"""Build an interactive social graph from photographs.

Images are analyzed with InsightFace and their face embeddings are grouped into
identities with DBSCAN. An image folder may contain images directly (all are
treated as one event) and/or event subfolders (each first-level subfolder is
an event). Profile photos are optional: each image in the profile folder should
show one person, and its filename (without extension) is used as that person's
label when it matches a detected identity.

People become graph nodes. An edge joins people seen together in an image or in
the same event. Image co-occurrence contributes 5 points for images with at
most three detected faces and 2 points for larger groups; event co-occurrence
contributes 1 point. Edge scores are accumulated and shown in the interactive
HTML graph. Faces DBSCAN marks as noise are omitted. Nodes without connections
(isolated nodes) are removed by default.

Each person in the graph receives a popularity score: every time someone appears
in an event, they gain 50 points, plus their in-degree (number of connections)
at the end.

Graph data is exported both as an interactive PyVis HTML visualization and
as a structured JSON file.

Usage::

    python create_social_graph.py PATH_TO_IMAGES
    python create_social_graph.py PATH_TO_IMAGES --disk-storage
    python create_social_graph.py PATH_TO_IMAGES --disk-storage --cache-dir my_cache
    python create_social_graph.py PATH_TO_IMAGES --profiles PATH_TO_PROFILES \\
        --output people_graph.html --json-output people_graph.json

See ``python create_social_graph.py --help`` for all options.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import gc
import json
from pathlib import Path
import shutil
from typing import Callable

import cv2
import networkx as nx
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from pyvis.network import Network
from tqdm import tqdm
from image_processing.image_utilities import face_crop_quality


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def image_files(folder: Path, recursive: bool = True, exclude_dirs: set[Path] | None = None) -> list[Path]:
    """Return supported image paths under *folder*, in stable sorted order."""
    exclude = exclude_dirs or set()
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    found: list[Path] = []
    for p in iterator:
        if any(p.is_relative_to(ex) for ex in exclude):
            continue
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            found.append(p)
    return sorted(found)


def read_image(path: Path) -> np.ndarray | None:
    """Decode an image, including paths containing non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def encode_crop(crop: np.ndarray) -> str | None:
    """Encode a face crop as a JPEG data URI for embedding in HTML/JSON."""
    mime = "image/jpeg"
    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    import base64
    return f"data:{mime};base64," + base64.b64encode(encoded).decode("ascii")


def encode_crop_file(path: Path) -> str | None:
    """Encode an existing JPEG crop on disk as a data URI."""
    try:
        import base64
        data = path.read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    except OSError:
        return None


def collect_events(image_root: Path, exclude_dirs: set[Path] | None = None) -> list[tuple[str, Path]]:
    """Return ``(event, image_path)`` pairs from a flat or grouped image folder.

    The first path component below the root identifies an event. Images placed
    directly in the root use the root folder's name as their event identifier.
    """
    found = image_files(image_root, exclude_dirs=exclude_dirs)
    pairs = []
    for path in found:
        relative = path.relative_to(image_root)
        event = relative.parts[0] if len(relative.parts) > 1 else image_root.name
        pairs.append((event, path))
    return pairs


def extract_image_faces(
    event_images: list[tuple[str, Path]],
    app: FaceAnalysis,
    disk_storage: bool = False,
    crops_dir: Path | None = None,
    embeddings_file=None,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_image: Callable[[Path, np.ndarray | None, list[tuple[int, int, int, int]], int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[list[np.ndarray], list[dict], dict[Path, int]]:
    """Scan images and detect faces, extracting normalized embeddings and crops."""
    vectors: list[np.ndarray] = []
    face_records: list[dict] = []
    image_face_counts: dict[Path, int] = {}
    total_imgs = len(event_images)
    face_idx = 0

    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    def check_cancel():
        if is_cancelled and is_cancelled():
            raise InterruptedError("Operation cancelled by user.")

    for idx, (event, path) in enumerate(tqdm(event_images, desc="Analyzing images", unit="image"), start=1):
        check_cancel()
        if on_progress:
            on_progress(idx, total_imgs, f"Scanning image [{idx}/{total_imgs}]: {path.name}")

        image = read_image(path)
        if image is None:
            log(f"Warning: could not read image: {path.name}")
            image_face_counts[path] = 0
            if on_image:
                on_image(path, None, [], 0, event)
            continue

        faces = app.get(image)
        image_face_counts[path] = len(faces)
        height, width = image.shape[:2]
        bboxes: list[tuple[int, int, int, int]] = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            bboxes.append((x1, y1, x2, y2))

            embedding = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if not norm:
                continue
            embedding_norm = embedding / norm
            has_crop = (x2 > x1 and y2 > y1)

            if disk_storage:
                assert embeddings_file is not None and crops_dir is not None
                embeddings_file.write(embedding_norm.tobytes())
                crop_path = None
                if has_crop:
                    crop = image[y1:y2, x1:x2]
                    crop_path = crops_dir / f"face_{face_idx:07d}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                face_records.append({"event": event, "image": path, "crop_path": crop_path})
            else:
                vectors.append(embedding_norm)
                crop = image[y1:y2, x1:x2].copy() if has_crop else None
                face_records.append({"event": event, "image": path, "crop": crop})

            face_idx += 1

        if on_image:
            on_image(path, image, bboxes, len(faces), event)

    return vectors, face_records, image_face_counts


def extract_profile_faces(
    profiles_root: Path,
    app: FaceAnalysis,
    exclude_dirs: set[Path] | None = None,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_image: Callable[[Path, np.ndarray | None, list[tuple[int, int, int, int]], int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Scan profile pictures and extract one representative embedding per person."""
    profile_images = image_files(profiles_root, exclude_dirs=exclude_dirs)
    profile_vectors: list[np.ndarray] = []
    profile_names: list[str] = []
    total = len(profile_images)

    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    def check_cancel():
        if is_cancelled and is_cancelled():
            raise InterruptedError("Operation cancelled by user.")

    log(f"Matching identities against {total} profile photos...")
    for idx, path in enumerate(tqdm(profile_images, desc="Analyzing profiles", unit="image"), start=1):
        check_cancel()
        if on_progress:
            on_progress(idx, total, f"Profile [{idx}/{total}]: {path.name}")

        image = read_image(path)
        if image is None:
            log(f"Warning: could not read profile image: {path.name}")
            continue

        faces = app.get(image)
        bboxes = []
        if faces:
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = best_face.bbox.astype(int)
            bboxes.append((x1, y1, x2, y2))
            embedding = np.asarray(best_face.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm:
                profile_vectors.append(embedding / norm)
                profile_names.append(path.stem)

        if on_image:
            on_image(path, image, bboxes, len(faces), "Profiles")

    return profile_vectors, profile_names


def cluster_identities(
    embeddings: np.ndarray,
    eps: float = 0.45,
    min_samples: int = 2,
) -> np.ndarray:
    """Group face embeddings into identity clusters with DBSCAN."""
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embeddings)


def match_profiles_to_identities(
    embeddings: np.ndarray,
    labels: np.ndarray,
    profile_vectors: list[np.ndarray],
    profile_names: list[str],
    profile_distance: float = 0.55,
    on_log: Callable[[str], None] | None = None,
) -> dict[int, str]:
    """Map cluster labels to profile names based on cosine similarity to cluster centers."""
    cluster_labels = sorted(int(label) for label in set(labels) if label >= 0)
    centers: dict[int, np.ndarray] = {}
    for label in cluster_labels:
        center = embeddings[labels == label].mean(axis=0)
        centers[label] = center / np.linalg.norm(center)

    cluster_names: dict[int, str] = {label: f"person_{label}" for label in cluster_labels}
    if profile_vectors:
        msg = "Assigning profile names to matching identities..."
        if on_log:
            on_log(msg)
        else:
            print(msg)
        profile_matrix = np.stack(profile_vectors)
        for label, center in centers.items():
            similarities = profile_matrix @ center
            best = int(np.argmax(similarities))
            if similarities[best] >= 1 - profile_distance:
                cluster_names[label] = profile_names[best]

    return cluster_names


def prepare_node_crops(
    face_records: list[dict],
    labels: np.ndarray,
    cluster_names: dict[int, str],
    disk_storage: bool = False,
) -> dict[str, str]:
    """Extract the highest-quality representative face crop for each cluster."""

    crops: dict[str, str] = {}
    best_scores: dict[str, float] = {}

    for record, label in zip(face_records, labels):
        if label < 0:
            continue

        name = cluster_names[int(label)]

        if disk_storage:
            crop_path = record.get("crop_path")
            if not crop_path or not crop_path.is_file():
                continue

            crop = cv2.imread(str(crop_path))
            if crop is None:
                continue

            score = face_crop_quality(crop)

            if score > best_scores.get(name, -1):
                data_uri = encode_crop_file(crop_path)
                if data_uri:
                    crops[name] = data_uri
                    best_scores[name] = score

        else:
            crop = record.get("crop")
            if crop is None:
                continue

            score = face_crop_quality(crop)

            if score > best_scores.get(name, -1):
                data_uri = encode_crop(crop)
                if data_uri:
                    crops[name] = data_uri
                    best_scores[name] = score

    return crops


def export_grouped_faces(
    grouped_dir: Path,
    face_records: list[dict],
    labels: np.ndarray,
    cluster_names: dict[int, str],
    disk_storage: bool = False,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    """Save or copy face crops into per-person directories."""
    msg = f"Organizing grouped faces into: {grouped_dir}"
    if on_log:
        on_log(msg)
    else:
        print(msg)
    grouped_dir.mkdir(parents=True, exist_ok=True)
    for i, (record, label) in enumerate(zip(face_records, labels)):
        if label < 0:
            continue
        person = cluster_names[int(label)]
        person_dir = grouped_dir / person
        person_dir.mkdir(parents=True, exist_ok=True)
        if disk_storage and record.get("crop_path"):
            shutil.copy2(record["crop_path"], person_dir / record["crop_path"].name)
        elif not disk_storage and record.get("crop") is not None:
            cv2.imwrite(str(person_dir / f"face_{i:07d}.jpg"), record["crop"])
    return grouped_dir


def construct_graph(
    face_records: list[dict],
    labels: np.ndarray,
    cluster_names: dict[int, str],
    image_face_counts: dict[Path, int],
    remove_isolated: bool = True,
    on_log: Callable[[str], None] | None = None,
) -> nx.Graph:
    """Build the social network graph with co-occurrence weighting and optional isolate removal."""
    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    log(f"Building connections for {len(cluster_names)} identified people...")
    event_people: dict[str, set[str]] = defaultdict(set)
    image_people: dict[Path, set[str]] = defaultdict(set)
    for record, label in zip(face_records, labels):
        if label < 0:
            continue
        person = cluster_names[int(label)]
        event_people[record["event"]].add(person)
        image_people[record["image"]].add(person)

    graph = nx.Graph()
    graph.add_nodes_from(sorted(set(cluster_names.values())))

    # Add stronger links for appearing together in a photo; larger group photos
    # get less weight (2 points) vs small group photos (5 points).
    for path, people in image_people.items():
        points = 5 if image_face_counts[path] <= 3 else 2
        members = sorted(people)
        for i, person_a in enumerate(members):
            for person_b in members[i + 1:]:
                graph.add_edge(person_a, person_b, weight=graph.get_edge_data(person_a, person_b, {}).get("weight", 0) + points)

    # Sharing an event adds 1 point, but only when there are multiple distinct events.
    # With a single event (flat folder), every person trivially shares it, which would
    # connect all people to each other and add noise rather than social signal.
    multiple_events = len(event_people) > 1
    if multiple_events:
        for people in event_people.values():
            members = sorted(people)
            for i, person_a in enumerate(members):
                for person_b in members[i + 1:]:
                    graph.add_edge(person_a, person_b, weight=graph.get_edge_data(person_a, person_b, {}).get("weight", 0) + 1)
    else:
        log("Single event detected (flat folder): skipping event co-occurrence edges to avoid connecting all people.")

    # Remove nodes without edges (isolated nodes)
    if remove_isolated:
        isolated = list(nx.isolates(graph))
        if isolated:
            graph.remove_nodes_from(isolated)
            log(f"Removed {len(isolated)} isolated person/node(s) with no connections (kept {len(graph.nodes)} connected nodes).")

    # Calculate popularity score for each person:
    # 50 points for each distinct event the person appears in (only meaningful when
    # multiple events exist), plus in-degree (number of connections).
    person_events: dict[str, set[str]] = defaultdict(set)
    for event, people in event_people.items():
        for person in people:
            person_events[person].add(event)

    for person in graph.nodes:
        num_events = len(person_events.get(person, set())) if multiple_events else 0
        deg = graph.degree[person]
        popularity = (num_events * 50) + deg
        graph.nodes[person]["event_count"] = num_events
        graph.nodes[person]["popularity"] = popularity

    # Assign deterministic positions so the visual graph remains stable
    if len(graph.nodes) > 0:
        pos = nx.spring_layout(graph, seed=42)
        for node, (x, y) in pos.items():
            graph.nodes[node]["x"] = round(float(x * 1000), 2)
            graph.nodes[node]["y"] = round(float(y * 1000), 2)

    return graph


def weight_to_color(weight: float, min_w: float, max_w: float) -> str:
    """Map connection weight to a hex color from cool blue (weak) to warm red (strong)."""
    if max_w <= min_w:
        t = 1.0
    else:
        t = max(0.0, min(1.0, (weight - min_w) / (max_w - min_w)))

    # Gradient: cool blue (t=0.0) -> purple (t=0.5) -> bright red (t=1.0)
    if t < 0.5:
        factor = t / 0.5
        r = int(96 + factor * (139 - 96))
        g = int(165 + factor * (92 - 165))
        b = int(250 + factor * (246 - 250))
    else:
        factor = (t - 0.5) / 0.5
        r = int(139 + factor * (239 - 139))
        g = int(92 + factor * (68 - 92))
        b = int(246 + factor * (68 - 246))

    return f"#{r:02x}{g:02x}{b:02x}"


def export_graph_json(
    graph: nx.Graph,
    crops: dict[str, str],
    output_path: Path,
    metadata: dict | None = None,
) -> Path:
    """Export graph structure and attributes to a JSON file."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = [data.get("weight", 1) for _, _, data in graph.edges(data=True)]
    min_weight = min(weights, default=1)
    max_weight = max(weights, default=1)

    nodes_data = []
    for node in sorted(graph.nodes):
        node_dict = {
            "id": node,
            "label": node,
            "degree": graph.degree[node],
            "popularity": graph.nodes[node].get("popularity", graph.degree[node]),
        }
        if "event_count" in graph.nodes[node]:
            node_dict["event_count"] = graph.nodes[node]["event_count"]
        if "x" in graph.nodes[node]:
            node_dict["x"] = graph.nodes[node]["x"]
            node_dict["y"] = graph.nodes[node]["y"]
        if node in crops:
            node_dict["image"] = crops[node]
        nodes_data.append(node_dict)

    edges_data = []
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        edges_data.append({
            "source": u,
            "target": v,
            "weight": weight,
            "value": round(weight / max_weight, 4),
            "color": weight_to_color(weight, min_weight, max_weight),
        })

    graph_payload = {
        "metadata": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "generated_at": datetime.datetime.now().isoformat(),
            **(metadata or {}),
        },
        "nodes": nodes_data,
        "edges": edges_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_payload, f, indent=2)

    return output_path


def export_graph_html(
    graph: nx.Graph,
    crops: dict[str, str],
    output_path: Path,
) -> Path:
    """Write an interactive PyVis HTML visualization."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = [data.get("weight", 1) for _, _, data in graph.edges(data=True)]
    min_weight = min(weights, default=1)
    max_weight = max(weights, default=1)
    net = Network(height="800px", width="100%", directed=False, bgcolor="#ffffff", font_color="#222222")

    for person in graph.nodes:
        pop = graph.nodes[person].get("popularity")
        deg = graph.degree[person]
        if pop is not None:
            evt_cnt = graph.nodes[person].get("event_count")
            evt_str = f"Events: {evt_cnt} | " if evt_cnt is not None else ""
            title = f"{person}\nPopularity: {pop}\n({evt_str}Connections: {deg})"
        else:
            title = person
        node = {"label": person, "title": title}
        if "x" in graph.nodes[person]:
            node["x"] = graph.nodes[person]["x"]
            node["y"] = graph.nodes[person]["y"]
        if person in crops:
            node.update(image=crops[person], shape="image")
        else:
            node.update(shape="dot", size=18)
        net.add_node(person, **node)

    for person_a, person_b, data in graph.edges(data=True):
        weight = data["weight"]
        edge_color = weight_to_color(weight, min_weight, max_weight)
        net.add_edge(person_a, person_b, color=edge_color, width=2, title=f"Score: {weight}")

    net.toggle_physics(False)
    net.write_html(str(output_path), open_browser=False)
    return output_path


def build_graph(
    args: argparse.Namespace | None = None,
    *,
    images: Path | str | None = None,
    profiles: Path | str | None = None,
    output: Path | str | None = None,
    json_output: Path | str | None = None,
    save_json: bool = True,
    save_bundle: bool = True,
    disk_storage: bool = False,
    cache_dir: Path | str | None = None,
    grouped_faces: bool = False,
    remove_isolated: bool = True,
    eps: float = 0.45,
    min_samples: int = 2,
    profile_distance: float = 0.55,
    model: str = "buffalo_l",
    ctx_id: int = -1,
    on_log: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_image: Callable[[Path, np.ndarray | None, list[tuple[int, int, int, int]], int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[Path, Path | None, Path | None]:
    """Analyze input photos and build social network graph files (HTML and JSON).

    Can be invoked with an ``argparse.Namespace`` (produced by :func:`parse_args`)
    or with explicit keyword arguments from external Python code or GUI.
    """
    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    def set_status(status: str):
        if on_status:
            on_status(status)

    def check_cancel():
        if is_cancelled and is_cancelled():
            raise InterruptedError("Operation cancelled by user.")

    # Resolve arguments
    if args is not None:
        images_val = args.images
        profiles_val = args.profiles
        output_val = args.output
        json_output_val = getattr(args, "json_output", None)
        save_json_val = not getattr(args, "no_json", False)
        save_bundle_val = not getattr(args, "no_bundle", False)
        disk_storage_val = getattr(args, "disk_storage", False)
        cache_dir_val = getattr(args, "cache_dir", None)
        grouped_faces_val = getattr(args, "grouped_faces", False)
        remove_isolated_val = not getattr(args, "keep_isolated", False)
        eps_val = getattr(args, "eps", 0.45)
        min_samples_val = getattr(args, "min_samples", 2)
        profile_distance_val = getattr(args, "profile_distance", 0.55)
        model_val = getattr(args, "model", "buffalo_l")
        ctx_id_val = getattr(args, "ctx_id", -1)
    else:
        images_val = images
        profiles_val = profiles
        output_val = output or Path("people_graph.html")
        json_output_val = json_output
        save_bundle_val = save_bundle
        save_json_val = save_json
        disk_storage_val = disk_storage
        cache_dir_val = cache_dir
        grouped_faces_val = grouped_faces
        remove_isolated_val = remove_isolated
        eps_val = eps
        min_samples_val = min_samples
        profile_distance_val = profile_distance
        model_val = model
        ctx_id_val = ctx_id
    print(args)
    if not images_val:
        raise ValueError("Images folder must be specified.")

    image_root = Path(images_val).resolve()
    if not image_root.is_dir():
        raise ValueError(f"Image folder does not exist: {image_root}")
    profiles_root = Path(profiles_val).resolve() if profiles_val else None
    if profiles_root and not profiles_root.is_dir():
        raise ValueError(f"Profile picture folder does not exist: {profiles_root}")

    # Configure disk storage
    if grouped_faces_val and not cache_dir_val:
        cache_dir_val = Path("graph_cache")
    if cache_dir_val is not None:
        disk_storage_val = True
    if disk_storage_val and cache_dir_val is None:
        cache_dir_val = Path("graph_cache")

    cache_dir_resolved = Path(cache_dir_val).resolve() if disk_storage_val and cache_dir_val else None
    crops_dir = cache_dir_resolved / "crops" if cache_dir_resolved else None
    embeddings_path = cache_dir_resolved / "embeddings.bin" if cache_dir_resolved else None
    embeddings_file = None

    if disk_storage_val and cache_dir_resolved and crops_dir and embeddings_path:
        cache_dir_resolved.mkdir(parents=True, exist_ok=True)
        if crops_dir.exists():
            shutil.rmtree(crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)
        if embeddings_path.exists():
            embeddings_path.unlink()
        embeddings_file = open(embeddings_path, "wb")
        log(f"Disk storage active: storing intermediate data in {cache_dir_resolved}")

    exclude_dirs = {cache_dir_resolved} if cache_dir_resolved else None
    event_images = collect_events(image_root, exclude_dirs=exclude_dirs)
    if not event_images:
        raise ValueError(f"No supported images found in {image_root}")

    output_html = Path(output_val).resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    if save_json_val:
        if json_output_val:
            output_json = Path(json_output_val).resolve()
        else:
            output_json = output_html.with_suffix(".json")
        output_json.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_json = None

    log(f"Found {len(event_images)} images across {len({event for event, _ in event_images})} events.")
    ctx_str = "CPU" if ctx_id_val == -1 else f"GPU device {ctx_id_val}"
    set_status(f"Loading face analysis model '{model_val}' on {ctx_str}...")
    log(f"Loading face analysis model '{model_val}' ({ctx_str})...")
    app = FaceAnalysis(name=model_val)
    app.prepare(ctx_id=ctx_id_val, det_size=(640, 640))

    set_status("Scanning images and detecting faces...")
    log("Detecting faces and extracting identity features...")
    try:
        vectors, face_records, image_face_counts = extract_image_faces(
            event_images=event_images,
            app=app,
            disk_storage=disk_storage_val,
            crops_dir=crops_dir,
            embeddings_file=embeddings_file,
            on_log=on_log,
            on_progress=on_progress,
            on_image=on_image,
            is_cancelled=is_cancelled,
        )
    finally:
        if embeddings_file is not None:
            try:
                embeddings_file.flush()
                embeddings_file.close()
            except Exception:
                pass

    check_cancel()
    if not face_records:
        raise ValueError("No faces were detected in the input images")

    set_status(f"Clustering {len(face_records)} faces into identities (DBSCAN)...")
    log(f"Detected {len(face_records)} faces. Grouping faces into identities...")
    if disk_storage_val and embeddings_path is not None:
        embeddings = np.memmap(
            embeddings_path,
            dtype=np.float32,
            mode="r",
            shape=(len(face_records), 512),
        )
    else:
        embeddings = np.stack(vectors)

    labels = cluster_identities(embeddings, eps=eps_val, min_samples=min_samples_val)

    if disk_storage_val and cache_dir_resolved is not None:
        np.save(cache_dir_resolved / "labels.npy", labels)

    check_cancel()
    # Profile matching
    profile_vectors: list[np.ndarray] = []
    profile_names: list[str] = []
    if profiles_root:
        set_status("Scanning profile photos...")
        profile_vectors, profile_names = extract_profile_faces(
            profiles_root=profiles_root,
            app=app,
            exclude_dirs=exclude_dirs,
            on_log=on_log,
            on_progress=on_progress,
            on_image=on_image,
            is_cancelled=is_cancelled,
        )

    cluster_names = match_profiles_to_identities(
        embeddings=embeddings,
        labels=labels,
        profile_vectors=profile_vectors,
        profile_names=profile_names,
        profile_distance=profile_distance_val,
        on_log=on_log,
    )

    # Free memory-mapped file resource on Windows
    if disk_storage_val:
        del embeddings
        gc.collect()

    check_cancel()
    # Save representative face crops for graph nodes
    crops = prepare_node_crops(
        face_records=face_records,
        labels=labels,
        cluster_names=cluster_names,
        disk_storage=disk_storage_val,
    )

    # Optionally organize clustered faces into per-person directories on disk
    if grouped_faces_val:
        set_status("Exporting grouped faces into folders...")
        grouped_dir = (cache_dir_resolved / "grouped_faces") if cache_dir_resolved else Path("grouped_faces").resolve()
        export_grouped_faces(
            grouped_dir=grouped_dir,
            face_records=face_records,
            labels=labels,
            cluster_names=cluster_names,
            disk_storage=disk_storage_val,
            on_log=on_log,
        )

    check_cancel()
    set_status("Building social connections...")
    graph = construct_graph(
        face_records=face_records,
        labels=labels,
        cluster_names=cluster_names,
        image_face_counts=image_face_counts,
        remove_isolated=remove_isolated_val,
        on_log=on_log,
    )

    check_cancel()
    # Export JSON graph if requested
    if output_json:
        set_status("Saving graph JSON...")
        log(f"Saving graph data in JSON format to: {output_json}")
        export_graph_json(
            graph=graph,
            crops=crops,
            output_path=output_json,
            metadata={
                "image_count": len(event_images),
                "faces_detected": len(face_records),
                "cluster_count": len(cluster_names),
                "removed_isolated": remove_isolated_val,
            },
        )

    # Export interactive HTML graph
    set_status("Saving interactive HTML graph...")
    log(f"Saving interactive graph with {len(graph.nodes)} people and {len(graph.edges)} connections to: {output_html}")
    export_graph_html(graph=graph, crops=crops, output_path=output_html)

    # Save serialized graph data bundle for continuous updates
    output_bundle: Path | None = None
    if save_bundle_val:
        try:
            set_status("Saving graph data bundle for future updates...")
            from utilities.graph_storage import save_graph_data, serialize_crop
            bundle_path = output_html.with_name(f"{output_html.stem}.graph_data.pkl")
            # Compact face record crops to JPEG bytes for storage
            compact_records = []
            for rec in face_records:
                c = dict(rec)
                if c.get("crop") is not None:
                    c["crop"] = serialize_crop(c["crop"])
                compact_records.append(c)
            if disk_storage_val and embeddings_path is not None:
                import numpy as _np
                embeddings_for_bundle = _np.array(
                    _np.memmap(embeddings_path, dtype=_np.float32, mode="r", shape=(len(face_records), 512))
                )
            else:
                embeddings_for_bundle = embeddings
            processed_images = {str(p.resolve()) for _, p in event_images}
            save_graph_data(
                bundle_path,
                graph=graph,
                crops=crops,
                embeddings=embeddings_for_bundle,
                face_records=compact_records,
                image_face_counts=image_face_counts,
                labels=labels,
                cluster_names=cluster_names,
                profile_vectors=profile_vectors,
                profile_names=profile_names,
                processed_images=processed_images,
                params={
                    "eps": eps_val,
                    "min_samples": min_samples_val,
                    "profile_distance": profile_distance_val,
                    "model": model_val,
                    "ctx_id": ctx_id_val,
                    "remove_isolated": remove_isolated_val,
                },
                metadata={
                    "image_count": len(event_images),
                    "faces_detected": len(face_records),
                    "cluster_count": len(cluster_names),
                },
            )
            output_bundle = bundle_path
            log(f"Graph data bundle saved to: {bundle_path}")
        except Exception as _exc:
            log(f"Warning: could not save graph data bundle: {_exc}")

    log(f"Successfully processed {len(event_images)} images; detected {len(face_records)} faces; created {len(graph.nodes)} people and {len(graph.edges)} connections.")
    log(f"HTML Graph saved to: {output_html}")
    if output_json:
        log(f"JSON Graph saved to: {output_json}")

    return output_html, output_json, output_bundle


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the image, model, and output settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Folder containing images (flat or grouped in event subfolders)")
    parser.add_argument("-p", "--profiles", type=Path, help="Optional folder of profile photos named after people")
    parser.add_argument("-o", "--output", type=Path, default=Path("people_graph.html"), help="Output HTML graph (default: people_graph.html)")
    parser.add_argument("-j", "--json-output", type=Path, default=None, help="Output JSON graph path (default: <output>.json)")
    parser.add_argument("--no-json", action="store_true", help="Do not save the JSON graph file")
    parser.add_argument("--no-bundle", action="store_true", help="Do not save the .graph_data.pkl bundle (disables continuous graph updates)")
    parser.add_argument(
        "--disk-storage",
        action="store_true",
        help="Store intermediate face embeddings and crops on hard disk instead of in memory to handle large datasets",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory to store intermediate disk data (defaults to ./graph_cache when --disk-storage is enabled)",
    )
    parser.add_argument(
        "--grouped-faces",
        action="store_true",
        help="Organize detected face crops into per-person directories on disk",
    )
    parser.add_argument(
        "--keep-isolated",
        action="store_true",
        help="Keep isolated nodes without edges (by default, nodes without connections are removed)",
    )
    parser.add_argument("--eps", type=float, default=0.45, help="DBSCAN cosine-distance threshold (default: 0.45)")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum faces per identity cluster (default: 2)")
    parser.add_argument("--profile-distance", type=float, default=0.55, help="Maximum cosine distance for profile matching (default: 0.55)")
    parser.add_argument("--model", default="buffalo_l", help="InsightFace model pack (default: buffalo_l)")
    parser.add_argument("--ctx-id", type=int, default=-1, help="Execution provider device id; -1 uses CPU")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        build_graph(parse_args())
    except (ValueError, OSError) as error:
        raise SystemExit(f"Error: {error}")
