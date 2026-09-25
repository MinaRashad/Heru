"""Create a social graph from event photos, with optional profile pictures.

Input images can be in one flat folder (treated as one event) or in event
subfolders. Profile image filenames are used as person labels when provided.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from pyvis.network import Network


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def image_files(folder: Path, recursive: bool = True) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path: Path):
    # imdecode handles non-ASCII paths on Windows as well as regular paths.
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def encode_crop(crop: np.ndarray) -> str | None:
    mime = "image/jpeg"
    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    import base64
    return f"data:{mime};base64," + base64.b64encode(encoded).decode("ascii")


def collect_events(image_root: Path) -> list[tuple[str, Path]]:
    """Return (event, image) pairs; root-level images form a single event."""
    found = image_files(image_root)
    pairs = []
    for path in found:
        relative = path.relative_to(image_root)
        event = relative.parts[0] if len(relative.parts) > 1 else image_root.name
        pairs.append((event, path))
    return pairs


def build_graph(args: argparse.Namespace) -> Path:
    image_root = args.images.resolve()
    if not image_root.is_dir():
        raise ValueError(f"Image folder does not exist: {image_root}")
    profiles_root = args.profiles.resolve() if args.profiles else None
    if profiles_root and not profiles_root.is_dir():
        raise ValueError(f"Profile picture folder does not exist: {profiles_root}")

    event_images = collect_events(image_root)
    if not event_images:
        raise ValueError(f"No supported images found in {image_root}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name=args.model)
    app.prepare(ctx_id=args.ctx_id, det_size=(640, 640))

    vectors: list[np.ndarray] = []
    face_records: list[dict] = []
    image_face_counts: dict[Path, int] = {}
    for event, path in event_images:
        image = read_image(path)
        if image is None:
            print(f"Warning: could not read image: {path}")
            image_face_counts[path] = 0
            continue
        faces = app.get(image)
        image_face_counts[path] = len(faces)
        height, width = image.shape[:2]
        for face in faces:
            embedding = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if not norm:
                continue
            vectors.append(embedding / norm)
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            crop = image[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
            face_records.append({"event": event, "image": path, "crop": crop})

    if not vectors:
        raise ValueError("No faces were detected in the input images")

    embeddings = np.stack(vectors)
    labels = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine").fit_predict(embeddings)

    # Profiles are matched to cluster centers only when a profile folder was given.
    profile_vectors: list[np.ndarray] = []
    profile_names: list[str] = []
    if profiles_root:
        for path in image_files(profiles_root):
            image = read_image(path)
            if image is None:
                print(f"Warning: could not read profile image: {path}")
                continue
            faces = app.get(image)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm:
                profile_vectors.append(embedding / norm)
                profile_names.append(path.stem)

    cluster_labels = sorted(int(label) for label in set(labels) if label >= 0)
    centers: dict[int, np.ndarray] = {}
    for label in cluster_labels:
        center = embeddings[labels == label].mean(axis=0)
        centers[label] = center / np.linalg.norm(center)

    cluster_names: dict[int, str] = {label: f"person_{label}" for label in cluster_labels}
    if profile_vectors:
        profile_matrix = np.stack(profile_vectors)
        for label, center in centers.items():
            similarities = profile_matrix @ center
            best = int(np.argmax(similarities))
            if similarities[best] >= 1 - args.profile_distance:
                cluster_names[label] = profile_names[best]

    # Save representative face crops for graph nodes. Data URIs in the HTML make
    # the graph portable as a single file.
    crops: dict[str, str] = {}
    for record, label in zip(face_records, labels):
        if label < 0 or record["crop"] is None:
            continue
        name = cluster_names[int(label)]
        if name not in crops:
            data_uri = encode_crop(record["crop"])
            if data_uri:
                crops[name] = data_uri

    # Track people by both event and source image. Unknown faces are excluded.
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
    for path, people in image_people.items():
        points = 5 if image_face_counts[path] <= 3 else 2
        members = sorted(people)
        for i, person_a in enumerate(members):
            for person_b in members[i + 1:]:
                graph.add_edge(person_a, person_b, weight=graph.get_edge_data(person_a, person_b, {}).get("weight", 0) + points)
    for people in event_people.values():
        members = sorted(people)
        for i, person_a in enumerate(members):
            for person_b in members[i + 1:]:
                graph.add_edge(person_a, person_b, weight=graph.get_edge_data(person_a, person_b, {}).get("weight", 0) + 1)

    max_weight = max((data["weight"] for _, _, data in graph.edges(data=True)), default=1)
    net = Network(height="800px", width="100%", directed=False, bgcolor="#ffffff", font_color="#222222")
    for person in graph.nodes:
        node = {"label": person, "title": person}
        if person in crops:
            node.update(image=crops[person], shape="image")
        else:
            node.update(shape="dot", size=18)
        net.add_node(person, **node)
    for person_a, person_b, data in graph.edges(data=True):
        weight = data["weight"]
        net.add_edge(person_a, person_b, value=weight / max_weight, title=f"Score: {weight}")
    net.write_html(str(output), open_browser=False)
    print(f"Processed {len(event_images)} images; detected {len(face_records)} faces; created {len(graph.nodes)} people and {len(graph.edges)} connections.")
    print(f"Graph saved to: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Folder containing images (flat or grouped in event subfolders)")
    parser.add_argument("-p", "--profiles", type=Path, help="Optional folder of profile photos named after people")
    parser.add_argument("-o", "--output", type=Path, default=Path("people_graph.html"), help="Output HTML graph (default: people_graph.html)")
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
