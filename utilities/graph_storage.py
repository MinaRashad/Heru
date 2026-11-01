"""Storage and serialization utilities for Social Eye graph data.

Handles saving and loading the complete state required to continuously update
a social graph (embeddings, face records, image counts, profile vectors, metadata,
and network structure) or load graph data for pruning.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any

import cv2
import networkx as nx
import numpy as np

try:
    from image_processing.image_utilities import face_crop_quality
except ImportError:
    import sys
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from image_processing.image_utilities import face_crop_quality


def serialize_crop(crop: np.ndarray | bytes | None) -> bytes | None:
    """Ensure a face crop is stored as compact JPEG bytes."""
    if crop is None:
        return None
    if isinstance(crop, bytes):
        return crop
    if isinstance(crop, np.ndarray):
        ok, encoded = cv2.imencode(".jpg", crop)
        if ok:
            return encoded.tobytes()
    return None


def compute_image_hash(image_input: bytes | Path | str | np.ndarray) -> str:
    """Compute SHA-256 hash of an image file, bytes, or array."""
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if p.is_file():
            hasher = hashlib.sha256()
            try:
                with open(p, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                return hasher.hexdigest()
            except Exception:
                pass
    elif isinstance(image_input, bytes):
        return hashlib.sha256(image_input).hexdigest()
    elif isinstance(image_input, np.ndarray):
        return hashlib.sha256(image_input.tobytes()).hexdigest()
    return ""


def crop_to_data_uri(crop: np.ndarray | bytes | str | None) -> str | None:
    """Convert crop (np.ndarray, JPEG bytes, or existing data URI) to a base64 data URI."""
    if crop is None:
        return None
    if isinstance(crop, str):
        return crop if crop.startswith("data:") else None
    if isinstance(crop, bytes):
        b64 = base64.b64encode(crop).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    if isinstance(crop, np.ndarray):
        ok, encoded = cv2.imencode(".jpg", crop)
        if ok:
            b64 = base64.b64encode(encoded).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    return None


def build_person_crops(
    face_records: list[dict],
    labels: np.ndarray,
    cluster_names: dict[int, str],
    disk_storage: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Extract highest quality representative image mapped to each person ID (Part 1).

    Parameters
    ----------
    face_records : list of dicts
        Face detection records.
    labels : np.ndarray
        Identity cluster labels.
    cluster_names : dict[int, str]
        Mapping from cluster label to person name/identifier.
    disk_storage : bool
        Whether crops are stored on disk.

    Returns
    -------
    tuple[dict[str, dict[str, Any]], dict[str, str]]
        - person_crops: person_id -> {person_id, crop, data_uri, quality, source_image, source_event}
        - crops: person_id -> data_uri (compatible with NetworkX & PyVis)
    """
    person_crops: dict[str, dict[str, Any]] = {}
    crops: dict[str, str] = {}
    best_scores: dict[str, float] = {}

    for record, label in zip(face_records, labels):
        if label < 0:
            continue
        person_name = cluster_names[int(label)]

        crop_img = None
        crop_bytes = None
        crop_path = record.get("crop_path")

        if disk_storage and crop_path:
            cp = Path(crop_path)
            if cp.is_file():
                crop_img = cv2.imread(str(cp))
                if crop_img is not None:
                    crop_bytes = serialize_crop(crop_img)
        else:
            raw_c = record.get("crop")
            if isinstance(raw_c, np.ndarray):
                crop_img = raw_c
                crop_bytes = serialize_crop(crop_img)
            elif isinstance(raw_c, bytes):
                crop_bytes = raw_c
                arr = np.frombuffer(raw_c, dtype=np.uint8)
                crop_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if crop_img is None:
            continue

        score = record.get("quality")
        if score is None:
            score = face_crop_quality(crop_img)

        if score > best_scores.get(person_name, -1.0):
            data_uri = crop_to_data_uri(crop_bytes)
            if data_uri:
                best_scores[person_name] = score
                crops[person_name] = data_uri
                person_crops[person_name] = {
                    "person_id": person_name,
                    "crop": crop_bytes,
                    "data_uri": data_uri,
                    "quality": float(score),
                    "source_image": str(record.get("image", "")),
                    "source_event": str(record.get("event", "")),
                }

    return person_crops, crops


def build_image_records(
    event_images: list[tuple[str, Path]] | list[dict[str, Any]],
    face_records: list[dict],
    labels: np.ndarray,
    cluster_names: dict[int, str],
    image_face_counts: dict[Path | str, int] | None = None,
    image_hashes: dict[Path | str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build standalone image records where each row is an image containing people IDs as a list (Part 2).

    Parameters
    ----------
    event_images : list of (event, path) tuples or existing image records
    face_records : list of dicts
    labels : np.ndarray
    cluster_names : dict[int, str]
    image_face_counts : dict, optional
    image_hashes : dict, optional
        Pre-cached image content hashes (SHA-256) keyed by image path.

    Returns
    -------
    list of dict
        Each row contains:
        - image_hash: SHA-256 hash of the image content
        - image_id: filename (e.g. img1.jpg)
        - image_path: normalized string path
        - event: event name
        - people: list of people IDs detected in this image (e.g. ['person_0', 'person_1'])
        - face_count: int
        - faces: list of face details (bounding boxes and quality scores)
    """
    hash_cache = image_hashes or {}
    counts = image_face_counts or {}

    # Map image path string to detected face records and their cluster labels
    image_to_faces: dict[str, list[tuple[dict, int]]] = {}
    image_to_event: dict[str, str] = {}

    for rec, label in zip(face_records, labels):
        img_str = str(Path(rec.get("image", "")).resolve())
        if img_str not in image_to_faces:
            image_to_faces[img_str] = []
        image_to_faces[img_str].append((rec, int(label)))
        if "event" in rec and rec["event"]:
            image_to_event[img_str] = str(rec["event"])

    # Determine unique set of images
    ordered_images: list[tuple[str, Path]] = []
    seen_paths = set()

    if event_images:
        for item in event_images:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                ev, p = item[0], Path(item[1])
                norm = str(p.resolve())
                if norm not in seen_paths:
                    ordered_images.append((str(ev), p))
                    seen_paths.add(norm)
            elif isinstance(item, dict):
                p_str = item.get("image_path") or item.get("image") or item.get("image_id", "")
                p = Path(p_str)
                ev = item.get("event", "")
                norm = str(p.resolve()) if p.exists() else p_str
                if norm not in seen_paths:
                    ordered_images.append((str(ev), p))
                    seen_paths.add(norm)

    # If any images in face_records weren't in event_images, include them
    for img_norm, face_list in image_to_faces.items():
        if img_norm not in seen_paths:
            ev = image_to_event.get(img_norm, Path(img_norm).parent.name)
            ordered_images.append((ev, Path(img_norm)))
            seen_paths.add(img_norm)

    image_records: list[dict[str, Any]] = []

    for event, path in ordered_images:
        norm_key = str(path.resolve()) if path.is_file() else str(path)
        matching_faces = image_to_faces.get(norm_key, [])

        # Collect people IDs (sorted unique) for faces in this image
        people_ids = sorted(list(set(
            cluster_names[lbl] for _, lbl in matching_faces if lbl >= 0 and lbl in cluster_names
        )))

        # SHA-256 hash of image content
        img_hash = ""
        if norm_key in hash_cache:
            img_hash = str(hash_cache[norm_key])
        elif path in hash_cache:
            img_hash = str(hash_cache[path])
        elif path.is_file():
            img_hash = compute_image_hash(path)

        # Total detected face count
        fc = counts.get(path, counts.get(norm_key, len(matching_faces)))

        # Detail face records for evidence
        faces_detail: list[dict[str, Any]] = []
        for face_rec, lbl in matching_faces:
            p_name = cluster_names.get(lbl) if lbl >= 0 else None
            faces_detail.append({
                "person_id": p_name,
                "cluster_label": lbl,
                "bbox": face_rec.get("bbox"),
                "quality": face_rec.get("quality"),
            })

        image_records.append({
            "image_hash": img_hash,
            "image_id": path.name if hasattr(path, "name") and path.name else str(path),
            "image_path": str(path),
            "event": str(event),
            "people": people_ids,
            "face_count": int(fc),
            "faces": faces_detail,
        })

    return image_records


def ensure_image_records(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Ensure a loaded graph bundle has standalone image records (Part 2).

    If the bundle was created with older versions, automatically synthesizes
    image records from face records and cluster assignments.
    """
    if "image_records" in bundle and bundle["image_records"]:
        return bundle["image_records"]

    face_records = bundle.get("face_records", [])
    labels = bundle.get("labels", [])
    cluster_names = bundle.get("cluster_names", {})
    image_face_counts = bundle.get("image_face_counts", {})

    image_records = build_image_records(
        event_images=[],
        face_records=face_records,
        labels=np.asarray(labels),
        cluster_names=cluster_names,
        image_face_counts=image_face_counts,
    )
    bundle["image_records"] = image_records
    return image_records


def save_graph_data(
    filepath: Path | str,
    *,
    graph: nx.Graph,
    crops: dict[str, str],
    embeddings: np.ndarray,
    face_records: list[dict],
    image_face_counts: dict[str | Path, int],
    labels: np.ndarray,
    cluster_names: dict[int, str],
    image_records: list[dict] | None = None,
    person_crops: dict[str, dict[str, Any]] | None = None,
    profile_vectors: list[np.ndarray] | None = None,
    profile_names: list[str] | None = None,
    processed_images: set[str | Path] | None = None,
    params: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save full graph state and features to disk so graph can be updated or rebuilt with new formulas.

    The saved bundle contains two standalone parts:
    1) Highest quality images mapped to each person ID (person_crops & crops).
    2) Each row is an image containing the people IDs as a list and image hash (image_records).
    """
    dest_path = Path(filepath).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize face_records to ensure compact storage
    compact_records = []
    for r in face_records:
        rec = {
            "event": str(r.get("event", "")),
            "image": str(r.get("image", "")),
        }
        if "crop_path" in r and r["crop_path"]:
            rec["crop_path"] = str(r["crop_path"])
        if "crop" in r and r["crop"] is not None:
            rec["crop"] = serialize_crop(r["crop"])
        if "bbox" in r and r["bbox"] is not None:
            rec["bbox"] = tuple(r["bbox"])
        if "quality" in r and r["quality"] is not None:
            rec["quality"] = float(r["quality"])
        compact_records.append(rec)

    # Normalize image_face_counts keys to strings
    normalized_counts = {str(Path(k).resolve()): v for k, v in image_face_counts.items()}

    # Normalize processed_images set to strings
    if processed_images is not None:
        proc_imgs = {str(Path(p).resolve()) for p in processed_images}
    else:
        proc_imgs = set(normalized_counts.keys())

    # Ensure Part 1 (person_crops) is populated
    if person_crops is None:
        person_crops, synthesized_crops = build_person_crops(
            face_records=compact_records,
            labels=labels,
            cluster_names=cluster_names,
        )
        if not crops:
            crops = synthesized_crops

    # Ensure Part 2 (image_records) is populated
    if image_records is None:
        image_records = build_image_records(
            event_images=[],
            face_records=compact_records,
            labels=labels,
            cluster_names=cluster_names,
            image_face_counts=normalized_counts,
        )
    else:
        # Normalize image_records
        compact_img_records = []
        for img_rec in image_records:
            row = {
                "image_hash": str(img_rec.get("image_hash") or ""),
                "image_id": str(img_rec.get("image_id") or ""),
                "image_path": str(img_rec.get("image_path") or ""),
                "event": str(img_rec.get("event") or ""),
                "people": list(img_rec.get("people", [])),
                "face_count": int(img_rec.get("face_count", len(img_rec.get("people", [])))),
            }
            if "faces" in img_rec:
                c_faces = []
                for f in img_rec["faces"]:
                    c_faces.append({
                        "person_id": f.get("person_id"),
                        "cluster_label": f.get("cluster_label"),
                        "bbox": tuple(f["bbox"]) if f.get("bbox") else None,
                        "quality": float(f["quality"]) if f.get("quality") is not None else None,
                    })
                row["faces"] = c_faces
            compact_img_records.append(row)
        image_records = compact_img_records

    # Ensure embeddings are numpy array float32
    if isinstance(embeddings, np.memmap):
        emb_array = np.array(embeddings, dtype=np.float32)
    else:
        emb_array = np.asarray(embeddings, dtype=np.float32)

    payload = {
        "version": 2,
        "format": "social_eye_graph_bundle",
        "created_at": metadata.get("created_at", datetime.datetime.now().isoformat()) if metadata else datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat(),
        "embeddings": emb_array,
        "face_records": compact_records,
        "image_face_counts": normalized_counts,
        "processed_images": proc_imgs,
        "labels": np.asarray(labels),
        "cluster_names": {int(k): str(v) for k, v in cluster_names.items()},
        "crops": crops,
        "person_crops": person_crops,
        "image_records": image_records,
        "profile_vectors": profile_vectors or [],
        "profile_names": profile_names or [],
        "params": params or {},
        "metadata": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "faces_detected": len(compact_records),
            "image_count": len(proc_imgs),
            "image_records_count": len(image_records),
            "standalone": True,
            **(metadata or {}),
        },
        "graph": graph,
    }

    temp_path = dest_path.with_name(f"{dest_path.name}.tmp")
    with open(temp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    if dest_path.exists():
        dest_path.unlink()
    temp_path.rename(dest_path)

    return dest_path

    temp_path = dest_path.with_name(f"{dest_path.name}.tmp")
    with open(temp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    if dest_path.exists():
        dest_path.unlink()
    temp_path.rename(dest_path)

    return dest_path


def load_graph_data(filepath: Path | str) -> dict[str, Any]:
    """Load serialized graph bundle from a pickle file."""
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Graph data bundle file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "embeddings" not in data or "graph" not in data:
        raise ValueError(f"File at {path} is not a valid Social Eye graph data bundle.")

    return data


def load_graph_from_json(json_path: Path | str) -> tuple[nx.Graph, dict[str, str], dict[str, Any]]:
    """Load a NetworkX graph and node crops directly from a Social Eye JSON file."""
    path = Path(json_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Graph JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = nx.Graph()
    crops: dict[str, str] = {}

    for node in data.get("nodes", []):
        node_id = node.get("id") or node.get("label")
        if not node_id:
            continue
        graph.add_node(node_id, **{k: v for k, v in node.items() if k not in ("id", "image")})
        if "image" in node and node["image"]:
            crops[node_id] = node["image"]

    for edge in data.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        weight = edge.get("weight", 1)
        if source and target:
            graph.add_edge(source, target, weight=weight)

    metadata = data.get("metadata", {})
    return graph, crops, metadata


def find_graph_data_file(path: Path | str) -> Path | None:
    """Find companion .graph_data.pkl or .pkl for a given file path."""
    p = Path(path).resolve()
    if p.is_file():
        if p.suffix.lower() == ".pkl":
            return p
        # Check <stem>.graph_data.pkl
        candidate1 = p.parent / f"{p.stem}.graph_data.pkl"
        if candidate1.is_file():
            return candidate1
        # Check <name>.graph_data.pkl
        candidate2 = p.with_name(f"{p.name}.graph_data.pkl")
        if candidate2.is_file():
            return candidate2
        # Check <stem>.pkl
        candidate3 = p.with_suffix(".pkl")
        if candidate3.is_file():
            return candidate3
    return None


def get_graph_summary(path: Path | str) -> dict[str, Any]:
    """Extract a summary of node/edge/image counts from either bundle, JSON, or HTML file."""
    p = Path(path).resolve()
    if not p.exists():
        return {"valid": False, "error": f"File does not exist: {p.name}"}

    # If it is a bundle file
    data_file = find_graph_data_file(p) if p.suffix.lower() != ".pkl" else p
    if data_file and data_file.is_file():
        try:
            bundle = load_graph_data(data_file)
            g: nx.Graph = bundle["graph"]
            meta = bundle.get("metadata", {})
            img_records = bundle.get("image_records", [])
            return {
                "valid": True,
                "has_bundle": True,
                "bundle_path": str(data_file),
                "nodes": len(g.nodes),
                "edges": len(g.edges),
                "images": len(bundle.get("processed_images", [])),
                "faces": len(bundle.get("face_records", [])),
                "has_image_records": bool(img_records),
                "image_records_count": len(img_records),
                "model": bundle.get("params", {}).get("model", "unknown"),
                "created_at": bundle.get("created_at", meta.get("created_at", "unknown")),
            }
        except Exception:
            pass

    # If it is a JSON file
    if p.suffix.lower() == ".json":
        try:
            g, crops, meta = load_graph_from_json(p)
            return {
                "valid": True,
                "has_bundle": False,
                "bundle_path": None,
                "nodes": len(g.nodes),
                "edges": len(g.edges),
                "images": meta.get("image_count", "N/A"),
                "faces": meta.get("faces_detected", "N/A"),
                "model": "N/A",
                "created_at": meta.get("generated_at", "unknown"),
            }
        except Exception as e:
            return {"valid": False, "error": f"Invalid JSON graph: {e}"}

    # If HTML file, check for companion JSON
    if p.suffix.lower() == ".html":
        json_cand = p.with_suffix(".json")
        if json_cand.is_file():
            return get_graph_summary(json_cand)

    return {"valid": False, "error": f"Unsupported or unrecognizable graph file: {p.name}"}
