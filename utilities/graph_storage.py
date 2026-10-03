"""Storage and serialization utilities for Social Eye graph data.

Handles saving and loading the complete state required to continuously update
a social graph (embeddings, face records, image counts, profile vectors, metadata,
and network structure) or load graph data for pruning.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
from pathlib import Path
import pickle
from typing import Any

import cv2
import networkx as nx
import numpy as np


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
    profile_vectors: list[np.ndarray] | None = None,
    profile_names: list[str] | None = None,
    processed_images: set[str | Path] | None = None,
    params: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save full graph state and features to disk so graph can be updated later."""
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
        compact_records.append(rec)

    # Normalize image_face_counts keys to strings
    normalized_counts = {str(Path(k).resolve()): v for k, v in image_face_counts.items()}

    # Normalize processed_images set to strings
    if processed_images is not None:
        proc_imgs = {str(Path(p).resolve()) for p in processed_images}
    else:
        proc_imgs = set(normalized_counts.keys())

    # Ensure embeddings are numpy array float32
    if isinstance(embeddings, np.memmap):
        emb_array = np.array(embeddings, dtype=np.float32)
    else:
        emb_array = np.asarray(embeddings, dtype=np.float32)

    payload = {
        "version": 1,
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
        "profile_vectors": profile_vectors or [],
        "profile_names": profile_names or [],
        "params": params or {},
        "metadata": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "faces_detected": len(compact_records),
            "image_count": len(proc_imgs),
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
            return {
                "valid": True,
                "has_bundle": True,
                "bundle_path": str(data_file),
                "nodes": len(g.nodes),
                "edges": len(g.edges),
                "images": len(bundle.get("processed_images", [])),
                "faces": len(bundle.get("face_records", [])),
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
