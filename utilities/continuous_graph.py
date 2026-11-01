"""Continuous Social Graph creation utilities for Social Eye.

Allows loading an existing social graph state bundle, scanning and detecting faces
in additional images, merging features, re-clustering identities, and updating
interactive HTML & JSON graphs along with the serialized state bundle.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import sys
from typing import Any, Callable

import cv2
from insightface.app import FaceAnalysis
import networkx as nx
import numpy as np

# Add project root to sys.path if not present
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from create_social_graph import (
    cluster_identities,
    collect_events,
    construct_graph,
    export_graph_html,
    export_graph_json,
    extract_image_faces,
    extract_profile_faces,
    match_profiles_to_identities,
)
from utilities.graph_storage import (
    build_image_records,
    build_person_crops,
    crop_to_data_uri,
    ensure_image_records,
    find_graph_data_file,
    load_graph_data,
    save_graph_data,
    serialize_crop,
)


def prepare_bundle_node_crops(
    face_records: list[dict],
    labels: np.ndarray,
    cluster_names: dict[int, str],
) -> dict[str, str]:
    """Extract representative face crop data URIs supporting both numpy arrays and JPEG bytes."""
    crops: dict[str, str] = {}
    for record, label in zip(face_records, labels):
        if label < 0:
            continue
        name = cluster_names[int(label)]
        if name in crops:
            continue

        raw_crop = record.get("crop")
        if raw_crop is not None:
            uri = crop_to_data_uri(raw_crop)
            if uri:
                crops[name] = uri
        elif record.get("crop_path"):
            cp = Path(record["crop_path"])
            if cp.is_file():
                try:
                    data = cp.read_bytes()
                    uri = crop_to_data_uri(data)
                    if uri:
                        crops[name] = uri
                except OSError:
                    pass
    return crops


def update_graph_with_images(
    graph_source: Path | str,
    new_images_dir: Path | str,
    *,
    new_profiles_dir: Path | str | None = None,
    output_html: Path | str | None = None,
    output_json: Path | str | None = None,
    output_data: Path | str | None = None,
    eps: float | None = None,
    min_samples: int | None = None,
    profile_distance: float | None = None,
    model: str | None = None,
    ctx_id: int | None = None,
    remove_isolated: bool | None = None,
    on_log: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_image: Callable[[Path, np.ndarray | None, list[tuple[int, int, int, int]], int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[Path, Path | None, Path]:
    """Load an existing graph bundle, process new images, and update the graph.

    Parameters
    ----------
    graph_source : Path or str
        Path to an existing .graph_data.pkl bundle, or a .json/.html with a companion bundle.
    new_images_dir : Path or str
        Directory containing new images to add to the graph.
    new_profiles_dir : Path or str, optional
        Optional folder with named profile photos to augment identity matching.
    output_html : Path or str, optional
        Destination path for updated HTML graph.
    output_json : Path or str, optional
        Destination path for updated JSON graph.
    output_data : Path or str, optional
        Destination path for updated .graph_data.pkl bundle.
    eps : float, optional
        DBSCAN eps override. If omitted, uses previous value.
    min_samples : int, optional
        DBSCAN min_samples override. If omitted, uses previous value.
    profile_distance : float, optional
        Profile distance override. If omitted, uses previous value.
    model : str, optional
        InsightFace model name override. If omitted, uses previous value.
    ctx_id : int, optional
        Device execution ID (-1 for CPU, 0+ for GPU). If omitted, uses previous value.
    remove_isolated : bool, optional
        Whether to remove isolated nodes from graph. If omitted, uses previous value.
    on_log, on_status, on_progress, on_image, is_cancelled : Callables, optional
        UI and progress hooks.

    Returns
    -------
    tuple[Path, Path | None, Path]
        (output_html_path, output_json_path, output_data_path)
    """
    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    def set_status(st: str):
        if on_status:
            on_status(st)

    def check_cancel():
        if is_cancelled and is_cancelled():
            raise InterruptedError("Operation cancelled by user.")

    src_path = Path(graph_source).resolve()
    bundle_file = find_graph_data_file(src_path) if src_path.suffix.lower() != ".pkl" else src_path
    if not bundle_file or not bundle_file.is_file():
        raise FileNotFoundError(
            f"Could not locate a graph data bundle (.graph_data.pkl) for: {src_path.name}\n"
            "A saved graph data bundle is required to continuously add new images."
        )

    set_status(f"Loading existing graph data bundle: {bundle_file.name}...")
    log(f"Loading previous graph data from: {bundle_file}")
    bundle = load_graph_data(bundle_file)

    prev_embeddings: np.ndarray = bundle["embeddings"]
    prev_face_records: list[dict] = bundle["face_records"]
    prev_image_face_counts: dict[str, int] = bundle["image_face_counts"]
    prev_processed_images: set[str] = set(bundle.get("processed_images", []))
    prev_profile_vectors: list[np.ndarray] = bundle.get("profile_vectors", [])
    prev_profile_names: list[str] = bundle.get("profile_names", [])
    prev_params: dict = bundle.get("params", {})

    log(f"Previous graph state: {len(prev_face_records)} faces across {len(prev_processed_images)} images.")

    # Resolve parameters with fallback to previous values
    eps_val = float(eps) if eps is not None else float(prev_params.get("eps", 0.45))
    min_samples_val = int(min_samples) if min_samples is not None else int(prev_params.get("min_samples", 2))
    profile_distance_val = float(profile_distance) if profile_distance is not None else float(prev_params.get("profile_distance", 0.55))
    model_val = str(model) if model is not None else str(prev_params.get("model", "buffalo_l"))
    ctx_id_val = int(ctx_id) if ctx_id is not None else int(prev_params.get("ctx_id", -1))
    remove_isolated_val = bool(remove_isolated) if remove_isolated is not None else bool(prev_params.get("remove_isolated", True))

    # Resolve output destinations
    if output_html:
        out_html = Path(output_html).resolve()
    else:
        out_html = bundle_file.parent / f"{bundle_file.stem}_updated.html"

    if output_json:
        out_json = Path(output_json).resolve()
    else:
        out_json = out_html.with_suffix(".json")

    if output_data:
        out_data = Path(output_data).resolve()
    else:
        out_data = out_html.parent / f"{out_html.stem}.graph_data.pkl"

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_data.parent.mkdir(parents=True, exist_ok=True)

    # Scan new images
    new_img_root = Path(new_images_dir).resolve()
    if not new_img_root.is_dir():
        raise ValueError(f"New images directory does not exist: {new_img_root}")

    all_candidate_events = collect_events(new_img_root)
    log(f"Found {len(all_candidate_events)} candidate image(s) in {new_img_root}")

    # Normalize paths for comparison
    normalized_prev_images = {str(Path(p).resolve()).lower() for p in prev_processed_images}

    new_event_images: list[tuple[str, Path]] = []
    for event, path in all_candidate_events:
        norm_p = str(path.resolve()).lower()
        if norm_p not in normalized_prev_images:
            new_event_images.append((event, path))

    skipped_count = len(all_candidate_events) - len(new_event_images)
    if skipped_count > 0:
        log(f"Skipping {skipped_count} image(s) already processed in previous graph.")

    if not new_event_images and not new_profiles_dir:
        log("No new images or profiles to process. Graph already up to date.")
        return out_html, out_json, out_data

    # Initialize FaceAnalysis
    ctx_str = "CPU" if ctx_id_val == -1 else f"GPU device {ctx_id_val}"
    set_status(f"Loading face model '{model_val}' on {ctx_str}...")
    log(f"Loading face analysis model '{model_val}' ({ctx_str})...")
    app = FaceAnalysis(name=model_val)
    app.prepare(ctx_id=ctx_id_val, det_size=(640, 640))

    new_vectors: list[np.ndarray] = []
    new_records: list[dict] = []
    new_counts: dict[Path, int] = {}

    if new_event_images:
        set_status(f"Scanning {len(new_event_images)} new images...")
        log(f"Detecting faces in {len(new_event_images)} new images...")
        new_vectors, new_records, new_counts = extract_image_faces(
            event_images=new_event_images,
            app=app,
            disk_storage=False,
            on_log=on_log,
            on_progress=on_progress,
            on_image=on_image,
            is_cancelled=is_cancelled,
        )
        check_cancel()
        log(f"Extracted {len(new_records)} faces from {len(new_event_images)} new images.")

    # Process additional profiles if supplied
    all_profile_vectors = list(prev_profile_vectors)
    all_profile_names = list(prev_profile_names)
    if new_profiles_dir:
        profiles_root = Path(new_profiles_dir).resolve()
        if profiles_root.is_dir():
            set_status("Scanning additional profile pictures...")
            p_vecs, p_names = extract_profile_faces(
                profiles_root=profiles_root,
                app=app,
                on_log=on_log,
                on_progress=on_progress,
                on_image=on_image,
                is_cancelled=is_cancelled,
            )
            check_cancel()
            all_profile_vectors.extend(p_vecs)
            all_profile_names.extend(p_names)

    # Combine existing + new data
    if new_vectors:
        all_embeddings = np.vstack([prev_embeddings, np.stack(new_vectors)])
        # Store newly detected crops as compressed JPEG bytes
        for rec in new_records:
            if rec.get("crop") is not None:
                rec["crop"] = serialize_crop(rec["crop"])
        all_face_records = prev_face_records + new_records
    else:
        all_embeddings = prev_embeddings
        all_face_records = prev_face_records

    all_image_face_counts = dict(prev_image_face_counts)
    for p_path, cnt in new_counts.items():
        all_image_face_counts[str(p_path.resolve())] = cnt

    all_processed_images = set(prev_processed_images)
    for _, p_path in new_event_images:
        all_processed_images.add(str(p_path.resolve()))

    check_cancel()
    set_status(f"Clustering combined {len(all_face_records)} faces (DBSCAN)...")
    log(f"Re-clustering total of {len(all_face_records)} faces across {len(all_processed_images)} images into identities...")
    labels = cluster_identities(all_embeddings, eps=eps_val, min_samples=min_samples_val)

    check_cancel()
    # Match profiles
    cluster_names = match_profiles_to_identities(
        embeddings=all_embeddings,
        labels=labels,
        profile_vectors=all_profile_vectors,
        profile_names=all_profile_names,
        profile_distance=profile_distance_val,
        on_log=on_log,
    )

    check_cancel()
    # Extract representative crops (Part 1: Highest quality images mapped to each person ID)
    person_crops, crops = build_person_crops(all_face_records, labels, cluster_names)
    if not crops:
        crops = prepare_bundle_node_crops(all_face_records, labels, cluster_names)

    # Build standalone image records (Part 2: each row is an image containing people IDs as a list)
    previous_event_by_path = {}
    for row in bundle.get("image_records", []):
        row_path = row.get("image_path")
        if row_path:
            previous_event_by_path[str(Path(row_path).resolve()).lower()] = row.get("event", "")
    previous_event_images = [
        (
            previous_event_by_path.get(str(Path(image_path).resolve()).lower(), Path(image_path).parent.name),
            Path(image_path),
        )
        for image_path in prev_processed_images
    ]
    image_records = build_image_records(
        event_images=previous_event_images + new_event_images,
        face_records=all_face_records,
        labels=labels,
        cluster_names=cluster_names,
        image_face_counts=all_image_face_counts,
    )
    check_cancel()
    # Construct updated graph
    set_status("Rebuilding updated social network graph...")
    graph = construct_graph(
        face_records=all_face_records,
        labels=labels,
        cluster_names=cluster_names,
        image_face_counts=all_image_face_counts,
        remove_isolated=remove_isolated_val,
        image_records=image_records,
        on_log=on_log,
    )

    check_cancel()
    # Export updated JSON
    set_status("Exporting updated JSON graph...")
    export_graph_json(
        graph=graph,
        crops=crops,
        output_path=out_json,
        metadata={
            "image_count": len(all_processed_images),
            "faces_detected": len(all_face_records),
            "cluster_count": len(cluster_names),
            "removed_isolated": remove_isolated_val,
            "continuous_update": True,
            "image_records_count": len(image_records),
        },
    )
    log(f"Updated JSON graph saved to: {out_json}")

    # Export updated HTML
    set_status("Exporting updated HTML visualization...")
    export_graph_html(graph=graph, crops=crops, output_path=out_html)
    log(f"Updated HTML graph saved to: {out_html}")

    # Save updated data bundle
    set_status("Saving updated graph data bundle...")
    save_graph_data(
        out_data,
        graph=graph,
        crops=crops,
        embeddings=all_embeddings,
        face_records=all_face_records,
        image_face_counts=all_image_face_counts,
        labels=labels,
        cluster_names=cluster_names,
        image_records=image_records,
        person_crops=person_crops,
        profile_vectors=all_profile_vectors,
        profile_names=all_profile_names,
        processed_images=all_processed_images,
        params={
            "eps": eps_val,
            "min_samples": min_samples_val,
            "profile_distance": profile_distance_val,
            "model": model_val,
            "ctx_id": ctx_id_val,
            "remove_isolated": remove_isolated_val,
        },
        metadata={
            "previous_bundle": str(bundle_file),
            "new_images_added": len(new_event_images),
            "new_faces_added": len(new_records),
            "image_records_count": len(image_records),
            "standalone": True,
        },
    )
    log(f"Updated graph data bundle saved to: {out_data}")

    set_status("Update complete!")
    log(f"Successfully updated graph: now has {len(graph.nodes)} people and {len(graph.edges)} connections.")
    return out_html, out_json, out_data


def rebuild_graph_from_bundle(
    graph_source: Path | str | dict[str, Any],
    *,
    output_html: Path | str | None = None,
    output_json: Path | str | None = None,
    output_bundle: Path | str | None = None,
    small_group_weight: float = 5.0,
    small_group_max_faces: int = 3,
    large_group_weight: float = 2.0,
    event_weight: float = 1.0,
    event_popularity_multiplier: float = 50.0,
    degree_popularity_multiplier: float = 1.0,
    remove_isolated: bool = True,
    on_log: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[nx.Graph, Path | None, Path | None, Path | None]:
    """Rebuild social graph from a standalone bundle using the formula defined in code.

    Operates in milliseconds without running face detection or DBSCAN clustering.
    Uses Part 1 (highest quality person crops) and Part 2 (image rows with people IDs).

    Parameters
    ----------
    graph_source : Path, str, or dict
        Path to .graph_data.pkl (or companion html/json), or pre-loaded bundle dict.
    output_html : Path or str, optional
        Destination for interactive HTML.
    output_json : Path or str, optional
        Destination for JSON graph.
    output_bundle : Path or str, optional
        Destination for updated .graph_data.pkl bundle.
    small_group_weight, small_group_max_faces, large_group_weight, event_weight,
    event_popularity_multiplier, degree_popularity_multiplier, remove_isolated : float/int/bool
        Formula configuration parameters.
    """
    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    def set_status(st: str):
        if on_status:
            on_status(st)

    def check_cancel():
        if is_cancelled and is_cancelled():
            raise InterruptedError("Operation cancelled by user.")

    set_status("Loading standalone graph bundle...")
    if isinstance(graph_source, dict):
        bundle = dict(graph_source)
        source_name = "in-memory bundle"
        bundle_file = None
    else:
        src_path = Path(graph_source).resolve()
        bundle_file = find_graph_data_file(src_path) if src_path.suffix.lower() != ".pkl" else src_path
        if not bundle_file or not bundle_file.is_file():
            raise FileNotFoundError(f"Could not locate graph data bundle (.graph_data.pkl) for: {src_path}")
        source_name = bundle_file.name
        log(f"Loading standalone bundle: {bundle_file.name}")
        bundle = load_graph_data(bundle_file)

    check_cancel()
    # Part 2: Standalone image records where each row is an image containing people IDs
    image_records = ensure_image_records(bundle)
    cluster_names = bundle.get("cluster_names", {})

    # Part 1: Highest quality person crops
    crops = bundle.get("crops", {})
    if not crops and "person_crops" in bundle:
        crops = {k: v.get("data_uri") for k, v in bundle["person_crops"].items() if v.get("data_uri")}

    log(f"Loaded {len(image_records)} standalone image records across {len(cluster_names)} person identities.")
    log(f"Applying formula: small_group={small_group_weight} (<= {small_group_max_faces} faces), large_group={large_group_weight}, event_weight={event_weight}, event_pop_mult={event_popularity_multiplier}, degree_pop_mult={degree_popularity_multiplier}, remove_isolated={remove_isolated}")

    set_status("Constructing social network with the current code formula...")
    graph = construct_graph(
        cluster_names=cluster_names,
        remove_isolated=remove_isolated,
        image_records=image_records,
        small_group_weight=small_group_weight,
        small_group_max_faces=small_group_max_faces,
        large_group_weight=large_group_weight,
        event_weight=event_weight,
        event_popularity_multiplier=event_popularity_multiplier,
        degree_popularity_multiplier=degree_popularity_multiplier,
        on_log=on_log,
    )

    check_cancel()
    out_html_path: Path | None = None
    out_json_path: Path | None = None
    out_bundle_path: Path | None = None

    if output_html:
        out_html_path = Path(output_html).resolve()
        out_html_path.parent.mkdir(parents=True, exist_ok=True)
        set_status("Exporting interactive HTML visualization...")
        export_graph_html(graph=graph, crops=crops, output_path=out_html_path)
        log(f"Interactive HTML graph exported to: {out_html_path}")

    if output_json:
        out_json_path = Path(output_json).resolve()
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        set_status("Exporting JSON graph...")
        export_graph_json(
            graph=graph,
            crops=crops,
            output_path=out_json_path,
            metadata={
                "rebuilt_from": source_name,
                "formula": {
                    "small_group_weight": small_group_weight,
                    "small_group_max_faces": small_group_max_faces,
                    "large_group_weight": large_group_weight,
                    "event_weight": event_weight,
                    "event_popularity_multiplier": event_popularity_multiplier,
                    "degree_popularity_multiplier": degree_popularity_multiplier,
                },
                "image_records_count": len(image_records),
            },
        )
        log(f"JSON graph exported to: {out_json_path}")

    if output_bundle:
        out_bundle_path = Path(output_bundle).resolve()
        out_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        set_status("Saving updated standalone graph bundle...")
        params = dict(bundle.get("params", {}))
        params["formula"] = {
            "small_group_weight": small_group_weight,
            "small_group_max_faces": small_group_max_faces,
            "large_group_weight": large_group_weight,
            "event_weight": event_weight,
            "event_popularity_multiplier": event_popularity_multiplier,
            "degree_popularity_multiplier": degree_popularity_multiplier,
        }
        meta = dict(bundle.get("metadata", {}))
        meta["rebuilt_from"] = source_name
        meta["rebuilt_at"] = datetime.datetime.now().isoformat()
        save_graph_data(
            out_bundle_path,
            graph=graph,
            crops=crops,
            embeddings=bundle.get("embeddings", np.empty((0, 512), dtype=np.float32)),
            face_records=bundle.get("face_records", []),
            image_face_counts=bundle.get("image_face_counts", {}),
            labels=bundle.get("labels", np.array([])),
            cluster_names=cluster_names,
            image_records=image_records,
            person_crops=bundle.get("person_crops"),
            profile_vectors=bundle.get("profile_vectors"),
            profile_names=bundle.get("profile_names"),
            processed_images=bundle.get("processed_images"),
            params=params,
            metadata=meta,
        )
        log(f"Updated standalone graph bundle saved to: {out_bundle_path}")

    set_status(f"Graph rebuild complete ({len(graph.nodes)} people, {len(graph.edges)} connections)!")
    return graph, out_html_path, out_json_path, out_bundle_path

    return out_html, out_json, out_data


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for continuous graph update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Path to existing graph (.graph_data.pkl or .json with companion bundle)")
    parser.add_argument("images", type=Path, help="Folder containing new images to add")
    parser.add_argument("-p", "--profiles", type=Path, default=None, help="Optional folder of new profile photos")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output updated HTML graph (default: <stem>_updated.html)")
    parser.add_argument("-j", "--json-output", type=Path, default=None, help="Output updated JSON graph (default: <output>.json)")
    parser.add_argument("-d", "--data-output", type=Path, default=None, help="Output updated graph bundle (default: <output>.graph_data.pkl)")
    parser.add_argument("--eps", type=float, default=None, help="DBSCAN cosine-distance threshold")
    parser.add_argument("--min-samples", type=int, default=None, help="Minimum faces per identity cluster")
    parser.add_argument("--profile-distance", type=float, default=None, help="Maximum cosine distance for profile matching")
    parser.add_argument("--model", default=None, help="InsightFace model pack")
    parser.add_argument("--ctx-id", type=int, default=None, help="Device execution id (-1 for CPU)")
    parser.add_argument("--keep-isolated", action="store_true", help="Keep isolated nodes without connections")
    return parser.parse_args()


def main():
    args = parse_args()
    remove_iso = not args.keep_isolated if args.keep_isolated else None
    update_graph_with_images(
        graph_source=args.graph,
        new_images_dir=args.images,
        new_profiles_dir=args.profiles,
        output_html=args.output,
        output_json=args.json_output,
        output_data=args.data_output,
        eps=args.eps,
        min_samples=args.min_samples,
        profile_distance=args.profile_distance,
        model=args.model,
        ctx_id=args.ctx_id,
        remove_isolated=remove_iso,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, OSError, InterruptedError) as err:
        sys.exit(f"Error: {err}")
