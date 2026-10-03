"""Graph pruning utilities for Social Eye.

Provides functionality to prune social networks to the top X highest-degree nodes
(the most connected individuals) and export the pruned network to HTML, JSON,
and serialized graph bundles.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import networkx as nx

# Add project root to sys.path if not present
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from create_social_graph import export_graph_html, export_graph_json
from utilities.graph_storage import (
    find_graph_data_file,
    load_graph_data,
    load_graph_from_json,
    save_graph_data,
)


def prune_graph_to_top_degree(
    source: Path | str | nx.Graph,
    top_x: int,
    *,
    crops: dict[str, str] | None = None,
    output_html: Path | str | None = None,
    output_json: Path | str | None = None,
    output_data: Path | str | None = None,
    remove_isolated: bool = False,
    on_log: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[nx.Graph, Path | None, Path | None, Path | None]:
    """Prune graph to retain only the top X nodes with the highest degree (most connections).

    Parameters
    ----------
    source : Path, str, or nx.Graph
        Source graph. Can be a NetworkX Graph, or path to a .graph_data.pkl, .json, or .html file.
    top_x : int
        Number of top degree nodes to keep (must be >= 1).
    crops : dict[str, str], optional
        Mapping of person name -> base64 crop URI (used if source is nx.Graph).
    output_html : Path or str, optional
        Destination path for pruned interactive HTML visualization.
    output_json : Path or str, optional
        Destination path for pruned JSON graph.
    output_data : Path or str, optional
        Destination path for pruned .graph_data.pkl bundle.
    remove_isolated : bool, default False
        Whether to remove nodes that become isolated after pruning edges to omitted nodes.
    on_log : Callable, optional
        Callback for logging messages.
    on_status : Callable, optional
        Callback for status updates.

    Returns
    -------
    tuple[nx.Graph, Path | None, Path | None, Path | None]
        (pruned_graph, output_html_path, output_json_path, output_data_path)
    """
    def log(msg: str):
        if on_log:
            on_log(msg)
        else:
            print(msg)

    def set_status(st: str):
        if on_status:
            on_status(st)

    if top_x < 1:
        raise ValueError(f"top_x must be an integer >= 1, got {top_x}")

    bundle: dict | None = None
    node_crops: dict[str, str] = crops or {}

    set_status("Loading graph for pruning...")
    if isinstance(source, nx.Graph):
        graph = source.copy()
    else:
        src_path = Path(source).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")

        # Check for bundle
        bundle_file = find_graph_data_file(src_path) if src_path.suffix.lower() != ".pkl" else src_path
        if bundle_file and bundle_file.is_file():
            log(f"Loading graph from bundle: {bundle_file.name}")
            bundle = load_graph_data(bundle_file)
            graph = bundle["graph"].copy()
            node_crops = bundle.get("crops", {})
        elif src_path.suffix.lower() == ".json":
            log(f"Loading graph from JSON: {src_path.name}")
            graph, json_crops, _ = load_graph_from_json(src_path)
            node_crops = json_crops
        elif src_path.suffix.lower() == ".html":
            json_candidate = src_path.with_suffix(".json")
            if json_candidate.is_file():
                log(f"Loading graph from companion JSON: {json_candidate.name}")
                graph, json_crops, _ = load_graph_from_json(json_candidate)
                node_crops = json_crops
            else:
                raise ValueError(f"Cannot prune from HTML without companion .json or .graph_data.pkl file: {src_path}")
        else:
            raise ValueError(f"Unsupported graph source format: {src_path}")

    total_nodes = len(graph.nodes)
    total_edges = len(graph.edges)
    log(f"Input graph contains {total_nodes} nodes and {total_edges} connections.")

    if total_nodes == 0:
        raise ValueError("Cannot prune an empty graph (0 nodes).")

    set_status(f"Pruning graph to top {top_x} degree nodes...")
    if top_x >= total_nodes:
        log(f"Graph node count ({total_nodes}) is <= top {top_x}. No nodes removed.")
        pruned_graph = graph.copy()
        kept_nodes = set(graph.nodes)
    else:
        # Rank nodes by degree descending; break ties by sum of edge weights descending, then node name
        def sort_key(n: str):
            deg = graph.degree[n]
            weighted_deg = sum(d.get("weight", 1) for _, _, d in graph.edges(n, data=True))
            return (-deg, -weighted_deg, str(n))

        sorted_nodes = sorted(graph.nodes, key=sort_key)
        kept_nodes = set(sorted_nodes[:top_x])
        pruned_nodes = set(sorted_nodes[top_x:])

        log(f"Keeping top {len(kept_nodes)} most connected people, pruning {len(pruned_nodes)} people:")
        for idx, n in enumerate(sorted_nodes[:top_x], start=1):
            deg = graph.degree[n]
            log(f"  #{idx}: {n} (degree = {deg})")

        pruned_graph = graph.subgraph(kept_nodes).copy()

    if remove_isolated:
        isolated = list(nx.isolates(pruned_graph))
        if isolated:
            pruned_graph.remove_nodes_from(isolated)
            log(f"Removed {len(isolated)} isolated node(s) after pruning.")

    pruned_node_count = len(pruned_graph.nodes)
    pruned_edge_count = len(pruned_graph.edges)
    log(f"Pruned graph contains {pruned_node_count} nodes and {pruned_edge_count} connections.")

    pruned_crops = {n: node_crops[n] for n in pruned_graph.nodes if n in node_crops}

    # Export pruned HTML
    out_html_path: Path | None = None
    if output_html:
        set_status("Exporting pruned HTML visualization...")
        out_html_path = Path(output_html).resolve()
        export_graph_html(pruned_graph, pruned_crops, out_html_path)
        log(f"Pruned HTML graph exported to: {out_html_path}")

    # Export pruned JSON
    out_json_path: Path | None = None
    if output_json:
        set_status("Exporting pruned JSON graph...")
        out_json_path = Path(output_json).resolve()
        metadata = {
            "pruned": True,
            "pruned_top_x": top_x,
            "original_nodes": total_nodes,
            "original_edges": total_edges,
            "kept_nodes": pruned_node_count,
            "kept_edges": pruned_edge_count,
        }
        export_graph_json(pruned_graph, pruned_crops, out_json_path, metadata=metadata)
        log(f"Pruned JSON graph exported to: {out_json_path}")

    # Export pruned bundle if requested and original was a bundle
    out_data_path: Path | None = None
    if output_data and bundle:
        set_status("Exporting pruned graph data bundle...")
        out_data_path = Path(output_data).resolve()
        save_graph_data(
            out_data_path,
            graph=pruned_graph,
            crops=pruned_crops,
            embeddings=bundle["embeddings"],
            face_records=bundle["face_records"],
            image_face_counts=bundle["image_face_counts"],
            labels=bundle["labels"],
            cluster_names=bundle["cluster_names"],
            profile_vectors=bundle.get("profile_vectors"),
            profile_names=bundle.get("profile_names"),
            processed_images=bundle.get("processed_images"),
            params=bundle.get("params"),
            metadata={
                **bundle.get("metadata", {}),
                "pruned_top_x": top_x,
            },
        )
        log(f"Pruned graph bundle exported to: {out_data_path}")

    set_status("Pruning complete.")
    return pruned_graph, out_html_path, out_json_path, out_data_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pruning a graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Path to input graph (.graph_data.pkl, .json, or .html)")
    parser.add_argument("-x", "--top-x", type=int, required=True, help="Number of top degree nodes to keep")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output pruned HTML graph (default: <stem>_top<X>.html)")
    parser.add_argument("-j", "--json-output", type=Path, default=None, help="Output pruned JSON graph (default: <output>.json)")
    parser.add_argument("--remove-isolated", action="store_true", help="Remove nodes that become isolated after edge pruning")
    return parser.parse_args()


def main():
    args = parse_args()
    src = args.graph.resolve()
    top_x = args.top_x

    out_html = args.output
    if not out_html:
        out_html = src.parent / f"{src.stem}_top{top_x}.html"

    out_json = args.json_output
    if not out_json:
        out_json = out_html.with_suffix(".json")

    prune_graph_to_top_degree(
        source=src,
        top_x=top_x,
        output_html=out_html,
        output_json=out_json,
        remove_isolated=args.remove_isolated,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, OSError) as err:
        sys.exit(f"Error: {err}")
