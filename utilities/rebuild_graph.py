"""CLI tool to rebuild a social graph from a standalone graph bundle.

Instantly reconstructs connection weights, evidence photos, and popularity scores
from standalone image records (images mapped to people IDs) without re-running
face detection or clustering.

Usage:
    python -m utilities.rebuild_graph people_graph.graph_data.pkl
    python utilities/rebuild_graph.py people_graph.graph_data.pkl -o custom_graph.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utilities.continuous_graph import rebuild_graph_from_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Path to input .graph_data.pkl bundle, .json, or .html")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output HTML graph file (default: <stem>_rebuilt.html)")
    parser.add_argument("-j", "--json-output", type=Path, default=None, help="Output JSON graph file (default: <stem>_rebuilt.json)")
    parser.add_argument("-d", "--data-output", type=Path, default=None, help="Output updated .graph_data.pkl bundle (default: <stem>_rebuilt.graph_data.pkl)")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.graph.resolve()
    stem = input_path.stem.replace(".graph_data", "")

    out_html = args.output.resolve() if args.output else input_path.parent / f"{stem}_rebuilt.html"
    out_json = args.json_output.resolve() if args.json_output else out_html.with_suffix(".json")
    out_bundle = args.data_output.resolve() if args.data_output else out_html.with_name(f"{out_html.stem}.graph_data.pkl")

    graph, html_path, json_path, bundle_path = rebuild_graph_from_bundle(
        input_path,
        output_html=out_html,
        output_json=out_json,
        output_bundle=out_bundle,
    )
    print(f"\nDone! Rebuilt social graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    if html_path:
        print(f"HTML:   {html_path}")
    if json_path:
        print(f"JSON:   {json_path}")
    if bundle_path:
        print(f"Bundle: {bundle_path}")


if __name__ == "__main__":
    main()
