# Heru

**Heru** is a face recognition and social network analysis system that reconstructs interpersonal networks from photograph collections. It analyzes images using InsightFace, groups detected faces into individual identities using DBSCAN clustering, computes connection weights and person popularity scores, and exports interactive network visualizations.

---

## Codebase Structure

```text
Heru/
│
├── create_social_graph.py          # Core pipeline: detection, clustering, graph construction & export
├── create_social_graph_gui.py      # Tkinter desktop GUI for graph generation & management
├── download_images.py              # Batch image downloader from CSV URLs with concurrency
│
└── utilities/
    ├── __init__.py                 # Package exports for utility functions
    ├── continuous_graph.py         # Incremental graph update pipeline & bundle rebuild
    ├── graph_pruning.py            # Graph pruning based on node popularity scores
    ├── graph_storage.py            # Standalone bundle serialization (.graph_data.pkl) & JSON I/O
    └── rebuild_graph.py            # Instant graph rebuild tool from standalone image records
```

---

## Module Breakdown

### 1. `create_social_graph.py` (Core Pipeline)
The core script providing the end-to-end processing pipeline:
- **Event & Image Discovery (`collect_events`, `image_files`)**: Scans input folders. Images placed in first-level subdirectories are grouped by event name; images directly in the root folder are treated as a single event.
- **Face Detection & Embeddings (`extract_image_faces`, `extract_profile_faces`)**: Runs InsightFace (`buffalo_l` or other models) to detect faces, extract normalized 512-dimensional embeddings, and save representative face crops. Supports on-disk caching via memory mapping for large datasets.
- **Identity Clustering (`cluster_identities`)**: Groups face embeddings into individual identities using DBSCAN with cosine distance.
- **Profile Matching (`match_profiles_to_identities`)**: Matches cluster centroids against an optional folder of labeled profile pictures to assign real names.
- **Graph Construction (`construct_graph`)**:
  - **Nodes**: Each identified person becomes a node in a NetworkX graph.
  - **Edge Weighting**:
    - Small-group photo co-occurrence ($\le 3$ people): **5 points**.
    - Large-group photo co-occurrence ($> 3$ people): **2 points**.
    - Event co-occurrence (appearing in the same event): **1 point**.
  - **Popularity Score**:
    Each person is assigned a popularity score calculated as:
    $$\text{Popularity} = (\text{Distinct Events} \times 50) + \text{In-Degree}$$
    where distinct events is the number of unique events the person appeared in, and in-degree is their number of connections in the graph.
- **Exports (`export_graph_html`, `export_graph_json`)**:
  - Interactive PyVis HTML visualization featuring circular face crops and hover tooltips showing popularity and connection counts.
  - Structured JSON format containing node metadata (coordinates, degree, popularity, event count) and edge weights.
  - Complete graph state bundle (`.graph_data.pkl`) for continuous updates and pruning.

### 2. `create_social_graph_gui.py` (Desktop GUI)
A Tkinter application (`SocialGraphGUI`) providing a visual interface:
- **Create Graph Tab**:
  - Folder selectors for images and optional profile photos.
  - Model and device options (CPU / GPU selection).
  - Clustering parameter tuning (DBSCAN $\varepsilon$, minimum samples, profile distance).
  - Real-time preview showing the current image being scanned with detected face bounding boxes.
  - Live log console and progress bar.
- **Manage Graph Tab**:
  - **Rebuild Graph**: Regenerates graph outputs from saved image records using the formula defined in code, without re-scanning or re-clustering.
  - **Image Records Viewer**: Inspects image hashes and the people IDs detected in each photo.
  - **Add New Images**: Incrementally expands an existing graph state bundle with new photos.
  - **Prune Graph**: Filters an existing graph down to the top $X$ most popular people.

### 3. `utilities/rebuild_graph.py` (Instant Graph Rebuild)
- Rebuilds an interactive HTML, JSON graph, and `.graph_data.pkl` bundle in milliseconds from standalone image records.
- Uses the graph formula defined in `create_social_graph.py`.
- Embeds co-occurrence photo evidence directly into graph edge tooltips and JSON payloads.

### 4. `utilities/graph_storage.py` (Standalone Bundle Serialization)
- **Standalone Two-Part Architecture**:
  1. **Person Crops (`person_crops` / `crops`)**: Maps each person ID to their highest-quality representative face crop, with quality scores, JPEG bytes, and data URIs.
  2. **Image Records (`image_records`)**: Each row stores an image hash, event, detected people IDs, face count, and face evidence metadata. Original image bytes are not copied into the bundle.
- **`save_graph_data` / `load_graph_data`**: Serializes and restores full standalone graph bundles (`.graph_data.pkl`).
- **`ensure_image_records`**: Automatically synthesizes image records for backward compatibility with older bundles.

### 5. `utilities/graph_pruning.py` (Pruning)
- **`prune_graph_to_top_popularity(source, top_x, ...)`**: Prunes a graph to retain only the top $X$ individuals with the highest popularity score.
- **Tie-Breaking**: Ties in popularity are broken by node degree, followed by sum of edge weights, then alphabetical name.
- **Backwards Compatibility**: Exposes `prune_graph_to_top_degree` as an alias.
- Supports input from `.graph_data.pkl` state bundles, `.json` graph files, or `.html` graphs with companion files.

### 6. `utilities/continuous_graph.py` (Incremental Updates & Rebuilding)
- **`update_graph_with_images(...)`**: Extends an existing social network with new images without re-scanning previously processed images.
- **`rebuild_graph_from_bundle(...)`**: Re-evaluates graph structure and weights from standalone image records using the formula defined in code.

### 7. `download_images.py` (Data Collection Utility)
- Standalone multi-threaded image downloader that parses gallery links from CSV files, groups downloads into folders by gallery title, and handles network retries.

---

## Quick Start

### 1. Launch the GUI
```bash
python create_social_graph_gui.py
```
Open the **Manage Graph** tab to rebuild a graph from an existing `.graph_data.pkl` file or inspect its image records.

### 2. Build a Graph via CLI
```bash
# Basic run with flat or event-organized photos
python create_social_graph.py path/to/images

# With profile pictures for identity naming and custom output
python create_social_graph.py path/to/images --profiles path/to/profiles --output people_graph.html
```

### 3. Rebuild Graph (Instant Rerun)
```bash
# Rebuild from standalone bundle using the formula defined in code
python utilities/rebuild_graph.py people_graph.graph_data.pkl -o rebuilt.html

# Or rebuild via create_social_graph.py
python create_social_graph.py --rebuild people_graph.graph_data.pkl -o rebuilt.html
```

### 4. Prune to Top Popular Nodes
```bash
python utilities/graph_pruning.py people_graph.graph_data.pkl --top-x 20 --output people_top20.html
```

### 5. Add New Images to an Existing Graph
```bash
python utilities/continuous_graph.py people_graph.graph_data.pkl path/to/new_images --output people_updated.html
```
