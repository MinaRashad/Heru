# Social Eye

**Social Eye** is a face recognition and social network analysis system that reconstructs interpersonal networks from photograph collections. It analyzes images using InsightFace, groups detected faces into individual identities using DBSCAN clustering, computes connection weights and person popularity scores, and exports interactive network visualizations.

---

## Codebase Structure

```text
Social Eye/
│
├── create_social_graph.py          # Core pipeline: detection, clustering, graph construction & export
├── create_social_graph_gui.py      # Tkinter desktop GUI for graph generation & management
├── download_images.py              # Batch image downloader from CSV URLs with concurrency
│
└── utilities/
    ├── __init__.py                 # Package exports for utility functions
    ├── continuous_graph.py         # Incremental graph update pipeline (adds new images to existing graphs)
    ├── graph_pruning.py            # Graph pruning based on node popularity scores
    └── graph_storage.py            # Bundle serialization (.graph_data.pkl), caching, and JSON I/O
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
  - **Add New Images**: Incrementally expands an existing graph state bundle with new photos.
  - **Prune Graph**: Filters an existing graph down to the top $X$ most popular people.

### 3. `utilities/graph_pruning.py` (Pruning)
- **`prune_graph_to_top_popularity(source, top_x, ...)`**: Prunes a graph to retain only the top $X$ individuals with the highest popularity score.
- **Tie-Breaking**: Ties in popularity are broken by node degree, followed by sum of edge weights, then alphabetical name.
- **Backwards Compatibility**: Exposes `prune_graph_to_top_degree` as an alias.
- Supports input from `.graph_data.pkl` state bundles, `.json` graph files, or `.html` graphs with companion files.

### 4. `utilities/continuous_graph.py` (Incremental Updates)
- **`update_graph_with_images(...)`**: Extends an existing social network with new images without re-scanning previously processed images.
- Merges previously saved embeddings with new face vectors, re-runs DBSCAN clustering, and regenerates updated HTML, JSON, and state bundle files.

### 5. `utilities/graph_storage.py` (Persistence & Serialization)
- **`save_graph_data` / `load_graph_data`**: Serializes and restores full graph bundles (`.graph_data.pkl`), including face embeddings, face crops, cluster labels, image counts, and NetworkX graph structures.
- **`load_graph_from_json`**: Reconstructs NetworkX graphs and node attributes directly from exported JSON files.
- **`crop_to_data_uri` / `serialize_crop`**: Handles compression and Base64 conversion of face crops.

### 6. `download_images.py` (Data Collection Utility)
- Standalone multi-threaded image downloader that parses gallery links from CSV files, groups downloads into folders by gallery title, and handles network retries.

---

## Quick Start

### 1. Launch the GUI
```bash
python create_social_graph_gui.py
```

### 2. Build a Graph via CLI
```bash
# Basic run with flat or event-organized photos
python create_social_graph.py path/to/images

# With profile pictures for identity naming and custom output
python create_social_graph.py path/to/images --profiles path/to/profiles --output people_graph.html
```

### 3. Prune to Top Popular Nodes
```bash
python utilities/graph_pruning.py people_graph.graph_data.pkl --top-x 20 --output people_top20.html
```

### 4. Add New Images to an Existing Graph
```bash
python utilities/continuous_graph.py people_graph.graph_data.pkl path/to/new_images --output people_updated.html
```
