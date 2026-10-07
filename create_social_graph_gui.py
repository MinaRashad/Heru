"""Graphical User Interface for create_social_graph.

Allows interactive configuration of all parameters with validation,
real-time preview of the image currently being scanned with face detections,
progress tracking, live logs, direct HTML graph preview, and JSON inspection.
"""

from __future__ import annotations

import os
from pathlib import Path
import queue
import sys
import threading
import webbrowser

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Import pipeline functions and constants from create_social_graph
from create_social_graph import (
    IMAGE_EXTENSIONS,
    build_graph,
    image_files,
)


class SocialGraphWorker(threading.Thread):
    """Background worker thread running graph generation via create_social_graph.build_graph."""

    def __init__(
        self,
        params: dict,
        ui_queue: queue.Queue,
        cancel_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.params = params
        self.ui_queue = ui_queue
        self.cancel_event = cancel_event

    def log(self, message: str):
        self.ui_queue.put(("log", message))

    def update_status(self, status: str):
        self.ui_queue.put(("status", status))

    def update_progress(self, current: int, total: int, message: str = ""):
        self.ui_queue.put(("progress", (current, total, message)))

    def update_image(
        self,
        path: Path,
        image: np.ndarray | None,
        bboxes: list[tuple[int, int, int, int]],
        face_count: int,
        event: str,
    ):
        self.ui_queue.put(("image_preview", (path, image, bboxes, face_count, event)))

    def run(self):
        try:
            html_path, json_path, bundle_path = build_graph(
                images=self.params["images"],
                profiles=self.params.get("profiles"),
                output=self.params["output"],
                json_output=self.params.get("json_output"),
                save_json=self.params.get("save_json", True),
                save_bundle=self.params.get("save_bundle", True),
                disk_storage=self.params.get("disk_storage", False),
                cache_dir=self.params.get("cache_dir"),
                grouped_faces=self.params.get("grouped_faces", False),
                remove_isolated=self.params.get("remove_isolated", True),
                eps=float(self.params.get("eps", 0.45)),
                min_samples=int(self.params.get("min_samples", 2)),
                profile_distance=float(self.params.get("profile_distance", 0.55)),
                model=self.params.get("model", "buffalo_l"),
                ctx_id=int(self.params.get("ctx_id", -1)),
                on_log=self.log,
                on_status=self.update_status,
                on_progress=self.update_progress,
                on_image=self.update_image,
                is_cancelled=lambda: self.cancel_event.is_set(),
            )
            self.ui_queue.put(
                (
                    "done",
                    (
                        str(html_path),
                        str(json_path) if json_path else None,
                        str(bundle_path) if bundle_path else None,
                    ),
                )
            )

        except InterruptedError as e:
            self.log(f"Stopped: {e}")
            self.update_status("Cancelled by user.")
            self.ui_queue.put(("cancelled", str(e)))
        except Exception as e:
            self.log(f"Error encountered: {e}")
            self.update_status(f"Error: {e}")
            self.ui_queue.put(("error", str(e)))


class SocialGraphGUI(tk.Tk):
    """Main application window for Create Social Graph GUI."""

    def __init__(self):
        super().__init__()
        self.title("Social Eye - Create Social Graph")
        self.geometry("1100x740")
        self.minsize(900, 580)

        # Apply modern style theme
        self.style = ttk.Style(self)
        available_themes = self.style.theme_names()
        if "clam" in available_themes:
            self.style.theme_use("clam")

        self.ui_queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker_thread: SocialGraphWorker | None = None
        self.last_output_path: str | None = None
        self.last_json_path: str | None = None
        self.last_bundle_path: str | None = None
        self.preview_image_ref = None

        # Manage-tab background worker
        self.manage_worker_thread: threading.Thread | None = None
        self.manage_cancel_event = threading.Event()

        self._configure_styles()
        self._build_ui()
        self._set_default_values()

        # Start periodic queue polling
        self.after(50, self._process_queue)

    def _configure_styles(self):
        self.configure(bg="#f4f6f9")
        self.style.configure(".", background="#f4f6f9", font=("Segoe UI", 9))
        self.style.configure("TFrame", background="#f4f6f9")
        self.style.configure("Card.TFrame", background="#ffffff", relief="flat")
        self.style.configure(
            "Header.TLabel",
            font=("Segoe UI", 14, "bold"),
            background="#f4f6f9",
            foreground="#1a202c",
        )
        self.style.configure(
            "Section.TLabel",
            font=("Segoe UI", 10, "bold"),
            background="#ffffff",
            foreground="#2b6cb0",
        )
        self.style.configure(
            "FieldLabel.TLabel",
            font=("Segoe UI", 9),
            background="#ffffff",
            foreground="#4a5568",
        )
        self.style.configure(
            "HelpText.TLabel",
            font=("Segoe UI", 8),
            background="#ffffff",
            foreground="#718096",
        )
        self.style.configure(
            "Status.TLabel",
            font=("Segoe UI", 9, "bold"),
            background="#ffffff",
            foreground="#2b6cb0",
        )
        self.style.configure(
            "Scanning.TLabel",
            font=("Segoe UI", 9),
            background="#ffffff",
            foreground="#1a202c",
        )

        self.style.configure(
            "Primary.TButton", font=("Segoe UI", 10, "bold"), padding=6
        )
        self.style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("Success.TButton", font=("Segoe UI", 9, "bold"), padding=5)

    def _build_ui(self):
        # Top banner
        banner = ttk.Frame(self, padding=(16, 12, 16, 8))
        banner.pack(side="top", fill="x")

        title_lbl = ttk.Label(
            banner, text="Social Eye Graph Generator", style="Header.TLabel"
        )
        title_lbl.pack(anchor="w")
        subtitle_lbl = ttk.Label(
            banner,
            text="Detect faces with InsightFace, cluster identities with DBSCAN, and build interactive HTML & JSON graphs.",
            font=("Segoe UI", 9),
            foreground="#4a5568",
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # Bottom Bar: MUST be packed side="bottom" BEFORE notebook so it is ALWAYS visible and never pushed off-screen
        self._build_bottom_bar()

        # Main notebook (tabs): packed side="top", fill="both", expand=True
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="top", fill="both", expand=True, padx=14, pady=(2, 0))

        # Tab 1: Create Graph
        create_tab = ttk.Frame(self.notebook, padding=0)
        self.notebook.add(create_tab, text="  Create Graph  ")
        self._build_create_tab(create_tab)

        # Tab 2: Manage Graph
        manage_tab = ttk.Frame(self.notebook, padding=0)
        self.notebook.add(manage_tab, text="  Manage Graph  ")
        self._build_manage_tab(manage_tab)

    def _build_create_tab(self, parent):
        """Build the Create Graph tab content (parameters + live preview)."""
        main_paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True)

        left_container = ttk.Frame(main_paned, width=470)
        left_container.pack_propagate(False)
        main_paned.add(left_container, weight=1)
        self._build_parameters_pane(left_container)

        right_container = ttk.Frame(main_paned, width=580)
        right_container.pack_propagate(False)
        main_paned.add(right_container, weight=2)
        self._build_preview_and_logs_pane(right_container)

    def _build_parameters_pane(self, parent):
        canvas = tk.Canvas(
            parent, bg="#ffffff", highlightthickness=1, highlightbackground="#e2e8f0"
        )
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="Card.TFrame", padding=14)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_canvas_configure(event):
            canvas.itemconfig(window_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(xscrollcommand=None, yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        row = 0
        # Section 1: Paths & Files
        sec1 = ttk.Label(
            scrollable_frame, text="1. Paths & Folders", style="Section.TLabel"
        )
        sec1.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # Images folder
        lbl_img = ttk.Label(
            scrollable_frame, text="Images Folder *:", style="FieldLabel.TLabel"
        )
        lbl_img.grid(row=row, column=0, sticky="w", pady=2)
        self.entry_images = ttk.Entry(scrollable_frame, width=28)
        self.entry_images.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        btn_browse_img = ttk.Button(
            scrollable_frame, text="Browse...", width=9, command=self._browse_images
        )
        btn_browse_img.grid(row=row, column=2, sticky="e", pady=2)
        row += 1
        h1 = ttk.Label(
            scrollable_frame,
            text="Root folder containing photos directly or in event subfolders.",
            style="HelpText.TLabel",
        )
        h1.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # Profiles folder
        lbl_prof = ttk.Label(
            scrollable_frame, text="Profiles Folder:", style="FieldLabel.TLabel"
        )
        lbl_prof.grid(row=row, column=0, sticky="w", pady=2)
        self.entry_profiles = ttk.Entry(scrollable_frame, width=28)
        self.entry_profiles.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        btn_browse_prof = ttk.Button(
            scrollable_frame, text="Browse...", width=9, command=self._browse_profiles
        )
        btn_browse_prof.grid(row=row, column=2, sticky="e", pady=2)
        row += 1
        h2 = ttk.Label(
            scrollable_frame,
            text="Optional folder of named profile photos to label identities.",
            style="HelpText.TLabel",
        )
        h2.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # Output HTML
        lbl_out = ttk.Label(
            scrollable_frame, text="Output HTML *:", style="FieldLabel.TLabel"
        )
        lbl_out.grid(row=row, column=0, sticky="w", pady=2)
        self.entry_output = ttk.Entry(scrollable_frame, width=28)
        self.entry_output.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        btn_browse_out = ttk.Button(
            scrollable_frame, text="Save As...", width=9, command=self._browse_output
        )
        btn_browse_out.grid(row=row, column=2, sticky="e", pady=2)
        row += 1
        h3 = ttk.Label(
            scrollable_frame,
            text="Destination path for the standalone interactive HTML graph.",
            style="HelpText.TLabel",
        )
        h3.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # Output JSON
        lbl_json = ttk.Label(
            scrollable_frame, text="Output JSON *:", style="FieldLabel.TLabel"
        )
        lbl_json.grid(row=row, column=0, sticky="w", pady=2)
        self.entry_json = ttk.Entry(scrollable_frame, width=28)
        self.entry_json.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        btn_browse_json = ttk.Button(
            scrollable_frame, text="Save As...", width=9, command=self._browse_json
        )
        btn_browse_json.grid(row=row, column=2, sticky="e", pady=2)
        row += 1
        h_json = ttk.Label(
            scrollable_frame,
            text="Destination path for the structured graph JSON file (nodes & edges).",
            style="HelpText.TLabel",
        )
        h_json.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        # Separator
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        # Section 2: Model & Execution Device
        sec2 = ttk.Label(
            scrollable_frame, text="2. Model & Execution Device", style="Section.TLabel"
        )
        sec2.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        lbl_model = ttk.Label(
            scrollable_frame, text="InsightFace Model:", style="FieldLabel.TLabel"
        )
        lbl_model.grid(row=row, column=0, sticky="w", pady=2)
        self.combo_model = ttk.Combobox(
            scrollable_frame,
            values=["buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2"],
            state="normal",
            width=26,
        )
        self.combo_model.grid(
            row=row, column=1, columnspan=2, sticky="w", padx=4, pady=2
        )
        row += 1
        h_model = ttk.Label(
            scrollable_frame,
            text="buffalo_l provides the highest face recognition accuracy.",
            style="HelpText.TLabel",
        )
        h_model.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        lbl_ctx = ttk.Label(
            scrollable_frame, text="Execution Device:", style="FieldLabel.TLabel"
        )
        lbl_ctx.grid(row=row, column=0, sticky="w", pady=2)
        self.combo_ctx = ttk.Combobox(
            scrollable_frame,
            values=["-1 (CPU)", "0 (GPU / CUDA Device 0)", "1 (GPU / CUDA Device 1)"],
            state="readonly",
            width=26,
        )
        self.combo_ctx.grid(row=row, column=1, columnspan=2, sticky="w", padx=4, pady=2)
        row += 1
        h_ctx = ttk.Label(
            scrollable_frame,
            text="-1 uses CPU (recommended if CUDA/GPU is not configured).",
            style="HelpText.TLabel",
        )
        h_ctx.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        # Separator
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        # Section 3: Clustering & Graph Rules
        sec3 = ttk.Label(
            scrollable_frame,
            text="3. Clustering & Graph Options",
            style="Section.TLabel",
        )
        sec3.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # eps
        lbl_eps = ttk.Label(
            scrollable_frame, text="DBSCAN eps:", style="FieldLabel.TLabel"
        )
        lbl_eps.grid(row=row, column=0, sticky="w", pady=2)
        self.spin_eps = ttk.Spinbox(
            scrollable_frame,
            from_=0.05,
            to=1.50,
            increment=0.01,
            format="%.2f",
            width=12,
        )
        self.spin_eps.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        row += 1
        h_eps = ttk.Label(
            scrollable_frame,
            text="Cosine distance threshold (default 0.45; lower=stricter, higher=looser).",
            style="HelpText.TLabel",
        )
        h_eps.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # min_samples
        lbl_ms = ttk.Label(
            scrollable_frame, text="Min Samples:", style="FieldLabel.TLabel"
        )
        lbl_ms.grid(row=row, column=0, sticky="w", pady=2)
        self.spin_min_samples = ttk.Spinbox(
            scrollable_frame, from_=1, to=100, increment=1, width=12
        )
        self.spin_min_samples.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        row += 1
        h_ms = ttk.Label(
            scrollable_frame,
            text="Minimum face occurrences to establish a distinct person (default: 2).",
            style="HelpText.TLabel",
        )
        h_ms.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # profile_distance
        lbl_pd = ttk.Label(
            scrollable_frame, text="Profile Distance:", style="FieldLabel.TLabel"
        )
        lbl_pd.grid(row=row, column=0, sticky="w", pady=2)
        self.spin_profile_dist = ttk.Spinbox(
            scrollable_frame,
            from_=0.05,
            to=1.50,
            increment=0.01,
            format="%.2f",
            width=12,
        )
        self.spin_profile_dist.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        row += 1
        h_pd = ttk.Label(
            scrollable_frame,
            text="Maximum cosine distance to match profile photo name (default: 0.55).",
            style="HelpText.TLabel",
        )
        h_pd.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # Remove isolated nodes checkbox
        self.var_remove_isolated = tk.BooleanVar(value=True)
        chk_isolated = ttk.Checkbutton(
            scrollable_frame,
            text="Remove Isolated Nodes (nodes with 0 connections)",
            variable=self.var_remove_isolated,
        )
        chk_isolated.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        h_iso = ttk.Label(
            scrollable_frame,
            text="Omits people who have no connections to anyone else in the network.",
            style="HelpText.TLabel",
        )
        h_iso.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        # Separator
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )
        row += 1

        # Section 4: Storage & Output Options
        sec4 = ttk.Label(
            scrollable_frame, text="4. Storage & Export Options", style="Section.TLabel"
        )
        sec4.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        self.var_disk_storage = tk.BooleanVar(value=False)
        chk_disk = ttk.Checkbutton(
            scrollable_frame,
            text="Enable Disk Storage (for large datasets)",
            variable=self.var_disk_storage,
            command=self._on_disk_storage_toggle,
        )
        chk_disk.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        h_disk = ttk.Label(
            scrollable_frame,
            text="Saves embeddings & crops to disk to minimize RAM usage.",
            style="HelpText.TLabel",
        )
        h_disk.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # Cache dir
        lbl_cache = ttk.Label(
            scrollable_frame, text="Cache Directory:", style="FieldLabel.TLabel"
        )
        lbl_cache.grid(row=row, column=0, sticky="w", pady=2)
        self.entry_cache_dir = ttk.Entry(scrollable_frame, width=28)
        self.entry_cache_dir.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        self.btn_browse_cache = ttk.Button(
            scrollable_frame, text="Browse...", width=9, command=self._browse_cache
        )
        self.btn_browse_cache.grid(row=row, column=2, sticky="e", pady=2)
        row += 1

        self.var_grouped_faces = tk.BooleanVar(value=False)
        chk_grouped = ttk.Checkbutton(
            scrollable_frame,
            text="Export Grouped Faces into Folders",
            variable=self.var_grouped_faces,
        )
        chk_grouped.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 2))
        row += 1
        # Separator & Start button inside the parameters panel for direct access
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(8, 12)
        )
        row += 1

        self.btn_start_param = ttk.Button(
            scrollable_frame,
            text="Start Scanning & Create Graph",
            style="Primary.TButton",
            command=self._start_scan,
        )
        self.btn_start_param.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8)
        )
        row += 1

        scrollable_frame.columnconfigure(1, weight=1)

    def _build_preview_and_logs_pane(self, parent):
        right_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        right_paned.pack(fill="both", expand=True)

        # 1. Live Image Scan Preview Card
        preview_card = ttk.Frame(right_paned, style="Card.TFrame", padding=12)
        right_paned.add(preview_card, weight=3)

        preview_header = ttk.Frame(preview_card, style="Card.TFrame")
        preview_header.pack(fill="x", pady=(0, 6))

        p_title = ttk.Label(
            preview_header, text="Live Scanner Preview", style="Section.TLabel"
        )
        p_title.pack(side="left")

        self.lbl_preview_event = ttk.Label(
            preview_header,
            text="",
            font=("Segoe UI", 9, "italic"),
            foreground="#4a5568",
        )
        self.lbl_preview_event.pack(side="right")

        # Current scanning image name label
        self.lbl_current_image = ttk.Label(
            preview_card,
            text="Waiting to start scan...",
            style="Scanning.TLabel",
            wraplength=480,
        )
        self.lbl_current_image.pack(anchor="w", pady=(0, 4))

        # Canvas for the preview image
        self.preview_canvas_frame = tk.Frame(
            preview_card, bg="#1a202c", relief="ridge", bd=1
        )
        self.preview_canvas_frame.pack(fill="both", expand=True, pady=4)

        self.preview_image_label = tk.Label(
            self.preview_canvas_frame,
            text="No scan in progress\n\nStart scanning to preview images and detected faces in real-time.",
            bg="#1a202c",
            fg="#a0aec0",
            font=("Segoe UI", 10),
            justify="center",
        )
        self.preview_image_label.pack(fill="both", expand=True)

        # Bottom info row under preview
        preview_info_frame = ttk.Frame(preview_card, style="Card.TFrame")
        preview_info_frame.pack(fill="x", pady=(6, 0))

        self.lbl_image_meta = ttk.Label(
            preview_info_frame, text="Resolution: -", style="FieldLabel.TLabel"
        )
        self.lbl_image_meta.pack(side="left")

        self.lbl_face_badge = ttk.Label(
            preview_info_frame,
            text="Faces detected: 0",
            font=("Segoe UI", 9, "bold"),
            foreground="#2b6cb0",
        )
        self.lbl_face_badge.pack(side="right")

        # 2. Activity Log / Console Card
        logs_card = ttk.Frame(right_paned, style="Card.TFrame", padding=12)
        right_paned.add(logs_card, weight=2)

        logs_header = ttk.Frame(logs_card, style="Card.TFrame")
        logs_header.pack(fill="x", pady=(0, 4))

        l_title = ttk.Label(logs_header, text="Activity Log", style="Section.TLabel")
        l_title.pack(side="left")

        btn_clear_log = ttk.Button(
            logs_header, text="Clear", width=7, command=self._clear_log
        )
        btn_clear_log.pack(side="right")

        self.txt_logs = scrolledtext.ScrolledText(
            logs_card,
            wrap=tk.WORD,
            height=8,
            font=("Consolas", 9),
            bg="#1a202c",
            fg="#edf2f7",
            insertbackground="#ffffff",
            relief="flat",
        )
        self.txt_logs.pack(fill="both", expand=True)

    # ── Manage Graph Tab ─────────────────────────────────────────────────────

    def _build_manage_tab(self, parent):
        """Build the Manage Graph tab: load a graph, add images, or prune."""
        outer = ttk.Frame(parent, padding=12)
        outer.pack(fill="both", expand=True)

        # ── Section: Select Existing Graph ──────────────────────────────────
        sec_graph = ttk.Label(
            outer, text="Select Existing Graph", style="Section.TLabel"
        )
        sec_graph.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        ttk.Label(
            outer, text="Graph File (.html/.json/.pkl):", style="FieldLabel.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=2)
        self.entry_mg_graph = ttk.Entry(outer, width=52)
        self.entry_mg_graph.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(
            outer, text="Browse...", width=9, command=self._mg_browse_graph
        ).grid(row=1, column=2, sticky="e", pady=2)

        ttk.Label(
            outer,
            text="Select the output HTML, JSON, or .graph_data.pkl from a previous graph build.",
            style="HelpText.TLabel",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 2))

        # Graph info badge
        self.lbl_mg_graph_info = ttk.Label(
            outer,
            text="No graph loaded.",
            font=("Segoe UI", 8, "italic"),
            foreground="#718096",
        )
        self.lbl_mg_graph_info.grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )

        ttk.Separator(outer, orient="horizontal").grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=8
        )

        # ── Two-column sub-panels ────────────────────────────────────────────
        panels = ttk.Frame(outer)
        panels.grid(row=5, column=0, columnspan=3, sticky="nsew")
        outer.rowconfigure(5, weight=1)
        outer.columnconfigure(1, weight=1)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)

        # Left panel: Add New Images
        add_card = ttk.LabelFrame(
            panels, text=" Add New Images (Continuous Creation) ", padding=10
        )
        add_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        add_card.columnconfigure(1, weight=1)
        self._build_mg_add_panel(add_card)

        # Right panel: Prune Graph
        prune_card = ttk.LabelFrame(
            panels, text=" Prune Graph (Top Popularity Nodes) ", padding=10
        )
        prune_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        prune_card.columnconfigure(1, weight=1)
        self._build_mg_prune_panel(prune_card)

        ttk.Separator(outer, orient="horizontal").grid(
            row=6, column=0, columnspan=3, sticky="ew", pady=8
        )

        # ── Manage tab log ──────────────────────────────────────────────────
        ttk.Label(outer, text="Manage Log", style="Section.TLabel").grid(
            row=7, column=0, columnspan=2, sticky="w"
        )
        btn_clear_mg = ttk.Button(
            outer, text="Clear", width=7, command=self._mg_clear_log
        )
        btn_clear_mg.grid(row=7, column=2, sticky="e")

        self.txt_mg_logs = scrolledtext.ScrolledText(
            outer,
            wrap=tk.WORD,
            height=7,
            font=("Consolas", 9),
            bg="#1a202c",
            fg="#edf2f7",
            insertbackground="#ffffff",
            relief="flat",
        )
        self.txt_mg_logs.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        outer.rowconfigure(8, weight=1)

    def _build_mg_add_panel(self, parent):
        """UI widgets for the 'Add New Images' section of Manage Graph."""
        row = 0
        ttk.Label(
            parent,
            text="Folder with new images to add to the selected graph.\n"
            "Only unprocessed images (not in the previous graph) are scanned.",
            style="HelpText.TLabel",
            wraplength=300,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        ttk.Label(parent, text="New Images Folder *:", style="FieldLabel.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.entry_mg_new_images = ttk.Entry(parent, width=22)
        self.entry_mg_new_images.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(
            parent, text="Browse...", width=8, command=self._mg_browse_new_images
        ).grid(row=row, column=2, sticky="e", pady=2)
        row += 1

        ttk.Label(parent, text="New Profiles Folder:", style="FieldLabel.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.entry_mg_new_profiles = ttk.Entry(parent, width=22)
        self.entry_mg_new_profiles.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(
            parent, text="Browse...", width=8, command=self._mg_browse_new_profiles
        ).grid(row=row, column=2, sticky="e", pady=2)
        row += 1

        ttk.Label(parent, text="Output Folder *:", style="FieldLabel.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.entry_mg_add_output = ttk.Entry(parent, width=22)
        self.entry_mg_add_output.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(
            parent, text="Browse...", width=8, command=self._mg_browse_add_output
        ).grid(row=row, column=2, sticky="e", pady=2)
        row += 1

        ttk.Label(
            parent,
            text="Output filename stem (e.g. 'graph_v2' → graph_v2.html/.json/.pkl)",
            style="HelpText.TLabel",
            wraplength=300,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))
        row += 1

        ttk.Label(parent, text="Output Name:", style="FieldLabel.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.entry_mg_add_name = ttk.Entry(parent, width=22)
        self.entry_mg_add_name.insert(0, "people_graph_updated")
        self.entry_mg_add_name.grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=4, pady=2
        )
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1

        self.btn_mg_add = ttk.Button(
            parent,
            text="Add Images & Update Graph",
            style="Primary.TButton",
            command=self._mg_run_add,
        )
        self.btn_mg_add.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self.btn_mg_cancel_add = ttk.Button(
            parent,
            text="Cancel",
            style="Danger.TButton",
            state="disabled",
            command=self._mg_cancel_add,
        )
        self.btn_mg_cancel_add.grid(row=row, column=2, sticky="e", padx=(4, 0), pady=2)
        row += 1

        self.btn_mg_open_add_result = ttk.Button(
            parent,
            text="Open Updated Graph",
            style="Success.TButton",
            state="disabled",
            command=self._mg_open_add_result,
        )
        self.btn_mg_open_add_result.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )

    def _build_mg_prune_panel(self, parent):
        """UI widgets for the 'Prune Graph' section of Manage Graph."""
        row = 0
        ttk.Label(
            parent,
            text="Keep only the top X most popular people in the graph.\n"
            "Nodes are ranked by popularity score (events attended + connections).",
            style="HelpText.TLabel",
            wraplength=300,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        ttk.Label(
            parent, text="Top X Nodes to Keep *:", style="FieldLabel.TLabel"
        ).grid(row=row, column=0, sticky="w", pady=2)
        self.spin_mg_top_x = ttk.Spinbox(
            parent, from_=1, to=10000, increment=1, width=10
        )
        self.spin_mg_top_x.set("20")
        self.spin_mg_top_x.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        row += 1

        self.var_mg_remove_isolated = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            parent,
            text="Remove nodes isolated after pruning",
            variable=self.var_mg_remove_isolated,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        self.var_mg_quality_popularity = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            parent,
            text="Add image quality",
            variable=self.var_mg_quality_popularity,
        ).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, text="Multiplier:", style="FieldLabel.TLabel").grid(
            row=row, column=1, sticky="e", padx=(0, 4), pady=2
        )
        self.spin_mg_quality_multiplier = ttk.Spinbox(
            parent, from_=0, to=100, increment=0.1, format="%.1f", width=7
        )
        self.spin_mg_quality_multiplier.set("1.0")
        self.spin_mg_quality_multiplier.grid(
            row=row, column=2, sticky="w", padx=4, pady=2
        )
        row += 1

        ttk.Label(parent, text="Output Folder *:", style="FieldLabel.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.entry_mg_prune_output = ttk.Entry(parent, width=22)
        self.entry_mg_prune_output.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(
            parent, text="Browse...", width=8, command=self._mg_browse_prune_output
        ).grid(row=row, column=2, sticky="e", pady=2)
        row += 1

        ttk.Label(parent, text="Output Name:", style="FieldLabel.TLabel").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.entry_mg_prune_name = ttk.Entry(parent, width=22)
        self.entry_mg_prune_name.insert(0, "people_graph_pruned")
        self.entry_mg_prune_name.grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=4, pady=2
        )
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1

        self.btn_mg_prune = ttk.Button(
            parent,
            text="Prune Graph",
            style="Primary.TButton",
            command=self._mg_run_prune,
        )
        self.btn_mg_prune.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        row += 1

        self.btn_mg_open_prune_result = ttk.Button(
            parent,
            text="Open Pruned Graph",
            style="Success.TButton",
            state="disabled",
            command=self._mg_open_prune_result,
        )
        self.btn_mg_open_prune_result.grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )

    # ── Manage Graph: state ───────────────────────────────────────────────────

    def _mg_clear_log(self):
        self.txt_mg_logs.delete("1.0", tk.END)

    def _mg_log(self, msg: str):
        self.txt_mg_logs.insert(tk.END, msg + "\n")
        self.txt_mg_logs.see(tk.END)

    def _mg_browse_graph(self):
        path = filedialog.askopenfilename(
            title="Select Existing Graph File",
            filetypes=[
                ("Graph bundle", "*.pkl"),
                ("JSON graph", "*.json"),
                ("HTML graph", "*.html"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.entry_mg_graph.delete(0, tk.END)
            self.entry_mg_graph.insert(0, path)
            self._mg_refresh_graph_info(path)

    def _mg_refresh_graph_info(self, path: str):
        try:
            from utilities.graph_storage import get_graph_summary

            info = get_graph_summary(path)
            if info.get("valid"):
                bundle_note = (
                    "[Bundle Available]"
                    if info.get("has_bundle")
                    else "[No Bundle (Pruning only)]"
                )
                txt = (
                    f"Nodes: {info['nodes']}  |  Edges: {info['edges']}  |  "
                    f"Images: {info['images']}  |  Faces: {info['faces']}  |  {bundle_note}"
                )
            else:
                txt = f"[Warning] Could not read: {info.get('error', 'unknown error')}"
        except Exception as exc:
            txt = f"[Error] Error inspecting graph: {exc}"
        self.lbl_mg_graph_info.configure(text=txt)

    def _mg_browse_new_images(self):
        path = filedialog.askdirectory(title="Select Folder with New Images")
        if path:
            self.entry_mg_new_images.delete(0, tk.END)
            self.entry_mg_new_images.insert(0, path)

    def _mg_browse_new_profiles(self):
        path = filedialog.askdirectory(
            title="Select Folder with New Profile Photos (Optional)"
        )
        if path:
            self.entry_mg_new_profiles.delete(0, tk.END)
            self.entry_mg_new_profiles.insert(0, path)

    def _mg_browse_add_output(self):
        path = filedialog.askdirectory(title="Select Output Folder for Updated Graph")
        if path:
            self.entry_mg_add_output.delete(0, tk.END)
            self.entry_mg_add_output.insert(0, path)

    def _mg_browse_prune_output(self):
        path = filedialog.askdirectory(title="Select Output Folder for Pruned Graph")
        if path:
            self.entry_mg_prune_output.delete(0, tk.END)
            self.entry_mg_prune_output.insert(0, path)

    # ── Add Images action ─────────────────────────────────────────────────────

    def _mg_run_add(self):
        from pathlib import Path

        graph_src = self.entry_mg_graph.get().strip()
        new_imgs = self.entry_mg_new_images.get().strip()
        new_profs = self.entry_mg_new_profiles.get().strip() or None
        out_folder = self.entry_mg_add_output.get().strip()
        out_name = self.entry_mg_add_name.get().strip() or "people_graph_updated"

        if not graph_src:
            messagebox.showerror(
                "Validation Error", "Please select an existing graph file.", parent=self
            )
            return
        if not new_imgs or not Path(new_imgs).is_dir():
            messagebox.showerror(
                "Validation Error",
                "Please select a valid folder with new images.",
                parent=self,
            )
            return
        if not out_folder:
            messagebox.showerror(
                "Validation Error", "Please select an output folder.", parent=self
            )
            return

        out_dir = Path(out_folder)
        out_html = out_dir / f"{out_name}.html"
        out_json = out_dir / f"{out_name}.json"
        out_data = out_dir / f"{out_name}.graph_data.pkl"

        self.btn_mg_add.configure(state="disabled")
        self.btn_mg_cancel_add.configure(state="normal")
        self.btn_mg_open_add_result.configure(state="disabled")
        self.mg_add_result_html: str | None = None
        self.manage_cancel_event.clear()

        self._mg_log(
            f"Starting update: adding images from '{new_imgs}' to graph '{Path(graph_src).name}'..."
        )

        def run_add():
            try:
                from utilities.continuous_graph import update_graph_with_images

                html_out, json_out, data_out = update_graph_with_images(
                    graph_source=graph_src,
                    new_images_dir=new_imgs,
                    new_profiles_dir=new_profs,
                    output_html=out_html,
                    output_json=out_json,
                    output_data=out_data,
                    on_log=lambda m: self.ui_queue.put(("mg_log", m)),
                    on_status=lambda s: self.ui_queue.put(("mg_status", s)),
                    on_progress=lambda c, t, m: self.ui_queue.put(
                        ("progress", (c, t, m))
                    ),
                    is_cancelled=lambda: self.manage_cancel_event.is_set(),
                )
                self.ui_queue.put(("mg_add_done", str(html_out)))
            except InterruptedError as e:
                self.ui_queue.put(("mg_add_cancelled", str(e)))
            except Exception as e:
                self.ui_queue.put(("mg_add_error", str(e)))

        self.manage_worker_thread = threading.Thread(target=run_add, daemon=True)
        self.manage_worker_thread.start()

    def _mg_cancel_add(self):
        self.manage_cancel_event.set()
        self.btn_mg_cancel_add.configure(state="disabled")
        self._mg_log("Cancellation requested...")

    def _mg_open_add_result(self):
        target = getattr(self, "mg_add_result_html", None)
        if target and os.path.exists(target):
            webbrowser.open(Path(target).resolve().as_uri())
        else:
            messagebox.showwarning(
                "File Not Found", "Updated HTML graph file not found.", parent=self
            )

    # ── Prune action ──────────────────────────────────────────────────────────

    def _mg_run_prune(self):
        from pathlib import Path

        graph_src = self.entry_mg_graph.get().strip()
        out_folder = self.entry_mg_prune_output.get().strip()
        out_name = self.entry_mg_prune_name.get().strip() or "people_graph_pruned"
        remove_iso = self.var_mg_remove_isolated.get()
        try:
            quality_multiplier = (
                float(self.spin_mg_quality_multiplier.get())
                if self.var_mg_quality_popularity.get()
                else 0.0
            )
            if quality_multiplier < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Validation Error", "Image quality multiplier must be non-negative.", parent=self
            )
            return

        if not graph_src:
            messagebox.showerror(
                "Validation Error", "Please select an existing graph file.", parent=self
            )
            return
        if not out_folder:
            messagebox.showerror(
                "Validation Error", "Please select an output folder.", parent=self
            )
            return

        try:
            top_x = int(self.spin_mg_top_x.get())
            if top_x < 1:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Validation Error", "Top X must be a positive integer.", parent=self
            )
            return

        out_dir = Path(out_folder)
        out_html = out_dir / f"{out_name}.html"
        out_json = out_dir / f"{out_name}.json"
        out_data_path = out_dir / f"{out_name}.graph_data.pkl"

        self.btn_mg_prune.configure(state="disabled")
        self.btn_mg_open_prune_result.configure(state="disabled")
        self.mg_prune_result_html = None
        self._mg_log(f"Pruning graph to top {top_x} most popular nodes...")

        def run_prune():
            try:
                from utilities.graph_pruning import prune_graph_to_top_popularity

                pruned, html_p, json_p, data_p = prune_graph_to_top_popularity(
                    source=graph_src,
                    top_x=top_x,
                    output_html=out_html,
                    output_json=out_json,
                    output_data=out_data_path,
                    remove_isolated=remove_iso,
                    image_quality_multiplier=quality_multiplier,
                    on_log=lambda m: self.ui_queue.put(("mg_log", m)),
                    on_status=lambda s: self.ui_queue.put(("mg_status", s)),
                )
                self.ui_queue.put(("mg_prune_done", str(html_p) if html_p else ""))
            except Exception as e:
                self.ui_queue.put(("mg_prune_error", str(e)))

        threading.Thread(target=run_prune, daemon=True).start()

    def _mg_open_prune_result(self):
        target = getattr(self, "mg_prune_result_html", None)
        if target and os.path.exists(target):
            webbrowser.open(Path(target).resolve().as_uri())
        else:
            messagebox.showwarning(
                "File Not Found", "Pruned HTML graph file not found.", parent=self
            )

    # ── Shared bottom bar ─────────────────────────────────────────────────────

    def _build_bottom_bar(self):
        bottom_frame = ttk.Frame(self, padding=(14, 8, 14, 12))
        bottom_frame.pack(fill="x", side="bottom")

        # Progress bar and status label row
        status_row = ttk.Frame(bottom_frame)
        status_row.pack(fill="x", pady=(0, 6))

        self.lbl_status = ttk.Label(status_row, text="Ready", style="Status.TLabel")
        self.lbl_status.pack(side="left")

        self.lbl_progress_percent = ttk.Label(
            status_row, text="0%", font=("Segoe UI", 9), foreground="#4a5568"
        )
        self.lbl_progress_percent.pack(side="right")

        self.progress_bar = ttk.Progressbar(
            bottom_frame, orient="horizontal", mode="determinate"
        )
        self.progress_bar.pack(fill="x", pady=(0, 8))

        # Controls & action buttons row
        btn_row = ttk.Frame(bottom_frame)
        btn_row.pack(fill="x")

        self.btn_start = ttk.Button(
            btn_row,
            text="▶ Start Scanning & Create Graph",
            style="Primary.TButton",
            command=self._start_scan,
        )
        self.btn_start.pack(side="left", padx=(0, 8))

        self.btn_cancel = ttk.Button(
            btn_row,
            text="⏹ Cancel",
            style="Danger.TButton",
            state="disabled",
            command=self._cancel_scan,
        )
        self.btn_cancel.pack(side="left", padx=(0, 8))

        self.btn_open_folder = ttk.Button(
            btn_row,
            text="📁 Open Folder",
            style="Success.TButton",
            state="disabled",
            command=self._open_output_folder,
        )
        self.btn_open_folder.pack(side="right", padx=(4, 0))

        self.btn_open_json = ttk.Button(
            btn_row,
            text="📄 Open JSON",
            style="Success.TButton",
            state="disabled",
            command=self._open_json_file,
        )
        self.btn_open_json.pack(side="right", padx=(4, 0))

        self.btn_open_browser = ttk.Button(
            btn_row,
            text="🌐 Open HTML Graph",
            style="Success.TButton",
            state="disabled",
            command=self._open_in_browser,
        )
        self.btn_open_browser.pack(side="right", padx=(4, 0))

    def _set_default_values(self):
        default_photos = Path("Photos").resolve()
        if default_photos.is_dir():
            self.entry_images.insert(0, str(default_photos))
        else:
            self.entry_images.insert(0, str(Path(".").resolve()))

        default_profiles = default_photos / "profile_pics"
        if default_profiles.is_dir():
            self.entry_profiles.insert(0, str(default_profiles))

        self.entry_output.insert(0, str(Path("people_graph.html").resolve()))
        self.entry_json.insert(0, str(Path("people_graph.json").resolve()))
        self.combo_model.set("buffalo_l")
        self.combo_ctx.current(0)  # -1 (CPU)

        self.spin_eps.set("0.45")
        self.spin_min_samples.set("2")
        self.spin_profile_dist.set("0.55")

        self.entry_cache_dir.insert(0, "graph_cache")
        self._on_disk_storage_toggle()

    def _on_disk_storage_toggle(self):
        is_on = self.var_disk_storage.get()
        state = "normal" if is_on else "disabled"
        self.entry_cache_dir.configure(state=state)
        self.btn_browse_cache.configure(state=state)

    def _browse_images(self):
        path = filedialog.askdirectory(title="Select Folder Containing Images")
        if path:
            self.entry_images.delete(0, tk.END)
            self.entry_images.insert(0, path)

    def _browse_profiles(self):
        path = filedialog.askdirectory(title="Select Optional Folder of Profile Photos")
        if path:
            self.entry_profiles.delete(0, tk.END)
            self.entry_profiles.insert(0, path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Choose Output Graph HTML Location",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
        )
        if path:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, path)
            json_path = str(Path(path).with_suffix(".json"))
            self.entry_json.delete(0, tk.END)
            self.entry_json.insert(0, json_path)

    def _browse_json(self):
        path = filedialog.asksaveasfilename(
            title="Choose Output Graph JSON Location",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.entry_json.delete(0, tk.END)
            self.entry_json.insert(0, path)

    def _browse_cache(self):
        path = filedialog.askdirectory(
            title="Select Cache Folder for Intermediate Files"
        )
        if path:
            self.entry_cache_dir.delete(0, tk.END)
            self.entry_cache_dir.insert(0, path)

    def _clear_log(self):
        self.txt_logs.delete("1.0", tk.END)

    def _log_message(self, message: str):
        self.txt_logs.insert(tk.END, message + "\n")
        self.txt_logs.see(tk.END)

    def _validate_inputs(self) -> dict | None:
        images_val = self.entry_images.get().strip()
        if not images_val:
            messagebox.showerror("Validation Error", "Please specify an Images Folder.")
            self.entry_images.focus()
            return None

        images_path = Path(images_val)
        if not images_path.is_dir():
            messagebox.showerror(
                "Validation Error", f"Images folder does not exist:\n{images_path}"
            )
            self.entry_images.focus()
            return None

        found_imgs = image_files(images_path)
        if not found_imgs:
            messagebox.showerror(
                "Validation Error",
                f"No supported images found in '{images_path}'.\nSupported types: {', '.join(IMAGE_EXTENSIONS)}",
            )
            return None

        profiles_val = self.entry_profiles.get().strip()
        if profiles_val:
            profiles_path = Path(profiles_val)
            if not profiles_path.is_dir():
                messagebox.showerror(
                    "Validation Error",
                    f"Profiles folder does not exist:\n{profiles_path}",
                )
                self.entry_profiles.focus()
                return None
        else:
            profiles_val = None

        output_val = self.entry_output.get().strip()
        if not output_val:
            messagebox.showerror(
                "Validation Error", "Please specify an Output HTML file path."
            )
            self.entry_output.focus()
            return None
        if not output_val.lower().endswith(".html"):
            output_val += ".html"
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, output_val)

        json_val = self.entry_json.get().strip()
        if json_val and not json_val.lower().endswith(".json"):
            json_val += ".json"
            self.entry_json.delete(0, tk.END)
            self.entry_json.insert(0, json_val)

        try:
            eps_val = float(self.spin_eps.get())
            if eps_val <= 0 or eps_val > 2.0:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Validation Error",
                "DBSCAN eps must be a positive number (typically between 0.05 and 1.5).",
            )
            self.spin_eps.focus()
            return None

        try:
            min_samples_val = int(self.spin_min_samples.get())
            if min_samples_val < 1:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Validation Error", "Min Samples must be an integer of 1 or greater."
            )
            self.spin_min_samples.focus()
            return None

        try:
            profile_dist_val = float(self.spin_profile_dist.get())
            if profile_dist_val < 0 or profile_dist_val > 2.0:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Validation Error",
                "Profile Distance must be a positive number between 0.0 and 2.0.",
            )
            self.spin_profile_dist.focus()
            return None

        model_val = self.combo_model.get().strip()
        if not model_val:
            model_val = "buffalo_l"

        ctx_raw = self.combo_ctx.get().strip()
        if "CPU" in ctx_raw or ctx_raw.startswith("-1"):
            ctx_id_val = -1
        elif "Device 0" in ctx_raw or ctx_raw.startswith("0"):
            ctx_id_val = 0
        elif "Device 1" in ctx_raw or ctx_raw.startswith("1"):
            ctx_id_val = 1
        else:
            try:
                ctx_id_val = int(ctx_raw.split()[0])
            except ValueError:
                ctx_id_val = -1

        cache_dir_val = (
            self.entry_cache_dir.get().strip()
            if self.var_disk_storage.get() or self.var_grouped_faces.get()
            else None
        )

        return {
            "images": images_val,
            "profiles": profiles_val,
            "output": output_val,
            "json_output": json_val or None,
            "save_json": bool(json_val),
            "model": model_val,
            "ctx_id": ctx_id_val,
            "eps": eps_val,
            "min_samples": min_samples_val,
            "profile_distance": profile_dist_val,
            "disk_storage": self.var_disk_storage.get(),
            "cache_dir": cache_dir_val,
            "grouped_faces": self.var_grouped_faces.get(),
            "remove_isolated": self.var_remove_isolated.get(),
        }

    def _start_scan(self):
        params = self._validate_inputs()
        if not params:
            return

        self.btn_start.configure(state="disabled")
        if hasattr(self, "btn_start_param"):
            self.btn_start_param.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_open_browser.configure(state="disabled")
        self.btn_open_json.configure(state="disabled")
        self.btn_open_folder.configure(state="disabled")

        self.progress_bar["value"] = 0
        self.lbl_progress_percent.configure(text="0%")
        self.lbl_status.configure(text="Starting scanner...")
        self.lbl_current_image.configure(text="Initializing scanning process...")
        self.lbl_preview_event.configure(text="")
        self.lbl_face_badge.configure(text="Faces detected: 0")

        self.cancel_event.clear()
        self.worker_thread = SocialGraphWorker(params, self.ui_queue, self.cancel_event)
        self.worker_thread.start()

    def _cancel_scan(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.lbl_status.configure(text="Stopping scanner...")
            self.cancel_event.set()
            self.btn_cancel.configure(state="disabled")

    def _process_queue(self):
        try:
            while True:
                msg_type, data = self.ui_queue.get_nowait()
                # ── Create Graph tab messages ──────────────────────────────
                if msg_type == "log":
                    self._log_message(data)
                elif msg_type == "status":
                    self.lbl_status.configure(text=data)
                elif msg_type == "progress":
                    cur, total, text = data
                    pct = int((cur / max(total, 1)) * 100)
                    self.progress_bar["maximum"] = total
                    self.progress_bar["value"] = cur
                    self.lbl_progress_percent.configure(text=f"{pct}% ({cur}/{total})")
                    if text:
                        self.lbl_current_image.configure(text=text)
                elif msg_type == "image_preview":
                    path, image, bboxes, face_count, event = data
                    self._render_preview(path, image, bboxes, face_count, event)
                elif msg_type == "done":
                    html_path, json_path, bundle_path = data
                    self.last_output_path = html_path
                    self.last_json_path = json_path
                    self.last_bundle_path = bundle_path
                    self._on_work_completed(success=True)
                elif msg_type == "cancelled":
                    self._on_work_completed(success=False, cancelled=True)
                elif msg_type == "error":
                    self._on_work_completed(success=False, error_msg=data)
                # ── Manage Graph tab messages ──────────────────────────────
                elif msg_type == "mg_log":
                    self._mg_log(data)
                elif msg_type == "mg_status":
                    self.lbl_status.configure(text=data)
                elif msg_type == "mg_add_done":
                    self.mg_add_result_html = data if data else None
                    self.btn_mg_add.configure(state="normal")
                    self.btn_mg_cancel_add.configure(state="disabled")
                    if data and os.path.exists(data):
                        self.btn_mg_open_add_result.configure(state="normal")
                    self._mg_log("[Success] Graph updated successfully!")
                    messagebox.showinfo(
                        "Success", f"Graph updated!\n\nHTML: {data}", parent=self
                    )
                elif msg_type == "mg_add_cancelled":
                    self.btn_mg_add.configure(state="normal")
                    self.btn_mg_cancel_add.configure(state="disabled")
                    self._mg_log("[Cancelled] Update cancelled.")
                    messagebox.showwarning(
                        "Cancelled", "Graph update was cancelled.", parent=self
                    )
                elif msg_type == "mg_add_error":
                    self.btn_mg_add.configure(state="normal")
                    self.btn_mg_cancel_add.configure(state="disabled")
                    self._mg_log(f"[Error] {data}")
                    messagebox.showerror(
                        "Error", f"Failed to update graph:\n\n{data}", parent=self
                    )
                elif msg_type == "mg_prune_done":
                    self.mg_prune_result_html = data if data else None
                    self.btn_mg_prune.configure(state="normal")
                    if data and os.path.exists(data):
                        self.btn_mg_open_prune_result.configure(state="normal")
                    self._mg_log("[Success] Graph pruned successfully!")
                    messagebox.showinfo(
                        "Success", f"Graph pruned!\n\nHTML: {data}", parent=self
                    )
                elif msg_type == "mg_prune_error":
                    self.btn_mg_prune.configure(state="normal")
                    self._mg_log(f"[Error] {data}")
                    messagebox.showerror(
                        "Error", f"Failed to prune graph:\n\n{data}", parent=self
                    )
        except queue.Empty:
            pass
        finally:
            self.after(50, self._process_queue)

    def _render_preview(
        self,
        path: Path,
        image: np.ndarray | None,
        bboxes: list[tuple[int, int, int, int]],
        face_count: int,
        event: str,
    ):
        self.lbl_preview_event.configure(text=f"Event: {event}")
        self.lbl_face_badge.configure(text=f"Faces detected: {face_count}")

        if image is None:
            self.lbl_image_meta.configure(text=f"File: {path.name} (Unreadable)")
            self.preview_image_label.configure(
                image="",
                text=f"Could not load image:\n{path.name}",
            )
            return

        h, w = image.shape[:2]
        self.lbl_image_meta.configure(text=f"{path.name}  |  {w} × {h} px")

        # Convert cv2 BGR to RGB
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw green bounding boxes for detected faces
        for x1, y1, x2, y2 in bboxes:
            cv2.rectangle(
                rgb_img,
                (x1, y1),
                (x2, y2),
                (40, 220, 100),
                max(2, int(max(w, h) / 350)),
            )

        # Determine canvas display dimensions
        canvas_w = max(self.preview_canvas_frame.winfo_width(), 360)
        canvas_h = max(self.preview_canvas_frame.winfo_height(), 240)

        # Scale down keeping aspect ratio
        pil_img = Image.fromarray(rgb_img)
        pil_img.thumbnail((canvas_w - 10, canvas_h - 10), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(pil_img)
        self.preview_image_ref = photo
        self.preview_image_label.configure(image=photo, text="")

    def _on_work_completed(
        self, success: bool, cancelled: bool = False, error_msg: str = ""
    ):
        self.btn_start.configure(state="normal")
        if hasattr(self, "btn_start_param"):
            self.btn_start_param.configure(state="normal")
        self.btn_cancel.configure(state="disabled")

        if success:
            self.btn_open_browser.configure(state="normal")
            if self.last_json_path and os.path.exists(self.last_json_path):
                self.btn_open_json.configure(state="normal")
            self.btn_open_folder.configure(state="normal")
            msg = f"Interactive Social Graph created successfully!\n\nHTML Graph: {self.last_output_path}"
            if self.last_json_path:
                msg += f"\nJSON Graph: {self.last_json_path}"
            messagebox.showinfo("Success", msg)
        elif cancelled:
            messagebox.showwarning("Cancelled", "Graph generation process was stopped.")
        else:
            messagebox.showerror(
                "Error", f"Failed to complete graph generation:\n\n{error_msg}"
            )

    def _open_in_browser(self):
        target = self.last_output_path or self.entry_output.get().strip()
        if target and os.path.exists(target):
            webbrowser.open(Path(target).resolve().as_uri())
        else:
            messagebox.showwarning(
                "File Not Found", "Output HTML file does not exist yet."
            )

    def _open_json_file(self):
        target = self.last_json_path or self.entry_json.get().strip()
        if target and os.path.exists(target):
            if sys.platform == "win32":
                os.startfile(target)
            else:
                webbrowser.open(Path(target).resolve().as_uri())
        else:
            messagebox.showwarning(
                "File Not Found", f"JSON file does not exist yet:\n{target}"
            )

    def _open_output_folder(self):
        out = self.last_output_path or self.entry_output.get().strip()
        folder = Path(out).parent.resolve()
        if folder.exists():
            if sys.platform == "win32":
                os.startfile(folder)
            else:
                webbrowser.open(folder.as_uri())
        else:
            messagebox.showwarning(
                "Folder Not Found", f"Folder does not exist:\n{folder}"
            )


def main():
    app = SocialGraphGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
