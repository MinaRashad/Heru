"""Unit and integration tests for standalone social graph bundle and formula testing."""

import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest

import cv2
import networkx as nx
import numpy as np

from create_social_graph import construct_graph, export_graph_html, export_graph_json
from utilities.graph_storage import (
    build_image_records,
    build_person_crops,
    ensure_image_records,
    load_graph_data,
    save_graph_data,
    serialize_crop,
)
from utilities.continuous_graph import rebuild_graph_from_bundle


class TestStandaloneGraph(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_social_eye_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synthetic_image(self, width: int = 200, height: int = 200, color=(100, 150, 200)) -> np.ndarray:
        img = np.full((height, width, 3), color, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
        return img

    def test_serialize_image_and_crop(self):
        img = self._create_synthetic_image()
        crop_bytes = serialize_crop(img[20:80, 20:80])
        self.assertIsInstance(crop_bytes, bytes)
        self.assertTrue(len(crop_bytes) > 0)

    def test_build_person_crops_highest_quality(self):
        # Create 2 crops for the same person: one low quality (flat), one higher quality (structured)
        flat_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        sharp_crop = self._create_synthetic_image(100, 100)

        face_records = [
            {"image": "img1.jpg", "event": "EventA", "crop": flat_crop, "quality": 10.0},
            {"image": "img2.jpg", "event": "EventB", "crop": sharp_crop, "quality": 95.0},
        ]
        labels = np.array([0, 0])
        cluster_names = {0: "person_0"}

        person_crops, crops = build_person_crops(face_records, labels, cluster_names)
        self.assertIn("person_0", person_crops)
        self.assertIn("person_0", crops)
        self.assertEqual(person_crops["person_0"]["quality"], 95.0)
        self.assertEqual(person_crops["person_0"]["source_image"], "img2.jpg")
        self.assertTrue(crops["person_0"].startswith("data:image/jpeg;base64,"))

    def test_build_image_records_standalone(self):
        img1 = self._create_synthetic_image()
        img2 = self._create_synthetic_image()

        p1 = self.temp_dir / "photo1.jpg"
        p2 = self.temp_dir / "photo2.jpg"
        cv2.imwrite(str(p1), img1)
        cv2.imwrite(str(p2), img2)

        face_records = [
            {"image": p1, "event": "Event1", "crop": img1[10:60, 10:60], "quality": 50.0},
            {"image": p1, "event": "Event1", "crop": img1[70:120, 70:120], "quality": 60.0},
            {"image": p2, "event": "Event2", "crop": img2[10:60, 10:60], "quality": 70.0},
        ]
        labels = np.array([0, 1, 1])
        cluster_names = {0: "Alice", 1: "Bob"}
        counts = {p1: 2, p2: 1}

        image_records = build_image_records(
            event_images=[("Event1", p1), ("Event2", p2)],
            face_records=face_records,
            labels=labels,
            cluster_names=cluster_names,
            image_face_counts=counts,
        )

        self.assertEqual(len(image_records), 2)
        # Verify first row
        row1 = image_records[0]
        self.assertEqual(row1["image_id"], "photo1.jpg")
        self.assertEqual(row1["people"], ["Alice", "Bob"])
        self.assertEqual(row1["face_count"], 2)
        self.assertEqual(len(row1["image_hash"]), 64)
        self.assertNotIn("image_data", row1)
        self.assertEqual(len(row1["faces"]), 2)

        # Verify second row
        row2 = image_records[1]
        self.assertEqual(row2["image_id"], "photo2.jpg")
        self.assertEqual(row2["people"], ["Bob"])
        self.assertEqual(row2["face_count"], 1)

    def test_construct_graph_from_image_records_and_evidence(self):
        image_records = [
            {
                "image_id": "group_small.jpg",
                "event": "Graduation",
                "people": ["Alice", "Bob"],
                "face_count": 2,
            },
            {
                "image_id": "group_large.jpg",
                "event": "Graduation",
                "people": ["Alice", "Bob", "Charlie", "David"],
                "face_count": 4,
            },
            {
                "image_id": "solo.jpg",
                "event": "Party",
                "people": ["Eve"],
                "face_count": 1,
            },
        ]

        # Formula: small group (<= 3 faces) = 10 pts, large group = 3 pts, event co-occurrence = 1 pt
        graph = construct_graph(
            image_records=image_records,
            small_group_weight=10.0,
            small_group_max_faces=3,
            large_group_weight=3.0,
            event_weight=1.0,
            event_popularity_multiplier=50.0,
            degree_popularity_multiplier=1.0,
            remove_isolated=True,
        )

        # Alice and Bob appear in group_small (10) and group_large (3)
        # There are 2 events (Graduation, Party). Graduation has Alice, Bob, Charlie, David.
        # So Alice and Bob also get 1 point from event co-occurrence -> total 14.0
        self.assertTrue(graph.has_edge("Alice", "Bob"))
        edge_data = graph["Alice"]["Bob"]
        self.assertEqual(edge_data["weight"], 14.0)

        # Check evidence photo tracking
        self.assertIn("evidence", edge_data)
        self.assertIn("group_small.jpg", edge_data["evidence"])
        self.assertIn("group_large.jpg", edge_data["evidence"])

        # Eve has no connections and remove_isolated=True, so Eve should be removed
        self.assertNotIn("Eve", graph.nodes)

    def test_save_and_rebuild_from_bundle(self):
        # Create a complete standalone bundle
        image_records = [
            {
                "image_id": "pic1.jpg",
                "image_path": str(self.temp_dir / "pic1.jpg"),
                "event": "Summit",
                "image_hash": "a" * 64,
                "people": ["PersonA", "PersonB"],
                "face_count": 2,
                "faces": [],
            },
            {
                "image_id": "pic2.jpg",
                "image_path": str(self.temp_dir / "pic2.jpg"),
                "event": "Dinner",
                "image_hash": "b" * 64,
                "people": ["PersonB", "PersonC"],
                "face_count": 2,
                "faces": [],
            },
        ]
        cluster_names = {0: "PersonA", 1: "PersonB", 2: "PersonC"}
        crops = {
            "PersonA": "data:image/jpeg;base64,mockA",
            "PersonB": "data:image/jpeg;base64,mockB",
            "PersonC": "data:image/jpeg;base64,mockC",
        }
        person_crops = {
            "PersonA": {"person_id": "PersonA", "crop": b"cropA", "data_uri": crops["PersonA"], "quality": 90.0},
            "PersonB": {"person_id": "PersonB", "crop": b"cropB", "data_uri": crops["PersonB"], "quality": 88.0},
            "PersonC": {"person_id": "PersonC", "crop": b"cropC", "data_uri": crops["PersonC"], "quality": 85.0},
        }

        g = construct_graph(image_records=image_records, cluster_names=cluster_names)
        bundle_path = self.temp_dir / "test.graph_data.pkl"

        save_graph_data(
            bundle_path,
            graph=g,
            crops=crops,
            embeddings=np.zeros((3, 512), dtype=np.float32),
            face_records=[
                {"event": "Summit", "image": "pic1.jpg", "crop": b"cropA"},
                {"event": "Summit", "image": "pic1.jpg", "crop": b"cropB"},
                {"event": "Dinner", "image": "pic2.jpg", "crop": b"cropC"},
            ],
            image_face_counts={"pic1.jpg": 2, "pic2.jpg": 1},
            labels=np.array([0, 1, 2]),
            cluster_names=cluster_names,
            image_records=image_records,
            person_crops=person_crops,
        )

        self.assertTrue(bundle_path.is_file())

        # Load bundle and verify Part 1 and Part 2 are intact
        loaded = load_graph_data(bundle_path)
        self.assertIn("image_records", loaded)
        self.assertIn("person_crops", loaded)
        self.assertEqual(len(loaded["image_records"]), 2)
        self.assertEqual(loaded["image_records"][0]["people"], ["PersonA", "PersonB"])
        self.assertEqual(loaded["image_records"][0]["image_hash"], "a" * 64)
        self.assertNotIn("image_data", loaded["image_records"][0])

        # Now test instant rebuild with a DIFFERENT formula!
        out_html = self.temp_dir / "rebuilt.html"
        out_json = self.temp_dir / "rebuilt.json"
        out_bundle = self.temp_dir / "rebuilt.graph_data.pkl"

        rebuilt_g, h_path, j_path, b_path = rebuild_graph_from_bundle(
            bundle_path,
            output_html=out_html,
            output_json=out_json,
            output_bundle=out_bundle,
            small_group_weight=20.0,  # Changed from default 5.0 to 20.0
            large_group_weight=1.0,
            event_weight=0.0,
            event_popularity_multiplier=100.0,
            degree_popularity_multiplier=2.0,
        )

        self.assertTrue(out_html.is_file())
        self.assertTrue(out_json.is_file())
        self.assertTrue(out_bundle.is_file())

        # Verify new weights applied
        self.assertEqual(rebuilt_g["PersonA"]["PersonB"]["weight"], 20.0)
        self.assertEqual(rebuilt_g["PersonB"]["PersonC"]["weight"], 20.0)

        # Verify evidence in exported JSON
        with open(out_json, "r", encoding="utf-8") as f:
            j_data = json.load(f)
        edge_ab = next(e for e in j_data["edges"] if e["source"] == "PersonA" and e["target"] == "PersonB")
        self.assertEqual(edge_ab["weight"], 20.0)
        self.assertEqual(edge_ab["evidence"], ["pic1.jpg"])


if __name__ == "__main__":
    unittest.main()
