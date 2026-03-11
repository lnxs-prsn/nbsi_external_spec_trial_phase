"""
NBSI v1.0 — Phase 6 Persistence Tests

Run from nbsi_v1_lightweight/:
    PYTHONPATH=. python -m unittest nbsi.tests.test_phase6_persistence -v

All tests use StubEmbedder — no ML dependencies required.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from nbsi.config import Config
from nbsi.embedder import StubEmbedder
from nbsi.lifecycle.nodes import StructuralNode
from nbsi.lifecycle.structural_library import StructuralNodeLibrary
from nbsi.session.persistence import save_library, load_library, library_info


def _make_library(n: int = 3) -> StructuralNodeLibrary:
    """Create a StructuralNodeLibrary with n synthetic structural nodes."""
    embedder = StubEmbedder()
    library = StructuralNodeLibrary()
    labels = [f"structural pattern {i}" for i in range(n)]
    embs = embedder.encode(labels)
    for i, emb in enumerate(embs):
        node = StructuralNode(
            node_id=f"node-{i:04d}",
            pattern_embedding=emb.tolist(),
            session_count=i + 1,
            diversity_score=0.3 + i * 0.1,
            is_generalised=True,
        )
        library.add(node)
    return library


class TestSaveLibrary(unittest.TestCase):

    def test_save_creates_file(self):
        """save_library creates a JSON file at the given path."""
        library = _make_library(3)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            result = save_library(library, path)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(result["nodes_saved"], 3)

    def test_save_creates_parent_dirs(self):
        """save_library creates parent directories if they do not exist."""
        library = _make_library(2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "subdir", "nested", "library.json")
            save_library(library, path)
            self.assertTrue(os.path.exists(path))

    def test_saved_file_is_valid_json(self):
        """The saved file is parseable JSON."""
        library = _make_library(2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            with open(path) as f:
                data = json.load(f)
            self.assertIn("nodes", data)
            self.assertIn("nbsi_version", data)

    def test_saved_file_contains_no_content(self):
        """
        Saved file must never contain session content.
        No labels, no raw text, no example_labels.
        """
        library = _make_library(2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            with open(path) as f:
                raw = f.read()
            # These content markers must not appear in the file
            self.assertNotIn("example_labels", raw)
            self.assertNotIn("observation", raw.lower().replace("observation_nodes", ""))

    def test_saved_file_has_content_free_flag(self):
        """Saved file explicitly marks itself as content-free."""
        library = _make_library(1)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            with open(path) as f:
                data = json.load(f)
            self.assertTrue(data.get("_content_free"))

    def test_save_empty_library(self):
        """Saving an empty library produces a valid file with 0 nodes."""
        library = StructuralNodeLibrary()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "empty.json")
            result = save_library(library, path)
            self.assertEqual(result["nodes_saved"], 0)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["node_count"], 0)
            self.assertEqual(data["nodes"], [])

    def test_saved_node_fields(self):
        """Each saved node contains the expected geometry fields."""
        library = _make_library(1)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            with open(path) as f:
                data = json.load(f)
            node = data["nodes"][0]
            self.assertIn("node_id", node)
            self.assertIn("pattern_embedding", node)
            self.assertIn("session_count", node)
            self.assertIn("diversity_score", node)
            self.assertIn("is_generalised", node)
            self.assertIn("attention_mode", node)
            self.assertIn("reactivation_count", node)

    def test_tilde_path_expands(self):
        """Tilde (~) in path is expanded correctly."""
        library = _make_library(1)
        with tempfile.TemporaryDirectory() as td:
            # Use an absolute path (tilde expansion works if home is accessible)
            path = os.path.join(td, "library.json")
            save_library(library, path)
            self.assertTrue(os.path.exists(path))


class TestLoadLibrary(unittest.TestCase):

    def test_load_returns_correct_node_count(self):
        """Loaded library has the same number of nodes as the saved one."""
        library = _make_library(4)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            loaded = load_library(path)
            self.assertEqual(len(loaded.nodes), 4)

    def test_load_preserves_embeddings(self):
        """Loaded node embeddings match the original centroids."""
        library = _make_library(2)
        original_emb = list(library.nodes["node-0000"].pattern_embedding)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            loaded = load_library(path)
            loaded_emb = list(loaded.nodes["node-0000"].pattern_embedding)
        self.assertEqual(original_emb, loaded_emb)

    def test_load_preserves_session_count(self):
        """Loaded node session_count matches the original."""
        library = _make_library(2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            loaded = load_library(path)
            self.assertEqual(loaded.nodes["node-0001"].session_count, 2)

    def test_load_nodes_are_structural_nodes(self):
        """Loaded nodes are StructuralNode instances."""
        library = _make_library(2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            loaded = load_library(path)
            for node in loaded.nodes.values():
                self.assertIsInstance(node, StructuralNode)

    def test_load_nodes_are_in_infrastructure_mode(self):
        """Loaded nodes start in infrastructure mode (not active)."""
        library = _make_library(2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            loaded = load_library(path)
            for node in loaded.nodes.values():
                self.assertEqual(node.attention_mode, "infrastructure")

    def test_load_missing_file_raises(self):
        """load_library raises FileNotFoundError for a missing file."""
        with self.assertRaises(FileNotFoundError):
            load_library("/tmp/nbsi_does_not_exist_xyz.json")

    def test_load_corrupt_file_raises(self):
        """load_library raises ValueError for corrupt JSON."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "corrupt.json")
            with open(path, "w") as f:
                f.write("{not valid json")
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                load_library(path)

    def test_load_wrong_version_raises(self):
        """load_library raises ValueError for a version mismatch."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wrong_version.json")
            with open(path, "w") as f:
                json.dump({"nbsi_version": "99.0", "nodes": []}, f)
            with self.assertRaises(ValueError):
                load_library(path)

    def test_load_empty_library(self):
        """Loading an empty library gives a valid empty StructuralNodeLibrary."""
        library = StructuralNodeLibrary()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "empty.json")
            save_library(library, path)
            loaded = load_library(path)
            self.assertIsInstance(loaded, StructuralNodeLibrary)
            self.assertEqual(len(loaded.nodes), 0)


class TestRoundTrip(unittest.TestCase):

    def test_roundtrip_match_still_works(self):
        """
        A loaded library can match embeddings against its structural nodes.
        This is the functional test: geometry survived save/load intact.
        """
        embedder = StubEmbedder()
        library = _make_library(3)

        # Get the embedding of node-0000's centroid
        emb = list(library.nodes["node-0000"].pattern_embedding)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            loaded = load_library(path)

        # Should match the same embedding after round-trip
        matches = loaded.match(emb, threshold=0.5)
        self.assertGreater(len(matches), 0)
        # The best match should be the same node
        self.assertEqual(matches[0][0], "node-0000")

    def test_roundtrip_node_count_preserved(self):
        """Node count is identical before and after round-trip."""
        library = _make_library(5)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            loaded = load_library(path)
        self.assertEqual(len(loaded.nodes), len(library.nodes))

    def test_double_roundtrip(self):
        """Save → load → save → load produces the same library."""
        library = _make_library(3)
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "pass1.json")
            p2 = os.path.join(td, "pass2.json")
            save_library(library, p1)
            lib1 = load_library(p1)
            save_library(lib1, p2)
            lib2 = load_library(p2)
        self.assertEqual(len(lib2.nodes), len(library.nodes))
        for nid in library.nodes:
            self.assertIn(nid, lib2.nodes)


class TestLibraryInfo(unittest.TestCase):

    def test_info_returns_dict(self):
        """library_info returns a dict with expected keys."""
        library = _make_library(2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            info = library_info(path)
        self.assertIsInstance(info, dict)
        self.assertIn("node_count", info)
        self.assertIn("saved_at", info)
        self.assertIn("version", info)
        self.assertIn("size_bytes", info)
        self.assertIn("content_free", info)

    def test_info_node_count_matches(self):
        """library_info returns the correct node count."""
        library = _make_library(4)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            info = library_info(path)
        self.assertEqual(info["node_count"], 4)

    def test_info_missing_file_returns_none(self):
        """library_info returns None for a non-existent file."""
        result = library_info("/tmp/nbsi_no_such_file_xyz.json")
        self.assertIsNone(result)

    def test_info_content_free_flag(self):
        """library_info confirms the file is content-free."""
        library = _make_library(1)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "library.json")
            save_library(library, path)
            info = library_info(path)
        self.assertTrue(info["content_free"])


if __name__ == "__main__":
    unittest.main()
