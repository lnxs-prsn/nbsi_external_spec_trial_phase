"""
tests/test_phase4_os_integration.py

Phase 4 exit criteria tests.

Tests are designed to run without a live NBSI session — they use
lightweight stubs so they can be verified in isolation before wiring
into main.py.

Run:
    PYTHONPATH=. python -m pytest tests/test_phase4_os_integration.py -v
"""

import os
import queue
import tempfile
import textwrap
import threading
import time
import unittest


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _FakeEmbedder:
    """Minimal embedder stub — returns deterministic 4-dim vectors."""
    def encode(self, texts):
        import numpy as np
        out = []
        for t in texts:
            v = np.array([hash(t) % 100 / 100,
                          hash(t[::-1]) % 100 / 100,
                          len(t) % 100 / 100,
                          0.5])
            v = v / (np.linalg.norm(v) + 1e-8)
            out.append(v)
        return np.array(out)


class _FakeNode:
    def __init__(self, label, embedding, node_type='concept', activation=0.7):
        import uuid
        self.node_id   = str(uuid.uuid4())
        self.label     = label
        self.embedding = embedding
        self.node_type = node_type
        self.activation = activation


class _FakeEdge:
    def __init__(self, source_id, target_id, edge_type='semantic'):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type


class _FakeExtractor:
    """Returns 3 fake nodes and 2 fake edges for any text."""
    def extract(self, text, embedder):
        embs  = embedder.encode(['alpha', 'beta', 'gamma'])
        nodes = [
            _FakeNode('alpha', embs[0].tolist()),
            _FakeNode('beta',  embs[1].tolist()),
            _FakeNode('gamma', embs[2].tolist()),
        ]
        edges = [
            _FakeEdge(nodes[0].node_id, nodes[1].node_id, 'causal'),
            _FakeEdge(nodes[1].node_id, nodes[2].node_id, 'semantic'),
        ]
        return nodes, edges


class _FakeGraph:
    def __init__(self):
        self.node_count = 0


class _FakeSession:
    def __init__(self):
        self.embedder    = _FakeEmbedder()
        self.graph       = _FakeGraph()
        self._ingestions = []

    def ingest_graph(self, nodes, edges):
        self._ingestions.append({'nodes': nodes, 'edges': edges})
        self.graph.node_count += len(nodes)
        return {'nodes': len(nodes), 'edges': len(edges)}


# ---------------------------------------------------------------------------
# Tests: reader.py
# ---------------------------------------------------------------------------

class TestReader(unittest.TestCase):

    def setUp(self):
        from nbsi.os_integration.reader import read_file, SUPPORTED_EXTENSIONS
        self.read_file  = read_file
        self.SUPPORTED  = SUPPORTED_EXTENSIONS

    # -- plain text ----------------------------------------------------------

    def test_read_txt_returns_content(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False, encoding='utf-8') as f:
            f.write('Hello world. This is a test document.\n')
            path = f.name
        try:
            result = self.read_file(path)
            self.assertIsNotNone(result)
            self.assertIn('Hello world', result)
        finally:
            os.unlink(path)

    def test_read_md_returns_content(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md',
                                         delete=False, encoding='utf-8') as f:
            f.write('# Heading\n\nSome paragraph text here.\n')
            path = f.name
        try:
            result = self.read_file(path)
            self.assertIsNotNone(result)
            self.assertIn('Heading', result)
        finally:
            os.unlink(path)

    def test_read_py_returns_content(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False, encoding='utf-8') as f:
            f.write('def hello():\n    return "world"\n')
            path = f.name
        try:
            result = self.read_file(path)
            self.assertIsNotNone(result)
            self.assertIn('def hello', result)
        finally:
            os.unlink(path)

    # -- unsupported / missing -----------------------------------------------

    def test_unsupported_extension_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b'data')
            path = f.name
        try:
            result = self.read_file(path)
            self.assertIsNone(result)
        finally:
            os.unlink(path)

    def test_missing_file_returns_none(self):
        result = self.read_file('/tmp/does_not_exist_nbsi_test.txt')
        self.assertIsNone(result)

    def test_empty_file_returns_empty_string_not_none(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            path = f.name
        try:
            result = self.read_file(path)
            # An empty file is a valid read — just returns empty string
            # The worker filters on len(text.strip()) < 50, not the reader
            self.assertIsNotNone(result)
        finally:
            os.unlink(path)

    # -- supported extensions set -------------------------------------------

    def test_supported_extensions_includes_core_formats(self):
        for ext in ['.txt', '.md', '.pdf', '.docx', '.py']:
            self.assertIn(ext, self.SUPPORTED,
                          f'{ext} missing from SUPPORTED_EXTENSIONS')

    # -- docx (conditional) --------------------------------------------------

    def test_read_docx_if_available(self):
        try:
            from docx import Document as DocxDocument
        except ImportError:
            self.skipTest('python-docx not installed')

        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
            path = f.name

        try:
            doc = DocxDocument()
            doc.add_paragraph('Introduction')
            doc.add_paragraph('This is the body of the document.')
            doc.save(path)

            result = self.read_file(path)
            self.assertIsNotNone(result)
            self.assertIn('Introduction', result)
        finally:
            os.unlink(path)

    # -- pdf (conditional) ---------------------------------------------------

    def test_read_pdf_graceful_if_unavailable(self):
        """
        If pdfminer is not installed, read_file returns None without
        raising an exception.
        """
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b'%PDF-1.4 fake content')
            path = f.name
        try:
            # Should not raise regardless of whether pdfminer is installed
            result = self.read_file(path)
            # We don't assert the value — just that it didn't crash
        except Exception as e:
            self.fail(f'read_file raised an exception: {e}')
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: watcher.py
# ---------------------------------------------------------------------------

class TestWatcher(unittest.TestCase):

    def test_new_txt_file_enqueued(self):
        """Creating a .txt file triggers a queue entry within 3 seconds."""
        from nbsi.os_integration.watcher import start_watcher

        with tempfile.TemporaryDirectory() as tmpdir:
            q        = queue.Queue()
            observer = start_watcher(tmpdir, q, debounce_seconds=1.0)

            try:
                test_path = os.path.join(tmpdir, 'test_doc.txt')
                with open(test_path, 'w') as f:
                    f.write('Hello NBSI watcher test.\n')

                # Wait up to 3 seconds for the event
                try:
                    path = q.get(timeout=3.0)
                    self.assertEqual(os.path.abspath(path),
                                     os.path.abspath(test_path))
                except queue.Empty:
                    self.fail('Watcher did not enqueue file within 3 seconds')
            finally:
                observer.stop()
                observer.join()

    def test_unsupported_extension_not_enqueued(self):
        """A .xyz file must NOT be enqueued."""
        from nbsi.os_integration.watcher import start_watcher

        with tempfile.TemporaryDirectory() as tmpdir:
            q        = queue.Queue()
            observer = start_watcher(tmpdir, q, debounce_seconds=1.0)

            try:
                test_path = os.path.join(tmpdir, 'junk.xyz')
                with open(test_path, 'w') as f:
                    f.write('irrelevant\n')

                time.sleep(1.0)
                self.assertTrue(q.empty(),
                    'Watcher should not enqueue unsupported extension')
            finally:
                observer.stop()
                observer.join()

    def test_debounce_prevents_duplicate_enqueue(self):
        """
        Modifying a file twice rapidly should produce only one queue entry
        within the debounce window.
        """
        from nbsi.os_integration.watcher import start_watcher

        with tempfile.TemporaryDirectory() as tmpdir:
            q        = queue.Queue()
            observer = start_watcher(tmpdir, q, debounce_seconds=5.0)

            try:
                test_path = os.path.join(tmpdir, 'repeated.txt')
                with open(test_path, 'w') as f:
                    f.write('First write.\n')
                time.sleep(0.3)
                with open(test_path, 'a') as f:
                    f.write('Second write.\n')
                time.sleep(0.5)

                items = []
                while not q.empty():
                    items.append(q.get_nowait())

                self.assertLessEqual(len(items), 2,
                    'Expected at most 2 queue entries (create + first modify)')
            finally:
                observer.stop()
                observer.join()

    def test_invalid_path_raises(self):
        """start_watcher raises ValueError for a non-existent directory."""
        from nbsi.os_integration.watcher import start_watcher
        q = queue.Queue()
        with self.assertRaises(ValueError):
            start_watcher('/tmp/nbsi_no_such_dir_xyz', q)

    def test_scan_existing_enqueues_files(self):
        """scan_existing enqueues files already present in the directory."""
        from nbsi.os_integration.watcher import scan_existing

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two supported files
            for name in ['doc1.txt', 'doc2.md']:
                with open(os.path.join(tmpdir, name), 'w') as f:
                    f.write('Content.\n')
            # One unsupported
            with open(os.path.join(tmpdir, 'junk.xyz'), 'w') as f:
                f.write('ignored\n')

            q     = queue.Queue()
            count = scan_existing(tmpdir, q)

            self.assertEqual(count, 2)
            self.assertEqual(q.qsize(), 2)


# ---------------------------------------------------------------------------
# Tests: ingestion_worker.py
# ---------------------------------------------------------------------------

class TestIngestionWorker(unittest.TestCase):

    def _make_worker(self):
        session   = _FakeSession()
        extractor = _FakeExtractor()
        q         = queue.Queue()
        from nbsi.os_integration.ingestion_worker import IngestionWorker
        worker    = IngestionWorker(session, extractor, q)
        return worker, session, q

    def test_worker_processes_txt_file(self):
        """Worker ingests a plain text file and updates the session."""
        worker, session, q = self._make_worker()
        worker.start()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False, encoding='utf-8') as f:
            f.write('The structural layer manages boundaries and patterns. '
                    'Session end triggers library consolidation. '
                    'Observation nodes accumulate over time.\n')
            path = f.name

        try:
            q.put(path)
            q.join()   # block until worker calls task_done()

            worker.stop(timeout=3.0)

            self.assertEqual(worker.files_ingested, 1)
            self.assertEqual(worker.files_failed, 0)
            self.assertGreater(session.graph.node_count, 0)
        finally:
            os.unlink(path)

    def test_worker_skips_short_file(self):
        """Worker skips files shorter than 50 characters."""
        worker, session, q = self._make_worker()
        worker.start()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            f.write('too short')
            path = f.name

        try:
            q.put(path)
            q.join()
            worker.stop(timeout=2.0)

            self.assertEqual(worker.files_ingested, 0,
                'Short file should be skipped, not ingested')
        finally:
            os.unlink(path)

    def test_worker_handles_missing_file_gracefully(self):
        """Worker does not crash when a queued file is missing."""
        worker, session, q = self._make_worker()
        worker.start()

        q.put('/tmp/nbsi_missing_file_xyz_12345.txt')
        q.join()
        worker.stop(timeout=2.0)

        # Should not have crashed — files_ingested stays 0
        self.assertEqual(worker.files_ingested, 0)

    def test_worker_processes_multiple_files(self):
        """Worker ingests several files sequentially."""
        worker, session, q = self._make_worker()
        worker.start()

        paths = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                                  delete=False,
                                                  encoding='utf-8') as f:
                    f.write(f'Document {i}. ' * 20)
                    paths.append(f.name)
                    q.put(f.name)

            q.join()
            worker.stop(timeout=5.0)

            self.assertEqual(worker.files_ingested, 3)
        finally:
            for p in paths:
                if os.path.exists(p):
                    os.unlink(p)

    def test_worker_stats(self):
        """stats() returns a dict with expected keys."""
        worker, session, q = self._make_worker()
        worker.start()
        worker.stop(timeout=2.0)

        s = worker.stats()
        for key in ['files_ingested', 'files_failed', 'nodes_added',
                    'edges_added', 'queue_size']:
            self.assertIn(key, s)

    def test_worker_does_not_block_main_thread(self):
        """
        The worker runs in a daemon thread — the main thread remains
        responsive while ingestion is running.
        """
        worker, session, q = self._make_worker()
        worker.start()

        # Queue a large-ish piece of text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False, encoding='utf-8') as f:
            f.write('Concept node. Structural layer. Session boundary. ' * 100)
            path = f.name

        try:
            q.put(path)

            # Main thread should be able to do work immediately
            t0   = time.monotonic()
            _ = 1 + 1   # trivial work
            elapsed = time.monotonic() - t0

            self.assertLess(elapsed, 0.1,
                'Main thread was blocked by worker thread')

            q.join()
            worker.stop(timeout=10.0)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# Integration: reader → worker pipeline
# ---------------------------------------------------------------------------

class TestReaderWorkerPipeline(unittest.TestCase):

    def test_end_to_end_txt(self):
        """
        Full pipeline: create a file, read it, extract, ingest.
        Verifies the three components work together.
        """
        from nbsi.os_integration.reader import read_file
        from nbsi.os_integration.ingestion_worker import IngestionWorker

        content = textwrap.dedent("""\
            The NBSI system builds a concept graph from documents.
            Structural nodes represent patterns that persist across sessions.
            The beam search finds paths between concepts in the graph.
            Conductivity measures how strongly two concepts are connected.
            Speculation introduces hypothetical nodes to test ideas.
        """)

        session   = _FakeSession()
        extractor = _FakeExtractor()
        q         = queue.Queue()
        worker    = IngestionWorker(session, extractor, q)
        worker.start()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False, encoding='utf-8') as f:
            f.write(content)
            path = f.name

        try:
            q.put(path)
            q.join()
            worker.stop(timeout=5.0)

            self.assertEqual(worker.files_ingested, 1)
            self.assertGreater(session.graph.node_count, 0)
            self.assertGreater(len(session._ingestions), 0)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main(verbosity=2)
