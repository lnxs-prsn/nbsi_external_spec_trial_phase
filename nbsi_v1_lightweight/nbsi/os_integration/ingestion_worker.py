"""
nbsi/os_integration/ingestion_worker.py

Background thread that drains the file queue and feeds documents
into the NBSI session via the Phase 4 pipeline.

Design:
    - Single daemon thread — never blocks a query
    - Pulls paths from queue.Queue one at a time
    - Runs each file through the full Phase 4 pipeline:
        document_reader -> chunker -> metadata_nodes -> spacy_extractor
    - Calls session.ingest_graph() with structured output
    - Reports progress to stdout
    - Gracefully stops on worker.stop() or process exit
"""

import queue
import threading
import time


class IngestionWorker:
    """
    Background ingestion worker.

    Parameters
    ----------
    session    : NBSISession    - the live session to ingest into
    extractor  : SpacyExtractor - the extraction pipeline
    queue      : queue.Queue    - receives file paths from the watcher

    Usage
    -----
        worker = IngestionWorker(session, extractor, ingest_queue)
        worker.start()
        # ... later ...
        worker.stop()
    """

    def __init__(self, session, extractor, ingest_queue: queue.Queue):
        self.session   = session
        self.extractor = extractor
        self.queue     = ingest_queue

        self._stop_event = threading.Event()
        self._thread     = threading.Thread(
            target=self._run,
            name='nbsi-ingestion',
            daemon=True,
        )

        self.files_ingested  = 0
        self.files_failed    = 0
        self.nodes_added     = 0
        self.edges_added     = 0

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        self._thread.start()
        print('[worker] Ingestion worker started')

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        print(f'[worker] Stopped - '
              f'{self.files_ingested} files ingested, '
              f'{self.nodes_added} nodes, '
              f'{self.edges_added} edges')

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stats(self) -> dict:
        return {
            'files_ingested': self.files_ingested,
            'files_failed':   self.files_failed,
            'nodes_added':    self.nodes_added,
            'edges_added':    self.edges_added,
            'queue_size':     self.queue.qsize(),
        }

    # -- main loop -----------------------------------------------------------

    def _run(self):
        while not self._stop_event.is_set():
            try:
                path = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._ingest_one(path)
            except Exception as e:
                print(f'[worker] Unexpected error on {path}: {e}')
                self.files_failed += 1
            finally:
                self.queue.task_done()

    # -- single file ---------------------------------------------------------

    def _ingest_one(self, path: str):
        from nbsi.ingestion.pipeline import ingest_documents

        t0 = time.monotonic()
        print(f'[worker] Ingesting {path}')

        report = ingest_documents(
            self.session, self.extractor, [path], verbose=False
        )

        if report.files_failed:
            print(f'[worker] Failed {path}: {report.results[0].error}')
            self.files_failed += 1
            return

        elapsed = time.monotonic() - t0
        r       = report.results[0]
        n_nodes = r.nodes_added
        n_edges = r.edges_added

        self.files_ingested += 1
        self.nodes_added    += n_nodes
        self.edges_added    += n_edges

        print(f'[worker] Done  {r.chunks} chunks  {n_nodes} nodes  '
              f'{n_edges} edges  ({elapsed:.1f}s)  '
              f'graph={self.session.graph.node_count} total')