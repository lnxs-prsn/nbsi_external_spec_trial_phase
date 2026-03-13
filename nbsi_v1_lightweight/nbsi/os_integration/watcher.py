"""
nbsi/os_integration/watcher.py

Watches the file system for new and modified documents.
Enqueues paths for the IngestionWorker — does not ingest directly.

Debounce: a file is only enqueued once per 10-second window.
This prevents repeated ingestion when an editor saves every few seconds.
"""

import os
import queue
import threading

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from nbsi.os_integration.reader import SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Event handler
# ---------------------------------------------------------------------------

class NBSIFileWatcher(FileSystemEventHandler):
    """
    Listens for file creation and modification events.
    Filters to supported extensions and debounces rapid events.
    """

    def __init__(self, ingest_queue: queue.Queue, debounce_seconds: float = 10.0):
        super().__init__()
        self.queue            = ingest_queue
        self.debounce_seconds = debounce_seconds
        self._seen            = set()       # paths currently in debounce window
        self._lock            = threading.Lock()

    # -- watchdog callbacks --------------------------------------------------

    def on_created(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_moved(self, event):
        # File renamed into the watched dir — treat destination as new
        if not event.is_directory:
            self._enqueue(event.dest_path)

    # -- internal ------------------------------------------------------------

    def _enqueue(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return

        with self._lock:
            if path in self._seen:
                return
            self._seen.add(path)

        self.queue.put(path)

        # After debounce window expires, allow re-ingestion if file changes again
        threading.Timer(
            self.debounce_seconds,
            lambda: self._expire(path)
        ).start()

    def _expire(self, path: str):
        with self._lock:
            self._seen.discard(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_watcher(
    watch_path: str,
    ingest_queue: queue.Queue,
    debounce_seconds: float = 10.0,
    recursive: bool = True,
) -> Observer:
    """
    Start watching watch_path for supported file events.
    Enqueues file paths into ingest_queue for the IngestionWorker.

    Returns the Observer — call observer.stop() on shutdown.

    Usage:
        q        = queue.Queue()
        observer = start_watcher('~/Documents', q)
        # ... later ...
        observer.stop()
        observer.join()
    """
    watch_path = os.path.expanduser(watch_path)

    if not os.path.isdir(watch_path):
        raise ValueError(f'[watcher] Watch path does not exist: {watch_path}')

    handler  = NBSIFileWatcher(ingest_queue, debounce_seconds=debounce_seconds)
    observer = Observer()
    observer.schedule(handler, watch_path, recursive=recursive)
    observer.start()

    print(f'[watcher] Watching {watch_path}  (recursive={recursive})')
    return observer


def scan_existing(
    watch_path: str,
    ingest_queue: queue.Queue,
) -> int:
    """
    Walk watch_path and enqueue any existing supported files.
    Useful on startup to ingest documents already present before the
    watcher was running.

    Returns the number of files enqueued.
    """
    watch_path = os.path.expanduser(watch_path)
    count = 0

    for root, _dirs, files in os.walk(watch_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                ingest_queue.put(os.path.join(root, fname))
                count += 1

    if count:
        print(f'[watcher] Queued {count} existing file(s) from {watch_path}')

    return count
