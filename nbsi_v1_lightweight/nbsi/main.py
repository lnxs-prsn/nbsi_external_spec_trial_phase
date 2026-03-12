"""
NBSI v1.0 Lightweight — main.py (Phase 7: Orchestration) (FIXED)

Full session loop with PERSISTENT observation nodes:

 startup → load structural library AND observation nodes from disk
 → start file watcher on a watched folder
 → start ingestion worker
 loop → accept queries from stdin
 → print reasoning paths
 shutdown → end session
 → save structural library AND observation nodes to disk
 → repeat on next run (library accumulates across sessions)

The structural layer AND observation nodes grow over time. 
Run many sessions against the same or related documents and watch both counts climb.

Usage
---
 # Watch a folder and query interactively
 python nbsi/main.py --watch ~/Documents/nbsi-inbox

 # Ingest specific files at startup then query
 python nbsi/main.py --files paper.docx notes.md

 # Both
 python nbsi/main.py --files paper.docx --watch ~/Documents/nbsi-inbox

 # Custom library path
 python nbsi/main.py --watch ~/docs --library ~/myproject/library.json

 # Skip synthesis (faster)
 python nbsi/main.py --files paper.docx --no-synth

Commands during interactive loop
-------------------------------
  Query the graph
 :ingest  Ingest a file immediately (blocking)
 :stats Show session and worker stats
 :library Show structural library info (includes observations)
 :save Save library now (also saved automatically on exit)
 :quit / :exit End session and save

Library location
--------------
 Default: ~/.nbsi/library.json
 Override: --library

The library is content-free — it stores only geometry (centroids),
never document text or session content.
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import time

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIBRARY_PATH_DEFAULT = os.path.expanduser("~/.nbsi/library.json")

BANNER = """
=================================================================
 NBSI v1.0 Lightweight (FIXED with persistent observations)
 spaCy · MiniLM · Local graph reasoning · Persistent library
================================================================="""

HELP_TEXT = """
Commands:
  Query the graph
 :ingest  Ingest a file immediately
 :stats Worker and session stats
 :library Structural library info (structural + observation nodes)
 :save Save library to disk now
 :help Show this message
 :quit / :exit Save and exit
"""

def _header(text: str) -> None:
    print(f"\n{'='*65}\n {text}\n{'='*65}")

def _info(text: str) -> None:
    print(f" {text}")

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _load_components(args):
    """Load embedder, extractor. Returns (embedder, extractor) or exits."""
    print("\n[1/4] Loading MiniLM embedder...")
    t0 = time.time()
    try:
        from nbsi.embedder import RealEmbedder
        embedder = RealEmbedder()
        print(f" Ready ({time.time()-t0:.1f}s)")
    except ImportError as e:
        print(f"\n[!] {e}")
        sys.exit(1)

    try:
        from nbsi.ingestion.spacy_extractor import SpacyExtractor
        extractor = SpacyExtractor()
    except OSError as e:
        print(f"\n[!] {e}")
        sys.exit(1)

    return embedder, extractor

def _load_library(library_path: str, embedder, config):
    """Load or create a StructuralNodeLibrary AND LifecycleEngine with observation nodes."""
    from nbsi.lifecycle.structural_library import StructuralNodeLibrary
    from nbsi.lifecycle.lifecycle_engine import LifecycleEngine
    from nbsi.session.persistence import load_library, library_info

    info = library_info(library_path)
    if info is None:
        print(f"[2/4] No library found at {library_path} — starting fresh")
        library = StructuralNodeLibrary()
        lifecycle = LifecycleEngine(library, embedder, config)
        return library, lifecycle

    print(f"[2/4] Loading library: {info['node_count']} structural nodes, "
          f"{info.get('observation_count', 0)} observation nodes "
          f"(saved {info['saved_at'][:19]})")
    try:
        library, lifecycle = load_library(library_path, embedder, config)
        print(f" Loaded {len(library.nodes)} structural nodes, "
              f"{len(lifecycle._observation_nodes)} observation nodes")
        return library, lifecycle
    except Exception as e:
        print(f"[!] Could not load library ({e}) — starting fresh")
        library = StructuralNodeLibrary()
        lifecycle = LifecycleEngine(library, embedder, config)
        return library, lifecycle

def _build_session(embedder, library, lifecycle):
    """Create and return an NBSISession with attached lifecycle."""
    from nbsi.config import Config
    from nbsi.session.session import NBSISession

    config = Config()
    config.MAX_NODES = 700
    config.BEAM_WIDTH = 8
    config.TOP_K_PATHS = 5

    session = NBSISession(
        structural_library=library,
        embedder=embedder,
        config=config,
    )
    # FIXED: Attach the loaded lifecycle (with observation nodes) to session
    session.lifecycle = lifecycle
    return session

# remove later
# def _start_watcher(watch_folder: str, ingest_queue: queue.Queue):
#     """Start FileWatcher on watch_folder. Returns watcher or None."""
#     if not watch_folder:
#         return None

#     watch_folder = os.path.expanduser(watch_folder)
#     if not os.path.isdir(watch_folder):
#         print(f"[!] Watch folder does not exist: {watch_folder}")
#         print(f" Create it and drop files in to ingest automatically.")
#         try:
#             os.makedirs(watch_folder, exist_ok=True)
#             print(f" Created: {watch_folder}")
#         except Exception as e:
#             print(f" Could not create folder: {e}")
#             return None

#     try:
#         from nbsi.os_integration.watcher import FileWatcher
#         watcher = FileWatcher(watch_folder, ingest_queue)
#         watcher.start()
#         print(f"[3/4] Watching: {watch_folder}")
#         return watcher
#     except Exception as e:
#         print(f"[!] Could not start watcher: {e}")
#         return None
def _start_watcher(watch_folder: str, ingest_queue: queue.Queue):
    """Start file watcher on watch_folder. Returns observer or None."""
    if not watch_folder:
        return None

    watch_folder = os.path.expanduser(watch_folder)
    if not os.path.isdir(watch_folder):
        print(f"[!] Watch folder does not exist: {watch_folder}")
        try:
            os.makedirs(watch_folder, exist_ok=True)
            print(f" Created: {watch_folder}")
        except Exception as e:
            print(f" Could not create folder: {e}")
            return None

    try:
        # from nbsi.os_integration.watcher import start_watcher     # replaced
        # observer = start_watcher(watch_folder, ingest_queue)
        from nbsi.os_integration.watcher import start_watcher, scan_existing
        observer = start_watcher(watch_folder, ingest_queue)
        scan_existing(watch_folder, ingest_queue)  # Add this line
        print(f"[3/4] Watching: {watch_folder}")
        return observer
    except Exception as e:
        print(f"[!] Could not start watcher: {e}")
        return None



def _ingest_startup_files(session, extractor, file_paths: list) -> None:
    """Ingest files specified at startup via --files."""
    if not file_paths:
        return

    from nbsi.ingestion.pipeline import ingest_documents

    print(f"\n[4/4] Ingesting {len(file_paths)} startup file(s)...")
    t0 = time.time()
    report = ingest_documents(session, extractor, file_paths, verbose=True)
    elapsed = time.time() - t0

    print(f"\n Done: {report.files_ok} ok, {report.files_failed} failed, "
          f"{report.total_nodes} nodes, {report.total_edges} edges "
          f"({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save_library(library, lifecycle, library_path: str) -> None:
    from nbsi.session.persistence import save_library
    try:
        result = save_library(library, lifecycle, library_path)
        print(f" Library saved: {result['nodes_saved']} structural, "
              f"{result['observations_saved']} observation nodes → {result['path']}")
    except Exception as e:
        print(f" [!] Could not save library: {e}")

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def _run_query(session, query: str, use_synthesis: bool) -> list:
    """Run a query and print results. Returns paths list."""
    t0 = time.time()
    paths = session.query(query)
    ms = (time.time() - t0) * 1000

    print(f"\n {len(paths)} paths found ({ms:.1f}ms)")

    if not paths:
        print(" No paths found. Try a broader query or ingest more documents.")
        return paths

    print(" " + "-"*60)
    for i, p in enumerate(paths, 1):
        chain = " → ".join(p["path"])
        conf = p["conductivity"]
        hops = p["length"] - 1
        print(f"\n Path {i} [{conf:.3f} conductivity · "
              f"{hops} hop{'s' if hops != 1 else ''}]")
        print(f" {chain}")

    if use_synthesis and paths:
        _run_synthesis(query, paths)

    return paths

def _run_synthesis(query: str, paths: list) -> None:
    """Attempt synthesis narration. Silent if model not found."""
    default_model = os.path.expanduser(
        "~/nbsi-models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )
    model_path = os.environ.get("NBSI_MODEL", default_model)

    if not os.path.exists(model_path):
        return  # Silent — user can add --no-synth or set NBSI_MODEL

    try:
        from nbsi.synthesis.synthesiser import Synthesiser
        synth = Synthesiser(model_path, n_threads=4, verbose=False)
        print(f"\n Narrating... (streaming)\n " + "-"*60)
        print(" ", end="", flush=True)
        for token in synth.narrate_streaming(query, paths):
            print(token, end="", flush=True)
        print("\n " + "-"*60)
    except ImportError:
        pass  # llama-cpp not installed — skip silently

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_stats(session, worker) -> None:
    summary = session.graph.node_count
    lib_summary = session.library.summary()
    obs_count = len(session.lifecycle._observation_nodes)
    print(f"\n Session graph nodes: {summary}")
    print(f" Structural library nodes: {lib_summary['total_structural_nodes']}")
    print(f" Observation nodes (accumulating): {obs_count}")
    if worker:
        s = worker.stats()
        print(f" Worker files ingested: {s['files_ingested']}")
        print(f" Worker files failed: {s['files_failed']}")
        print(f" Worker nodes added: {s['nodes_added']}")
        print(f" Worker queue size: {s['queue_size']}")

def _cmd_library(library, lifecycle, library_path: str) -> None:
    from nbsi.session.persistence import library_info
    info = library_info(library_path)
    lib_count = library.summary()['total_structural_nodes']
    obs_count = len(lifecycle._observation_nodes)
    print(f"\n Structural nodes (in memory): {lib_count}")
    print(f" Observation nodes (in memory): {obs_count}")
    if info:
        print(f" Last saved: {info['saved_at'][:19]}")
        print(f" Saved structural: {info['node_count']}")
        print(f" Saved observation: {info.get('observation_count', 0)}")
        print(f" File: {info['path']}")
        print(f" Size: {info['size_bytes']} bytes")
    else:
        print(f" Not yet saved to disk")

def _cmd_ingest(session, extractor, path_arg: str) -> None:
    path = os.path.expanduser(path_arg.strip())
    if not os.path.isfile(path):
        print(f" [!] File not found: {path}")
        return

    from nbsi.ingestion.pipeline import ingest_documents
    print(f" Ingesting {path}...")
    t0 = time.time()
    report = ingest_documents(session, extractor, [path], verbose=False)
    elapsed = time.time() - t0

    if report.files_failed:
        print(f" [!] Failed: {report.results[0].error}")
    else:
        r = report.results[0]
        print(f" Done: {r.chunks} chunks · {r.nodes_added} nodes · "
              f"{r.edges_added} edges ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args) -> None:
    library_path = os.path.expanduser(args.library)

    print(BANNER)

    # -- Load components
    embedder, extractor = _load_components(args)

    # -- Load library (or start fresh) - FIXED: returns (library, lifecycle)
    from nbsi.config import Config
    config = Config()
    library, lifecycle = _load_library(library_path, embedder, config)

    # -- Build session with attached lifecycle
    session = _build_session(embedder, library, lifecycle)
    print(f" Session ready")

    # -- Start watcher
    ingest_queue = queue.Queue()
    watcher = _start_watcher(args.watch, ingest_queue) if args.watch else None

    # -- Start ingestion worker (even if no watcher — handles :ingest commands)
    from nbsi.os_integration.ingestion_worker import IngestionWorker
    worker = IngestionWorker(session, extractor, ingest_queue)
    worker.start()

    # -- Ingest startup files
    _ingest_startup_files(session, extractor,
                          [os.path.expanduser(p) for p in (args.files or [])])

    use_synthesis = not args.no_synth

    # -- Graceful shutdown on Ctrl+C
    def _shutdown(sig=None, frame=None):
        print("\n\n[Shutting down...]")
        # if watcher:       # replaced remove later
        #     watcher.stop()
        if watcher:
            watcher.stop()
            watcher.join()
        worker.stop()
        summary = session.end_session()
        lib_nodes = summary['structural_library']['total_structural_nodes']
        obs_nodes = len(session.lifecycle._observation_nodes)
        print(f" Session ended. Graph destroyed.")
        print(f" Structural nodes: {lib_nodes}")
        print(f" Observation nodes: {obs_nodes}")
        _save_library(library, session.lifecycle, library_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # -- Interactive loop
    print(f"\n{'='*65}")
    print(f" Ready. Type a query or :help for commands.")
    lib_nodes = library.summary()['total_structural_nodes']
    obs_nodes = len(lifecycle._observation_nodes)
    if lib_nodes == 0 and obs_nodes == 0:
        print(f" Library is empty — ingest documents to build it.")
    else:
        print(f" Library: {lib_nodes} structural, {obs_nodes} observation nodes loaded.")
    print(f"{'='*65}\n")

    while True:
        try:
            line = input("nbsi> ").strip()
        except EOFError:
            _shutdown()
            break

        if not line:
            continue

        # -- Commands
        if line in (":quit", ":exit", ":q"):
            _shutdown()
            break

        elif line == ":help":
            print(HELP_TEXT)

        elif line == ":stats":
            _cmd_stats(session, worker)

        elif line == ":library":
            _cmd_library(library, lifecycle, library_path)

        elif line == ":save":
            _save_library(library, lifecycle, library_path)

        elif line.startswith(":ingest "):
            path_arg = line[len(":ingest "):]
            _cmd_ingest(session, extractor, path_arg)

        elif line.startswith(":"):
            print(f" Unknown command: {line} (type :help for commands)")

        else:
            # -- Query
            _run_query(session, line, use_synthesis)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NBSI v1.0 — interactive session with persistent structural library and observation nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--watch", metavar="FOLDER",
        help="Watch a folder for new files and ingest them automatically",
    )
    parser.add_argument(
        "--files", metavar="FILE", nargs="+",
        help="Ingest these files at startup before entering the query loop",
    )
    parser.add_argument(
        "--library", metavar="PATH",
        default=LIBRARY_PATH_DEFAULT,
        help=f"Structural library file (default: {LIBRARY_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--no-synth", action="store_true",
        help="Skip LLM narration (faster, no model required)",
    )
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
