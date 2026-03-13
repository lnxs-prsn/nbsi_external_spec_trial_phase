"""
NBSI v1.0 — Persistence

Saves and loads the StructuralNodeLibrary to/from disk.

What is saved:
  - StructuralNode: node_id, pattern_embedding (centroid), session_count,
    diversity_score, is_generalised, attention_mode, reactivation_count
  - Library metadata: version, saved_at, node_count

What is NOT saved:
  - Any content, labels, or raw embeddings from sessions
  - Observation nodes (transient — they live only within a session lifecycle)
  - The ConceptGraph, edges, or any operational layer state
  - Session distributions or entity counts (scoring data — not structural state)

Format: JSON — human-readable, inspectable, no binary blobs.

This separation is the architectural guarantee: loading a saved library
gives you geometry only. No session content is recoverable.

Usage:
    from nbsi.session.persistence import save_library, load_library

    save_library(session.library, "~/.nbsi/library.json")

    library = load_library("~/.nbsi/library.json")
    session = NBSISession(structural_library=library, ...)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from nbsi.lifecycle.nodes import StructuralNode
from nbsi.lifecycle.structural_library import StructuralNodeLibrary

_FORMAT_VERSION = "1.0"


# ── Save ──────────────────────────────────────────────────────────────────────

def save_library(library: StructuralNodeLibrary, path: str) -> dict:
    """
    Serialise the StructuralNodeLibrary to a JSON file.

    Saves geometry only — no content, no labels, no session data.
    Creates parent directories if they do not exist.

    Returns a summary dict: {path, nodes_saved, saved_at}.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_data = []
    for node in library.nodes.values():
        nodes_data.append({
            "node_id":          node.node_id,
            "pattern_embedding": node.pattern_embedding,   # centroid only
            "session_count":    node.session_count,
            "diversity_score":  node.diversity_score,
            "is_generalised":   node.is_generalised,
            "attention_mode":   node.attention_mode,
            "reactivation_count": node.reactivation_count,
        })

    payload = {
        "nbsi_version":  _FORMAT_VERSION,
        "saved_at":      datetime.now(timezone.utc).isoformat(),
        "node_count":    len(nodes_data),
        "nodes":         nodes_data,
        # Explicit statement: no content is stored
        "_content_free": True,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "path":        str(path),
        "nodes_saved": len(nodes_data),
        "saved_at":    payload["saved_at"],
    }


# ── Load ──────────────────────────────────────────────────────────────────────

def load_library(path: str) -> StructuralNodeLibrary:
    """
    Deserialise a saved StructuralNodeLibrary from a JSON file.

    Returns a populated StructuralNodeLibrary ready to pass to NBSISession.
    Raises FileNotFoundError if the file does not exist.
    Raises ValueError on format mismatch or corrupt data.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"[persistence] Library file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    _validate(payload, path)

    library = StructuralNodeLibrary()

    for nd in payload["nodes"]:
        node = StructuralNode(
            node_id=nd["node_id"],
            pattern_embedding=nd["pattern_embedding"],
            session_count=nd.get("session_count", 0),
            diversity_score=nd.get("diversity_score", 0.0),
            is_generalised=nd.get("is_generalised", True),
        )
        # Restore runtime state
        node.attention_mode    = nd.get("attention_mode", "infrastructure")
        node.reactivation_count = nd.get("reactivation_count", 0)
        library.add(node)

    return library


# ── Info ──────────────────────────────────────────────────────────────────────

def library_info(path: str) -> Optional[dict]:
    """
    Return metadata about a saved library without fully loading it.
    Returns None if the file does not exist.
    Raises ValueError on corrupt data.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    _validate(payload, path)

    return {
        "path":       str(path),
        "version":    payload.get("nbsi_version"),
        "saved_at":   payload.get("saved_at"),
        "node_count": payload.get("node_count", 0),
        "content_free": payload.get("_content_free", False),
        "size_bytes": os.path.getsize(path),
    }


# ── Internal ──────────────────────────────────────────────────────────────────

def _validate(payload: dict, path: Path) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"[persistence] Invalid library file (not a JSON object): {path}")
    if "nodes" not in payload:
        raise ValueError(f"[persistence] Invalid library file (missing 'nodes' key): {path}")
    version = payload.get("nbsi_version")
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"[persistence] Version mismatch: file is '{version}', "
            f"expected '{_FORMAT_VERSION}': {path}"
        )
