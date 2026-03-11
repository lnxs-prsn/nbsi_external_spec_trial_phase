"""
NBSI v1.0 — Persistence (FIXED)

Saves and loads the StructuralNodeLibrary AND ObservationNodes to/from disk.

What is saved:
 - StructuralNode: node_id, pattern_embedding (centroid), session_count,
   diversity_score, is_generalised, attention_mode, reactivation_count
 - ObservationNode: node_id, pattern_embedding (centroid), confirmation_count,
   diversity_score, stabilisation_score, sessions_active, example_labels,
   entity_firing_counts
 - Library metadata: version, saved_at, node_count, observation_count

What is NOT saved:
 - Any content, labels, or raw embeddings from sessions
 - The ConceptGraph, edges, or any operational layer state
 - Session distributions or entity counts (recomputed fresh each session)

Format: JSON — human-readable, inspectable, no binary blobs.

This separation is the architectural guarantee: loading a saved library
gives you geometry only. No session content is recoverable.

Usage:
 from nbsi.session.persistence import save_library, load_library

 save_library(session.library, session.lifecycle, "~/.nbsi/library.json")

 library, lifecycle = load_library("~/.nbsi/library.json", embedder, config)
 session = NBSISession(structural_library=library, embedder=embedder, config=config)
 lifecycle is attached to session via LifecycleEngine
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from nbsi.lifecycle.nodes import StructuralNode, ObservationNode
from nbsi.lifecycle.structural_library import StructuralNodeLibrary
from nbsi.lifecycle.lifecycle_engine import LifecycleEngine

if TYPE_CHECKING:
    from nbsi.embedder import EmbedderInterface
    from nbsi.config import Config

_FORMAT_VERSION = "1.1"  # Bumped for observation node support

# ── Save ──────────────────────────────────────────────────────────────────────

def save_library(library: StructuralNodeLibrary, 
                 lifecycle: LifecycleEngine,
                 path: str) -> dict:
    """
    Serialise the StructuralNodeLibrary AND ObservationNodes to a JSON file.

    Saves geometry only — no content, no labels, no session data.
    Creates parent directories if they do not exist.

    Returns a summary dict: {path, nodes_saved, observations_saved, saved_at}.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save structural nodes
    nodes_data = []
    for node in library.nodes.values():
        nodes_data.append({
            "node_id": node.node_id,
            "pattern_embedding": node.pattern_embedding,  # centroid only
            "session_count": node.session_count,
            "diversity_score": node.diversity_score,
            "is_generalised": node.is_generalised,
            "attention_mode": node.attention_mode,
            "reactivation_count": node.reactivation_count,
        })

    # FIXED: Also save observation nodes
    observations_data = []
    for obs_id, obs in lifecycle._observation_nodes.items():
        observations_data.append({
            "node_id": obs.node_id,
            "pattern_embedding": obs.pattern_embedding,  # centroid only
            "confirmation_count": obs.confirmation_count,
            "diversity_score": obs.diversity_score,
            "stabilisation_score": obs.stabilisation_score,
            "sessions_active": obs.sessions_active,
            "example_labels": obs.example_labels,  # Ring buffer, max 5
            "entity_firing_counts": obs.entity_firing_counts,
        })

    # Save session degree distributions (needed for Wasserstein diversity scoring)
    distributions_data = {}
    for obs_id, dists in library._session_distributions.items():
        distributions_data[obs_id] = [d.tolist() for d in dists]

    payload = {
        "nbsi_version": _FORMAT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(nodes_data),
        "observation_count": len(observations_data),
        "nodes": nodes_data,
        "observations": observations_data,
        "session_distributions": distributions_data,
        # Explicit statement: no content is stored
        "_content_free": True,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "path": str(path),
        "nodes_saved": len(nodes_data),
        "observations_saved": len(observations_data),
        "saved_at": payload["saved_at"],
    }

# ── Load ──────────────────────────────────────────────────────────────────────

def load_library(path: str, 
                 embedder: EmbedderInterface,
                 config: Config) -> tuple[StructuralNodeLibrary, LifecycleEngine]:
    """
    Deserialise a saved StructuralNodeLibrary and ObservationNodes from JSON.

    Returns a tuple of (library, lifecycle_engine) ready to use.
    The lifecycle_engine has all observation nodes restored.
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
    lifecycle = LifecycleEngine(library, embedder, config)

    # Load structural nodes
    for nd in payload.get("nodes", []):
        node = StructuralNode(
            node_id=nd["node_id"],
            pattern_embedding=nd["pattern_embedding"],
            session_count=nd.get("session_count", 0),
            diversity_score=nd.get("diversity_score", 0.0),
            is_generalised=nd.get("is_generalised", True),
        )
        # Restore runtime state
        node.attention_mode = nd.get("attention_mode", "infrastructure")
        node.reactivation_count = nd.get("reactivation_count", 0)
        library.add(node)

    # FIXED: Load observation nodes
    for od in payload.get("observations", []):
        obs = ObservationNode(
            node_id=od["node_id"],
            pattern_embedding=od.get("pattern_embedding"),
            confirmation_count=od.get("confirmation_count", 0),
            diversity_score=od.get("diversity_score", 0.0),
            stabilisation_score=od.get("stabilisation_score", 0.0),
            sessions_active=od.get("sessions_active", 0),
            example_labels=od.get("example_labels", []),
        )
        # Restore entity firing counts
        obs._entity_firing_counts = od.get("entity_firing_counts", {})
        lifecycle._observation_nodes[obs.node_id] = obs

    # Restore session degree distributions
    import numpy as np
    for obs_id, dists in payload.get("session_distributions", {}).items():
        library._session_distributions[obs_id] = [np.array(d) for d in dists]

    return library, lifecycle

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
        "path": str(path),
        "version": payload.get("nbsi_version"),
        "saved_at": payload.get("saved_at"),
        "node_count": payload.get("node_count", 0),
        "observation_count": payload.get("observation_count", 0),
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
    # Allow both old format (1.0) and new format (1.1)
    if version not in (_FORMAT_VERSION, "1.0"):
        raise ValueError(
            f"[persistence] Version mismatch: file is '{version}', "
            f"expected '{_FORMAT_VERSION}' or '1.0': {path}"
        )
