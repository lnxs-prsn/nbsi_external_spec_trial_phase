"""
NBSI v1.0 — StructuralNodeLibrary

The servant's structural identity. Persists across all sessions.
Contains ONLY refined processing patterns (centroids).
No content from any session is stored here.

The library grows by refinement, not accumulation:
  - Early: new sessions frequently produce new node candidates. Library expands.
  - Mature: sessions confirm and refine existing nodes. Growth rate falls.
  - Asymptote: every pattern is cleanly matched. Growth stops. Peak capability.

FAISS upgrade path: replace find_similar() internals with IndexFlatIP.
Interface is unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from nbsi.lifecycle.nodes import StructuralNode


class StructuralNodeLibrary:
    def __init__(self):
        self._nodes: dict[str, StructuralNode] = {}
        # Session degree distributions per observation node (for diversity scoring)
        # Maps obs_node_id -> list of np.ndarray degree distributions
        self._session_distributions: dict[str, list[np.ndarray]] = {}
        # Entity firing counts per observation node (for generalisation constraint)
        self._entity_counts: dict[str, dict[str, int]] = {}

    # ── Node management ───────────────────────────────────────────────────────
    def add(self, node: StructuralNode) -> None:
        self._nodes[node.node_id] = node

    def get(self, node_id: str) -> Optional[StructuralNode]:
        return self._nodes.get(node_id)

    def remove(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)

    @property
    def nodes(self) -> dict[str, StructuralNode]:
        return self._nodes

    # ── Similarity matching ───────────────────────────────────────────────────
    def match(self, embedding: list[float],
              threshold: float = 0.75) -> list[tuple[str, float]]:
        """
        Find structural nodes that fire for the given embedding.
        Returns list of (node_id, similarity) sorted desc.
        FAISS upgrade: replace this function's body with IndexFlatIP query.
        """
        if not embedding:
            return []
        qv = np.array(embedding)
        qn = np.linalg.norm(qv)
        if qn < 1e-10:
            return []
        qv = qv / qn
        results = []
        for nid, node in self._nodes.items():
            if node.pattern_embedding is None:
                continue
            ev = np.array(node.pattern_embedding)
            en = np.linalg.norm(ev)
            if en < 1e-10:
                continue
            sim = float(np.dot(qv, ev / en))
            if sim >= threshold:
                results.append((nid, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_firing_nodes(self, embedding: list[float],
                         threshold: float = 0.75) -> list[StructuralNode]:
        """Return StructuralNode objects that fire for this embedding."""
        return [self._nodes[nid] for nid, _ in self.match(embedding, threshold)
                if nid in self._nodes]

    # ── Diversity tracking ────────────────────────────────────────────────────
    def record_session_distribution(self, obs_node_id: str,
                                    degree_dist: np.ndarray) -> None:
        if obs_node_id not in self._session_distributions:
            self._session_distributions[obs_node_id] = []
        self._session_distributions[obs_node_id].append(degree_dist)

    def get_session_distributions(self, obs_node_id: str) -> list[np.ndarray]:
        return self._session_distributions.get(obs_node_id, [])

    # ── Entity count tracking ─────────────────────────────────────────────────
    def update_entity_counts(self, obs_node_id: str,
                             entity_counts: dict[str, int]) -> None:
        if obs_node_id not in self._entity_counts:
            self._entity_counts[obs_node_id] = {}
        for eid, count in entity_counts.items():
            self._entity_counts[obs_node_id][eid] = \
                self._entity_counts[obs_node_id].get(eid, 0) + count

    def get_entity_counts(self, obs_node_id: str) -> dict[str, int]:
        return dict(self._entity_counts.get(obs_node_id, {}))

    # ── Efficiency metrics ────────────────────────────────────────────────────
    def infrastructure_cost(self) -> dict:
        total = len(self._nodes)
        active = sum(1 for n in self._nodes.values()
                     if n.attention_mode == "active")
        return {
            "total_structural_nodes": total,
            "infrastructure": total - active,
            "active": active,
            "infrastructure_fraction": (total - active) / max(total, 1),
        }

    def summary(self) -> dict:
        cost = self.infrastructure_cost()
        generalised = sum(1 for n in self._nodes.values() if n.is_generalised)
        cost["generalised_nodes"] = generalised
        cost["observation_candidates"] = len(self._session_distributions)
        return cost
