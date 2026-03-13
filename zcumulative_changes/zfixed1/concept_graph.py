"""
NBSI v1.0 — ConceptGraph

Session-scoped graph. Built fresh each session. Destroyed at session end.
Backed by NetworkX DiGraph with label-based node lookup for ensemble mode.

Key design decisions:
  - Nodes indexed by node_id (UUID) in NetworkX
  - Label index maintained separately for ensemble intersection (labels, not IDs)
  - Betweenness centrality pruning respects protected flag
  - Degree distribution exposed for lifecycle diversity scoring
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np

from nbsi.graph.node import ConceptNode
from nbsi.graph.edge import ConceptEdge


class ConceptGraph:
    def __init__(self):
        self._nx = nx.DiGraph()
        self._nodes: dict[str, ConceptNode] = {}       # node_id -> ConceptNode
        self._edges: dict[tuple, ConceptEdge] = {}     # (src_id, tgt_id) -> ConceptEdge
        self._label_index: dict[str, str] = {}         # label -> node_id

    # ── Node operations ───────────────────────────────────────────────────────
    def add_node(self, node: ConceptNode) -> str:
        """Add node. Returns node_id. Merges if label already exists."""
        existing_id = self._label_index.get(node.label)
        if existing_id:
            # Merge: take higher activation, preserve protected status
            existing = self._nodes[existing_id]
            existing.activation = max(existing.activation, node.activation)
            existing.agreement = max(existing.agreement, node.agreement)
            existing.protected = existing.protected or node.protected
            if node.embedding and not existing.embedding:
                existing.embedding = node.embedding
            return existing_id
        self._nodes[node.node_id] = node
        self._label_index[node.label] = node.node_id
        self._nx.add_node(node.node_id)
        return node.node_id

    def get_node(self, node_id: str) -> Optional[ConceptNode]:
        return self._nodes.get(node_id)

    def get_node_by_label(self, label: str) -> Optional[ConceptNode]:
        nid = self._label_index.get(label.lower().strip())
        return self._nodes.get(nid) if nid else None

    def remove_node(self, node_id: str) -> None:
        node = self._nodes.pop(node_id, None)
        if node:
            self._label_index.pop(node.label, None)
            # Remove associated edges
            for key in list(self._edges.keys()):
                if key[0] == node_id or key[1] == node_id:
                    del self._edges[key]
            self._nx.remove_node(node_id)

    # ── Edge operations ───────────────────────────────────────────────────────
    def add_edge(self, edge: ConceptEdge) -> None:
        """Add edge. Replaces if same (src, tgt) already exists."""
        key = (edge.source_id, edge.target_id)
        self._edges[key] = edge
        self._nx.add_edge(
            edge.source_id, edge.target_id,
            weight=edge.effective_weight,
            edge_type=edge.edge_type
        )

    def get_edge(self, source_id: str, target_id: str) -> Optional[ConceptEdge]:
        return self._edges.get((source_id, target_id))

    def edges_from(self, node_id: str) -> list[ConceptEdge]:
        return [self._edges[(node_id, tgt)] for tgt in self._nx.successors(node_id)
                if (node_id, tgt) in self._edges]

    def edges_to(self, node_id: str) -> list[ConceptEdge]:
        return [self._edges[(src, node_id)] for src in self._nx.predecessors(node_id)
                if (src, node_id) in self._edges]

    def all_edges(self) -> list[ConceptEdge]:
        return list(self._edges.values())

    # ── Structural metrics ────────────────────────────────────────────────────
    def betweenness_centrality(self) -> dict[str, float]:
        """Betweenness on the NX graph using effective_weight as edge cost."""
        if len(self._nodes) < 2:
            return {nid: 0.0 for nid in self._nodes}
        # Update NX edge weights from effective_weight before computing
        for (src, tgt), edge in self._edges.items():
            if self._nx.has_edge(src, tgt):
                self._nx[src][tgt]['weight'] = edge.effective_weight
        return nx.betweenness_centrality(self._nx, weight='weight', normalized=True)

    def degree_distribution(self) -> np.ndarray:
        """Return degree sequence as numpy array (for Wasserstein diversity scoring)."""
        degrees = [d for _, d in self._nx.degree()]
        return np.array(sorted(degrees), dtype=float) if degrees else np.array([0.0])

    def prune_by_betweenness(self, max_nodes: int) -> list[str]:
        """
        Prune to max_nodes using betweenness centrality.
        Protected nodes are immune. Returns list of pruned node_ids.
        """
        if len(self._nodes) <= max_nodes:
            return []
        bc = self.betweenness_centrality()
        # Sort unprotected nodes by betweenness ascending (lowest first = prune first)
        candidates = [
            (nid, score) for nid, score in bc.items()
            if not self._nodes[nid].protected
        ]
        candidates.sort(key=lambda x: x[1])
        n_to_prune = len(self._nodes) - max_nodes
        pruned = []
        for nid, _ in candidates[:n_to_prune]:
            self.remove_node(nid)
            pruned.append(nid)
        return pruned

    def protect_anchor_adjacent(self, anchor_node_id: str) -> None:
        """Mark anchor node and all direct neighbours as protected."""
        if anchor_node_id in self._nodes:
            self._nodes[anchor_node_id].protected = True
        for neighbour_id in list(self._nx.successors(anchor_node_id)) + \
                            list(self._nx.predecessors(anchor_node_id)):
            if neighbour_id in self._nodes:
                self._nodes[neighbour_id].protected = True

    # ── Similarity search (linear — FAISS upgrade path is clear) ─────────────
    def find_similar_nodes(self, query_embedding: list[float],
                           threshold: float = 0.0,
                           top_k: int = 10) -> list[tuple[str, float]]:
        """
        Linear cosine similarity search.
        Returns list of (node_id, similarity) sorted desc.
        Replace internals with FAISS when available.
        """
        if not query_embedding:
            return []
        qv = np.array(query_embedding)
        qn = np.linalg.norm(qv)
        if qn < 1e-10:
            return []
        qv = qv / qn
        results = []
        for nid, node in self._nodes.items():
            if node.embedding is None:
                continue
            ev = np.array(node.embedding)
            en = np.linalg.norm(ev)
            if en < 1e-10:
                continue
            sim = float(np.dot(qv, ev / en))
            if sim >= threshold:
                results.append((nid, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── Stability ─────────────────────────────────────────────────────────────
    def node_labels(self) -> set[str]:
        return set(self._label_index.keys())

    # ── Inspection ───────────────────────────────────────────────────────────
    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def summary(self) -> dict:
        protected = sum(1 for n in self._nodes.values() if n.protected)
        sem_active = sum(1 for e in self._edges.values() if e.has_active_sem)
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "protected_nodes": protected,
            "sem_active_edges": sem_active,
        }

    def reset(self) -> None:
        """Hard reset — destroys all session data."""
        self._nx.clear()
        self._nodes.clear()
        self._edges.clear()
        self._label_index.clear()
