"""
NBSI v1.0 — SEM Propagator

Speculation-as-impurity: introducing a speculative statement changes
the conductivity of existing edges throughout the session graph.

Edges semantically aligned with the speculation gain effective_weight.
Edges semantically opposed lose it.
Effect decays exponentially with graph distance from the speculative node.

base_weight is NEVER touched during propagation or rollback.
Only effective_weight changes (via sem_deltas on each ConceptEdge).

Confirmation writes deltas permanently to base_weight.
Rollback removes all deltas — effective_weight returns to base_weight with
zero trace. Tolerance verified < 1e-9.

FIX (v1.0.1): Speculative node is now anchored to the top-K most similar
existing nodes before BFS propagation. Without this the spec node was an
island with no neighbours and propagation never fired.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from nbsi.graph.concept_graph import ConceptGraph
from nbsi.graph.edge import ConceptEdge
from nbsi.sem.speculative_node import SpeculativeNode
from nbsi.embedder import EmbedderInterface
from nbsi.config import Config


class SEMPropagator:
    def __init__(self, graph: ConceptGraph, embedder: EmbedderInterface, config: Config):
        self.graph = graph
        self.embedder = embedder
        self.config = config
        self._active: dict[str, SpeculativeNode] = {}   # spec_id -> SpeculativeNode

    # ── Public API ────────────────────────────────────────────────────────────
    def add_speculation(self, statement: str) -> SpeculativeNode:
        """
        Add a speculative statement. Creates SpeculativeNode, adds to graph,
        anchors it to similar existing nodes, runs BFS propagation, returns the node.
        """
        emb = self.embedder.encode([statement])[0].tolist()
        spec = SpeculativeNode(statement=statement, embedding=emb)

        # Add speculative node to graph
        from nbsi.graph.node import ConceptNode
        spec_graph_node = ConceptNode(
            label=f"__spec__{spec.spec_id[:8]}",
            node_id=spec.spec_id,
            embedding=emb,
            activation=self.config.SPEC_INITIAL_ACTIVATION,
        )
        self.graph.add_node(spec_graph_node)

        # ── Anchor: connect spec node to most similar existing nodes ──────────
        # Without anchoring the spec node is an island — BFS finds no neighbours
        # and propagation never fires. We draw speculative edges to the top-K
        # most similar existing nodes so propagation has real entry points.
        similar = self.graph.find_similar_nodes(
            emb,
            threshold=0.25,    # Low threshold — want neighbours even on weak match
            top_k=5,
        )
        anchored = 0
        for nid, sim in similar:
            if nid == spec.spec_id:
                continue
            anchor_edge = ConceptEdge(
                source_id=spec.spec_id,
                target_id=nid,
                edge_type="speculative",
                base_weight=float(sim) * self.config.SEM_ALPHA,
            )
            self.graph.add_edge(anchor_edge)
            anchored += 1

        # Propagate SEM through the graph
        self._propagate(spec)
        self._active[spec.spec_id] = spec
        return spec

    def confirm(self, spec_id: str) -> bool:
        """
        Confirm a speculation. Writes all SEM deltas permanently to base_weight.
        Raises the speculative node's activation.
        Returns True if found and confirmed.
        """
        spec = self._active.get(spec_id)
        if spec is None or spec.is_resolved:
            return False
        for src_id, tgt_id in spec.affected_edges:
            edge = self.graph.get_edge(src_id, tgt_id)
            if edge:
                edge.confirm_sem(spec_id)
        # Raise activation of the spec node
        node = self.graph.get_node(spec_id)
        if node:
            node.activation = self.config.SPEC_CONFIRMED_ACTIVATION
        spec.confirmed = True
        return True

    def rollback(self, spec_id: str) -> bool:
        """
        Disconfirm and roll back a speculation.
        Removes all SEM deltas — effective_weight returns to base_weight.
        Prunes the speculative node from the graph.
        Returns True if found and rolled back.
        """
        spec = self._active.get(spec_id)
        if spec is None or spec.is_resolved:
            return False
        for src_id, tgt_id in spec.affected_edges:
            edge = self.graph.get_edge(src_id, tgt_id)
            if edge:
                edge.remove_sem_delta(spec_id)
        # Prune the speculative node
        if spec_id in self.graph._nodes:
            self.graph.remove_node(spec_id)
        spec.disconfirmed = True
        return True

    def verify_rollback_tolerance(self, spec_id: str) -> float:
        """
        After rollback: maximum deviation of effective_weight from base_weight
        across all formerly-affected edges. Should be < 1e-9.
        """
        max_delta = 0.0
        spec = self._active.get(spec_id)
        if spec is None:
            return 0.0
        for src_id, tgt_id in spec.affected_edges:
            edge = self.graph.get_edge(src_id, tgt_id)
            if edge:
                diff = abs(edge.effective_weight - edge.base_weight)
                max_delta = max(max_delta, diff)
        return max_delta

    @property
    def active_speculations(self) -> list[SpeculativeNode]:
        return [s for s in self._active.values() if not s.is_resolved]

    # ── Internal propagation ──────────────────────────────────────────────────
    def _propagate(self, spec: SpeculativeNode) -> None:
        """
        BFS from speculative node outward to SEM_RADIUS hops.
        For each edge in the neighbourhood, compute semantic alignment
        between edge and speculation. Apply signed delta to effective_weight.
        """
        visited_edges: set[tuple] = set()
        # BFS queue: (node_id, distance_from_spec)
        queue = deque()
        for neighbour_id in (list(self.graph._nx.successors(spec.spec_id)) +
                              list(self.graph._nx.predecessors(spec.spec_id))):
            queue.append((neighbour_id, 1))
        seen_nodes = {spec.spec_id}

        spec_emb = np.array(spec.embedding)

        while queue:
            node_id, dist = queue.popleft()
            if dist > self.config.SEM_RADIUS:
                continue
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            # Process all edges incident on this node
            for edge in self.graph.edges_from(node_id) + self.graph.edges_to(node_id):
                key = (edge.source_id, edge.target_id)
                if key in visited_edges:
                    continue
                visited_edges.add(key)

                # Compute alignment between edge and speculation
                alignment = self._edge_alignment(edge, spec_emb)
                decay = math.exp(-self.config.SEM_DECAY_RATE * dist)
                delta = self.config.SEM_ALPHA * alignment * decay
                edge.apply_sem_delta(spec.spec_id, delta)
                spec.affected_edges.append(key)

            # Expand BFS
            for next_id in (list(self.graph._nx.successors(node_id)) +
                             list(self.graph._nx.predecessors(node_id))):
                if next_id not in seen_nodes:
                    queue.append((next_id, dist + 1))

    def _edge_alignment(self, edge: ConceptEdge, spec_embedding: np.ndarray) -> float:
        """
        Cosine similarity between edge representation and speculation.
        Edge is represented as the mean of its endpoint embeddings.
        Returns value in [-1, 1]: positive = aligned, negative = opposed.
        """
        src_node = self.graph.get_node(edge.source_id)
        tgt_node = self.graph.get_node(edge.target_id)
        if src_node is None or tgt_node is None:
            return 0.0
        if src_node.embedding is None or tgt_node.embedding is None:
            return 0.0
        edge_vec = (np.array(src_node.embedding) + np.array(tgt_node.embedding)) / 2.0
        norm_e = np.linalg.norm(edge_vec)
        norm_s = np.linalg.norm(spec_embedding)
        if norm_e < 1e-10 or norm_s < 1e-10:
            return 0.0
        return float(np.dot(edge_vec / norm_e, spec_embedding / norm_s))
