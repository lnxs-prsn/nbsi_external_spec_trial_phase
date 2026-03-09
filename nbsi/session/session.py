"""
NBSI v1.0 — NBSISession

Orchestrates the two-layer architecture:
  Structural layer: StructuralNodeLibrary — persists across sessions (passed in, not owned)
  Operational layer: ConceptGraph + SEM + QueryEngine — destroyed at session end

The two-layer separation is enforced here:
  - Nothing from the operational layer enters the structural library except
    pattern-geometry updates (centroid adjustments from reactivation/stabilisation)
  - The structural library is never reset by session end
  - The operational layer is always completely reset at session end

Session lifecycle:
  1. Wake    — structural complete, operational empty
  2. Ingest  — build session graph from source; structural nodes shape it
  3. Speculate (optional) — add speculation; SEM rewires conductivity
  4. Query   — beam search through session graph
  5. End     — post-session lifecycle processing; operational layer destroyed
"""
from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from nbsi.config import Config
from nbsi.embedder import EmbedderInterface
from nbsi.graph.concept_graph import ConceptGraph
from nbsi.graph.node import ConceptNode
from nbsi.graph.edge import ConceptEdge, EDGE_TYPE_WEIGHTS
from nbsi.sem.propagator import SEMPropagator
from nbsi.sem.speculative_node import SpeculativeNode
from nbsi.reasoning.conductivity import ConductivityEngine, QueryEngine, ReasoningPath
from nbsi.reasoning.ensemble import measure_stability, ensemble_query
from nbsi.lifecycle.lifecycle_engine import LifecycleEngine
from nbsi.lifecycle.structural_library import StructuralNodeLibrary


class NBSISession:
    def __init__(self,
                 structural_library: StructuralNodeLibrary,
                 embedder: EmbedderInterface,
                 config: Config,
                 entity_id: str = "default"):
        self.session_id = str(uuid.uuid4())
        self.entity_id = entity_id
        self.config = config
        self.embedder = embedder

        # Structural layer — shared, not owned by session
        self.library = structural_library
        self.lifecycle = LifecycleEngine(structural_library, embedder, config)

        # Operational layer — session-scoped
        self._reset_operational()

    # ── Operational layer management ──────────────────────────────────────────
    def _reset_operational(self) -> None:
        """Hard reset of all operational state. Structural layer untouched."""
        self.graph = ConceptGraph()
        self.graph_ensemble: Optional[ConceptGraph] = None
        self.sem = SEMPropagator(self.graph, self.embedder, self.config)
        self._conductivity = ConductivityEngine(self.graph, self.config)
        self._query_engine = QueryEngine(self.graph, self._conductivity, self.config)
        self.stability_score: Optional[float] = None
        self.ensemble_mode: bool = False
        self._session_degree_dist: Optional[np.ndarray] = None
        self._ingest_count: int = 0
        self._query_count: int = 0

    # ── Ingestion ─────────────────────────────────────────────────────────────
    def ingest_graph(self, nodes: list[ConceptNode],
                     edges: list[ConceptEdge]) -> dict:
        """
        Primary ingestion path. Accepts pre-built nodes and edges
        (from LLM extraction pipeline once that is wired up).
        Applies structural shaping, prunes, measures stability.
        """
        # Add nodes with embeddings
        for node in nodes:
            if node.embedding is None and node.label:
                emb = self.embedder.encode([node.label])[0].tolist()
                node.embedding = emb
            self.graph.add_node(node)

        # Add edges
        for edge in edges:
            self.graph.add_edge(edge)

        # Apply structural shaping from library
        shaping_stats = self._apply_structural_shaping()

        # Prune to MAX_NODES
        pruned = self.graph.prune_by_betweenness(self.config.MAX_NODES)

        self._session_degree_dist = self.graph.degree_distribution()
        self._ingest_count += 1

        return {
            "session_id": self.session_id,
            "nodes": self.graph.node_count,
            "edges": self.graph.edge_count,
            "pruned": len(pruned),
            **shaping_stats,
            **self.graph.summary(),
        }

    def ingest_text_simple(self, text: str) -> dict:
        """
        Simple ingestion from raw text using sentence-level chunking.
        Produces ConceptNodes from sentences and semantic edges from similarity.
        Suitable for testing without LLM extraction.
        LLM extraction pipeline replaces the chunking/extraction part of this method.
        """
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 10]
        if not sentences:
            return {"error": "No content extracted"}

        embeddings = self.embedder.encode(sentences)
        nodes = []
        for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
            # Use first 6 words as label
            words = sent.split()[:6]
            label = " ".join(words).lower()
            node = ConceptNode(
                label=label,
                embedding=emb.tolist(),
                activation=0.7,
                recency=1.0 - (i / max(len(sentences), 1)) * 0.2,  # Slight recency decay
            )
            nodes.append(node)

        # Add nodes
        node_ids = []
        for node in nodes:
            nid = self.graph.add_node(node)
            node_ids.append(nid)

        # Sequential edges (sentence order)
        edges = []
        for i in range(len(node_ids) - 1):
            edge = ConceptEdge(
                source_id=node_ids[i],
                target_id=node_ids[i + 1],
                edge_type="sequential",
            )
            self.graph.add_edge(edge)
            edges.append(edge)

        # Semantic edges (cosine similarity above threshold)
        emb_array = embeddings
        for i in range(len(node_ids)):
            for j in range(i + 2, min(i + 6, len(node_ids))):  # Local window
                vi = emb_array[i]
                vj = emb_array[j]
                sim = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-10))
                if sim >= self.config.SEMANTIC_EDGE_THRESHOLD:
                    edge = ConceptEdge(
                        source_id=node_ids[i],
                        target_id=node_ids[j],
                        edge_type="semantic",
                        base_weight=sim * EDGE_TYPE_WEIGHTS["semantic"],
                        agreement=sim,
                    )
                    self.graph.add_edge(edge)

        return self.ingest_graph([], [])  # Trigger shaping and pruning

    def _apply_structural_shaping(self) -> dict:
        """
        For each session node, check structural library.
        Firing structural nodes boost session node activation.
        Non-matching nodes go to lifecycle observation.
        """
        structural_firings = 0
        observation_nodes_created = 0
        reactivations = 0

        for node in list(self.graph._nodes.values()):
            if node.embedding is None:
                continue
            matches = self.library.match(
                node.embedding,
                threshold=self.config.REACTIVATION_HIGH_THRESHOLD
            )
            if matches:
                structural_firings += 1
                top_sim = matches[0][1]
                node.activation = min(1.0, node.activation + top_sim * 0.15)
                # Check anomaly zone for reactivation
                reactivated_ids = self.lifecycle.check_for_reactivation(node.embedding)
                for rid in reactivated_ids:
                    self.lifecycle.refine_reactivated(rid, node.embedding)
                    reactivations += 1
            else:
                # No structural match — create observation node
                self.lifecycle.observe(node.label, node.embedding, self.entity_id)
                observation_nodes_created += 1

        return {
            "structural_nodes_firing": structural_firings,
            "observation_nodes_created": observation_nodes_created,
            "reactivations": reactivations,
        }

    # ── Stability ─────────────────────────────────────────────────────────────
    def measure_stability(self, graph_reversed: ConceptGraph) -> float:
        """Measure stability between primary and reversed-order graph."""
        sample_embeddings = []
        sample_nodes = list(self.graph._nodes.values())[:self.config.STABILITY_SAMPLE_QUERIES]
        for node in sample_nodes:
            if node.embedding:
                sample_embeddings.append(node.embedding)
        self.stability_score = measure_stability(
            self.graph, graph_reversed, sample_embeddings, self.config
        )
        self.ensemble_mode = self.stability_score < self.config.STABILITY_THRESHOLD
        return self.stability_score

    # ── Query ─────────────────────────────────────────────────────────────────
    def query(self, query_text: str) -> list[dict]:
        """
        Query the session graph. Returns list of path dicts with conductivity scores.
        """
        query_emb = self.embedder.encode([query_text])[0].tolist()
        self._query_count += 1

        if self.ensemble_mode and self.graph_ensemble:
            paths = ensemble_query(
                self.graph, self.graph_ensemble, query_emb, self.config
            )
        else:
            paths = self._query_engine.query(query_emb)

        return [p.to_dict(self.graph) for p in paths]

    # ── Speculation ───────────────────────────────────────────────────────────
    def speculate(self, statement: str) -> dict:
        """Add a speculation. Returns spec_id and SEM stats."""
        spec = self.sem.add_speculation(statement)
        return {
            "spec_id": spec.spec_id,
            "statement": spec.statement,
            "status": spec.status,
            "edges_affected": len(spec.affected_edges),
        }

    def confirm_speculation(self, spec_id: str) -> bool:
        return self.sem.confirm(spec_id)

    def rollback_speculation(self, spec_id: str) -> dict:
        success = self.sem.rollback(spec_id)
        tolerance = self.sem.verify_rollback_tolerance(spec_id)
        return {
            "success": success,
            "rollback_tolerance": tolerance,
            "tolerance_ok": tolerance < 1e-9,
        }

    # ── Session end ───────────────────────────────────────────────────────────
    def end_session(self) -> dict:
        """
        End session:
        1. Run post-session lifecycle (evaluate observation nodes for stabilisation)
        2. Hard reset operational layer
        3. Structural library persists untouched

        Returns summary of session and lifecycle activity.
        """
        lifecycle_summary = {}
        if self._session_degree_dist is not None:
            lifecycle_summary = self.lifecycle.post_session(
                self._session_degree_dist, self.entity_id
            )

        graph_summary = self.graph.summary()
        lib_summary = self.library.summary()

        # Hard reset operational layer
        self._reset_operational()

        return {
            "session_id": self.session_id,
            "session_graph_destroyed": True,
            "graph_at_end": graph_summary,
            "lifecycle": lifecycle_summary,
            "structural_library": lib_summary,
        }

    # ── State inspection ──────────────────────────────────────────────────────
    def state(self) -> dict:
        return {
            "session_id": self.session_id,
            "entity_id": self.entity_id,
            "nodes": self.graph.node_count,
            "edges": self.graph.edge_count,
            "stability_score": self.stability_score,
            "ensemble_mode": self.ensemble_mode,
            "active_speculations": len(self.sem.active_speculations),
            "observation_nodes": self.lifecycle.observation_count,
            "structural_nodes": len(self.library.nodes),
            "library": self.library.infrastructure_cost(),
        }
