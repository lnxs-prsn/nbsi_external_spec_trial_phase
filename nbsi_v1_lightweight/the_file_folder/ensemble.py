"""
NBSI v1.0 — Ensemble Mode

When stability falls below threshold, a second graph is built from
the source in reversed chunk order. Queries return only paths that
appear in BOTH graphs — paths grounded in the source's structure
regardless of reading order.

Critical: path intersection uses node LABELS not node IDs.
Two graphs assign different UUIDs to the same concept.
Labels are canonical (lowercased, from source).
"""
from __future__ import annotations

import numpy as np

from nbsi.graph.concept_graph import ConceptGraph
from nbsi.reasoning.conductivity import ConductivityEngine, QueryEngine, ReasoningPath
from nbsi.config import Config


def measure_stability(graph_a: ConceptGraph,
                      graph_b: ConceptGraph,
                      sample_embeddings: list[list[float]],
                      config: Config) -> float:
    """
    Stability = fraction of top-K paths that appear in both graphs
    (label-based intersection) across sample queries.

    Returns score in [0, 1]. 1.0 = perfectly stable.
    """
    if not sample_embeddings:
        return 1.0

    engine_a = QueryEngine(graph_a, ConductivityEngine(graph_a, config), config)
    engine_b = QueryEngine(graph_b, ConductivityEngine(graph_b, config), config)

    match_count = 0
    total = 0

    for emb in sample_embeddings:
        paths_a = engine_a.query(emb)
        paths_b = engine_b.query(emb)

        # Convert to label sequences for comparison
        labels_a = {
            tuple(
                graph_a.get_node(nid).label
                for nid in p.node_ids
                if graph_a.get_node(nid)
            )
            for p in paths_a
        }
        labels_b = {
            tuple(
                graph_b.get_node(nid).label
                for nid in p.node_ids
                if graph_b.get_node(nid)
            )
            for p in paths_b
        }

        intersection = labels_a & labels_b
        union = labels_a | labels_b

        if union:
            match_count += len(intersection)
            total += len(union)

    return match_count / total if total > 0 else 1.0


def ensemble_query(graph_primary: ConceptGraph,
                   graph_ensemble: ConceptGraph,
                   query_embedding: list[float],
                   config: Config) -> list[ReasoningPath]:
    """
    Return only paths whose label sequence appears in both graphs.
    Uses primary graph's paths and conductivity scores.
    Warns if intersection is empty.
    """
    engine_p = QueryEngine(graph_primary, ConductivityEngine(graph_primary, config), config)
    engine_e = QueryEngine(graph_ensemble, ConductivityEngine(graph_ensemble, config), config)

    paths_p = engine_p.query(query_embedding)
    paths_e = engine_e.query(query_embedding)

    # Label sequences from ensemble graph
    ensemble_label_seqs = {
        tuple(
            graph_ensemble.get_node(nid).label
            for nid in p.node_ids
            if graph_ensemble.get_node(nid)
        )
        for p in paths_e
    }

    # Filter primary paths to those whose label sequence is in ensemble
    intersection_paths = []
    for p in paths_p:
        label_seq = tuple(
            graph_primary.get_node(nid).label
            for nid in p.node_ids
            if graph_primary.get_node(nid)
        )
        if label_seq in ensemble_label_seqs:
            intersection_paths.append(p)

    if not intersection_paths:
        # Warn but return primary paths rather than empty
        print("[NBSI][WARN] Ensemble intersection empty — returning primary paths. "
              "Source may be highly order-dependent.")
        return paths_p

    return intersection_paths
