"""
NBSI v1.0 — Conductivity and Beam Search Traversal

Conductivity formula (geometric mean, length-normalised):
    C(P, q) = [ PROD f(ni, ni+1) ] ^ (1 / path_length)

where:
    f(ni, nj) = effective_weight(edge) × activation(nj) × recency(nj) × agreement(nj)

Geometric mean properties:
  - Length-normalised: longer paths not penalised purely by length
  - Single-factor resilient: one low-quality node dampens proportionally
  - Zero edge: drives score to zero (broken link is meaningful)

Beam search: maintains top-K partial paths by conductivity at each step.
Anchor nodes: query-matching nodes set as protected before traversal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nbsi.graph.concept_graph import ConceptGraph
from nbsi.graph.node import ConceptNode
from nbsi.config import Config


@dataclass
class ReasoningPath:
    node_ids: list[str] = field(default_factory=list)
    conductivity: float = 0.0
    hop_scores: list[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.node_ids)

    def to_dict(self, graph: ConceptGraph) -> dict:
        labels = []
        for nid in self.node_ids:
            node = graph.get_node(nid)
            labels.append(node.label if node else nid)
        return {
            "path": labels,
            "node_ids": self.node_ids,
            "conductivity": round(self.conductivity, 6),
            "hop_scores": [round(s, 4) for s in self.hop_scores],
            "length": self.length,
        }


class ConductivityEngine:
    def __init__(self, graph: ConceptGraph, config: Config):
        self.graph = graph
        self.config = config

    def hop_score(self, source_id: str, target_id: str) -> float:
        """
        Score for a single hop: edge effective_weight × target activation
        × target recency × target agreement.
        Returns 0.0 if edge not found.
        """
        edge = self.graph.get_edge(source_id, target_id)
        if edge is None:
            return 0.0
        target = self.graph.get_node(target_id)
        if target is None:
            return 0.0
        return (edge.effective_weight
                * target.activation
                * target.recency
                * target.agreement)

    def path_conductivity(self, node_ids: list[str]) -> tuple[float, list[float]]:
        """
        Geometric mean conductivity for a path.
        Returns (conductivity, list_of_hop_scores).
        Zero if any hop score is zero (broken link).
        """
        if len(node_ids) < 2:
            return (0.0, [])
        hop_scores = []
        for i in range(len(node_ids) - 1):
            s = self.hop_score(node_ids[i], node_ids[i + 1])
            hop_scores.append(s)
        # Geometric mean
        if any(s <= 0 for s in hop_scores):
            return (0.0, hop_scores)
        log_sum = sum(math.log(s) for s in hop_scores)
        conductivity = math.exp(log_sum / len(hop_scores))
        return (conductivity, hop_scores)

    def beam_search(self,
                    start_node_id: str,
                    max_depth: int = None,
                    beam_width: int = None,
                    visited_global: set = None) -> list[ReasoningPath]:
        """
        Beam search from start_node_id.
        Returns list of completed paths sorted by conductivity desc.
        """
        max_depth = max_depth or self.config.MAX_PATH_DEPTH
        beam_width = beam_width or self.config.BEAM_WIDTH
        visited_global = visited_global or set()

        # Each beam entry: (partial_path as list of node_ids, partial log-score sum, hop_scores)
        beam: list[tuple[list[str], float, list[float]]] = [
            ([start_node_id], 0.0, [])
        ]
        completed: list[ReasoningPath] = []

        for depth in range(max_depth):
            candidates = []
            for path, log_score, scores in beam:
                current = path[-1]
                successors = list(self.graph._nx.successors(current))
                expanded = False
                for nxt in successors:
                    if nxt in path:  # No cycles
                        continue
                    if nxt in visited_global:
                        continue
                    s = self.hop_score(current, nxt)
                    if s <= 0:
                        continue
                    new_log = log_score + math.log(s)
                    new_path = path + [nxt]
                    new_scores = scores + [s]
                    candidates.append((new_path, new_log, new_scores))
                    expanded = True
                if not expanded and len(path) > 1:
                    # Dead end — save as completed path
                    if log_score > -math.inf:
                        cond = math.exp(log_score / max(len(scores), 1))
                        completed.append(ReasoningPath(
                            node_ids=list(path),
                            conductivity=cond,
                            hop_scores=list(scores)
                        ))
            if not candidates:
                break
            # Keep top beam_width by log-score (normalised by length)
            candidates.sort(
                key=lambda x: x[1] / max(len(x[2]), 1),
                reverse=True
            )
            beam = candidates[:beam_width]

        # Save remaining beam entries as completed paths
        for path, log_score, scores in beam:
            if len(path) > 1 and log_score > -math.inf:
                cond = math.exp(log_score / max(len(scores), 1))
                completed.append(ReasoningPath(
                    node_ids=list(path),
                    conductivity=cond,
                    hop_scores=list(scores)
                ))

        # Deduplicate and sort
        seen = set()
        unique = []
        for p in sorted(completed, key=lambda x: x.conductivity, reverse=True):
            key = tuple(p.node_ids)
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique[:self.config.TOP_K_PATHS]


class QueryEngine:
    def __init__(self, graph: ConceptGraph, conductivity: ConductivityEngine, config: Config):
        self.graph = graph
        self.conductivity = conductivity
        self.config = config

    def find_anchor_nodes(self,
                          query_embedding: list[float],
                          top_k: int = 3) -> list[str]:
        """Find top-k nodes most similar to query embedding. These become anchors."""
        results = self.graph.find_similar_nodes(
            query_embedding,
            threshold=0.0,
            top_k=top_k
        )
        return [nid for nid, _ in results]

    def query(self,
              query_embedding: list[float],
              top_k_anchors: int = 3) -> list[ReasoningPath]:
        """
        Full query: find anchors, protect them, run beam search from each,
        merge and return top-K paths by conductivity.
        """
        anchor_ids = self.find_anchor_nodes(query_embedding, top_k=top_k_anchors)
        if not anchor_ids:
            return []

        # Protect anchor-adjacent nodes from pruning
        for anchor_id in anchor_ids:
            self.graph.protect_anchor_adjacent(anchor_id)

        all_paths: list[ReasoningPath] = []
        visited_across_anchors: set[str] = set()

        for anchor_id in anchor_ids:
            paths = self.conductivity.beam_search(
                start_node_id=anchor_id,
                visited_global=visited_across_anchors,
            )
            all_paths.extend(paths)
            # Add anchor's visited set to global to encourage diverse paths
            for p in paths:
                visited_across_anchors.update(p.node_ids[1:])  # Not the anchor itself

        # Sort by conductivity, deduplicate
        seen = set()
        unique = []
        for p in sorted(all_paths, key=lambda x: x.conductivity, reverse=True):
            key = tuple(p.node_ids)
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique[:self.config.TOP_K_PATHS]
