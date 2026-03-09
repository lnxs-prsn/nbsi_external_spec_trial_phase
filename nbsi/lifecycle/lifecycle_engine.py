"""
NBSI v1.0 — LifecycleEngine

Implements all three lifecycle stages:

Stage 1 — Observation:
    Content that matches no structural node creates an ObservationNode.
    ObservationNode is expensive (active attention), provisional.

Stage 2 — Stabilisation:
    After a session, observation nodes are evaluated.
    Nodes that cross MIN_CONFIRMATIONS AND STABILISATION_THRESHOLD
    are promoted to StructuralNode. Content is discarded; only centroid survives.
    Generalisation constraint enforced before promotion.

Stage 3 — Reactivation:
    Content in the anomaly zone (REACTIVATION_LOW < sim < REACTIVATION_HIGH)
    reactivates a structural node temporarily.
    refine_reactivated() adjusts the centroid boundary and returns node to infrastructure.
    The anomaly is the mechanism of refinement — not a problem to be solved.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
from scipy.stats import wasserstein_distance

from nbsi.lifecycle.nodes import ObservationNode, StructuralNode
from nbsi.lifecycle.structural_library import StructuralNodeLibrary
from nbsi.embedder import EmbedderInterface
from nbsi.config import Config


class LifecycleEngine:
    def __init__(self,
                 library: StructuralNodeLibrary,
                 embedder: EmbedderInterface,
                 config: Config):
        self.library = library
        self.embedder = embedder
        self.config = config
        self._observation_nodes: dict[str, ObservationNode] = {}

    # ── Stage 1: Observation ──────────────────────────────────────────────────
    def observe(self, label: str, embedding: list[float],
                entity_id: str = "unknown") -> str:
        """
        Called when content matches no structural node.
        Merges into nearby observation node if one exists, else creates new.
        Returns observation node_id.
        """
        qv = np.array(embedding)
        qn = np.linalg.norm(qv)
        if qn > 1e-10:
            qv = qv / qn

        # Check for merge with existing observation node
        for obs in self._observation_nodes.values():
            if obs.pattern_embedding is None:
                continue
            ev = np.array(obs.pattern_embedding)
            en = np.linalg.norm(ev)
            if en < 1e-10:
                continue
            sim = float(np.dot(qv, ev / en))
            if sim >= self.config.OBSERVATION_MERGE_THRESHOLD:
                obs.update_pattern(embedding, label, entity_id)
                return obs.node_id

        # New observation node
        obs = ObservationNode()
        obs.update_pattern(embedding, label, entity_id)
        self._observation_nodes[obs.node_id] = obs
        return obs.node_id

    def get_observation_node(self, obs_id: str) -> Optional[ObservationNode]:
        return self._observation_nodes.get(obs_id)

    # ── Stage 2: Stabilisation ────────────────────────────────────────────────
    def evaluate_for_stabilisation(self, obs_id: str,
                                   session_degree_distributions: list[np.ndarray]) -> bool:
        """
        Evaluate whether an observation node should be promoted to structural.
        Returns True if promoted.

        Requirements:
          1. confirmation_count >= MIN_CONFIRMATIONS
          2. Wasserstein diversity across session distributions >= DIVERSITY_THRESHOLD
          3. stabilisation_score >= STABILISATION_THRESHOLD
          4. Generalisation constraint: not entity-specific
        """
        obs = self._observation_nodes.get(obs_id)
        if obs is None:
            return False

        if obs.confirmation_count < self.config.MIN_CONFIRMATIONS:
            return False

        if len(session_degree_distributions) < 2:
            return False

        # Compute pairwise Wasserstein distance as diversity measure
        pairwise = [
            wasserstein_distance(d1, d2)
            for d1, d2 in combinations(session_degree_distributions, 2)
        ]
        diversity = float(np.mean(pairwise))

        score = obs.compute_stabilisation_score(diversity)
        if score < self.config.STABILISATION_THRESHOLD:
            return False

        # Generalisation constraint — must pass before promotion
        if not self._check_generalisation(obs):
            return False

        self._promote(obs_id)
        return True

    def _check_generalisation(self, obs: ObservationNode) -> bool:
        """
        Returns False if the observation node fires predominantly for one entity.
        A node that fires >ENTITY_SPECIFICITY_THRESHOLD fraction from one entity
        must not be promoted — it encodes entity-specific knowledge.
        """
        counts = obs.entity_firing_counts
        total = sum(counts.values())
        if total == 0:
            return True
        max_fraction = max(counts.values()) / total
        return max_fraction < self.config.ENTITY_SPECIFICITY_THRESHOLD

    def _promote(self, obs_id: str) -> None:
        """Promote to StructuralNode. Only centroid survives — no content."""
        obs = self._observation_nodes.pop(obs_id)
        structural = StructuralNode(
            node_id=obs.node_id,
            pattern_embedding=list(obs.pattern_embedding),  # Centroid only
            session_count=obs.sessions_active,
            diversity_score=obs.diversity_score,
            is_generalised=True,
        )
        self.library.add(structural)

    # ── Stage 3: Reactivation ─────────────────────────────────────────────────
    def check_for_reactivation(self, content_embedding: list[float]) -> list[str]:
        """
        Content in the anomaly zone (LOW < sim < HIGH) triggers reactivation.
        The anomaly zone: content is close enough to partially match but not
        close enough to fire confidently. This is the eclipse region.

        Returns list of reactivated structural node IDs.
        """
        if not content_embedding:
            return []
        qv = np.array(content_embedding)
        qn = np.linalg.norm(qv)
        if qn < 1e-10:
            return []
        qv = qv / qn

        reactivated = []
        low = self.config.REACTIVATION_LOW_THRESHOLD
        high = self.config.REACTIVATION_HIGH_THRESHOLD

        for node in self.library.nodes.values():
            if node.pattern_embedding is None:
                continue
            ev = np.array(node.pattern_embedding)
            en = np.linalg.norm(ev)
            if en < 1e-10:
                continue
            sim = float(np.dot(qv, ev / en))
            if low < sim < high:  # Anomaly zone
                node.reactivate()
                reactivated.append(node.node_id)

        return reactivated

    def refine_reactivated(self, node_id: str,
                           content_embedding: list[float]) -> bool:
        """
        Refine a reactivated structural node using the anomalous content.
        Adjusts centroid boundary with small step (REFINEMENT_ALPHA).
        Content is NOT stored — only the centroid moves.
        Returns node to infrastructure after refinement.
        Returns True if refinement applied.
        """
        node = self.library.get(node_id)
        if node is None or node.attention_mode != "active":
            return False
        if node.pattern_embedding is None:
            return False

        alpha = self.config.REFINEMENT_ALPHA
        node.pattern_embedding = [
            node.pattern_embedding[i] * (1 - alpha) + content_embedding[i] * alpha
            for i in range(len(node.pattern_embedding))
        ]
        # Re-normalise centroid
        ev = np.array(node.pattern_embedding)
        en = np.linalg.norm(ev)
        if en > 1e-10:
            node.pattern_embedding = (ev / en).tolist()

        node.return_to_infrastructure()
        return True

    # ── Post-session processing ───────────────────────────────────────────────
    def post_session(self, session_degree_dist: np.ndarray,
                     entity_id: str = "unknown") -> dict:
        """
        Called after session end (before operational layer reset).
        Records session degree distribution for each active observation node.
        Evaluates all observation nodes for stabilisation.
        Returns summary of promotions and active nodes.
        """
        promoted = []
        for obs_id in list(self._observation_nodes.keys()):
            obs = self._observation_nodes.get(obs_id)
            if obs is None:
                continue
            obs.sessions_active += 1
            # Record this session's degree distribution for diversity scoring
            self.library.record_session_distribution(obs_id, session_degree_dist)
            self.library.update_entity_counts(obs_id, obs.entity_firing_counts)
            dists = self.library.get_session_distributions(obs_id)
            if self.evaluate_for_stabilisation(obs_id, dists):
                promoted.append(obs_id)

        # Prune observation nodes that have never stabilised after MAX sessions
        stale = []
        for obs_id, obs in list(self._observation_nodes.items()):
            if obs.sessions_active >= self.config.MAX_OBSERVATION_SESSIONS:
                stale.append(obs_id)
                del self._observation_nodes[obs_id]

        return {
            "observation_nodes_evaluated": len(self._observation_nodes) + len(promoted),
            "nodes_promoted": len(promoted),
            "nodes_still_in_observation": len(self._observation_nodes),
            "stale_nodes_pruned": len(stale),
        }

    @property
    def observation_count(self) -> int:
        return len(self._observation_nodes)
