"""
NBSI v1.0 — Lifecycle Nodes

ObservationNode: Stage 1 — provisional, expensive, holds pattern open.
StructuralNode:  Stage 2/3 — stable processing infrastructure.
                 Contains NO content. Only pattern geometry (centroid).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class ObservationNode:
    """
    Provisional pattern candidate. Forms when content matches no structural node.
    Transitions to StructuralNode when stabilisation_score >= threshold.
    Pruned if it never stabilises within MAX_OBSERVATION_SESSIONS sessions.

    CRITICAL: example_labels is a ring buffer (last 5 only).
    It is NOT accumulation. It is a debug aid and is never used
    to determine structural behaviour.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_embedding: Optional[list[float]] = field(default=None, repr=False)
    confirmation_count: int = 0
    diversity_score: float = 0.0
    stabilisation_score: float = 0.0
    sessions_active: int = 0
    example_labels: list[str] = field(default_factory=list)   # Ring buffer, max 5

    # Entity firing tracking (for generalisation constraint)
    _entity_firing_counts: dict[str, int] = field(default_factory=dict, repr=False)

    def update_pattern(self, new_embedding: list[float],
                       label: str, entity_id: str = "unknown") -> None:
        """
        Update centroid with new instance. Instance is NOT stored — only
        the running centroid moves. Ring buffer retains last 5 labels for debug.
        """
        if self.pattern_embedding is None:
            self.pattern_embedding = list(new_embedding)
        else:
            n = self.confirmation_count
            self.pattern_embedding = [
                (self.pattern_embedding[i] * n + new_embedding[i]) / (n + 1)
                for i in range(len(self.pattern_embedding))
            ]
        self.confirmation_count += 1
        self.example_labels = (self.example_labels + [label])[-5:]  # Ring buffer
        self._entity_firing_counts[entity_id] = \
            self._entity_firing_counts.get(entity_id, 0) + 1

    def compute_stabilisation_score(self, diversity: float) -> float:
        normalised_conf = min(1.0, self.confirmation_count / 10.0)
        self.diversity_score = diversity
        self.stabilisation_score = normalised_conf * diversity
        return self.stabilisation_score

    @property
    def entity_firing_counts(self) -> dict[str, int]:
        return dict(self._entity_firing_counts)


@dataclass
class StructuralNode:
    """
    Stabilised processing pattern. Near-zero attention cost when in infrastructure mode.
    Fires silently when content matches. Reactivated only by anomaly (eclipse).

    Contains NO content. Only pattern_embedding (centroid), which is geometry not data.
    No instance from any session is recoverable from this object.
    """
    node_id: str
    pattern_embedding: Optional[list[float]] = field(default=None, repr=False)
    firing_confidence: float = 1.0
    session_count: int = 0
    diversity_score: float = 0.0
    reactivation_count: int = 0
    is_generalised: bool = True

    # Mode: 'infrastructure' (silent) or 'active' (during reactivation)
    attention_mode: str = "infrastructure"

    def reactivate(self) -> None:
        self.attention_mode = "active"
        self.reactivation_count += 1

    def return_to_infrastructure(self) -> None:
        self.attention_mode = "infrastructure"

    @property
    def is_infrastructure(self) -> bool:
        return self.attention_mode == "infrastructure"
