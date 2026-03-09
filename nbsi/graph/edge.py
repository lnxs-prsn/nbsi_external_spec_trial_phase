"""
NBSI v1.0 — ConceptEdge

The base_weight / effective_weight separation is the most important
implementation decision in the session graph.

base_weight    : Source-derived. Set at ingestion. Never changed by SEM.
                 Only changes on speculation confirmation (sem_confirm).
effective_weight: What conductivity uses. Equals base_weight when no
                  speculation is active. SEM modifies this, not base_weight.
                  Rollback restores effective_weight = base_weight without trace.

sem_deltas     : Dict[spec_node_id -> delta_value]. Summed to compute
                 effective_weight. Cleared on rollback.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


EdgeType = Literal[
    "causal",
    "constitutive",
    "contrastive",
    "sequential",
    "analogical",
    "evidential",
    "semantic",      # Auto-created from cosine similarity
    "speculative",   # Edges from SpeculativeNode
]

# Baseline weight by edge type — reflects epistemic strength
EDGE_TYPE_WEIGHTS: dict[str, float] = {
    "causal":       0.85,
    "constitutive": 0.80,
    "evidential":   0.80,
    "sequential":   0.70,
    "contrastive":  0.65,
    "analogical":   0.60,
    "semantic":     0.55,
    "speculative":  0.40,
}


@dataclass
class ConceptEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType = "semantic"
    base_weight: float = 0.6
    agreement: float = 1.0       # Cross-LLM agreement on this edge's existence

    # SEM state — never touches base_weight
    _sem_deltas: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Initialise base_weight from type if not explicitly set
        if self.base_weight == 0.6 and self.edge_type in EDGE_TYPE_WEIGHTS:
            self.base_weight = EDGE_TYPE_WEIGHTS[self.edge_type]

    @property
    def effective_weight(self) -> float:
        """What conductivity uses. SEM-modulated but never < 0."""
        delta_sum = sum(self._sem_deltas.values())
        return max(0.0, self.base_weight * (1.0 + delta_sum))

    def apply_sem_delta(self, spec_node_id: str, delta: float) -> None:
        """Add or update a SEM delta for a speculative node. Does NOT touch base_weight."""
        self._sem_deltas[spec_node_id] = delta

    def remove_sem_delta(self, spec_node_id: str) -> None:
        """Remove delta on rollback. Restores effective_weight toward base_weight."""
        self._sem_deltas.pop(spec_node_id, None)

    def confirm_sem(self, spec_node_id: str) -> None:
        """
        Write the SEM delta permanently into base_weight.
        Called on speculation confirmation. Removes the delta entry.
        """
        delta = self._sem_deltas.pop(spec_node_id, 0.0)
        self.base_weight = max(0.0, self.base_weight * (1.0 + delta))

    def clear_all_sem(self) -> None:
        """Full rollback — all deltas removed."""
        self._sem_deltas.clear()

    @property
    def has_active_sem(self) -> bool:
        return len(self._sem_deltas) > 0
