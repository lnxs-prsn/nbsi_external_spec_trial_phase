"""
NBSI v1.0 — ConceptNode

A node in the session graph. Content-derived, session-scoped.
Destroyed at session end. Never enters the structural library.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class ConceptNode:
    """
    A concept extracted from source text for the current session.

    label       : Canonical lowercased name. Used for label-based node identity
                  in ensemble mode (not UUID-based — two graphs assign different
                  UUIDs to the same concept).
    embedding   : 384-dim unit vector from embedder. None until set.
    activation  : [0, 1]. Set at ingestion; boosted when structural nodes fire.
    protected   : True for anchor-adjacent nodes — immune to betweenness pruning.
    node_type   : Extraction hint. Does not affect graph logic.
    agreement   : Cross-LLM agreement score [0, 1]. Set during multi-LLM extraction.
    recency     : [0, 1]. 1.0 at ingestion, decays if recency decay is enabled.
    """
    label: str
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[list[float]] = field(default=None, repr=False)
    activation: float = 0.7
    protected: bool = False
    node_type: str = "concept"          # concept | entity | event | relation
    agreement: float = 1.0
    recency: float = 1.0

    def __post_init__(self):
        self.label = self.label.lower().strip()

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if isinstance(other, ConceptNode):
            return self.node_id == other.node_id
        return False
