"""
NBSI v1.0 — SpeculativeNode

A provisional node added mid-session via speculation.
Lower initial activation than source nodes.
Triggers SEM propagation on creation.
Resolved by confirmation (writes to base_weight) or rollback (no trace).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class SpeculativeNode:
    statement: str
    spec_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[list[float]] = field(default=None, repr=False)
    activation: float = 0.3          # Below source node range (0.6–1.0)
    confirmed: bool = False
    disconfirmed: bool = False
    affected_edges: list[tuple] = field(default_factory=list)  # (src_id, tgt_id) pairs

    @property
    def is_resolved(self) -> bool:
        return self.confirmed or self.disconfirmed

    @property
    def status(self) -> str:
        if self.confirmed:
            return "confirmed"
        if self.disconfirmed:
            return "disconfirmed"
        return "active"
