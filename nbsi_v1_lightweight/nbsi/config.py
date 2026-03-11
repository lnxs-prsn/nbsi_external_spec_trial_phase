"""
NBSI v1.0 External Architecture — Configuration
All parameters in one place. Defaults from the engineering spec.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ── Ingestion ────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Graph ────────────────────────────────────────────────────────────────
    MAX_NODES: int = 200
    SEMANTIC_EDGE_THRESHOLD: float = 0.75   # Cosine sim for auto semantic edges
    BETWEENNESS_PRUNE_KEEP: float = 0.6     # Keep top 60% by betweenness

    # ── Traversal ────────────────────────────────────────────────────────────
    MAX_PATH_DEPTH: int = 5
    BEAM_WIDTH: int = 10
    TOP_K_PATHS: int = 5

    # ── SEM ──────────────────────────────────────────────────────────────────
    SEM_ALPHA: float = 0.4           # Max fractional weight change
    SEM_DECAY_RATE: float = 0.5      # Exponential decay with graph distance
    SEM_RADIUS: int = 2              # Max hop radius
    SPEC_INITIAL_ACTIVATION: float = 0.3
    SPEC_CONFIRMED_ACTIVATION: float = 0.7

    # ── Ensemble / Stability ─────────────────────────────────────────────────
    STABILITY_THRESHOLD: float = 0.80
    STABILITY_SAMPLE_QUERIES: int = 5

    # ── Lifecycle ────────────────────────────────────────────────────────────
    OBSERVATION_MERGE_THRESHOLD: float = 0.88
    MIN_CONFIRMATIONS: int = 5
    STABILISATION_THRESHOLD: float = 0.40
    REACTIVATION_LOW_THRESHOLD: float = 0.45
    REACTIVATION_HIGH_THRESHOLD: float = 0.75
    REFINEMENT_ALPHA: float = 0.05
    MAX_OBSERVATION_SESSIONS: int = 20      # Prune if never stabilises
    ENTITY_SPECIFICITY_THRESHOLD: float = 1.01  # change back to 0.60
    DIVERSITY_THRESHOLD: float = 0.0           # change back to 0.30


# Singleton default config
default_config = Config()
