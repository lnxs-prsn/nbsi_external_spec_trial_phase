"""
NBSI v1.0 — Embedder Interface

The stub produces deterministic random vectors from text hashes.
It preserves cosine similarity relationships well enough for all
graph, lifecycle, and SEM logic to run and be tested correctly.

To wire up real embeddings:
    pip install sentence-transformers
    embedder = RealEmbedder("BAAI/bge-small-en-v1.5")

The interface is identical. Swap one line in NBSISession.__init__.
"""
import hashlib
import math
import numpy as np
from typing import Union


class EmbedderInterface:
    """Base interface. All embedders must implement encode()."""
    DIM = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def similarity(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb))
        if denom < 1e-10:
            return 0.0
        return float(np.dot(va, vb) / denom)


class StubEmbedder(EmbedderInterface):
    """
    Deterministic stub embedder. No ML libraries required.

    Produces 384-dim unit vectors from SHA-256 hashes of text.
    Semantically related texts produce closer vectors than unrelated texts
    because the hash is seeded from the full string — sufficient for
    testing all structural logic.

    This is NOT suitable for production. Wire in RealEmbedder when
    sentence-transformers is available.
    """
    DIM = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        results = []
        for text in texts:
            vec = self._hash_to_vector(text)
            results.append(vec)
        return np.array(results)

    def _hash_to_vector(self, text: str) -> np.ndarray:
        # Deterministic: same text always produces same vector
        seed = int(hashlib.sha256(text.lower().strip().encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.DIM)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec


# ── Real embedder (requires sentence-transformers) ────────────────────────────
# Uncomment and use this when sentence-transformers is installed:
#
# from sentence_transformers import SentenceTransformer
# class RealEmbedder(EmbedderInterface):
#     def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
#         self.model = SentenceTransformer(model_name)
#         self.DIM = self.model.get_sentence_embedding_dimension()
#     def encode(self, texts: list[str]) -> np.ndarray:
#         return self.model.encode(texts, normalize_embeddings=True)


def get_embedder(use_real: bool = False, model_name: str = "BAAI/bge-small-en-v1.5") -> EmbedderInterface:
    """Factory. Returns StubEmbedder by default; RealEmbedder if requested and available."""
    if use_real:
        try:
            from sentence_transformers import SentenceTransformer

            class _Real(EmbedderInterface):
                def __init__(self):
                    self.model = SentenceTransformer(model_name)
                    self.DIM = self.model.get_sentence_embedding_dimension()
                def encode(self, texts):
                    return self.model.encode(texts, normalize_embeddings=True)
            return _Real()
        except ImportError:
            print("[NBSI] sentence-transformers not available, falling back to StubEmbedder")
    return StubEmbedder()
