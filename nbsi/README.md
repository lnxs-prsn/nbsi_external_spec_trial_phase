# NBSI v1.0 — Node-Based Semantic Intelligence (External Architecture)

A session-based graph reasoning engine with structural persistence and lifecycle-aware node management.

## What runs right now

```bash
python3 run_tests.py        # 33 tests, all pass, no ML dependencies
```

## What requires dependencies

```bash
pip install -r requirements.txt
# Then use RealEmbedder and LLM clients in NBSISession
```

## Project structure

```
nbsi/
  config.py                  # All parameters with defaults
  embedder.py                # StubEmbedder (works now) + RealEmbedder (uncomment)
  graph/
    node.py                  # ConceptNode
    edge.py                  # ConceptEdge — base_weight / effective_weight split
    concept_graph.py         # ConceptGraph backed by NetworkX
  sem/
    speculative_node.py      # SpeculativeNode
    propagator.py            # SEMPropagator — speculation-as-impurity operator
  lifecycle/
    nodes.py                 # ObservationNode, StructuralNode
    structural_library.py    # StructuralNodeLibrary — the persistent structural layer
    lifecycle_engine.py      # LifecycleEngine — observe / stabilise / reactivate
  reasoning/
    conductivity.py          # ConductivityEngine (geometric mean), QueryEngine, BeamSearch
    ensemble.py              # Stability measurement, dual-graph intersection
  session/
    session.py               # NBSISession — full two-layer orchestration
  run_tests.py               # 33 tests, pure stdlib runner
```

## Architecture in one paragraph

Two layers. The **structural layer** (StructuralNodeLibrary) persists across all sessions and contains only refined processing patterns — centroids with no content. The **operational layer** (ConceptGraph + SEMPropagator) is built fresh each session and destroyed completely at session end. Nothing content-bearing crosses from operational to structural. The servant cannot develop interests because it retains no entity-specific data.

## The node lifecycle

1. **Observation** — novel content creates an ObservationNode. Expensive. Provisional.
2. **Stabilisation** — node confirmed across diverse sessions is promoted to StructuralNode. Infrastructure: fires silently, near-zero cost.
3. **Reactivation** — anomalous content (eclipse) pulls a structural node back to active attention for refinement. Returns to infrastructure after.

## Wiring up real embeddings

In `embedder.py`, uncomment `RealEmbedder` and change one line in `NBSISession.__init__`:

```python
# Before:
self.embedder = StubEmbedder()

# After:
from nbsi.embedder import get_embedder
self.embedder = get_embedder(use_real=True, model_name="BAAI/bge-small-en-v1.5")
```

## Key design decisions

- **base_weight / effective_weight separation**: SEM never touches base_weight. Rollback is always clean. Verified tolerance < 1e-9.
- **Label-based identity**: Ensemble mode uses node labels not UUIDs. Two graphs built from the same source assign different UUIDs to the same concept.
- **Generalisation constraint**: A structural node that fires >60% for one entity is not promoted. This is what makes the obedience property hold.
- **Protected nodes**: Anchor-adjacent nodes are immune to betweenness pruning. The query cannot destroy the nodes it depends on.
- **FAISS upgrade path**: `StructuralNodeLibrary.match()` and `ConceptGraph.find_similar_nodes()` use linear search. Replace their internals with `faiss-cpu` IndexFlatIP for O(log N) matching at scale.
