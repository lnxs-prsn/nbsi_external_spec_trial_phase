# NBSI v1.0 Lightweight
## Node-Based Semantic Intelligence — External Architecture

A session-based graph reasoning engine with structural persistence.
No GPU required. No API calls required. Runs on a Raspberry Pi.

---

## Quickstart — Core only (no ML deps)

```bash
pip install networkx numpy scipy
PYTHONPATH=. python3 run_tests.py   # 33/33 tests pass
```

## Quickstart — Lightweight demo (real embeddings, real paths)

```bash
pip install spacy sentence-transformers networkx numpy scipy
python -m spacy download en_core_web_sm

# Run demo with built-in sample text
PYTHONPATH=. python3 demo.py

# Run demo with your own file
PYTHONPATH=. python3 demo.py yourfile.txt

# Run demo with your own file and query
PYTHONPATH=. python3 demo.py yourfile.txt "your question here"
```

## With uv (faster)

```bash
uv venv && source .venv/bin/activate
uv pip install spacy sentence-transformers networkx numpy scipy
python -m spacy download en_core_web_sm
PYTHONPATH=. python3 demo.py
```

---

## What demo.py does

1. Loads MiniLM-L6-v2 embedder (80MB, CPU only)
2. Extracts nodes and edges from your text using spaCy
3. Builds a session graph with real semantic embeddings
4. Queries the graph — returns reasoning paths with conductivity scores
5. Demonstrates SEM speculation — rewires the graph, shows the change
6. Rolls back the speculation — verifies zero trace (tolerance < 1e-9)
7. Ends the session — destroys the graph, structural library persists

---

## Project structure

```
nbsi/
  config.py                    All parameters
  embedder.py                  StubEmbedder (tests) + RealEmbedder (demo)
  demo.py                      Runnable demo — start here
  run_tests.py                 33 tests, pure stdlib runner

  ingestion/
    spacy_extractor.py         spaCy NLP → ConceptNodes + ConceptEdges

  graph/
    node.py                    ConceptNode
    edge.py                    ConceptEdge (base/effective weight split)
    concept_graph.py           ConceptGraph (NetworkX)

  sem/
    speculative_node.py        SpeculativeNode
    propagator.py              SEMPropagator (speculation-as-impurity)

  lifecycle/
    nodes.py                   ObservationNode, StructuralNode
    structural_library.py      StructuralNodeLibrary (persistent layer)
    lifecycle_engine.py        LifecycleEngine (observe/stabilise/reactivate)

  reasoning/
    conductivity.py            ConductivityEngine, BeamSearch, QueryEngine
    ensemble.py                Stability measurement, dual-graph intersection

  session/
    session.py                 NBSISession (orchestrates both layers)
```

---

## Next step — synthesis (narration)

Add llama-cpp to get natural language answers from the paths:

```bash
pip install llama-cpp-python
# Download a model (e.g. Qwen2.5 1.5B Q4 ~900MB)
# See NBSI_Lightweight_Build_Plan.docx Section 4
```

No API calls. Runs locally. Works on a Raspberry Pi 5.
