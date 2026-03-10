# NBSI — Node-Based Semantic Intelligence
### Implementation Repository

A session-based graph reasoning engine with structural persistence.  
Runs entirely on CPU. No GPU. No API calls. No content stored between sessions.

**Papers and theory:** https://github.com/lnxs-prsn/nbsi_theory_and_plan  
**Live overview:** https://huggingface.co/spaces/Node-Based-S-I/the_theory

---

## Repository structure

This repository contains two independent but related projects that evolved
sequentially. Each is a complete, self-contained Python package.

```
nbsi_external_spec_trial_phase/
│
├── nbsi/                        # Project 1 — Core engine
│   ├── run_tests.py             #   33 tests, stdlib only, no ML deps
│   ├── config.py                #   All parameters
│   ├── embedder.py              #   StubEmbedder (tests) + RealEmbedder (MiniLM)
│   ├── graph/                   #   ConceptGraph, ConceptNode, ConceptEdge
│   ├── sem/                     #   SEMPropagator, SpeculativeNode
│   ├── lifecycle/               #   LifecycleEngine, StructuralNodeLibrary
│   ├── reasoning/               #   ConductivityEngine, BeamSearch, QueryEngine
│   ├── session/                 #   NBSISession — orchestrates both layers
│   └── tests/                   #   Full test suite
│
├── nbsi_v1_lightweight/         # Project 2 — Lightweight stack
│   └── nbsi/
│       ├── demo.py              #   Start here — run against any document
│       ├── ingestion/           #   spaCy extraction pipeline
│       │   └── spacy_extractor.py
│       ├── synthesis/           #   llama-cpp synthesis (Phase 3 — complete)
│       ├── os_integration/      #   File watcher (Phase 5 — not yet built)
│       └── ...                  #   All core engine files (updated versions)
│
├── zfixed1/                     # Patched files from Phase 2 bug fixes
├── zfixed2/                     # Patched files from Phase 2 bug fixes
├── zphases_1_to_7/              # Build progress documentation
└── zqueries/                    # Test queries used during verification
```

---

## Which project should I use?

| I want to... | Use |
|---|---|
| Verify the core engine works | `nbsi/` — run `run_tests.py` |
| Run reasoning on a real document | `nbsi_v1_lightweight/nbsi/` — run `demo.py` |
| Build on the architecture | Start with `nbsi_v1_lightweight/` — it has all fixes applied |
| Understand the internals | Start with `nbsi/` — cleaner, fully tested, no ML deps |

**Important:** `nbsi_v1_lightweight/` contains updated versions of the core
engine files with three bug fixes applied that are not yet in `nbsi/`.
If you are building on this project, use the lightweight directory as your base.

---

## Project 1 — Core engine (`nbsi/`)

The pure engine. No spaCy. No sentence-transformers. Just the graph, the
lifecycle, the SEM operator, and beam search. Three dependencies only.

### Setup

```bash
cd nbsi

uv venv --python 3.12
source .venv/bin/activate

uv pip install networkx numpy scipy
```

### Verify

```bash
PYTHONPATH=. python run_tests.py
```

Expected output:

```
======================================================================
  NBSI v1.0 — Test Suite
======================================================================
  ✓  Graph: add_node returns id and increments count
  ... (33 tests)
======================================================================
  Results: 33 passed, 0 failed, 33 total
======================================================================
```

### Note on Python version

Python 3.14 is not supported — spaCy (used in Project 2) does not yet
support it. Use Python 3.11 or 3.12.

```bash
uv python install 3.12
uv venv --python 3.12
```

---

## Project 2 — Lightweight stack (`nbsi_v1_lightweight/nbsi/`)

The full CPU stack. spaCy extracts nodes and edges from documents. MiniLM
provides real semantic embeddings. The core engine reasons over the graph.
No GPU. No API calls.

### Setup

```bash
cd nbsi_v1_lightweight/nbsi

uv venv --python 3.12
source .venv/bin/activate

uv pip install spacy sentence-transformers networkx numpy scipy
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```
```bash
# Phase 3 — synthesis (optional but recommended)
sudo dnf install -y gcc gcc-c++ cmake make   # Fedora
# sudo apt install -y build-essential cmake  # Ubuntu/Debian
uv pip install llama-cpp-python
mkdir -p ~/nbsi-models
wget -P ~/nbsi-models https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```
### Run the demo

```bash
# Built-in sample text — no file needed
PYTHONPATH=. python demo.py

# Your own text, PDF, or DOCX file
PYTHONPATH=. python demo.py yourfile.txt

# Your own file with your own query
PYTHONPATH=. python demo.py yourfile.txt "your question here"

# With synthesis (default — requires model downloaded above)
PYTHONPATH=. python demo.py yourfile.txt "your question"

# Without synthesis (faster, no model needed)
PYTHONPATH=. python demo.py yourfile.txt "your question" --no-synth

```

### Verify core tests still pass

```bash
PYTHONPATH=. python run_tests.py   # 33/33
```

### Supported document formats

| Format | Extension | Extra dependency |
|---|---|---|
| Plain text | .txt .md .py | None |
| PDF | .pdf | `uv pip install pdfminer.six` |
| Word document | .docx | `uv pip install python-docx` |

---

## Verified results

Tested against the NBSI External Architecture Paper (19724 chars, 838 nodes extracted):

| Query | Paths | Top conductivity |
|---|---|---|
| what is the structural layer | 5 | 0.442 |
| how does speculation change the graph | 3 | 0.490 |
| what happens at session end | 5 | 0.595 |
| should the session graph persist | 5 | 0.490 |

**SEM verified:**
- 46 edges affected on semantically relevant speculation
- Conductivity change: -0.058 confirmed
- Direction correct: graph resisted speculation contradicting source material
- Rollback tolerance: 0.00e+00

---

## Bug fixes applied in `nbsi_v1_lightweight/`

Three bugs were found and fixed during real-document testing in Phase 2.
These fixes are in `nbsi_v1_lightweight/` but not yet backported to `nbsi/`.

| File | Fix |
|---|---|
| `sem/propagator.py` | Speculative node now anchored to similar existing nodes before BFS — was an island, propagation never fired |
| `graph/concept_graph.py` | `add_edge` guards both endpoints against `_nodes` — phantom edges caused zero hop scores and 0 paths found |
| `session/session.py` | `ingest_graph` remaps merged node IDs in edges — edges were referencing discarded IDs after label merging |

The fixed files are also available individually in `zfixed1/` and `zfixed2/`.

---

## Build phases

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Complete | Core engine — 33/33 tests passing |
| 2 | ✅ Complete | Real embeddings + spaCy extraction + bug fixes |
| 3 | ✅ Complete | Synthesis — llama-cpp local narration (Qwen2.5-1.5B Q4) |
| 4 | 🔲 Pending | Document reader expansion |
| 5 | 🔲 Pending | OS file watcher |
| 6 | 🔲 Pending | Persistence across sessions |
| 7 | 🔲 Pending | Full orchestration via main.py |

---
Each completed phase has its own branch in this repository.
Checkout a specific phase to see the project at that exact stage:

```bash
git checkout main      # Phases 1 and 2 — stable base
git checkout phase_3   # Phase 3 — synthesis added
```
---


## Architecture overview

```
Query
  ↓
QueryEngine — finds anchor nodes by semantic similarity
  ↓
BeamSearch — explores graph from anchors using conductivity scores
  ↓
ConductivityEngine — geometric mean path scoring
  ↓
Reasoning paths returned — ranked by conductivity

Speculation (optional, any time during session)
  ↓
SEMPropagator — anchors spec node, BFS through graph
  ↓
Edge effective_weights shift — base_weights never touched
  ↓
Same query returns different paths — graph landscape changed
  ↓
Rollback — zero trace, tolerance < 1e-9
```

**Two layers — never cross:**

```
Operational layer          Structural layer
(session-scoped)           (permanent)
─────────────────          ────────────────
ConceptGraph               StructuralNodeLibrary
SEMPropagator              StructuralNodes (centroids only)
QueryEngine                No content — geometry only
Destroyed at session end   Persists forever
```

---

## Licence

Code: GNU General Public License v3.0  
Papers: Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Links

- **Theory and papers:** https://github.com/lnxs-prsn/nbsi_theory_and_plan
- **Hugging Face:** https://huggingface.co/spaces/Node-Based-S-I/the_theory
