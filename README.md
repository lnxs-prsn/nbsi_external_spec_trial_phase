# NBSI — Node-Based Semantic Intelligence
### Implementation Repository

A session-based graph reasoning engine with structural persistence.  
Runs entirely on CPU. No GPU. No API calls. No content stored between sessions.

**Papers and theory:** https://github.com/lnxs-prsn/nbsi_theory_and_plan  
**Live overview:** https://huggingface.co/spaces/Node-Based-S-I/the_theory

---

## Repository structure

```
nbsi_external_spec_trial_phase/
│
├── nbsi/                        # Project 1 — Core engine
│   ├── run_tests.py             #   33 tests, stdlib only, no ML deps
│   ├── config.py
│   ├── embedder.py
│   ├── graph/
│   ├── sem/
│   ├── lifecycle/
│   ├── reasoning/
│   ├── session/
│   └── tests/
│
├── nbsi_v1_lightweight/         # Project 2 — Full stack (start here)
│   └── nbsi/
│       ├── demo.py              #   Entry point — run against any document
│       ├── ingestion/
│       │   ├── spacy_extractor.py   # spaCy + MiniLM concept extraction
│       │   ├── chunker.py           # Splits documents into focused sections
│       │   ├── document_reader.py   # Structure-aware reading (PDF/DOCX/HTML/MD)
│       │   ├── metadata_nodes.py    # Injects title/heading anchors into graph
│       │   └── pipeline.py          # Multi-document ingestion API
│       ├── synthesis/
│       │   └── synthesiser.py       # llama-cpp local LLM narration
│       ├── graph/
│       ├── sem/
│       ├── lifecycle/
│       ├── reasoning/
│       └── session/
│
├── zcumulative_changes
    ├── zfixed1/                     # Phase 2 bug fix patches
    ├── zfixed1/                     # Phase 2 bug fix patches
    ├── zfixed2/                     # Phase 2 bug fix patches
    ├── zphases_1_to_7/              # Build progress notes
    └── zqueries/                    # Test queries used during verification
```

---

## Which project should I use?

| I want to... | Use |
|---|---|
| Verify the core engine works | `nbsi/` — run `run_tests.py` |
| Run reasoning on a real document | `nbsi_v1_lightweight/nbsi/` — run `demo.py` |
| Build on the architecture | `nbsi_v1_lightweight/` — has all fixes and Phase 4 ingestion |
| Understand the internals | `nbsi/` — clean, fully tested, no ML deps |

**Important:** `nbsi_v1_lightweight/` contains updated versions of all core
engine files with three bug fixes applied that are not yet backported to `nbsi/`.
Use the lightweight directory as your base for any new work.

---

## Project 1 — Core engine (`nbsi/`)

The pure engine. No spaCy. No sentence-transformers. Just the graph, the
lifecycle, the SEM operator, and beam search.

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
# Expected: 33 passed, 0 failed
```

---

## Project 2 — Full stack (`nbsi_v1_lightweight/nbsi/`)

The complete CPU stack. Reads structured documents, splits them into focused
chunks, extracts concept graphs with spaCy and MiniLM, and reasons over them
with beam search. Optional local LLM narration via llama-cpp.

### Setup

```bash
cd nbsi_v1_lightweight/nbsi

uv venv --python 3.12
source .venv/bin/activate

uv pip install spacy sentence-transformers networkx numpy scipy
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Optional — for PDF support
uv pip install pdfminer.six

# Optional — for DOCX support
uv pip install python-docx

# Optional — for synthesis (local LLM narration)
uv pip install llama-cpp-python
```

### Run the demo

Run from inside `nbsi_v1_lightweight/nbsi/`:

```bash
# Built-in sample text — no file needed
PYTHONPATH=.. python demo.py

# Your own file with default query
PYTHONPATH=.. python demo.py yourfile.txt

# Your own file with your own query
PYTHONPATH=.. python demo.py yourfile.txt "your question here"

# Skip LLM narration (faster)
PYTHONPATH=.. python demo.py yourfile.txt "your question" --no-synth
```

### Supported document formats

| Format | Extensions | Dependency |
|---|---|---|
| Plain text | .txt .md .py .js .ts .rst | None |
| HTML | .html .htm | None |
| PDF | .pdf | `pdfminer.six` |
| Word document | .docx | `python-docx` |

### LLM synthesis (optional)

The synthesiser narrates reasoning paths in plain language using a local
quantised model. No internet connection required after download.

```bash
mkdir -p ~/nbsi-models

# Recommended — Qwen2.5 1.5B (laptop, ~900MB)
wget -P ~/nbsi-models https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Alternative — Phi-3 Mini 3.8B (Raspberry Pi 5, ~2.4GB)
wget -P ~/nbsi-models https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

To use a different model path:

```bash
NBSI_MODEL=~/nbsi-models/your-model.gguf PYTHONPATH=.. python demo.py yourfile.txt "query"
```

### Run all tests

```bash
# Core engine tests (no ML deps required)
PYTHONPATH=.. python run_tests.py
# Expected: 33 passed, 0 failed

# Phase 4 ingestion tests
PYTHONPATH=.. python -m unittest nbsi.tests.test_phase4_real -v
# Expected: 40 passed, 0 failed
```

---

## How ingestion works (Phase 4)

When you pass a file to `demo.py`, it goes through four stages before
the graph is queried:

```
File
  ↓
document_reader.py — reads structure (headings, sections, metadata)
  ↓
chunker.py — splits each section into ~3000 char focused chunks
  ↓
metadata_nodes.py — injects title and heading nodes as protected anchors
  ↓
spacy_extractor.py — extracts concept nodes and edges per chunk
  ↓
session.ingest_graph() — unified graph across all chunks
  ↓
Query
```

The NBSI External Paper (19,724 chars) produces 38 chunks and 611 nodes
rather than the 838 nodes from flat single-pass extraction. More importantly,
the graph has structure: section heading nodes act as anchors, so queries
like "what does section 3 cover" find direct paths rather than drifting
through generic concept chains.

### Multi-document ingestion

To ingest a folder of documents into one unified session graph:

```python
from nbsi.embedder import RealEmbedder
from nbsi.ingestion.spacy_extractor import SpacyExtractor
from nbsi.ingestion.pipeline import ingest_documents, ingest_folder
from nbsi.session.session import NBSISession
from nbsi.lifecycle.structural_library import StructuralNodeLibrary
from nbsi.config import Config

session   = NBSISession(StructuralNodeLibrary(), RealEmbedder(), Config())
extractor = SpacyExtractor()

# Ingest a list of files
report = ingest_documents(session, extractor, ['paper.pdf', 'notes.md'])
print(report.summary())

# Or ingest an entire folder
report = ingest_folder(session, extractor, '~/Documents/research')
print(report.summary())

# Query across all ingested documents
paths = session.query("your question here")
```

---

## Verified results

Tested against the NBSI External Architecture Paper with the updated pipeline:

| Query | Paths | Top conductivity |
|---|---|---|
| what happens at session end | 5 | 0.720 |
| what is the structural layer | 5 | — |
| how does speculation change the graph | 5 | — |

**Before Phase 4 (flat extraction):**
- 838 nodes extracted → top path conductivity 0.595
- Paths drifted through generic concept chains
- Speculation effect: +0.000 (graph too sparse to propagate)

**After Phase 4 (pipeline):**
- 611 nodes across 38 chunks → top path conductivity 0.720
- Paths land directly on document section headings
- Speculation effect: +0.116 (graph dense enough for SEM propagation)

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
  ↓
Synthesiser (optional) — local LLM narrates paths in plain language

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

## Bug fixes applied in `nbsi_v1_lightweight/`

Three bugs were found during real-document testing in Phase 2.
Fixed in `nbsi_v1_lightweight/`, not yet backported to `nbsi/`.

| File | Fix |
|---|---|
| `sem/propagator.py` | Speculative node anchored to top-5 similar existing nodes — was an island, propagation never fired |
| `graph/concept_graph.py` | `add_edge` guards both endpoints — phantom edges caused zero hop scores and 0 paths found |
| `session/session.py` | `ingest_graph` remaps merged node IDs in edges — edges referenced discarded IDs after label merging |

---

## Build phases

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Complete | Core engine — 33/33 tests |
| 2 | ✅ Complete | Real embeddings + spaCy extraction + bug fixes |
| 3 | ✅ Complete | Synthesis — llama-cpp, Qwen2.5-1.5B + Phi-3 Mini verified |
| 4 | ✅ Complete | Document ingestion — chunker, structure-aware readers, metadata nodes, multi-document pipeline — 40/40 tests |
| 5 | ✅ Built, pending integration | OS integration — file reader, directory watcher, ingestion worker — 21/21 tests |
| 6 | 🔲 Pending | Persistence across sessions |
| 7 | 🔲 Pending | Full orchestration via main.py |

---

Each completed phase has its own branch in this repository.
Checkout a specific phase to see the project at that exact stage:

```bash
git checkout main      # Phases 1 and 2 — stable base
git checkout phase_3   # Phase 3 — synthesis added
git checkout phase_4   # Phase 4 — Document ingestion added
```

---

## Note on Python version

Python 3.14 is not supported — spaCy's pydantic v1 dependency does not
support it. Use Python 3.11 or 3.12.

```bash
uv python install 3.12
uv venv --python 3.12
```

---

## Licence

Code: GNU General Public License v3.0  
Papers: Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Links

- **Theory and papers:** https://github.com/lnxs-prsn/nbsi_theory_and_plan
- **Hugging Face:** https://huggingface.co/spaces/Node-Based-S-I/the_theory
