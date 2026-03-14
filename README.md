ITS ALL AI NONSENSE BUT PROJECT WAS FUN WORKING WITH 
IT HAS NOTHING RELATED TO REASONING. ITS JUST BIASED KNOWLEDGE STORAGE AT BEST. 



# NBSI — Node-Based Semantic Intelligence
### Implementation Repository
A local document reasoning tool. Give it documents, ask questions, get answers traced through the actual structure of what you gave it.

No internet connection required after setup. No API calls. No data leaves your machine.

**Papers and theory:** https://github.com/lnxs-prsn/nbsi_theory_and_plan  
**Live overview:** https://huggingface.co/spaces/Node-Based-S-I/the_theory

---

## What it does

You give it one or more documents — PDF, Word, plain text, HTML. It reads them, builds a concept graph, and lets you query that graph. Answers are reasoning paths — traces through connected concepts in your documents — not generated text.

Optionally a small local language model narrates those paths in plain English.

Over multiple sessions it builds a structural layer — patterns that kept appearing across different documents get promoted into persistent landmarks. The more you use it on related material the better it orients itself.

## What it does not do

- It does not chat or have a conversation
- It does not search the internet
- It does not remember the content of your documents — only geometric patterns
- The structural layer takes many sessions to develop — do not expect it after one run

## Hardware

Runs on CPU only. Tested on:
- Linux laptop (Fedora)
- Raspberry Pi 5

Minimum: 4GB RAM. The embedding model loads ~500MB into memory. The optional local LLM needs an additional 1-3GB depending on model.

## Who this is for

You need to be comfortable with a terminal. If you can run `git clone` and follow install instructions you can run this. No Python experience required beyond that.

---

## Getting started

→ See [INSTALL.md](nbsi_v1_lightweight/INSTALL.md) for full setup instructions for Linux, macOS, Windows, and Raspberry Pi.

---

## Repository structure
```
nbsi_external_spec_trial_phase/
│
├── README.md                        # This file
├── LICENSE
│
├── nbsi_v1_lightweight/             # The project — start here
│   ├── INSTALL.md                   # Full installation instructions
│   ├── nbsi/
│   │   ├── demo.py                  # Quick start — run against any document
│   │   ├── main.py                  # Interactive session with persistent library
│   │   ├── config.py
│   │   ├── ingestion/               # Document reading and concept extraction
│   │   ├── lifecycle/               # Structural layer — observation and promotion
│   │   ├── os_integration/          # File watcher and ingestion worker
│   │   ├── reasoning/               # Beam search and conductivity scoring
│   │   ├── sem/                     # Speculation and rollback
│   │   ├── session/                 # Session management and persistence
│   │   ├── synthesis/               # Optional local LLM narration
│   │   └── tests/                   # All tests
│   ├── run_diversity_test.sh        # Automated multi-session diversity testing
│   └── the_file_folder/             # Sample documents for testing
│
└── zcumulative_changes/             # Build history and patch notes
```

---

## Build history

| Phase | Description |
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
| 3 | 🔲 Pending | Synthesis — llama-cpp local narration |
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
