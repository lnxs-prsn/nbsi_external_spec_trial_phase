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
| 1 | Core engine — graph, SEM operator, beam search, lifecycle |
| 2 | Real embeddings + spaCy extraction + three bug fixes |
| 3 | Local LLM narration via llama-cpp |
| 4 | Document ingestion — chunker, structure-aware readers, multi-document pipeline |
| 5 | OS integration — file watcher, ingestion worker |
| 6 | Persistence — library saves and loads across sessions |
| 7 | Full orchestration — main.py, persistent sessions, watcher integration, persistence bug fixes |

---

## Licence

Code: GNU General Public License v3.0  
Papers: Creative Commons Attribution 4.0 International (CC BY 4.0)