# Installation Guide

## Before you start

You need:
- A terminal (Command Prompt or PowerShell on Windows)
- Git
- About 2GB free disk space (more if you want the local LLM)

That is all. You do not need Python installed — the setup instructions below handle it.

---

## Step 1 — Install system dependencies

These are required before installing any Python packages.

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install git build-essential cmake python3-dev
```

### Fedora / RHEL
```bash
sudo dnf install git gcc gcc-c++ cmake python3-devel
```

### Raspberry Pi (Raspberry Pi OS)
```bash
sudo apt update
sudo apt install git build-essential cmake python3-dev libopenblas-dev
```
Note: the optional LLM will compile from source on Pi — allow 15-20 minutes.

### macOS
```bash
xcode-select --install
```
If you have Homebrew: `brew install cmake`

### Windows
1. Install Git: https://git-scm.com/download/win
2. Install Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   — Select "Desktop development with C++" during install
3. Install cmake: https://cmake.org/download/

---

## Step 2 — Install uv

uv manages Python and packages. It replaces pip and pyenv in one tool.
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal after installing uv.

---

## Step 3 — Clone the repository
```bash
git clone https://github.com/lnxs-prsn/nbsi_external_spec_trial_phase
cd nbsi_external_spec_trial_phase/nbsi_v1_lightweight/nbsi
```

---

## Step 4 — Create environment and install packages
```bash
uv venv --python 3.12
source .venv/bin/activate
```

Windows:
```
.venv\Scripts\activate
```

Then install:
```bash
uv pip install spacy sentence-transformers networkx numpy scipy watchdog python-docx pdfminer.six
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

This will take a few minutes. The sentence-transformers package downloads a small embedding model (~90MB) on first run.

---

## Step 5 — Run it

From inside `nbsi_v1_lightweight/nbsi/`:
```bash
# Built-in sample — no file needed
PYTHONPATH=.. python demo.py

# Your own file
PYTHONPATH=.. python demo.py yourfile.pdf "your question here" --no-synth

# Your own file, Word document
PYTHONPATH=.. python demo.py yourfile.docx "your question here" --no-synth
```

Windows — replace `PYTHONPATH=..` with:
```
set PYTHONPATH=.. && python demo.py
```

You should see the document being read, chunks being extracted, and then reasoning paths returned for your query.

---

## Interactive session with persistent library

The demo runs one session and exits. For persistent sessions that accumulate knowledge across runs use main.py instead.

Run from inside `nbsi_v1_lightweight/`:
```bash
# Watch a folder — any file dropped in gets ingested automatically
PYTHONPATH=. python nbsi/main.py --watch ~/nbsi-inbox --no-synth

# Load a file immediately on startup
PYTHONPATH=. python nbsi/main.py --files nbsi/NBSI_External_Paper.docx --no-synth
```

The library saves automatically at `~/.nbsi/library.json` when you exit.
Run multiple sessions against different documents to accumulate structural patterns.

**Commands inside the session:**

| Command | What it does |
|---|---|
| `:help` | Show all commands |
| `:ingest path/to/file` | Ingest a file mid-session |
| `:stats` | Graph and library statistics |
| `:library` | Show structural node details |
| `:speculate concept` | Run speculation on a concept |
| `:rollback` | Undo last speculation |
| `:exit` | Save and exit |

---

## Optional — Local LLM narration

Without this, reasoning paths are returned as concept chains. With this, a small local model narrates them in plain English.
```bash
uv pip install llama-cpp-python
mkdir -p ~/nbsi-models

# Laptop — Qwen2.5 1.5B (~900MB)
wget -P ~/nbsi-models https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Raspberry Pi — Phi-3 Mini (~2.4GB)
wget -P ~/nbsi-models https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

Then run without `--no-synth`:
```bash
PYTHONPATH=.. python demo.py yourfile.pdf "your question"
```

---

## Run the tests
```bash
# From inside nbsi_v1_lightweight/nbsi/
PYTHONPATH=.. python run_tests.py
# Expected: 33 passed

PYTHONPATH=.. python -m unittest tests.test_phase4_real -v
# Expected: 40 passed

PYTHONPATH=.. python -m unittest tests.test_phase4_os_integration -v
# Expected: 21 passed

PYTHONPATH=.. python -m unittest tests.test_phase6_persistence -v
# Expected: 24 passed
```

---

## Troubleshooting

**llama-cpp-python fails to install**
It compiles from source. Verify cmake is available: `cmake --version`
If not found install it — see Step 1 for your OS.
If you do not need narration just use `--no-synth` and skip this package entirely.

**spaCy model download fails**
Try the direct install instead:
```bash
python -m spacy download en_core_web_sm
```

**numpy errors on Raspberry Pi**
```bash
sudo apt install libopenblas-dev
uv pip install --force-reinstall numpy
```

**PYTHONPATH not recognised on Windows**
In CMD: `set PYTHONPATH=.. && python demo.py`
In PowerShell: `$env:PYTHONPATH=".."; python demo.py`

**Python version error**
Python 3.14 is not supported — spaCy does not support it yet.
Install 3.12 explicitly: `uv python install 3.12` then `uv venv --python 3.12`