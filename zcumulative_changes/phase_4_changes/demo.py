"""
NBSI v1.0 Lightweight — Demo (Phase 3: with synthesis)

Ingests a document using spaCy + MiniLM.
Queries the resulting graph.
Returns real reasoning paths with conductivity scores.
Narrates the paths in natural language using a local LLM.

NO API calls. NO GPU required.

Usage:
    python demo.py                               # Built-in sample text
    python demo.py myfile.txt                    # Your own file
    python demo.py myfile.txt "your query"       # Custom query
    python demo.py myfile.txt "query" --no-synth # Skip synthesis

Requirements (core):
    uv pip install spacy sentence-transformers networkx numpy scipy
    python -m spacy download en_core_web_sm

Requirements (synthesis):
    uv pip install llama-cpp-python
    # Download model to ~/nbsi-models/ — see README
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default model path — change if you saved the model elsewhere
DEFAULT_MODEL = os.path.expanduser(
    "~/nbsi-models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)

SAMPLE_TEXT = """
Artificial intelligence is transforming the modern world through machine learning and
deep neural networks. Machine learning algorithms enable computers to learn patterns
from large datasets without being explicitly programmed for each task.

Deep learning, a subset of machine learning, uses neural networks with many layers to
process complex information. These neural networks are inspired by the human brain and
can recognise images, understand speech, and translate languages.

Natural language processing allows machines to understand and generate human language.
Because language is complex and ambiguous, natural language processing requires
significant computational resources and training data.

Reinforcement learning enables AI systems to learn through trial and error by receiving
rewards for correct actions. This approach has led to breakthroughs in game playing,
robotics, and autonomous vehicles.

However, AI systems face significant challenges including bias in training data,
lack of transparency, and high energy consumption. Despite these challenges, AI
continues to advance rapidly and is being applied across healthcare, finance,
education, and scientific research.

The future of AI depends on solving problems of safety, alignment, and interpretability.
Researchers are working on methods to ensure AI systems behave as intended and remain
under human control. These safety measures are critical as AI becomes more capable.
"""


def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        try:
            from pdfminer.high_level import extract_text
            return extract_text(path)
        except ImportError:
            print("[!] pdfminer.six not installed. Install with: uv pip install pdfminer.six")
            sys.exit(1)
    elif ext == '.docx':
        try:
            from docx import Document
            doc = Document(path)
            return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            print("[!] python-docx not installed. Install with: uv pip install python-docx")
            sys.exit(1)
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


def run_demo(text: str, query: str, use_synthesis: bool = True):
    print("\n" + "="*65)
    print("  NBSI v1.0 Lightweight — Demo")
    print("  spaCy · MiniLM · Local graph reasoning · llama-cpp")
    print("="*65)

    # ── [1/5] Load embedder ───────────────────────────────────────────
    print("\n[1/5] Loading MiniLM embedder...")
    t0 = time.time()
    try:
        from nbsi.embedder import RealEmbedder
        embedder = RealEmbedder()
        print(f"      Ready ({time.time()-t0:.1f}s)")
    except ImportError as e:
        print(f"\n[!] {e}")
        sys.exit(1)

    # ── [2/5] Extract nodes and edges ────────────────────────────────
    print("\n[2/5] Extracting graph from text (spaCy + MiniLM)...")
    t0 = time.time()
    try:
        from nbsi.ingestion.spacy_extractor import SpacyExtractor
        extractor = SpacyExtractor()
    except OSError as e:
        print(f"\n[!] {e}")
        sys.exit(1)

    nodes, edges = extractor.extract(text, embedder)
    print(f"      {len(nodes)} nodes · {len(edges)} edges ({time.time()-t0:.1f}s)")

    if not nodes:
        print("\n[!] No nodes extracted. Text may be too short.")
        sys.exit(1)

    # ── [3/5] Build session graph ─────────────────────────────────────
    print("\n[3/5] Building session graph...")
    t0 = time.time()
    from nbsi.config import Config
    from nbsi.lifecycle.structural_library import StructuralNodeLibrary
    from nbsi.session.session import NBSISession

    config = Config()
    config.MAX_NODES = 700
    config.BEAM_WIDTH = 8
    config.TOP_K_PATHS = 5

    library = StructuralNodeLibrary()
    session = NBSISession(structural_library=library, embedder=embedder, config=config)
    result  = session.ingest_graph(nodes, edges)

    print(f"      Graph: {result['nodes']} nodes · {result['edges']} edges ({time.time()-t0:.1f}s)")
    print(f"      Structural nodes firing:    {result['structural_nodes_firing']}")
    print(f"      Observation nodes created:  {result['observation_nodes_created']}")

    # ── [4/5] Query ───────────────────────────────────────────────────
    print(f"\n[4/5] Querying: \"{query}\"")
    t0     = time.time()
    paths  = session.query(query)
    elapsed = time.time() - t0

    print(f"      {len(paths)} paths found ({elapsed*1000:.1f}ms)\n")

    if not paths:
        print("  No paths found. Try a different query or a longer document.")
    else:
        print("  Reasoning paths (highest conductivity first):")
        print("  " + "-"*60)
        for i, p in enumerate(paths, 1):
            chain = " → ".join(p["path"])
            conf  = p["conductivity"]
            hops  = p["length"] - 1
            print(f"\n  Path {i}  [{conf:.3f} conductivity · {hops} hop{'s' if hops!=1 else ''}]")
            print(f"  {chain}")

    # ── [5/5] Synthesis ───────────────────────────────────────────────
    if use_synthesis and paths:
        print("\n" + "="*65)
        print("  Synthesis — local LLM narration")
        print("="*65)

        model_path = os.environ.get("NBSI_MODEL", DEFAULT_MODEL)

        if not os.path.exists(model_path):
            print(f"\n  [!] Model not found at: {model_path}")
            print(f"  Download with:")
            print(f"  mkdir -p ~/nbsi-models")
            print(f"  wget -P ~/nbsi-models https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf")
            print("\n  Skipping synthesis. Run with --no-synth to suppress this message.")
        else:
            try:
                from nbsi.synthesis.synthesiser import Synthesiser
                synth = Synthesiser(model_path, n_threads=4, verbose=False)

                print(f"\n  Narrating paths for: \"{query}\"")
                print("  (this takes 20-40 seconds on CPU — streaming output below)\n")
                print("  " + "-"*60)
                print("  ", end="", flush=True)

                # Stream tokens as they generate
                for token in synth.narrate_streaming(query, paths):
                    print(token, end="", flush=True)
                print("\n  " + "-"*60)

            except ImportError:
                print("\n  [!] llama-cpp-python not installed.")
                print("  Install with: uv pip install llama-cpp-python")

    # ── Speculation demo ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Speculation demo")
    print("="*65)
    spec_statement = "the session graph should persist across sessions for better continuity"
    print(f"\n  Adding: \"{spec_statement}\"")
    spec = session.speculate(spec_statement)
    print(f"  Spec ID:        {spec['spec_id'][:12]}...")
    print(f"  Edges affected: {spec['edges_affected']}")

    print(f"\n  Re-querying after speculation: \"{query}\"")
    paths_after = session.query(query)
    if paths_after:
        print(f"  Top path conductivity after:  {paths_after[0]['conductivity']:.3f}")
        if paths:
            print(f"  Top path conductivity before: {paths[0]['conductivity']:.3f}")
            delta     = paths_after[0]['conductivity'] - paths[0]['conductivity']
            direction = "↑ increased" if delta > 0 else "↓ decreased" if delta < 0 else "unchanged"
            print(f"  Change: {direction} ({delta:+.3f}) — SEM rewired the graph")

    print("\n  Rolling back speculation...")
    rollback = session.rollback_speculation(spec['spec_id'])
    print(f"  Rollback tolerance: {rollback['rollback_tolerance']:.2e} "
          f"(must be < 1e-9: {rollback['tolerance_ok']})")

    # ── Session end ───────────────────────────────────────────────────
    print("\n" + "="*65)
    summary = session.end_session()
    print(f"  Session ended.")
    print(f"  Graph destroyed:          {summary['session_graph_destroyed']}")
    print(f"  Nodes remaining in graph: {session.graph.node_count}")
    print(f"  Structural library:       {summary['structural_library']['total_structural_nodes']} nodes (persists)")
    print("="*65 + "\n")


if __name__ == "__main__":
    text_arg    = sys.argv[1] if len(sys.argv) > 1 else None
    query_arg   = sys.argv[2] if len(sys.argv) > 2 else None
    no_synth    = "--no-synth" in sys.argv

    if text_arg and not text_arg.startswith("--"):
        if not os.path.exists(text_arg):
            print(f"[!] File not found: {text_arg}")
            sys.exit(1)
        text = load_text(text_arg)
        print(f"Loaded: {text_arg} ({len(text)} chars)")
    else:
        text = SAMPLE_TEXT
        print("Using built-in sample text (AI overview).")
        print("To use your own: python demo.py yourfile.txt")

    query = query_arg or "what are the challenges facing AI development"
    run_demo(text, query, use_synthesis=not no_synth)
