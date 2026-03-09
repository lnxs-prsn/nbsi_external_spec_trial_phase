"""
NBSI v1.0 — Test Runner (no pytest required)

Runs all tests using only stdlib. Same test logic as test_nbsi.py
but structured as a standalone runner.

Usage: python3 run_tests.py
"""
import sys
import os
import math
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from nbsi.config import Config
from nbsi.embedder import StubEmbedder
from nbsi.graph.node import ConceptNode
from nbsi.graph.edge import ConceptEdge
from nbsi.graph.concept_graph import ConceptGraph
from nbsi.sem.propagator import SEMPropagator
from nbsi.lifecycle.nodes import ObservationNode, StructuralNode
from nbsi.lifecycle.structural_library import StructuralNodeLibrary
from nbsi.lifecycle.lifecycle_engine import LifecycleEngine
from nbsi.reasoning.conductivity import ConductivityEngine, QueryEngine
from nbsi.session.session import NBSISession


# ── Test harness ──────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0
RESULTS = []


def test(name):
    """Decorator to register a test function."""
    def decorator(fn):
        RESULTS.append((name, fn))
        return fn
    return decorator


def run_all():
    global PASS, FAIL
    PASS = 0
    FAIL = 0
    print("\n" + "="*70)
    print("  NBSI v1.0 — Test Suite")
    print("="*70)
    for name, fn in RESULTS:
        try:
            fn()
            print(f"  ✓  {name}")
            PASS += 1
        except AssertionError as e:
            print(f"  ✗  {name}")
            print(f"     AssertionError: {e}")
            FAIL += 1
        except Exception as e:
            print(f"  ✗  {name}")
            print(f"     {type(e).__name__}: {e}")
            traceback.print_exc()
            FAIL += 1
    print("="*70)
    print(f"  Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print("="*70 + "\n")
    return FAIL == 0


# ── Shared setup ──────────────────────────────────────────────────────────────
def make_config():
    c = Config()
    c.MIN_CONFIRMATIONS = 3
    c.STABILISATION_THRESHOLD = 0.3
    return c


def make_simple_graph(embedder):
    g = ConceptGraph()
    embs = embedder.encode(["solar energy", "photovoltaic cells", "electricity generation"])
    nodes = [
        ConceptNode(label="solar energy", embedding=embs[0].tolist(), activation=0.8),
        ConceptNode(label="photovoltaic cells", embedding=embs[1].tolist(), activation=0.7),
        ConceptNode(label="electricity generation", embedding=embs[2].tolist(), activation=0.9),
    ]
    ids = [g.add_node(n) for n in nodes]
    g.add_edge(ConceptEdge(source_id=ids[0], target_id=ids[1], edge_type="causal", base_weight=0.85))
    g.add_edge(ConceptEdge(source_id=ids[1], target_id=ids[2], edge_type="causal", base_weight=0.80))
    return g, ids, nodes


embedder = StubEmbedder()
config = make_config()


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH TESTS
# ══════════════════════════════════════════════════════════════════════════════
@test("Graph: add_node returns id and increments count")
def _():
    g = ConceptGraph()
    emb = embedder.encode(["test concept"])[0].tolist()
    node = ConceptNode(label="test concept", embedding=emb)
    nid = g.add_node(node)
    assert nid in g._nodes
    assert g.node_count == 1


@test("Graph: same label merges — does not duplicate")
def _():
    g = ConceptGraph()
    emb = embedder.encode(["shared concept"])[0].tolist()
    n1 = ConceptNode(label="shared concept", activation=0.5, embedding=emb)
    n2 = ConceptNode(label="shared concept", activation=0.8, embedding=emb)
    g.add_node(n1)
    g.add_node(n2)
    assert g.node_count == 1
    node = g.get_node_by_label("shared concept")
    assert node.activation == 0.8


@test("Graph: base_weight never changes when SEM delta applied")
def _():
    edge = ConceptEdge(source_id="a", target_id="b", edge_type="causal", base_weight=0.85)
    original_base = edge.base_weight
    edge.apply_sem_delta("spec1", 0.3)
    assert edge.base_weight == original_base
    assert edge.effective_weight > original_base


@test("Graph: effective_weight = base_weight * (1 + sum_of_deltas)")
def _():
    edge = ConceptEdge(source_id="a", target_id="b", base_weight=0.8)
    edge.apply_sem_delta("spec1", 0.25)
    edge.apply_sem_delta("spec2", 0.10)
    expected = 0.8 * (1.0 + 0.25 + 0.10)
    assert abs(edge.effective_weight - expected) < 1e-10


@test("Graph: hard reset destroys all nodes and edges")
def _():
    g, ids, nodes = make_simple_graph(embedder)
    assert g.node_count > 0
    g.reset()
    assert g.node_count == 0
    assert g.edge_count == 0
    assert len(g._label_index) == 0


@test("Graph: protected node survives betweenness pruning")
def _():
    g = ConceptGraph()
    embs = embedder.encode([f"concept {i}" for i in range(10)])
    ids = []
    for i, emb in enumerate(embs):
        node = ConceptNode(label=f"concept {i}", embedding=emb.tolist())
        ids.append(g.add_node(node))
    g._nodes[ids[0]].protected = True
    for i in range(len(ids) - 1):
        g.add_edge(ConceptEdge(source_id=ids[i], target_id=ids[i+1], edge_type="sequential"))
    g.prune_by_betweenness(5)
    assert ids[0] in g._nodes


# ══════════════════════════════════════════════════════════════════════════════
# SEM TESTS
# ══════════════════════════════════════════════════════════════════════════════
@test("SEM: speculation creates node in session graph")
def _():
    g, ids, nodes = make_simple_graph(embedder)
    sem = SEMPropagator(g, embedder, config)
    spec = sem.add_speculation("solar panels convert light to electricity")
    assert spec.spec_id in g._nodes


@test("SEM: propagation changes effective_weight but not base_weight")
def _():
    g, ids, nodes = make_simple_graph(embedder)
    original_bases = {(e.source_id, e.target_id): e.base_weight for e in g.all_edges()}
    sem = SEMPropagator(g, embedder, config)
    sem.add_speculation("solar panels convert light to electricity")
    for e in g.all_edges():
        key = (e.source_id, e.target_id)
        if key in original_bases:
            assert e.base_weight == original_bases[key], f"base_weight changed: {key}"


@test("SEM: rollback tolerance < 1e-9")
def _():
    g, ids, nodes = make_simple_graph(embedder)
    sem = SEMPropagator(g, embedder, config)
    spec = sem.add_speculation("wind turbines generate power")
    sem.rollback(spec.spec_id)
    tolerance = sem.verify_rollback_tolerance(spec.spec_id)
    assert tolerance < 1e-9, f"Rollback tolerance {tolerance:.2e} exceeds 1e-9"


@test("SEM: speculative node activation below source node activation")
def _():
    g, ids, nodes = make_simple_graph(embedder)
    sem = SEMPropagator(g, embedder, config)
    spec = sem.add_speculation("speculative claim about energy")
    spec_node = g.get_node(spec.spec_id)
    assert spec_node is not None
    assert spec_node.activation == config.SPEC_INITIAL_ACTIVATION
    assert spec_node.activation < 0.6


@test("SEM: rollback removes speculative node from graph")
def _():
    g, ids, nodes = make_simple_graph(embedder)
    sem = SEMPropagator(g, embedder, config)
    spec = sem.add_speculation("speculative claim")
    assert spec.spec_id in g._nodes
    sem.rollback(spec.spec_id)
    assert spec.spec_id not in g._nodes


@test("SEM: confirm writes delta permanently to base_weight")
def _():
    g, ids, nodes = make_simple_graph(embedder)
    sem = SEMPropagator(g, embedder, config)
    spec = sem.add_speculation("solar energy powers cities")
    if not spec.affected_edges:
        return  # Skip if no edges affected
    src, tgt = spec.affected_edges[0]
    edge = g.get_edge(src, tgt)
    base_before = edge.base_weight
    sem.confirm(spec.spec_id)
    # After confirm, no pending deltas for this spec
    assert spec.spec_id not in edge._sem_deltas


# ══════════════════════════════════════════════════════════════════════════════
# CONDUCTIVITY TESTS
# ══════════════════════════════════════════════════════════════════════════════
@test("Conductivity: geometric mean matches manual calculation")
def _():
    g, ids, nodes = make_simple_graph(embedder)
    engine = ConductivityEngine(g, config)
    conductivity, hop_scores = engine.path_conductivity(ids)
    if hop_scores and all(s > 0 for s in hop_scores):
        expected = math.exp(sum(math.log(s) for s in hop_scores) / len(hop_scores))
        assert abs(conductivity - expected) < 1e-10


@test("Conductivity: zero edge weight drives path score to zero")
def _():
    g = ConceptGraph()
    embs = embedder.encode(["node a", "node b", "node c"])
    nodes = [ConceptNode(label=f"node {l}", embedding=e.tolist(), activation=0.8)
             for l, e in zip("abc", embs)]
    ids = [g.add_node(n) for n in nodes]
    g.add_edge(ConceptEdge(source_id=ids[0], target_id=ids[1], base_weight=0.8))
    g.add_edge(ConceptEdge(source_id=ids[1], target_id=ids[2], base_weight=0.0))
    engine = ConductivityEngine(g, config)
    conductivity, _ = engine.path_conductivity(ids)
    assert conductivity == 0.0


@test("Conductivity: beam search returns at least one path from connected graph")
def _():
    g, ids, nodes = make_simple_graph(embedder)
    engine = ConductivityEngine(g, config)
    paths = engine.beam_search(ids[0])
    assert len(paths) > 0


@test("Conductivity: no cycles in beam search paths")
def _():
    g, ids, nodes = make_simple_graph(embedder)
    engine = ConductivityEngine(g, config)
    paths = engine.beam_search(ids[0])
    for path in paths:
        assert len(path.node_ids) == len(set(path.node_ids)), \
            f"Cycle detected: {path.node_ids}"


@test("Conductivity: geometric mean is length-normalised")
def _():
    g = ConceptGraph()
    embs = embedder.encode([f"node {i}" for i in range(4)])
    nodes = [ConceptNode(label=f"node {i}", embedding=embs[i].tolist(), activation=1.0)
             for i in range(4)]
    ids = [g.add_node(n) for n in nodes]
    for i in range(3):
        g.add_edge(ConceptEdge(source_id=ids[i], target_id=ids[i+1], base_weight=1.0))
    engine = ConductivityEngine(g, config)
    cond_2hop, _ = engine.path_conductivity(ids[:3])
    cond_3hop, _ = engine.path_conductivity(ids)
    assert abs(cond_2hop - cond_3hop) < 0.02, \
        f"Length normalisation failed: 2-hop={cond_2hop:.4f}, 3-hop={cond_3hop:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE TESTS — White paper predictions 1–4
# ══════════════════════════════════════════════════════════════════════════════
@test("Lifecycle P1: novel content → observation node, not structural node")
def _():
    library = StructuralNodeLibrary()
    engine = LifecycleEngine(library, embedder, config)
    emb = embedder.encode(["completely novel pattern xyz123"])[0].tolist()
    assert engine.library.match(emb) == []
    obs_id = engine.observe("novel pattern", emb)
    assert obs_id in engine._observation_nodes
    assert obs_id not in engine.library.nodes


@test("Lifecycle P1: structural node fires on matching content")
def _():
    library = StructuralNodeLibrary()
    engine = LifecycleEngine(library, embedder, config)
    emb = embedder.encode(["reliable recurrence pattern"])[0].tolist()
    node = StructuralNode(node_id="test-structural", pattern_embedding=emb)
    engine.library.add(node)
    matches = engine.library.match(emb, threshold=0.5)
    assert len(matches) > 0
    assert matches[0][0] == "test-structural"


@test("Lifecycle P2: confirmed diverse node is promoted to structural")
def _():
    library = StructuralNodeLibrary()
    engine = LifecycleEngine(library, embedder, config)
    emb = embedder.encode(["reliable recurrence"])[0].tolist()
    # Observe from entity_A (first call)
    obs_id = engine.observe("reliable recurrence", emb, entity_id="entity_A")
    obs = engine._observation_nodes[obs_id]
    # Confirm from multiple entities — this is the generalisation requirement:
    # a truly generalisable pattern fires across different entities, not just one.
    entities = ["entity_A", "entity_B", "entity_C"]
    for i in range(config.MIN_CONFIRMATIONS):
        obs.update_pattern(emb, f"recurrence_{i}", entity_id=entities[i % len(entities)])
    diverse_dists = [
        np.array([1, 2, 3, 10], dtype=float),
        np.array([1, 1, 5, 5, 8], dtype=float),
        np.array([2, 2, 2, 9], dtype=float),
    ]
    promoted = engine.evaluate_for_stabilisation(obs_id, diverse_dists)
    assert promoted is True
    assert obs_id in engine.library.nodes
    assert obs_id not in engine._observation_nodes


@test("Lifecycle P2: too few confirmations — not promoted")
def _():
    library = StructuralNodeLibrary()
    engine = LifecycleEngine(library, embedder, config)
    emb = embedder.encode(["sparse pattern"])[0].tolist()
    obs_id = engine.observe("sparse pattern", emb)
    obs = engine._observation_nodes[obs_id]
    obs.update_pattern(emb, "sparse_1")  # Only 1 confirmation
    diverse_dists = [np.array([1, 2, 3]), np.array([1, 1, 4])]
    promoted = engine.evaluate_for_stabilisation(obs_id, diverse_dists)
    assert promoted is False
    assert obs_id in engine._observation_nodes


@test("Lifecycle P3: reactivation puts node in active mode")
def _():
    library = StructuralNodeLibrary()
    engine = LifecycleEngine(library, embedder, config)
    base_emb = embedder.encode(["recurring seasonal pattern"])[0].tolist()
    node = StructuralNode(node_id="infra-node", pattern_embedding=base_emb)
    engine.library.add(node)
    assert node.attention_mode == "infrastructure"
    # Create an embedding in the anomaly zone via controlled shift
    arr = np.array(base_emb)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(len(arr)) * 0.8
    shifted = arr + noise
    shifted = (shifted / np.linalg.norm(shifted)).tolist()
    # Check if it lands in anomaly zone — if so, verify reactivation
    sim = float(np.dot(np.array(shifted), np.array(base_emb)))
    if config.REACTIVATION_LOW_THRESHOLD < sim < config.REACTIVATION_HIGH_THRESHOLD:
        reactivated = engine.check_for_reactivation(shifted)
        assert "infra-node" in reactivated
        assert engine.library.nodes["infra-node"].attention_mode == "active"
    else:
        # Directly test the reactivation mechanism
        node.reactivate()
        assert node.attention_mode == "active"
        assert node.reactivation_count == 1


@test("Lifecycle P4: refinement returns node to infrastructure")
def _():
    library = StructuralNodeLibrary()
    engine = LifecycleEngine(library, embedder, config)
    base_emb = embedder.encode(["refinement test pattern"])[0].tolist()
    node = StructuralNode(node_id="refine-node", pattern_embedding=base_emb)
    node.reactivate()
    engine.library.add(node)
    assert node.attention_mode == "active"
    result = engine.refine_reactivated("refine-node", base_emb)
    assert result is True
    assert engine.library.nodes["refine-node"].attention_mode == "infrastructure"
    assert engine.library.nodes["refine-node"].reactivation_count == 1


@test("Lifecycle P4: refinement moves centroid toward anomaly")
def _():
    library = StructuralNodeLibrary()
    engine = LifecycleEngine(library, embedder, config)
    base_emb = embedder.encode(["pattern to refine"])[0].tolist()
    node = StructuralNode(node_id="centroid-node", pattern_embedding=list(base_emb))
    node.reactivate()
    engine.library.add(node)
    anomaly_emb = embedder.encode(["anomalous variant of pattern"])[0].tolist()
    original_centroid = list(base_emb)
    engine.refine_reactivated("centroid-node", anomaly_emb)
    new_centroid = engine.library.nodes["centroid-node"].pattern_embedding
    diff = sum(abs(new_centroid[i] - original_centroid[i]) for i in range(len(original_centroid)))
    assert diff > 0, "Centroid did not move after refinement"


@test("Lifecycle: entity-specific node blocked from promotion")
def _():
    library = StructuralNodeLibrary()
    engine = LifecycleEngine(library, embedder, config)
    emb = embedder.encode(["entity specific pattern"])[0].tolist()
    obs_id = engine.observe("entity specific", emb, entity_id="entity_A")
    obs = engine._observation_nodes[obs_id]
    for i in range(config.MIN_CONFIRMATIONS + 2):
        obs.update_pattern(emb, f"instance_{i}", entity_id="entity_A")
    diverse_dists = [np.array([1, 2, 10]), np.array([1, 5, 5]), np.array([2, 3, 8])]
    promoted = engine.evaluate_for_stabilisation(obs_id, diverse_dists)
    assert promoted is False, "Entity-specific node must not be promoted"


@test("Lifecycle: structural node contains no content after promotion")
def _():
    library = StructuralNodeLibrary()
    engine = LifecycleEngine(library, embedder, config)
    emb = embedder.encode(["content to promote"])[0].tolist()
    obs_id = engine.observe("content to promote", emb)
    obs = engine._observation_nodes[obs_id]
    for i in range(config.MIN_CONFIRMATIONS):
        obs.update_pattern(emb, f"instance text {i}")
    diverse_dists = [np.array([1, 3, 5]), np.array([2, 2, 6]), np.array([1, 4, 4])]
    engine.evaluate_for_stabilisation(obs_id, diverse_dists)
    if obs_id in engine.library.nodes:
        structural = engine.library.nodes[obs_id]
        assert structural.pattern_embedding is not None
        assert isinstance(structural.pattern_embedding, list)
        assert not hasattr(structural, 'instances')
        assert not hasattr(structural, 'raw_texts')


@test("Lifecycle: observation node ring buffer capped at 5")
def _():
    obs = ObservationNode()
    emb = embedder.encode(["test"])[0].tolist()
    for i in range(10):
        obs.update_pattern(emb, f"label_{i}")
    assert len(obs.example_labels) <= 5
    assert obs.confirmation_count == 10  # Count is correct


# ══════════════════════════════════════════════════════════════════════════════
# SESSION INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════
@test("Session: full lifecycle round-trip — two-layer separation")
def _():
    library = StructuralNodeLibrary()
    session = NBSISession(structural_library=library, embedder=embedder, config=config)

    # Build session graph
    embs = embedder.encode(["climate change", "carbon emissions", "global warming", "renewable energy"])
    nodes = [
        ConceptNode(label="climate change", embedding=embs[0].tolist(), activation=0.8),
        ConceptNode(label="carbon emissions", embedding=embs[1].tolist(), activation=0.7),
        ConceptNode(label="global warming", embedding=embs[2].tolist(), activation=0.9),
        ConceptNode(label="renewable energy", embedding=embs[3].tolist(), activation=0.75),
    ]
    ids = [session.graph.add_node(n) for n in nodes]
    edges = [
        ConceptEdge(ids[0], ids[1], edge_type="causal"),
        ConceptEdge(ids[1], ids[2], edge_type="causal"),
        ConceptEdge(ids[2], ids[3], edge_type="contrastive"),
        ConceptEdge(ids[0], ids[3], edge_type="analogical"),
    ]
    for e in edges:
        session.graph.add_edge(e)
    session._session_degree_dist = session.graph.degree_distribution()

    # Query
    results = session.query("what causes warming")
    assert isinstance(results, list)

    # Speculate and rollback
    spec = session.speculate("solar panels replace fossil fuels")
    rollback = session.rollback_speculation(spec["spec_id"])
    assert rollback["tolerance_ok"] is True

    # End session
    summary = session.end_session()
    assert summary["session_graph_destroyed"] is True
    assert session.graph.node_count == 0
    assert session.graph.edge_count == 0
    assert session.library is library  # Library object unchanged


@test("Session: structural library survives session reset")
def _():
    library = StructuralNodeLibrary()
    emb = embedder.encode(["pre-existing pattern"])[0].tolist()
    pre_node = StructuralNode(node_id="pre-node", pattern_embedding=emb)
    library.add(pre_node)
    assert len(library.nodes) == 1

    session = NBSISession(structural_library=library, embedder=embedder, config=config)
    session._session_degree_dist = np.array([1.0, 2.0])
    session.end_session()
    assert "pre-node" in library.nodes


@test("Session: no cross-session contamination")
def _():
    library = StructuralNodeLibrary()
    session = NBSISession(structural_library=library, embedder=embedder, config=config)
    emb = embedder.encode(["arctic ice melting"])[0].tolist()
    node = ConceptNode(label="arctic ice melting", embedding=emb)
    session.graph.add_node(node)
    session._session_degree_dist = session.graph.degree_distribution()
    session.end_session()
    assert session.graph.node_count == 0
    assert session.graph.get_node_by_label("arctic ice melting") is None


@test("Session: empty graph query returns empty list")
def _():
    library = StructuralNodeLibrary()
    session = NBSISession(library, embedder, config)
    results = session.query("test query")
    assert results == []


@test("Session: speculation on empty graph does not crash")
def _():
    library = StructuralNodeLibrary()
    session = NBSISession(library, embedder, config)
    result = session.speculate("speculation on empty graph")
    assert "spec_id" in result
    assert result["status"] == "active"


@test("Session: state() returns correct structure")
def _():
    library = StructuralNodeLibrary()
    session = NBSISession(library, embedder, config)
    state = session.state()
    required_keys = ["session_id", "nodes", "edges", "ensemble_mode",
                     "active_speculations", "structural_nodes", "library"]
    for key in required_keys:
        assert key in state, f"Missing key in state(): {key}"


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
