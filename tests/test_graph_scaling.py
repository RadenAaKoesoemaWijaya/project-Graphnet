import numpy as np
import pandas as pd

from model import create_claim_graph, create_knn_graph


def test_knn_graph_enforces_node_limit():
    features = np.random.default_rng(42).normal(size=(21, 4)).astype(np.float32)

    node_features, edge_index = create_knn_graph(features, k=3, max_nodes=10)

    assert node_features.shape[0] == 10
    assert edge_index.shape[0] == 2
    assert int(edge_index.max()) < 10


def test_star_graph_resets_index_and_enforces_edge_limit():
    frame = pd.DataFrame(
        {
            "value": np.arange(20, dtype=np.float32),
            "provider_id": ["provider"] * 20,
        },
        index=np.arange(100, 120),
    )

    node_features, edge_index = create_claim_graph(
        frame,
        ["value"],
        method="star",
        max_nodes=10,
        max_edges=6,
    )

    assert node_features.shape[0] == 10
    assert edge_index.shape == (2, 6)
    assert int(edge_index.max()) < 10


# ── Tests for build_anomaly_subgraph ─────────────────────────────────────────

import torch
from model import build_anomaly_subgraph


def _make_star_graph(n: int = 100):
    """Return (node_features, edge_index_np, gnn_scores) for a simple star graph."""
    rng = np.random.default_rng(0)
    node_features = rng.normal(size=(n, 5)).astype(np.float32)
    # Star topology: node 0 connected to all others
    src = [0] * (n - 1) + list(range(1, n))
    dst = list(range(1, n)) + [0] * (n - 1)
    ei = np.array([src, dst], dtype=np.int64)
    gnn_scores = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    # Make nodes 10–14 clearly anomalous
    gnn_scores[10:15] = 0.95
    return node_features, ei, gnn_scores


def test_build_anomaly_subgraph_basic():
    """Subgraph must be non-empty and respect max_viz_nodes."""
    node_features, ei, scores = _make_star_graph(100)
    result = build_anomaly_subgraph(
        node_features=node_features,
        edge_index=ei,
        gnn_scores=scores,
        top_k_anomalies=10,
        hop=1,
        max_viz_nodes=50,
    )
    sub_ids = result['sub_node_ids']
    sub_ei  = result['sub_edge_index']

    assert len(sub_ids) > 0, "subgraph must contain at least one node"
    assert len(sub_ids) <= 50, "must respect max_viz_nodes"
    assert result['n_total_nodes'] == 100
    assert result['n_total_edges'] == ei.shape[1]
    # sub_edge_index uses remapped IDs → all values must be < len(sub_ids)
    if sub_ei.shape[1] > 0:
        assert int(sub_ei.max()) < len(sub_ids)


def test_build_anomaly_subgraph_seeds_included():
    """All top-K anomaly seeds must appear in the subgraph."""
    node_features, ei, scores = _make_star_graph(100)
    result = build_anomaly_subgraph(
        node_features=node_features,
        edge_index=ei,
        gnn_scores=scores,
        top_k_anomalies=5,
        hop=1,
        max_viz_nodes=300,
    )
    sub_ids_set = set(result['sub_node_ids'])
    # Nodes 10–14 have score 0.95 — they must be seeds
    for seed_node in range(10, 15):
        assert seed_node in sub_ids_set, f"seed node {seed_node} missing from subgraph"

    is_seed = result['is_seed']
    assert is_seed.dtype == bool
    assert is_seed.sum() >= 5, "at least 5 seed nodes expected"


def test_build_anomaly_subgraph_scores_shape():
    """sub_scores must match sub_node_ids length."""
    node_features, ei, scores = _make_star_graph(80)
    result = build_anomaly_subgraph(
        node_features=node_features,
        edge_index=ei,
        gnn_scores=scores,
        top_k_anomalies=15,
        max_viz_nodes=200,
    )
    assert result['sub_scores'].shape[0] == len(result['sub_node_ids'])
    assert result['is_seed'].shape[0] == len(result['sub_node_ids'])


def test_build_anomaly_subgraph_with_edge_type():
    """edge_type is propagated to sub_edge_type with matching length."""
    node_features, ei, scores = _make_star_graph(60)
    n_edges = ei.shape[1]
    edge_type = np.zeros(n_edges, dtype=np.int64)
    edge_type[: n_edges // 2] = 1  # half provider, half patient

    result = build_anomaly_subgraph(
        node_features=node_features,
        edge_index=ei,
        gnn_scores=scores,
        edge_type=edge_type,
        top_k_anomalies=10,
        max_viz_nodes=100,
    )
    sub_et = result['sub_edge_type']
    sub_ei = result['sub_edge_index']
    assert sub_et is not None, "sub_edge_type must be set when edge_type supplied"
    assert sub_et.shape[0] == sub_ei.shape[1], "edge_type len must match edge count"


def test_build_anomaly_subgraph_torch_tensor_input():
    """Accepts torch.LongTensor edge_index (common in-memory format)."""
    node_features, ei_np, scores = _make_star_graph(50)
    ei_torch = torch.LongTensor(ei_np)
    result = build_anomaly_subgraph(
        node_features=node_features,
        edge_index=ei_torch,
        gnn_scores=scores,
        top_k_anomalies=5,
        max_viz_nodes=100,
    )
    assert len(result['sub_node_ids']) > 0


def test_build_anomaly_subgraph_degenerate_single_node():
    """Single-node graph must not crash; returns at least that node."""
    nf   = np.zeros((1, 3), dtype=np.float32)
    ei   = np.array([[0], [0]], dtype=np.int64)  # self-loop
    sc   = np.array([0.9])
    result = build_anomaly_subgraph(nf, ei, sc, top_k_anomalies=5, max_viz_nodes=50)
    assert result['n_total_nodes'] == 1
    assert len(result['sub_node_ids']) == 1


def test_build_anomaly_subgraph_all_low_scores():
    """When all scores are below threshold, top-K seeds still populate subgraph."""
    node_features, ei, _ = _make_star_graph(40)
    low_scores = np.full(40, 0.1)   # all below default threshold 0.5
    result = build_anomaly_subgraph(
        node_features=node_features,
        edge_index=ei,
        gnn_scores=low_scores,
        top_k_anomalies=5,
        max_viz_nodes=100,
        anomaly_threshold=0.5,
    )
    # Even with no above-threshold nodes, top-K fallback ensures non-empty result
    assert len(result['sub_node_ids']) > 0
