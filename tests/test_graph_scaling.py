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