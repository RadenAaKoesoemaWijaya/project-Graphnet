import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from model import (
    CombinedAnomalyDetector,
    ClaimAnomalyXGBoostModel,
    InsuranceAnomalyGNNModel,
    clean_gpu_memory,
    TORCH_AVAILABLE,
)
from fraud_risk_pipeline import compute_patient_level_fuzzy_similarity_scores
from preprocessing_optimized import get_cached_pseudo_labels, _PSEUDO_LABEL_CACHE


def test_clean_gpu_memory_safety():
    """Verify clean_gpu_memory executes without exceptions regardless of torch/cuda state."""
    clean_gpu_memory()


def test_xgboost_hardware_parameter_initialization():
    """Verify ClaimAnomalyXGBoostModel sets device parameter properly."""
    model = ClaimAnomalyXGBoostModel(model_type='xgboost')
    assert 'device' in model.params
    assert model.params['device'] in ['cuda', 'cpu']


def test_xgboost_cuda_fallback_resilience():
    """Verify XGBoost falls back to CPU if a CUDA fit failure occurs."""
    model = ClaimAnomalyXGBoostModel(model_type='xgboost', device='cuda')
    X = np.random.randn(40, 4)
    y = np.random.randint(0, 2, size=40)
    # Ensure both classes exist
    y[0] = 0
    y[1] = 1
    # fit should run and if cuda is not supported, fallback to cpu gracefully
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (40, 2)


def test_gnn_inference_fallback_with_self_loops():
    """Verify that when NeighborLoader fails, chunked fallback executes with self-loops and does not crash."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")

    import torch

    detector = CombinedAnomalyDetector(
        algorithms=['gnn'],
        random_state=42,
        verbose=False
    )
    # Mock trained GNN model
    feature_dim = 8
    num_nodes = 50
    detector.gnn_model = InsuranceAnomalyGNNModel(
        num_features=feature_dim,
        num_classes=2,
        hidden_channels=16,
        num_heads=1,
        num_layers=1
    )
    detector.gnn_weight = 1.0
    detector.isolation_forest = None
    detector.autoencoder = None
    detector.xgboost_model = None

    features = np.random.randn(num_nodes, feature_dim).astype(np.float32)
    # Create simple edge_index (line graph)
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    edge_index = np.array([src, dst])

    # Fit dummy scaler & imputer
    detector.imputer.fit(features)
    detector.scaler.fit(features)

    # Patch NeighborLoader to simulate a runtime exception so it triggers the fallback
    with patch.dict('sys.modules', {'torch_geometric.loader': MagicMock(NeighborLoader=MagicMock(side_effect=RuntimeError("Simulated loader failure")))}):
        probs, ind = detector.predict_anomaly_probability(features, edge_index=edge_index, device='cpu')
        assert len(probs) == num_nodes
        assert 'gnn' in ind
        # Crucial check: GNN scores must NOT be all zeros due to silent crash!
        assert not np.all(ind['gnn'] == 0.0)


def test_fuzzy_similarity_optimization_parity():
    """Verify optimized compute_patient_level_fuzzy_similarity_scores computes correct scores and handles time windows."""
    df = pd.DataFrame({
        "_astina_row_id": [101, 102, 103, 104],
        "patient_id": ["P001", "P001", "P001", "P002"],
        "billing_date": ["2024-01-01", "2024-01-05", "2024-03-01", "2024-01-02"],
        "service_code": ["SRV_A", "SRV_A", "SRV_A", "SRV_B"],
        "provider_id": ["PRV_1", "PRV_1", "PRV_1", "PRV_2"],
        "diagnosis_code": ["DIAG_1", "DIAG_1", "DIAG_1", "DIAG_2"],
        "total_claim_amount": [500000, 505000, 500000, 1000000],
    })

    scores = compute_patient_level_fuzzy_similarity_scores(df, max_window_days=30, min_similarity=0.8)
    assert len(scores) == 4
    assert isinstance(scores, pd.Series)
    assert scores.name == "fuzzy_similarity_score"

    # Row 101 and 102 are 4 days apart with same service, provider, patient -> high similarity
    assert scores.iloc[0] > 0.8
    assert scores.iloc[1] > 0.8

    # Row 103 is 60 days later -> gap > 30 days -> score should be 0.0
    assert scores.iloc[2] == 0.0

    # Row 104 is patient P002 (single claim) -> score should be 0.0
    assert scores.iloc[3] == 0.0


def test_pseudo_label_caching():
    """Verify get_cached_pseudo_labels caches results and avoids redundant fits."""
    _PSEUDO_LABEL_CACHE.clear()
    X = pd.DataFrame(np.random.randn(60, 5), columns=[f"col_{i}" for i in range(5)])

    labels1 = get_cached_pseudo_labels(X)
    assert len(_PSEUDO_LABEL_CACHE) == 1
    assert len(labels1) == 60

    # Second call should retrieve from cache
    labels2 = get_cached_pseudo_labels(X)
    assert len(_PSEUDO_LABEL_CACHE) == 1
    assert np.array_equal(labels1, labels2)
