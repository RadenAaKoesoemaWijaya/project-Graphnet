import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from model import (
    get_adaptive_gnn_threshold,
    get_optimal_batch_size,
    create_claim_graph,
    create_knn_graph,
    CombinedAnomalyDetector,
    TORCH_AVAILABLE,
)


def test_adaptive_gnn_threshold_cpu():
    """Verify adaptive threshold returns sensible values on CPU."""
    threshold = get_adaptive_gnn_threshold(device=None, feature_dim=10, num_nodes=10000)
    assert isinstance(threshold, int)
    assert 1000 <= threshold <= 10000


def test_adaptive_gnn_threshold_cuda_mocked():
    """Verify adaptive threshold handles mocked CUDA device properties safely."""
    mock_device = MagicMock()
    mock_device.type = 'cuda'

    with patch('model.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 2 * 1024**3  # 2GB
        mock_torch.cuda.get_device_properties.return_value = mock_props

        thresh_small = get_adaptive_gnn_threshold(device=mock_device, feature_dim=16, num_nodes=10000)
        assert thresh_small == 3000

        mock_props.total_memory = 12 * 1024**3  # 12GB
        thresh_large = get_adaptive_gnn_threshold(device=mock_device, feature_dim=16, num_nodes=10000)
        assert thresh_large == 8000


def test_get_optimal_batch_size_fallback():
    """Verify optimal batch size calculation fallback when memory tools are constrained."""
    batch_size = get_optimal_batch_size(device=None, num_samples=5000, feature_dim=20, default_batch=512)
    assert isinstance(batch_size, int)
    assert batch_size > 0
    assert batch_size <= 16384


def test_detector_initialization_and_resilience():
    """Verify detector initializes and runs baseline anomaly algorithms even if GNN is skipped."""
    detector = CombinedAnomalyDetector(
        algorithms=['isolation_forest', 'xgboost'],
        random_state=42,
        verbose=False
    )
    X = np.random.randn(50, 5)
    detector.fit(X)
    preds, individual = detector.predict_anomaly_probability(X)
    assert len(preds) == 50
    assert 'isolation_forest' in individual
