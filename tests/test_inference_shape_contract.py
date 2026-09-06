"""IMP-2 Integration tests: train(N) then predict(M) with M != N.

These tests guard against the exact bugs that previously slipped through
the 100%-passing 53-test suite:
  * BUG-1: GNN producing per-node probabilities instead of per-claim.
  * BUG-2: HDBSCAN/DBSCAN returning training-length arrays on inference.

Every individual-algorithm probability vector and the final ensemble
probability MUST have length equal to the number of rows passed to
predict_anomaly_probability / predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import CombinedAnomalyDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n_rows: int, n_features: int = 10, seed: int = 42, with_entity_cols: bool = True):
    """Return (X_df, feature_cols).

    The first 80% rows are low-variance "normal" samples; the last 20% are
    high-magnitude "anomaly" samples. This guarantees that the consensus
    pseudo-labeler sees two classes, so XGBoost/LightGBM do not abort with
    "Invalid classes inferred from unique values of `y`".
    """
    rng = np.random.default_rng(seed)
    n_normal = int(0.8 * n_rows)
    n_anom = n_rows - n_normal
    base = rng.normal(loc=0.0, scale=0.5, size=(n_normal, n_features))
    anomalies = rng.normal(loc=5.0, scale=1.5, size=(n_anom, n_features))
    values = np.vstack([base, anomalies]).astype(np.float32)
    # Light shuffle so anomalies are not all at the tail (preserves 2-class mix).
    order = rng.permutation(n_rows)
    values = values[order]

    data = {f"feat_{i}": values[:, i] for i in range(n_features)}
    if with_entity_cols:
        data["provider_id"] = [f"P{i % 7}" for i in range(n_rows)]
        data["patient_id"] = [f"PT{i % 13}" for i in range(n_rows)]
        data["diagnosis_code"] = [f"D{i % 5}" for i in range(n_rows)]
    df = pd.DataFrame(data)
    feature_cols = [f"feat_{i}" for i in range(n_features)]
    return df, feature_cols


_TRAIN_N = 200
_TEST_M_SMALL = 37
_TEST_M_LARGE = 311


def _auto_device():
    """Use CUDA if available — matches model.fit() auto-detection — so we
    never train on GPU then manually call predict(device='cpu') which would
    trigger AE device-placement errors.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# IMP-2 core tests — shape is the contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algorithms", [
    ["isolation_forest"],
    ["autoencoder"],
    ["dbscan"],
    ["xgboost"],
    ["gnn"],
    ["isolation_forest", "autoencoder", "dbscan", "xgboost", "gnn"],
])
def test_ensemble_predict_shape_when_test_smaller_than_train(algorithms):
    """Train on N rows, predict on M < N rows. All outputs must be (M,)."""
    train_df, feats = _make_dataset(_TRAIN_N, seed=1)
    test_df, _ = _make_dataset(_TEST_M_SMALL, seed=2)

    X_train = train_df[feats].values
    X_test = test_df[feats].values
    dev = _auto_device()

    detector = CombinedAnomalyDetector(
        algorithms=algorithms,
        random_state=42,
        verbose=False,
        xgboost_params={"model_type": "xgboost", "n_estimators": 20, "max_depth": 3},
        autoencoder_params={"epochs": 3, "hidden_dims": [12], "encoding_dim": 6},
        dbscan_params={"eps": 2.5, "min_samples": 5, "min_cluster_size": 5},
        gnn_params={"hidden_channels": 12, "num_heads": 1, "num_layers": 1, "epochs": 3},
    )
    detector.fit(X_train, labels=None, device=dev)

    probs, individual = detector.predict_anomaly_probability(X_test, device=dev)

    assert probs.shape == (_TEST_M_SMALL,), (
        f"ensemble output shape {probs.shape} != ({_TEST_M_SMALL},)"
    )
    for name, arr in individual.items():
        arr = np.asarray(arr)
        assert arr.shape == (_TEST_M_SMALL,), (
            f"algorithm '{name}' individual shape {arr.shape} != ({_TEST_M_SMALL},)"
        )
        assert np.isfinite(arr).all(), f"algorithm '{name}' contains NaN/Inf"


@pytest.mark.parametrize("algorithms", [
    ["isolation_forest"],
    ["autoencoder"],
    ["dbscan"],
    ["xgboost"],
    ["gnn"],
    ["isolation_forest", "autoencoder", "dbscan", "xgboost", "gnn"],
])
def test_ensemble_predict_shape_when_test_larger_than_train(algorithms):
    """Train on N rows, predict on M > N rows. All outputs must be (M,).

    This is the harder case and previously tripped HDBSCAN (training-length
    outlier_scores_ reused directly) and GNN when NeighborLoader produced
    per-node instead of per-claim probabilities.
    """
    train_df, feats = _make_dataset(_TRAIN_N, seed=3)
    test_df, _ = _make_dataset(_TEST_M_LARGE, seed=4)

    X_train = train_df[feats].values
    X_test = test_df[feats].values
    dev = _auto_device()

    detector = CombinedAnomalyDetector(
        algorithms=algorithms,
        random_state=43,
        verbose=False,
        xgboost_params={"model_type": "xgboost", "n_estimators": 20, "max_depth": 3},
        autoencoder_params={"epochs": 3, "hidden_dims": [12], "encoding_dim": 6},
        dbscan_params={"eps": 2.5, "min_samples": 5, "min_cluster_size": 5},
        gnn_params={"hidden_channels": 12, "num_heads": 1, "num_layers": 1, "epochs": 3},
    )
    detector.fit(X_train, labels=None, device=dev)

    probs, individual = detector.predict_anomaly_probability(X_test, device=dev)

    assert probs.shape == (_TEST_M_LARGE,), (
        f"ensemble output shape {probs.shape} != ({_TEST_M_LARGE},)"
    )
    for name, arr in individual.items():
        arr = np.asarray(arr)
        assert arr.shape == (_TEST_M_LARGE,), (
            f"algorithm '{name}' individual shape {arr.shape} != ({_TEST_M_LARGE},)"
        )
        assert np.isfinite(arr).all(), f"algorithm '{name}' contains NaN/Inf"


def test_dbscan_core_samples_cache_is_used_for_mismatched_inference_shape():
    """When dbscan core samples are cached at fit-time, inference with a
    different-length dataset MUST still produce an output of length M (not N).
    """
    train_df, feats = _make_dataset(_TRAIN_N, seed=5)
    test_df, _ = _make_dataset(_TEST_M_LARGE, seed=6)

    X_train = train_df[feats].values
    X_test = test_df[feats].values
    dev = _auto_device()

    detector = CombinedAnomalyDetector(
        algorithms=["dbscan"],
        random_state=44,
        verbose=False,
        dbscan_params={"eps": 2.0, "min_samples": 5, "min_cluster_size": 5},
    )
    detector.fit(X_train, labels=None, device=dev)

    # Sanity: the detector must have cached core samples for this test to
    # exercise the intended code path. If it does not, bump eps/min_samples.
    assert getattr(detector, "_dbscan_core_samples", None) is not None, (
        "HDBSCAN/DBSCAN did not cache any core samples; cannot validate "
        "mismatched-shape inference path."
    )

    probs, individual = detector.predict_anomaly_probability(X_test, device=dev)

    assert probs.shape == (_TEST_M_LARGE,)
    dbscan_arr = np.asarray(individual["dbscan"])
    assert dbscan_arr.shape == (_TEST_M_LARGE,)
    # Scores must be in [0, 1] even for mismatched-inference fallback.
    assert (dbscan_arr >= 0.0).all() and (dbscan_arr <= 1.0).all()


def test_gnn_shape_guard_triggers_without_crash_for_obviously_mismatched():
    """If we hand-craft a situation where the GNN reports the wrong length,
    the shape guard must emit zeros of the correct length instead of crashing
    or silently returning a length-mismatched vector.

    This test requires a real PyTorch installation — skip cleanly when
    PyTorch is unavailable (the TORCH_AVAILABLE shim in model.py replaces
    ``torch`` with a dummy object that does not expose ``torch.nn``).
    """
    from model import TORCH_AVAILABLE

    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch (TORCH_AVAILABLE=False) is not installed; skipping GNN mock test.")

    import torch
    from model import InsuranceAnomalyGNNModel  # noqa: F401 — imported for symbol availability

    train_df, feats = _make_dataset(1000, n_features=8, seed=7)  # big enough to not skip GNN
    test_df, _ = _make_dataset(_TEST_M_SMALL, n_features=8, seed=8)

    X_train = train_df[feats].values
    X_test = test_df[feats].values
    dev = _auto_device()

    detector = CombinedAnomalyDetector(
        algorithms=["gnn"],
        random_state=45,
        verbose=False,
        gnn_params={"hidden_channels": 8, "num_heads": 1, "num_layers": 1, "epochs": 2},
    )
    detector.fit(X_train, labels=None, device=dev)

    # Replace the (potentially correctly-sized) trained GNN with a model that
    # returns an intentionally too-large output vector so we exercise the
    # new shape guard deterministically on every run.
    class _SizedMockGNN(torch.nn.Module):
        def __init__(self, n_out_nodes: int, feat_dim: int = 8):
            super().__init__()
            self._n_out = n_out_nodes
            self.fc = torch.nn.Linear(feat_dim, 2)

        def forward(self, x, edge_index, batch=None, edge_attr=None):
            # Ignore actual graph — just emit exactly _n_out rows every call.
            dummy = torch.zeros(self._n_out, x.shape[1], dtype=x.dtype, device=x.device)
            return self.fc(dummy)

    detector.gnn_model = _SizedMockGNN(n_out_nodes=len(X_test) * 4, feat_dim=8)  # 4:1 ratio
    detector.gnn_weight = 1.0

    probs, individual = detector.predict_anomaly_probability(X_test, device=dev)

    # The ensemble must still be exactly the length of X_test.
    assert probs.shape == (_TEST_M_SMALL,)
    gnn_arr = np.asarray(individual["gnn"])
    assert gnn_arr.shape == (_TEST_M_SMALL,)
