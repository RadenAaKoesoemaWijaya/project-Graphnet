import pytest
import numpy as np
import pandas as pd
from model import OptunaEnsembleOptimizer, CombinedAnomalyDetector, DynamicWeightOptimizer
from model_explainer import ConceptDriftDetector, AdaptiveLearningManager

def test_optuna_ensemble_optimizer_simplex_constraint():
    """Verify that OptunaEnsembleOptimizer returns weights summing to 1.0 and non-negative."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    
    # Generate mock scores with varying precision/recall
    individual_scores = {
        'isolation': np.clip(y_true * 0.7 + rng.normal(0.2, 0.1, n), 0, 1),
        'autoencoder': np.clip(y_true * 0.8 + rng.normal(0.3, 0.15, n), 0, 1),
        'xgboost': np.clip(y_true * 0.9 + rng.normal(0.1, 0.05, n), 0, 1),
    }
    
    optimizer = OptunaEnsembleOptimizer(n_trials=10, timeout=30, lambda_fpr=0.5, cv_folds=3, random_state=42)
    result = optimizer.optimize(individual_scores, y_true=y_true)
    
    assert result['status'] == 'success'
    weights = result['weights']
    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-5)
    for k, v in weights.items():
        assert v >= 0.0, f"Weight for {k} is negative: {v}"

def test_optuna_ensemble_fpr_minimization():
    """Verify that Optuna Ensemble tuning prioritizes high precision model to reduce FPR."""
    rng = np.random.default_rng(123)
    n = 300
    y_true = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    
    # Model A has high FP (noisy on negative samples)
    score_a = np.where(y_true == 1, 0.8, 0.45 + rng.uniform(0, 0.3, n))
    # Model B is clean (low FP on negative samples)
    score_b = np.where(y_true == 1, 0.75, 0.1 + rng.uniform(0, 0.2, n))
    
    individual_scores = {
        'noisy_algo': score_a,
        'clean_algo': score_b
    }
    
    optimizer = OptunaEnsembleOptimizer(n_trials=15, timeout=30, lambda_fpr=1.0, beta=0.5, random_state=42)
    result = optimizer.optimize(individual_scores, y_true=y_true)
    
    weights = result['weights']
    # Clean algorithm should receive higher weight to reduce false positives
    assert weights['clean_algo'] > weights['noisy_algo']

def test_combined_anomaly_detector_ensemble_optimization():
    """Verify integration of optimize_ensemble_weights inside CombinedAnomalyDetector."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    # Add anomalies
    X[:10] += 5.0
    labels = np.zeros(100, dtype=int)
    labels[:10] = 1
    
    detector = CombinedAnomalyDetector(
        algorithms=['isolation_forest', 'xgboost'],
        use_dynamic_weights=True
    )
    
    detector.fit(
        X, labels=labels,
        optimize_ensemble_weights=True,
        optuna_n_trials=5,
        optuna_timeout=15
    )
    
    assert detector.weight_optimization_results is not None
    assert 'weights' in detector.weight_optimization_results
    assert np.isclose(detector.isolation_weight + detector.xgboost_weight, 1.0, atol=1e-4)

def test_concept_drift_detector_auto_retrain_trigger():
    """Verify ConceptDriftDetector triggers automated retraining on significant distribution shift."""
    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame(rng.normal(0, 1, (200, 4)), columns=[f'feat_{i}' for i in range(4)])
    
    drift_detector = ConceptDriftDetector(reference_data=ref_df, threshold=0.05)
    
    # Create shifted data (covariate shift)
    shifted_df = pd.DataFrame(rng.normal(5, 2, (200, 4)), columns=[f'feat_{i}' for i in range(4)])
    
    callback_executed = []
    def mock_retrain(data, **kwargs):
        callback_executed.append(True)
        return {'status': 'retrained_successfully'}
    
    outcome = drift_detector.check_and_trigger_retraining(
        new_data=shifted_df,
        min_drift_feature_pct=0.25,
        retrain_callback=mock_retrain
    )
    
    assert outcome['drift_detected'] is True
    assert outcome['retraining_triggered'] is True
    assert len(callback_executed) == 1
    assert outcome['retraining_result'] == {'status': 'retrained_successfully'}

def test_champion_challenger_quality_gate():
    """Verify Champion-Challenger Quality Gate logic in AdaptiveLearningManager."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 4))
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    
    champ = CombinedAnomalyDetector(algorithms=['isolation_forest'])
    champ.fit(X, labels=y)
    
    chal = CombinedAnomalyDetector(algorithms=['isolation_forest', 'xgboost'])
    chal.fit(X, labels=y)
    
    manager = AdaptiveLearningManager(detector=champ)
    gate_res = manager.evaluate_champion_challenger(
        champion_detector=champ,
        challenger_detector=chal,
        validation_data=X,
        validation_labels=y
    )
    
    assert 'decision' in gate_res
    assert 'promoted' in gate_res
    assert 'champion_metrics' in gate_res
    assert 'challenger_metrics' in gate_res
