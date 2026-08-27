import pytest
import pandas as pd
import numpy as np
from preprocessing_optimized import (
    apply_select_k_best,
    apply_mutual_info_selection,
    apply_tree_based_selection,
    apply_pca_reduction,
    filter_correlated_features,
    filter_low_variance_features
)

@pytest.fixture
def sample_claims_data():
    np.random.seed(42)
    n_samples = 100
    
    # Create synthetic features with known relationships
    billed_amount = np.random.uniform(100, 10000, n_samples)
    # Perfectly correlated feature with billed_amount
    billed_amount_clone = billed_amount * 1.0001 + np.random.normal(0, 0.01, n_samples)
    # Low variance feature (almost constant)
    constant_feat = np.ones(n_samples) * 5.0
    constant_feat[0] = 5.000001
    # Normalized small scale feature with good variance (e.g. probability / ratio)
    ratio_feat = np.random.uniform(0.1, 0.9, n_samples)
    # Anomaly indicator feature
    high_risk_score = np.random.exponential(2, n_samples)
    other_feat_1 = np.random.normal(50, 10, n_samples)
    other_feat_2 = np.random.normal(10, 2, n_samples)

    df = pd.DataFrame({
        'billed_amount': billed_amount,
        'billed_amount_clone': billed_amount_clone,
        'constant_feat': constant_feat,
        'ratio_feat': ratio_feat,
        'high_risk_score': high_risk_score,
        'other_feat_1': other_feat_1,
        'other_feat_2': other_feat_2
    })
    return df

def test_apply_select_k_best_f_classif(sample_claims_data):
    feature_cols = sample_claims_data.columns.tolist()
    selected, scores_df = apply_select_k_best(
        sample_claims_data, feature_cols, k=4, score_func_name="f_classif"
    )
    assert len(selected) == 4
    assert isinstance(scores_df, pd.DataFrame)
    assert 'Feature' in scores_df.columns
    assert 'Score' in scores_df.columns
    assert len(scores_df) == len(feature_cols)

def test_apply_select_k_best_mutual_info(sample_claims_data):
    feature_cols = sample_claims_data.columns.tolist()
    selected, scores_df = apply_select_k_best(
        sample_claims_data, feature_cols, k=3, score_func_name="mutual_info_classif"
    )
    assert len(selected) == 3
    assert isinstance(scores_df, pd.DataFrame)
    assert len(scores_df) == len(feature_cols)

def test_apply_tree_based_selection(sample_claims_data):
    feature_cols = sample_claims_data.columns.tolist()
    selected, importances_df, model_name = apply_tree_based_selection(
        sample_claims_data, feature_cols, k=3
    )
    assert len(selected) == 3
    assert isinstance(importances_df, pd.DataFrame)
    assert model_name in ["LightGBM", "XGBoost", "RandomForest"]

def test_apply_pca_reduction(sample_claims_data):
    feature_cols = ['billed_amount', 'ratio_feat', 'high_risk_score', 'other_feat_1', 'other_feat_2']
    df_pca, pca_cols, explained_variance = apply_pca_reduction(
        sample_claims_data, feature_cols, n_components=3
    )
    assert len(pca_cols) == 3
    assert df_pca.shape[1] == 3
    assert len(explained_variance) == 3
    assert sum(explained_variance) > 0.0

def test_filter_correlated_features_importance_aware(sample_claims_data):
    features = ['billed_amount', 'billed_amount_clone', 'ratio_feat']
    # Give billed_amount higher score than billed_amount_clone
    scores_df = pd.DataFrame({
        'Feature': ['billed_amount', 'billed_amount_clone', 'ratio_feat'],
        'Score': [100.0, 5.0, 50.0]
    })
    kept, removed = filter_correlated_features(
        sample_claims_data, features, correlation_threshold=0.9, feature_scores_df=scores_df
    )
    # billed_amount_clone should be removed, billed_amount kept
    assert 'billed_amount' in kept
    assert 'billed_amount_clone' in removed
    assert 'ratio_feat' in kept

def test_filter_low_variance_features_scale_invariant(sample_claims_data):
    features = ['billed_amount', 'constant_feat', 'ratio_feat']
    # Constant feat should be removed, ratio_feat (0.1 to 0.9) must be kept even though its raw variance is small
    kept, removed = filter_low_variance_features(
        sample_claims_data, features, variance_threshold=0.001
    )
    assert 'constant_feat' in removed
    assert 'ratio_feat' in kept
    assert 'billed_amount' in kept
