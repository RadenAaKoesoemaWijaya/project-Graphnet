"""
Test suite for Adaptive Threshold Engine
"""

import pytest
import pandas as pd
import numpy as np
from adaptive_threshold_engine import AdaptiveThresholdEngine


class TestAdaptiveThresholdEngine:
    """Test adaptive threshold engine functionality."""

    def test_calculate_iqr_threshold(self):
        """Test IQR-based threshold calculation."""
        engine = AdaptiveThresholdEngine()

        # Create test data with clear outliers
        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
            {"amount": 11, "provider_id": "P1"},
            {"amount": 13, "provider_id": "P1"},
            {"amount": 100, "provider_id": "P1"},  # Outlier
        ])

        result = engine.calculate_dynamic_threshold(data, "amount", method="iqr")

        assert result["method"] == "iqr"
        assert result["upper_threshold"] > 13  # Should be above normal values
        assert result["sample_size"] == 5

    def test_calculate_zscore_threshold(self):
        """Test Z-score-based threshold calculation."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
            {"amount": 11, "provider_id": "P1"},
            {"amount": 13, "provider_id": "P1"},
            {"amount": 100, "provider_id": "P1"},
        ])

        result = engine.calculate_dynamic_threshold(data, "amount", method="zscore")

        assert result["method"] == "zscore"
        assert "mean" in result
        assert "std" in result

    def test_calculate_percentile_threshold(self):
        """Test percentile-based threshold calculation."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
            {"amount": 11, "provider_id": "P1"},
            {"amount": 13, "provider_id": "P1"},
            {"amount": 100, "provider_id": "P1"},
        ])

        result = engine.calculate_dynamic_threshold(data, "amount", method="percentile")

        assert result["method"] == "percentile"
        assert result["percentile_99"] > 0  # Should have a value
        assert result["sample_size"] == 5

    def test_calculate_mad_threshold(self):
        """Test MAD-based threshold calculation."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
            {"amount": 11, "provider_id": "P1"},
            {"amount": 13, "provider_id": "P1"},
            {"amount": 100, "provider_id": "P1"},
        ])

        result = engine.calculate_dynamic_threshold(data, "amount", method="robust_mad")

        assert result["method"] == "robust_mad"
        assert "median" in result
        assert "mad" in result

    def test_provider_specific_threshold(self):
        """Test provider-specific threshold calculation."""
        engine = AdaptiveThresholdEngine()
        engine.min_samples_for_threshold = 3  # Lower threshold for testing

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
            {"amount": 11, "provider_id": "P1"},
            {"amount": 13, "provider_id": "P1"},
            {"amount": 1000, "provider_id": "P2"},  # High-volume provider
            {"amount": 1100, "provider_id": "P2"},
            {"amount": 1050, "provider_id": "P2"},
            {"amount": 1150, "provider_id": "P2"},
        ])

        result_p1 = engine.calculate_dynamic_threshold(data, "amount", provider_id="P1", method="iqr")
        result_p2 = engine.calculate_dynamic_threshold(data, "amount", provider_id="P2", method="iqr")

        # P2 should have higher threshold due to higher amounts
        assert result_p2["upper_threshold"] >= result_p1["upper_threshold"]

    def test_insufficient_data_default(self):
        """Test default threshold when insufficient data."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
        ])  # Only 2 samples (< min_samples_for_threshold)

        result = engine.calculate_dynamic_threshold(data, "amount", method="iqr")

        assert result["is_default"] == True
        assert result["sample_size"] == 0

    def test_multi_method_threshold(self):
        """Test calculating thresholds with multiple methods."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
            {"amount": 11, "provider_id": "P1"},
            {"amount": 13, "provider_id": "P1"},
            {"amount": 100, "provider_id": "P1"},
        ])

        results = engine.calculate_multi_method_threshold(data, "amount")

        assert "iqr" in results
        assert "zscore" in results
        assert "percentile" in results
        assert "robust_mad" in results

    def test_conservative_threshold(self):
        """Test conservative threshold (highest upper threshold)."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
            {"amount": 11, "provider_id": "P1"},
            {"amount": 13, "provider_id": "P1"},
            {"amount": 100, "provider_id": "P1"},
        ])

        result = engine.get_conservative_threshold(data, "amount")

        assert result["method"] == "conservative"
        assert result["upper_threshold"] > 0

    def test_aggressive_threshold(self):
        """Test aggressive threshold (lowest upper threshold)."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
            {"amount": 11, "provider_id": "P1"},
            {"amount": 13, "provider_id": "P1"},
            {"amount": 100, "provider_id": "P1"},
        ])

        result = engine.get_aggressive_threshold(data, "amount")

        assert result["method"] == "aggressive"
        assert result["upper_threshold"] > 0

    def test_baseline_cache(self):
        """Test baseline cache functionality."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
            {"amount": 11, "provider_id": "P1"},
        ])

        engine.update_baseline_cache(data, cache_key="test")

        assert "test" in engine.baseline_cache
        assert "amount" in engine.baseline_cache["test"]["columns"]

    def test_get_baseline_value(self):
        """Test retrieving baseline value from cache."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
            {"amount": 11, "provider_id": "P1"},
        ])

        engine.update_baseline_cache(data, cache_key="test")

        median = engine.get_baseline_value("amount", "median", cache_key="test")

        assert median is not None
        assert median > 0

    def test_clear_baseline_cache(self):
        """Test clearing baseline cache."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([
            {"amount": 10, "provider_id": "P1"},
            {"amount": 12, "provider_id": "P1"},
        ])

        engine.update_baseline_cache(data, cache_key="test")
        assert "test" in engine.baseline_cache

        engine.clear_baseline_cache(cache_key="test")
        assert "test" not in engine.baseline_cache

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame()
        result = engine.calculate_dynamic_threshold(data, "amount", method="iqr")

        assert result["is_default"] == True

    def test_missing_column(self):
        """Test with missing column."""
        engine = AdaptiveThresholdEngine()

        data = pd.DataFrame([{"other_column": 10}])
        result = engine.calculate_dynamic_threshold(data, "amount", method="iqr")

        assert result["is_default"] == True
