"""
Test suite for Robust Outlier Detection
"""

import pytest
import pandas as pd
import numpy as np
from robust_outlier_detector import RobustOutlierDetector


class TestRobustOutlierDetector:
    """Test robust outlier detection methods."""

    def test_detect_outliers_mad(self):
        """Test MAD-based outlier detection."""
        detector = RobustOutlierDetector()

        # Create test data with clear outliers
        data = pd.Series([10, 12, 11, 13, 10, 12, 11, 100, 10, 12])
        outliers = detector.detect_outliers_mad(data, threshold=3.0)

        # 100 should be detected as outlier
        assert outliers.iloc[7] == True
        # Others should not be outliers
        assert outliers.iloc[0] == False
        assert outliers.iloc[1] == False

    def test_detect_outliers_iqr(self):
        """Test IQR-based outlier detection."""
        detector = RobustOutlierDetector()

        # Create test data with clear outliers
        data = pd.Series([10, 12, 11, 13, 10, 12, 11, 100, 10, 12])
        outliers = detector.detect_outliers_iqr(data, iqr_multiplier=1.5)

        # 100 should be detected as outlier
        assert outliers.iloc[7] == True
        # Others should not be outliers
        assert outliers.iloc[0] == False

    def test_detect_outliers_empty_series(self):
        """Test outlier detection with empty series."""
        detector = RobustOutlierDetector()

        data = pd.Series([])
        outliers = detector.detect_outliers_mad(data)

        assert len(outliers) == 0

    def test_detect_outliers_small_series(self):
        """Test outlier detection with small series (less than 3 items)."""
        detector = RobustOutlierDetector()

        data = pd.Series([10, 12])
        outliers = detector.detect_outliers_mad(data)

        # Should return False for both (insufficient data)
        assert outliers.iloc[0] == False
        assert outliers.iloc[1] == False

    def test_calculate_outlier_score(self):
        """Test outlier score calculation."""
        detector = RobustOutlierDetector()

        data = pd.Series([10, 12, 11, 13, 10, 12, 11, 100, 10, 12])
        scores = detector.calculate_outlier_score(data, method="mad")

        # Outlier should have higher score
        assert scores.iloc[7] > scores.iloc[0]
        # Scores should be between 0 and 1
        assert all(scores >= 0.0)
        assert all(scores <= 1.0)

    def test_get_method_statistics(self):
        """Test method statistics calculation."""
        detector = RobustOutlierDetector()

        data = pd.Series([10, 12, 11, 13, 10, 12, 11, 100, 10, 12])
        stats = detector.get_method_statistics(data, method="mad")

        assert "count" in stats
        assert "mean" in stats
        assert "median" in stats
        assert "mad" in stats
        assert stats["count"] == 10

    def test_no_outliers(self):
        """Test with data that has no outliers."""
        detector = RobustOutlierDetector()

        data = pd.Series([10, 12, 11, 13, 10, 12, 11, 10, 12, 11])
        outliers = detector.detect_outliers_mad(data, threshold=3.0)

        # None should be outliers
        assert not outliers.any()

    def test_multiple_outliers(self):
        """Test detection of multiple outliers."""
        detector = RobustOutlierDetector()

        data = pd.Series([10, 12, 11, 100, 10, 12, 11, 200, 10, 12])
        outliers = detector.detect_outliers_mad(data, threshold=3.0)

        # Both 100 and 200 should be outliers
        assert outliers.iloc[3] == True
        assert outliers.iloc[7] == True
