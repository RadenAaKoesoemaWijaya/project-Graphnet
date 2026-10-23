"""
Test suite for Historical Baseline System
"""

import pytest
import pandas as pd
import numpy as np
from historical_baseline_system import HistoricalBaselineSystem
import tempfile
import shutil


class TestHistoricalBaselineSystem:
    """Test historical baseline system functionality."""

    def test_update_baseline(self):
        """Test updating baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            df = pd.DataFrame([
                {
                    "claim_id": "CLM001",
                    "patient_id": "PAT001",
                    "provider_id": "PROV001",
                    "service_code": "CONS001",
                    "amount": 500000,
                    "billing_date": "2024-01-15"
                },
                {
                    "claim_id": "CLM002",
                    "patient_id": "PAT002",
                    "provider_id": "PROV002",
                    "service_code": "BT001",
                    "amount": 600000,
                    "billing_date": "2024-01-16"
                }
            ])

            system.update_baseline(df)

            assert "provider" in system.baselines
            assert "cost" in system.baselines
            assert "seasonal" in system.baselines
            assert "patient" in system.baselines

    def test_calculate_provider_baseline(self):
        """Test provider baseline calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            df = pd.DataFrame([
                {"provider_id": "PROV001", "patient_id": "PAT001", "amount": 500000, "billing_date": "2024-01-15"},
                {"provider_id": "PROV001", "patient_id": "PAT002", "amount": 600000, "billing_date": "2024-01-16"},
                {"provider_id": "PROV002", "patient_id": "PAT003", "amount": 700000, "billing_date": "2024-01-17"},
            ])

            system.update_baseline(df)

            provider_baseline = system.baselines["provider"]

            assert "PROV001" in provider_baseline
            assert "PROV002" in provider_baseline
            assert provider_baseline["PROV001"]["total_claims"] == 2
            assert provider_baseline["PROV002"]["total_claims"] == 1

    def test_calculate_cost_baseline(self):
        """Test cost baseline calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            df = pd.DataFrame([
                {"amount": 500000},
                {"amount": 600000},
                {"amount": 700000},
                {"amount": 1000000},  # Outlier
            ])

            system.update_baseline(df)

            cost_baseline = system.baselines["cost"]["global"]

            assert "mean" in cost_baseline
            assert "median" in cost_baseline
            assert "std" in cost_baseline
            assert cost_baseline["mean"] > 0

    def test_calculate_seasonal_baseline(self):
        """Test seasonal baseline calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            df = pd.DataFrame([
                {"billing_date": "2024-01-15", "amount": 500000},
                {"billing_date": "2024-06-15", "amount": 600000},
                {"billing_date": "2024-12-15", "amount": 700000},
            ])

            system.update_baseline(df)

            seasonal_baseline = system.baselines["seasonal"]

            assert "by_month" in seasonal_baseline
            assert len(seasonal_baseline["by_month"]) > 0

    def test_get_baseline_value(self):
        """Test retrieving baseline value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            df = pd.DataFrame([
                {"amount": 500000},
                {"amount": 600000},
                {"amount": 700000},
            ])

            system.update_baseline(df)

            mean = system.get_baseline_value("cost", "global", "mean")

            assert mean is not None
            assert mean > 0

    def test_get_baseline_value_default(self):
        """Test retrieving baseline value with default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            value = system.get_baseline_value("cost", "nonexistent", "mean", default=999)

            assert value == 999

    def test_detect_concept_drift(self):
        """Test concept drift detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            # Create baseline
            baseline_df = pd.DataFrame([
                {"amount": 500000},
                {"amount": 600000},
                {"amount": 700000},
            ])

            system.update_baseline(baseline_df)

            # Create current data with drift
            current_df = pd.DataFrame([
                {"amount": 5000000},  # 10x higher
                {"amount": 6000000},
                {"amount": 7000000},
            ])

            drift_results = system.detect_concept_drift(current_df, threshold=0.2)

            assert "cost_mean" in drift_results
            assert drift_results["cost_mean"] == True  # Should detect drift

    def test_detect_no_concept_drift(self):
        """Test no concept drift detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            baseline_df = pd.DataFrame([
                {"amount": 500000},
                {"amount": 600000},
                {"amount": 700000},
            ])

            system.update_baseline(baseline_df)

            # Similar data
            current_df = pd.DataFrame([
                {"amount": 510000},
                {"amount": 610000},
                {"amount": 710000},
            ])

            drift_results = system.detect_concept_drift(current_df, threshold=0.2)

            assert drift_results["cost_mean"] == False  # Should not detect drift

    def test_save_and_load_baselines(self):
        """Test saving and loading baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system1 = HistoricalBaselineSystem(storage_path=tmpdir)

            df = pd.DataFrame([
                {"amount": 500000, "provider_id": "PROV001", "patient_id": "PAT001"},
            ])

            system1.update_baseline(df)

            # Create new system instance and load
            system2 = HistoricalBaselineSystem(storage_path=tmpdir)
            system2.load_baselines()

            assert "cost" in system2.baselines
            assert "provider" in system2.baselines

    def test_get_baseline_summary(self):
        """Test getting baseline summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            df = pd.DataFrame([
                {"amount": 500000, "provider_id": "PROV001", "patient_id": "PAT001", "service_code": "CONS001"},
            ])

            system.update_baseline(df)

            summary = system.get_baseline_summary()

            assert "last_updated" in summary
            assert "record_count" in summary
            assert summary["record_count"] == 1

    def test_clear_baselines(self):
        """Test clearing baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            df = pd.DataFrame([{"amount": 500000}])
            system.update_baseline(df)

            assert len(system.baselines) > 0

            system.clear_baselines()

            assert len(system.baselines) == 0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            df = pd.DataFrame()
            system.update_baseline(df)

            # Should not crash, baselines should remain empty
            assert len(system.baselines) == 0

    def test_missing_columns(self):
        """Test with missing required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = HistoricalBaselineSystem(storage_path=tmpdir)

            df = pd.DataFrame([{"other_column": 10}])
            system.update_baseline(df)

            # Should handle gracefully
            assert "metadata" in system.baselines
