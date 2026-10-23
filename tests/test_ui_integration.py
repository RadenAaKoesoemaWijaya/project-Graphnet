"""
Test suite for UI Integration of Phase 1 & 2 Features
"""

import pytest
import pandas as pd
import numpy as np


class TestUIIntegration:
    """Test UI integration for Phase 1 & 2 features."""

    def test_detection_page_import(self):
        """Test that detection page can be imported."""
        try:
            from ui.pages.detection import show_detection_page
            assert show_detection_page is not None
        except ImportError as e:
            pytest.fail(f"Failed to import detection page: {e}")

    def test_settings_page_import(self):
        """Test that settings page can be imported."""
        try:
            from ui.pages.settings import show_settings_page
            assert show_settings_page is not None
        except ImportError as e:
            pytest.fail(f"Failed to import settings page: {e}")

    def test_pipeline_with_phase2_enabled(self):
        """Test pipeline with Phase 2 enabled."""
        from fraud_risk_pipeline import enrich_claims_with_business_risk_features

        df = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "diagnosis_code": "E11",
                "billing_date": "2024-01-15",
                "service_date": "2024-01-15",
                "billed_amount": 500000,
                "paid_amount": 450000,
                "allowed_amount": 450000,
                "claim_status": "Settled",
                "patient_age": 45,
                "length_of_stay": 0,
                "quantity": 1
            }
        ])

        result, summary = enrich_claims_with_business_risk_features(df, enable_phase2=True)

        assert result is not None
        assert not result.empty
        assert summary["total_claims"] == 1

    def test_pipeline_with_phase2_disabled(self):
        """Test pipeline with Phase 2 disabled."""
        from fraud_risk_pipeline import enrich_claims_with_business_risk_features

        df = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "diagnosis_code": "E11",
                "billing_date": "2024-01-15",
                "service_date": "2024-01-15",
                "billed_amount": 500000,
                "paid_amount": 450000,
                "allowed_amount": 450000,
                "claim_status": "Settled",
                "patient_age": 45,
                "length_of_stay": 0,
                "quantity": 1
            }
        ])

        result, summary = enrich_claims_with_business_risk_features(df, enable_phase2=False)

        assert result is not None
        assert not result.empty
        assert summary["total_claims"] == 1

    def test_phase2_columns_present(self):
        """Test that Phase 2 columns are present in results."""
        from fraud_risk_pipeline import enrich_claims_with_business_risk_features

        df = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "diagnosis_code": "E11",
                "billing_date": "2024-01-15",
                "service_date": "2024-01-15",
                "billed_amount": 500000,
                "paid_amount": 450000,
                "allowed_amount": 450000,
                "claim_status": "Settled",
                "patient_age": 45,
                "length_of_stay": 0,
                "quantity": 1
            }
        ])

        result, _ = enrich_claims_with_business_risk_features(df, enable_phase2=True)

        # Check Phase 2 columns
        assert "composite_risk_score" in result.columns
        assert "context_suppressed" in result.columns
        assert "final_business_risk_flag" in result.columns

    def test_whitelist_manager_integration(self):
        """Test whitelist manager integration."""
        try:
            from whitelist_manager import WhitelistManager
            whitelist_mgr = WhitelistManager()

            stats = whitelist_mgr.get_whitelist_statistics()

            assert stats is not None
            assert "chronic_conditions" in stats
        except ImportError as e:
            pytest.skip(f"Whitelist manager not available: {e}")

    def test_adaptive_threshold_engine_integration(self):
        """Test adaptive threshold engine integration."""
        try:
            from adaptive_threshold_engine import AdaptiveThresholdEngine
            engine = AdaptiveThresholdEngine()

            df = pd.DataFrame([
                {"amount": 10, "provider_id": "P1"},
                {"amount": 12, "provider_id": "P1"},
                {"amount": 11, "provider_id": "P1"},
            ])

            result = engine.calculate_dynamic_threshold(df, "amount", method="iqr")

            assert result is not None
            assert "upper_threshold" in result
        except ImportError as e:
            pytest.skip(f"Adaptive threshold engine not available: {e}")

    def test_context_aware_risk_scorer_integration(self):
        """Test context-aware risk scorer integration."""
        try:
            from context_aware_risk_scorer import ContextAwareRiskScorer
            scorer = ContextAwareRiskScorer()

            claim = pd.Series({
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001"
            })

            result = scorer.calculate_composite_risk_score(claim, {}, {})

            assert result is not None
            assert "composite_risk_score" in result
        except ImportError as e:
            pytest.skip(f"Context-aware risk scorer not available: {e}")

    def test_historical_baseline_system_integration(self):
        """Test historical baseline system integration."""
        try:
            from historical_baseline_system import HistoricalBaselineSystem
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                system = HistoricalBaselineSystem(storage_path=tmpdir)

                df = pd.DataFrame([
                    {"amount": 500000, "provider_id": "PROV001", "patient_id": "PAT001"}
                ])

                system.update_baseline(df)

                summary = system.get_baseline_summary()

                assert summary is not None
                assert "last_updated" in summary
        except ImportError as e:
            pytest.skip(f"Historical baseline system not available: {e}")

    def test_empty_dataframe_handling(self):
        """Test empty DataFrame handling with Phase 2."""
        from fraud_risk_pipeline import enrich_claims_with_business_risk_features

        df = pd.DataFrame()
        result, summary = enrich_claims_with_business_risk_features(df, enable_phase2=True)

        assert result.empty
        assert summary["total_claims"] == 0

    def test_missing_columns_handling(self):
        """Test missing columns handling with Phase 2."""
        from fraud_risk_pipeline import enrich_claims_with_business_risk_features

        df = pd.DataFrame([{"other_column": 10}])
        result, summary = enrich_claims_with_business_risk_features(df, enable_phase2=True)

        # Should handle gracefully
        assert result is not None
