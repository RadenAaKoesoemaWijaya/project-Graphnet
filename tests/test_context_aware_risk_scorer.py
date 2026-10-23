"""
Test suite for Context-Aware Risk Scorer
"""

import pytest
import pandas as pd
import numpy as np
from context_aware_risk_scorer import ContextAwareRiskScorer


class TestContextAwareRiskScorer:
    """Test context-aware risk scorer functionality."""

    def test_calculate_composite_risk_score(self):
        """Test composite risk score calculation."""
        scorer = ContextAwareRiskScorer()

        claim = pd.Series({
            "claim_id": "CLM001",
            "patient_id": "PAT001",
            "provider_id": "PROV001",
            "amount": 500000,
            "billing_date": "2024-01-15"
        })

        module_flags = {
            "repeat_billing": 1,
            "phantom_service": 0
        }

        module_scores = {
            "repeat_billing": 0.8,
            "phantom_service": 0.0
        }

        result = scorer.calculate_composite_risk_score(claim, module_flags, module_scores)

        assert "composite_risk_score" in result
        assert 0.0 <= result["composite_risk_score"] <= 1.0
        assert "patient_risk_factor" in result
        assert "provider_risk_factor" in result

    def test_should_suppress_flag_low_composite(self):
        """Test flag suppression for low composite score."""
        scorer = ContextAwareRiskScorer()

        should_suppress = scorer._should_suppress_flag(
            composite_score=0.2,
            patient_risk_factor=0.5,
            provider_risk_factor=0.5
        )

        assert should_suppress == True

    def test_should_suppress_flag_low_patient_risk(self):
        """Test flag suppression for low patient risk."""
        scorer = ContextAwareRiskScorer()

        should_suppress = scorer._should_suppress_flag(
            composite_score=0.4,
            patient_risk_factor=0.1,
            provider_risk_factor=0.2
        )

        assert should_suppress == True

    def test_should_not_suppress_flag_high_risk(self):
        """Test flag not suppressed for high risk."""
        scorer = ContextAwareRiskScorer()

        should_suppress = scorer._should_suppress_flag(
            composite_score=0.7,
            patient_risk_factor=0.6,
            provider_risk_factor=0.5
        )

        assert should_suppress == False

    def test_module_correlation_single_module(self):
        """Test module correlation with single module."""
        scorer = ContextAwareRiskScorer()

        module_flags = {"repeat_billing": 1}

        correlation = scorer._calculate_module_correlation(module_flags)

        assert correlation == 1.0

    def test_module_correlation_multiple_modules(self):
        """Test module correlation with multiple modules."""
        scorer = ContextAwareRiskScorer()

        module_flags = {
            "repeat_billing": 1,
            "phantom_service": 1,
            "upcoding_unbundling": 0
        }

        correlation = scorer._calculate_module_correlation(module_flags)

        assert correlation > 0.5  # Multiple modules triggered

    def test_module_correlation_high_risk_combination(self):
        """Test module correlation with high-risk combination."""
        scorer = ContextAwareRiskScorer()

        module_flags = {
            "repeat_billing": 1,
            "inflated_bill_cloning": 1
        }

        correlation = scorer._calculate_module_correlation(module_flags)

        assert correlation > 0.5  # High-risk combination

    def test_seasonal_adjustment_holiday_season(self):
        """Test seasonal adjustment for holiday season."""
        scorer = ContextAwareRiskScorer()

        claim = pd.Series({
            "billing_date": "2024-12-15"  # December - holiday season
        })

        seasonal_factor = scorer._get_seasonal_adjustment(claim)

        assert seasonal_factor >= 0.5  # Higher risk during holidays

    def test_seasonal_adjustment_neutral(self):
        """Test seasonal adjustment for neutral season."""
        scorer = ContextAwareRiskScorer()

        claim = pd.Series({
            "billing_date": "2024-05-15"  # May - neutral
        })

        seasonal_factor = scorer._get_seasonal_adjustment(claim)

        assert seasonal_factor == 0.5  # Neutral

    def test_aggregate_module_scores(self):
        """Test aggregating module scores."""
        scorer = ContextAwareRiskScorer()

        module_scores = {
            "repeat_billing": 0.8,
            "phantom_service": 0.6,
            "upcoding_unbundling": 0.9
        }

        aggregated = scorer._aggregate_module_scores(module_scores)

        assert aggregated == 0.9  # Should return max score

    def test_update_patient_history(self):
        """Test updating patient history."""
        scorer = ContextAwareRiskScorer()

        df = pd.DataFrame([
            {"patient_id": "PAT001", "amount": 500000, "billing_date": "2024-01-15"},
            {"patient_id": "PAT002", "amount": 600000, "billing_date": "2024-01-16"},
        ])

        scorer.update_patient_history(df)

        assert "PAT001" in scorer.patient_history
        assert "PAT002" in scorer.patient_history

    def test_set_provider_reputation(self):
        """Test setting provider reputation."""
        scorer = ContextAwareRiskScorer()

        scorer.set_provider_reputation("PROV001", 0.8)

        assert scorer.provider_reputation["PROV001"] == 0.8

    def test_get_context_statistics(self):
        """Test getting context statistics."""
        scorer = ContextAwareRiskScorer()

        scorer.set_provider_reputation("PROV001", 0.8)

        stats = scorer.get_context_statistics()

        assert "providers_cached" in stats
        assert stats["providers_cached"] == 1

    def test_empty_module_flags(self):
        """Test with empty module flags."""
        scorer = ContextAwareRiskScorer()

        claim = pd.Series({
            "claim_id": "CLM001",
            "patient_id": "PAT001",
            "provider_id": "PROV001"
        })

        result = scorer.calculate_composite_risk_score(claim, {}, {})

        # Should be low score due to no module triggers
        assert result["composite_risk_score"] < 0.5
        assert result["aggregate_module_score"] == 0.0

    def test_suppression_reason(self):
        """Test suppression reason generation."""
        scorer = ContextAwareRiskScorer()

        reason = scorer._get_suppression_reason(
            patient_risk_factor=0.1,
            provider_risk_factor=0.2,
            composite_score=0.2
        )

        assert "Low composite risk score" in reason or "Low patient historical risk" in reason
