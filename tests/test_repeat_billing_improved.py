"""
Test suite for improved Repeat Billing Detector with adaptive threshold and whitelist
"""

import pytest
import pandas as pd
import numpy as np
from repeat_billing_detector import RepeatBillingDetector
from whitelist_manager import WhitelistManager


class TestRepeatBillingDetectorImproved:
    """Test improved repeat billing detector."""

    def test_detect_repeat_claims_basic(self):
        """Test basic repeat billing detection."""
        detector = RepeatBillingDetector(
            temporal_window_days=30,
            fuzzy_threshold=0.85,
            use_adaptive_threshold=False
        )

        # Create test data with repeat billing
        data = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-01",
                "amount": 500000,
                "diagnosis_code": "Z00"
            },
            {
                "claim_id": "CLM002",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-15",
                "amount": 500000,
                "diagnosis_code": "Z00"
            }
        ])

        result = detector.detect_repeat_claims(data)

        assert len(result) == 1
        assert result.iloc[0]["first_claim_id"] == "CLM001"
        assert result.iloc[0]["repeat_claim_id"] == "CLM002"
        assert result.iloc[0]["time_gap_days"] == 14

    def test_detect_repeat_claims_outside_window(self):
        """Test repeat billing outside temporal window."""
        detector = RepeatBillingDetector(
            temporal_window_days=30,
            fuzzy_threshold=0.85,
            use_adaptive_threshold=False
        )

        # Create test data with repeat billing outside window
        data = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-01",
                "amount": 500000,
                "diagnosis_code": "Z00"
            },
            {
                "claim_id": "CLM002",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-02-15",
                "amount": 500000,
                "diagnosis_code": "Z00"
            }
        ])

        result = detector.detect_repeat_claims(data)

        # Should not detect (outside 30-day window)
        assert len(result) == 0

    def test_adaptive_threshold_disabled(self):
        """Test with adaptive threshold disabled."""
        detector = RepeatBillingDetector(
            temporal_window_days=30,
            fuzzy_threshold=0.85,
            use_adaptive_threshold=False
        )

        # Create test data
        data = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-01",
                "amount": 500000,
                "diagnosis_code": "Z00"
            },
            {
                "claim_id": "CLM002",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-15",
                "amount": 500000,
                "diagnosis_code": "Z00"
            }
        ])

        result = detector.detect_repeat_claims(data)

        # Should detect with fixed threshold
        assert len(result) == 1

    def test_whitelist_chronic_condition(self):
        """Test whitelist for chronic conditions."""
        whitelist_mgr = WhitelistManager()
        detector = RepeatBillingDetector(
            temporal_window_days=30,
            fuzzy_threshold=0.85,
            use_adaptive_threshold=False,
            whitelist_manager=whitelist_mgr
        )

        # Create test data with diabetes (chronic condition)
        data = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "BT001",
                "billing_date": "2024-01-01",
                "amount": 500000,
                "diagnosis_code": "E11"
            },
            {
                "claim_id": "CLM002",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "BT001",
                "billing_date": "2024-01-15",
                "amount": 500000,
                "diagnosis_code": "E11"
            }
        ])

        result = detector.detect_repeat_claims(data)

        # Should not flag (whitelisted chronic condition)
        assert len(result) == 0

    def test_whitelist_disabled(self):
        """Test with whitelist disabled (None)."""
        detector = RepeatBillingDetector(
            temporal_window_days=30,
            fuzzy_threshold=0.85,
            use_adaptive_threshold=False,
            whitelist_manager=None
        )

        # Create test data with diabetes
        data = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "BT001",
                "billing_date": "2024-01-01",
                "amount": 500000,
                "diagnosis_code": "E11"
            },
            {
                "claim_id": "CLM002",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "BT001",
                "billing_date": "2024-01-15",
                "amount": 500000,
                "diagnosis_code": "E11"
            }
        ])

        result = detector.detect_repeat_claims(data)

        # Should flag (no whitelist)
        assert len(result) == 1

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        detector = RepeatBillingDetector()

        data = pd.DataFrame()
        result = detector.detect_repeat_claims(data)

        assert len(result) == 0
        assert "first_claim_id" in result.columns

    def test_missing_columns(self):
        """Test with missing required columns."""
        detector = RepeatBillingDetector()

        # Missing amount column
        data = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-01"
            }
        ])

        with pytest.raises(ValueError, match="missing required columns"):
            detector.detect_repeat_claims(data)

    def test_amount_variance(self):
        """Test amount variance calculation."""
        detector = RepeatBillingDetector(
            temporal_window_days=30,
            fuzzy_threshold=0.85,
            amount_variance_pct=20.0,  # Allow 20% variance
            use_adaptive_threshold=False
        )

        # Create test data with different amounts
        data = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-01",
                "amount": 500000,
                "diagnosis_code": "Z00"
            },
            {
                "claim_id": "CLM002",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-15",
                "amount": 550000,  # 10% variance (within 20% threshold)
                "diagnosis_code": "Z00"
            }
        ])

        result = detector.detect_repeat_claims(data)

        # Should still detect with acceptable variance
        assert len(result) == 1
        assert result.iloc[0]["amount_variance_pct"] <= 20.0

    def test_risk_score_calculation(self):
        """Test risk score calculation."""
        detector = RepeatBillingDetector(
            temporal_window_days=30,
            fuzzy_threshold=0.85,
            use_adaptive_threshold=False
        )

        data = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-01",
                "amount": 500000,
                "diagnosis_code": "Z00"
            },
            {
                "claim_id": "CLM002",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-15",
                "amount": 500000,
                "diagnosis_code": "Z00"
            }
        ])

        result = detector.detect_repeat_claims(data)

        # Risk score should be between 0 and 1
        assert 0.0 <= result.iloc[0]["risk_score"] <= 1.0
        # Similarity score should be high
        assert result.iloc[0]["similarity_score"] >= 0.85

    def test_verbose_logging(self):
        """Test verbose logging."""
        detector = RepeatBillingDetector(
            temporal_window_days=30,
            fuzzy_threshold=0.85,
            use_adaptive_threshold=False,
            verbose=True
        )

        data = pd.DataFrame([
            {
                "claim_id": "CLM001",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-01",
                "amount": 500000,
                "diagnosis_code": "Z00"
            },
            {
                "claim_id": "CLM002",
                "patient_id": "PAT001",
                "provider_id": "PROV001",
                "service_code": "CONS001",
                "billing_date": "2024-01-15",
                "amount": 500000,
                "diagnosis_code": "Z00"
            }
        ])

        result = detector.detect_repeat_claims(data)

        # Should still work with verbose enabled
        assert len(result) == 1
