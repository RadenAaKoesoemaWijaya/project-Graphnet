"""
Test suite for Whitelist Manager
"""

import pytest
import pandas as pd
from whitelist_manager import WhitelistManager


class TestWhitelistManager:
    """Test whitelist manager functionality."""

    def test_chronic_condition_whitelist(self):
        """Test chronic condition whitelist functionality."""
        manager = WhitelistManager()

        # Test diabetes (E11)
        is_legitimate, reason = manager.is_legitimate_repeat(
            diagnosis_code="E11",
            service_code="BT001",
            patient_id="PAT001",
            time_gap_days=15
        )

        assert is_legitimate == True
        assert "Type 2 Diabetes" in reason

    def test_chronic_condition_outside_frequency(self):
        """Test chronic condition outside expected frequency."""
        manager = WhitelistManager()

        # Test diabetes with gap > 30 days
        is_legitimate, reason = manager.is_legitimate_repeat(
            diagnosis_code="E11",
            service_code="BT001",
            patient_id="PAT001",
            time_gap_days=45
        )

        # Should still be legitimate but with different reason
        assert is_legitimate == True

    def test_non_chronic_condition(self):
        """Test non-chronic condition not whitelisted."""
        manager = WhitelistManager()

        # Test non-chronic condition
        is_legitimate, reason = manager.is_legitimate_repeat(
            diagnosis_code="Z00",  # General examination
            service_code="CONS001",
            patient_id="PAT001",
            time_gap_days=15
        )

        assert is_legitimate == False
        assert reason == ""

    def test_add_provider_exception(self):
        """Test adding provider volume exception."""
        manager = WhitelistManager()

        manager.add_provider_exception(
            provider_id="PROV001",
            max_daily_claims=50,
            reason="High-volume tertiary hospital"
        )

        is_allowed, reason = manager.check_provider_capacity_exception(
            provider_id="PROV001",
            daily_claim_count=30
        )

        assert is_allowed == True
        assert "volume exception" in reason.lower()

    def test_provider_exception_exceeded(self):
        """Test provider exception when exceeded."""
        manager = WhitelistManager()

        manager.add_provider_exception(
            provider_id="PROV001",
            max_daily_claims=50,
            reason="High-volume tertiary hospital"
        )

        is_allowed, reason = manager.check_provider_capacity_exception(
            provider_id="PROV001",
            daily_claim_count=60
        )

        assert is_allowed == False

    def test_no_provider_exception(self):
        """Test without provider exception."""
        manager = WhitelistManager()

        is_allowed, reason = manager.check_provider_capacity_exception(
            provider_id="PROV999",
            daily_claim_count=30
        )

        assert is_allowed == False
        assert reason == ""

    def test_add_service_combination_whitelist(self):
        """Test adding service combination whitelist."""
        manager = WhitelistManager()

        manager.add_service_combination_whitelist(
            diagnosis_prefix="I10",
            service_code="BT001",
            max_repeat_days=30,
            reason="Hypertension monitoring"
        )

        is_allowed, reason = manager.check_service_combination_whitelist(
            diagnosis_code="I10",
            service_code="BT001",
            time_gap_days=15
        )

        assert is_allowed == True

    def test_service_combination_exceeded(self):
        """Test service combination when exceeded max days."""
        manager = WhitelistManager()

        manager.add_service_combination_whitelist(
            diagnosis_prefix="I10",
            service_code="BT001",
            max_repeat_days=30,
            reason="Hypertension monitoring"
        )

        is_allowed, reason = manager.check_service_combination_whitelist(
            diagnosis_code="I10",
            service_code="BT001",
            time_gap_days=45
        )

        assert is_allowed == False
        assert "exceeds max repeat days" in reason.lower()

    def test_add_patient_history_exception(self):
        """Test adding patient history exception."""
        manager = WhitelistManager()

        manager.add_patient_history_exception(
            patient_id="PAT001",
            diagnosis_prefix="Z00",  # Non-chronic condition
            max_gap_days=14,
            reason="Patient history exception"
        )

        is_legitimate, reason = manager.is_legitimate_repeat(
            diagnosis_code="Z00.1",
            service_code="CONS001",
            patient_id="PAT001",
            time_gap_days=10
        )

        assert is_legitimate == True
        assert "Patient history exception" in reason

    def test_get_whitelist_statistics(self):
        """Test getting whitelist statistics."""
        manager = WhitelistManager()

        stats = manager.get_whitelist_statistics()

        assert "chronic_conditions" in stats
        assert "provider_exceptions" in stats
        assert "service_combinations" in stats
        assert "patient_exceptions" in stats

        # Should have chronic conditions by default
        assert stats["chronic_conditions"] > 0

    def test_empty_diagnosis_code(self):
        """Test with empty diagnosis code."""
        manager = WhitelistManager()

        is_legitimate, reason = manager.is_legitimate_repeat(
            diagnosis_code="",
            service_code="CONS001",
            patient_id="PAT001",
            time_gap_days=15
        )

        assert is_legitimate == False
        assert reason == ""
