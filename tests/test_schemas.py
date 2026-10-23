"""
Test suite for Pydantic schemas (Phase 1 - Data Validation)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestPydanticSchemas:
    """Test Pydantic schema validation"""

    def test_claim_data_valid(self):
        """Test valid claim data"""
        from schemas import ClaimData, ClaimStatus

        claim = ClaimData(
            claim_id="CLM001",
            patient_id="PAT001",
            provider_id="PROV001",
            service_code="CONS001",
            diagnosis_code="E11",
            billing_date=datetime.now() - timedelta(days=5),
            service_date=datetime.now() - timedelta(days=5),
            billed_amount=500000,
            paid_amount=450000,
            allowed_amount=450000,
            claim_status=ClaimStatus.SETTLED,
            patient_age=45,
            length_of_stay=0,
            quantity=1
        )

        assert claim.claim_id == "CLM001"
        assert claim.billed_amount == 500000
        assert claim.claim_status == ClaimStatus.SETTLED

    def test_claim_data_invalid_future_date(self):
        """Test claim data with future date (should fail)"""
        from schemas import ClaimData

        with pytest.raises(ValueError, match="Date cannot be in the future"):
            ClaimData(
                claim_id="CLM001",
                patient_id="PAT001",
                provider_id="PROV001",
                service_code="CONS001",
                billing_date=datetime.now() + timedelta(days=5),
                service_date=datetime.now() + timedelta(days=5),
                billed_amount=500000,
                paid_amount=450000,
                allowed_amount=450000,
                claim_status="Settled",
                patient_age=45,
                length_of_stay=0,
                quantity=1
            )

    def test_claim_data_invalid_paid_exceeds_billed(self):
        """Test claim data where paid amount exceeds billed (should fail)"""
        from schemas import ClaimData

        with pytest.raises(ValueError, match="Paid amount cannot exceed billed amount"):
            ClaimData(
                claim_id="CLM001",
                patient_id="PAT001",
                provider_id="PROV001",
                service_code="CONS001",
                billing_date=datetime.now() - timedelta(days=5),
                service_date=datetime.now() - timedelta(days=5),
                billed_amount=500000,
                paid_amount=600000,  # Exceeds billed
                allowed_amount=450000,
                claim_status="Settled",
                patient_age=45,
                length_of_stay=0,
                quantity=1
            )

    def test_claim_data_invalid_status(self):
        """Test claim data with invalid status (should fail)"""
        from schemas import ClaimData

        with pytest.raises(ValueError):
            ClaimData(
                claim_id="CLM001",
                patient_id="PAT001",
                provider_id="PROV001",
                service_code="CONS001",
                billing_date=datetime.now() - timedelta(days=5),
                service_date=datetime.now() - timedelta(days=5),
                billed_amount=500000,
                paid_amount=450000,
                allowed_amount=450000,
                claim_status="InvalidStatus",
                patient_age=45,
                length_of_stay=0,
                quantity=1
            )

    def test_claim_data_invalid_age(self):
        """Test claim data with invalid age (should fail)"""
        from schemas import ClaimData

        with pytest.raises(ValueError):
            ClaimData(
                claim_id="CLM001",
                patient_id="PAT001",
                provider_id="PROV001",
                service_code="CONS001",
                billing_date=datetime.now() - timedelta(days=5),
                service_date=datetime.now() - timedelta(days=5),
                billed_amount=500000,
                paid_amount=450000,
                allowed_amount=450000,
                claim_status="Settled",
                patient_age=150,  # Invalid (>120)
                length_of_stay=0,
                quantity=1
            )

    def test_claim_data_invalid_los(self):
        """Test claim data with invalid length of stay (should fail)"""
        from schemas import ClaimData

        with pytest.raises(ValueError, match="Length of stay unusually long"):
            ClaimData(
                claim_id="CLM001",
                patient_id="PAT001",
                provider_id="PROV001",
                service_code="CONS001",
                billing_date=datetime.now() - timedelta(days=5),
                service_date=datetime.now() - timedelta(days=5),
                billed_amount=500000,
                paid_amount=450000,
                allowed_amount=450000,
                claim_status="Settled",
                patient_age=45,
                length_of_stay=200,  # Invalid (>180)
                quantity=1
            )

    def test_detection_config_valid(self):
        """Test valid detection configuration"""
        from schemas import DetectionConfig

        config = DetectionConfig(
            enable_phase2=True,
            threshold_method="iqr",
            context_aware_enabled=True,
            suppression_threshold=0.25,
            historical_window_days=90
        )

        assert config.enable_phase2 == True
        assert config.threshold_method == "iqr"
        assert config.suppression_threshold == 0.25

    def test_detection_config_invalid_method(self):
        """Test detection config with invalid method (should fail)"""
        from schemas import DetectionConfig

        with pytest.raises(ValueError):
            DetectionConfig(
                enable_phase2=True,
                threshold_method="invalid_method",
                context_aware_enabled=True,
                suppression_threshold=0.25
            )

    def test_detection_config_invalid_threshold(self):
        """Test detection config with invalid threshold (should fail)"""
        from schemas import DetectionConfig

        with pytest.raises(ValueError):
            DetectionConfig(
                enable_phase2=True,
                threshold_method="iqr",
                context_aware_enabled=True,
                suppression_threshold=0.75  # Invalid (>0.5)
            )

    def test_whitelist_exception_valid(self):
        """Test valid whitelist exception"""
        from schemas import WhitelistException

        exception = WhitelistException(
            provider_id="PROV001",
            max_daily_claims=100,
            reason="High-volume legitimate provider",
            expires_at=datetime.now() + timedelta(days=30)
        )

        assert exception.provider_id == "PROV001"
        assert exception.max_daily_claims == 100

    def test_whitelist_exception_invalid_expiration(self):
        """Test whitelist exception with past expiration (should fail)"""
        from schemas import WhitelistException

        with pytest.raises(ValueError, match="Expiration date must be in the future"):
            WhitelistException(
                provider_id="PROV001",
                max_daily_claims=100,
                reason="Test",
                expires_at=datetime.now() - timedelta(days=1)  # Past date
            )

    def test_chronic_condition_valid(self):
        """Test valid chronic condition"""
        from schemas import ChronicCondition

        condition = ChronicCondition(
            icd10_prefix="E11",
            condition_name="Type 2 Diabetes",
            expected_frequency_days=30
        )

        assert condition.icd10_prefix == "E11"
        assert condition.expected_frequency_days == 30

    def test_chronic_condition_invalid_prefix(self):
        """Test chronic condition with invalid ICD-10 prefix (should fail)"""
        from schemas import ChronicCondition

        with pytest.raises(ValueError):
            ChronicCondition(
                icd10_prefix="INVALID",  # Invalid pattern
                condition_name="Test",
                expected_frequency_days=30
            )

    def test_dataframe_converter_valid_row(self):
        """Test DataFrame converter with valid row"""
        from schemas import DataFrameToPydanticConverter

        row = pd.Series({
            'claim_id': 'CLM001',
            'patient_id': 'PAT001',
            'provider_id': 'PROV001',
            'service_code': 'CONS001',
            'diagnosis_code': 'E11',
            'billing_date': '2024-01-15',
            'service_date': '2024-01-15',
            'billed_amount': 500000,
            'paid_amount': 450000,
            'allowed_amount': 450000,
            'claim_status': 'Settled',
            'patient_age': 45,
            'length_of_stay': 0,
            'quantity': 1
        })

        claim = DataFrameToPydanticConverter.convert_row_to_claim(row)
        assert claim.claim_id == "CLM001"
        assert claim.billed_amount == 500000

    def test_dataframe_converter_invalid_row(self):
        """Test DataFrame converter with invalid row"""
        from schemas import DataFrameToPydanticConverter

        row = pd.Series({
            'claim_id': 'CLM001',
            'patient_id': 'PAT001',
            'provider_id': 'PROV001',
            'service_code': 'CONS001',
            'diagnosis_code': 'E11',
            'billing_date': 'invalid_date',  # Invalid date
            'service_date': '2024-01-15',
            'billed_amount': 500000,
            'paid_amount': 450000,
            'allowed_amount': 450000,
            'claim_status': 'Settled',
            'patient_age': 45,
            'length_of_stay': 0,
            'quantity': 1
        })

        with pytest.raises(ValueError, match="billing_date is required"):
            DataFrameToPydanticConverter.convert_row_to_claim(row)

    def test_dataframe_validation_valid(self):
        """Test DataFrame validation with valid data"""
        from schemas import DataFrameToPydanticConverter

        df = pd.DataFrame([
            {
                'claim_id': 'CLM001',
                'patient_id': 'PAT001',
                'provider_id': 'PROV001',
                'service_code': 'CONS001',
                'diagnosis_code': 'E11',
                'billing_date': '2024-01-15',
                'service_date': '2024-01-15',
                'billed_amount': 500000,
                'paid_amount': 450000,
                'allowed_amount': 450000,
                'claim_status': 'Settled',
                'patient_age': 45,
                'length_of_stay': 0,
                'quantity': 1
            }
        ])

        summary = DataFrameToPydanticConverter.validate_dataframe(df)
        assert summary.is_valid == True
        assert summary.valid_claims == 1
        assert summary.invalid_claims == 0

    def test_dataframe_validation_invalid(self):
        """Test DataFrame validation with invalid data"""
        from schemas import DataFrameToPydanticConverter

        df = pd.DataFrame([
            {
                'claim_id': 'CLM001',
                'patient_id': 'PAT001',
                'provider_id': 'PROV001',
                'service_code': 'CONS001',
                'diagnosis_code': 'E11',
                'billing_date': 'invalid_date',  # Invalid
                'service_date': '2024-01-15',
                'billed_amount': 500000,
                'paid_amount': 450000,
                'allowed_amount': 450000,
                'claim_status': 'Settled',
                'patient_age': 45,
                'length_of_stay': 0,
                'quantity': 1
            }
        ])

        summary = DataFrameToPydanticConverter.validate_dataframe(df)
        assert summary.is_valid == False
        assert summary.invalid_claims == 1
        assert len(summary.validation_errors) > 0

    def test_dataframe_validation_empty(self):
        """Test DataFrame validation with empty data"""
        from schemas import DataFrameToPydanticConverter

        df = pd.DataFrame()

        summary = DataFrameToPydanticConverter.validate_dataframe(df)
        assert summary.is_valid == False
        assert summary.total_claims == 0
        assert "empty" in summary.validation_errors[0].lower()

    def test_data_validator_pydantic_integration(self):
        """Test data_validator.py Pydantic integration"""
        from data_validator import pydantic_validation, PYDANTIC_AVAILABLE

        if not PYDANTIC_AVAILABLE:
            pytest.skip("Pydantic not available")

        df = pd.DataFrame([
            {
                'claim_id': 'CLM001',
                'patient_id': 'PAT001',
                'provider_id': 'PROV001',
                'service_code': 'CONS001',
                'diagnosis_code': 'E11',
                'billing_date': '2024-01-15',
                'service_date': '2024-01-15',
                'billed_amount': 500000,
                'paid_amount': 450000,
                'allowed_amount': 450000,
                'claim_status': 'Settled',
                'patient_age': 45,
                'length_of_stay': 0,
                'quantity': 1
            }
        ])

        is_valid, summary = pydantic_validation(df)
        assert is_valid == True
        assert summary.valid_claims == 1
