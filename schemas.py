"""
Pydantic schemas for ASTINA data validation
Phase 1: Data Validation Layer
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np


class ClaimStatus(str, Enum):
    """Claim status enumeration"""
    SETTLED = "Settled"
    PENDING = "Pending"
    REJECTED = "Rejected"
    PAID = "Paid"
    PARTIALLY_PAID = "Partially Paid"
    UNDER_REVIEW = "Under Review"


class ClaimData(BaseModel):
    """Core claim data schema with validation"""
    claim_id: str = Field(..., min_length=1, max_length=50, description="Unique claim identifier")
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient identifier")
    provider_id: str = Field(..., min_length=1, max_length=50, description="Provider/facility identifier")
    service_code: str = Field(..., min_length=1, max_length=20, description="Medical service code")
    diagnosis_code: Optional[str] = Field(None, max_length=20, description="ICD-10 diagnosis code")
    billing_date: datetime = Field(..., description="Billing date")
    service_date: datetime = Field(..., description="Service date")
    billed_amount: float = Field(..., ge=0, description="Billed amount")
    paid_amount: float = Field(..., ge=0, description="Paid amount")
    allowed_amount: float = Field(..., ge=0, description="Allowed amount")
    claim_status: ClaimStatus = Field(..., description="Claim status")
    patient_age: int = Field(..., ge=0, le=120, description="Patient age")
    length_of_stay: int = Field(..., ge=0, le=365, description="Length of stay in days")
    quantity: int = Field(..., ge=0, description="Service quantity")

    @validator('billing_date', 'service_date')
    def validate_dates_not_future(cls, v):
        """Validate that dates are not in the future"""
        if v > datetime.now():
            raise ValueError("Date cannot be in the future")
        return v

    @validator('service_date')
    def validate_service_date_not_before_billing(cls, v, values):
        """Validate that service date is not before billing date"""
        if 'billing_date' in values and v < values['billing_date']:
            raise ValueError("Service date cannot be before billing date")
        return v

    @validator('paid_amount')
    def validate_paid_amount_not_exceed_billed(cls, v, values):
        """Validate that paid amount does not exceed billed amount"""
        if 'billed_amount' in values and v > values['billed_amount']:
            raise ValueError("Paid amount cannot exceed billed amount")
        return v

    @validator('allowed_amount')
    def validate_allowed_amount_not_exceed_billed(cls, v, values):
        """Validate that allowed amount does not exceed billed amount"""
        if 'billed_amount' in values and v > values['billed_amount']:
            raise ValueError("Allowed amount cannot exceed billed amount")
        return v

    @validator('length_of_stay')
    def validate_length_of_stay_with_service_date(cls, v, values):
        """Validate length of stay is reasonable for the service"""
        if v > 0 and 'service_date' in values:
            # For inpatient services, length of stay should be reasonable
            if v > 180:  # More than 6 months is unusual
                raise ValueError("Length of stay unusually long (>180 days)")
        return v

    class Config:
        """Pydantic configuration"""
        extra = "forbid"  # Reject extra fields
        str_strip_whitespace = True  # Strip whitespace from strings


class DetectionConfig(BaseModel):
    """Detection configuration schema"""
    enable_phase2: bool = Field(True, description="Enable Phase 2 features")
    threshold_method: str = Field("iqr", pattern=r"^(iqr|zscore|percentile|robust_mad)$", description="Threshold method")
    context_aware_enabled: bool = Field(True, description="Enable context-aware scoring")
    suppression_threshold: float = Field(0.25, ge=0.0, le=0.5, description="Flag suppression threshold")
    historical_window_days: int = Field(90, ge=7, le=365, description="Historical window in days")

    class Config:
        extra = "forbid"


class DetectionResult(BaseModel):
    """Detection result schema"""
    claim_id: str
    anomaly_probability: float = Field(..., ge=0, le=1, description="Anomaly probability from ML ensemble")
    final_risk_score: float = Field(..., ge=0, le=1, description="Final risk score")
    composite_risk_score: float = Field(..., ge=0, le=1, description="Composite risk score with context")
    context_suppressed: bool = Field(default=False, description="Flag was suppressed by context-aware scoring")
    final_business_risk_flag: bool = Field(default=False, description="Final business risk flag")
    risk_category: str = Field(..., description="Risk category")
    severity: str = Field(..., description="Severity level")

    @validator('severity')
    def validate_severity(cls, v):
        """Validate severity level"""
        if v not in ["Low", "Medium", "High"]:
            raise ValueError("Severity must be one of: Low, Medium, High")
        return v

    class Config:
        extra = "forbid"


class ValidationSummary(BaseModel):
    """Validation summary schema"""
    is_valid: bool
    total_claims: int
    valid_claims: int
    invalid_claims: int
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class WhitelistException(BaseModel):
    """Whitelist exception schema"""
    provider_id: str = Field(..., min_length=1, max_length=50)
    max_daily_claims: int = Field(..., ge=1, le=1000, description="Maximum daily claims allowed")
    reason: str = Field(..., min_length=1, max_length=500, description="Reason for exception")
    expires_at: Optional[datetime] = Field(None, description="Expiration date for exception")

    @validator('expires_at')
    def validate_expiration_date(cls, v):
        """Validate that expiration date is in the future"""
        if v and v < datetime.now():
            raise ValueError("Expiration date must be in the future")
        return v

    class Config:
        extra = "forbid"


class ChronicCondition(BaseModel):
    """Chronic condition schema for whitelist"""
    icd10_prefix: str = Field(..., min_length=3, max_length=5, description="ICD-10 prefix (e.g., E11)")
    condition_name: str = Field(..., min_length=1, max_length=100, description="Condition name")
    expected_frequency_days: int = Field(..., ge=1, le=365, description="Expected frequency in days")

    @validator('icd10_prefix')
    def validate_icd10_prefix(cls, v):
        """Validate ICD-10 prefix format"""
        import re
        if not re.match(r"^[A-Z]\d{2}$", v):
            raise ValueError("ICD-10 prefix must be in format A## (e.g., E11)")
        return v

    class Config:
        extra = "forbid"


class DataFrameToPydanticConverter:
    """Converter from pandas DataFrame to Pydantic models"""

    @staticmethod
    def convert_row_to_claim(row: pd.Series) -> ClaimData:
        """Convert a single DataFrame row to ClaimData model"""
        try:
            # Convert dates
            billing_date = pd.to_datetime(row.get('billing_date'), errors='coerce')
            service_date = pd.to_datetime(row.get('service_date'), errors='coerce')

            if pd.isna(billing_date):
                raise ValueError("billing_date is required and must be a valid date")
            if pd.isna(service_date):
                raise ValueError("service_date is required and must be a valid date")

            # Convert claim status
            claim_status_str = str(row.get('claim_status', 'Pending'))
            try:
                claim_status = ClaimStatus(claim_status_str)
            except ValueError:
                raise ValueError(f"Invalid claim_status: {claim_status_str}. Must be one of: {[s.value for s in ClaimStatus]}")

            return ClaimData(
                claim_id=str(row.get('claim_id', '')),
                patient_id=str(row.get('patient_id', '')),
                provider_id=str(row.get('provider_id', '')),
                service_code=str(row.get('service_code', '')),
                diagnosis_code=str(row.get('diagnosis_code')) if pd.notna(row.get('diagnosis_code')) else None,
                billing_date=billing_date,
                service_date=service_date,
                billed_amount=float(row.get('billed_amount', 0)),
                paid_amount=float(row.get('paid_amount', 0)),
                allowed_amount=float(row.get('allowed_amount', 0)),
                claim_status=claim_status,
                patient_age=int(row.get('patient_age', 0)),
                length_of_stay=int(row.get('length_of_stay', 0)),
                quantity=int(row.get('quantity', 1))
            )
        except Exception as e:
            raise ValueError(f"Row validation failed: {str(e)}")

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> ValidationSummary:
        """
        Validate entire DataFrame using Pydantic models

        Args:
            df: Input DataFrame

        Returns:
            ValidationSummary with validation results
        """
        if df.empty:
            return ValidationSummary(
                is_valid=False,
                total_claims=0,
                valid_claims=0,
                invalid_claims=0,
                validation_errors=["DataFrame is empty"]
            )

        validation_errors = []
        validation_warnings = []
        valid_claims = 0
        invalid_claims = 0

        for idx, row in df.iterrows():
            try:
                DataFrameToPydanticConverter.convert_row_to_claim(row)
                valid_claims += 1
            except Exception as e:
                invalid_claims += 1
                error_msg = f"Row {idx}: {str(e)}"
                validation_errors.append(error_msg)

        # Add warnings for common issues
        if df.duplicated().sum() > 0:
            validation_warnings.append(f"Found {df.duplicated().sum()} duplicate rows")

        if df.isnull().sum().sum() > 0:
            validation_warnings.append(f"Found {df.isnull().sum().sum()} missing values")

        is_valid = invalid_claims == 0

        return ValidationSummary(
            is_valid=is_valid,
            total_claims=len(df),
            valid_claims=valid_claims,
            invalid_claims=invalid_claims,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings
        )
