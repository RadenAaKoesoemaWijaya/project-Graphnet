import numpy as np
import pandas as pd
import pytest

from fraud_risk_pipeline import (
    normalize_claims_dataframe,
    enrich_claims_with_business_risk_features,
    run_integrated_claim_risk_pipeline,
    compute_patient_level_fuzzy_similarity_scores,
)
from phantom_service_rules import PhantomServiceRuleEngine
from provider_capacity_validator import ProviderCapacityValidator
from repeat_billing_detector import RepeatBillingDetector
from claim_status_validator import ClaimStatusValidator
from medication_device_fraud_rules import detect_medication_and_device_fraud


def test_normalize_claims_dataframe_empty():
    df_empty = pd.DataFrame()
    normalized = normalize_claims_dataframe(df_empty)
    assert normalized.empty
    assert "amount" in normalized.columns
    assert "billing_date" in normalized.columns
    assert "claim_id" in normalized.columns


def test_normalize_claims_dataframe_dirty_types():
    df_dirty = pd.DataFrame([
        {"claim_id": 101, "patient_id": "P1", "provider_id": "PR1", "amount": "$1,500.50", "billing_date": "2026/01/15"},
        {"claim_id": 102, "patient_id": "P2", "provider_id": "PR2", "amount": "invalid_val", "billing_date": "not-a-date"},
        {"claim_id": None, "patient_id": None, "provider_id": None, "amount": None, "billing_date": None},
    ])
    clean = normalize_claims_dataframe(df_dirty)
    assert len(clean) == 3
    assert clean.iloc[0]["amount"] == 0.0 or isinstance(clean.iloc[0]["amount"], (int, float))
    assert clean.iloc[1]["amount"] == 0.0
    assert pd.isna(clean.iloc[1]["billing_date"])
    assert pd.isna(clean.iloc[2]["billing_date"])


def test_fuzzy_similarity_with_single_row():
    df_single = pd.DataFrame([
        {"claim_id": "C1", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "amount": 1000.0, "billing_date": "2026-01-01"}
    ])
    scores = compute_patient_level_fuzzy_similarity_scores(df_single)
    assert len(scores) == 1
    assert scores.iloc[0] == 0.0


def test_fuzzy_similarity_preserves_non_contiguous_row_mapping():
    df = pd.DataFrame([
        {"claim_id": "A1", "patient_id": "PA", "provider_id": "PR", "service_code": "CT001", "amount": 1000, "billing_date": "2026-01-01"},
        {"claim_id": "B1", "patient_id": "PB", "provider_id": "PR", "service_code": "XR001", "amount": 2000, "billing_date": "2026-01-01"},
        {"claim_id": "A2", "patient_id": "PA", "provider_id": "PR", "service_code": "CT001", "amount": 1000, "billing_date": "2026-01-02"},
        {"claim_id": "B2", "patient_id": "PB", "provider_id": "PR", "service_code": "XR001", "amount": 2000, "billing_date": "2026-01-02"},
    ], index=[10, 20, 30, 40])

    scores = compute_patient_level_fuzzy_similarity_scores(df)
    assert list(scores.index) == [10, 20, 30, 40]
    assert scores.loc[10] == scores.loc[30]
    assert scores.loc[20] == scores.loc[40]


def test_enrich_claims_with_business_risk_features_dirty_dataset():
    df_dirty = pd.DataFrame([
        {"claim_id": "C_NORM_1", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "amount": 1000.0, "billing_date": "2026-01-01", "service_date": "2026-01-01"},
        {"claim_id": "C_NORM_2", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "amount": 1000.0, "billing_date": "2026-01-02", "service_date": "2026-01-02"},
        {"claim_id": "C_BAD_1", "patient_id": "P2", "provider_id": "PR2", "service_code": "BAD_CODE", "amount": -500.0, "billing_date": "invalid", "service_date": "invalid"},
        {"claim_id": "C_EXTREME", "patient_id": "P3", "provider_id": "PR3", "service_code": "XR001", "amount": 999999999.0, "billing_date": "2026-02-01", "service_date": "2026-02-01"},
    ])
    enriched, summary = enrich_claims_with_business_risk_features(df_dirty)
    assert len(enriched) == 4
    assert "business_risk_score" in enriched.columns
    assert "business_risk_flag" in enriched.columns
    assert summary["total_claims"] == 4
    assert 0.0 <= summary["avg_business_risk_score"] <= 1.0


def test_run_integrated_claim_risk_pipeline_without_ml_model():
    df = pd.DataFrame([
        {"claim_id": "C10", "patient_id": "P10", "provider_id": "PR10", "service_code": "CT001", "amount": 2500.0, "billing_date": "2026-01-10", "service_date": "2026-01-10", "diagnosis_code": "A01"},
        {"claim_id": "C11", "patient_id": "P10", "provider_id": "PR10", "service_code": "CT001", "amount": 2500.0, "billing_date": "2026-01-12", "service_date": "2026-01-12", "diagnosis_code": "A01"},
    ])
    result, summary = run_integrated_claim_risk_pipeline(df, ml_model=None, chunk_size=100)
    assert not result.empty
    assert "final_risk_score" in result.columns
    assert "final_risk_flag" in result.columns
    assert "risk_category" in result.columns
    assert summary["total_claims"] == 2


def test_pipeline_result_is_independent_of_chunk_size():
    df = pd.DataFrame([
        {"claim_id": "C1", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "amount": 1000, "billing_date": "2026-01-01", "service_date": "2026-01-01"},
        {"claim_id": "C2", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "amount": 1000, "billing_date": "2026-01-02", "service_date": "2026-01-02"},
    ])
    full, _ = run_integrated_claim_risk_pipeline(df, chunk_size=None)
    chunked, _ = run_integrated_claim_risk_pipeline(df, chunk_size=1)
    assert full["repeat_billing_flag"].tolist() == chunked["repeat_billing_flag"].tolist()
    assert full["repeat_billing_score"].tolist() == chunked["repeat_billing_score"].tolist()


def test_duplicate_payment_excludes_current_claim_and_only_paid_counts():
    class FakeConnection:
        def execute(self, *_args):
            return [
                {"claim_id": "C1", "status": "PAID"},
                {"claim_id": "C0", "status": "PENDING"},
            ]

    validator = ClaimStatusValidator(db_connection=FakeConnection())
    duplicate, history, message = validator.check_duplicate_payment({
        "claim_id": "C1", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001",
    })
    assert duplicate is False
    assert [item["claim_id"] for item in history] == ["C0"]
    assert "belum ada pembayaran PAID duplikat" in message


def test_phantom_engine_edge_cases():
    engine = PhantomServiceRuleEngine()
    # Missing service code
    valid, violations = engine.validate_claim({})
    assert not valid
    assert "Kode layanan tidak ditemukan" in violations[0]

    # Invalid age formats
    valid, violations = engine.validate_claim({"service_code": "CT001", "patient_age": "not-a-number"})
    assert valid or not violations  # Doesn't crash on invalid age format

    valid, violations = engine.validate_claim({"service_code": "CT001", "patient_age": 0.01})
    assert not valid
    assert any("usia < 1 bulan" in v for v in violations)


def test_provider_capacity_edge_cases():
    validator = ProviderCapacityValidator()
    # Empty dataframe
    is_feasible, violations, util = validator.validate_provider_schedule(pd.DataFrame(), "PR_TEST", "2026-01-01")
    assert is_feasible
    assert util == 0.0

    # Normal dataframe
    df = pd.DataFrame([
        {"provider_id": "PR_TEST", "service_date": "2026-01-01", "service_code": "CONS001"},
        {"provider_id": "PR_TEST", "service_date": "2026-01-01", "service_code": "CONS001"},
    ])
    is_feasible, violations, util = validator.validate_provider_schedule(df, "PR_TEST", "2026-01-01")
    assert is_feasible
    assert util > 0.0


def test_provider_capacity_is_attributed_by_calendar_date():
    rows = []
    for index in range(16):
        rows.append({
            "claim_id": f"CAP-{index}",
            "patient_id": f"P-{index}",
            "provider_id": "PR-CAP",
            "service_code": "CONS001",
            "amount": 100.0,
            "billing_date": f"2026-01-01 0{index % 9}:00:00",
            "service_date": f"2026-01-01 {index % 9}:00:00",
        })
    rows.append({
        "claim_id": "CAP-NEXT",
        "patient_id": "P-NEXT",
        "provider_id": "PR-CAP",
        "service_code": "CONS001",
        "amount": 100.0,
        "billing_date": "2026-01-02",
        "service_date": "2026-01-02",
    })
    enriched, _ = enrich_claims_with_business_risk_features(pd.DataFrame(rows))
    assert enriched.loc[enriched["claim_id"] == "CAP-NEXT", "provider_capacity_flag"].iloc[0] == 0
    assert enriched.loc[enriched["claim_id"] == "CAP-0", "provider_capacity_flag"].iloc[0] == 1


def test_medication_zero_delivery_is_flagged():
    result = detect_medication_and_device_fraud(pd.DataFrame([
        {
            "claim_id": "MED-1",
            "patient_id": "P1",
            "provider_id": "PR1",
            "amount": 100.0,
            "item_code": "DRUG-1",
            "quantity_billed": 10,
            "quantity_delivered": 0,
            "unit_price": 10.0,
        }
    ]))
    assert not result.empty
    assert result.iloc[0]["claim_id"] == "MED-1"
