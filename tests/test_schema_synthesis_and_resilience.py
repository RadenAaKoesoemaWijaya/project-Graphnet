"""
Resilience & Contract Tests for Schema Harmonizer, Semantic Aliasing, and Business Rules Circuit Breaker.
"""
import pytest
import pandas as pd
import numpy as np

from schema_harmonizer import SchemaHarmonizer
from fraud_risk_pipeline import (
    normalize_claims_dataframe,
    enrich_claims_with_business_risk_features,
    run_integrated_claim_risk_pipeline,
)
from length_of_stay_rules import detect_prolonged_stay_and_readmission


def test_zero_crash_on_minimal_dataset():
    """System must not crash or throw exceptions when given only minimal columns."""
    df_minimal = pd.DataFrame([
        {"claim_id": "CLM-001", "amount": 2500000.0},
        {"claim_id": "CLM-002", "amount": 7500000.0},
    ])
    result, summary = run_integrated_claim_risk_pipeline(df_minimal)
    assert not result.empty
    assert len(result) == 2
    assert "business_risk_score" in result.columns
    assert "final_risk_score" in result.columns
    assert (result["business_risk_score"] >= 0.0).all() and (result["business_risk_score"] <= 1.0).all()
    assert (result["final_risk_score"] >= 0.0).all() and (result["final_risk_score"] <= 1.0).all()
    assert summary["total_claims"] == 2
    # Check circuit breaker marked skipped rules
    assert "rule_readiness" in summary
    assert summary["active_rules_count"] < summary["total_rules_count"]


def test_indonesian_column_aliases():
    """System must transparently resolve Indonesian and industry synonyms to canonical columns."""
    df_indo = pd.DataFrame([
        {
            "no_klaim": "KLAIM-001",
            "no_peserta": "PESERTA-01",
            "kode_faskes": "RSUD-001",
            "kode_tindakan": "99213",
            "diagnosa": "J06.9",
            "tgl_klaim": "2024-02-01",
            "tgl_pelayanan": "2024-01-28",
            "biaya_tagihan": "15,000,000",
            "biaya_dibayar": "12,000,000",
            "lama_rawat": 3,
            "jumlah": 2,
            "status": "APPROVED",
        },
        {
            "no_klaim": "KLAIM-002",
            "no_peserta": "PESERTA-01",
            "kode_faskes": "RSUD-001",
            "kode_tindakan": "99213",
            "diagnosa": "J06.9",
            "tgl_klaim": "2024-02-05",
            "tgl_pelayanan": "2024-02-04",
            "biaya_tagihan": "15,000,000",
            "biaya_dibayar": "12,000,000",
            "lama_rawat": 1,
            "jumlah": 1,
            "status": "APPROVED",
        },
    ])

    clean, report = SchemaHarmonizer.harmonize_claims_schema(df_indo)
    assert clean["claim_id"].iloc[0] == "KLAIM-001"
    assert clean["patient_id"].iloc[0] == "PESERTA-01"
    assert clean["provider_id"].iloc[0] == "RSUD-001"
    assert clean["service_code"].iloc[0] == "99213"
    assert clean["diagnosis_code"].iloc[0] == "J06.9"
    assert clean["amount"].iloc[0] == 15000000.0
    assert clean["billed_amount"].iloc[0] == 15000000.0
    assert clean["paid_amount"].iloc[0] == 12000000.0
    assert clean["length_of_stay"].iloc[0] == 3
    assert clean["quantity"].iloc[0] == 2

    # Check readiness
    readiness = SchemaHarmonizer.evaluate_rule_readiness(clean)
    assert readiness["repeat_billing"]["ready"] is True
    assert readiness["provider_capacity"]["ready"] is True
    assert readiness["phantom_service"]["ready"] is True


def test_deterministic_date_and_los_derivation():
    """System must derive missing admission/discharge dates from service_date and length_of_stay."""
    df_los = pd.DataFrame([
        {
            "claim_id": "CLM-LOS-1",
            "patient_id": "PAT-01",
            "provider_id": "PROV-01",
            "service_date": "2024-03-01",
            "length_of_stay": 10,
            "amount": 25000000.0,
            "diagnosis_code": "I10",
        }
    ])
    clean = normalize_claims_dataframe(df_los)
    assert "admission_date" in clean.columns
    assert "discharge_date" in clean.columns
    assert clean["admission_date"].iloc[0] == pd.Timestamp("2024-03-01")
    assert clean["discharge_date"].iloc[0] == pd.Timestamp("2024-03-11")

    # Verify length_of_stay_rules executes without missing column error
    los_results = detect_prolonged_stay_and_readmission(clean)
    assert isinstance(los_results, pd.DataFrame)


def test_dynamic_weight_renormalization_circuit_breaker():
    """When some rules are skipped due to missing prerequisites, weights must re-normalize to 1.0."""
    # Create dataset with phantom service and upcoding, but missing dates (so repeat billing is skipped)
    df_no_dates = pd.DataFrame([
        {
            "claim_id": f"CLM-{i}",
            "patient_id": f"PAT-{i}",
            "provider_id": "PROV-01",
            "service_code": "SUR001",
            "diagnosis_code": "K21.0",
            "amount": 5000000.0,
        }
        for i in range(5)
    ])
    clean, summary = enrich_claims_with_business_risk_features(df_no_dates)
    assert summary["rule_readiness"]["repeat_billing"]["ready"] is False
    assert summary["rule_readiness"]["repeat_billing"]["status"] == "SKIPPED"
    # business_risk_score must still be well-bounded
    assert (clean["business_risk_score"] >= 0.0).all() and (clean["business_risk_score"] <= 1.0).all()


def test_provenance_tag_preservation():
    """Imputed columns must be tagged in metadata attrs to prevent false positives."""
    df_simple = pd.DataFrame([
        {"claim_id": "C1", "amount": 1000.0}
    ])
    clean, report = SchemaHarmonizer.harmonize_claims_schema(df_simple)
    assert "quantity" in clean.attrs["_imputed_columns"]
    assert "claim_status" in clean.attrs["_imputed_columns"]
    assert clean["quantity"].iloc[0] == 1
    assert clean["claim_status"].iloc[0] == "APPROVED"


def test_empty_dataframe_resilience():
    """Empty dataframe must return clean structure with canonical columns without raising."""
    df_empty = pd.DataFrame()
    clean, report = SchemaHarmonizer.harmonize_claims_schema(df_empty)
    assert clean.empty
    assert "claim_id" in clean.columns
    assert "amount" in clean.columns
    assert "billing_date" in clean.columns
    assert "service_date" in clean.columns
