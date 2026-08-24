import pandas as pd

from repeat_billing_detector import RepeatBillingDetector
from fuzzy_claim_matcher import FuzzyClaimMatcher
from phantom_service_rules import PhantomServiceRuleEngine
from provider_capacity_validator import ProviderCapacityValidator
from fraud_risk_pipeline import run_integrated_claim_risk_pipeline
from upcoding_unbundling_rules import detect_upcoding_and_unbundling
from inflated_bill_cloning_rules import detect_inflated_bill_and_cloning
from length_of_stay_rules import detect_prolonged_stay_and_readmission
from medication_device_fraud_rules import detect_medication_and_device_fraud


def test_repeat_billing_detects_similar_claims():
    df = pd.DataFrame([
        {"claim_id": "C1", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "billing_date": "2026-01-01", "service_date": "2026-01-01", "amount": 1000.0, "diagnosis_code": "A01"},
        {"claim_id": "C2", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "billing_date": "2026-01-10", "service_date": "2026-01-10", "amount": 1050.0, "diagnosis_code": "A01"},
    ])
    result = RepeatBillingDetector(temporal_window_days=30, fuzzy_threshold=0.75).detect_repeat_claims(df)
    assert not result.empty
    assert result.iloc[0]["risk_score"] > 0.5


def test_fuzzy_claim_match_uses_weighted_similarity():
    matcher = FuzzyClaimMatcher()
    score = matcher.calculate_claim_similarity(
        {"patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "amount": 1000, "service_date": "2026-01-01", "diagnosis_code": "A01"},
        {"patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "amount": 990, "service_date": "2026-01-03", "diagnosis_code": "A01"},
    )
    assert 0.7 <= score <= 1.0


def test_phantom_rules_flag_invalid_service_and_frequency():
    engine = PhantomServiceRuleEngine()
    valid, violations = engine.validate_claim({
        "service_code": "BAD999",
        "provider_id": "PR1",
        "patient_age": 2,
        "patient_gender": "F",
    })
    assert not valid
    assert violations

    df = pd.DataFrame([
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
        {"patient_id": "P2", "service_code": "CONS001", "service_date": "2026-01-15"},
    ])
    is_violation, reason = engine.check_frequency_violation(df, "P2", "CONS001", pd.Timestamp("2026-01-15"))
    assert is_violation
    assert reason


def test_provider_capacity_validator_flags_over_capacity():
    validator = ProviderCapacityValidator()
    df = pd.DataFrame([
        {"provider_id": "PR9", "service_code": "CT_SCAN", "service_date": "2026-02-10"},
        {"provider_id": "PR9", "service_code": "CT_SCAN", "service_date": "2026-02-10"},
        {"provider_id": "PR9", "service_code": "CT_SCAN", "service_date": "2026-02-10"},
    ])
    is_feasible, violations, utilization = validator.validate_provider_schedule(df, "PR9", "2026-02-10")
    assert not is_feasible
    assert violations
    assert utilization > 0


def test_pipeline_handles_empty_and_invalid_inputs_gracefully():
    empty = pd.DataFrame([])
    enriched, summary = run_integrated_claim_risk_pipeline(empty)
    assert len(enriched) == 0
    assert "total_claims" in summary

    invalid_df = pd.DataFrame([
        {"claim_id": "C1", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "billing_date": "bad-date", "amount": "oops"},
    ])
    result, _ = run_integrated_claim_risk_pipeline(invalid_df)
    assert len(result) == 1
    assert result["amount"].notna().all()


def test_repeat_billing_ignores_large_time_gap():
    df = pd.DataFrame([
        {"claim_id": "C1", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "billing_date": "2026-01-01", "service_date": "2026-01-01", "amount": 1000.0, "diagnosis_code": "A01"},
        {"claim_id": "C2", "patient_id": "P1", "provider_id": "PR1", "service_code": "CT001", "billing_date": "2026-06-01", "service_date": "2026-06-01", "amount": 1005.0, "diagnosis_code": "A01"},
    ])
    result = RepeatBillingDetector(temporal_window_days=30).detect_repeat_claims(df)
    assert result.empty


def test_upcoding_detector_flags_high_amount_pattern():
    df = pd.DataFrame([
        {"claim_id": "U1", "patient_id": "P10", "provider_id": "PR10", "diagnosis_code": "A01", "procedure_code": "A01", "amount": 15000.0},
        {"claim_id": "U2", "patient_id": "P11", "provider_id": "PR11", "diagnosis_code": "B02", "procedure_code": "C03", "amount": 400.0},
    ])
    result = detect_upcoding_and_unbundling(df)
    assert not result.empty
    assert result["flag"].sum() >= 1


def test_cloning_detector_handles_empty_and_duplicate_like_records():
    empty = pd.DataFrame([])
    empty_result = detect_inflated_bill_and_cloning(empty)
    assert empty_result.empty

    df = pd.DataFrame([
        {"claim_id": "K1", "patient_id": "P20", "provider_id": "PR20", "service_date": "2026-03-01", "diagnosis_code": "A99", "amount": 2000.0},
        {"claim_id": "K2", "patient_id": "P20", "provider_id": "PR20", "service_date": "2026-03-01", "diagnosis_code": "A99", "amount": 2050.0},
        {"claim_id": "K3", "patient_id": "P20", "provider_id": "PR20", "service_date": "2026-03-02", "diagnosis_code": "A99", "amount": 300.0},
    ])
    result = detect_inflated_bill_and_cloning(df)
    assert not result.empty
    assert result["flag"].sum() >= 1


def test_length_of_stay_detector_flags_long_stay():
    df = pd.DataFrame([
        {"claim_id": "L1", "patient_id": "P30", "provider_id": "PR30", "admission_date": "2026-04-01", "discharge_date": "2026-04-20", "diagnosis_code": "D01"},
        {"claim_id": "L2", "patient_id": "P31", "provider_id": "PR31", "admission_date": "2026-04-02", "discharge_date": "2026-04-04", "diagnosis_code": "D02"},
    ])
    result = detect_prolonged_stay_and_readmission(df)
    assert not result.empty
    assert result["flag"].sum() >= 1


def test_medication_device_detector_flags_quantity_mismatch():
    df = pd.DataFrame([
        {"claim_id": "M1", "patient_id": "P40", "provider_id": "PR40", "amount": 800.0, "quantity_billed": 10, "quantity_delivered": 2, "unit_price": 80},
        {"claim_id": "M2", "patient_id": "P41", "provider_id": "PR41", "amount": 100.0, "quantity_billed": 1, "quantity_delivered": 1, "unit_price": 100},
    ])
    result = detect_medication_and_device_fraud(df)
    assert not result.empty
    assert result["flag"].sum() >= 1


def test_patient_level_fuzzy_similarity_is_window_limited():
    df = pd.DataFrame([
        {"claim_id": "F1", "patient_id": "P50", "provider_id": "PR50", "service_code": "CT001", "billing_date": "2026-01-01", "service_date": "2026-01-01", "amount": 1000.0, "diagnosis_code": "A01"},
        {"claim_id": "F2", "patient_id": "P50", "provider_id": "PR50", "service_code": "CT001", "billing_date": "2026-01-05", "service_date": "2026-01-05", "amount": 1010.0, "diagnosis_code": "A01"},
        {"claim_id": "F3", "patient_id": "P50", "provider_id": "PR50", "service_code": "CT001", "billing_date": "2026-04-01", "service_date": "2026-04-01", "amount": 1050.0, "diagnosis_code": "A01"},
    ])
    from fraud_risk_pipeline import compute_patient_level_fuzzy_similarity_scores

    scores = compute_patient_level_fuzzy_similarity_scores(df, max_window_days=30)
    assert len(scores) == len(df)
    assert scores[0] > 0.7
    assert scores[2] >= 0.0


def test_chunked_pipeline_handles_large_dataframe():
    rows = []
    for i in range(1200):
        rows.append({
            "claim_id": f"C{i:04d}",
            "patient_id": f"P{i % 50}",
            "provider_id": "PR1",
            "service_code": "CT001",
            "billing_date": "2026-01-01",
            "service_date": "2026-01-01",
            "amount": 1000.0 + (i % 10),
            "diagnosis_code": "A01",
            "status": "Approved",
        })

    df = pd.DataFrame(rows)
    result, summary = run_integrated_claim_risk_pipeline(df, chunk_size=250)
    assert len(result) == len(df)
    assert summary["total_claims"] == len(df)
    assert "final_high_risk_claims" in summary


def test_feature_importance_supports_sampling_budget():
    import numpy as np
    from model import CombinedAnomalyDetector
    from model_explainer import ModelExplainer

    rng = np.random.RandomState(42)
    X = rng.randn(80, 10)
    detector = CombinedAnomalyDetector()
    detector.fit(X)

    explainer = ModelExplainer(detector=detector, feature_names=[f"f{i}" for i in range(X.shape[1])])
    assert explainer.initialize_explainers(X[:20])

    importance = explainer.get_feature_importance("isolation_forest", X=X, max_samples=32)
    assert importance is not None
    assert len(importance) == X.shape[1]
    assert importance["importance"].sum() > 0


def test_feature_importance_unsupported_model_has_clear_message(monkeypatch):
    import numpy as np
    from model import CombinedAnomalyDetector
    from model_explainer import ModelExplainer

    rng = np.random.RandomState(7)
    X = rng.randn(50, 6)
    detector = CombinedAnomalyDetector()
    detector.fit(X)

    explainer = ModelExplainer(detector=detector, feature_names=[f"f{i}" for i in range(X.shape[1])])
    assert explainer.initialize_explainers(X[:20])

    captured = []
    monkeypatch.setattr("streamlit.warning", lambda msg: captured.append(str(msg)))
    monkeypatch.setattr("streamlit.error", lambda msg: captured.append(str(msg)))

    result = explainer.get_feature_importance("autoencoder", X=X, max_samples=32)
    assert result is None
    assert any(
        "tidak didukung" in msg.lower() or "tidak sesuai" in msg.lower() or "unsupported" in msg.lower()
        for msg in captured
    )
