from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from claim_status_validator import ClaimStatusValidator
from fuzzy_claim_matcher import FuzzyClaimMatcher
from inflated_bill_cloning_rules import detect_inflated_bill_and_cloning
from length_of_stay_rules import detect_prolonged_stay_and_readmission
from medication_device_fraud_rules import detect_medication_and_device_fraud
from phantom_service_rules import PhantomServiceRuleEngine
from provider_capacity_validator import ProviderCapacityValidator
from repeat_billing_detector import RepeatBillingDetector
from upcoding_unbundling_rules import detect_upcoding_and_unbundling

logger = logging.getLogger("astina.fraud_risk_pipeline")

REQUIRED_CLAIM_FIELDS = {
    "claim_id",
    "patient_id",
    "provider_id",
    "service_code",
    "billing_date",
    "amount",
}


def normalize_claims_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("DataFrame klaim tidak boleh None")
    clean = df.copy()
    if "_astina_row_id" not in clean.columns:
        clean["_astina_row_id"] = np.arange(len(clean), dtype=np.int64)
    for column in ["claim_id", "patient_id", "provider_id", "service_code", "diagnosis_code"]:
        if column not in clean.columns:
            clean[column] = ""
    for column in ["billing_date", "service_date"]:
        if column not in clean.columns:
            clean[column] = pd.NaT
    if "amount" not in clean.columns:
        clean["amount"] = 0.0
    clean["amount"] = pd.to_numeric(clean["amount"], errors="coerce").fillna(0.0)
    clean["billing_date"] = pd.to_datetime(clean["billing_date"], errors="coerce")
    clean["service_date"] = pd.to_datetime(clean["service_date"], errors="coerce")
    return clean


def compute_patient_level_fuzzy_similarity_scores(
    df: pd.DataFrame,
    max_window_days: int = 30,
    min_similarity: float = 0.8,
) -> pd.Series:
    """Compute a bounded patient-level fuzzy similarity score without quadratic full-frame scans."""
    clean = normalize_claims_dataframe(df)
    if clean.empty:
        return pd.Series(dtype=float, index=clean.index, name="fuzzy_similarity_score")

    matcher = FuzzyClaimMatcher()
    scores: Dict[int, float] = {}

    for patient_id, patient_group in clean.groupby("patient_id", dropna=False):
        patient_group = patient_group.sort_values("billing_date")
        if patient_group.empty:
            continue

        for row_id, row in patient_group.set_index("_astina_row_id").iterrows():
            base_date = pd.to_datetime(row.get("billing_date"), errors="coerce")
            if pd.isna(base_date):
                scores[int(row_id)] = 0.0
                continue

            candidates = patient_group[patient_group["_astina_row_id"] != row_id].copy()
            if candidates.empty:
                scores[int(row_id)] = 0.0
                continue

            candidates["time_gap_days"] = (pd.to_datetime(candidates["billing_date"], errors="coerce") - base_date).dt.days.abs()
            candidates = candidates[candidates["time_gap_days"].le(max_window_days)]
            if candidates.empty:
                scores[int(row_id)] = 0.0
                continue

            best_score = 0.0
            for _, candidate in candidates.iterrows():
                score = matcher.calculate_claim_similarity(row.to_dict(), candidate.to_dict())
                if score > best_score:
                    best_score = score

            scores[int(row_id)] = float(best_score if best_score >= min_similarity else 0.0)

    ordered_scores = pd.Series(
        [scores.get(int(row_id), 0.0) for row_id in clean["_astina_row_id"]],
        index=clean.index,
        name="fuzzy_similarity_score",
    )
    return ordered_scores


def derive_risk_category(row: pd.Series) -> str:
    """Map a claim row to its dominant risk category."""
    flag_checks = [
        (row.get('repeat_billing_flag', 0) == 1, 'Repeat Billing'),
        (row.get('phantom_service_flag', 0) == 1, 'Phantom Service'),
        (row.get('provider_capacity_flag', 0) == 1, 'Provider Capacity'),
        (row.get('upcoding_unbundling_flag', 0) == 1, 'Upcoding'),
        (row.get('inflated_bill_cloning_flag', 0) == 1, 'Inflated Bill / Cloning'),
        (row.get('prolonged_stay_readmission_flag', 0) == 1, 'Prolonged Stay'),
        (row.get('medication_device_fraud_flag', 0) == 1, 'Medication / Device'),
        (row.get('duplicate_payment_flag', 0) == 1, 'Duplicate Payment'),
    ]
    for matched, label in flag_checks:
        if matched:
            return label
    if row.get('final_risk_flag', 0) == 1 or row.get('anomaly_prediction', 0) == 1 or row.get('business_risk_flag', 0) == 1:
        return 'High Anomaly'
    return 'Normal / Low Risk'


def enrich_claims_with_business_risk_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    clean = normalize_claims_dataframe(df)
    if clean.empty:
        return clean, {
            "total_claims": 0,
            "repeat_billing_cases": 0,
            "phantom_service_cases": 0,
            "provider_capacity_issues": 0,
            "upcoding_unbundling_cases": 0,
            "inflated_bill_cloning_cases": 0,
            "prolonged_stay_readmission_cases": 0,
            "medication_device_fraud_cases": 0,
            "avg_business_risk_score": 0.0,
            "high_risk_claims": 0,
        }

    # Parallel evaluation of independent rule modules
    repeat_detector = RepeatBillingDetector(temporal_window_days=30, fuzzy_threshold=0.8)
    phantom_engine = PhantomServiceRuleEngine()
    capacity_validator = ProviderCapacityValidator()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f_repeat = executor.submit(repeat_detector.detect_repeat_claims, clean)
        f_phantom = executor.submit(phantom_engine.validate_claims_dataframe, clean)
        f_capacity = executor.submit(capacity_validator.validate_all_providers_batch, clean)
        f_fuzzy = executor.submit(compute_patient_level_fuzzy_similarity_scores, clean, 30, 0.8)
        f_upcoding = executor.submit(detect_upcoding_and_unbundling, clean)
        f_inflated = executor.submit(detect_inflated_bill_and_cloning, clean)
        f_stay = executor.submit(detect_prolonged_stay_and_readmission, clean)
        f_med = executor.submit(detect_medication_and_device_fraud, clean)

        repeat_results = f_repeat.result()
        phantom_df = f_phantom.result()
        provider_df = f_capacity.result()
        fuzzy_scores = f_fuzzy.result()
        upcoding_df = f_upcoding.result()
        inflated_df = f_inflated.result()
        stay_df = f_stay.result()
        med_df = f_med.result()

    clean["fuzzy_similarity_score"] = fuzzy_scores

    # Repeat billing mapping
    repeat_lookup = {}
    for _, row in repeat_results.iterrows():
        first_key = str(row["first_claim_id"])
        repeat_key = str(row["repeat_claim_id"])
        score = float(row["risk_score"])
        repeat_lookup[first_key] = max(repeat_lookup.get(first_key, 0.0), score)
        repeat_lookup[repeat_key] = max(repeat_lookup.get(repeat_key, 0.0), score)

    clean["repeat_billing_flag"] = clean["claim_id"].astype(str).map(lambda claim_id: 1 if str(claim_id) in repeat_lookup else 0).astype(int)
    clean["repeat_billing_score"] = clean["claim_id"].astype(str).map(lambda claim_id: float(repeat_lookup.get(str(claim_id), 0.0))).fillna(0.0)

    # Phantom service mapping
    clean["phantom_service_flag"] = 0
    if not phantom_df.empty:
        phantom_lookup = {str(claim_id): 1 for claim_id in phantom_df["claim_id"].astype(str).dropna().unique()}
        clean["phantom_service_flag"] = clean["claim_id"].astype(str).map(phantom_lookup).fillna(0).astype(int)
    clean["phantom_service_score"] = clean["phantom_service_flag"].astype(float) * 0.9

    # Provider capacity mapping
    clean["provider_capacity_flag"] = 0
    if not provider_df.empty:
        provider_dates = provider_df.copy()
        provider_dates["_capacity_date"] = pd.to_datetime(provider_dates["service_date"], errors="coerce").dt.normalize()
        claim_dates = pd.to_datetime(clean["service_date"], errors="coerce").dt.normalize()
        capacity_keys = set(zip(provider_dates["provider_id"].astype(str), provider_dates["_capacity_date"]))
        clean["provider_capacity_flag"] = [
            int((str(provider_id), service_date) in capacity_keys)
            for provider_id, service_date in zip(clean["provider_id"], claim_dates)
        ]
    clean["provider_capacity_score"] = clean["provider_capacity_flag"].astype(float) * 0.8

    # Additional modules mapping
    additional_modules = {
        "upcoding_unbundling": upcoding_df,
        "inflated_bill_cloning": inflated_df,
        "prolonged_stay_readmission": stay_df,
        "medication_device_fraud": med_df,
    }

    for module_name, module_df in additional_modules.items():
        if module_df is None or module_df.empty or "claim_id" not in module_df.columns:
            clean[f"{module_name}_flag"] = 0
            clean[f"{module_name}_score"] = 0.0
            continue

        module_flags = module_df.groupby("claim_id", dropna=False)["risk_score"].max().to_dict()
        score_lookup = {str(claim_id): float(score) for claim_id, score in module_flags.items()}

        clean[f"{module_name}_flag"] = clean["claim_id"].map(lambda claim_id: int(score_lookup.get(str(claim_id), 0.0) > 0.0)).fillna(0).astype(int)
        clean[f"{module_name}_score"] = clean["claim_id"].map(lambda claim_id: float(score_lookup.get(str(claim_id), 0.0))).fillna(0.0)

    additional_columns = [
        col for col in [
            "upcoding_unbundling_score",
            "inflated_bill_cloning_score",
            "prolonged_stay_readmission_score",
            "medication_device_fraud_score",
        ]
        if col in clean.columns
    ]
    if additional_columns:
        additional_total = sum(clean[col].astype(float) for col in additional_columns)
    else:
        additional_total = pd.Series(0.0, index=clean.index)
    clean["additional_fraud_score"] = additional_total.clip(0, 1)
    clean["additional_fraud_flag"] = (clean["additional_fraud_score"] >= 0.5).astype(int)

    clean["business_risk_score"] = (
        clean["repeat_billing_score"] * 0.40 +
        clean["phantom_service_score"] * 0.20 +
        clean["provider_capacity_score"] * 0.15 +
        clean["fuzzy_similarity_score"] * 0.15 +
        clean["additional_fraud_score"] * 0.10
    ).clip(0, 1)
    clean["business_risk_flag"] = (clean["business_risk_score"] >= 0.6).astype(int)

    summary = {
        "total_claims": len(clean),
        "repeat_billing_cases": len(repeat_results),
        "phantom_service_cases": len(phantom_df),
        "provider_capacity_issues": len(provider_df),
        "upcoding_unbundling_cases": int(clean["upcoding_unbundling_flag"].sum()) if "upcoding_unbundling_flag" in clean.columns else 0,
        "inflated_bill_cloning_cases": int(clean["inflated_bill_cloning_flag"].sum()) if "inflated_bill_cloning_flag" in clean.columns else 0,
        "prolonged_stay_readmission_cases": int(clean["prolonged_stay_readmission_flag"].sum()) if "prolonged_stay_readmission_flag" in clean.columns else 0,
        "medication_device_fraud_cases": int(clean["medication_device_fraud_flag"].sum()) if "medication_device_fraud_flag" in clean.columns else 0,
        "avg_business_risk_score": float(clean["business_risk_score"].mean()) if not clean.empty else 0.0,
        "high_risk_claims": int(clean["business_risk_flag"].sum()),
    }
    return clean, summary


def _aggregate_risk_summary(frames: list[pd.DataFrame]) -> Dict[str, Any]:
    if not frames:
        return {
            "total_claims": 0,
            "repeat_billing_cases": 0,
            "phantom_service_cases": 0,
            "provider_capacity_issues": 0,
            "upcoding_unbundling_cases": 0,
            "inflated_bill_cloning_cases": 0,
            "prolonged_stay_readmission_cases": 0,
            "medication_device_fraud_cases": 0,
            "avg_business_risk_score": 0.0,
            "high_risk_claims": 0,
            "duplicate_payment_claims": 0,
            "final_high_risk_claims": 0,
        }

    combined = pd.concat(frames, ignore_index=True)
    summary = {
        "total_claims": int(len(combined)),
        "repeat_billing_cases": int(combined["repeat_billing_flag"].sum()) if "repeat_billing_flag" in combined.columns else 0,
        "phantom_service_cases": int(combined["phantom_service_flag"].sum()) if "phantom_service_flag" in combined.columns else 0,
        "provider_capacity_issues": int(combined["provider_capacity_flag"].sum()) if "provider_capacity_flag" in combined.columns else 0,
        "upcoding_unbundling_cases": int(combined["upcoding_unbundling_flag"].sum()) if "upcoding_unbundling_flag" in combined.columns else 0,
        "inflated_bill_cloning_cases": int(combined["inflated_bill_cloning_flag"].sum()) if "inflated_bill_cloning_flag" in combined.columns else 0,
        "prolonged_stay_readmission_cases": int(combined["prolonged_stay_readmission_flag"].sum()) if "prolonged_stay_readmission_flag" in combined.columns else 0,
        "medication_device_fraud_cases": int(combined["medication_device_fraud_flag"].sum()) if "medication_device_fraud_flag" in combined.columns else 0,
        "avg_business_risk_score": float(combined["business_risk_score"].mean()) if "business_risk_score" in combined.columns else 0.0,
        "high_risk_claims": int(combined["business_risk_flag"].sum()) if "business_risk_flag" in combined.columns else 0,
        "duplicate_payment_claims": int(combined["duplicate_payment_flag"].sum()) if "duplicate_payment_flag" in combined.columns else 0,
        "final_high_risk_claims": int(combined["final_risk_flag"].sum()) if "final_risk_flag" in combined.columns else 0,
    }
    return summary


def run_integrated_claim_risk_pipeline(
    df: pd.DataFrame,
    db_connection=None,
    chunk_size: int | None = None,
    ml_model=None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df is None or df.empty:
        return pd.DataFrame(), {
            "total_claims": 0,
            "repeat_billing_cases": 0,
            "phantom_service_cases": 0,
            "provider_capacity_issues": 0,
            "upcoding_unbundling_cases": 0,
            "inflated_bill_cloning_cases": 0,
            "prolonged_stay_readmission_cases": 0,
            "medication_device_fraud_cases": 0,
            "avg_business_risk_score": 0.0,
            "high_risk_claims": 0,
            "duplicate_payment_claims": 0,
            "final_high_risk_claims": 0,
        }

    enriched, summary = enrich_claims_with_business_risk_features(df)
    validator = ClaimStatusValidator(db_connection=db_connection)
    status_flags = []
    for idx, claim in enriched.iterrows():
        duplicate, history, message = validator.check_duplicate_payment(claim.to_dict())
        matched_statuses = sorted({
            str(item.get("status", "")).upper()
            for item in history if item.get("status")
        })
        status_flags.append({
            "_astina_row_id": claim.get("_astina_row_id", idx),
            "duplicate_payment_flag": int(duplicate),
            "duplicate_payment_status": "|".join(matched_statuses),
            "status_message": message,
        })
    status_df = pd.DataFrame(status_flags)
    if not status_df.empty:
        enriched = enriched.merge(status_df, on="_astina_row_id", how="left", validate="one_to_one")
        enriched["duplicate_payment_flag"] = enriched["duplicate_payment_flag"].fillna(0).astype(int)
    else:
        enriched["duplicate_payment_flag"] = 0
        enriched["duplicate_payment_status"] = ""
        enriched["status_message"] = "Tidak ada riwayat pembayaran"

    enriched["business_risk_score"] = (
        enriched["business_risk_score"]
        + enriched["duplicate_payment_flag"].astype(float) * 0.10
    ).clip(0, 1)
    enriched["business_risk_flag"] = (enriched["business_risk_score"] >= 0.6).astype(int)

    # Hybrid risk calculation (combining rule-based and ML if provided)
    if ml_model is not None and hasattr(ml_model, "predict_proba"):
        try:
            # If ML model score is available
            ml_scores = ml_model.predict_proba(df)[:, 1] if hasattr(ml_model, "predict_proba") else np.zeros(len(df))
            enriched["ml_anomaly_score"] = ml_scores
            enriched["final_risk_score"] = (
                enriched["business_risk_score"] * 0.5 +
                enriched["ml_anomaly_score"] * 0.3 +
                enriched["duplicate_payment_flag"].astype(float) * 0.2
            ).clip(0, 1)
        except Exception as e:
            logger.warning(f"ML scoring fallback to rule score: {e}")
            enriched["final_risk_score"] = (
                enriched["business_risk_score"] * 0.7 +
                enriched["duplicate_payment_flag"].astype(float) * 0.3
            ).clip(0, 1)
    else:
        enriched["final_risk_score"] = (
            enriched["business_risk_score"] * 0.7 +
            enriched["duplicate_payment_flag"].astype(float) * 0.3
        ).clip(0, 1)

    enriched["final_risk_flag"] = (enriched["final_risk_score"] >= 0.65).astype(int)
    enriched["risk_category"] = enriched.apply(derive_risk_category, axis=1)

    summary["duplicate_payment_claims"] = int(enriched["duplicate_payment_flag"].sum())
    summary["final_high_risk_claims"] = int(enriched["final_risk_flag"].sum())
    return enriched, summary
