from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("astina.inflated_bill_cloning")

FRAUD_COLUMNS = [
    "claim_id",
    "patient_id",
    "provider_id",
    "module_name",
    "flag",
    "risk_score",
    "severity",
    "reason",
    "evidence",
]


def _empty_fraud_result() -> pd.DataFrame:
    return pd.DataFrame(columns=FRAUD_COLUMNS)


def _safe_numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def detect_inflated_bill_and_cloning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect inflated billing and cloning patterns.

    Output schema:
    - claim_id: claim identifier
    - patient_id: patient identifier
    - provider_id: provider identifier
    - module_name: 'inflated_bill_cloning'
    - flag: int 0/1
    - risk_score: float between 0 and 1
    - severity: 'low' | 'medium' | 'high'
    - reason: string explanation
    - evidence: dictionary summary encoded as string
    """
    if df is None or df.empty:
        return _empty_fraud_result()

    required_columns = {"claim_id", "patient_id", "provider_id", "amount"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        logger.warning("Missing required columns for inflated bill detection: %s", missing)
        return _empty_fraud_result()

    clean_df = df.copy()
    for col in ["diagnosis_code", "procedure_code", "service_date"]:
        if col not in clean_df.columns:
            clean_df[col] = ""
    clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce").fillna(0.0)

    results: List[Dict[str, Any]] = []

    for idx, row in clean_df.iterrows():
        issues: List[str] = []
        score = 0.0
        amount = _safe_numeric(row.get("amount"), 0.0)
        patient_id = str(row.get("patient_id") or "")
        provider_id = str(row.get("provider_id") or "")
        service_date = str(row.get("service_date") or "")
        diagnosis = str(row.get("diagnosis_code") or "")

        if amount > 0:
            median_amount = float(clean_df["amount"].median() or 0.0)
            if median_amount > 0 and amount > median_amount * 2.5:
                issues.append("Claim amount is far above the median benchmark.")
                score = max(score, 0.7)

        same_patient_records = clean_df[
            (clean_df["patient_id"].astype(str) == patient_id)
            & (clean_df["provider_id"].astype(str) == provider_id)
            & (clean_df["service_date"].astype(str) == service_date)
        ]
        if not same_patient_records.empty and len(same_patient_records) > 1:
            issues.append("Multiple similar claims exist for the same patient/provider/date combo.")
            score = max(score, 0.8)

        if diagnosis and diagnosis in clean_df["diagnosis_code"].astype(str).values:
            same_diag = clean_df[(clean_df["diagnosis_code"].astype(str) == diagnosis)]
            if not same_diag.empty and len(same_diag) > 2:
                issues.append("Repeated diagnosis pattern suggests possible cloning or repeated billing behavior.")
                score = max(score, 0.5)

        if score <= 0.0:
            continue

        if score >= 0.75:
            severity = "high"
        elif score >= 0.45:
            severity = "medium"
        else:
            severity = "low"

        evidence = {
            "amount": amount,
            "patient_id": patient_id,
            "provider_id": provider_id,
            "service_date": service_date,
            "diagnosis_code": diagnosis,
            "rule_matches": issues,
        }

        results.append({
            "claim_id": row.get("claim_id", idx),
            "patient_id": patient_id,
            "provider_id": provider_id,
            "module_name": "inflated_bill_cloning",
            "flag": 1,
            "risk_score": round(float(min(score, 1.0)), 4),
            "severity": severity,
            "reason": "; ".join(issues),
            "evidence": str(evidence),
        })

    return pd.DataFrame(results, columns=FRAUD_COLUMNS) if results else _empty_fraud_result()


class InflatedBillCloningDetector:
    """Template class wrapper around the inflated-bill and cloning detector."""

    def __init__(self):
        self.module_name = "inflated_bill_cloning"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_inflated_bill_and_cloning(df)
