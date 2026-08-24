from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("astina.length_of_stay")

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


def detect_prolonged_stay_and_readmission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect prolonged stay and readmission anomalies.

    Output schema:
    - claim_id
    - patient_id
    - provider_id
    - module_name
    - flag
    - risk_score
    - severity
    - reason
    - evidence
    """
    if df is None or df.empty:
        return _empty_fraud_result()

    required_columns = {"claim_id", "patient_id", "provider_id", "admission_date", "discharge_date"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        logger.warning("Missing required columns for LOS detection: %s", missing)
        return _empty_fraud_result()

    clean_df = df.copy()
    for col in ["diagnosis_code", "procedure_code"]:
        if col not in clean_df.columns:
            clean_df[col] = ""

    clean_df["admission_date"] = pd.to_datetime(clean_df["admission_date"], errors="coerce")
    clean_df["discharge_date"] = pd.to_datetime(clean_df["discharge_date"], errors="coerce")
    clean_df = clean_df.dropna(subset=["admission_date", "discharge_date"]).copy()
    if clean_df.empty:
        return _empty_fraud_result()

    clean_df["los_days"] = (clean_df["discharge_date"] - clean_df["admission_date"]).dt.days
    results: List[Dict[str, Any]] = []

    for idx, row in clean_df.iterrows():
        issues: List[str] = []
        score = 0.0
        patient_id = str(row.get("patient_id") or "")
        provider_id = str(row.get("provider_id") or "")
        diagnosis = str(row.get("diagnosis_code") or "")
        los_days = int(row.get("los_days") or 0)

        if los_days > 0:
            median_los = float(clean_df["los_days"].median() or 0.0)
            if median_los > 0 and los_days > median_los * 1.5:
                issues.append("Length of stay is significantly above benchmark.")
                score = max(score, 0.7)

        same_patient = clean_df[clean_df["patient_id"].astype(str) == patient_id]
        if not same_patient.empty:
            same_patient = same_patient.sort_values("admission_date")
            patient_dates = same_patient["admission_date"].tolist()
            for prev_date in patient_dates:
                if pd.isna(prev_date):
                    continue
                gap_days = abs((pd.Timestamp(row["admission_date"]) - pd.Timestamp(prev_date)).days)
                if 0 < gap_days <= 30 and str(row.get("diagnosis_code") or "") == str(same_patient.loc[same_patient["admission_date"] == prev_date, "diagnosis_code"].iloc[0]) if not same_patient.empty else "":
                    issues.append("Readmission within a short interval with similar diagnosis pattern.")
                    score = max(score, 0.8)
                    break

        if score <= 0.0:
            continue

        if score >= 0.75:
            severity = "high"
        elif score >= 0.45:
            severity = "medium"
        else:
            severity = "low"

        evidence = {
            "los_days": los_days,
            "patient_id": patient_id,
            "provider_id": provider_id,
            "diagnosis_code": diagnosis,
            "rule_matches": issues,
        }

        results.append({
            "claim_id": row.get("claim_id", idx),
            "patient_id": patient_id,
            "provider_id": provider_id,
            "module_name": "prolonged_stay_readmission",
            "flag": 1,
            "risk_score": round(float(min(score, 1.0)), 4),
            "severity": severity,
            "reason": "; ".join(issues),
            "evidence": str(evidence),
        })

    return pd.DataFrame(results, columns=FRAUD_COLUMNS) if results else _empty_fraud_result()


class LengthOfStayDetector:
    """Template class wrapper around the prolonged-stay and readmission detector."""

    def __init__(self):
        self.module_name = "prolonged_stay_readmission"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_prolonged_stay_and_readmission(df)
