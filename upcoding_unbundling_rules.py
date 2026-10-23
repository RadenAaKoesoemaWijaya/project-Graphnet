from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import numpy as np

try:
    from robust_outlier_detector import RobustOutlierDetector
    _has_robust_detector = True
except ImportError:
    _has_robust_detector = False
    RobustOutlierDetector = None

logger = logging.getLogger("astina.upcoding_unbundling")

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


def detect_upcoding_and_unbundling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect upcoding and unbundling patterns in claim data.

    Output schema:
    - claim_id: identifier of the claim flagged
    - patient_id: patient identifier
    - provider_id: provider identifier
    - module_name: 'upcoding_unbundling'
    - flag: integer 0/1
    - risk_score: float in [0.0, 1.0]
    - severity: 'low' | 'medium' | 'high'
    - reason: concise explanation
    - evidence: JSON-like string or dictionary-like summary
    """
    if df is None or df.empty:
        return _empty_fraud_result()

    required_columns = {"claim_id", "patient_id", "provider_id", "amount"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        logger.warning("Missing required columns for upcoding detection: %s", missing)
        return _empty_fraud_result()

    clean_df = df.copy()
    for col in ["claim_id", "patient_id", "provider_id", "diagnosis_code", "procedure_code"]:
        if col not in clean_df.columns:
            clean_df[col] = ""
    clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce").fillna(0.0)

    # Use robust outlier detection if available
    if _has_robust_detector and len(clean_df) >= 10:
        detector = RobustOutlierDetector()
        amount_outliers = detector.detect_outliers_mad(clean_df["amount"], threshold=3.0)
        outlier_scores = detector.calculate_outlier_score(clean_df["amount"], method="mad")
    else:
        amount_outliers = pd.Series([False] * len(clean_df), index=clean_df.index)
        outlier_scores = pd.Series([0.0] * len(clean_df), index=clean_df.index)

    results: List[Dict[str, Any]] = []

    for idx, row in clean_df.iterrows():
        issues: List[str] = []
        score = 0.0
        diagnosis = str(row.get("diagnosis_code") or "").strip()
        procedure = str(row.get("procedure_code") or "").strip()
        amount = _safe_numeric(row.get("amount"), 0.0)

        if diagnosis and procedure and diagnosis.lower() == procedure.lower():
            issues.append("Diagnosis and procedure appear indistinguishable or duplicated.")
            score = max(score, 0.35)

        # Check amount outlier using robust detection
        if amount > 0 and amount_outliers.loc[idx]:
            outlier_score = outlier_scores.loc[idx]
            if outlier_score > 0.7:  # Extreme outlier
                issues.append(f"Claim amount is extreme outlier (score: {outlier_score:.2f})")
                score = max(score, 0.6)
            elif outlier_score > 0.4:  # Moderate outlier
                issues.append(f"Claim amount is moderate outlier (score: {outlier_score:.2f})")
                score = max(score, 0.4)
        elif amount > 0 and not _has_robust_detector:
            # Fallback to median-based for small datasets
            if amount > float(clean_df["amount"].median() or 0.0) * 2.5:
                issues.append("Claim amount exceeds the median benchmark substantially.")
                score = max(score, 0.6)

        if diagnosis and procedure and len(diagnosis) > 0 and len(procedure) > 0 and diagnosis[0] != procedure[0]:
            if amount > 0:
                issues.append("Diagnosis/procedure pattern is unusually inconsistent with typical billing behavior.")
                score = max(score, 0.4)

        if score <= 0.0:
            continue

        if score >= 0.75:
            severity = "high"
        elif score >= 0.45:
            severity = "medium"
        else:
            severity = "low"

        evidence = {
            "diagnosis_code": diagnosis,
            "procedure_code": procedure,
            "amount": amount,
            "outlier_score": float(outlier_scores.loc[idx]) if _has_robust_detector else 0.0,
            "rule_matches": issues,
        }

        results.append({
            "claim_id": row.get("claim_id", idx),
            "patient_id": row.get("patient_id", ""),
            "provider_id": row.get("provider_id", ""),
            "module_name": "upcoding_unbundling",
            "flag": 1,
            "risk_score": round(float(min(score, 1.0)), 4),
            "severity": severity,
            "reason": "; ".join(issues),
            "evidence": str(evidence),
        })

    return pd.DataFrame(results, columns=FRAUD_COLUMNS) if results else _empty_fraud_result()


class UpcodingUnbundlingDetector:
    """Template class wrapper around the rule-based detector."""

    def __init__(self):
        self.module_name = "upcoding_unbundling"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_upcoding_and_unbundling(df)
