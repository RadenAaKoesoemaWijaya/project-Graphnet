from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("astina.medication_device_fraud")

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


def detect_medication_and_device_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect medication and device billing manipulation.

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

    required_columns = {"claim_id", "patient_id", "provider_id", "amount"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        logger.warning("Missing required columns for medication/device checks: %s", missing)
        return _empty_fraud_result()

    clean_df = df.copy()
    for col in ["item_code", "item_name", "quantity_billed", "quantity_delivered", "unit_price", "diagnosis_code"]:
        if col not in clean_df.columns:
            clean_df[col] = ""
    clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce").fillna(0.0)
    clean_df["quantity_billed"] = pd.to_numeric(clean_df["quantity_billed"], errors="coerce").fillna(0.0)
    clean_df["quantity_delivered"] = pd.to_numeric(clean_df["quantity_delivered"], errors="coerce").fillna(0.0)
    clean_df["unit_price"] = pd.to_numeric(clean_df["unit_price"], errors="coerce").fillna(0.0)

    results: List[Dict[str, Any]] = []

    for idx, row in clean_df.iterrows():
        issues: List[str] = []
        score = 0.0
        amount = _safe_numeric(row.get("amount"), 0.0)
        quantity_billed = _safe_numeric(row.get("quantity_billed"), 0.0)
        quantity_delivered = _safe_numeric(row.get("quantity_delivered"), 0.0)
        unit_price = _safe_numeric(row.get("unit_price"), 0.0)

        if quantity_billed > 0:
            discrepancy = abs(quantity_billed - quantity_delivered) / max(quantity_billed, 1.0)
            if discrepancy > 0.3:
                issues.append("Billed quantity materially exceeds delivered quantity.")
                score = max(score, 0.7)

        if unit_price > 0:
            median_price = float(clean_df["unit_price"].median() or 0.0)
            if median_price > 0 and unit_price > median_price * 2.5:
                issues.append("Unit price is substantially above normal benchmark.")
                score = max(score, 0.75)

        if amount > 0 and quantity_billed > 0 and unit_price > 0:
            computed_amount = quantity_billed * unit_price
            if abs(amount - computed_amount) / max(abs(computed_amount), 1.0) > 0.4:
                issues.append("Claim amount is inconsistent with billed quantity and unit price.")
                score = max(score, 0.65)

        if score <= 0.0:
            continue

        if score >= 0.75:
            severity = "high"
        elif score >= 0.45:
            severity = "medium"
        else:
            severity = "low"

        evidence = {
            "item_code": row.get("item_code", ""),
            "item_name": row.get("item_name", ""),
            "quantity_billed": quantity_billed,
            "quantity_delivered": quantity_delivered,
            "unit_price": unit_price,
            "amount": amount,
            "rule_matches": issues,
        }

        results.append({
            "claim_id": row.get("claim_id", idx),
            "patient_id": row.get("patient_id", ""),
            "provider_id": row.get("provider_id", ""),
            "module_name": "medication_device_fraud",
            "flag": 1,
            "risk_score": round(float(min(score, 1.0)), 4),
            "severity": severity,
            "reason": "; ".join(issues),
            "evidence": str(evidence),
        })

    return pd.DataFrame(results, columns=FRAUD_COLUMNS) if results else _empty_fraud_result()


class MedicationDeviceFraudDetector:
    """Template class wrapper around the medication/device fraud detector."""

    def __init__(self):
        self.module_name = "medication_device_fraud"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_medication_and_device_fraud(df)
