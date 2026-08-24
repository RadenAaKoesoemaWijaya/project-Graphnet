from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from fuzzywuzzy import fuzz
except ImportError:  # pragma: no cover
    class _FallbackFuzz:
        @staticmethod
        def ratio(a: str, b: str) -> float:
            import difflib

            return difflib.SequenceMatcher(None, str(a), str(b)).ratio() * 100.0

        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            import difflib

            left = set(str(a).split())
            right = set(str(b).split())
            if not left and not right:
                return 100.0
            if not left or not right:
                return 0.0
            union = left | right
            overlap = left & right
            score = len(overlap) / len(union) if union else 0.0
            return score * 100.0

    fuzz = _FallbackFuzz()

logger = logging.getLogger("astina.repeat_billing")


class RepeatBillingDetector:
    """Detect repeat billing patterns in claim history within a temporal window."""

    def __init__(
        self,
        temporal_window_days: int = 30,
        fuzzy_threshold: float = 0.85,
        amount_variance_pct: float = 5.0,
        verbose: bool = False,
    ):
        self.temporal_window = int(temporal_window_days)
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.amount_variance_pct = float(amount_variance_pct)
        self.verbose = bool(verbose)

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def detect_repeat_claims(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                "first_claim_id",
                "repeat_claim_id",
                "patient_id",
                "provider_id",
                "service_code",
                "first_billing_date",
                "repeat_billing_date",
                "time_gap_days",
                "first_amount",
                "repeat_amount",
                "amount_variance_pct",
                "similarity_score",
                "risk_score",
                "detection_reason",
            ])

        required = {"claim_id", "patient_id", "provider_id", "service_code", "billing_date", "amount"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        clean_df = df.copy()
        clean_df["billing_date"] = pd.to_datetime(clean_df["billing_date"], errors="coerce")
        clean_df = clean_df.dropna(subset=["billing_date"]).copy()

        if clean_df.empty:
            return pd.DataFrame(columns=[
                "first_claim_id",
                "repeat_claim_id",
                "patient_id",
                "provider_id",
                "service_code",
                "first_billing_date",
                "repeat_billing_date",
                "time_gap_days",
                "first_amount",
                "repeat_amount",
                "amount_variance_pct",
                "similarity_score",
                "risk_score",
                "detection_reason",
            ])

        results: List[Dict] = []
        grouped = clean_df.groupby(["patient_id", "provider_id", "service_code"], dropna=False)

        for (patient_id, provider_id, service_code), group in grouped:
            if len(group) <= 1:
                continue

            ordered = group.sort_values("billing_date").reset_index(drop=True)

            # Sliding-window comparison keeps only claims within the active temporal window.
            # This preserves the original logic while avoiding unnecessary comparisons outside
            # the valid repeat-billing timeframe.
            for i in range(len(ordered) - 1):
                claim1 = ordered.iloc[i]
                claim1_date = pd.Timestamp(claim1["billing_date"])

                for j in range(i + 1, len(ordered)):
                    claim2 = ordered.iloc[j]
                    claim2_date = pd.Timestamp(claim2["billing_date"])
                    time_gap = int((claim2_date - claim1_date).days)

                    if time_gap <= 0:
                        continue
                    if time_gap > self.temporal_window:
                        break

                    similarity = self._calculate_claim_similarity(claim1, claim2)
                    variance = self._calculate_amount_variance(float(claim1.get("amount", 0.0) or 0.0), float(claim2.get("amount", 0.0) or 0.0))
                    if similarity < self.fuzzy_threshold:
                        continue

                    risk_score = self._calculate_risk_score(time_gap, similarity, variance)
                    results.append({
                        "first_claim_id": claim1.get("claim_id"),
                        "repeat_claim_id": claim2.get("claim_id"),
                        "patient_id": patient_id,
                        "provider_id": provider_id,
                        "service_code": service_code,
                        "first_billing_date": claim1["billing_date"].date(),
                        "repeat_billing_date": claim2["billing_date"].date(),
                        "time_gap_days": time_gap,
                        "first_amount": float(claim1.get("amount", 0.0) or 0.0),
                        "repeat_amount": float(claim2.get("amount", 0.0) or 0.0),
                        "amount_variance_pct": variance,
                        "similarity_score": similarity,
                        "risk_score": risk_score,
                        "detection_reason": self._get_reason(time_gap, similarity, variance),
                    })

        result_df = pd.DataFrame(results, columns=[
            "first_claim_id",
            "repeat_claim_id",
            "patient_id",
            "provider_id",
            "service_code",
            "first_billing_date",
            "repeat_billing_date",
            "time_gap_days",
            "first_amount",
            "repeat_amount",
            "amount_variance_pct",
            "similarity_score",
            "risk_score",
            "detection_reason",
        ])
        if result_df.empty:
            return result_df.copy()

        if self.verbose:
            logger.info("Detected %s possible repeat billing cases", len(result_df))
        return result_df.sort_values(["risk_score", "similarity_score"], ascending=False).reset_index(drop=True)

    def _calculate_claim_similarity(self, claim1: pd.Series, claim2: pd.Series) -> float:
        service_code_score = self._string_similarity(claim1.get("service_code", ""), claim2.get("service_code", ""))
        diagnosis_score = self._string_similarity(claim1.get("diagnosis_code", ""), claim2.get("diagnosis_code", ""))
        amount_score = self._amount_similarity(float(claim1.get("amount", 0.0) or 0.0), float(claim2.get("amount", 0.0) or 0.0))

        service_date1 = pd.to_datetime(claim1.get("service_date", claim1.get("billing_date")), errors="coerce")
        service_date2 = pd.to_datetime(claim2.get("service_date", claim2.get("billing_date")), errors="coerce")
        date_score = self._date_similarity(service_date1, service_date2)

        weights = [0.35, 0.20, 0.25, 0.20]
        combined = [service_code_score, diagnosis_score, amount_score, date_score]
        total_weight = sum(weights)
        return float(np.clip(np.average(combined, weights=weights), 0.0, 1.0)) if total_weight else 0.0

    def _string_similarity(self, value1: object, value2: object) -> float:
        if pd.isna(value1) or pd.isna(value2):
            return 0.0
        if str(value1) == str(value2):
            return 1.0
        ratio = fuzz.ratio(str(value1), str(value2)) / 100.0
        return float(np.clip(ratio, 0.0, 1.0))

    def _amount_similarity(self, amount1: float, amount2: float) -> float:
        if amount1 == 0 and amount2 == 0:
            return 1.0
        if amount1 == 0 or amount2 == 0:
            return 0.0

        variance = self._calculate_amount_variance(amount1, amount2) / 100.0
        tolerance = max(self.amount_variance_pct / 100.0, 1e-6)
        if variance <= tolerance:
            return 1.0
        return float(max(0.0, 1.0 - (variance - tolerance) / tolerance))

    def _date_similarity(self, date1, date2) -> float:
        if pd.isna(date1) or pd.isna(date2):
            return 0.0
        if date1 == date2:
            return 1.0
        try:
            gap_days = abs((pd.Timestamp(date2) - pd.Timestamp(date1)).days)
            if gap_days <= 3:
                return 0.95
            if gap_days <= 7:
                return 0.85
            if gap_days <= 14:
                return 0.7
            return 0.5
        except Exception:
            return 0.0

    def _calculate_amount_variance(self, amount1: float, amount2: float) -> float:
        if not np.isfinite(amount1) or not np.isfinite(amount2):
            return 100.0
        a1 = float(amount1)
        a2 = float(amount2)
        max_amount = max(abs(a1), abs(a2), 1.0)
        variance = abs(a1 - a2) / max_amount * 100.0
        return float(np.clip(variance, 0.0, 100.0))

    def _calculate_risk_score(self, time_gap: int, similarity: float, amount_variance: float) -> float:
        if time_gap <= 7:
            time_factor = 0.95
        elif time_gap <= 14:
            time_factor = 0.85
        elif time_gap <= 21:
            time_factor = 0.75
        else:
            time_factor = 0.6

        similarity_factor = float(np.clip(similarity, 0.0, 1.0))
        amount_factor = 0.95 if amount_variance <= self.amount_variance_pct else 0.8 if amount_variance <= self.amount_variance_pct * 2 else 0.6

        risk = (time_factor * 0.35) + (similarity_factor * 0.45) + (amount_factor * 0.20)
        return float(np.clip(risk, 0.0, 1.0))

    def _get_reason(self, time_gap: int, similarity: float, amount_variance: float) -> str:
        reasons = []
        if time_gap <= 14:
            reasons.append(f"Jarak klaim {time_gap} hari")
        if similarity >= 0.95:
            reasons.append("Klaim hampir identik")
        elif similarity >= 0.85:
            reasons.append("Klaim sangat mirip")
        if amount_variance <= self.amount_variance_pct:
            reasons.append("Nilai klaim konsisten")
        if not reasons:
            return "Pola repeat billing terdeteksi"
        return "; ".join(reasons)

    def get_repeat_claim_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        result = self.detect_repeat_claims(df)
        if result.empty:
            return []
        return list(zip(result["first_claim_id"].astype(str), result["repeat_claim_id"].astype(str)))
