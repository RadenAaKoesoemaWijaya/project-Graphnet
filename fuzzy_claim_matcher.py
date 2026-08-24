from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from fuzzywuzzy import fuzz
except ImportError:  # pragma: no cover
    class _FallbackFuzz:
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
            return (len(overlap) / len(union)) * 100.0 if union else 0.0

        @staticmethod
        def ratio(a: str, b: str) -> float:
            import difflib

            return difflib.SequenceMatcher(None, str(a), str(b)).ratio() * 100.0

    fuzz = _FallbackFuzz()

logger = logging.getLogger("astina.fuzzy_matcher")


class FuzzyClaimMatcher:
    """Weighted similarity scorer for near-duplicate claims."""

    def __init__(self, string_threshold: float = 0.85, amount_threshold: float = 0.95, amount_variance_pct: float = 5.0):
        self.string_threshold = float(string_threshold)
        self.amount_threshold = float(amount_threshold)
        self.amount_variance_pct = float(amount_variance_pct)

    def calculate_claim_similarity(self, claim1: Dict, claim2: Dict, weight_config: Optional[Dict] = None) -> float:
        if weight_config is None:
            weight_config = {
                "service_code": 0.25,
                "patient_id": 0.25,
                "provider_id": 0.15,
                "amount": 0.20,
                "service_date": 0.10,
                "diagnosis_code": 0.05,
            }

        scores = {}
        scores["service_code"] = self._string_similarity(claim1.get("service_code", ""), claim2.get("service_code", ""))
        scores["patient_id"] = 1.0 if str(claim1.get("patient_id", "")) == str(claim2.get("patient_id", "")) else 0.0
        scores["provider_id"] = 1.0 if str(claim1.get("provider_id", "")) == str(claim2.get("provider_id", "")) else 0.0
        scores["amount"] = self._amount_similarity(float(claim1.get("amount", 0.0) or 0.0), float(claim2.get("amount", 0.0) or 0.0))
        scores["service_date"] = self._date_similarity(claim1.get("service_date"), claim2.get("service_date"))
        scores["diagnosis_code"] = self._string_similarity(claim1.get("diagnosis_code", ""), claim2.get("diagnosis_code", ""))

        total_weight = sum(weight_config.values())
        if total_weight <= 0:
            return 0.0

        weighted_sum = 0.0
        for field, weight in weight_config.items():
            if field in scores:
                weighted_sum += scores[field] * weight
        return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))

    def _string_similarity(self, str1: object, str2: object) -> float:
        if pd.isna(str1) or pd.isna(str2):
            return 0.0
        if str(str1) == str(str2):
            return 1.0
        ratio = fuzz.token_set_ratio(str(str1), str(str2)) / 100.0
        return float(np.clip(ratio, 0.0, 1.0))

    def _amount_similarity(self, amount1: float, amount2: float) -> float:
        if amount1 == 0 and amount2 == 0:
            return 1.0
        if amount1 == 0 or amount2 == 0:
            return 0.0

        variance = abs(amount1 - amount2) / max(abs(amount1), abs(amount2), 1.0)
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
            gap_days = abs((pd.to_datetime(date2) - pd.to_datetime(date1)).days)
            if gap_days <= 7:
                return 0.9
            if gap_days <= 14:
                return 0.7
            return 0.5
        except Exception:
            return 0.0

    def find_similar_claims(self, df: pd.DataFrame, target_claim_index: int, time_window_days: int = 30, min_similarity: Optional[float] = None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["claim_id", "similarity_score", "time_gap_days", "index"])
        if not 0 <= target_claim_index < len(df):
            raise IndexError("target_claim_index out of bounds")

        if min_similarity is None:
            min_similarity = self.string_threshold

        target_claim = df.iloc[target_claim_index].to_dict()
        similar = []
        target_patient = str(target_claim.get("patient_id", ""))
        target_provider = str(target_claim.get("provider_id", ""))
        target_service = str(target_claim.get("service_code", "")).strip()
        target_diagnosis = str(target_claim.get("diagnosis_code", "")).strip()
        target_amount = float(target_claim.get("amount", 0.0) or 0.0)

        try:
            target_date = pd.to_datetime(target_claim.get("billing_date"))
        except Exception:
            target_date = pd.NaT

        candidates = df[df.index != target_claim_index].copy()
        if target_date is pd.NaT:
            return pd.DataFrame(columns=["claim_id", "similarity_score", "time_gap_days", "index"])

        for idx, candidate in candidates.iterrows():
            if str(candidate.get("patient_id", "")) != target_patient:
                continue
            if str(candidate.get("provider_id", "")) != target_provider:
                continue

            candidate_service = str(candidate.get("service_code", "")).strip()
            if target_service and candidate_service and candidate_service != target_service:
                continue

            candidate_diagnosis = str(candidate.get("diagnosis_code", "")).strip()
            if target_diagnosis and candidate_diagnosis and candidate_diagnosis != target_diagnosis:
                continue

            try:
                cand_date = pd.to_datetime(candidate.get("billing_date"))
                time_gap = abs((cand_date - target_date).days)
            except Exception:
                continue
            if time_gap > time_window_days:
                continue

            candidate_amount = float(candidate.get("amount", 0.0) or 0.0)
            amount_ratio = abs(candidate_amount - target_amount) / max(max(abs(candidate_amount), abs(target_amount)), 1.0)
            if amount_ratio > 0.35:
                continue

            score = self.calculate_claim_similarity(target_claim, candidate.to_dict())
            if score >= min_similarity:
                similar.append({
                    "claim_id": candidate.get("claim_id", idx),
                    "similarity_score": score,
                    "time_gap_days": time_gap,
                    "index": idx,
                })

        return pd.DataFrame(similar).sort_values("similarity_score", ascending=False).reset_index(drop=True)
