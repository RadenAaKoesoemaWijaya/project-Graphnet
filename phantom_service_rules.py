from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("astina.phantom_service")


class PhantomServiceRuleEngine:
    """Rule-based validation for phantom service claims with vectorized batch support."""

    def __init__(self):
        self.valid_services = {
            "CT001": "CT_SCAN",
            "XR001": "X_RAY",
            "BT001": "BLOOD_TEST",
            "CONS001": "CONSULTATION",
            "US001": "ULTRASOUND",
            "LAB001": "LABORATORY",
            "SUR001": "SURGERY",
        }

        self.max_frequency_per_day = {
            "CT_SCAN": 2,
            "X_RAY": 4,
            "BLOOD_TEST": 5,
            "CONSULTATION": 10,
            "ULTRASOUND": 5,
            "LABORATORY": 10,
            "SURGERY": 1,
        }

        self.provider_specialization = {}
        self.unrealistic_patterns = [
            {"service_type": "CT_SCAN", "patient_age_max": 1, "reason": "CT Scan tidak seharusnya untuk pasien usia < 1 bulan."},
            {"service_type": "SURGERY", "max_per_day": 1, "reason": "Tidak boleh ada lebih dari 1 operasi per hari untuk pasien yang sama."},
        ]

    def validate_claim(self, claim: Dict) -> Tuple[bool, List[str]]:
        violations: List[str] = []
        service_code = str(claim.get("service_code", "") or "")
        if not service_code:
            violations.append("Kode layanan tidak ditemukan.")
            return False, violations

        if service_code not in self.valid_services:
            violations.append(f"Kode layanan tidak valid: {service_code}")
            return False, violations

        service_type = self.valid_services[service_code]
        provider_id = claim.get("provider_id")
        if provider_id in self.provider_specialization and service_type not in self.provider_specialization[provider_id]:
            violations.append(f"Provider {provider_id} tidak memiliki spesialisasi untuk {service_type}.")

        patient_age = claim.get("patient_age")
        if patient_age is not None:
            try:
                patient_age = float(patient_age)
            except Exception:
                patient_age = None

        for pattern in self.unrealistic_patterns:
            if pattern.get("service_type") == service_type:
                if pattern.get("patient_age_max") is not None and patient_age is not None and patient_age < pattern["patient_age_max"]:
                    violations.append(pattern["reason"])
                if pattern.get("max_per_day") is not None and patient_age is not None and patient_age <= 0:
                    violations.append(pattern["reason"])

        return len(violations) == 0, violations

    def validate_claims_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized batch validation across an entire claims dataframe for massive performance boost."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["claim_id", "service_code", "phantom_violations"])

        clean = df.copy()
        if "claim_id" not in clean.columns:
            clean["claim_id"] = clean.index.astype(str)
        if "service_code" not in clean.columns:
            clean["service_code"] = ""

        violations_series = pd.Series([""] * len(clean), index=clean.index, dtype=object)

        # 1. Missing / Invalid service code check
        service_codes = clean["service_code"].fillna("").astype(str).str.strip()
        missing_mask = service_codes == ""
        invalid_mask = (~service_codes.isin(self.valid_services)) & (~missing_mask)

        violations_series[missing_mask] = violations_series[missing_mask].apply(
            lambda v: (v + "; " if v else "") + "Kode layanan tidak ditemukan."
        )
        violations_series[invalid_mask] = [
            (v + "; " if v else "") + f"Kode layanan tidak valid: {code}"
            for v, code in zip(violations_series[invalid_mask], service_codes[invalid_mask])
        ]

        # 2. Unrealistic pattern check (age-based)
        if "patient_age" in clean.columns:
            patient_ages = pd.to_numeric(clean["patient_age"], errors="coerce")
            ct_under_age = (service_codes == "CT001") & (patient_ages < 1) & (patient_ages.notna())
            violations_series[ct_under_age] = violations_series[ct_under_age].apply(
                lambda v: (v + "; " if v else "") + "CT Scan tidak seharusnya untuk pasien usia < 1 bulan."
            )

        # 3. Frequency check per patient + service + service_date
        if "patient_id" in clean.columns and "service_date" in clean.columns:
            dates = pd.to_datetime(clean["service_date"], errors="coerce")
            valid_date_mask = dates.notna()
            if valid_date_mask.any():
                sub_freq = clean[valid_date_mask].copy()
                sub_freq["_date_only"] = dates[valid_date_mask].dt.date
                counts = sub_freq.groupby(["patient_id", "service_code", "_date_only"])["claim_id"].transform("count")
                service_types = sub_freq["service_code"].map(self.valid_services)
                max_limits = service_types.map(self.max_frequency_per_day).fillna(10)
                freq_violations = counts > max_limits
                if freq_violations.any():
                    violating_indices = sub_freq[freq_violations].index
                    violations_series[violating_indices] = violations_series[violating_indices].apply(
                        lambda v: (v + "; " if v else "") + "Frekuensi layanan melebihi batas harian yang wajar."
                    )

        has_violation = violations_series != ""
        if not has_violation.any():
            return pd.DataFrame(columns=["claim_id", "service_code", "phantom_violations"])

        res = pd.DataFrame({
            "claim_id": clean.loc[has_violation, "claim_id"],
            "service_code": clean.loc[has_violation, "service_code"],
            "phantom_violations": violations_series[has_violation],
        }).reset_index(drop=True)
        return res

    def check_frequency_violation(self, df: pd.DataFrame, patient_id: str, service_code: str, service_date) -> Tuple[bool, Optional[str]]:
        if df is None or df.empty:
            return False, None
        if service_code not in self.valid_services:
            return False, None

        service_type = self.valid_services[service_code]
        max_allowed = self.max_frequency_per_day.get(service_type, 10)
        service_date = pd.to_datetime(service_date, errors="coerce")
        if pd.isna(service_date):
            return False, None

        subset = df[(df.get("patient_id", pd.Series(dtype=object)) == patient_id) & (df.get("service_code", pd.Series(dtype=object)) == service_code)]
        if subset.empty:
            return False, None

        matched_dates = pd.to_datetime(subset.get("service_date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
        same_day_count = int((matched_dates == service_date).sum())
        if same_day_count > max_allowed:
            return True, f"Melebihi batas maksimal: {same_day_count} {service_type} dalam 1 hari (max: {max_allowed})."
        return False, None

    def add_custom_rule(self, rule_name: str, validation_func) -> None:
        setattr(self, f"_rule_{rule_name}", validation_func)
        logger.info("Added custom rule: %s", rule_name)

    def validate_provider_service_match(self, provider_id: str, service_code: str) -> bool:
        if service_code not in self.valid_services:
            return False
        if not provider_id:
            return True
        if provider_id not in self.provider_specialization:
            return True
        service_type = self.valid_services[service_code]
        return service_type in self.provider_specialization[provider_id]
