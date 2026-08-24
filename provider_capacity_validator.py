from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

from config import SERVICE_CAPACITY

logger = logging.getLogger("astina.provider_capacity")

SERVICE_CODE_TO_TYPE = {
    "CT001": "CT_SCAN",
    "XR001": "X_RAY",
    "BT001": "BLOOD_TEST",
    "CONS001": "CONSULTATION",
    "US001": "ULTRASOUND",
    "LAB001": "LABORATORY",
    "SUR001": "SURGERY",
}


class ProviderCapacityValidator:
    """Check whether provider capacity is feasible for a given service schedule."""

    def __init__(self, service_capacity: Dict = None):
        self.service_capacity = service_capacity or SERVICE_CAPACITY

    def _normalize_service_type(self, code: str) -> str:
        code_str = str(code or "").strip().upper()
        return SERVICE_CODE_TO_TYPE.get(code_str, code_str)

    def validate_provider_schedule(self, df: pd.DataFrame, provider_id: str, service_date) -> Tuple[bool, List[str], float]:
        if df is None or df.empty:
            return True, [], 0.0
        if provider_id is None:
            return True, [], 0.0

        date_value = pd.to_datetime(service_date, errors="coerce")
        if pd.isna(date_value):
            return True, [], 0.0

        df_dates = pd.to_datetime(df.get("service_date"), errors="coerce")
        subset = df[(df.get("provider_id") == provider_id) & (df_dates.dt.date == date_value.date())]
        if subset.empty:
            return True, [], 0.0

        violations: List[str] = []
        total_minutes = 0.0

        # Map each row's service code to normalized type
        service_types = subset.get("service_code", pd.Series(dtype=object)).map(self._normalize_service_type)
        type_counts = service_types.value_counts()

        for service_type, count in type_counts.items():
            service_config = self.service_capacity.get(service_type)
            if not service_config:
                continue

            max_allowed = int(service_config.get("max_per_day", 999))
            if count > max_allowed:
                violations.append(f"Provider {provider_id} melebihi kapasitas {service_type}: {count}/{max_allowed} layanan per hari.")
            duration = float(service_config.get("duration_minutes", 0.0))
            total_minutes += duration * count

        if total_minutes > 480:
            violations.append(f"Provider {provider_id} melebihi total kapasitas harian: {total_minutes:.0f}/480 menit.")

        utilization = min(1.0, total_minutes / 480.0) if total_minutes else 0.0
        return len(violations) == 0, violations, float(utilization)

    def validate_all_providers_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized grouped validation for all providers in the dataset."""
        if df is None or df.empty or "provider_id" not in df.columns or "service_date" not in df.columns:
            return pd.DataFrame(columns=["provider_id", "service_date", "utilization_pct", "violations"])

        clean = df.copy()
        clean["_service_date_clean"] = pd.to_datetime(clean["service_date"], errors="coerce").dt.normalize()
        valid = clean[clean["_service_date_clean"].notna() & clean["provider_id"].notna()]
        if valid.empty:
            return pd.DataFrame(columns=["provider_id", "service_date", "utilization_pct", "violations"])

        results = []
        grouped = valid.groupby(["provider_id", "_service_date_clean"])
        for (provider_id, s_date), group in grouped:
            is_feasible, violations, utilization = self.validate_provider_schedule(group, provider_id, s_date)
            if not is_feasible:
                results.append({
                    "provider_id": provider_id,
                    "service_date": s_date,
                    "utilization_pct": utilization,
                    "violations": "; ".join(violations),
                })

        return pd.DataFrame(results) if results else pd.DataFrame(columns=["provider_id", "service_date", "utilization_pct", "violations"])
