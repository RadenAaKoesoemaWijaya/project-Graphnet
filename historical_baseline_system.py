"""
Historical Baseline System for ASTINA

Maintains and updates historical baselines for:
- Provider claim patterns
- Patient claim frequency
- Service cost distributions
- Seasonal variations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os

logger = logging.getLogger("astina.historical_baseline")


class HistoricalBaselineSystem:
    """
    Maintains and updates historical baselines.
    """

    def __init__(self, storage_path: str = "cache/baselines/"):
        self.storage_path = storage_path
        self.baselines = {}
        self.load_baselines()

    def update_baseline(
        self,
        df: pd.DataFrame,
        time_window_days: int = 90,
        force_update: bool = False
    ):
        """
        Update baselines with new data.
        Should be called periodically (e.g., weekly/monthly).

        Args:
            df: DataFrame dengan claims data
            time_window_days: Time window untuk historical data
            force_update: Force update even if recently updated
        """
        if df is None or df.empty:
            logger.warning("Cannot update baseline: empty DataFrame")
            return

        logger.info(f"Updating baselines with {len(df)} records")

        # Update provider baselines
        provider_baseline = self._calculate_provider_baseline(df)
        self.baselines["provider"] = provider_baseline

        # Update service cost baselines
        cost_baseline = self._calculate_cost_baseline(df)
        self.baselines["cost"] = cost_baseline

        # Update seasonal baselines
        seasonal_baseline = self._calculate_seasonal_baseline(df)
        self.baselines["seasonal"] = seasonal_baseline

        # Update patient frequency baselines
        patient_baseline = self._calculate_patient_baseline(df)
        self.baselines["patient"] = patient_baseline

        # Add metadata
        self.baselines["metadata"] = {
            "updated_at": datetime.now().isoformat(),
            "record_count": len(df),
            "time_window_days": time_window_days
        }

        self.save_baselines()
        logger.info("Baselines updated successfully")

    def _calculate_provider_baseline(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate provider-level baselines."""
        if "provider_id" not in df.columns:
            return {}

        provider_stats = {}

        for provider_id, group in df.groupby("provider_id"):
            stats = {
                "total_claims": len(group),
                "unique_patients": group["patient_id"].nunique() if "patient_id" in group.columns else 0,
                "avg_daily_claims": 0.0,
                "total_amount": 0.0,
                "avg_amount": 0.0,
                "amount_std": 0.0,
                "last_claim_date": None
            }

            if "amount" in group.columns:
                stats["total_amount"] = float(group["amount"].sum())
                stats["avg_amount"] = float(group["amount"].mean())
                stats["amount_std"] = float(group["amount"].std())

            if "billing_date" in group.columns:
                dates = pd.to_datetime(group["billing_date"], errors="coerce")
                valid_dates = dates.dropna()
                if not valid_dates.empty:
                    stats["last_claim_date"] = valid_dates.max().isoformat()

                    # Calculate daily claims
                    date_range = (valid_dates.max() - valid_dates.min()).days
                    if date_range > 0:
                        stats["avg_daily_claims"] = len(group) / date_range

            provider_stats[provider_id] = stats

        return provider_stats

    def _calculate_cost_baseline(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate cost distribution baselines."""
        if "amount" not in df.columns:
            return {}

        amounts = df["amount"].dropna()

        if amounts.empty:
            return {}

        return {
            "global": {
                "count": len(amounts),
                "mean": float(amounts.mean()),
                "median": float(amounts.median()),
                "std": float(amounts.std()),
                "min": float(amounts.min()),
                "max": float(amounts.max()),
                "q1": float(amounts.quantile(0.25)),
                "q3": float(amounts.quantile(0.75)),
                "iqr": float(amounts.quantile(0.75) - amounts.quantile(0.25)),
                "p95": float(amounts.quantile(0.95)),
                "p99": float(amounts.quantile(0.99))
            },
            "by_service": self._calculate_cost_by_service(df)
        }

    def _calculate_cost_by_service(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate cost baselines by service type."""
        if "service_code" not in df.columns or "amount" not in df.columns:
            return {}

        service_stats = {}

        for service_code, group in df.groupby("service_code"):
            amounts = group["amount"].dropna()
            if len(amounts) >= 3:
                service_stats[service_code] = {
                    "count": len(amounts),
                    "mean": float(amounts.mean()),
                    "median": float(amounts.median()),
                    "std": float(amounts.std()),
                    "p95": float(amounts.quantile(0.95))
                }

        return service_stats

    def _calculate_seasonal_baseline(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate seasonal patterns."""
        if "billing_date" not in df.columns:
            return {}

        df_copy = df.copy()
        df_copy["billing_date"] = pd.to_datetime(df_copy["billing_date"], errors="coerce")
        df_copy = df_copy.dropna(subset=["billing_date"])

        if df_copy.empty:
            return {}

        df_copy["month"] = df_copy["billing_date"].dt.month
        df_copy["day_of_week"] = df_copy["billing_date"].dt.dayofweek
        df_copy["hour"] = df_copy["billing_date"].dt.hour

        seasonal_stats = {
            "by_month": {},
            "by_day_of_week": {},
            "by_hour": {}
        }

        # Monthly patterns
        for month, group in df_copy.groupby("month"):
            seasonal_stats["by_month"][int(month)] = {
                "claim_count": len(group),
                "avg_amount": float(group["amount"].mean()) if "amount" in group.columns else 0.0
            }

        # Day of week patterns
        for dow, group in df_copy.groupby("day_of_week"):
            seasonal_stats["by_day_of_week"][int(dow)] = {
                "claim_count": len(group),
                "avg_amount": float(group["amount"].mean()) if "amount" in group.columns else 0.0
            }

        # Hourly patterns
        for hour, group in df_copy.groupby("hour"):
            seasonal_stats["by_hour"][int(hour)] = {
                "claim_count": len(group),
                "avg_amount": float(group["amount"].mean()) if "amount" in group.columns else 0.0
            }

        return seasonal_stats

    def _calculate_patient_baseline(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate patient-level baselines."""
        if "patient_id" not in df.columns:
            return {}

        patient_stats = {}

        for patient_id, group in df.groupby("patient_id"):
            stats = {
                "total_claims": len(group),
                "unique_providers": group["provider_id"].nunique() if "provider_id" in group.columns else 0,
                "total_amount": 0.0,
                "avg_amount": 0.0,
                "claim_frequency": 0.0
            }

            if "amount" in group.columns:
                stats["total_amount"] = float(group["amount"].sum())
                stats["avg_amount"] = float(group["amount"].mean())

            if "billing_date" in group.columns:
                dates = pd.to_datetime(group["billing_date"], errors="coerce")
                valid_dates = dates.dropna()
                if not valid_dates.empty:
                    date_range = (valid_dates.max() - valid_dates.min()).days
                    if date_range > 0:
                        stats["claim_frequency"] = len(group) / date_range

            patient_stats[patient_id] = stats

        return patient_stats

    def get_baseline_value(
        self,
        baseline_type: str,
        key: str,
        metric: str,
        default: Any = None
    ) -> Any:
        """
        Get baseline value for specific key and metric.

        Args:
            baseline_type: Type of baseline ('provider', 'cost', 'seasonal', 'patient')
            key: Key (e.g., provider_id, service_code, month)
            metric: Metric name (e.g., 'mean', 'median', 'count')
            default: Default value if not found

        Returns:
            Baseline value or default
        """
        if baseline_type not in self.baselines:
            return default

        baseline = self.baselines[baseline_type]

        if baseline_type == "cost":
            if key == "global":
                return baseline.get("global", {}).get(metric, default)
            elif key == "by_service":
                return baseline.get("by_service", {})
            else:
                return baseline.get("by_service", {}).get(key, {}).get(metric, default)
        else:
            return baseline.get(key, {}).get(metric, default)

    def detect_concept_drift(
        self,
        current_df: pd.DataFrame,
        threshold: float = 0.2
    ) -> Dict[str, bool]:
        """
        Detect if current data deviates significantly from baseline.

        Args:
            current_df: Current DataFrame to check
            threshold: Drift threshold (default 0.2 = 20% deviation)

        Returns:
            Dictionary mapping metric -> drift_detected (bool)
        """
        drift_results = {}

        if "cost" not in self.baselines:
            logger.warning("No cost baseline available for drift detection")
            return drift_results

        cost_baseline = self.baselines["cost"].get("global", {})

        if "amount" not in current_df.columns:
            return drift_results

        current_amounts = current_df["amount"].dropna()

        if current_amounts.empty:
            return drift_results

        # Compare mean
        baseline_mean = cost_baseline.get("mean", 0)
        current_mean = current_amounts.mean()

        if baseline_mean > 0:
            drift_pct = abs(current_mean - baseline_mean) / baseline_mean
            drift_results["cost_mean"] = drift_pct > threshold

        # Compare median
        baseline_median = cost_baseline.get("median", 0)
        current_median = current_amounts.median()

        if baseline_median > 0:
            drift_pct = abs(current_median - baseline_median) / baseline_median
            drift_results["cost_median"] = drift_pct > threshold

        # Compare std
        baseline_std = cost_baseline.get("std", 0)
        current_std = current_amounts.std()

        if baseline_std > 0:
            drift_pct = abs(current_std - baseline_std) / baseline_std
            drift_results["cost_std"] = drift_pct > threshold

        # Compare volume
        baseline_count = cost_baseline.get("count", 0)
        current_count = len(current_amounts)

        if baseline_count > 0:
            drift_pct = abs(current_count - baseline_count) / baseline_count
            drift_results["volume"] = drift_pct > threshold

        # Log drift detection
        drifted_metrics = [k for k, v in drift_results.items() if v]
        if drifted_metrics:
            logger.warning(f"Concept drift detected in: {', '.join(drifted_metrics)}")

        return drift_results

    def save_baselines(self):
        """Save baselines to storage."""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            file_path = os.path.join(self.storage_path, "baselines.json")

            with open(file_path, 'w') as f:
                json.dump(self.baselines, f, indent=2, default=str)

            logger.info(f"Baselines saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    def load_baselines(self):
        """Load baselines from storage."""
        try:
            file_path = os.path.join(self.storage_path, "baselines.json")

            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.baselines = json.load(f)
                logger.info(f"Baselines loaded from {file_path}")
            else:
                logger.info("No existing baselines found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load baselines: {e}")
            self.baselines = {}

    def get_baseline_summary(self) -> Dict[str, Any]:
        """
        Get summary of current baselines.

        Returns:
            Dictionary dengan summary information
        """
        summary = {
            "last_updated": self.baselines.get("metadata", {}).get("updated_at", "Never"),
            "record_count": self.baselines.get("metadata", {}).get("record_count", 0),
            "baseline_types": list(self.baselines.keys())
        }

        if "provider" in self.baselines:
            summary["provider_count"] = len(self.baselines["provider"])

        if "patient" in self.baselines:
            summary["patient_count"] = len(self.baselines["patient"])

        if "cost" in self.baselines:
            summary["service_count"] = len(self.baselines["cost"].get("by_service", {}))

        return summary

    def clear_baselines(self):
        """Clear all baselines."""
        self.baselines = {}
        logger.info("All baselines cleared")
