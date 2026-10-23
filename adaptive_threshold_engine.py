"""
Adaptive Threshold Engine for ASTINA

Calculates dynamic thresholds based on:
- Historical baseline (rolling window)
- Provider-specific patterns
- Seasonal adjustments
- Statistical outlier detection (IQR, Z-score, MAD)
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger("astina.adaptive_threshold")


class AdaptiveThresholdEngine:
    """
    Engine untuk menghitung threshold dinamis berdasarkan multiple methods.
    """

    def __init__(
        self,
        historical_window_days: int = 90,
        min_samples_for_threshold: int = 30,
        default_threshold_multiplier: float = 3.0
    ):
        self.historical_window_days = int(historical_window_days)
        self.min_samples_for_threshold = int(min_samples_for_threshold)
        self.default_threshold_multiplier = float(default_threshold_multiplier)
        self.baseline_cache = {}  # Cache untuk historical baselines

    def calculate_dynamic_threshold(
        self,
        df: pd.DataFrame,
        column: str,
        provider_id: str = None,
        method: str = "iqr",
        date_column: str = "billing_date"
    ) -> Dict[str, float]:
        """
        Calculate dynamic threshold with multiple methods:
        - IQR-based: Q3 + 1.5 * IQR (robust to outliers)
        - Z-score-based: mean + 3 * std (standard)
        - Percentile-based: 95th or 99th percentile
        - Robust MAD: median + 3 * MAD (median absolute deviation)

        Args:
            df: DataFrame dengan data
            column: Column name untuk calculate threshold
            provider_id: Optional provider ID untuk provider-specific threshold
            method: Method to use ('iqr', 'zscore', 'percentile', 'robust_mad')
            date_column: Date column name untuk historical filtering

        Returns:
            Dictionary dengan threshold information
        """
        if df is None or df.empty:
            return self._get_default_threshold(method)

        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return self._get_default_threshold(method)

        # Filter by provider if specified
        if provider_id and "provider_id" in df.columns:
            subset = df[df['provider_id'] == provider_id]
            if len(subset) < self.min_samples_for_threshold:
                logger.info(f"Insufficient data for provider {provider_id}, using global threshold")
                subset = df
        else:
            subset = df

        # Filter by historical window if date column exists
        if date_column in df.columns:
            try:
                subset[date_column] = pd.to_datetime(subset[date_column], errors="coerce")
                cutoff_date = datetime.now() - timedelta(days=self.historical_window_days)
                subset = subset[subset[date_column] >= cutoff_date]
            except Exception as e:
                logger.warning(f"Failed to filter by date: {e}")

        # Check minimum sample size
        if len(subset) < self.min_samples_for_threshold:
            logger.info(f"Insufficient data ({len(subset)} samples), using global threshold")
            subset = df

        values = subset[column].dropna()

        if len(values) < 3:
            logger.warning(f"Insufficient valid values ({len(values)}), using default threshold")
            return self._get_default_threshold(method)

        return self._calculate_threshold_by_method(values, method)

    def _calculate_threshold_by_method(
        self,
        values: pd.Series,
        method: str
    ) -> Dict[str, float]:
        """Calculate threshold using specified method."""
        if method == "iqr":
            return self._calculate_iqr_threshold(values)
        elif method == "zscore":
            return self._calculate_zscore_threshold(values)
        elif method == "percentile":
            return self._calculate_percentile_threshold(values)
        elif method == "robust_mad":
            return self._calculate_mad_threshold(values)
        else:
            logger.warning(f"Unknown method: {method}, using IQR")
            return self._calculate_iqr_threshold(values)

    def _calculate_iqr_threshold(self, values: pd.Series) -> Dict[str, float]:
        """Calculate threshold using IQR method."""
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1

        upper_threshold = Q3 + 1.5 * IQR
        lower_threshold = Q1 - 1.5 * IQR

        return {
            "upper_threshold": float(upper_threshold),
            "lower_threshold": float(lower_threshold),
            "method": "iqr",
            "q1": float(Q1),
            "q3": float(Q3),
            "iqr": float(IQR),
            "sample_size": len(values),
            "is_global": False
        }

    def _calculate_zscore_threshold(self, values: pd.Series) -> Dict[str, float]:
        """Calculate threshold using Z-score method."""
        mean = values.mean()
        std = values.std()

        upper_threshold = mean + 3 * std
        lower_threshold = mean - 3 * std

        return {
            "upper_threshold": float(upper_threshold),
            "lower_threshold": float(lower_threshold),
            "method": "zscore",
            "mean": float(mean),
            "std": float(std),
            "sample_size": len(values),
            "is_global": False
        }

    def _calculate_percentile_threshold(self, values: pd.Series) -> Dict[str, float]:
        """Calculate threshold using percentile method."""
        upper_threshold = values.quantile(0.99)
        lower_threshold = values.quantile(0.01)

        return {
            "upper_threshold": float(upper_threshold),
            "lower_threshold": float(lower_threshold),
            "method": "percentile",
            "percentile_99": float(upper_threshold),
            "percentile_1": float(lower_threshold),
            "sample_size": len(values),
            "is_global": False
        }

    def _calculate_mad_threshold(self, values: pd.Series) -> Dict[str, float]:
        """Calculate threshold using Robust MAD method."""
        median = values.median()
        mad = (values - median).abs().median()

        upper_threshold = median + 3 * mad
        lower_threshold = median - 3 * mad

        return {
            "upper_threshold": float(upper_threshold),
            "lower_threshold": float(lower_threshold),
            "method": "robust_mad",
            "median": float(median),
            "mad": float(mad),
            "sample_size": len(values),
            "is_global": False
        }

    def _get_default_threshold(self, method: str) -> Dict[str, float]:
        """Get default threshold when insufficient data."""
        return {
            "upper_threshold": float('inf'),
            "lower_threshold": float('-inf'),
            "method": method,
            "sample_size": 0,
            "is_global": True,
            "is_default": True
        }

    def calculate_multi_method_threshold(
        self,
        df: pd.DataFrame,
        column: str,
        provider_id: str = None,
        methods: list = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate thresholds using multiple methods and return all results.

        Args:
            df: DataFrame dengan data
            column: Column name
            provider_id: Optional provider ID
            methods: List of methods to use (default: ['iqr', 'zscore', 'percentile', 'robust_mad'])

        Returns:
            Dictionary mapping method name to threshold result
        """
        if methods is None:
            methods = ['iqr', 'zscore', 'percentile', 'robust_mad']

        results = {}
        for method in methods:
            try:
                results[method] = self.calculate_dynamic_threshold(
                    df, column, provider_id, method
                )
            except Exception as e:
                logger.warning(f"Failed to calculate {method} threshold: {e}")
                results[method] = self._get_default_threshold(method)

        return results

    def get_conservative_threshold(
        self,
        df: pd.DataFrame,
        column: str,
        provider_id: str = None
    ) -> Dict[str, float]:
        """
        Get conservative threshold (highest upper threshold from all methods).
        Useful untuk mengurangi false positives.

        Returns:
            Dictionary dengan conservative threshold
        """
        multi_results = self.calculate_multi_method_threshold(df, column, provider_id)

        upper_thresholds = [
            r['upper_threshold'] for r in multi_results.values()
            if not r.get('is_default', False) and np.isfinite(r['upper_threshold'])
        ]

        if not upper_thresholds:
            return self._get_default_threshold('conservative')

        # Use highest upper threshold (most conservative)
        conservative_upper = max(upper_thresholds)

        return {
            "upper_threshold": float(conservative_upper),
            "lower_threshold": float('-inf'),
            "method": "conservative",
            "sample_size": len(df),
            "is_global": False,
            "source_methods": list(multi_results.keys())
        }

    def get_aggressive_threshold(
        self,
        df: pd.DataFrame,
        column: str,
        provider_id: str = None
    ) -> Dict[str, float]:
        """
        Get aggressive threshold (lowest upper threshold from all methods).
        Useful untuk maximizing detection sensitivity.

        Returns:
            Dictionary dengan aggressive threshold
        """
        multi_results = self.calculate_multi_method_threshold(df, column, provider_id)

        upper_thresholds = [
            r['upper_threshold'] for r in multi_results.values()
            if not r.get('is_default', False) and np.isfinite(r['upper_threshold'])
        ]

        if not upper_thresholds:
            return self._get_default_threshold('aggressive')

        # Use lowest upper threshold (most aggressive)
        aggressive_upper = min(upper_thresholds)

        return {
            "upper_threshold": float(aggressive_upper),
            "lower_threshold": float('-inf'),
            "method": "aggressive",
            "sample_size": len(df),
            "is_global": False,
            "source_methods": list(multi_results.keys())
        }

    def update_baseline_cache(
        self,
        df: pd.DataFrame,
        cache_key: str = "global"
    ):
        """
        Update baseline cache dengan current data statistics.

        Args:
            df: DataFrame dengan data
            cache_key: Key untuk cache (e.g., 'global', provider_id)
        """
        if df is None or df.empty:
            return

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        self.baseline_cache[cache_key] = {
            "updated_at": datetime.now().isoformat(),
            "sample_size": len(df),
            "columns": {}
        }

        for col in numeric_columns:
            values = df[col].dropna()
            if len(values) >= 3:
                self.baseline_cache[cache_key]["columns"][col] = {
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                    "mad": float((values - values.median()).abs().median())
                }

        logger.info(f"Updated baseline cache for {cache_key}")

    def get_baseline_value(
        self,
        column: str,
        statistic: str = "median",
        cache_key: str = "global"
    ) -> Optional[float]:
        """
        Get baseline value dari cache.

        Args:
            column: Column name
            statistic: Statistic to retrieve ('mean', 'median', 'std', 'q1', 'q3', 'iqr', 'mad')
            cache_key: Cache key

        Returns:
            Baseline value or None if not found
        """
        if cache_key not in self.baseline_cache:
            return None

        if column not in self.baseline_cache[cache_key]["columns"]:
            return None

        return self.baseline_cache[cache_key]["columns"][column].get(statistic)

    def clear_baseline_cache(self, cache_key: str = None):
        """
        Clear baseline cache.

        Args:
            cache_key: Specific key to clear, or None to clear all
        """
        if cache_key:
            if cache_key in self.baseline_cache:
                del self.baseline_cache[cache_key]
                logger.info(f"Cleared baseline cache for {cache_key}")
        else:
            self.baseline_cache.clear()
            logger.info("Cleared all baseline cache")
