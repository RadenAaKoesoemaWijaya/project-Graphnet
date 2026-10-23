"""
Robust Outlier Detection for ASTINA

Provides robust statistical methods for outlier detection to reduce false positives:
- Median Absolute Deviation (MAD)
- Interquartile Range (IQR)
- Modified Z-Score
- Isolation Forest
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("astina.robust_outlier")


class RobustOutlierDetector:
    """
    Robust outlier detection using multiple statistical methods.
    More robust than standard deviation for skewed distributions.
    """

    @staticmethod
    def detect_outliers_mad(
        values: pd.Series,
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers using Median Absolute Deviation.
        More robust than standard deviation for skewed distributions.

        Args:
            values: Series of numeric values
            threshold: MAD threshold (default 3.0)

        Returns:
            Boolean Series indicating outliers
        """
        if values.empty or len(values) < 3:
            return pd.Series([False] * len(values), index=values.index)

        median = values.median()
        mad = (values - median).abs().median()

        # Avoid division by zero
        if mad == 0:
            return pd.Series([False] * len(values), index=values.index)

        # Modified Z-score
        modified_z_scores = 0.6745 * (values - median) / mad
        outliers = (modified_z_scores.abs() > threshold)

        return outliers

    @staticmethod
    def detect_outliers_iqr(
        values: pd.Series,
        iqr_multiplier: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers using IQR method.
        Standard method in boxplot analysis.

        Args:
            values: Series of numeric values
            iqr_multiplier: IQR multiplier (default 1.5)

        Returns:
            Boolean Series indicating outliers
        """
        if values.empty or len(values) < 4:
            return pd.Series([False] * len(values), index=values.index)

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1

        # Avoid division by zero
        if IQR == 0:
            return pd.Series([False] * len(values), index=values.index)

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        outliers = (values < lower_bound) | (values > upper_bound)

        return outliers

    @staticmethod
    def detect_outliers_isolation_forest(
        values: pd.Series,
        contamination: float = 0.05,
        random_state: int = 42
    ) -> pd.Series:
        """
        Detect outliers using Isolation Forest.
        Good for high-dimensional data.

        Args:
            values: Series of numeric values
            contamination: Expected proportion of outliers (default 0.05)
            random_state: Random seed for reproducibility

        Returns:
            Boolean Series indicating outliers
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("scikit-learn not available, falling back to MAD method")
            return RobustOutlierDetector.detect_outliers_mad(values)

        if values.empty or len(values) < 10:
            return pd.Series([False] * len(values), index=values.index)

        clf = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )

        values_reshaped = values.values.reshape(-1, 1)
        predictions = clf.fit_predict(values_reshaped)

        outliers = pd.Series(predictions == -1, index=values.index)

        return outliers

    @staticmethod
    def calculate_outlier_score(
        values: pd.Series,
        method: str = "mad"
    ) -> pd.Series:
        """
        Calculate outlier score (0-1) for each value.
        Higher score = more likely to be outlier.

        Args:
            values: Series of numeric values
            method: Detection method ('mad', 'iqr', 'isolation_forest')

        Returns:
            Series of outlier scores (0-1)
        """
        if values.empty:
            return pd.Series([0.0] * len(values), index=values.index)

        if method == "mad":
            median = values.median()
            mad = (values - median).abs().median()

            if mad == 0:
                return pd.Series([0.0] * len(values), index=values.index)

            # Modified Z-score converted to 0-1 scale
            modified_z_scores = 0.6745 * (values - median) / mad
            scores = np.clip(np.abs(modified_z_scores) / 5.0, 0.0, 1.0)  # Normalize by 5 MAD
            return pd.Series(scores, index=values.index)

        elif method == "iqr":
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                return pd.Series([0.0] * len(values), index=values.index)

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Calculate distance from bounds
            lower_dist = np.maximum(0, lower_bound - values)
            upper_dist = np.maximum(0, values - upper_bound)
            max_dist = max(lower_dist.max(), upper_dist.max(), 1.0)

            scores = (lower_dist + upper_dist) / max_dist
            return pd.Series(np.clip(scores, 0.0, 1.0), index=values.index)

        elif method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest
            except ImportError:
                logger.warning("scikit-learn not available, falling back to MAD method")
                return RobustOutlierDetector.calculate_outlier_score(values, "mad")

            if len(values) < 10:
                return pd.Series([0.0] * len(values), index=values.index)

            clf = IsolationForest(
                contamination=0.05,
                random_state=42
            )

            values_reshaped = values.values.reshape(-1, 1)
            scores = -clf.score_samples(values_reshaped)  # Negative score = more anomalous

            # Normalize to 0-1
            min_score = scores.min()
            max_score = scores.max()
            if max_score - min_score > 0:
                normalized = (scores - min_score) / (max_score - min_score)
            else:
                normalized = np.zeros_like(scores)

            return pd.Series(normalized, index=values.index)

        else:
            logger.warning(f"Unknown method: {method}, falling back to MAD")
            return RobustOutlierDetector.calculate_outlier_score(values, "mad")

    @staticmethod
    def get_method_statistics(
        values: pd.Series,
        method: str = "mad"
    ) -> Dict[str, float]:
        """
        Get statistics about the outlier detection method.

        Args:
            values: Series of numeric values
            method: Detection method

        Returns:
            Dictionary with statistics
        """
        if values.empty:
            return {}

        stats = {
            "count": len(values),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max())
        }

        if method == "mad":
            median = values.median()
            mad = (values - median).abs().median()
            stats["mad"] = float(mad)
            stats["q1"] = float(values.quantile(0.25))
            stats["q3"] = float(values.quantile(0.75))

        elif method == "iqr":
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            stats["q1"] = float(Q1)
            stats["q3"] = float(Q3)
            stats["iqr"] = float(IQR)

        return stats
