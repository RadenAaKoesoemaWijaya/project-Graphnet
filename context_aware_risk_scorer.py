"""
Context-Aware Risk Scorer for ASTINA

Calculates composite risk score considering multiple factors:
- Historical baseline for patient/provider
- Cross-module signal correlation
- Seasonal patterns
- Provider reputation score
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger("astina.context_risk")


class ContextAwareRiskScorer:
    """
    Calculates composite risk score with context awareness.
    """

    def __init__(self):
        self.patient_history = {}  # patient_id -> historical_claims
        self.provider_reputation = {}  # provider_id -> reputation_score (0-1)
        self.seasonal_patterns = {}  # (month, service_type) -> expected_volume
        self.reputation_cache = {}  # Cache untuk reputation calculations

    def calculate_composite_risk_score(
        self,
        claim: pd.Series,
        module_flags: Dict[str, int],
        module_scores: Dict[str, float],
        df_historical: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Calculate composite risk score with context awareness.

        Args:
            claim: Claim data as Series
            module_flags: Dictionary mapping module_name -> flag (0/1)
            module_scores: Dictionary mapping module_name -> risk_score (0-1)
            df_historical: Historical claims data untuk context

        Returns:
            Dictionary dengan composite risk score and factors
        """
        patient_id = claim.get("patient_id")
        provider_id = claim.get("provider_id")

        # Factor 1: Historical baseline for patient
        patient_risk_factor = self._get_patient_risk_factor(
            patient_id, claim, df_historical
        )

        # Factor 2: Provider reputation
        provider_risk_factor = self._get_provider_risk_factor(
            provider_id, df_historical
        )

        # Factor 3: Cross-module correlation
        correlation_factor = self._calculate_module_correlation(module_flags)

        # Factor 4: Seasonal adjustment
        seasonal_factor = self._get_seasonal_adjustment(claim)

        # Factor 5: Aggregate module scores
        aggregate_module_score = self._aggregate_module_scores(module_scores)

        # Composite score (weighted)
        composite_score = (
            aggregate_module_score * 0.40 +
            patient_risk_factor * 0.20 +
            provider_risk_factor * 0.15 +
            correlation_factor * 0.15 +
            seasonal_factor * 0.10
        )

        # Determine if flag should be suppressed
        should_suppress = self._should_suppress_flag(
            composite_score,
            patient_risk_factor,
            provider_risk_factor
        )

        suppression_reason = self._get_suppression_reason(
            patient_risk_factor,
            provider_risk_factor,
            composite_score
        )

        return {
            "composite_risk_score": float(np.clip(composite_score, 0.0, 1.0)),
            "patient_risk_factor": patient_risk_factor,
            "provider_risk_factor": provider_risk_factor,
            "correlation_factor": correlation_factor,
            "seasonal_factor": seasonal_factor,
            "aggregate_module_score": aggregate_module_score,
            "should_suppress_flag": should_suppress,
            "suppression_reason": suppression_reason,
            "modules_triggered": [k for k, v in module_flags.items() if v == 1]
        }

    def _get_patient_risk_factor(
        self,
        patient_id: str,
        claim: pd.Series,
        df_historical: pd.DataFrame = None
    ) -> float:
        """
        Calculate patient risk factor based on historical patterns.

        Returns:
            Risk factor between 0.0 (low risk) and 1.0 (high risk)
        """
        if not patient_id or df_historical is None or df_historical.empty:
            return 0.5  # Neutral score if no historical data

        # Get patient's historical claims
        patient_claims = df_historical[df_historical['patient_id'] == patient_id]

        if len(patient_claims) < 3:
            return 0.5  # Insufficient data

        # Calculate risk based on:
        # 1. Frequency of claims
        # 2. Previous flag history
        # 3. Amount patterns

        risk_factors = []

        # Factor 1: Claim frequency (high frequency = higher risk)
        time_span_days = (
            pd.to_datetime(patient_claims['billing_date']).max() -
            pd.to_datetime(patient_claims['billing_date']).min()
        ).days
        if time_span_days > 0:
            claims_per_month = len(patient_claims) / (time_span_days / 30)
            frequency_risk = min(claims_per_month / 10.0, 1.0)  # Cap at 10 claims/month
            risk_factors.append(frequency_risk)

        # Factor 2: Previous flags (more flags = higher risk)
        flag_columns = [c for c in patient_claims.columns if c.endswith('_flag')]
        if flag_columns:
            total_flags = patient_claims[flag_columns].sum().sum()
            flag_ratio = total_flags / (len(patient_claims) * len(flag_columns))
            risk_factors.append(flag_ratio)

        # Factor 3: Amount variance (high variance = higher risk)
        if 'amount' in patient_claims.columns:
            amount_std = patient_claims['amount'].std()
            amount_mean = patient_claims['amount'].mean()
            if amount_mean > 0:
                coefficient_of_variation = amount_std / amount_mean
                cv_risk = min(coefficient_of_variation / 2.0, 1.0)
                risk_factors.append(cv_risk)

        # Average all risk factors
        if risk_factors:
            return float(np.mean(risk_factors))
        else:
            return 0.5

    def _get_provider_risk_factor(
        self,
        provider_id: str,
        df_historical: pd.DataFrame = None
    ) -> float:
        """
        Calculate provider reputation/risk factor.

        Returns:
            Risk factor between 0.0 (good reputation) and 1.0 (high risk)
        """
        if not provider_id or df_historical is None or df_historical.empty:
            return 0.5  # Neutral score

        # Check cache first
        cache_key = f"{provider_id}_{len(df_historical)}"
        if cache_key in self.reputation_cache:
            return self.reputation_cache[cache_key]

        # Get provider's historical claims
        provider_claims = df_historical[df_historical['provider_id'] == provider_id]

        if len(provider_claims) < 10:
            return 0.5  # Insufficient data

        risk_factors = []

        # Factor 1: Flag rate for provider
        flag_columns = [c for c in provider_claims.columns if c.endswith('_flag')]
        if flag_columns:
            total_flags = provider_claims[flag_columns].sum().sum()
            flag_ratio = total_flags / (len(provider_claims) * len(flag_columns))
            risk_factors.append(flag_ratio)

        # Factor 2: Amount patterns
        if 'amount' in provider_claims.columns:
            # Check for unusual high amounts
            provider_mean = provider_claims['amount'].mean()
            global_mean = df_historical['amount'].mean()
            if global_mean > 0:
                amount_ratio = provider_mean / global_mean
                amount_risk = min(max(amount_ratio - 1.0, 0) / 2.0, 1.0)
                risk_factors.append(amount_risk)

        # Factor 3: Unique patients (low diversity = higher risk)
        unique_patients = provider_claims['patient_id'].nunique()
        diversity_ratio = unique_patients / len(provider_claims)
        diversity_risk = 1.0 - diversity_ratio  # Low diversity = high risk
        risk_factors.append(diversity_risk)

        # Average all risk factors
        if risk_factors:
            reputation_score = float(np.mean(risk_factors))
        else:
            reputation_score = 0.5

        # Cache the result
        self.reputation_cache[cache_key] = reputation_score
        self.provider_reputation[provider_id] = reputation_score

        return reputation_score

    def _calculate_module_correlation(
        self,
        module_flags: Dict[str, int]
    ) -> float:
        """
        Calculate correlation factor based on multiple modules triggering.

        Returns:
            Correlation factor between 0.0 (no correlation) and 1.0 (high correlation)
        """
        if not module_flags:
            return 0.0

        triggered_modules = [k for k, v in module_flags.items() if v == 1]
        total_modules = len(module_flags)

        if total_modules == 0:
            return 0.0

        # More modules triggering = higher correlation risk
        correlation_score = len(triggered_modules) / total_modules

        # Weight by module type (some combinations are more suspicious)
        high_risk_combinations = [
            {'repeat_billing', 'inflated_bill_cloning'},
            {'repeat_billing', 'phantom_service'},
            {'upcoding_unbundling', 'inflated_bill_cloning'},
            {'provider_capacity', 'phantom_service'}
        ]

        triggered_set = set(triggered_modules)
        for combo in high_risk_combinations:
            if combo.issubset(triggered_set):
                correlation_score = min(correlation_score + 0.2, 1.0)
                break

        return float(correlation_score)

    def _get_seasonal_adjustment(
        self,
        claim: pd.Series
    ) -> float:
        """
        Calculate seasonal adjustment factor.

        Returns:
            Seasonal factor between 0.0 (low risk season) and 1.0 (high risk season)
        """
        try:
            billing_date = pd.to_datetime(claim.get("billing_date"), errors="coerce")
            if pd.isna(billing_date):
                return 0.5

            month = billing_date.month

            # Define seasonal patterns (can be customized)
            # Higher risk during holiday seasons (Dec) and flu season (Jan-Feb)
            high_risk_months = [12, 1, 2]  # Holiday and flu season
            medium_risk_months = [6, 7, 8]  # Summer vacation

            if month in high_risk_months:
                return 0.6  # Slightly higher risk
            elif month in medium_risk_months:
                return 0.4  # Slightly lower risk
            else:
                return 0.5  # Neutral
        except Exception as e:
            logger.warning(f"Failed to calculate seasonal adjustment: {e}")
            return 0.5

    def _aggregate_module_scores(
        self,
        module_scores: Dict[str, float]
    ) -> float:
        """
        Aggregate module scores into single score.

        Returns:
            Aggregated score between 0.0 and 1.0
        """
        if not module_scores:
            return 0.0

        scores = [v for v in module_scores.values() if v is not None]

        if not scores:
            return 0.0

        # Use max score (most conservative)
        return float(max(scores))

    def _should_suppress_flag(
        self,
        composite_score: float,
        patient_risk_factor: float,
        provider_risk_factor: float
    ) -> bool:
        """
        Determine if flag should be suppressed based on context.

        Returns:
            True if flag should be suppressed, False otherwise
        """
        # Suppress if composite score is very low
        if composite_score < 0.25:
            return True

        # Suppress if patient has low historical risk AND provider has good reputation
        if patient_risk_factor < 0.2 and provider_risk_factor < 0.3:
            return True

        # Suppress if composite score is low and both factors are neutral
        if composite_score < 0.35 and patient_risk_factor < 0.4 and provider_risk_factor < 0.4:
            return True

        return False

    def _get_suppression_reason(
        self,
        patient_risk_factor: float,
        provider_risk_factor: float,
        composite_score: float
    ) -> str:
        """Get reason for flag suppression."""
        reasons = []

        if composite_score < 0.25:
            reasons.append("Low composite risk score")

        if patient_risk_factor < 0.2:
            reasons.append("Low patient historical risk")

        if provider_risk_factor < 0.3:
            reasons.append("Good provider reputation")

        if patient_risk_factor < 0.4 and provider_risk_factor < 0.4 and composite_score < 0.35:
            reasons.append("Neutral risk factors")

        return "; ".join(reasons) if reasons else ""

    def update_patient_history(
        self,
        df: pd.DataFrame
    ):
        """
        Update patient history cache.

        Args:
            df: DataFrame dengan claims data
        """
        if df is None or df.empty or 'patient_id' not in df.columns:
            return

        for patient_id, group in df.groupby('patient_id'):
            self.patient_history[patient_id] = {
                "claim_count": len(group),
                "last_claim_date": pd.to_datetime(group['billing_date']).max().isoformat(),
                "total_amount": float(group['amount'].sum()) if 'amount' in group.columns else 0.0
            }

        logger.info(f"Updated patient history for {len(self.patient_history)} patients")

    def set_provider_reputation(
        self,
        provider_id: str,
        reputation_score: float
    ):
        """
        Manually set provider reputation score.

        Args:
            provider_id: Provider identifier
            reputation_score: Score between 0.0 (good) and 1.0 (high risk)
        """
        self.provider_reputation[provider_id] = float(np.clip(reputation_score, 0.0, 1.0))
        logger.info(f"Set provider reputation for {provider_id}: {reputation_score}")

    def get_context_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about context data.

        Returns:
            Dictionary dengan statistics
        """
        return {
            "patients_cached": len(self.patient_history),
            "providers_cached": len(self.provider_reputation),
            "reputation_cache_size": len(self.reputation_cache),
            "seasonal_patterns_count": len(self.seasonal_patterns)
        }
