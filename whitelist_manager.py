"""
Whitelist Manager for ASTINA

Manages whitelist and exception rules for legitimate business scenarios to reduce false positives:
- Chronic condition whitelist
- Provider volume exceptions
- Service combination whitelist
- Patient history exceptions
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger("astina.whitelist")


class WhitelistManager:
    """
    Manages whitelist and exception rules for legitimate business scenarios.
    """

    def __init__(self):
        # Chronic condition ICD-10 codes requiring regular treatment
        self.chronic_condition_whitelist = {
            # Diabetes - regular lab tests and medication
            "E11": {
                "condition": "Type 2 Diabetes",
                "expected_frequency_days": 30,  # Monthly
                "allowed_services": ["BT001", "LAB001", "CONS001"],
                "description": "Regular diabetes management"
            },
            "E10": {
                "condition": "Type 1 Diabetes",
                "expected_frequency_days": 30,
                "allowed_services": ["BT001", "LAB001", "CONS001"],
                "description": "Regular diabetes management"
            },
            # Hypertension - regular BP checks
            "I10": {
                "condition": "Essential Hypertension",
                "expected_frequency_days": 30,
                "allowed_services": ["CONS001", "BT001"],
                "description": "Regular BP monitoring"
            },
            "I11": {
                "condition": "Hypertensive Heart Disease",
                "expected_frequency_days": 30,
                "allowed_services": ["CONS001", "BT001", "XR001"],
                "description": "Regular cardiac monitoring"
            },
            # Asthma - regular inhaler prescriptions
            "J45": {
                "condition": "Asthma",
                "expected_frequency_days": 30,
                "allowed_services": ["CONS001", "LAB001"],
                "description": "Regular asthma management"
            },
            "J46": {
                "condition": "Chronic Obstructive Pulmonary Disease",
                "expected_frequency_days": 30,
                "allowed_services": ["CONS001", "BT001", "XR001"],
                "description": "Regular COPD management"
            },
            # Chronic pain - physical therapy
            "M54": {
                "condition": "Low Back Pain",
                "expected_frequency_days": 14,
                "allowed_services": ["CONS001", "US001", "XR001"],
                "description": "Regular pain management"
            },
            "M53": {
                "condition": "Other Spinal Disorders",
                "expected_frequency_days": 14,
                "allowed_services": ["CONS001", "US001"],
                "description": "Regular spinal management"
            },
            # Kidney disease - regular dialysis
            "N18": {
                "condition": "Chronic Kidney Disease",
                "expected_frequency_days": 7,
                "allowed_services": ["LAB001", "CONS001"],
                "description": "Regular dialysis monitoring"
            },
        }

        # Provider volume exceptions (provider_id -> config)
        self.provider_volume_exceptions = {}

        # Service combination whitelist (diagnosis_prefix, service_code) -> allowed
        self.service_combination_whitelist = {}

        # Patient history exceptions (patient_id -> history)
        self.patient_history_exceptions = {}

        # Load from storage if available
        self._load_whitelist_from_storage()

    def is_legitimate_repeat(
        self,
        diagnosis_code: str,
        service_code: str,
        patient_id: str,
        time_gap_days: int
    ) -> Tuple[bool, str]:
        """
        Check if repeat billing is legitimate based on:
        - Chronic condition diagnosis
        - Expected treatment frequency
        - Patient history

        Args:
            diagnosis_code: ICD-10 diagnosis code
            service_code: Service code
            patient_id: Patient identifier
            time_gap_days: Days between claims

        Returns:
            (is_legitimate, reason)
        """
        if not diagnosis_code:
            return False, ""

        # Check chronic condition whitelist
        for prefix, config in self.chronic_condition_whitelist.items():
            if diagnosis_code.startswith(prefix):
                # Chronic condition - check if repeat is within expected frequency
                expected_freq = config["expected_frequency_days"]
                allowed_services = config["allowed_services"]

                # Check if service is allowed for this condition
                if service_code in allowed_services or not allowed_services:
                    if time_gap_days <= expected_freq:
                        condition = config["condition"]
                        return True, f"Legitimate repeat for {condition} (expected frequency: {expected_freq} days)"
                    elif time_gap_days <= expected_freq * 2:
                        # Slightly outside expected but still reasonable
                        condition = config["condition"]
                        return True, f"Reasonable repeat for {condition} (slightly outside expected frequency)"

        # Check patient history exceptions
        if patient_id in self.patient_history_exceptions:
            patient_config = self.patient_history_exceptions[patient_id]
            if "allowed_repeat_patterns" in patient_config:
                for pattern in patient_config["allowed_repeat_patterns"]:
                    if (pattern.get("diagnosis_prefix") and
                        diagnosis_code.startswith(pattern["diagnosis_prefix"]) and
                        time_gap_days <= pattern.get("max_gap_days", 30)):
                        return True, f"Patient history exception: {pattern.get('reason', 'Custom exception')}"

        return False, ""

    def add_provider_exception(
        self,
        provider_id: str,
        max_daily_claims: int,
        reason: str,
        valid_until: Optional[datetime] = None
    ):
        """
        Add exception for high-volume legitimate providers.

        Args:
            provider_id: Provider identifier
            max_daily_claims: Maximum allowed daily claims
            reason: Reason for exception
            valid_until: Optional expiration date
        """
        self.provider_volume_exceptions[provider_id] = {
            "max_daily_claims": int(max_daily_claims),
            "reason": reason,
            "added_at": datetime.now(),
            "valid_until": valid_until
        }
        logger.info(f"Added provider exception for {provider_id}: {reason}")
        self._save_whitelist_to_storage()

    def check_provider_capacity_exception(
        self,
        provider_id: str,
        daily_claim_count: int,
        service_type: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Check if provider has volume exception.

        Args:
            provider_id: Provider identifier
            daily_claim_count: Number of claims on the day
            service_type: Optional service type for specific exceptions

        Returns:
            (is_allowed, reason)
        """
        if provider_id not in self.provider_volume_exceptions:
            return False, ""

        exception = self.provider_volume_exceptions[provider_id]

        # Check if exception is still valid
        if exception["valid_until"] and datetime.now() > exception["valid_until"]:
            logger.info(f"Provider exception for {provider_id} has expired")
            del self.provider_volume_exceptions[provider_id]
            self._save_whitelist_to_storage()
            return False, ""

        # Check specific service type exceptions
        if service_type and "service_type_exceptions" in exception:
            service_exceptions = exception["service_type_exceptions"]
            if service_type in service_exceptions:
                max_allowed = service_exceptions[service_type]
                if daily_claim_count <= max_allowed:
                    return True, f"Provider has service-specific exception for {service_type}: {exception['reason']}"

        # Check general exception
        if daily_claim_count <= exception["max_daily_claims"]:
            return True, f"Provider has volume exception: {exception['reason']}"

        return False, ""

    def add_service_combination_whitelist(
        self,
        diagnosis_prefix: str,
        service_code: str,
        max_repeat_days: int = 30,
        reason: str = ""
    ):
        """
        Add whitelist for specific diagnosis-service combinations.

        Args:
            diagnosis_prefix: ICD-10 code prefix
            service_code: Service code
            max_repeat_days: Maximum days between repeats
            reason: Reason for whitelist
        """
        key = (diagnosis_prefix, service_code)
        self.service_combination_whitelist[key] = {
            "max_repeat_days": max_repeat_days,
            "reason": reason,
            "added_at": datetime.now()
        }
        logger.info(f"Added service combination whitelist: {diagnosis_prefix} + {service_code}")
        self._save_whitelist_to_storage()

    def check_service_combination_whitelist(
        self,
        diagnosis_code: str,
        service_code: str,
        time_gap_days: int
    ) -> Tuple[bool, str]:
        """
        Check if service combination is whitelisted.

        Args:
            diagnosis_code: ICD-10 diagnosis code
            service_code: Service code
            time_gap_days: Days between claims

        Returns:
            (is_allowed, reason)
        """
        for (prefix, svc_code), config in self.service_combination_whitelist.items():
            if diagnosis_code.startswith(prefix) and service_code == svc_code:
                if time_gap_days <= config["max_repeat_days"]:
                    return True, f"Whitelisted combination: {config['reason']}"
                else:
                    return False, f"Whitelisted but exceeds max repeat days ({config['max_repeat_days']})"

        return False, ""

    def add_patient_history_exception(
        self,
        patient_id: str,
        diagnosis_prefix: str,
        max_gap_days: int,
        reason: str = ""
    ):
        """
        Add exception for specific patient history.

        Args:
            patient_id: Patient identifier
            diagnosis_prefix: ICD-10 code prefix
            max_gap_days: Maximum days between claims
            reason: Reason for exception
        """
        if patient_id not in self.patient_history_exceptions:
            self.patient_history_exceptions[patient_id] = {
                "allowed_repeat_patterns": [],
                "added_at": datetime.now()
            }

        self.patient_history_exceptions[patient_id]["allowed_repeat_patterns"].append({
            "diagnosis_prefix": diagnosis_prefix,
            "max_gap_days": max_gap_days,
            "reason": reason
        })
        logger.info(f"Added patient history exception for {patient_id}")
        self._save_whitelist_to_storage()

    def get_whitelist_statistics(self) -> Dict[str, int]:
        """
        Get statistics about whitelist configuration.

        Returns:
            Dictionary with statistics
        """
        return {
            "chronic_conditions": len(self.chronic_condition_whitelist),
            "provider_exceptions": len(self.provider_volume_exceptions),
            "service_combinations": len(self.service_combination_whitelist),
            "patient_exceptions": len(self.patient_history_exceptions)
        }

    def _load_whitelist_from_storage(self):
        """Load whitelist from storage file if available."""
        import json
        import os

        storage_path = "cache/whitelist_config.json"
        if os.path.exists(storage_path):
            try:
                with open(storage_path, 'r') as f:
                    data = json.load(f)
                    self.provider_volume_exceptions = data.get("provider_exceptions", {})
                    self.service_combination_whitelist = {
                        tuple(k): v for k, v in data.get("service_combinations", {}).items()
                    }
                    self.patient_history_exceptions = data.get("patient_exceptions", {})
                logger.info(f"Loaded whitelist from {storage_path}")
            except Exception as e:
                logger.warning(f"Failed to load whitelist from storage: {e}")

    def _save_whitelist_to_storage(self):
        """Save whitelist to storage file."""
        import json
        import os

        storage_path = "cache/whitelist_config.json"
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        try:
            data = {
                "provider_exceptions": self.provider_volume_exceptions,
                "service_combinations": {
                    list(k): v for k, v in self.service_combination_whitelist.items()
                },
                "patient_exceptions": self.patient_history_exceptions
            }
            with open(storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved whitelist to {storage_path}")
        except Exception as e:
            logger.warning(f"Failed to save whitelist to storage: {e}")
