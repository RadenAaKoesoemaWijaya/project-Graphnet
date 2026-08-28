"""
PII (Personally Identifiable Information) Masking & Anonymization

This module masks sensitive fields in audit logs to comply with:
- HIPAA (Health Insurance Portability and Accountability Act)
- GDPR (General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- PIPEDA (Personal Information Protection and Electronic Documents Act)

Security Category: High Priority (Phase 1)
"""

import re
import os
import hashlib
from typing import Any, Dict, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PII PATTERNS
# ============================================================================

PII_PATTERNS = {
    # Medical/Healthcare Specific
    'patient_id': r'(?i)(patient[_\s]?id|pid)',
    'medical_record': r'(?i)(mrn|medical[_\s]?record|chart[_\s]?number)',
    'diagnosis': r'(?i)(diagnosis|diagnostic|icd)',
    'treatment': r'(?i)(treatment|procedure|therapy)',
    'medication': r'(?i)(medication|drug|medicine|prescription)',
    'health_condition': r'(?i)(condition|disease|illness|symptom)',
    
    # Personal Information
    'ssn': r'\d{3}-\d{2}-\d{4}',  # Social Security Number
    'phone': r'[\+]?[(]?\d{3}[)]?[-\s.]?\d{3}[-\s.]?\d{4}',  # Phone numbers
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',  # CC numbers
    'date_of_birth': r'(?i)(dob|date[_\s]?of[_\s]?birth|birthdate)',
    
    # Provider Information
    'provider_id': r'(?i)(provider[_\s]?id|prv[_\s]?)',
    'clinic_id': r'(?i)(clinic[_\s]?id|facility[_\s]?id)',
    'hospital_id': r'(?i)(hospital[_\s]?id|org[_\s]?id)',
    
    # Financial Information
    'amount': r'(?i)(amount|cost|price|fee|charge|premium|deductible)',
    'bank_account': r'(?i)(account[_\s]?number|account[_\s]?id)',
    
    # Location Information
    'address': r'(?i)(address|street|city|state|zip|postal)',
}

# Fields that should ALWAYS be masked in logs
SENSITIVE_FIELDS = {
    # Healthcare
    'patient_id',
    'medical_record_number',
    'mrn',
    'diagnosis',
    'treatment',
    'medication',
    'health_condition',
    'procedure',
    'icd_code',
    'cpt_code',
    
    # Personal
    'ssn',
    'social_security_number',
    'phone',
    'email',
    'credit_card',
    'credit_card_number',
    'date_of_birth',
    'dob',
    'name',
    'first_name',
    'last_name',
    'address',
    'zip_code',
    'postal_code',
    
    # Provider
    'provider_id',
    'clinic_id',
    'hospital_id',
    'npi',
    'national_provider_id',
    
    # Financial
    'amount',
    'cost',
    'price',
    'fee',
    'charge',
    'premium',
    'deductible',
    'claim_amount',
    'paid_amount',
    'bank_account',
    'account_number',
    
    # Identifiers
    'claim_id',
    'transaction_id',
    'reference_number',
}

# ============================================================================
# PII MASKING UTILITIES
# ============================================================================

class PIIMasker:
    """Mask sensitive fields to protect PII in audit logs"""
    
    @staticmethod
    def hash_pii(value: str, salt: str = None) -> str:
        """
        One-way hash of PII for anonymization.
        
        Useful for analytics without exposing original values.
        Same input always produces same hash (for grouping).
        """
        if salt is None:
            salt = os.getenv('PII_HASH_SALT', 'astina-default-salt')
        
        if not isinstance(value, str):
            value = str(value)
        
        combined = f"{value}{salt}"
        hash_result = hashlib.sha256(combined.encode()).hexdigest()[:8]
        return f"HASH_{hash_result}"
    
    @staticmethod
    def mask_value(value: Any, field_name: str = '') -> str:
        """
        Mask a single PII value.
        
        Strategies:
        - Hash: For anonymization
        - Partial: Show only last N characters
        - Category: Show only type (e.g., "PHONE_NUMBER")
        """
        if value is None:
            return "NULL"
        
        value_str = str(value).strip()
        if not value_str:
            return "EMPTY"
        
        # Detect field type and apply appropriate masking
        if any(pattern in field_name.lower() for pattern in ['ssn', 'social_security']):
            return f"SSN_MASKED"
        elif any(pattern in field_name.lower() for pattern in ['credit_card', 'cc_']):
            return f"CARD_MASKED"
        elif any(pattern in field_name.lower() for pattern in ['diagnosis', 'treatment', 'medication']):
            return f"MEDICAL_INFO_MASKED"
        elif any(pattern in field_name.lower() for pattern in ['amount', 'cost', 'price', 'fee', 'charge']):
            # For amounts, show category only
            try:
                float_val = float(value_str)
                return f"AMOUNT_{int(float_val/1000)}K"  # Group by thousands
            except:
                return "AMOUNT_MASKED"
        elif any(pattern in field_name.lower() for pattern in ['id', 'number']):
            # For IDs, show last 3 chars
            if len(value_str) > 3:
                return f"***{value_str[-3:]}"
            else:
                return "***"
        else:
            # Default: mask with partial reveal
            if len(value_str) > 3:
                return f"***{value_str[-3:]}"
            else:
                return "***"
    
    @staticmethod
    def is_sensitive_field(field_name: str) -> bool:
        """
        Check if a field name represents sensitive PII.
        Excludes statistical metrics, model features, and boolean flags.
        """
        if not field_name or not isinstance(field_name, str):
            return False
            
        field_lower = field_name.lower().strip()
        
        # Rule flags, scores, probabilities, and ML/statistical indicators are NEVER PII
        non_pii_suffixes = (
            '_flag', '_score', '_probability', '_prob', '_ratio', 
            '_diff', '_count', '_idx', '_index', '_cluster', '_cluster_id', 
            '_status', '_type', '_rate', '_std', '_mean', '_sum', 
            '_min', '_max', '_median', '_norm', '_normed', '_scaled', 
            '_encoded', '_weight'
        )
        if any(field_lower.endswith(suffix) for suffix in non_pii_suffixes):
            return False
            
        non_pii_prefixes = ('is_', 'has_', 'anomaly_', 'risk_', 'rule_', 'stat_', 'metric_', 'flag_')
        if any(field_lower.startswith(prefix) for prefix in non_pii_prefixes):
            return False

        # Check against sensitive fields
        return any(
            sensitive.lower() in field_lower
            for sensitive in SENSITIVE_FIELDS
        )

    @staticmethod
    def mask_dict(data: Dict[str, Any], strategy: str = 'mask') -> Dict[str, Any]:
        """
        Mask all sensitive fields in a dictionary.
        
        Args:
            data: Dictionary to mask
            strategy: 'mask' (partial), 'hash' (full anonymization), 'remove' (delete field)
        
        Returns:
            Dictionary with masked values
        """
        if not isinstance(data, dict):
            return data
        
        masked = {}
        
        for field, value in data.items():
            # Check if field is sensitive
            if not PIIMasker.is_sensitive_field(field):
                masked[field] = value
                continue
            
            # Apply masking strategy
            if strategy == 'remove':
                # Skip this field entirely
                continue
            elif strategy == 'hash':
                # One-way hash for anonymization
                masked[field] = PIIMasker.hash_pii(value)
            else:  # 'mask' (default)
                # Partial mask
                masked[field] = PIIMasker.mask_value(value, field)
        
        return masked
    
    @staticmethod
    def mask_dataframe_columns(df, strategy: str = 'mask'):
        """
        Mask sensitive columns in a DataFrame.
        
        Useful for preventing PII leakage when logging sample data.
        """
        try:
            df = df.copy()
            
            for col in df.columns:
                # Check if column name indicates sensitive data
                if PIIMasker.is_sensitive_field(col):
                    if strategy == 'remove':
                        df = df.drop(columns=[col])
                    elif strategy == 'hash':
                        df[col] = df[col].apply(lambda x: PIIMasker.hash_pii(x) if x else None)
                    else:  # 'mask'
                        df[col] = df[col].apply(lambda x: PIIMasker.mask_value(x, col) if x else None)
            
            return df
        except Exception as e:
            logger.warning(f"Failed to mask dataframe columns: {e}")
            return df

# ============================================================================
# ERROR MESSAGE SANITIZATION
# ============================================================================

class ErrorSanitizer:
    """Remove PII from error messages before showing to users"""
    
    # Regex patterns for common PII in error messages
    PII_REGEX_PATTERNS = {
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone': r'[\+]?[(]?\d{3}[)]?[-\s.]?\d{3}[-\s.]?\d{4}',
        'numbers': r'\b\d{4,}\b',  # 4+ digit numbers
    }
    
    @staticmethod
    def sanitize_message(message: str) -> str:
        """Remove PII from error/warning messages"""
        
        if not isinstance(message, str):
            return str(message)
        
        sanitized = message
        
        # Replace SSN
        sanitized = re.sub(
            ErrorSanitizer.PII_REGEX_PATTERNS['ssn'],
            'SSN_REDACTED',
            sanitized
        )
        
        # Replace credit cards
        sanitized = re.sub(
            ErrorSanitizer.PII_REGEX_PATTERNS['credit_card'],
            'CARD_REDACTED',
            sanitized
        )
        
        # Replace emails
        sanitized = re.sub(
            ErrorSanitizer.PII_REGEX_PATTERNS['email'],
            '[EMAIL_REDACTED]',
            sanitized
        )
        
        # Replace phone numbers
        sanitized = re.sub(
            ErrorSanitizer.PII_REGEX_PATTERNS['phone'],
            '[PHONE_REDACTED]',
            sanitized
        )
        
        return sanitized

# ============================================================================
# LOG RETENTION & SECURE DELETION
# ============================================================================

class SecureDelete:
    """Securely delete files with multi-pass overwrite"""
    
    @staticmethod
    def secure_delete_file(file_path: str, passes: int = 3) -> bool:
        """
        Securely delete file by overwriting with random data.
        
        Args:
            file_path: Path to file to delete
            passes: Number of overwrite passes (DoD standard: 3)
        
        Returns:
            True if successful
        """
        try:
            import os
            
            if not os.path.exists(file_path):
                return True
            
            file_size = os.path.getsize(file_path)
            
            # Multi-pass overwrite
            with open(file_path, 'ba+', buffering=0) as f:
                for pass_num in range(passes):
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Finally, delete the file
            os.remove(file_path)
            logger.info(f"Securely deleted: {file_path}")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to securely delete {file_path}: {e}")
            # Fallback to regular delete
            try:
                os.remove(file_path)
                return True
            except:
                return False

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Usage Examples:

1. Mask sensitive fields in audit logs:
    
    from audit_trail import log_event
    
    # Before masking
    details = {
        'patient_id': '12345678',
        'diagnosis': 'diabetic neuropathy',
        'amount': 5000,
        'provider_id': 'PRV-001'
    }
    
    # Mask the details
    from pii_masker import PIIMasker
    masked_details = PIIMasker.mask_dict(details)
    
    # Log with masked details
    log_event('inference', 'anomaly_detected', masked_details)

2. Sanitize error messages:
    
    try:
        # some operation
        pass
    except Exception as e:
        from pii_masker import ErrorSanitizer
        safe_message = ErrorSanitizer.sanitize_message(str(e))
        st.error(f"Error: {safe_message}")

3. Mask DataFrame before logging sample:
    
    from pii_masker import PIIMasker
    df_masked = PIIMasker.mask_dataframe_columns(df_sample)
    logger.info(f"Sample data shape: {df_masked.shape}")

4. Secure file deletion:
    
    from pii_masker import SecureDelete
    SecureDelete.secure_delete_file('/tmp/sensitive_data.parquet')
"""
