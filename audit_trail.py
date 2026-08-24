"""
Audit trail system for ASTINA to track user activities and system events.
Provides comprehensive logging for compliance and debugging.
Supports both local JSON logs and Google Cloud Logging integration.
Includes PII masking to protect sensitive health data.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

try:
    from google.cloud import logging as cloud_logging
    CLOUD_LOGGING_AVAILABLE = True
except ImportError:
    CLOUD_LOGGING_AVAILABLE = False

# Import PII masking utilities
try:
    from pii_masker import PIIMasker, ErrorSanitizer
    PII_MASKING_AVAILABLE = True
except ImportError:
    PII_MASKING_AVAILABLE = False
    PIIMasker = None
    ErrorSanitizer = None

logger = logging.getLogger("graphnet.audit_trail")

class AuditTrail:
    """Comprehensive audit trail system for tracking user and system activities"""
    
    def __init__(self, log_dir: str = "logs/audit", use_cloud_logging: bool = True):
        """
        Initialize audit trail system.
        
        Args:
            log_dir: Directory to store audit logs (local fallback)
            use_cloud_logging: Enable Google Cloud Logging integration (auto-enabled on Cloud Run)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily log file
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"audit_{today}.log"
        
        # In-memory buffer for recent events
        self.recent_events: List[Dict[str, Any]] = []
        self.max_buffer_size = 100
        
        # Cloud Logging setup
        self.cloud_logger = None
        if use_cloud_logging and CLOUD_LOGGING_AVAILABLE:
            try:
                cloud_log_client = cloud_logging.Client()
                self.cloud_logger = cloud_log_client.logger("astina-audit-trail")
                logger.info("✅ Cloud Logging enabled for audit trail")
            except Exception as e:
                logger.warning(f"Cloud Logging unavailable: {e}. Using local logs only.")
                self.cloud_logger = None
        
        # Auto-detect Cloud Run environment
        self.is_cloud_run = os.getenv('K_SERVICE') is not None
        if self.is_cloud_run and not self.cloud_logger:
            logger.warning("⚠️ Running on Cloud Run but Cloud Logging not configured. "
                         "Audit logs will be local only.")
    
    def _get_client_info(self) -> Dict[str, str]:
        """Get client information for audit records"""
        client_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_agent': 'unknown',
            'session_id': 'unknown'
        }
        
        if STREAMLIT_AVAILABLE and st:
            if hasattr(st.context, 'headers'):
                client_info['user_agent'] = st.context.headers.get('user-agent', 'unknown')
            if 'session_id' in st.session_state:
                client_info['session_id'] = st.session_state.get('session_id', 'unknown')
        
        return client_info
    
    def log_event(
        self,
        event_type: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        severity: str = "INFO"
    ) -> None:
        """
        Log an audit event to both local storage and Cloud Logging.
        
        Args:
            event_type: Type of event (e.g., 'data_upload', 'model_training', 'anomaly_detection')
            action: Action performed (e.g., 'upload', 'train', 'detect')
            resource: Resource affected (e.g., file name, model name)
            details: Additional event details
            user_id: User identifier
            severity: Event severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Mask sensitive details if PII masking is available
        if details and PII_MASKING_AVAILABLE and PIIMasker:
            details = PIIMasker.mask_dict(details, strategy='mask')
        
        event = {
            **self._get_client_info(),
            'event_type': event_type,
            'action': action,
            'resource': resource,
            'details': details or {},
            'user_id': user_id or 'anonymous',
            'severity': severity,
            'environment': 'cloud_run' if self.is_cloud_run else 'local'
        }
        
        # Add to in-memory buffer
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_buffer_size:
            self.recent_events.pop(0)
        
        # Write to local log file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
        
        # Send to Cloud Logging if available
        if self.cloud_logger:
            try:
                self.cloud_logger.log_struct(
                    event,
                    severity=severity,
                    labels={
                        'event_type': event_type,
                        'action': action,
                        'resource': resource[:100]  # Truncate resource name for label
                    }
                )
            except Exception as e:
                logger.debug(f"Cloud Logging write failed (non-fatal): {e}")
        
        # Also log to system logger
        log_method = {
            'DEBUG': logger.debug,
            'INFO': logger.info,
            'WARNING': logger.warning,
            'ERROR': logger.error,
            'CRITICAL': logger.critical
        }.get(severity, logger.info)
        
        log_method(f"Audit: {event_type} - {action} on {resource} by {user_id}")
    
    def log_data_upload(
        self,
        file_name: str,
        file_size: int,
        row_count: int,
        column_count: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log data upload event"""
        details = {
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'row_count': row_count,
            'column_count': column_count,
            'success': success,
            'error_message': error_message
        }
        
        self.log_event(
            event_type='data_upload',
            action='upload',
            resource=file_name,
            details=details,
            severity='INFO' if success else 'ERROR'
        )
    
    def log_preprocessing(
        self,
        original_rows: int,
        original_cols: int,
        processed_rows: int,
        processed_cols: int,
        processing_time: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log data preprocessing event"""
        details = {
            'original_rows': original_rows,
            'original_cols': original_cols,
            'processed_rows': processed_rows,
            'processed_cols': processed_cols,
            'processing_time_seconds': round(processing_time, 2),
            'success': success,
            'error_message': error_message
        }
        
        self.log_event(
            event_type='data_preprocessing',
            action='preprocess',
            resource='dataset',
            details=details,
            severity='INFO' if success else 'ERROR'
        )
    
    def log_model_training(
        self,
        model_type: str,
        algorithm: str,
        training_samples: int,
        feature_count: int,
        training_time: float,
        success: bool = True,
        metrics: Optional[Dict[str, float]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Log model training event"""
        details = {
            'model_type': model_type,
            'algorithm': algorithm,
            'training_samples': training_samples,
            'feature_count': feature_count,
            'training_time_seconds': round(training_time, 2),
            'success': success,
            'metrics': metrics or {},
            'error_message': error_message
        }
        
        self.log_event(
            event_type='model_training',
            action='train',
            resource=model_type,
            details=details,
            severity='INFO' if success else 'ERROR'
        )
    
    def log_anomaly_detection(
        self,
        detection_samples: int,
        anomaly_count: int,
        anomaly_rate: float,
        detection_time: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log anomaly detection event"""
        details = {
            'detection_samples': detection_samples,
            'anomaly_count': anomaly_count,
            'anomaly_rate': round(anomaly_rate, 4),
            'detection_time_seconds': round(detection_time, 2),
            'success': success,
            'error_message': error_message
        }
        
        self.log_event(
            event_type='anomaly_detection',
            action='detect',
            resource='dataset',
            details=details,
            severity='INFO' if success else 'ERROR'
        )
    
    def log_model_export(
        self,
        model_name: str,
        model_size: int,
        export_time: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log model export event"""
        details = {
            'model_name': model_name,
            'model_size_bytes': model_size,
            'model_size_mb': round(model_size / (1024 * 1024), 2),
            'export_time_seconds': round(export_time, 2),
            'success': success,
            'error_message': error_message
        }
        
        self.log_event(
            event_type='model_export',
            action='export',
            resource=model_name,
            details=details,
            severity='INFO' if success else 'ERROR'
        )
    
    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = "WARNING"
    ) -> None:
        """Log security-related event"""
        details = {
            'description': description,
            'security_event': True
        }
        
        self.log_event(
            event_type='security',
            action=event_type,
            resource='system',
            details=details,
            severity=severity
        )
    
    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent audit events from buffer.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        return self.recent_events[-limit:]
    
    def get_events_by_type(self, event_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get events filtered by type from log file.
        
        Args:
            event_type: Type of events to filter
            limit: Maximum number of events to return
            
        Returns:
            List of filtered events
        """
        events = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('event_type') == event_type:
                            events.append(event)
                            if len(events) >= limit:
                                break
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        
        return events
    
    def get_events_by_date(self, date: str) -> List[Dict[str, Any]]:
        """
        Get events for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of events for the date
        """
        log_file = self.log_dir / f"audit_{date}.log"
        events = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        
        return events
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """
        Clean up audit logs older than specified days.
        
        Args:
            days_to_keep: Number of days of logs to keep
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in self.log_dir.glob("audit_*.log"):
            try:
                file_date = datetime.strptime(log_file.stem.split("_")[1], "%Y-%m-%d")
                if file_date < cutoff_date:
                    log_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old audit log: {log_file}")
            except (ValueError, IndexError):
                continue
        
        return deleted_count

# Global audit trail instance
_audit_trail: Optional[AuditTrail] = None

def get_audit_trail() -> AuditTrail:
    """Get or create global audit trail instance"""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = AuditTrail()
    return _audit_trail

def log_data_upload(file_name: str, file_size: int, row_count: int, column_count: int, 
                    success: bool = True, error_message: Optional[str] = None) -> None:
    """Convenience function to log data upload"""
    get_audit_trail().log_data_upload(file_name, file_size, row_count, column_count, 
                                       success, error_message)

def log_preprocessing(original_rows: int, original_cols: int, processed_rows: int, 
                     processed_cols: int, processing_time: float, success: bool = True,
                     error_message: Optional[str] = None) -> None:
    """Convenience function to log preprocessing"""
    get_audit_trail().log_preprocessing(original_rows, original_cols, processed_rows,
                                        processed_cols, processing_time, success, error_message)

def log_model_training(model_type: str, algorithm: str, training_samples: int, 
                      feature_count: int, training_time: float, success: bool = True,
                      metrics: Optional[Dict[str, float]] = None, 
                      error_message: Optional[str] = None) -> None:
    """Convenience function to log model training"""
    get_audit_trail().log_model_training(model_type, algorithm, training_samples,
                                         feature_count, training_time, success, metrics, error_message)

def log_anomaly_detection(detection_samples: int, anomaly_count: int, anomaly_rate: float,
                         detection_time: float, success: bool = True,
                         error_message: Optional[str] = None) -> None:
    """Convenience function to log anomaly detection"""
    get_audit_trail().log_anomaly_detection(detection_samples, anomaly_count, anomaly_rate,
                                            detection_time, success, error_message)

from datetime import timedelta
