"""
Enhanced monitoring metrics for ASTINA with performance tracking and alerting.
Provides comprehensive metrics collection for system observability.
"""
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

logger = logging.getLogger("graphnet.enhanced_metrics")

class PerformanceMetrics:
    """Track performance metrics for various operations"""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance metrics tracker.
        
        Args:
            max_history: Maximum number of metric records to keep
        """
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def record_timing(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record timing metric for an operation.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            metadata: Additional metadata about the operation
        """
        with self.lock:
            self.metrics[f"{operation}_timing"].append({
                'timestamp': datetime.utcnow().isoformat(),
                'duration': duration,
                'metadata': metadata or {}
            })
            self.counters[f"{operation}_count"] += 1
    
    def record_counter(self, metric_name: str, increment: int = 1) -> None:
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the counter
            increment: Amount to increment by
        """
        with self.lock:
            self.counters[metric_name] += increment
    
    def set_gauge(self, metric_name: str, value: float) -> None:
        """
        Set a gauge metric value.
        
        Args:
            metric_name: Name of the gauge
            value: Current value
        """
        with self.lock:
            self.gauges[metric_name] = value
    
    def get_timing_stats(self, operation: str) -> Dict[str, float]:
        """
        Get statistics for timing metrics.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Dictionary with timing statistics
        """
        with self.lock:
            timings = self.metrics.get(f"{operation}_timing", deque())
            if not timings:
                return {}
            
            durations = [t['duration'] for t in timings]
            return {
                'count': len(durations),
                'min': min(durations),
                'max': max(durations),
                'mean': sum(durations) / len(durations),
                'median': sorted(durations)[len(durations) // 2],
                'p95': sorted(durations)[int(len(durations) * 0.95)],
                'p99': sorted(durations)[int(len(durations) * 0.99)]
            }
    
    def get_counter(self, metric_name: str) -> int:
        """Get current counter value"""
        with self.lock:
            return self.counters.get(metric_name, 0)
    
    def get_gauge(self, metric_name: str) -> Optional[float]:
        """Get current gauge value"""
        with self.lock:
            return self.gauges.get(metric_name)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'timing_operations': [k.replace('_timing', '') for k in self.metrics.keys() if k.endswith('_timing')]
            }
    
    def reset_counter(self, metric_name: str) -> None:
        """Reset a counter to zero"""
        with self.lock:
            self.counters[metric_name] = 0

class HealthChecker:
    """Monitor system health and resource usage"""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.last_results: Dict[str, Dict[str, Any]] = {}
    
    def register_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """
        Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """
        Run a specific health check.
        
        Args:
            name: Name of the check to run
            
        Returns:
            Dictionary with check result
        """
        if name not in self.checks:
            return {'name': name, 'status': 'unknown', 'error': 'Check not registered'}
        
        start_time = time.time()
        try:
            result = self.checks[name]()
            duration = time.time() - start_time
            
            check_result = {
                'name': name,
                'status': 'healthy' if result else 'unhealthy',
                'duration': duration,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.last_results[name] = check_result
            return check_result
            
        except Exception as e:
            duration = time.time() - start_time
            check_result = {
                'name': name,
                'status': 'error',
                'error': str(e),
                'duration': duration,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.last_results[name] = check_result
            return check_result
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary with all check results
        """
        results = {}
        overall_healthy = True
        
        for name in self.checks:
            result = self.run_check(name)
            results[name] = result
            if result['status'] != 'healthy':
                overall_healthy = False
        
        return {
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'checks': results,
            'timestamp': datetime.utcnow().isoformat()
        }

class AlertManager:
    """Manage alerts based on metric thresholds"""
    
    def __init__(self):
        self.thresholds: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
    
    def set_threshold(self, metric_name: str, operator: str, value: float, severity: str = "warning") -> None:
        """
        Set alert threshold for a metric.
        
        Args:
            metric_name: Name of the metric to monitor
            operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
            value: Threshold value
            severity: Alert severity ('info', 'warning', 'error', 'critical')
        """
        self.thresholds[metric_name] = {
            'operator': operator,
            'value': value,
            'severity': severity
        }
    
    def check_threshold(self, metric_name: str, current_value: float) -> Optional[Dict[str, Any]]:
        """
        Check if current value triggers alert threshold.
        
        Args:
            metric_name: Name of the metric
            current_value: Current value to check
            
        Returns:
            Alert dictionary if threshold triggered, None otherwise
        """
        if metric_name not in self.thresholds:
            return None
        
        threshold = self.thresholds[metric_name]
        operator = threshold['operator']
        threshold_value = threshold['value']
        
        triggered = False
        if operator == '>' and current_value > threshold_value:
            triggered = True
        elif operator == '<' and current_value < threshold_value:
            triggered = True
        elif operator == '>=' and current_value >= threshold_value:
            triggered = True
        elif operator == '<=' and current_value <= threshold_value:
            triggered = True
        elif operator == '==' and current_value == threshold_value:
            triggered = True
        elif operator == '!=' and current_value != threshold_value:
            triggered = True
        
        if triggered:
            alert = {
                'metric_name': metric_name,
                'current_value': current_value,
                'threshold_value': threshold_value,
                'operator': operator,
                'severity': threshold['severity'],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.alerts.append(alert)
            
            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
            return alert
        
        return None
    
    def register_alert_callback(self, callback: Callable) -> None:
        """
        Register a callback function to be called when alert triggers.
        
        Args:
            callback: Function to call with alert data
        """
        self.alert_callbacks.append(callback)
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alerts[-limit:]

class MetricsCollector:
    """Main metrics collection system"""
    
    def __init__(self):
        self.performance = PerformanceMetrics()
        self.health = HealthChecker()
        self.alerts = AlertManager()
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Set default alert thresholds
        self._set_default_thresholds()
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks"""
        import psutil
        import os
        
        def check_memory_usage() -> bool:
            """Check if memory usage is acceptable"""
            try:
                memory_percent = psutil.virtual_memory().percent
                return memory_percent < 90  # Alert if > 90% memory usage
            except:
                return True  # Assume healthy if check fails
        
        def check_disk_space() -> bool:
            """Check if disk space is acceptable"""
            try:
                disk_usage = psutil.disk_usage(os.getcwd())
                return disk_usage.percent < 90  # Alert if > 90% disk usage
            except:
                return True
        
        self.health.register_check('memory_usage', check_memory_usage)
        self.health.register_check('disk_space', check_disk_space)
    
    def _set_default_thresholds(self) -> None:
        """Set default alert thresholds"""
        # Alert if processing time exceeds 5 minutes
        self.alerts.set_threshold('preprocessing_timing', '>', 300, 'warning')
        
        # Alert if model training time exceeds 1 hour
        self.alerts.set_threshold('model_training_timing', '>', 3600, 'warning')
        
        # Alert if error rate exceeds 5%
        self.alerts.set_threshold('error_rate', '>', 0.05, 'error')
    
    def record_operation(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an operation with timing.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            metadata: Additional metadata
        """
        self.performance.record_timing(operation, duration, metadata)
        
        # Check for timing alerts
        timing_metric = f"{operation}_timing"
        stats = self.performance.get_timing_stats(operation)
        if stats:
            avg_duration = stats.get('mean', 0)
            self.alerts.check_threshold(timing_metric, avg_duration)
    
    def increment_counter(self, metric_name: str, increment: int = 1) -> None:
        """Increment a counter metric"""
        self.performance.record_counter(metric_name, increment)
    
    def set_gauge(self, metric_name: str, value: float) -> None:
        """Set a gauge metric"""
        self.performance.set_gauge(metric_name, value)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        health_status = self.health.run_all_checks()
        performance_metrics = self.performance.get_all_metrics()
        recent_alerts = self.alerts.get_recent_alerts(10)
        
        return {
            'health': health_status,
            'performance': performance_metrics,
            'recent_alerts': recent_alerts,
            'timestamp': datetime.utcnow().isoformat()
        }

# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

def record_operation(operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to record operation timing"""
    get_metrics_collector().record_operation(operation, duration, metadata)

def increment_counter(metric_name: str, increment: int = 1) -> None:
    """Convenience function to increment counter"""
    get_metrics_collector().increment_counter(metric_name, increment)

def set_gauge(metric_name: str, value: float) -> None:
    """Convenience function to set gauge"""
    get_metrics_collector().set_gauge(metric_name, value)

def get_system_status() -> Dict[str, Any]:
    """Convenience function to get system status"""
    return get_metrics_collector().get_system_status()
