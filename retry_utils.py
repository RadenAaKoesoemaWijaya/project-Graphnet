"""
Retry mechanism for handling transient failures in ASTINA.
Provides exponential backoff and circuit breaker patterns.
"""
import time
import logging
from functools import wraps
from typing import Callable, Optional, Any, Type
from datetime import datetime, timedelta

logger = logging.getLogger("graphnet.retry_utils")

class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is open, requests fail immediately
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def _should_attempt_request(self) -> bool:
        """Determine if request should be attempted based on circuit state"""
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            # Check if recovery timeout has passed
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
                return True
            return False
        elif self.state == 'HALF_OPEN':
            return True
        return False
    
    def _record_success(self):
        """Record successful request"""
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            logger.info("Circuit breaker transitioning to CLOSED state")
    
    def _record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will remain open for {self.recovery_timeout} seconds"
            )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if not self._should_attempt_request():
            raise Exception(
                f"Circuit breaker is OPEN. Too many failures ({self.failure_count}). "
                f"Retry after {self.recovery_timeout} seconds"
            )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure()
            raise e

def retry_on_exception(
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for retrying function on exception with exponential backoff.
    
    Args:
        config: Retry configuration
        exceptions: Tuple of exceptions to retry on
        on_retry: Callback function called before each retry
        
    Example:
        @retry_on_exception(max_attempts=3, base_delay=1.0)
        def load_data(file_path):
            return pd.read_parquet(file_path)
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        # Last attempt failed, raise exception
                        logger.error(
                            f"Function {func.__name__} failed after {config.max_attempts} attempts. "
                            f"Final error: {str(e)}"
                        )
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter if enabled
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    # Call on_retry callback if provided
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception if last_exception else Exception("Retry failed")
        
        return wrapper
    return decorator

def safe_file_operation(operation: str):
    """
    Decorator for safe file operations with retry logic.
    
    Args:
        operation: Description of the file operation
        
    Example:
        @safe_file_operation("reading parquet file")
        def read_parquet(path):
            return pd.read_parquet(path)
    """
    config = RetryConfig(max_attempts=3, base_delay=0.5, max_delay=10.0)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            def on_retry(attempt: int, error: Exception, delay: float):
                logger.info(f"Retrying {operation} (attempt {attempt}) after {delay:.2f}s")
            
            return retry_on_exception(
                config=config,
                exceptions=(IOError, OSError, FileNotFoundError, PermissionError),
                on_retry=on_retry
            )(func)(*args, **kwargs)
        
        return wrapper
    return decorator

def safe_network_operation(operation: str):
    """
    Decorator for safe network operations with retry logic.
    
    Args:
        operation: Description of the network operation
        
    Example:
        @safe_network_operation("downloading model")
        def download_model(url):
            return requests.get(url)
    """
    config = RetryConfig(max_attempts=5, base_delay=1.0, max_delay=30.0)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            def on_retry(attempt: int, error: Exception, delay: float):
                logger.info(f"Retrying {operation} (attempt {attempt}) after {delay:.2f}s")
            
            return retry_on_exception(
                config=config,
                exceptions=(ConnectionError, TimeoutError, OSError),
                on_retry=on_retry
            )(func)(*args, **kwargs)
        
        return wrapper
    return decorator

class RetryContext:
    """Context manager for retry operations"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.attempts = 0
        self.errors = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.errors.append(exc_val)
            logger.error(f"Error in retry context: {exc_val}")
        return False  # Don't suppress exceptions
    
    def attempt(self, func: Callable, *args, **kwargs) -> Any:
        """
        Attempt function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        while self.attempts < self.config.max_attempts:
            try:
                self.attempts += 1
                return func(*args, **kwargs)
            except Exception as e:
                self.errors.append(e)
                
                if self.attempts >= self.config.max_attempts:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
                    raise e
                
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** (self.attempts - 1)),
                    self.config.max_delay
                )
                
                logger.warning(
                    f"Attempt {self.attempts} failed: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
        
        raise Exception("Retry context failed")
