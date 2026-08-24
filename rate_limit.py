"""
Rate Limiting & Quota Management for ASTINA

This module provides rate limiting functionality to prevent:
- DDoS attacks (volume exhaustion)
- Resource exhaustion (CPU/memory from large jobs)
- Abuse of expensive operations
- Service unavailability

Security Category: Medium Priority (Phase 1 Quick Win)
"""

import os
import time
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# ============================================================================
# RATE LIMIT CONFIGURATION
# ============================================================================

# Get limits from environment variables with sensible defaults
RATE_LIMITS = {
    'analyst': {
        'uploads_per_day': int(os.getenv('RATE_LIMIT_UPLOADS_PER_DAY', '10')),
        'training_jobs_per_day': int(os.getenv('RATE_LIMIT_TRAINING_PER_DAY', '5')),
        'inference_per_minute': int(os.getenv('RATE_LIMIT_INFERENCE_PER_MINUTE', '100')),
        'concurrent_uploads': 2,
        'max_file_size_gb': 2,
    },
    'admin': {
        'uploads_per_day': 100,
        'training_jobs_per_day': 50,
        'inference_per_minute': 1000,
        'concurrent_uploads': 10,
        'max_file_size_gb': 10,
    },
    'viewer': {
        'uploads_per_day': 0,  # No upload permission
        'training_jobs_per_day': 0,
        'inference_per_minute': 50,
        'concurrent_uploads': 0,
        'max_file_size_gb': 0,
    }
}

# Default quota for unauthenticated users (very restrictive)
UNAUTHENTICATED_LIMITS = {
    'uploads_per_day': 1,
    'training_jobs_per_day': 0,
    'inference_per_minute': 10,
    'concurrent_uploads': 1,
    'max_file_size_gb': 0.5,
}

# ============================================================================
# IN-MEMORY RATE LIMITING (for single-instance deployment)
# ============================================================================

class RateLimiter:
    """In-memory rate limiter for single-instance deployments"""
    
    def __init__(self):
        # Track requests per user per endpoint
        # Format: {user_id: {endpoint: deque([timestamp1, timestamp2, ...])}}
        self.request_history = defaultdict(lambda: defaultdict(deque))
        
        # Track daily quota usage
        # Format: {date: {user_id: {action: count}}}
        self.daily_usage = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Track concurrent operations
        # Format: {user_id: {action: count}}
        self.concurrent_ops = defaultdict(lambda: defaultdict(int))
        
        # Cleanup interval (hours)
        self.cleanup_interval = 24
    
    def is_rate_limited(
        self,
        user_id: str,
        action: str,
        limit: int,
        window_seconds: int = 60
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if user exceeded rate limit for action.
        
        Args:
            user_id: Unique user identifier
            action: Action name (e.g., 'upload', 'inference')
            limit: Maximum requests allowed in window
            window_seconds: Time window for limit (default: 60 seconds)
        
        Returns:
            (is_limited, error_message)
        """
        now = time.time()
        key = f"{action}"
        
        # Get request history for this user+action
        history = self.request_history[user_id][key]
        
        # Remove old requests outside the window
        while history and (now - history[0]) > window_seconds:
            history.popleft()
        
        # Check if limit exceeded
        if len(history) >= limit:
            oldest_request = history[0]
            reset_time = oldest_request + window_seconds
            reset_seconds = int(reset_time - now)
            
            return True, f"Rate limit exceeded ({len(history)}/{limit}). Reset in {reset_seconds}s"
        
        # Record this request
        history.append(now)
        return False, None
    
    def check_daily_quota(
        self,
        user_id: str,
        action: str,
        limit: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if user exceeded daily quota.
        
        Args:
            user_id: Unique user identifier
            action: Action name (e.g., 'uploads', 'training_jobs')
            limit: Maximum per day
        
        Returns:
            (is_limited, error_message)
        """
        today = datetime.utcnow().date().isoformat()
        current_count = self.daily_usage[today][user_id][action]
        
        if current_count >= limit:
            return True, f"Daily quota exceeded for {action}. Limit: {limit}/day"
        
        return False, None
    
    def increment_daily_quota(self, user_id: str, action: str):
        """Increment daily quota counter"""
        today = datetime.utcnow().date().isoformat()
        self.daily_usage[today][user_id][action] += 1
    
    def check_concurrent_limit(
        self,
        user_id: str,
        action: str,
        limit: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if user exceeded concurrent operation limit.
        
        Args:
            user_id: Unique user identifier
            action: Action name (e.g., 'upload')
            limit: Maximum concurrent operations
        
        Returns:
            (is_limited, error_message)
        """
        current = self.concurrent_ops[user_id][action]
        
        if current >= limit:
            return True, f"Too many concurrent {action} operations ({current}/{limit})"
        
        return False, None
    
    def increment_concurrent(self, user_id: str, action: str):
        """Increment concurrent operation counter"""
        self.concurrent_ops[user_id][action] += 1
    
    def decrement_concurrent(self, user_id: str, action: str):
        """Decrement concurrent operation counter"""
        self.concurrent_ops[user_id][action] = max(0, self.concurrent_ops[user_id][action] - 1)
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get rate limit statistics for user"""
        today = datetime.utcnow().date().isoformat()
        
        return {
            'user_id': user_id,
            'daily_usage': dict(self.daily_usage[today].get(user_id, {})),
            'concurrent_ops': dict(self.concurrent_ops.get(user_id, {})),
            'rate_limit_requests': {
                action: list(history)[-10:]  # Last 10 requests
                for action, history in self.request_history[user_id].items()
            }
        }

# Global instance
rate_limiter = RateLimiter()

# ============================================================================
# QUOTA CHECKER
# ============================================================================

def check_user_quota(
    user_id: str,
    action: str,
    user_role: str = 'analyst'
) -> Tuple[bool, Optional[str]]:
    """
    Check if user has permission and quota to perform action.
    
    Args:
        user_id: User identifier
        action: Action to perform (upload, train, inference)
        user_role: User role for quota lookup
    
    Returns:
        (allowed, error_message)
    """
    # Get user's role limits
    limits = RATE_LIMITS.get(user_role, UNAUTHENTICATED_LIMITS)
    
    # Map actions to quota keys
    quota_map = {
        'upload': ('uploads_per_day', 60),           # 60 second window
        'training': ('training_jobs_per_day', 3600),  # 1 hour window
        'inference': ('inference_per_minute', 60),    # 1 minute window
    }
    
    if action not in quota_map:
        return True, None  # Unknown action, allow
    
    quota_key, window = quota_map[action]
    daily_limit = limits.get(quota_key, 0)
    
    # Check daily quota
    if daily_limit > 0:
        is_limited, error = rate_limiter.check_daily_quota(user_id, quota_key, daily_limit)
        if is_limited:
            return False, error
    elif daily_limit == 0:
        return False, f"User role '{user_role}' does not have permission for '{action}'"
    
    return True, None

def increment_quota(user_id: str, action: str):
    """Increment quota counter after successful action"""
    quota_map = {
        'upload': 'uploads_per_day',
        'training': 'training_jobs_per_day',
        'inference': 'inference_per_minute',
    }
    
    if action in quota_map:
        rate_limiter.increment_daily_quota(user_id, quota_map[action])

# ============================================================================
# STREAMLIT INTEGRATION
# ============================================================================

def check_upload_quota(user_id: str = None) -> Tuple[bool, Optional[str]]:
    """Check upload quota for current user"""
    if user_id is None:
        user_id = "anonymous"
    
    # For now, use analyst as default role (will be updated with auth in Phase 2)
    allowed, error = check_user_quota(user_id, 'upload', 'analyst')
    if not allowed:
        return False, error
    
    return True, None

def check_training_quota(user_id: str = None) -> Tuple[bool, Optional[str]]:
    """Check training quota for current user"""
    if user_id is None:
        user_id = "anonymous"
    
    allowed, error = check_user_quota(user_id, 'training', 'analyst')
    if not allowed:
        return False, error
    
    return True, None

def check_inference_quota(user_id: str = None) -> Tuple[bool, Optional[str]]:
    """Check inference quota for current user"""
    if user_id is None:
        user_id = "anonymous"
    
    allowed, error = check_user_quota(user_id, 'inference', 'analyst')
    if not allowed:
        return False, error
    
    return True, None

# ============================================================================
# USAGE EXAMPLES FOR STREAMLIT
# ============================================================================

"""
Usage in Streamlit pages:

from retry_utils import check_upload_quota, check_training_quota, increment_quota

# In data_collection.py - File upload
if uploaded_file is not None:
    user_id = st.session_state.get('username', 'anonymous')
    allowed, error = check_upload_quota(user_id)
    
    if not allowed:
        st.error(f"❌ {error}")
        st.stop()
    
    # Process upload...
    increment_quota(user_id, 'upload')
    st.success("✅ Upload successful")

# In training.py - Model training
if st.button("🚀 Mulai Training Model"):
    user_id = st.session_state.get('username', 'anonymous')
    allowed, error = check_training_quota(user_id)
    
    if not allowed:
        st.error(f"❌ {error}")
        st.stop()
    
    # Start training...
    increment_quota(user_id, 'training')
"""
