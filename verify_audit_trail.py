#!/usr/bin/env python3
"""Verify audit_trail Cloud Logging integration"""
from audit_trail import AuditTrail, CLOUD_LOGGING_AVAILABLE
import os

print('AUDIT TRAIL VERIFICATION')
print('=' * 60)
print(f'Cloud Logging Available: {CLOUD_LOGGING_AVAILABLE}')
print(f'Cloud Run Environment: {os.getenv("K_SERVICE") is not None}')

# Test audit trail creation
try:
    audit = AuditTrail()
    print(f'✅ AuditTrail initialized successfully')
    print(f'   Log directory: {audit.log_dir}')
    status = 'Enabled' if audit.cloud_logger else 'Disabled (fallback to local)'
    print(f'   Cloud logger: {status}')
    
    # Test logging an event
    audit.log_event(
        event_type='test',
        action='verify',
        resource='audit_trail.py',
        details={'test': True}
    )
    print(f'✅ Test event logged successfully')
    print(f'   Events in buffer: {len(audit.recent_events)}')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()

print('=' * 60)
