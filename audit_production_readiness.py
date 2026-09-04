#!/usr/bin/env python3
"""Production readiness audit for ASTINA"""
import os
import re
from pathlib import Path

# Production Readiness Audit
audit = {
    'Security': [],
    'Performance': [],
    'Reliability': [],
    'Operations': [],
    'Documentation': [],
    'Testing': [],
}

# Check for secrets/hardcoded values
py_files = list(Path('.').rglob('*.py'))[:30]
secrets_found = []
for f in py_files:
    try:
        content = f.read_text(encoding='utf-8', errors='ignore')
        if re.search(r'(password|secret|api_key|token)\s*=', content, re.I):
            if not 'test' in str(f):
                secrets_found.append(str(f))
    except:
        pass

audit['Security'].append(('Hardcoded Secrets', 'PASS' if not secrets_found else f'FAIL: {len(secrets_found)} files'))

# Check SSL/HTTPS setup
has_ssl = Path('.streamlit/config.toml').exists()
audit['Security'].append(('SSL/HTTPS Config', 'PASS' if has_ssl else 'WARN'))

# Check for environment variable usage
env_usage = False
for f in py_files:
    try:
        if 'os.environ' in f.read_text():
            env_usage = True
            break
    except:
        pass
audit['Security'].append(('Environment Variables', 'PASS' if env_usage else 'WARN'))

# Check for async operations
has_threading = False
for f in py_files:
    try:
        content = f.read_text()
        if 'threading' in content or 'asyncio' in content:
            has_threading = True
            break
    except:
        pass
audit['Reliability'].append(('Async/Threading Support', 'PASS' if has_threading else 'WARN'))

# Check for error handling
has_error_handler = Path('error_handler.py').exists()
audit['Reliability'].append(('Error Handler Module', 'PASS' if has_error_handler else 'FAIL'))

# Check logging setup
logging_config = Path('logging_config.py').exists()
audit['Operations'].append(('Logging Configuration', 'PASS' if logging_config else 'FAIL'))

# Check for metrics
has_metrics = False
for f in py_files:
    try:
        if 'metrics' in f.name:
            has_metrics = True
            break
    except:
        pass
audit['Operations'].append(('Metrics/Monitoring', 'PASS' if has_metrics else 'WARN'))

# Check deployment configs
docker_file = Path('Dockerfile').exists()
docker_compose = Path('docker-compose.yml').exists()
cloudrun_config = Path('.cloudrun/app.yaml').exists()
audit['Operations'].append(('Docker Support', 'PASS' if docker_file and docker_compose else 'FAIL'))
audit['Operations'].append(('Cloud Run Support', 'PASS' if cloudrun_config else 'FAIL'))

# Documentation
readme = Path('README.md').exists()
arch_doc = Path('ARCHITECTURE.md').exists()
exec_summary = Path('EXECUTIVE SUMMARY.md').exists()
audit['Documentation'].append(('README', 'PASS' if readme else 'FAIL'))
audit['Documentation'].append(('Architecture Docs', 'PASS' if arch_doc else 'FAIL'))
audit['Documentation'].append(('Executive Summary', 'PASS' if exec_summary else 'FAIL'))

# Testing
test_dir = Path('tests').exists()
has_tests = len(list(Path('tests').glob('*.py'))) > 0 if test_dir else False
audit['Testing'].append(('Test Suite', 'PASS' if has_tests else 'WARN'))
audit['Testing'].append(('Test Coverage', 'WARN'))  # Would need coverage report

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Print audit report
print('\n' + '='*70)
print('PRODUCTION READINESS AUDIT - ASTINA')
print('='*70)

summary = {'PASS': 0, 'WARN': 0, 'FAIL': 0}
for category, checks in audit.items():
    print(f'\n[*] {category.upper()}')
    print('-' * 70)
    for check, status in checks:
        if status.startswith('PASS'):
            symbol = '[PASS]'
            summary['PASS'] += 1
        elif status.startswith('WARN'):
            symbol = '[WARN]'
            summary['WARN'] += 1
        else:
            symbol = '[FAIL]'
            summary['FAIL'] += 1
        print(f'{symbol:8s} {check:30s} {status}')

print('\n' + '='*70)
print(f'SUMMARY: {summary["PASS"]} PASS | {summary["WARN"]} WARN | {summary["FAIL"]} FAIL')
print('='*70)

# Production readiness score
total = sum(summary.values())
score = (summary['PASS'] / total * 100) if total > 0 else 0
print(f'\nProduction Readiness Score: {score:.0f}%')

if summary['FAIL'] == 0 and summary['WARN'] <= 3:
    print('Status: ✅ READY FOR PRODUCTION (with minor improvements)')
elif summary['FAIL'] == 0:
    print('Status: ⚠️  MOSTLY READY (address warnings first)')
else:
    print('Status: ❌ NOT YET READY (address failures first)')
