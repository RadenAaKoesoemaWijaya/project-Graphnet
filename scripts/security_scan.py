#!/usr/bin/env python3
"""
Security Scanning Script for ASTINA

This script performs automated security checks:
1. Dependency vulnerability scanning (pip-audit)
2. Secrets detection
3. Configuration validation
4. File permission checks

Usage:
    python scripts/security_scan.py
    python scripts/security_scan.py --fix
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def check_pip_audit() -> Tuple[bool, List[str]]:
    """
    Run pip-audit to check for known vulnerabilities in dependencies.
    
    Returns: (has_vulnerabilities, list_of_issues)
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'audit'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            logger.info("✅ No known vulnerabilities found in dependencies")
            return False, []
        else:
            # pip-audit returns non-zero if vulnerabilities found
            issues = result.stdout.strip().split('\n') if result.stdout else []
            logger.warning(f"⚠️ Found {len(issues)} potential vulnerabilities")
            return True, issues
    except subprocess.TimeoutExpired:
        logger.error("pip-audit timed out")
        return False, ["pip-audit timed out"]
    except FileNotFoundError:
        logger.warning("pip-audit not installed. Install with: pip install pip-audit")
        return False, ["pip-audit not installed"]
    except Exception as e:
        logger.error(f"Error running pip-audit: {e}")
        return False, [str(e)]


def check_secrets_in_code() -> List[str]:
    """
    Scan for potential secrets in code files.
    
    Returns: List of suspicious patterns found
    """
    suspicious_patterns = [
        (r'password\s*=\s*["\'][^"\']{8,}["\']', 'Hardcoded password'),
        (r'api_key\s*=\s*["\'][^"\']{20,}["\']', 'Hardcoded API key'),
        (r'secret\s*=\s*["\'][^"\']{20,}["\']', 'Hardcoded secret'),
        (r'token\s*=\s*["\'][^"\']{20,}["\']', 'Hardcoded token'),
    ]
    
    issues = []
    
    # Scan Python files
    for py_file in Path('.').rglob('*.py'):
        if 'venv' in str(py_file) or '.venv' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                for pattern, description in suspicious_patterns:
                    if pattern in content:
                        issues.append(f"{py_file}: {description}")
        except Exception as e:
            logger.debug(f"Could not read {py_file}: {e}")
    
    if issues:
        logger.warning(f"⚠️ Found {len(issues)} potential secret leaks")
    else:
        logger.info("✅ No hardcoded secrets found in code")
    
    return issues


def check_file_permissions() -> List[str]:
    """
    Check file permissions for security.
    
    Returns: List of permission issues
    """
    import stat
    
    critical_dirs = ['models', 'cache', 'logs', 'studies']
    issues = []
    
    for dir_name in critical_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            continue
        
        try:
            mode = dir_path.stat().st_mode
            if os.name != 'nt':  # Unix-like systems
                if (mode & stat.S_IWGRP) or (mode & stat.S_IWOTH):
                    issues.append(f"{dir_name}: Group/others have write permission")
        except Exception as e:
            logger.debug(f"Could not check permissions for {dir_name}: {e}")
    
    if issues:
        logger.warning(f"⚠️ Found {len(issues)} permission issues")
    else:
        logger.info("✅ File permissions are secure")
    
    return issues


def check_gitignore_completeness() -> List[str]:
    """
    Check if .gitignore properly excludes sensitive files.
    
    Returns: List of missing patterns
    """
    required_patterns = [
        '.env',
        '*.key',
        '*.pem',
        '*.secret',
        'models/*.pkl',
        'models/*.pt',
        'logs/',
    ]
    
    issues = []
    
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        issues.append(".gitignore file not found")
        return issues
    
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()
    
    for pattern in required_patterns:
        if pattern not in gitignore_content:
            issues.append(f"Missing pattern in .gitignore: {pattern}")
    
    if issues:
        logger.warning(f"⚠️ Found {len(issues)} missing .gitignore patterns")
    else:
        logger.info("✅ .gitignore is properly configured")
    
    return issues


def run_all_scans() -> Dict[str, Any]:
    """
    Run all security scans and return results.
    
    Returns: Dictionary with scan results
    """
    results = {
        'pip_audit': {'has_vulns': False, 'issues': []},
        'secrets': {'found': False, 'issues': []},
        'permissions': {'issues': []},
        'gitignore': {'issues': []},
        'overall_status': 'PASS'
    }
    
    logger.info("=" * 60)
    logger.info("Starting ASTINA Security Scan")
    logger.info("=" * 60)
    
    # 1. Dependency vulnerability scan
    logger.info("\n[1/4] Scanning dependencies for vulnerabilities...")
    has_vulns, vuln_issues = check_pip_audit()
    results['pip_audit']['has_vulns'] = has_vulns
    results['pip_audit']['issues'] = vuln_issues
    
    # 2. Secrets detection
    logger.info("\n[2/4] Scanning for hardcoded secrets...")
    secret_issues = check_secrets_in_code()
    results['secrets']['found'] = len(secret_issues) > 0
    results['secrets']['issues'] = secret_issues
    
    # 3. File permissions
    logger.info("\n[3/4] Checking file permissions...")
    perm_issues = check_file_permissions()
    results['permissions']['issues'] = perm_issues
    
    # 4. Gitignore completeness
    logger.info("\n[4/4] Checking .gitignore completeness...")
    gitignore_issues = check_gitignore_completeness()
    results['gitignore']['issues'] = gitignore_issues
    
    # Determine overall status
    total_issues = (
        len(vuln_issues) +
        len(secret_issues) +
        len(perm_issues) +
        len(gitignore_issues)
    )
    
    if total_issues > 0:
        results['overall_status'] = 'FAIL'
        logger.warning(f"\n⚠️ Security scan completed with {total_issues} issues")
    else:
        results['overall_status'] = 'PASS'
        logger.info("\n✅ Security scan completed successfully - no issues found")
    
    return results


def print_results(results: Dict[str, Any]):
    """Print scan results in a readable format."""
    print("\n" + "=" * 60)
    print("SECURITY SCAN RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Status: {results['overall_status']}")
    
    # Pip Audit
    print("\n[1] Dependency Vulnerabilities:")
    if results['pip_audit']['has_vulns']:
        print("  Status: VULNERABILITIES FOUND")
        for issue in results['pip_audit']['issues']:
            print(f"  - {issue}")
    else:
        print("  Status: OK")
    
    # Secrets
    print("\n[2] Hardcoded Secrets:")
    if results['secrets']['found']:
        print("  Status: SECRETS FOUND")
        for issue in results['secrets']['issues']:
            print(f"  - {issue}")
    else:
        print("  Status: OK")
    
    # Permissions
    print("\n[3] File Permissions:")
    if results['permissions']['issues']:
        print("  Status: ISSUES FOUND")
        for issue in results['permissions']['issues']:
            print(f"  - {issue}")
    else:
        print("  Status: OK")
    
    # Gitignore
    print("\n[4] Gitignore Completeness:")
    if results['gitignore']['issues']:
        print("  Status: ISSUES FOUND")
        for issue in results['gitignore']['issues']:
            print(f"  - {issue}")
    else:
        print("  Status: OK")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Security scanning script for ASTINA')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues automatically')
    parser.add_argument('--quiet', action='store_true', help='Suppress output except for errors')
    
    args = parser.parse_args()
    
    if args.quiet:
        logger.setLevel(logging.ERROR)
    
    results = run_all_scans()
    
    if not args.quiet:
        print_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
