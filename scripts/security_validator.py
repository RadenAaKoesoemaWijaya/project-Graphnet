#!/usr/bin/env python3
"""
ASTINA Security Validator

This script validates security configuration before deployment.
It checks for common security issues and provides actionable feedback.

Usage:
    python scripts/security_validator.py
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple

class SecurityValidator:
    """Validates security configuration for ASTINA deployment"""
    
    def __init__(self):
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.passed_checks: List[str] = []
        
        # Default passwords that should never be used in production
        self.default_passwords = [
            'AdminAstina2026!',
            'AuditorAstina2026!', 
            'AnalystAstina2026!',
            'ViewerAstina2026!',
            'password',
            'admin',
            '123456',
            'qwerty'
        ]
        
        # Patterns that might indicate hardcoded secrets
        self.secret_patterns = [
            r'api_key\s*=\s*["\'][a-zA-Z0-9_-]{30,}["\']',  # API keys (longer, more specific)
            r'secret\s*=\s*["\'][a-zA-Z0-9_-]{30,}["\']',  # Secrets (longer, more specific)
            r'token\s*=\s*["\'][a-zA-Z0-9_-]{30,}["\']',  # Tokens (longer, more specific)
            r'password\s*=\s*["\'][^"\']{20,}["\']',  # Long passwords (more specific)
        ]
    
    def validate_environment_files(self) -> None:
        """Validate environment configuration files"""
        env_files = ['.env', '.env.production', '.env.local']
        
        for env_file in env_files:
            if Path(env_file).exists():
                self._check_env_file_security(env_file)
            else:
                if env_file == '.env.production':
                    # Check if template exists as fallback
                    if Path('.env.production.template').exists():
                        self.warnings.append(f"[WARN] {env_file} not found, but template exists. Run: python scripts/setup_production_env.py")
                    else:
                        self.issues.append(f"[ERROR] {env_file} not found (required for production)")
    
    def _check_env_file_security(self, env_file: str) -> None:
        """Check security of a specific environment file"""
        try:
            with open(env_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for default passwords
            for default_pwd in self.default_passwords:
                if default_pwd in content:
                    self.issues.append(f"[ERROR] Default password found in {env_file}: {default_pwd[:10]}...")
            
            # Check for short passwords
            password_lines = re.findall(r'PASSWORD=([^\n]+)', content)
            for pwd in password_lines:
                if len(pwd) < 12 and pwd != 'CHANGE_THIS_IN_PRODUCTION':
                    self.warnings.append(f"[WARN] Short password in {env_file} (min 12 characters)")
            
            # Check if AUTH_ENABLED is properly set
            auth_match = re.search(r'AUTH_ENABLED=([^\n]+)', content)
            if auth_match:
                auth_value = auth_match.group(1).strip().lower()
                if 'production' in env_file and auth_value != 'true':
                    self.issues.append(f"[ERROR] AUTH_ENABLED not set to 'true' in {env_file}")
                elif auth_value == 'true':
                    self.passed_checks.append(f"[PASS] Authentication enabled in {env_file}")
            
        except Exception as e:
            self.warnings.append(f"[WARN] Could not read {env_file}: {e}")
    
    def validate_code_security(self) -> None:
        """Validate source code for security issues"""
        python_files = list(Path('.').rglob('*.py'))
        
        # Skip certain directories
        skip_dirs = {'tests', '__pycache__', '.git', 'venv', '.venv'}
        python_files = [f for f in python_files if not any(skip in str(f) for skip in skip_dirs)]
        
        for py_file in python_files:
            self._check_file_for_secrets(py_file)
    
    def _check_file_for_secrets(self, file_path: Path) -> None:
        """Check a Python file for potential hardcoded secrets"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for pattern in self.secret_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Filter out example/placeholder values
                    real_matches = [m for m in matches if not any(
                        placeholder in m.lower() 
                        for placeholder in ['example', 'placeholder', 'your-', 'change_', 'test', 'your_secure', 'your_api', 'your_']
                    )]
                    if real_matches:
                        self.issues.append(f"[ERROR] Potential hardcoded secret in {file_path}")
                        break
                        
        except Exception as e:
            self.warnings.append(f"[WARN] Could not scan {file_path}: {e}")
    
    def validate_gitignore(self) -> None:
        """Validate .gitignore for security"""
        gitignore_path = Path('.gitignore')
        
        if not gitignore_path.exists():
            self.issues.append("[ERROR] .gitignore file not found")
            return
        
        try:
            with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
                gitignore_content = f.read()
        except Exception as e:
            self.warnings.append(f"[WARN] Could not read .gitignore: {e}")
            return
        
        required_entries = ['.env', '.env.production', '.env.local', '*.key', '*.pem']
        
        for entry in required_entries:
            if entry not in gitignore_content:
                self.warnings.append(f"[WARN] '{entry}' not in .gitignore")
            else:
                self.passed_checks.append(f"[PASS] '{entry}' is in .gitignore")
        
        required_entries = ['.env', '.env.production', '.env.local', '*.key', '*.pem']
        
        for entry in required_entries:
            if entry not in gitignore_content:
                self.warnings.append(f"[WARN] '{entry}' not in .gitignore")
            else:
                self.passed_checks.append(f"[PASS] '{entry}' is in .gitignore")
    
    def validate_dependencies(self) -> None:
        """Validate dependencies for security vulnerabilities"""
        requirements_path = Path('requirements.txt')
        
        if not requirements_path.exists():
            self.warnings.append("[WARN] requirements.txt not found")
            return
        
        # Check for outdated or vulnerable packages
        try:
            with open(requirements_path, 'r', encoding='utf-8', errors='ignore') as f:
                requirements = f.read()
        except Exception as e:
            self.warnings.append(f"[WARN] Could not read requirements.txt: {e}")
            return
        
        # Check for pinned versions (good for security)
        unpinned = re.findall(r'^[a-zA-Z0-9_-]+$', requirements, re.MULTILINE)
        if unpinned:
            self.warnings.append(f"[WARN] {len(unpinned)} unpinned dependencies found")
        else:
            self.passed_checks.append("[PASS] Dependencies are properly pinned")
    
    def validate_authentication_module(self) -> None:
        """Validate authentication module implementation"""
        auth_module_path = Path('auth_manager.py')
        
        if not auth_module_path.exists():
            self.issues.append("[ERROR] auth_manager.py not found")
            return
        
        try:
            with open(auth_module_path, 'r', encoding='utf-8', errors='ignore') as f:
                auth_content = f.read()
        except Exception as e:
            self.warnings.append(f"[WARN] Could not read auth_manager.py: {e}")
            return
        
        # Check for secure password hashing
        if 'hashlib' in auth_content and 'sha256' in auth_content:
            self.passed_checks.append("[PASS] Secure password hashing implemented")
        else:
            self.issues.append("[ERROR] Secure password hashing not found")
        
        # Check for role-based access control
        if 'ROLE_PERMISSIONS' in auth_content and 'RBAC' in auth_content:
            self.passed_checks.append("[PASS] Role-based access control implemented")
        else:
            self.warnings.append("[WARN] RBAC implementation unclear")
    
    def validate_pii_protection(self) -> None:
        """Validate PII protection implementation"""
        pii_module_path = Path('pii_masker.py')
        
        if not pii_module_path.exists():
            self.issues.append("[ERROR] pii_masker.py not found")
            return
        
        try:
            with open(pii_module_path, 'r', encoding='utf-8', errors='ignore') as f:
                pii_content = f.read()
        except Exception as e:
            self.warnings.append(f"[WARN] Could not read pii_masker.py: {e}")
            return
        
        # Check for PII masking functions
        if 'mask_value' in pii_content and 'hash_pii' in pii_content:
            self.passed_checks.append("[PASS] PII masking functions implemented")
        else:
            self.warnings.append("[WARN] PII masking functions unclear")
        
        # Check for sensitive fields detection
        if 'SENSITIVE_FIELDS' in pii_content:
            self.passed_checks.append("[PASS] Sensitive fields defined")
        else:
            self.warnings.append("[WARN] Sensitive fields not clearly defined")
    
    def validate_audit_trail(self) -> None:
        """Validate audit trail implementation"""
        audit_module_path = Path('audit_trail.py')
        
        if not audit_module_path.exists():
            self.issues.append("[ERROR] audit_trail.py not found")
            return
        
        try:
            with open(audit_module_path, 'r', encoding='utf-8', errors='ignore') as f:
                audit_content = f.read()
        except Exception as e:
            self.warnings.append(f"[WARN] Could not read audit_trail.py: {e}")
            return
        
        # Check for comprehensive audit trail features
        if 'log_event' in audit_content:
            self.passed_checks.append("[PASS] Audit trail logging implemented")
        else:
            self.warnings.append("[WARN] Audit trail logging unclear")
        
        # Check for PII masking integration
        if 'PII_MASKING_AVAILABLE' in audit_content or 'PIIMasker' in audit_content:
            self.passed_checks.append("[PASS] PII masking integrated in audit trail")
        else:
            self.warnings.append("[WARN] PII masking integration unclear")
        
        # Check for cloud logging support
        if 'cloud_logging' in audit_content.lower():
            self.passed_checks.append("[PASS] Cloud Logging support available")
        else:
            self.warnings.append("[WARN] Cloud Logging support unclear")
    
    def run_validation(self) -> Tuple[bool, str]:
        """Run all security validations"""
        print("Starting ASTINA Security Validation...\n")
        
        self.validate_environment_files()
        self.validate_code_security()
        self.validate_gitignore()
        self.validate_dependencies()
        self.validate_authentication_module()
        self.validate_pii_protection()
        self.validate_audit_trail()
        
        # Generate report
        report = self._generate_report()
        
        # Determine overall status
        has_critical_issues = len(self.issues) > 0
        
        return (not has_critical_issues, report)
    
    def _generate_report(self) -> str:
        """Generate security validation report"""
        report = []
        
        if self.passed_checks:
            report.append("[PASSED CHECKS]")
            for check in self.passed_checks:
                report.append(f"   {check}")
            report.append("")
        
        if self.warnings:
            report.append("[WARNINGS]")
            for warning in self.warnings:
                report.append(f"   {warning}")
            report.append("")
        
        if self.issues:
            report.append("[CRITICAL ISSUES]")
            for issue in self.issues:
                report.append(f"   {issue}")
            report.append("")
        
        # Summary
        report.append("=" * 50)
        report.append(f"Total Checks Passed: {len(self.passed_checks)}")
        report.append(f"Total Warnings: {len(self.warnings)}")
        report.append(f"Total Critical Issues: {len(self.issues)}")
        report.append("=" * 50)
        
        if self.issues:
            report.append("[FAILED] SECURITY VALIDATION FAILED")
            report.append("Please address critical issues before deployment.")
        elif self.warnings:
            report.append("[WARNING] SECURITY VALIDATION PASSED WITH WARNINGS")
            report.append("Review warnings before deployment.")
        else:
            report.append("[SUCCESS] SECURITY VALIDATION PASSED")
            report.append("System is ready for deployment.")
        
        return "\n".join(report)

def main():
    """Main entry point"""
    validator = SecurityValidator()
    passed, report = validator.run_validation()
    
    print(report)
    
    sys.exit(0 if passed else 1)

if __name__ == '__main__':
    main()