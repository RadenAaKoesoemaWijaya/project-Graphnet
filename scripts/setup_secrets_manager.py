#!/usr/bin/env python3
"""
ASTINA Google Secret Manager Setup Script

This script helps setup Google Secret Manager for ASTINA production deployment.
It creates secrets and grants Cloud Run service account access.

Prerequisites:
- gcloud CLI installed and configured
- Project ID set: gcloud config set project YOUR_PROJECT_ID
- Appropriate IAM permissions (Secret Manager Admin, Service Account Admin)

Usage:
    python scripts/setup_secrets_manager.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd: str, check: bool = True) -> tuple:
    """Run shell command and return (success, output)"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def get_project_id() -> str:
    """Get current Google Cloud project ID"""
    success, output = run_command("gcloud config get-value project")
    if success:
        project_id = output.strip()
        if project_id:
            return project_id
    return None

def get_project_number(project_id: str) -> str:
    """Get project number from project ID"""
    success, output = run_command(f"gcloud projects describe {project_id} --format='value(projectNumber)'")
    if success:
        return output.strip()
    return None

def create_secret(secret_name: str, secret_value: str) -> bool:
    """Create a secret in Secret Manager"""
    print(f"Creating secret: {secret_name}")
    
    # Create secret
    cmd = f'echo "{secret_value}" | gcloud secrets create {secret_name} --data-file=-'
    success, output = run_command(cmd, check=False)
    
    if success:
        print(f"  ✅ Secret created: {secret_name}")
        return True
    else:
        # Secret might already exist, try to add version
        print(f"  ⚠️  Secret might already exist, adding new version...")
        cmd = f'echo "{secret_value}" | gcloud secrets versions add {secret_name} --data-file=-'
        success, output = run_command(cmd, check=False)
        if success:
            print(f"  ✅ New version added: {secret_name}")
            return True
        else:
            print(f"  ❌ Failed to create secret: {secret_name}")
            print(f"     Error: {output}")
            return False

def grant_secret_access(secret_name: str, service_account: str) -> bool:
    """Grant Secret Manager access to Cloud Run service account"""
    print(f"Granting access to {service_account} for secret: {secret_name}")
    
    cmd = f'gcloud secrets add-iam-policy-binding {secret_name} --member="serviceAccount:{service_account}" --role="roles/secretmanager.secretAccessor"'
    success, output = run_command(cmd, check=False)
    
    if success:
        print(f"  ✅ Access granted")
        return True
    else:
        print(f"  ⚠️  Access might already be granted or failed")
        return True  # Continue even if this fails

def setup_astina_secrets():
    """Setup all ASTINA secrets in Secret Manager"""
    print("=" * 60)
    print("ASTINA Google Secret Manager Setup")
    print("=" * 60)
    
    # Get project ID
    project_id = get_project_id()
    if not project_id:
        print("ERROR: Project ID not set")
        print("Run: gcloud config set project YOUR_PROJECT_ID")
        return False
    
    print(f"\nProject ID: {project_id}")
    
    # Get project number
    project_number = get_project_number(project_id)
    if not project_number:
        print("ERROR: Could not get project number")
        return False
    
    print(f"Project Number: {project_number}")
    
    # Cloud Run service account
    service_account = f"{project_number}-compute@developer.gserviceaccount.com"
    print(f"Service Account: {service_account}")
    
    print("\n" + "=" * 60)
    print("Secret Creation")
    print("=" * 60)
    
    # Collect secret values
    print("\nEnter secret values (or press Enter to skip):")
    print("=" * 60)
    
    secrets = {}
    
    # Admin password
    admin_password = input("Admin password (or leave empty to skip): ").strip()
    if admin_password:
        secrets['astina-admin-password'] = admin_password
    
    # Auditor password
    auditor_password = input("Auditor password (or leave empty to skip): ").strip()
    if auditor_password:
        secrets['astina-auditor-password'] = auditor_password
    
    # Analyst password
    analyst_password = input("Analyst password (or leave empty to skip): ").strip()
    if analyst_password:
        secrets['astina-analyst-password'] = analyst_password
    
    # Viewer password
    viewer_password = input("Viewer password (or leave empty to skip): ").strip()
    if viewer_password:
        secrets['astina-viewer-password'] = viewer_password
    
    # Gemini API key
    gemini_key = input("Gemini API key (or leave empty to skip): ").strip()
    if gemini_key:
        secrets['gemini-api-key'] = gemini_key
    
    # OpenAI API key
    openai_key = input("OpenAI API key (or leave empty to skip): ").strip()
    if openai_key:
        secrets['openai-api-key'] = openai_key
    
    # Database password
    db_password = input("Database password (or leave empty to skip): ").strip()
    if db_password:
        secrets['database-password'] = db_password
    
    if not secrets:
        print("\nNo secrets to create. Exiting.")
        return True
    
    # Create secrets
    print("\n" + "=" * 60)
    print("Creating Secrets...")
    print("=" * 60)
    
    all_success = True
    for secret_name, secret_value in secrets.items():
        if not create_secret(secret_name, secret_value):
            all_success = False
    
    # Grant access
    print("\n" + "=" * 60)
    print("Granting Access to Cloud Run Service Account")
    print("=" * 60)
    
    for secret_name in secrets.keys():
        grant_secret_access(secret_name, service_account)
    
    # Generate deployment command
    print("\n" + "=" * 60)
    print("Deployment Command")
    print("=" * 60)
    
    secret_refs = []
    for secret_name in secrets.keys():
        secret_refs.append(f"{secret_name.upper()}={secret_name}:latest")
    
    secret_refs_str = ",".join(secret_refs)
    
    print(f"\nTo deploy with secrets, run:")
    print(f"gcloud run deploy astina \\")
    print(f"  --region=asia-southeast2 \\")
    print(f"  --set-secrets=\"{secret_refs_str}\"")
    
    # Update cloudbuild.yaml
    print("\n" + "=" * 60)
    print("Update Cloud Build Configuration")
    print("=" * 60)
    
    print(f"\nUpdate cloudbuild.yaml deploy step with:")
    print(f"--set-secrets=\"{secret_refs_str}\"")
    
    print("\n" + "=" * 60)
    if all_success:
        print("✅ SECRET MANAGER SETUP COMPLETED")
    else:
        print("⚠️  SECRET MANAGER SETUP COMPLETED WITH ERRORS")
    print("=" * 60)
    
    return all_success

def main():
    """Main entry point"""
    try:
        success = setup_astina_secrets()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()