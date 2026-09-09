#!/usr/bin/env python3
"""
ASTINA Production Environment Setup Script

This script helps setup production environment configuration safely.
It generates secure passwords and creates .env.production from template.

Usage:
    python scripts/setup_production_env.py
"""

import os
import sys
import secrets
import shutil
from pathlib import Path

def generate_secure_password(length: int = 16) -> str:
    """Generate a cryptographically secure random password"""
    return secrets.token_urlsafe(length)

def setup_production_environment():
    """Setup production environment configuration"""
    print("Setting up ASTINA Production Environment...")
    print("=" * 60)
    
    # Check if template exists
    template_path = Path('.env.production.template')
    if not template_path.exists():
        print("ERROR: .env.production.template not found")
        print("Please ensure .env.production.template exists in the project root")
        return False
    
    # Generate secure passwords
    print("\nGenerating secure passwords...")
    admin_password = generate_secure_password(16)
    auditor_password = generate_secure_password(16)
    analyst_password = generate_secure_password(16)
    viewer_password = generate_secure_password(16)
    
    print(f"Admin Password: {admin_password}")
    print(f"Auditor Password: {auditor_password}")
    print(f"Analyst Password: {analyst_password}")
    print(f"Viewer Password: {viewer_password}")
    
    # Read template
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Replace placeholders
    production_content = template_content.replace(
        'CHANGE_THIS_IN_PRODUCTION_MIN_12_CHARS', admin_password, 1
    ).replace(
        'CHANGE_THIS_IN_PRODUCTION_MIN_12_CHARS', auditor_password, 1
    ).replace(
        'CHANGE_THIS_IN_PRODUCTION_MIN_12_CHARS', analyst_password, 1
    ).replace(
        'CHANGE_THIS_IN_PRODUCTION_MIN_12_CHARS', viewer_password, 1
    ).replace(
        'CHANGE_THIS_IN_PRODUCTION_OR_USE_SECRET_MANAGER', 'your-api-key-here'
    )
    
    # Write to .env.production
    production_path = Path('.env.production')
    with open(production_path, 'w') as f:
        f.write(production_content)
    
    print(f"\n✅ Created .env.production with secure configuration")
    print(f"   File location: {production_path.absolute()}")
    
    # Ask user for additional configuration
    print("\nOptional Configuration:")
    print("=" * 60)
    
    # GCS Bucket
    gcs_bucket = input("Enter Google Cloud Storage bucket name (or press Enter to skip): ").strip()
    if gcs_bucket:
        with open(production_path, 'r') as f:
            content = f.read()
        content = content.replace('your-production-bucket-name-here', gcs_bucket)
        with open(production_path, 'w') as f:
            f.write(content)
        print(f"✅ GCS bucket configured: {gcs_bucket}")
    
    # Database (optional)
    use_db = input("Do you want to configure database? (y/n): ").strip().lower()
    if use_db == 'y':
        db_host = input("Enter database host (default: localhost): ").strip() or 'localhost'
        db_port = input("Enter database port (default: 5432): ").strip() or '5432'
        db_name = input("Enter database name (default: astina): ").strip() or 'astina'
        db_user = input("Enter database user (default: postgres): ").strip() or 'postgres'
        db_password = generate_secure_password(16)
        
        with open(production_path, 'r') as f:
            content = f.read()
        content = content.replace('localhost', db_host, 1)
        content = content.replace('5432', db_port, 1)
        content = content.replace('astina', db_name, 1)
        content = content.replace('postgres', db_user, 1)
        content = content.replace('CHANGE_THIS_IN_PRODUCTION_OR_USE_SECRET_MANAGER', db_password)
        content = content.replace('ENABLE_DATABASE=0', 'ENABLE_DATABASE=1')
        
        with open(production_path, 'w') as f:
            f.write(content)
        
        print("✅ Database configuration completed")
    
    # LLM Provider
    llm_provider = input("Select LLM provider (heuristic/gemini/openai/ollama) [default: heuristic]: ").strip().lower() or 'heuristic'
    if llm_provider != 'heuristic':
        with open(production_path, 'r') as f:
            content = f.read()
        content = content.replace('LLM_PROVIDER=heuristic', f'LLM_PROVIDER={llm_provider}')
        
        if llm_provider in ['gemini', 'openai']:
            api_key = input(f"Enter {llm_provider.upper()} API key: ").strip()
            if api_key:
                if llm_provider == 'gemini':
                    content = content.replace('your-api-key-here', api_key)
                else:
                    content = content.replace('your-api-key-here', api_key)
                print(f"✅ {llm_provider.upper()} API key configured")
        
        with open(production_path, 'w') as f:
            f.write(content)
    
    print("\n" + "=" * 60)
    print("PRODUCTION ENVIRONMENT SETUP COMPLETED")
    print("=" * 60)
    print("\nIMPORTANT SECURITY NOTES:")
    print("1. Save the generated passwords in a secure password manager")
    print("2. Never commit .env.production to version control")
    print("3. Consider using Google Secret Manager for production")
    print("4. Rotate passwords every 90 days")
    print("5. Test authentication before deploying to production")
    
    print(f"\nGenerated passwords saved to: {production_path.absolute()}")
    print("You can customize these values by editing the file directly.")
    
    return True

def main():
    """Main entry point"""
    try:
        success = setup_production_environment()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()