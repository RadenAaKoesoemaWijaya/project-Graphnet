#!/usr/bin/env python3
"""
ASTINA Backup Scheduler

This script automates backup of model artifacts, cache, and important data.
It supports local backup and Google Cloud Storage backup.

Usage:
    python scripts/backup_scheduler.py --local
    python scripts/backup_scheduler.py --gcs --bucket your-bucket-name
    python scripts/backup_scheduler.py --local --gcs --bucket your-bucket-name
"""

import argparse
import sys
import os
import shutil
import gzip
import tarfile
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/backup.log')
        ]
    )
    return logging.getLogger(__name__)

def create_local_backup(
    backup_dir: str = "backups",
    sources: Optional[List[str]] = None,
    compress: bool = True
) -> dict:
    """
    Create local backup of specified directories
    
    Args:
        backup_dir: Directory to store backups
        sources: List of source directories to backup
        compress: Whether to compress the backup
    
    Returns:
        Dictionary with backup statistics
    """
    logger = setup_logging()
    
    if sources is None:
        sources = ['models', 'cache', 'logs']
    
    logger.info("=" * 60)
    logger.info("ASTINA Local Backup Started")
    logger.info("=" * 60)
    
    # Create backup directory
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"astina_backup_{timestamp}"
    
    if compress:
        backup_filename += ".tar.gz"
    else:
        backup_filename += ".tar"
    
    backup_file = backup_path / backup_filename
    
    logger.info(f"Backup file: {backup_file}")
    logger.info(f"Sources: {sources}")
    
    # Check which sources exist
    existing_sources = []
    for source in sources:
        source_path = Path(source)
        if source_path.exists():
            existing_sources.append(source)
            logger.info(f"  ✓ {source} ({source_path.stat().st_size / (1024*1024):.2f} MB)")
        else:
            logger.warning(f"  ✗ {source} (not found)")
    
    if not existing_sources:
        logger.error("No sources found to backup")
        return {'success': False, 'backup_file': None}
    
    # Create backup
    logger.info("\nCreating backup...")
    try:
        if compress:
            with tarfile.open(backup_file, "w:gz") as tar:
                for source in existing_sources:
                    tar.add(source, arcname=source)
        else:
            with tarfile.open(backup_file, "w") as tar:
                for source in existing_sources:
                    tar.add(source, arcname=source)
        
        backup_size = backup_file.stat().st_size / (1024*1024)
        logger.info(f"✅ Backup created successfully")
        logger.info(f"   Size: {backup_size:.2f} MB")
        logger.info(f"   Location: {backup_file.absolute()}")
        
        return {
            'success': True,
            'backup_file': str(backup_file.absolute()),
            'backup_size_mb': backup_size,
            'sources_backed_up': existing_sources,
            'timestamp': timestamp
        }
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'backup_file': None}

def cleanup_old_backups(
    backup_dir: str = "backups",
    keep_count: int = 5
) -> int:
    """
    Clean up old backups, keeping only the most recent ones
    
    Args:
        backup_dir: Directory containing backups
        keep_count: Number of recent backups to keep
    
    Returns:
        Number of backups removed
    """
    logger = setup_logging()
    
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        logger.info(f"Backup directory {backup_dir} does not exist")
        return 0
    
    # Get all backup files
    backup_files = sorted(
        backup_path.glob("astina_backup_*.tar*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    logger.info(f"Found {len(backup_files)} backup files")
    
    # Remove old backups
    removed_count = 0
    for old_backup in backup_files[keep_count:]:
        try:
            old_backup.unlink()
            logger.info(f"Removed old backup: {old_backup.name}")
            removed_count += 1
        except Exception as e:
            logger.error(f"Failed to remove {old_backup.name}: {e}")
    
    logger.info(f"Removed {removed_count} old backups")
    return removed_count

def create_gcs_backup(
    bucket_name: str,
    sources: Optional[List[str]] = None,
    compress: bool = True
) -> dict:
    """
    Create backup to Google Cloud Storage
    
    Args:
        bucket_name: GCS bucket name
        sources: List of source directories to backup
        compress: Whether to compress the backup
    
    Returns:
        Dictionary with backup statistics
    """
    logger = setup_logging()
    
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage not installed")
        logger.error("Install with: pip install google-cloud-storage")
        return {'success': False, 'error': 'google-cloud-storage not installed'}
    
    if sources is None:
        sources = ['models', 'cache', 'logs']
    
    logger.info("=" * 60)
    logger.info("ASTINA GCS Backup Started")
    logger.info(f"Bucket: {bucket_name}")
    logger.info("=" * 60)
    
    # First create local backup
    local_result = create_local_backup(
        backup_dir="temp_backups",
        sources=sources,
        compress=compress
    )
    
    if not local_result['success']:
        return {'success': False, 'error': 'Local backup failed'}
    
    local_backup_file = local_result['backup_file']
    
    # Upload to GCS
    logger.info("\nUploading to Google Cloud Storage...")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Create blob name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"astina_backups/astina_backup_{timestamp}.tar.gz"
        
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_backup_file)
        
        logger.info(f"✅ Upload completed successfully")
        logger.info(f"   GCS Path: gs://{bucket_name}/{blob_name}")
        
        # Clean up local backup
        Path(local_backup_file).unlink()
        Path("temp_backups").rmdir()
        
        return {
            'success': True,
            'gcs_path': f"gs://{bucket_name}/{blob_name}",
            'backup_size_mb': local_result['backup_size_mb'],
            'timestamp': timestamp
        }
        
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='ASTINA Backup Scheduler - Automated Model & Data Backup'
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Create local backup'
    )
    parser.add_argument(
        '--gcs',
        action='store_true',
        help='Create Google Cloud Storage backup'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        help='GCS bucket name (required for --gcs)'
    )
    parser.add_argument(
        '--sources',
        type=str,
        nargs='+',
        default=['models', 'cache', 'logs'],
        help='Source directories to backup'
    )
    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Do not compress backup'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up old backups (keep last 5)'
    )
    parser.add_argument(
        '--keep-count',
        type=int,
        default=5,
        help='Number of recent backups to keep (default: 5)'
    )
    
    args = parser.parse_args()
    
    if not args.local and not args.gcs and not args.cleanup:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Perform cleanup if requested
        if args.cleanup:
            cleanup_old_backups(keep_count=args.keep_count)
        
        # Create local backup
        if args.local:
            result = create_local_backup(
                sources=args.sources,
                compress=not args.no_compress
            )
            if not result['success']:
                sys.exit(1)
        
        # Create GCS backup
        if args.gcs:
            if not args.bucket:
                print("ERROR: --bucket is required for --gcs")
                sys.exit(1)
            result = create_gcs_backup(
                bucket_name=args.bucket,
                sources=args.sources,
                compress=not args.no_compress
            )
            if not result['success']:
                sys.exit(1)
        
        sys.exit(0)
        
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Backup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()