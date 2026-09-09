#!/usr/bin/env python3
"""
ASTINA Cache Cleanup Scheduler

This script automates cache cleanup for data governance and privacy compliance.
It can be run as a cron job, systemd service, or Cloud Scheduler job.

Usage:
    python scripts/cache_cleanup_scheduler.py --max-age-hours 24
    python scripts/cache_cleanup_scheduler.py --dry-run
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache_manager import purge_expired_cache, get_cache_stats, CACHE_DIR, ensure_cache_dir

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/cache_cleanup.log')
        ]
    )
    return logging.getLogger(__name__)

def cleanup_cache(max_age_hours: float = 24.0, dry_run: bool = False) -> dict:
    """
    Perform cache cleanup with data governance logging
    
    Args:
        max_age_hours: Maximum age of cache files in hours before deletion
        dry_run: If True, only report what would be deleted without actually deleting
    
    Returns:
        Dictionary with cleanup statistics
    """
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("ASTINA Cache Cleanup Started")
    logger.info(f"Max Age: {max_age_hours} hours")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("=" * 60)
    
    # Ensure cache directory exists
    ensure_cache_dir()
    
    # Get cache statistics before cleanup
    stats_before = get_cache_stats()
    logger.info(f"Cache Statistics Before:")
    logger.info(f"  Total Files: {stats_before['total_files']}")
    logger.info(f"  Total Size: {stats_before['total_size_mb']:.2f} MB")
    logger.info(f"  Oldest File: {stats_before['oldest_file']}")
    logger.info(f"  Newest File: {stats_before['newest_file']}")
    
    if dry_run:
        logger.info("\n[Dry Run Mode] - No files will be deleted")
        logger.info("To perform actual cleanup, remove --dry-run flag")
        return {
            'dry_run': True,
            'stats_before': stats_before,
            'purged_count': 0,
            'stats_after': stats_before
        }
    
    # Perform cleanup
    logger.info("\nStarting cache cleanup...")
    purged_count = purge_expired_cache(max_age_hours=max_age_hours)
    
    # Get cache statistics after cleanup
    stats_after = get_cache_stats()
    logger.info(f"\nCache Statistics After:")
    logger.info(f"  Total Files: {stats_after['total_files']}")
    logger.info(f"  Total Size: {stats_after['total_size_mb']:.2f} MB")
    logger.info(f"  Oldest File: {stats_after['oldest_file']}")
    logger.info(f"  Newest File: {stats_after['newest_file']}")
    
    # Calculate cleanup metrics
    files_removed = stats_before['total_files'] - stats_after['total_files']
    size_freed_mb = stats_before['total_size_mb'] - stats_after['total_size_mb']
    
    logger.info("\n" + "=" * 60)
    logger.info("Cleanup Summary:")
    logger.info(f"  Files Purged: {purged_count}")
    logger.info(f"  Files Removed: {files_removed}")
    logger.info(f"  Size Freed: {size_freed_mb:.2f} MB")
    logger.info("=" * 60)
    
    return {
        'dry_run': False,
        'stats_before': stats_before,
        'purged_count': purged_count,
        'stats_after': stats_after,
        'files_removed': files_removed,
        'size_freed_mb': size_freed_mb
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='ASTINA Cache Cleanup Scheduler - Data Governance & Privacy Compliance'
    )
    parser.add_argument(
        '--max-age-hours',
        type=float,
        default=24.0,
        help='Maximum age of cache files in hours before deletion (default: 24)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report what would be deleted without actually deleting'
    )
    
    args = parser.parse_args()
    
    try:
        result = cleanup_cache(
            max_age_hours=args.max_age_hours,
            dry_run=args.dry_run
        )
        sys.exit(0)
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Cache cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()