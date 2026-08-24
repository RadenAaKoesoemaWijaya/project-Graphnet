import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd

# File upload configuration for large files
MAX_FILE_SIZE = 3 * 1024 * 1024 * 1024  # 3GiB in bytes
CHUNK_SIZE = 50 * 1024 * 1024  # 50MB chunks for processing

# Temporary directory for processed data
TEMP_DATA_DIR = os.path.join(tempfile.gettempdir(), "astina_temp_data")
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# Memory optimization settings
MEMORY_LIMIT_MB = 4096  # 4GB memory limit for processing
PANDAS_DTYPE_OPTIMIZATION = True
USE_CATEGORICAL_OPTIMIZATION = True

# Streamlit file uploader configuration
UPLOADER_CONFIG = {
    'type': ['csv', 'parquet', 'xlsx', 'json'],
    'max_file_size': MAX_FILE_SIZE,
    'help': f'Upload file up to {MAX_FILE_SIZE // (1024*1024*1024)}GiB. Large files are processed in partitions.'
}

# Processing configuration for large datasets
LARGE_DATASET_CONFIG = {
    'chunk_size': 50000,  # Reduced from 100,000 for memory stability
    'sample_size': 10000,  # sample size for initial analysis
    'max_memory_usage': 0.8,  # maximum memory usage percentage
    'enable_progress_bar': True,
    'enable_caching': True
}

# GNN sampling defaults for large graph training.
GNN_SAMPLING_CONFIG = {
    'use_neighbor_sampling': True,
    'sampling_threshold_nodes': 20000,
    'batch_size_cpu': 512,
    'batch_size_gpu': 2048,
    'num_neighbors': [15, 10],
}

# Column type optimization
DTYPE_OPTIMIZATION = {
    'int64': 'int32',
    'float64': 'float32',
    'object': 'category'  # for columns with low cardinality
}

# Memory management
MEMORY_CONFIG = {
    'gc_threshold': 0.8,  # garbage collection threshold
    'clear_intermediate': True,  # clear intermediate dataframes
    'use_lazy_loading': True  # load data only when needed
}

REPEAT_BILLING_CONFIG = {
    'enabled': True,
    'temporal_window_days': 30,
    'fuzzy_match_threshold': 0.85,
    'amount_variance_tolerance_pct': 5.0,
    'min_risk_score_for_alert': 0.7,
}

PHANTOM_SERVICE_CONFIG = {
    'enabled': True,
    'enable_rule_validation': True,
    'enable_provider_capacity_check': True,
    'enable_frequency_check': True,
    'rules_update_interval_days': 7,
}

SERVICE_CAPACITY = {
    'CT_SCAN': {'max_per_day': 2, 'duration_minutes': 30, 'requires_specialist': True},
    'X_RAY': {'max_per_day': 5, 'duration_minutes': 10, 'requires_specialist': False},
    'BLOOD_TEST': {'max_per_day': 20, 'duration_minutes': 5, 'requires_specialist': False},
    'CONSULTATION': {'max_per_day': 15, 'duration_minutes': 20, 'requires_specialist': True},
    'ULTRASOUND': {'max_per_day': 5, 'duration_minutes': 25, 'requires_specialist': True},
    'LABORATORY': {'max_per_day': 10, 'duration_minutes': 8, 'requires_specialist': False},
    'SURGERY': {'max_per_day': 1, 'duration_minutes': 180, 'requires_specialist': True},
}

UNREALISTIC_PATTERNS = {
    'max_surgeries_per_day': 1,
    'max_ct_scans_per_day_per_patient': 1,
    'min_time_between_same_service_days': 14,
}

DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'username': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD'),  # MUST be set via environment variable, no default!
    'database': os.getenv('DB_NAME', 'astina'),
    'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
    'timeout_seconds': int(os.getenv('DB_TIMEOUT', '5')),
}

# Validate critical database configuration
def _validate_database_config():
    """Validate database config if database mode is enabled."""
    db_enabled = os.getenv("ENABLE_DATABASE", "0").lower() in ("1", "true", "yes")
    if db_enabled and DATABASE_CONFIG['password'] is None:
        import warnings
        warning_category = getattr(warnings, 'SecurityWarning', UserWarning)
        warnings.warn(
            "⚠️  DB_PASSWORD environment variable not set. "
            "If you need database connectivity, please set it:\n"
            "  export DB_PASSWORD='your_secure_password'\n"
            "For production, use Secret Manager or similar secure vault.",
            warning_category,
            stacklevel=2,
        )
    return True

# Run validation on module import
_validate_database_config()

def check_file_size(file_size_bytes):
    """Check if file size is within limits"""
    if file_size_bytes > MAX_FILE_SIZE:
        raise ValueError(f"File size ({file_size_bytes / (1024*1024*1024):.2f}GB) exceeds maximum allowed size ({MAX_FILE_SIZE / (1024*1024*1024)}GB)")
    return True

def get_optimal_chunk_size(file_size_bytes):
    """Calculate optimal chunk size based on file size"""
    if file_size_bytes < 100 * 1024 * 1024:  # < 100MB
        return None  # Process whole file
    elif file_size_bytes < 500 * 1024 * 1024:  # < 500MB
        return 50000  # 50k rows per chunk
    elif file_size_bytes < 1024 * 1024 * 1024:  # < 1GB
        return 50000  # 50k rows per chunk
    else:  # >= 1GB
        return 100000  # 100k rows per chunk

def optimize_memory_usage(df):
    """Optimize DataFrame memory usage"""
    if not PANDAS_DTYPE_OPTIMIZATION:
        return df

    original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB

    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        col_type = df[col].dtype

        if col_type == 'int64':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)

        elif col_type == 'float64':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    # Optimize object columns
    if USE_CATEGORICAL_OPTIMIZATION:
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:  # Less than 50% unique values
                # Don't convert to category if there are missing values to avoid setitem error
                if df[col].isnull().sum() == 0:
                    df[col] = df[col].astype('category')

    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
    memory_saved = original_memory - optimized_memory

    return df

def optimize_memory_usage_aggressive(df):
    """More aggressive memory optimization for large datasets"""
    original_memory = df.memory_usage(deep=True).sum() / 1024**2  # GB

    # Downcast integers using pandas to_numeric
    for col in df.select_dtypes(include=['int64', 'int32', 'int16']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # Downcast floats using pandas to_numeric
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Convert object to category more aggressively (increased threshold from 50% to 30%)
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.3:  # Less than 30% unique
            # Don't convert to category if there are missing values to avoid setitem error
            if df[col].isnull().sum() == 0:
                df[col] = df[col].astype('category')

    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    memory_saved = original_memory - optimized_memory

    return df

import numpy as np  # Add this import at the top of the file