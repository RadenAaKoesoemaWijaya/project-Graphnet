import pandas as pd
import numpy as np
import streamlit as st
import polars as pl
import psutil
import os
import gc
from large_file_processor import preprocess_large_dataset, optimize_categorical_encoding_large, smart_data_type_optimization
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def check_memory_threshold(threshold_mb=8000, warning_threshold_mb=6000, stop_on_critical=False):
    """
    Check if memory usage exceeds threshold and take action if needed.
    Made less aggressive to prevent early termination.

    Args:
        threshold_mb: Critical threshold in MB (default 8GB, increased from 4GB)
        warning_threshold_mb: Warning threshold in MB (default 6GB, increased from 3GB)
        stop_on_critical: If True, raise exception when critical threshold is reached (default False)

    Returns:
        memory_mb: Current memory usage in MB
        is_critical: Whether memory usage exceeds critical threshold
        should_continue: Whether processing should continue (always True unless stop_on_critical=True and critical)
    """
    memory_mb = get_memory_usage_mb()
    is_critical = memory_mb > threshold_mb
    is_warning = memory_mb > warning_threshold_mb
    should_continue = True

    if is_critical:
        if stop_on_critical:
            st.error(f"❌ Memory usage critical: {memory_mb:.0f}MB. Processing stopped. Consider reducing chunk size or dataset size.")
            gc.collect()
            should_continue = False
        else:
            st.warning(f"⚠️ Memory usage critical: {memory_mb:.0f}MB. Garbage collection triggered.")
            gc.collect()
    elif is_warning:
        st.info(f"ℹ️ Memory usage high: {memory_mb:.0f}MB. Processing will continue.")

    return memory_mb, is_critical, should_continue

def detect_and_handle_outliers(df, numerical_columns, method='iqr', threshold=3.0, action='cap'):
    """
    Detect and handle outliers using multiple methods.
    
    Args:
        df: Input dataframe
        numerical_columns: List of numerical columns to process
        method: 'iqr' or 'zscore' detection method
        threshold: Threshold for zscore method (default 3.0)
        action: 'cap' to cap outliers, 'remove' to remove rows, 'flag' to add indicator
    
    Returns:
        df: Processed dataframe
        outlier_metadata: Dictionary with outlier statistics
    """
    # Only copy if we need to remove rows (action='remove')
    if action == 'remove':
        df_processed = df.copy()
    else:
        df_processed = df  # Work in-place for cap and flag actions
    outlier_metadata = {}
    total_outliers = 0
    
    for col in numerical_columns:
        try:
            if col not in df_processed.columns:
                continue
                
            # Skip if too many missing values
            if df_processed[col].isnull().sum() / len(df_processed) > 0.5:
                continue
                
            if method == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                
            elif method == 'zscore':
                mean_val = df_processed[col].mean()
                std_val = df_processed[col].std()
                if std_val > 1e-6:  # Avoid division by zero
                    z_scores = np.abs((df_processed[col] - mean_val) / std_val)
                    outliers = z_scores > threshold
                else:
                    outliers = pd.Series([False] * len(df_processed), index=df_processed.index)
            else:
                continue
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                total_outliers += outlier_count
                
                if action == 'cap':
                    # Cap outliers to bounds
                    if method == 'iqr':
                        df_processed[col] = np.where(
                            df_processed[col] > upper_bound, upper_bound,
                            np.where(df_processed[col] < lower_bound, lower_bound, df_processed[col])
                        )
                    elif method == 'zscore':
                        mean_val = df_processed[col].mean()
                        std_val = df_processed[col].std()
                        upper_bound = mean_val + threshold * std_val
                        lower_bound = mean_val - threshold * std_val
                        df_processed[col] = np.where(
                            df_processed[col] > upper_bound, upper_bound,
                            np.where(df_processed[col] < lower_bound, lower_bound, df_processed[col])
                        )
                    
                    outlier_metadata[col] = {
                        'outliers_capped': int(outlier_count),
                        'method': method,
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
                    
                elif action == 'flag':
                    # Add outlier indicator column
                    df_processed[f'{col}_outlier'] = outliers.astype(np.int8)
                    outlier_metadata[col] = {
                        'outliers_flagged': int(outlier_count),
                        'method': method
                    }
                    
                elif action == 'remove':
                    # Remove outlier rows (use with caution)
                    df_processed = df_processed[~outliers].copy()
                    outlier_metadata[col] = {
                        'outliers_removed': int(outlier_count),
                        'method': method
                    }
                    
        except Exception as e:
            continue
    
    outlier_metadata['total_outliers_handled'] = total_outliers
    outlier_metadata['method_used'] = method
    outlier_metadata['action_taken'] = action
    
    return df_processed, outlier_metadata

def validate_data_ranges(df, numerical_columns):
    """
    Validate numerical columns against logical ranges and fix invalid values.
    
    Args:
        df: Input dataframe
        numerical_columns: List of numerical columns to validate
    
    Returns:
        df: Processed dataframe with fixed invalid values
        validation_metadata: Dictionary with validation statistics
    """
    # Work in-place to avoid unnecessary copy
    df_processed = df
    validation_metadata = {}
    
    # Age validation (0-120 years)
    age_cols = [col for col in numerical_columns if 'age' in col.lower()]
    for col in age_cols:
        try:
            if col not in df_processed.columns:
                continue
                
            invalid_age = (df_processed[col] < 0) | (df_processed[col] > 120)
            invalid_count = invalid_age.sum()
            
            if invalid_count > 0:
                df_processed[col] = df_processed[col].clip(0, 120)
                validation_metadata[col] = {
                    'invalid_values_capped': int(invalid_count),
                    'validation_type': 'age_range',
                    'min_allowed': 0,
                    'max_allowed': 120
                }
        except Exception as e:
            continue
    
    # Amount validation (no negative amounts for financial data)
    amount_cols = [col for col in numerical_columns if any(
        keyword in col.lower() for keyword in ['amount', 'cost', 'price', 'fee', 'charge', 'paid', 'billed']
    )]
    for col in amount_cols:
        try:
            if col not in df_processed.columns:
                continue
                
            negative_amount = df_processed[col] < 0
            negative_count = negative_amount.sum()
            
            if negative_count > 0:
                df_processed[col] = df_processed[col].abs()
                validation_metadata[col] = {
                    'negative_values_fixed': int(negative_count),
                    'validation_type': 'non_negative_amount'
                }
        except Exception as e:
            continue
    
    # Percentage validation (0-100 or 0-1)
    percentage_cols = [col for col in numerical_columns if any(
        keyword in col.lower() for keyword in ['percent', 'rate', 'ratio', 'pct']
    )]
    for col in percentage_cols:
        try:
            if col not in df_processed.columns:
                continue
                
            # Check if values are in 0-100 range or 0-1 range
            max_val = df_processed[col].max()
            min_val = df_processed[col].min()
            
            if max_val > 100 and min_val >= 0:
                # Likely percentage values that exceed 100
                invalid_high = df_processed[col] > 100
                invalid_count = invalid_high.sum()
                if invalid_count > 0:
                    df_processed[col] = df_processed[col].clip(0, 100)
                    validation_metadata[col] = {
                        'invalid_percentage_capped': int(invalid_count),
                        'validation_type': 'percentage_range_0_100'
                    }
            elif max_val > 1 and min_val >= 0 and max_val <= 100:
                # Could be either 0-1 or 0-100, normalize to 0-1 if it looks like percentage
                if max_val > 1:
                    df_processed[col] = df_processed[col] / 100
                    validation_metadata[col] = {
                        'normalized_to_0_1': True,
                        'validation_type': 'percentage_normalization'
                    }
        except Exception as e:
            continue
    
    # Count/quantity validation (no negative counts)
    count_cols = [col for col in numerical_columns if any(
        keyword in col.lower() for keyword in ['count', 'num', 'quantity', 'total', 'number']
    )]
    for col in count_cols:
        try:
            if col not in df_processed.columns:
                continue
                
            negative_count = df_processed[col] < 0
            negative_count_val = negative_count.sum()
            
            if negative_count_val > 0:
                df_processed[col] = df_processed[col].abs()
                validation_metadata[col] = {
                    'negative_counts_fixed': int(negative_count_val),
                    'validation_type': 'non_negative_count'
                }
        except Exception as e:
            continue
    
    validation_metadata['total_validations_performed'] = len(validation_metadata)
    
    return df_processed, validation_metadata

def pre_select_important_features(df, numerical_columns, top_k=15):
    """
    Safely selects the most important numerical features based on variance 
    and data quality before complex engineering.
    """
    if not numerical_columns:
        return []
        
    feature_stats = []
    
    for col in numerical_columns:
        try:
            # Skip ID-like columns (high cardinality but not meaningful as continuous)
            if any(id_key in col.lower() for id_key in ['id', 'key', 'code', 'identifier']):
                continue
                
            # Check for zero or near-zero variance
            std_dev = df[col].std()
            if pd.isna(std_dev) or std_dev < 1e-6:
                continue
                
            # Check for too many missing values
            missing_rate = df[col].isnull().sum() / len(df)
            if missing_rate > 0.3: # Skip if more than 30% missing
                continue
                
            # Calculate normalized variance (coefficient of variation)
            mean_val = abs(df[col].mean())
            if mean_val > 1e-6:
                score = std_dev / mean_val
            else:
                score = std_dev
                
            feature_stats.append({'column': col, 'score': score})
        except:
            continue
            
    if not feature_stats:
        # Fallback to first few columns if scoring fails
        return numerical_columns[:min(len(numerical_columns), top_k)]
        
    # Sort by score descending and take top K
    sorted_features = sorted(feature_stats, key=lambda x: x['score'], reverse=True)
    selected = [item['column'] for item in sorted_features[:top_k]]
    
    return selected

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from dataframe with metadata tracking.

    Args:
        df: Input dataframe
        subset: Column names to consider for identifying duplicates (None = all columns)
        keep: Which duplicate to keep ('first', 'last', or False to drop all)

    Returns:
        df_processed: Dataframe with duplicates removed
        duplicate_metadata: Dictionary with duplicate removal statistics
    """
    original_count = len(df)
    df_processed = df.copy()

    # Identify duplicates
    duplicates_mask = df_processed.duplicated(subset=subset, keep=keep)
    duplicate_count = duplicates_mask.sum()

    if duplicate_count > 0:
        df_processed = df_processed[~duplicates_mask].reset_index(drop=True)

    duplicate_metadata = {
        'original_rows': original_count,
        'duplicates_removed': int(duplicate_count),
        'final_rows': len(df_processed),
        'duplicate_rate': float(duplicate_count / original_count) if original_count > 0 else 0.0,
        'subset': subset,
        'keep': keep
    }

    return df_processed, duplicate_metadata

def enhanced_missing_handling(df, use_robust_imputation=True):
    """
    Enhanced missing value handling with robust statistics and indicators, optimized for big data.

    Args:
        df: Input dataframe
        use_robust_imputation: If True, use robust statistics (median) that are less sensitive to outliers

    Returns:
        df_processed: Processed dataframe
        missing_metadata: Dictionary with missing value statistics
    """
    # Work in-place to avoid unnecessary copy
    df_processed = df
    missing_metadata = {}

    total_len = len(df_processed)
    missing_counts = df_processed.isnull().sum()
    missing_rates = missing_counts / total_len

    cols_with_missing = missing_counts[missing_counts > 0].index

    # 1. Create missing indicators in bulk
    indicator_cols = [col for col in cols_with_missing if missing_rates[col] < 0.5]
    if indicator_cols:
        for col in indicator_cols:
            df_processed[f'{col}_missing'] = df_processed[col].isnull().astype(np.int8)

        for col in indicator_cols:
            missing_metadata[col] = {'missing_count': int(missing_counts[col]), 'missing_rate': float(missing_rates[col])}

    # 2. Drop columns with too many missing values
    cols_to_drop = [col for col in cols_with_missing if missing_rates[col] > 0.5]
    if cols_to_drop:
        df_processed.drop(columns=cols_to_drop, inplace=True)
        for col in cols_to_drop:
            missing_metadata[col] = {'action': 'dropped', 'reason': 'high_missing_rate'}

    # 3. Handle remaining missing values
    remaining_missing = [col for col in cols_with_missing if col not in cols_to_drop]
    
    if remaining_missing:
        # Separate numerical and categorical
        num_dtypes = ['float64', 'float32', 'int64', 'int32', 'float16', 'int16', 'int8']
        num_cols = [col for col in remaining_missing if df_processed[col].dtype in num_dtypes]
        cat_cols = [col for col in remaining_missing if col not in num_cols]
        
        # Numerical: enhanced imputation with robust statistics
        if num_cols:
            if use_robust_imputation:
                # Use median (robust to outliers) for numerical columns
                medians = df_processed[num_cols].median()
                df_processed[num_cols] = df_processed[num_cols].fillna(medians)
                for col in num_cols:
                    missing_metadata[col] = {
                        'action': 'median_imputed_robust',
                        'value': float(medians[col]) if not pd.isna(medians[col]) else 0.0
                    }
            else:
                # Use mean for numerical columns (faster but less robust)
                means = df_processed[num_cols].mean()
                df_processed[num_cols] = df_processed[num_cols].fillna(means)
                for col in num_cols:
                    missing_metadata[col] = {
                        'action': 'mean_imputed',
                        'value': float(means[col]) if not pd.isna(means[col]) else 0.0
                    }
                
        # Categorical: enhanced imputation with fallback strategies
        if cat_cols:
            for col in cat_cols:
                missing_rate = df_processed[col].isnull().sum() / len(df_processed)

                # Convert to string if categorical to avoid setitem error
                if pd.api.types.is_categorical_dtype(df_processed[col]):
                    df_processed[col] = df_processed[col].astype(str)

                if missing_rate > 0.3:
                    # High missing rate: use 'Unknown' to avoid bias
                    df_processed[col] = df_processed[col].fillna('Unknown')
                    missing_metadata[col] = {
                        'action': 'unknown_imputed',
                        'reason': 'high_missing_rate'
                    }
                else:
                    # Low missing rate: use mode
                    mode_val = df_processed[col].mode()
                    if len(mode_val) > 0:
                        df_processed[col] = df_processed[col].fillna(mode_val[0])
                        missing_metadata[col] = {
                            'action': 'mode_imputed',
                            'value': str(mode_val[0])
                        }
                    else:
                        # Fallback to 'Unknown' if mode fails
                        df_processed[col] = df_processed[col].fillna('Unknown')
                        missing_metadata[col] = {'action': 'unknown_imputed_fallback'}
                    
    missing_metadata['total_missing_columns'] = len(cols_with_missing)
    missing_metadata['columns_dropped'] = len(cols_to_drop)
    missing_metadata['robust_imputation_used'] = use_robust_imputation
                    
    return df_processed, missing_metadata

def enhanced_categorical_encoding(df, categorical_columns):
    """Enhanced categorical encoding with multiple strategies"""
    df_processed = df  # Use reference to avoid memory duplication
    encoding_metadata = {}

    low_cardinality_cols = []
    new_encoded_features = []

    for col in categorical_columns:
        try:
            # Convert to string if categorical to avoid setitem error
            if pd.api.types.is_categorical_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].astype(str)

            cardinality = df_processed[col].nunique()

            # Skip if column has too many unique values (likely IDs)
            if cardinality > len(df_processed) * 0.5:
                encoding_metadata[col] = {'strategy': 'skipped', 'reason': 'high_cardinality'}
                continue

            # Strategy selection based on cardinality
            if cardinality <= 5:
                # One-Hot Encoding in bulk later
                low_cardinality_cols.append(col)
                encoding_metadata[col] = {
                    'strategy': 'one_hot',
                    'cardinality': cardinality
                }

            elif cardinality <= 20:
                # Label Encoding + Frequency Encoding for medium cardinality
                df_processed[f'{col}_encoded'] = pd.factorize(df_processed[col])[0]

                # Frequency encoding with rare category grouping
                freq_map = df_processed[col].value_counts()
                rare_threshold = max(3, len(df_processed) * 0.01)  # 1% or minimum 3

                # Group rare categories
                freq_map_filtered = freq_map[freq_map >= rare_threshold]
                df_processed[f'{col}_freq_encoded'] = df_processed[col].map(freq_map_filtered).fillna(0)

                encoding_metadata[col] = {
                    'strategy': 'label_frequency',
                    'features': [f'{col}_encoded', f'{col}_freq_encoded'],
                    'cardinality': cardinality,
                    'rare_threshold': rare_threshold
                }

            else:
                # Advanced frequency encoding for high cardinality
                freq_map = df_processed[col].value_counts()
                
                # Create frequency bins
                freq_bins = pd.qcut(freq_map, q=10, labels=False, duplicates='drop')
                freq_binned_map = freq_map.groupby(freq_bins).mean().to_dict()
                
                df_processed[f'{col}_freq_binned'] = df_processed[col].map(freq_map).map(freq_binned_map)
                df_processed[f'{col}_freq_encoded'] = df_processed[col].map(freq_map)
                
                encoding_metadata[col] = {
                    'strategy': 'frequency_binned',
                    'features': [f'{col}_freq_binned', f'{col}_freq_encoded'],
                    'cardinality': cardinality
                }
                
        except Exception as e:
            encoding_metadata[col] = {'strategy': 'failed', 'error': str(e)}
            continue
            
    if low_cardinality_cols:
        df_processed = pd.get_dummies(df_processed, columns=low_cardinality_cols, drop_first=True)
        # Update features in metadata
        for col in low_cardinality_cols:
            encoded_features = [c for c in df_processed.columns if c.startswith(f"{col}_")]
            encoding_metadata[col]['features'] = encoded_features
    
    return df_processed, encoding_metadata

def preprocess_insurance_claims_optimized(df, enable_large_file_handling=True, enable_outlier_detection=True, enable_data_validation=True):
    """
    Enhanced preprocessing with large file support - adapts to any dataset structure
    
    Args:
        df: Input dataframe
        enable_large_file_handling: Enable parallel processing for large datasets
        enable_outlier_detection: Enable outlier detection and handling
        enable_data_validation: Enable data range validation
    
    Returns:
        df_processed: Processed dataframe
        final_features: List of final features for modeling
        preprocessing_metadata: Dictionary with preprocessing statistics
    """
    # Check if dataset is large
    if enable_large_file_handling and len(df) > 100000:  # 100k+ rows
        return preprocess_large_dataset(df)
    
    # Original preprocessing for smaller datasets
    # Work in-place to avoid unnecessary copy
    df_processed = df
    
    # Store original columns for reference
    original_columns = df.columns.tolist()
    
    # 1. ENHANCED MISSING VALUE HANDLING (with robust statistics)
    memory_mb, is_critical, should_continue = check_memory_threshold(warning_threshold_mb=2000)
    if not should_continue:
        raise MemoryError("Memory threshold exceeded during missing value handling")
    df_processed, missing_metadata = enhanced_missing_handling(df_processed, use_robust_imputation=True)

    # 1.5 DATA VALIDATION - Validate logical ranges before feature engineering
    memory_mb, is_critical, should_continue = check_memory_threshold(warning_threshold_mb=2500)
    if not should_continue:
        raise MemoryError("Memory threshold exceeded during data validation")
    if enable_data_validation:
        numerical_columns_for_validation = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        df_processed, validation_metadata = validate_data_ranges(df_processed, numerical_columns_for_validation)
    else:
        validation_metadata = {}

    # 1.6 OUTLIER DETECTION AND HANDLING - Detect and handle outliers
    memory_mb, is_critical, should_continue = check_memory_threshold(warning_threshold_mb=3000)
    if not should_continue:
        raise MemoryError("Memory threshold exceeded during outlier detection")
    if enable_outlier_detection:
        numerical_columns_for_outliers = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        # Use IQR method with capping (safer than removal)
        df_processed, outlier_metadata = detect_and_handle_outliers(
            df_processed,
            numerical_columns_for_outliers,
            method='iqr',
            action='cap'
        )
    else:
        outlier_metadata = {}

    # 2. DATE PROCESSING - Flexible date detection and processing
    memory_mb, is_critical, should_continue = check_memory_threshold(warning_threshold_mb=3500)
    if not should_continue:
        raise MemoryError("Memory threshold exceeded during date processing")
    date_columns = [col for col in df_processed.columns if any(
        keyword in col.lower() for keyword in ['date', 'time', 'created', 'submitted']
    )]
    
    new_date_features = []
    for col in date_columns:
        try:
            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            # Extract temporal features
            df_processed[f'{col}_day_of_week'] = df_processed[col].dt.dayofweek
            df_processed[f'{col}_month'] = df_processed[col].dt.month
            df_processed[f'{col}_year'] = df_processed[col].dt.year
            df_processed[f'{col}_day'] = df_processed[col].dt.day
            df_processed[f'{col}_quarter'] = df_processed[col].dt.quarter
        except:
            continue
    
    # 3. ENHANCED CATEGORICAL ENCODING
    categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Exclude date columns from categorical processing
    categorical_columns = [col for col in categorical_columns if col not in date_columns]
    
    if len(categorical_columns) > 0:
        if len(df) > 50000:  # Use optimized encoding for larger datasets
            df_processed, encoding_metadata = optimize_categorical_encoding_large(df_processed, categorical_columns)
        else:
            df_processed, encoding_metadata = enhanced_categorical_encoding(df_processed, categorical_columns)
    else:
        encoding_metadata = {}
    
    # 4. NUMERICAL FEATURE ENGINEERING - Automatic detection and enhancement
    numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove encoded columns from original numerical list to avoid duplication
    original_numerical = [col for col in numerical_columns if not col.endswith('_encoded')]
    
    # --- PRE-SELECTION STEP ---
    # Identify top informative features before complex engineering to avoid explosion
    important_numerical = pre_select_important_features(df_processed, original_numerical, top_k=15)
    st.info(f"🎯 Seleksi fitur: Menggunakan {len(important_numerical)} fitur utama untuk rekayasa fitur kompleks.")
    
    # Apply smart data type optimization
    df_processed = smart_data_type_optimization(df_processed)
    
    # 4.1 Ratio features for amount/financial columns - OPTIMIZED WITH POLARS
    amount_columns = [col for col in important_numerical if any(
        keyword in col.lower() for keyword in ['amount', 'cost', 'price', 'fee', 'charge']
    )]
    
    if amount_columns:
        # Limit ratio calculations to avoid O(n²) complexity
        # Only create ratios for top 5 amount columns to prevent feature explosion
        amount_columns_limited = amount_columns[:5]
        
        # Convert to Polars for parallel processing
        pl_df = pl.from_pandas(df_processed[amount_columns_limited])
        ratio_exprs = []
        
        # Create ratios between amount columns - limited combinations
        for i, col1 in enumerate(amount_columns_limited):
            # Only create ratios with the next 2 columns instead of all remaining columns
            for col2 in amount_columns_limited[i+1:i+3]:
                ratio_col = f'{col1}_to_{col2}_ratio'
                # Express ratio calculation
                ratio_exprs.append(
                    (pl.col(col1) / (pl.col(col2) + 1e-8)).alias(ratio_col)
                )
        
        if ratio_exprs:
            ratios_pl = pl_df.select(ratio_exprs)
            # Handle inf/nan and convert back to pandas
            ratios_df = ratios_pl.to_pandas()
            ratios_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Fillna in bulk
            medians = ratios_df.median()
            ratios_df.fillna(medians, inplace=True)
            ratios_df.fillna(0, inplace=True) # Fallback if median is NaN
            
            for c in ratios_df.columns:
                df_processed[c] = ratios_df[c]
    
    # 4.2 Age-related features
    age_columns = [col for col in original_numerical if any(
        keyword in col.lower() for keyword in ['age', 'years', 'old']
    )]
    
    new_age_features = []
    for col in age_columns:
        try:
            # Age groups
            group = pd.cut(df_processed[col], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
            df_processed[f'{col}_group_encoded'] = pd.factorize(group)[0]
            
            # Age squared for non-linear relationships
            df_processed[f'{col}_squared'] = df_processed[col] ** 2
        except:
            continue
    
    # 4.3 Count/quantity features
    count_columns = [col for col in original_numerical if any(
        keyword in col.lower() for keyword in ['count', 'num', 'quantity', 'total', 'number']
    )]
    
    new_count_features = []
    for col in count_columns:
        try:
            # Binary indicators for high values
            threshold = df_processed[col].quantile(0.75)
            df_processed[f'{col}_high'] = (df_processed[col] > threshold).astype(np.int8)
            
            # Log transform for skewed distributions
            if df_processed[col].min() >= 0:
                df_processed[f'{col}_log'] = np.log1p(df_processed[col])
        except:
            continue
    
    # 4.4 Time/duration features
    time_columns = [col for col in original_numerical if any(
        keyword in col.lower() for keyword in ['days', 'duration', 'time', 'period']
    )]
    
    # 4.5 Domain-specific insurance claim features
    try:
        # Claim amount ratio features (if multiple amount columns exist)
        if len(amount_columns) > 1:
            billed_col = next((col for col in amount_columns if 'billed' in col.lower()), None)
            paid_col = next((col for col in amount_columns if 'paid' in col.lower()), None)
            allowed_col = next((col for col in amount_columns if 'allowed' in col.lower()), None)
            
            if billed_col and paid_col and billed_col in df_processed.columns and paid_col in df_processed.columns:
                # Payment ratio (how much was actually paid vs billed)
                df_processed['payment_ratio'] = np.where(
                    df_processed[billed_col] > 0,
                    df_processed[paid_col] / df_processed[billed_col],
                    0
                )
                # Cap at 1.0 for cases where paid > billed
                df_processed['payment_ratio'] = df_processed['payment_ratio'].clip(0, 1.0)
                
            if billed_col and allowed_col and billed_col in df_processed.columns and allowed_col in df_processed.columns:
                # Allowance ratio (how much was allowed vs billed)
                df_processed['allowance_ratio'] = np.where(
                    df_processed[billed_col] > 0,
                    df_processed[allowed_col] / df_processed[billed_col],
                    0
                )
                df_processed['allowance_ratio'] = df_processed['allowance_ratio'].clip(0, 1.0)
                
        # Service frequency features (if count columns exist)
        if len(count_columns) > 0:
            for col in count_columns:
                if 'service' in col.lower() or 'num' in col.lower():
                    # High service count indicator
                    threshold = df_processed[col].quantile(0.9)
                    df_processed[f'{col}_very_high'] = (df_processed[col] > threshold).astype(np.int8)
                    
        # Risk scoring features based on combinations
        if len(amount_columns) > 0 and len(time_columns) > 0:
            # High amount + quick submission combination (potential fraud indicator)
            amount_col = amount_columns[0]
            time_col = time_columns[0]
            if amount_col in df_processed.columns and time_col in df_processed.columns:
                high_amount = df_processed[amount_col] > df_processed[amount_col].quantile(0.75)
                quick_submission = df_processed[time_col] < df_processed[time_col].quantile(0.25)
                df_processed['high_amount_quick_submit'] = (high_amount & quick_submission).astype(np.int8)
                
    except Exception as e:
        print(f"Error in domain-specific feature engineering: {e}")
    
    # 4.4 Time/duration features (continued)
    new_time_features = []
    for col in time_columns:
        try:
            # Binary indicators for unusual time periods
            if 'days' in col.lower() or 'duration' in col.lower():
                df_processed[f'{col}_late'] = (df_processed[col] > df_processed[col].quantile(0.9)).astype(np.int8)
                df_processed[f'{col}_quick'] = (df_processed[col] < df_processed[col].quantile(0.1)).astype(np.int8)
        except:
            continue
    
    # 5. STATISTICAL FEATURES - Rolling and aggregation features
    if len(df_processed) > 10:
        try:
            # Create features based on statistical properties
            valid_cols = [col for col in important_numerical if df_processed[col].std() > 0]
            if valid_cols:
                # Z-score and Percentile rank iteratively to save memory
                for valid_col in valid_cols:
                    mean_val = df_processed[valid_col].mean()
                    std_val = df_processed[valid_col].std()
                    if std_val > 0:
                        df_processed[f'{valid_col}_zscore'] = (df_processed[valid_col] - mean_val) / std_val
                    
                    df_processed[f'{valid_col}_pct_rank'] = df_processed[valid_col].rank(pct=True)
        except Exception as e:
            print(f"Error in statistical features: {e}")
    
    # 6. INTERACTION FEATURES - OPTIMIZED WITH POLARS
    try:
        # Use important features identified during pre-selection
        # Further limit to top 5 features to prevent feature explosion
        top_numerical = important_numerical[:5]
        
        if len(top_numerical) > 1:
            pl_df = pl.from_pandas(df_processed[top_numerical])
            interaction_exprs = []
            
            # Only create interactions between adjacent features (linear complexity instead of quadratic)
            for i in range(len(top_numerical) - 1):
                col1 = top_numerical[i]
                col2 = top_numerical[i + 1]
                interaction_col = f'{col1}_x_{col2}'
                interaction_exprs.append(
                    (pl.col(col1) * pl.col(col2)).alias(interaction_col)
                )
            
            if interaction_exprs:
                interactions_pl = pl_df.select(interaction_exprs)
                interactions_df = interactions_pl.to_pandas()
                for c in interactions_df.columns:
                    df_processed[c] = interactions_df[c]
    except Exception as e:
        print(f"Interaction features optimization error: {e}")
    
    # 7. FINAL FEATURE SELECTION - Get all numerical features
    all_numerical = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Pre-calculate missing rates and stds in bulk for faster filtering
    missing_rates = df_processed[all_numerical].isnull().sum() / len(df_processed)
    stds = df_processed[all_numerical].std()
    
    # Remove ID columns and original categorical columns
    id_keywords = ['id', 'identifier', 'key', 'code']
    final_features = []
    
    for col in all_numerical:
        # Skip if it's an ID column
        if any(keyword in col.lower() for keyword in id_keywords):
            if not col.endswith('_encoded'):  # Keep encoded IDs
                continue
        
        # Skip if it has too many missing values
        if missing_rates[col] > 0.5:
            continue
        
        # Skip if it has zero or near-zero variance
        if stds[col] < 1e-6:
            continue
        
        final_features.append(col)
    
    # 8. DATA QUALITY CHECKS
    # Fill remaining missing values in bulk
    cols_to_fill = [col for col in final_features if missing_rates[col] > 0]
    if cols_to_fill:
        medians = df_processed[cols_to_fill].median()
        df_processed[cols_to_fill] = df_processed[cols_to_fill].fillna(medians)
    
    # FINAL CHECK: Ensure NO NaNs in final features
    # Check for any remaining NaNs and fill them with 0 or median
    for col in final_features:
        if df_processed[col].isnull().any():
            # Fill with median (if possible) or 0
            median_val = df_processed[col].median()
            if pd.isna(median_val):
                df_processed[col] = df_processed[col].fillna(0)
            else:
                df_processed[col] = df_processed[col].fillna(median_val)
    
    # Store metadata for transparency
    # Hitung statistik untuk metadata
    date_columns_count = len(date_columns)
    categorical_columns_count = len(categorical_columns)
    numerical_columns_count = len(original_numerical)
    
    preprocessing_metadata = {
        'original_columns': original_columns,
        'original_columns_count': len(original_columns),
        'final_features_count': len(final_features),
        'final_features': final_features,
        'dataset_size': len(df),
        'date_columns_count': date_columns_count,
        'categorical_columns_count': categorical_columns_count,
        'numerical_columns_count': numerical_columns_count,
        'enhanced_encoding_metadata': encoding_metadata,
        'missing_value_metadata': missing_metadata,
        'outlier_metadata': outlier_metadata,
        'validation_metadata': validation_metadata,
        'processing_method': 'optimized_standard_with_validation',
        'outlier_detection_enabled': enable_outlier_detection,
        'data_validation_enabled': enable_data_validation
    }
    
    return df_processed, final_features, preprocessing_metadata

@st.cache_data(ttl=1800, max_entries=10)
def apply_mutual_info_selection(df, feature_columns, target_col=None, k=20, sample_size=10000):
    """
    Applies Mutual Information selection. Scalable for big data via sampling.
    Cached for 30 minutes to avoid recomputation.
    """
    # Handle lazy loading dict response
    if isinstance(df, dict):
        if df.get('lazy'):
            df = pd.read_parquet(df['path'])
        else:
            df = df

    # Sample data if too large for performance
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    # Fill missing values with median (better than 0 for numerical features)
    X = df_sample[feature_columns].copy()
    medians = X.median()
    X = X.fillna(medians).fillna(0) # Fallback to 0 if median is NaN

    # If no target provided, use Isolation Forest to find outliers as pseudo-labels
    if target_col is None or target_col not in df.columns:
        iso = IsolationForest(contamination=0.1, random_state=42)
        y = iso.fit_predict(X)
        y = (y == -1).astype(int)
    else:
        y = df_sample[target_col].fillna(df_sample[target_col].median() if pd.api.types.is_numeric_dtype(df_sample[target_col]) else df_sample[target_col].mode()[0])

    # Calculate MI
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(feature_columns)))
    selector.fit(X, y)

    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_columns[i] for i in selected_indices]

    # Get scores for visualization
    scores = selector.scores_
    feature_scores = pd.DataFrame({'Feature': feature_columns, 'Score': scores}).sort_values(by='Score', ascending=False)

    return selected_features, feature_scores

@st.cache_data(ttl=1800, max_entries=10)
def apply_tree_based_selection(df, feature_columns, target_col=None, k=20, sample_size=20000):
    """
    Applies Tree-based (XGBoost/LGBM/RF) selection.
    Cached for 30 minutes to avoid recomputation.
    """
    # Handle lazy loading dict response
    if isinstance(df, dict):
        if df.get('lazy'):
            df = pd.read_parquet(df['path'])
        else:
            df = df

    # Sample data
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
        
    # Fill missing values with median (better than 0 for numerical features)
    X = df_sample[feature_columns].copy()
    medians = X.median()
    X = X.fillna(medians).fillna(0) # Fallback to 0 if median is NaN
    
    if target_col is None or target_col not in df.columns:
        iso = IsolationForest(contamination=0.1, random_state=42)
        y = iso.fit_predict(X)
        y = (y == -1).astype(int)
    else:
        y = df_sample[target_col].fillna(df_sample[target_col].median() if pd.api.types.is_numeric_dtype(df_sample[target_col]) else df_sample[target_col].mode()[0])
    
    # Prefer LightGBM for speed on big data, then XGBoost, then RF
    model_name = "RandomForest"
    if lgb:
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        model_name = "LightGBM"
    elif xgb:
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model_name = "XGBoost"
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    model.fit(X, y)
    
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'Feature': feature_columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    selected_features = feature_importances.head(k)['Feature'].tolist()
    
    return selected_features, feature_importances, model_name

@st.cache_data(ttl=1800, max_entries=10)
def apply_pca_reduction(df, feature_columns, n_components=0.95):
    """
    Applies PCA reduction. Returns a DF with principal components.
    Cached for 30 minutes to avoid recomputation.
    """
    # Handle lazy loading dict response
    if isinstance(df, dict):
        if df.get('lazy'):
            df = pd.read_parquet(df['path'])
        else:
            df = df

    from sklearn.preprocessing import StandardScaler

    # Fill missing values with median
    X = df[feature_columns].copy()
    medians = X.median()
    X = X.fillna(medians).fillna(0) # Fallback to 0 if median is NaN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_cols = [f'PCA_Component_{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    
    # Variance explained info
    explained_variance = pca.explained_variance_ratio_
    
    return df_pca, pca_cols, explained_variance