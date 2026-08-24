import pandas as pd
import numpy as np
import streamlit as st
import gc
from config import LARGE_DATASET_CONFIG
from joblib import Parallel, delayed

def process_single_chunk(chunk, all_columns=None, enable_outlier_detection=True, enable_data_validation=True):
    """Worker function for parallel chunk processing"""
    from preprocessing_optimized import preprocess_insurance_claims_optimized
    # Convert view to copy for processing (preprocessing functions need modifiable dataframes)
    chunk_copy = chunk.copy()
    processed_chunk, _, _ = preprocess_insurance_claims_optimized(
        chunk_copy, 
        enable_large_file_handling=False,
        enable_outlier_detection=enable_outlier_detection,
        enable_data_validation=enable_data_validation
    )
    
    if all_columns:
        processed_chunk = processed_chunk.reindex(columns=all_columns)
    
    return smart_data_type_optimization(processed_chunk)

def preprocess_large_dataset(df, chunk_size=50000, progress_bar=True, enable_outlier_detection=True, enable_data_validation=True):
    """
    Optimized preprocessing for large datasets with parallel execution support
    
    Args:
        df: Input dataframe
        chunk_size: Size of chunks for parallel processing
        progress_bar: Show progress bar
        enable_outlier_detection: Enable outlier detection and handling
        enable_data_validation: Enable data range validation
    """
    
    if len(df) <= chunk_size:
        from preprocessing_optimized import preprocess_insurance_claims_optimized
        return preprocess_insurance_claims_optimized(
            df, 
            enable_large_file_handling=False,
            enable_outlier_detection=enable_outlier_detection,
            enable_data_validation=enable_data_validation
        )
    
    total_rows = len(df)
    st.info(f"🔄 Dataset besar terdeteksi ({total_rows:,} baris). Memproses secara paralel...")
    
    # Prepare chunks - use views instead of copies to reduce memory overhead
    chunks = [df.iloc[i:min(i + chunk_size, total_rows)] for i in range(0, total_rows, chunk_size)]
    total_chunks = len(chunks)
    
    progress_bar_st = None
    status_text = None
    if progress_bar:
        progress_bar_st = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Memproses {total_chunks} chunk secara paralel...")
    
    try:
        # Use joblib for parallel processing with multiprocessing for CPU-bound tasks
        # Changed from threading to multiprocessing for better CPU utilization
        import os
        cpu_count = os.cpu_count() or 2
        optimal_jobs = min(2, max(1, cpu_count // 2))
        processed_chunks = Parallel(n_jobs=optimal_jobs, backend="threading")(
            delayed(process_single_chunk)(chunk, None, enable_outlier_detection, enable_data_validation) for chunk in chunks
        )
        
        if progress_bar:
            progress_bar_st.progress(1.0)
            status_text.text("Merging processed chunks...")

        # Clear original chunks to free memory
        del chunks
        gc.collect()

        # Get all columns from all chunks
        all_cols = set()
        for chunk in processed_chunks:
            all_cols.update(chunk.columns.tolist())
        all_cols = sorted(list(all_cols))

        # Reindex chunks to match columns before concat
        for i in range(len(processed_chunks)):
            processed_chunks[i] = processed_chunks[i].reindex(columns=all_cols)
            
        # Detect numerical columns
        all_numerical = processed_chunks[0].select_dtypes(include=[np.number]).columns.tolist()
        
        # Merge
        df_processed = pd.concat(processed_chunks, ignore_index=True)
        
        # Clean up
        chunks_count = len(processed_chunks)
        processed_chunks.clear()
        del processed_chunks
        gc.collect()

        id_keywords = ['id', 'identifier', 'key', 'code']
        final_features = []
        for col in all_numerical:
            if any(keyword in col.lower() for keyword in id_keywords):
                if not col.endswith('_encoded'):
                    continue
            if df_processed[col].isnull().sum() / len(df_processed) > 0.5:
                continue
            if df_processed[col].std() < 1e-6:
                continue
            final_features.append(col)
        
        for col in final_features:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)

        preprocessing_metadata = {
            'original_columns': df.columns.tolist(),
            'original_columns_count': len(df.columns),
            'date_columns_found': [],
            'date_columns_count': 0,
            'categorical_columns_found': [],
            'categorical_columns_count': 0,
            'numerical_columns_found': final_features,
            'numerical_columns_count': len(final_features),
            'final_features_count': len(final_features),
            'final_features': final_features,
            'enhanced_encoding_metadata': {},
            'missing_value_metadata': {},
            'outlier_metadata': {},
            'validation_metadata': {},
            'encoding_strategies_used': [],
            'total_rows_processed': total_rows,
            'chunks_processed': chunks_count,
            'processing_method': 'parallel_chunked_optimization',
            'outlier_detection_enabled': enable_outlier_detection,
            'data_validation_enabled': enable_data_validation
        }
        
        st.success(f"✅ Dataset besar berhasil diproses secara paralel ({total_rows:,} baris)")
        
        return df_processed, final_features, preprocessing_metadata

    except Exception as e:
        st.error(f"Error processing large dataset: {str(e)}")
        return None, None, None
    
    finally:
        if progress_bar_st is not None:
            progress_bar_st.empty()
        if status_text is not None:
            status_text.empty()

def optimize_categorical_encoding_large(df, categorical_columns, sample_size=10000):
    """
    Optimized categorical encoding for large datasets using sampling
    """
    encoding_metadata = {}
    low_cardinality_cols = []
    new_features = []
    
    for col in categorical_columns:
        try:
            # Use sample for cardinality analysis
            sample_data = df[col].sample(min(sample_size, len(df)))
            cardinality = sample_data.nunique()
            
            if cardinality <= 5:
                # One-Hot Encoding for low cardinality
                low_cardinality_cols.append(col)
                encoding_metadata[col] = {
                    'strategy': 'one_hot',
                    'cardinality': cardinality
                }
                
            elif cardinality <= 50:
                # Label Encoding + Frequency Encoding
                new_features.append(pd.Series(pd.factorize(df[col])[0], name=f'{col}_encoded', index=df.index))
                
                # Sample-based frequency encoding
                freq_map = sample_data.value_counts()
                new_features.append(df[col].map(freq_map).fillna(0).rename(f'{col}_freq_encoded'))
                
                encoding_metadata[col] = {
                    'strategy': 'label_frequency',
                    'features': [f'{col}_encoded', f'{col}_freq_encoded'],
                    'cardinality': cardinality
                }
                
            else:
                # Frequency encoding only for high cardinality
                freq_map = sample_data.value_counts()
                new_features.append(df[col].map(freq_map).fillna(0).rename(f'{col}_freq_encoded'))
                
                encoding_metadata[col] = {
                    'strategy': 'frequency_only',
                    'features': [f'{col}_freq_encoded'],
                    'cardinality': cardinality
                }
                
        except Exception as e:
            encoding_metadata[col] = {'strategy': 'failed', 'error': str(e)}
            continue
            
    if new_features:
        df = pd.concat([df] + new_features, axis=1)
        
    if low_cardinality_cols:
        df = pd.get_dummies(df, columns=low_cardinality_cols, drop_first=True)
        for col in low_cardinality_cols:
            encoded_features = [c for c in df.columns if c.startswith(f"{col}_")]
            encoding_metadata[col]['features'] = encoded_features
    
    return df, encoding_metadata

def smart_data_type_optimization(df, memory_threshold=0.8):
    """
    Smart data type optimization based on memory usage
    """
    # Check current memory usage - use shallow check first for performance
    current_memory_shallow = df.memory_usage(deep=False).sum() / (1024**3)  # GB
    
    if current_memory_shallow < 0.5:  # Less than 500MB
        return df  # No optimization needed
    
    # Only do deep check if shallow check indicates high memory usage
    current_memory = df.memory_usage(deep=True).sum() / (1024**3)  # GB
    
    if current_memory < 0.5:  # Less than 500MB
        return df  # No optimization needed
    
    st.info(f"🧠 Optimasi memory (penggunaan: {current_memory:.2f}GB)")
    
    # Optimize based on memory pressure
    if current_memory > memory_threshold:
        # Aggressive optimization for high memory usage
        for col in df.select_dtypes(include=['int64']).columns:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
        
        # Convert float64 to float32
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
    else:
        # Conservative optimization
        for col in df.select_dtypes(include=['int64']).columns:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        # Only convert to float32 if precision allows
        for col in df.select_dtypes(include=['float64']).columns:
            if df[col].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0).max() <= 6:
                df[col] = df[col].astype(np.float32)
    
    # Optimize categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.1:  # Less than 10% unique
            # Don't convert to category if there are missing values to avoid setitem error
            if df[col].isnull().sum() == 0:
                df[col] = df[col].astype('category')
    
    optimized_memory = df.memory_usage(deep=True).sum() / (1024**3)
    memory_saved = current_memory - optimized_memory
    
    if memory_saved > 0.1:  # Saved more than 100MB
        st.success(f"💾 Memory dioptimalkan: {memory_saved:.2f}GB berhasil dihemat")
    
    return df