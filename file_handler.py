import pandas as pd
import numpy as np
import streamlit as st
import polars as pl
from config import *
import gc
import os
from tqdm import tqdm
import tempfile
import shutil
import tempfile
import uuid


def stream_csv_to_parquet(file_path, output_path=None, chunk_size=50000, progress_bar=True):
    """Write a CSV to Parquet in bounded-memory batches.

    This is the storage boundary for large-file workflows. It deliberately
    returns a path and row count instead of a full DataFrame.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    file_size = os.path.getsize(file_path)
    check_file_size(file_size)
    if output_path is None:
        output_path = os.path.join(
            TEMP_DATA_DIR,
            f"raw_{uuid.uuid4().hex}.parquet",
        )

    writer = None
    total_rows = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk = optimize_memory_usage(chunk)
            chunk = fix_arrow_compatibility(chunk)
            for column in chunk.select_dtypes(include=["category"]).columns:
                chunk[column] = chunk[column].astype(str)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(chunk)
            del table, chunk
            gc.collect()
    except Exception:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise
    finally:
        if writer is not None:
            writer.close()

    return output_path, total_rows

def read_with_polars(file_path, file_type='csv'):
    """
    Read file using Polars for high-performance big data handling
    """
    try:
        if file_type == 'csv':
            df_pl = pl.read_csv(file_path, rechunk=False)
        elif file_type == 'parquet':
            df_pl = pl.read_parquet(file_path)
        elif file_type == 'json':
            df_pl = pl.read_json(file_path)
        else:
            return None
            
        df_pd = df_pl.to_pandas()
        del df_pl
        gc.collect()
        return df_pd
    except Exception as e:
        st.warning(f"Polars reading failed, falling back to Pandas: {str(e)}")
        return None

def read_large_csv(file_path, chunk_size=None, progress_bar=True):
    """
    Read large CSV files with chunking and memory optimization
    """
    # Get file size
    file_size = os.path.getsize(file_path)
    check_file_size(file_size)

    # Determine chunk size
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(file_size)

    if chunk_size is None or file_size < 50 * 1024 * 1024:  # < 50MB
        # Read whole file
        df = pd.read_csv(file_path)
        return optimize_memory_usage(df)

    # Materialize through a temporary Parquet file instead of retaining every
    # input chunk until a final pd.concat operation.
    parquet_path, _ = stream_csv_to_parquet(
        file_path,
        chunk_size=chunk_size,
        progress_bar=progress_bar,
    )
    try:
        df = pd.read_parquet(parquet_path)
        return optimize_memory_usage(df)
    finally:
        if os.path.exists(parquet_path):
            os.unlink(parquet_path)
        gc.collect()

def fix_arrow_compatibility(df):
    """
    Fix Arrow serialization errors without converting optimized int32/float32 back to 64-bit.
    PyArrow natively supports int8, int16, int32, int64, float32, float64, and categorical dtypes.
    Only convert raw object columns to string.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('str')
    return df

def read_file_with_optimization(uploaded_file, file_type='csv'):
    """
    Read uploaded file with streaming buffer and memory optimization for large files
    """
    # Check file size
    file_size = uploaded_file.size
    check_file_size(file_size)

    if file_size > 100 * 1024 * 1024 and file_type in ['xlsx', 'xls']:
        st.warning("⚠️ Format Excel sangat lambat dan tidak efisien untuk file >100MB. Sangat disarankan menggunakan Parquet atau CSV.")

    # Stream upload buffer to disk in 8MB chunks to prevent memory explosion
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, tmp_file, length=8 * 1024 * 1024)
        tmp_file_path = tmp_file.name

    try:
        if file_type == 'csv':
            df = read_large_csv(tmp_file_path, progress_bar=True)
        elif file_type == 'parquet':
            st.info("🚀 Menggunakan Polars engine untuk Parquet...")
            df = read_with_polars(tmp_file_path, 'parquet')
            if df is None:
                df = pd.read_parquet(tmp_file_path)
            df = optimize_memory_usage(df)
        elif file_type in ['xlsx', 'xls']:
            excel_errors = ['#NULL!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#NUM!', '#N/A', '#N/A!']
            errors_to_replace = ['']
            for error in excel_errors:
                errors_to_replace.extend([
                    error, error.lower(), error.upper(),
                    error + ' ', ' ' + error
                ])
            df = pd.read_excel(tmp_file_path,
                             na_values=errors_to_replace,
                             keep_default_na=True)
            df = optimize_memory_usage(df)
        elif file_type == 'json':
            df = pd.read_json(tmp_file_path)
            df = optimize_memory_usage(df)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Fix Arrow compatibility without expanding memory
        df = fix_arrow_compatibility(df)

        return df

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

def process_dataframe_in_chunks(df, processing_func, chunk_size=10000, progress_bar=True):
    """
    Process DataFrame in chunks to handle large datasets
    """
    if len(df) <= chunk_size:
        return processing_func(df)
    
    chunks = []
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    progress_bar_st = None
    status_text = None
    if progress_bar and st is not None:
        progress_bar_st = st.progress(0)
        status_text = st.empty()
    
    try:
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()
            
            # Process chunk
            processed_chunk = processing_func(chunk)
            chunks.append(processed_chunk)
            
            if progress_bar and st is not None and progress_bar_st is not None:
                progress = (i + 1) / total_chunks
                progress_bar_st.progress(progress)
                if status_text is not None:
                    status_text.text(f"Processing chunk {i + 1}/{total_chunks}")
            
            # Force garbage collection
            if i % 5 == 0:
                gc.collect()
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None
    
    finally:
        if progress_bar_st is not None:
            progress_bar_st.empty()
        if status_text is not None:
            status_text.empty()
    
    # Combine processed chunks
    result = pd.concat(chunks, ignore_index=True)
    
    # Clear intermediate data
    del chunks
    gc.collect()
    
    return result

def get_file_info(uploaded_file):
    """
    Get file information and size analysis
    """
    file_size = uploaded_file.size
    file_size_mb = file_size / (1024 * 1024)
    file_size_gb = file_size / (1024 * 1024 * 1024)
    
    info = {
        'size_bytes': file_size,
        'size_mb': file_size_mb,
        'size_gb': file_size_gb,
        'size_category': 'small' if file_size_mb < 100 else 'medium' if file_size_mb < 500 else 'large',
        'recommended_chunk_size': get_optimal_chunk_size(file_size)
    }
    
    return info

def optimize_dataframe_memory(df):
    """
    Comprehensive DataFrame memory optimization
    """
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
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
    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            # Don't convert to category if there are missing values to avoid setitem error
            if df[col].isnull().sum() == 0:
                df[col] = df[col].astype('category')
    
    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    memory_saved = original_memory - optimized_memory
    memory_saved_percent = (memory_saved / original_memory) * 100
    
    return df, {
        'original_memory_mb': original_memory,
        'optimized_memory_mb': optimized_memory,
        'memory_saved_mb': memory_saved,
        'memory_saved_percent': memory_saved_percent
    }

def ingest_file_to_raw_parquet(uploaded_file, file_type='csv'):
    """
    Stream uploaded file directly to a raw Parquet file on disk without loading into RAM.
    Returns (raw_parquet_path, total_rows, schema_dict)
    """
    file_size = uploaded_file.size
    check_file_size(file_size)
    
    unique_id = uuid.uuid4().hex
    raw_parquet_path = os.path.join(TEMP_DATA_DIR, f"raw_{unique_id}.parquet")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, tmp_file, length=8 * 1024 * 1024)
        tmp_file_path = tmp_file.name

    try:
        if file_type == 'csv':
            stream_csv_to_parquet(tmp_file_path, output_path=raw_parquet_path, progress_bar=False)
        elif file_type == 'parquet':
            shutil.copyfile(tmp_file_path, raw_parquet_path)
        else:
            # Fallback for Excel / JSON
            df = read_file_with_optimization(uploaded_file, file_type)
            df = fix_arrow_compatibility(df)
            df.to_parquet(raw_parquet_path, index=False, compression="zstd")
            del df
            gc.collect()

        # Extract schema and row count using Polars Lazy scan (zero-copy / low memory)
        lf = pl.scan_parquet(raw_parquet_path)
        total_rows = lf.select(pl.len()).collect().item()
        schema = lf.collect_schema()
        
        schema_dict = {
            'columns': list(schema.names()),
            'dtypes': {name: str(dtype) for name, dtype in schema.items()},
            'total_rows': total_rows
        }
        return raw_parquet_path, total_rows, schema_dict
    finally:
        if os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

def get_parquet_sample(parquet_path: str, n: int = 5000) -> pd.DataFrame:
    """
    Read a representative head sample from a Parquet file without loading the entire dataset.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    return pl.scan_parquet(parquet_path).head(n).collect().to_pandas()

def show_file_size_warning(file_size_gb):
    """
    Show warning for large files
    """
    if file_size_gb > 1.0:
        st.warning(f"⚠️ File besar terdeteksi ({file_size_gb:.2f}GB). Proses akan membutuhkan waktu lebih lama.")
        st.info("💡 Tips: File akan diproses secara bertahap untuk mengoptimalkan penggunaan memory.")
    elif file_size_gb > 0.5:
        st.info(f"📊 File medium ({file_size_gb:.2f}GB) akan diproses dengan optimasi memory.")

def save_processed_data(df, prefix="processed"):
    """
    Save processed DataFrame or file path to Parquet file in TEMP_DATA_DIR
    Returns the file path
    """
    if isinstance(df, str) and os.path.exists(df):
        return df

    # Apply Arrow compatibility fix before saving
    df_copy = fix_arrow_compatibility(df.copy())

    unique_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DATA_DIR, f"{prefix}_{unique_id}.parquet")
    df_copy.to_parquet(file_path, index=False, compression="snappy")
    return file_path

def load_processed_data(file_path, lazy_threshold_mb=50):
    """
    Load processed DataFrame from Parquet file with lazy loading option for large files
    Reduced threshold from 100MB to 50MB for better performance
    """
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB

    if file_size > lazy_threshold_mb:
        # For large files, return metadata first and load on demand
        try:
            # Load only metadata (column names and dtypes)
            df_meta = pd.read_parquet(file_path, columns=[])
            return {
                'path': file_path,
                'metadata': df_meta,
                'lazy': True,
                'size_mb': file_size,
                'message': f"Dataset besar ({file_size:.1f}MB). Data akan dimuat sesuai kebutuhan."
            }
        except Exception as e:
            st.warning(f"Lazy loading gagal, memuat penuh: {e}")
            return pd.read_parquet(file_path)

    return pd.read_parquet(file_path)

def cleanup_temp_data(keep_files=None):
    """
    Clean up old temporary files (older than 1 day)
    """
    import time
    now = time.time()
    one_day_ago = now - (24 * 60 * 60)
    
    keep_files = keep_files or []
    
    for filename in os.listdir(TEMP_DATA_DIR):
        file_path = os.path.join(TEMP_DATA_DIR, filename)
        try:
            if os.path.isfile(file_path) and file_path not in keep_files:
                file_time = os.path.getmtime(file_path)
                if file_time < one_day_ago:
                    os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up {file_path}: {e}")