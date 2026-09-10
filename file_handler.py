import pandas as pd
import numpy as np
import streamlit as st
import polars as pl
from config import *
import gc
import os
import csv as _csv
import gzip
import logging
from tqdm import tqdm
import tempfile
import shutil
import uuid

logger = logging.getLogger("astina.file_handler")


def _report_progress(progress_callback, stage, percent, message):
    if progress_callback is None:
        return
    try:
        progress_callback(stage, max(0.0, min(100.0, float(percent))), message)
    except Exception:
        logger.debug("Progress callback failed", exc_info=True)


def normalize_upload_type(file_type, filename=""):
    """Normalize uploader extension including .csv.gz."""
    name = (filename or "").lower()
    ft = (file_type or "csv").lower().lstrip(".")
    if name.endswith(".csv.gz") or ft in {"gz", "csv.gz"}:
        return "csv.gz"
    if ft == "excel":
        return "xlsx"
    if ft == "json":
        return "json"
    return ft


def check_ingest_resources(file_size_bytes, copies=None):
    """Fail fast when temp disk (or RAM) cannot hold ingest copies."""
    copies = INGEST_DISK_COPIES if copies is None else copies
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    usage = shutil.disk_usage(TEMP_DATA_DIR)
    needed = int(file_size_bytes * copies) + INGEST_DISK_SLACK_BYTES
    if usage.free < needed:
        raise ValueError(
            f"Ruang disk tidak cukup untuk ingest. Diperlukan sekitar {needed / (1024**3):.2f} GB "
            f"bebas di {TEMP_DATA_DIR} (tersedia {usage.free / (1024**3):.2f} GB). "
            "Kosongkan disk atau set TEMP_DATA_DIR ke volume yang lebih besar."
        )
    try:
        import psutil
        # Only gate RAM for large uploads; small files and CI hosts with low
        # "available" pages must still ingest.
        if file_size_bytes >= 100 * 1024 * 1024:
            available = psutil.virtual_memory().available
            if available < 256 * 1024 * 1024:
                raise ValueError(
                    "RAM tersedia kurang dari 256 MB untuk file besar. "
                    "Tutup aplikasi lain sebelum mengunggah."
                )
    except ImportError:
        pass
    return True


def can_materialize_dataset(file_size_bytes, safety_factor=2.0, reserve_bytes=1024 * 1024 * 1024):
    """Whether it is safe to load a dataset fully into pandas."""
    try:
        import psutil
        available = psutil.virtual_memory().available
        return available > int(file_size_bytes * safety_factor) + reserve_bytes
    except Exception:
        return file_size_bytes <= MAX_DIRECT_DETECTION_FILE_SIZE


def _open_csv_text(file_path, encoding, errors="strict"):
    if str(file_path).lower().endswith(".gz"):
        return gzip.open(file_path, "rt", encoding=encoding, errors=errors)
    return open(file_path, "r", encoding=encoding, errors=errors)


def sniff_csv_dialect(file_path, sample_size=None):
    """Detect encoding and delimiter for CSV / CSV.GZ without reading the whole file."""
    sample_size = CSV_SNIFF_BYTES if sample_size is None else sample_size
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    try:
        opener = gzip.open if str(file_path).lower().endswith(".gz") else open
        with opener(file_path, "rb") as raw_file:
            prefix = raw_file.read(4)
        if prefix.startswith(b"\xff\xfe") or prefix.startswith(b"\xfe\xff"):
            encodings = ["utf-16", "utf-8-sig", "utf-8", "latin-1"]
        elif prefix.startswith(b"\xef\xbb\xbf"):
            encodings = ["utf-8-sig", "utf-8", "latin-1"]
    except Exception:
        pass

    last_error = None
    for encoding in encodings:
        try:
            with _open_csv_text(file_path, encoding, errors="strict") as handle:
                sample = handle.read(sample_size)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            continue
        if not sample:
            continue
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
            separator = dialect.delimiter
        except _csv.Error:
            separator = max([",", ";", "\t", "|"], key=sample.count)
            if sample.count(separator) == 0:
                separator = ","
        return {"encoding": encoding, "separator": separator}

    if last_error:
        logger.warning("CSV sniff fallback to latin-1: %s", last_error)
    return {"encoding": "latin-1", "separator": ","}


def _stream_upload_to_path(uploaded_file, destination_path, progress_callback=None):
    """Write an upload buffer to disk in 8MB chunks (single copy)."""
    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
    total_size = int(getattr(uploaded_file, "size", 0) or 0)
    copied = 0
    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    with open(destination_path, "wb") as out_file:
        while True:
            chunk = uploaded_file.read(UPLOAD_STREAM_BYTES)
            if not chunk:
                break
            out_file.write(chunk)
            copied += len(chunk)
            if total_size:
                _report_progress(
                    progress_callback,
                    "buffer",
                    10.0 + min(25.0, 25.0 * copied / total_size),
                    f"Menulis ke disk ({copied / (1024**2):.1f} MB)...",
                )
    return destination_path


def _pandas_csv_to_parquet(file_path, output_path, chunk_size, dialect, progress_callback=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    writer_closed = False
    total_rows = 0
    read_kwargs = {
        "chunksize": chunk_size,
        "sep": dialect.get("separator", ","),
        "encoding": dialect.get("encoding", "utf-8"),
        "low_memory": True,
        "on_bad_lines": "warn",
    }
    try:
        for chunk_number, chunk in enumerate(pd.read_csv(file_path, **read_kwargs), start=1):
            chunk = optimize_memory_usage(chunk)
            chunk = fix_arrow_compatibility(chunk)
            for column in chunk.select_dtypes(include=["category"]).columns:
                chunk[column] = chunk[column].astype(str)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema, safe=False)
            writer.write_table(table)
            total_rows += len(chunk)
            del table, chunk
            if chunk_number % 50 == 0:
                gc.collect()
                _report_progress(
                    progress_callback,
                    "convert",
                    min(85.0, 40.0 + chunk_number * 0.2),
                    f"Konversi CSV chunk {chunk_number}...",
                )
    except TypeError:
        read_kwargs.pop("on_bad_lines", None)
        return _pandas_csv_to_parquet_legacy(file_path, output_path, chunk_size, dialect, progress_callback)
    except Exception:
        if writer is not None:
            writer.close()
            writer_closed = True
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise
    finally:
        if writer is not None and not writer_closed:
            writer.close()
    return output_path, total_rows


def _pandas_csv_to_parquet_legacy(file_path, output_path, chunk_size, dialect, progress_callback=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    writer_closed = False
    total_rows = 0
    try:
        reader = pd.read_csv(
            file_path,
            chunksize=chunk_size,
            sep=dialect.get("separator", ","),
            encoding=dialect.get("encoding", "utf-8"),
            low_memory=True,
        )
        for chunk_number, chunk in enumerate(reader, start=1):
            chunk = optimize_memory_usage(chunk)
            chunk = fix_arrow_compatibility(chunk)
            for column in chunk.select_dtypes(include=["category"]).columns:
                chunk[column] = chunk[column].astype(str)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema, safe=False)
            writer.write_table(table)
            total_rows += len(chunk)
            del table, chunk
            if chunk_number % 50 == 0:
                gc.collect()
    except Exception:
        if writer is not None:
            writer.close()
            writer_closed = True
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise
    finally:
        if writer is not None and not writer_closed:
            writer.close()
    return output_path, total_rows


def stream_csv_to_parquet(file_path, output_path=None, chunk_size=50000, progress_bar=True, progress_callback=None):
    """Write a CSV (or .csv.gz) to Parquet in bounded-memory streaming.

    Prefers Polars scan/sink. Falls back to pandas chunks with sniffed
    delimiter/encoding. Returns a path and row count, not a DataFrame.
    """
    file_size = os.path.getsize(file_path)
    check_file_size(file_size)
    check_ingest_resources(file_size)
    if output_path is None:
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)
        output_path = os.path.join(TEMP_DATA_DIR, f"raw_{uuid.uuid4().hex}.parquet")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    dialect = sniff_csv_dialect(file_path)
    _report_progress(
        progress_callback,
        "sniff",
        15.0,
        f"Format CSV: pemisah '{dialect['separator']}', encoding {dialect['encoding']}",
    )

    # Polars scan_csv only accepts utf8 / utf8-lossy. Non-UTF encodings use pandas.
    use_polars = dialect["encoding"] in {"utf-8", "utf-8-sig"}
    if use_polars:
        try:
            _report_progress(progress_callback, "convert", 35.0, "Streaming CSV ke Parquet (Polars)...")
            lf = pl.scan_csv(
                file_path,
                separator=dialect["separator"],
                encoding="utf8-lossy",
                infer_schema_length=10000,
                ignore_errors=True,
                try_parse_dates=False,
                low_memory=True,
            )
            lf.sink_parquet(output_path, compression="zstd")
            total_rows = pl.scan_parquet(output_path).select(pl.len()).collect().item()
            if total_rows == 0:
                raise ValueError("CSV tidak mengandung baris data.")
            _report_progress(progress_callback, "convert", 90.0, f"Parquet siap ({total_rows:,} baris).")
            return output_path, int(total_rows)
        except Exception as exc:
            logger.warning("Polars CSV ingest failed (%s); falling back to pandas chunks.", exc)
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except Exception:
                    pass

    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(file_size) or 50000
    return _pandas_csv_to_parquet(
        file_path,
        output_path,
        chunk_size=chunk_size,
        dialect=dialect,
        progress_callback=progress_callback,
    )

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
        dialect = sniff_csv_dialect(file_path)
        try:
            df = pd.read_csv(
                file_path,
                sep=dialect["separator"],
                encoding=dialect["encoding"],
            )
        except Exception:
            df = pd.read_csv(file_path, sep=None, engine="python", encoding="latin-1")
        if df.empty:
            raise ValueError(
                "File berhasil dibaca tetapi tidak mengandung baris data (hanya header). "
                "Pastikan file memiliki minimal 1 baris data di bawah baris header."
            )
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
            # Try to convert object columns to string to avoid Arrow serialization issues
            # This handles mixed-type columns that can cause ArrowInvalid errors
            try:
                df[col] = df[col].astype('str')
            except Exception:
                # If conversion fails, try converting to nullable string dtype
                df[col] = df[col].astype('string')
    return df

def _read_local_file_with_optimization(file_path, file_type):
    """Read a local file after it has been safely buffered to disk."""
    if file_type in ('csv', 'csv.gz'):
        df = read_large_csv(file_path, progress_bar=True)
    elif file_type == 'parquet':
        st.info("🚀 Menggunakan Polars engine untuk Parquet...")
        df = read_with_polars(file_path, 'parquet')
        if df is None:
            df = pd.read_parquet(file_path)
        df = optimize_memory_usage(df)
    elif file_type in ['xlsx', 'xls']:
        excel_errors = ['#NULL!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#NUM!', '#N/A', '#N/A!']
        errors_to_replace = ['']
        for error in excel_errors:
            errors_to_replace.extend([
                error, error.lower(), error.upper(),
                error + ' ', ' ' + error
            ])
        df = pd.read_excel(
            file_path,
            na_values=errors_to_replace,
            keep_default_na=True,
        )
        if df.empty:
            raise ValueError(
                "File Excel berhasil dibaca tetapi tidak memiliki baris data. "
                "Pastikan sheet pertama berisi header dan minimal satu baris data."
            )
        df = optimize_memory_usage(df)
    elif file_type == 'json':
        df = pd.read_json(file_path)
        df = optimize_memory_usage(df)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return fix_arrow_compatibility(df)

def read_file_with_optimization(uploaded_file, file_type='csv'):
    """
    Read uploaded file with streaming buffer and memory optimization for large files
    """
    filename = getattr(uploaded_file, "name", "") or ""
    file_type = normalize_upload_type(file_type, filename)

    file_size = uploaded_file.size
    check_file_size(file_size)
    check_ingest_resources(file_size, copies=2.0)

    if file_type in ['xlsx', 'xls'] and file_size > MAX_EXCEL_FILE_SIZE:
        limit_mb = MAX_EXCEL_FILE_SIZE / (1024 * 1024)
        raise ValueError(
            f"File Excel dibatasi {limit_mb:.0f}MB karena parser Excel menggunakan memory penuh. "
            "Gunakan CSV atau Parquet untuk dataset yang lebih besar."
        )

    suffix = ".csv.gz" if file_type == "csv.gz" else f".{file_type}"
    tmp_file_path = os.path.join(TEMP_DATA_DIR, f"read_{uuid.uuid4().hex}{suffix}")
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    _stream_upload_to_path(uploaded_file, tmp_file_path)

    try:
        return _read_local_file_with_optimization(tmp_file_path, file_type)
    finally:
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

def ingest_file_to_raw_parquet(uploaded_file, file_type='csv', progress_callback=None):
    """
    Stream uploaded file to a raw Parquet file on disk without loading into RAM.
    Parquet uploads are written once (no extra temp copy). CSV is sniffed then
    streamed. Returns (raw_parquet_path, total_rows, schema_dict).
    """
    filename = getattr(uploaded_file, "name", "") or ""
    file_type = normalize_upload_type(file_type, filename)
    file_size = int(getattr(uploaded_file, "size", 0) or 0)
    check_file_size(file_size)
    check_ingest_resources(file_size)

    if file_type in ["xlsx", "xls"] and file_size > MAX_EXCEL_FILE_SIZE:
        limit_mb = MAX_EXCEL_FILE_SIZE / (1024 * 1024)
        raise ValueError(
            f"File Excel dibatasi {limit_mb:.0f}MB karena parser Excel menggunakan memory penuh. "
            "Gunakan CSV atau Parquet untuk dataset yang lebih besar."
        )

    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    unique_id = uuid.uuid4().hex
    raw_parquet_path = os.path.join(TEMP_DATA_DIR, f"raw_{unique_id}.parquet")
    tmp_file_path = None

    try:
        _report_progress(progress_callback, "start", 5.0, "Memulai ingest ke disk...")
        if file_type == "parquet":
            _stream_upload_to_path(uploaded_file, raw_parquet_path, progress_callback)
        elif file_type in ("csv", "csv.gz"):
            suffix = ".csv.gz" if file_type == "csv.gz" else ".csv"
            tmp_file_path = os.path.join(TEMP_DATA_DIR, f"upload_{unique_id}{suffix}")
            _stream_upload_to_path(uploaded_file, tmp_file_path, progress_callback)
            stream_csv_to_parquet(
                tmp_file_path,
                output_path=raw_parquet_path,
                progress_bar=False,
                progress_callback=progress_callback,
            )
        else:
            suffix = f".{file_type}"
            tmp_file_path = os.path.join(TEMP_DATA_DIR, f"upload_{unique_id}{suffix}")
            _stream_upload_to_path(uploaded_file, tmp_file_path, progress_callback)
            _report_progress(progress_callback, "convert", 50.0, f"Mengonversi {file_type} ke Parquet...")
            if file_type == "json":
                try:
                    pl.scan_ndjson(tmp_file_path).sink_parquet(raw_parquet_path, compression="zstd")
                except Exception:
                    df = _read_local_file_with_optimization(tmp_file_path, file_type)
                    df.to_parquet(raw_parquet_path, index=False, compression="zstd")
                    del df
                    gc.collect()
            else:
                df = _read_local_file_with_optimization(tmp_file_path, file_type)
                df.to_parquet(raw_parquet_path, index=False, compression="zstd")
                del df
                gc.collect()

        lf = pl.scan_parquet(raw_parquet_path)
        total_rows = lf.select(pl.len()).collect().item()
        if total_rows == 0:
            raise ValueError(
                "File berhasil dibaca tetapi tidak mengandung baris data. "
                "Pastikan file memiliki header dan minimal satu baris data."
            )
        schema = lf.collect_schema()
        schema_dict = {
            "columns": list(schema.names()),
            "dtypes": {name: str(dtype) for name, dtype in schema.items()},
            "total_rows": int(total_rows),
        }
        _report_progress(progress_callback, "done", 100.0, f"Ingest selesai ({int(total_rows):,} baris).")
        return raw_parquet_path, int(total_rows), schema_dict
    except Exception:
        if os.path.exists(raw_parquet_path):
            try:
                os.unlink(raw_parquet_path)
            except Exception:
                pass
        raise
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass


def load_parquet_bounded(parquet_path, max_rows=100000):
    """Load parquet fully only when RAM is safe; otherwise return a head sample."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    file_size = os.path.getsize(parquet_path)
    total_rows = int(pl.scan_parquet(parquet_path).select(pl.len()).collect().item())
    if can_materialize_dataset(file_size) and total_rows <= max_rows * 20:
        return pd.read_parquet(parquet_path), False, total_rows
    sample_n = min(max_rows, total_rows)
    return get_parquet_sample(parquet_path, n=sample_n), True, total_rows

def get_parquet_sample(parquet_path: str, n: int = 5000) -> pd.DataFrame:
    """
    Read a representative head sample from a Parquet file without loading the entire dataset.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    return pl.scan_parquet(parquet_path).head(n).collect().to_pandas()

def remove_duplicates_from_parquet(input_path, output_path=None, subset=None, keep='first'):
    """Remove duplicate rows from Parquet without materializing pandas data."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Parquet file not found: {input_path}")
    if keep not in ('first', 'last', False):
        raise ValueError("keep must be 'first', 'last', or False")

    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(TEMP_DATA_DIR, f"deduplicated_{uuid.uuid4().hex}.parquet")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    lf = pl.scan_parquet(input_path)
    schema = lf.collect_schema()
    columns = list(schema.names())
    if subset is not None:
        missing_columns = [column for column in subset if column not in columns]
        if missing_columns:
            raise ValueError(f"Kolom deduplikasi tidak ditemukan: {', '.join(missing_columns)}")

    original_rows = lf.select(pl.len()).collect().item()
    polars_keep = 'none' if keep is False else keep
    maintain_order = original_rows < 500_000
    try:
        lf.unique(
            subset=subset,
            keep=polars_keep,
            maintain_order=maintain_order,
        ).sink_parquet(output_path, compression='zstd')
        final_rows = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    except Exception:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise

    duplicate_count = original_rows - final_rows
    metadata = {
        'original_rows': int(original_rows),
        'duplicates_removed': int(duplicate_count),
        'final_rows': int(final_rows),
        'duplicate_rate': float(duplicate_count / original_rows) if original_rows else 0.0,
        'subset': subset,
        'keep': keep,
    }
    return output_path, metadata

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

    # Ensure TEMP_DATA_DIR exists before attempting to write
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)

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