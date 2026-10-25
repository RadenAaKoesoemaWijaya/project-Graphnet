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
    """Check disk and RAM resources for ingest.

    Returns
    -------
    tuple(bool, str | None)
        (ok, warning_message).  ``ok`` is always True (disk issues still
        raise).  ``warning_message`` is a human-readable string when RAM
        is low but the upload can still proceed, or ``None`` when
        resources look fine.

    Raises
    ------
    ValueError
        Only when *disk* space is insufficient (unrecoverable).
    """
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

    warning_msg = None
    try:
        import psutil
        gc.collect()
        available = psutil.virtual_memory().available
        avail_mb = available / (1024 * 1024)
        file_mb = file_size_bytes / (1024 * 1024)

        # Tiered RAM check:
        # 1. Critical: RAM < 64 MB AND file > 500 MB → hard block
        # 2. Low: RAM < 128 MB AND file > 100 MB → soft warning
        # 3. Otherwise → proceed normally
        critical_ram_mb = int(os.getenv("ASTINA_CRITICAL_RAM_MB", "64"))
        warn_ram_mb = int(os.getenv("ASTINA_MIN_AVAILABLE_RAM_MB", "128"))
        large_file_threshold_mb = 500

        if avail_mb < critical_ram_mb and file_mb > large_file_threshold_mb:
            raise ValueError(
                f"RAM tersedia ({avail_mb:.0f} MB) sangat rendah dan file terlalu besar ({file_mb:.0f} MB). "
                f"Minimal {critical_ram_mb} MB RAM diperlukan untuk file >500 MB. "
                "Tutup aplikasi lain atau gunakan file yang lebih kecil."
            )
        elif avail_mb < warn_ram_mb and file_mb > 100:
            warning_msg = (
                f"⚠️ RAM tersedia rendah ({avail_mb:.0f} MB). "
                "Proses upload tetap dilanjutkan dengan mode hemat memori. "
                "Untuk performa optimal, tutup aplikasi lain."
            )
            logger.warning(
                "Low RAM warning: %.0f MB available, file size %.0f MB",
                avail_mb, file_mb,
            )
    except ImportError:
        pass

    return True, warning_msg


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
        handle = None
        try:
            handle = _open_csv_text(file_path, encoding, errors="strict")
            sample = handle.read(sample_size)
            handle.close()
        except UnicodeDecodeError as exc:
            last_error = exc
            if handle:
                handle.close()
            continue
        except PermissionError as exc:
            logger.error(f"Permission denied reading file {file_path}: {exc}")
            last_error = exc
            if handle:
                handle.close()
            continue
        except Exception as exc:
            last_error = exc
            if handle:
                handle.close()
            continue
        if not sample:
            if handle:
                handle.close()
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
    sep = str(dialect.get("separator", ","))
    encoding = str(dialect.get("encoding", "utf-8"))
    try:
        reader = pd.read_csv(
            file_path,
            chunksize=int(chunk_size),
            sep=sep,
            encoding=encoding,
            low_memory=True,
            on_bad_lines="warn",
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
                _report_progress(
                    progress_callback,
                    "convert",
                    min(85.0, 40.0 + chunk_number * 0.2),
                    f"Konversi CSV chunk {chunk_number}...",
                )
    except TypeError:
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
    check_ingest_resources(file_size)  # warning ignored in internal path
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


def stream_excel_to_parquet(file_path, output_path=None, chunk_size=25000, progress_callback=None):
    """Write an Excel (.xlsx / .xls) to Parquet with memory safety.

    Uses openpyxl read_only iterator for .xlsx to stream row chunks directly to Parquet.
    Falls back to pandas for .xls or complex sheets.
    """
    file_size = os.path.getsize(file_path)
    check_file_size(file_size, "xlsx")
    check_ingest_resources(file_size)  # warning ignored in internal path
    if output_path is None:
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)
        output_path = os.path.join(TEMP_DATA_DIR, f"raw_{uuid.uuid4().hex}.parquet")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    excel_errors = {'#NULL!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#NUM!', '#N/A', '#N/A!'}

    if str(file_path).lower().endswith(".xlsx"):
        try:
            import openpyxl
            import pyarrow as pa
            import pyarrow.parquet as pq

            wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            sheet_names = wb.sheetnames
            if not sheet_names:
                raise ValueError("Workbook Excel tidak memiliki sheet.")
            ws = wb[sheet_names[0]]
            rows_iter = ws.iter_rows(values_only=True)

            header_row = None
            for row in rows_iter:
                if row and any(cell is not None for cell in row):
                    header_row = [
                        str(cell).strip() if cell is not None else f"col_{idx}"
                        for idx, cell in enumerate(row)
                    ]
                    break
            if not header_row:
                wb.close()
                raise ValueError("File Excel tidak memiliki baris header.")

            writer = None
            chunk_buffer = []
            total_rows = 0

            for row in rows_iter:
                if row is None or not any(c is not None for c in row):
                    continue
                cleaned_row = [
                    None if (isinstance(c, str) and c.strip().upper() in excel_errors) else c
                    for c in row[:len(header_row)]
                ]
                if len(cleaned_row) < len(header_row):
                    cleaned_row.extend([None] * (len(header_row) - len(cleaned_row)))
                chunk_buffer.append(cleaned_row)
                total_rows += 1

                if len(chunk_buffer) >= chunk_size:
                    chunk_df = pd.DataFrame(chunk_buffer, columns=header_row)
                    chunk_df = optimize_memory_usage(chunk_df)
                    chunk_df = fix_arrow_compatibility(chunk_df)
                    table = pa.Table.from_pandas(chunk_df)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                    writer.write_table(table)
                    chunk_buffer = []
                    del chunk_df, table
                    _report_progress(progress_callback, "convert", 65.0, f"Mengonversi Excel ({total_rows:,} baris)...")

            if chunk_buffer:
                chunk_df = pd.DataFrame(chunk_buffer, columns=header_row)
                chunk_df = optimize_memory_usage(chunk_df)
                chunk_df = fix_arrow_compatibility(chunk_df)
                table = pa.Table.from_pandas(chunk_df)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                chunk_buffer = []
                del chunk_df, table

            if writer:
                writer.close()
            wb.close()

            if total_rows == 0:
                raise ValueError("File Excel tidak mengandung baris data.")

            _report_progress(progress_callback, "convert", 90.0, f"Excel siap ({total_rows:,} baris).")
            return output_path, total_rows
        except Exception as exc:
            logger.warning("Streaming Excel gagal (%s); falling back to pandas.", exc)
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except Exception:
                    pass

    # Fallback to pandas
    df = _read_local_file_with_optimization(file_path, "xlsx" if str(file_path).lower().endswith(".xlsx") else "xls")
    df.to_parquet(output_path, index=False, compression="zstd")
    row_count = len(df)
    del df
    gc.collect()
    return output_path, row_count


def read_with_polars(file_path, file_type='csv'):
    """
    Read file using Polars for high-performance big data handling
    """
    try:
        if file_type == 'csv':
            df_pl = pl.read_csv(file_path)
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
    check_file_size(file_size, file_type)
    check_ingest_resources(file_size, copies=2.0)  # warning ignored in internal path

    if file_type in ['xlsx', 'xls'] and file_size > MAX_EXCEL_FILE_SIZE:
        limit_mb = MAX_EXCEL_FILE_SIZE / (1024 * 1024)
        raise ValueError(
            f"File Excel dibatasi {limit_mb:.0f}MB karena parser Excel menggunakan memory penuh. "
            "Gunakan CSV atau Parquet untuk dataset yang lebih besar."
        )

    if file_type == 'parquet' and file_size > MAX_PARQUET_FILE_SIZE:
        limit_mb = MAX_PARQUET_FILE_SIZE / (1024 * 1024)
        limit_gb = MAX_PARQUET_FILE_SIZE / (1024 * 1024 * 1024)
        raise ValueError(
            f"File Parquet dibatasi {limit_mb:.0f}MB ({limit_gb:.1f}GB) untuk menjaga kestabilan memori lokal. "
            "Gunakan CSV streaming untuk dataset hingga 3GB."
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
    filename = getattr(uploaded_file, "name", "") or ""
    file_type = normalize_upload_type("", filename)
    max_allowed = get_max_file_size_for_type(file_type)

    info = {
        'size_bytes': file_size,
        'size_mb': file_size_mb,
        'size_gb': file_size_gb,
        'file_type': file_type,
        'max_allowed_bytes': max_allowed,
        'max_allowed_mb': max_allowed / (1024 * 1024),
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
    check_file_size(file_size, file_type)
    check_ingest_resources(file_size)  # warning ignored in internal path

    if file_type in ["xlsx", "xls"] and file_size > MAX_EXCEL_FILE_SIZE:
        limit_mb = MAX_EXCEL_FILE_SIZE / (1024 * 1024)
        raise ValueError(
            f"File Excel dibatasi {limit_mb:.0f}MB karena parser Excel menggunakan memory penuh. "
            "Gunakan CSV atau Parquet untuk dataset yang lebih besar."
        )

    if file_type == "parquet" and file_size > MAX_PARQUET_FILE_SIZE:
        limit_mb = MAX_PARQUET_FILE_SIZE / (1024 * 1024)
        limit_gb = MAX_PARQUET_FILE_SIZE / (1024 * 1024 * 1024)
        raise ValueError(
            f"File Parquet dibatasi {limit_mb:.0f}MB ({limit_gb:.1f}GB) untuk menjaga kestabilan memori lokal. "
            "Gunakan CSV streaming untuk dataset hingga 3GB."
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
        elif file_type in ("xlsx", "xls"):
            suffix = f".{file_type}"
            tmp_file_path = os.path.join(TEMP_DATA_DIR, f"upload_{unique_id}{suffix}")
            _stream_upload_to_path(uploaded_file, tmp_file_path, progress_callback)
            _report_progress(progress_callback, "convert", 45.0, "Mengonversi Excel ke Parquet secara streaming...")
            stream_excel_to_parquet(
                tmp_file_path,
                output_path=raw_parquet_path,
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


def validate_join_columns(df1_columns, df2_columns, join_columns):
    """
    Validate that join columns exist and are available.
    
    Args:
        df1_columns: List of column names in first dataset
        df2_columns: List of column names in second dataset
        join_columns: List of column names to use as join keys
    
    Returns:
        (is_valid, error_message)
    """
    if not join_columns:
        return False, "Kolom join tidak boleh kosong"
    
    missing_in_df1 = [col for col in join_columns if col not in df1_columns]
    missing_in_df2 = [col for col in join_columns if col not in df2_columns]
    
    if missing_in_df1 or missing_in_df2:
        error_parts = []
        if missing_in_df1:
            error_parts.append(f"tidak ada di dataset 1: {', '.join(missing_in_df1)}")
        if missing_in_df2:
            error_parts.append(f"tidak ada di dataset 2: {', '.join(missing_in_df2)}")
        return False, f"Kolom join {', '.join(error_parts)}"
    
    return True, None


def estimate_merge_size(df1_rows, df2_rows, join_type, key_cardinality_estimate=None):
    """
    Estimate the size of merged dataset to prevent OOM.
    
    Args:
        df1_rows: Number of rows in first dataset
        df2_rows: Number of rows in second dataset
        join_type: Type of join ('inner', 'left', 'outer', 'cross')
        key_cardinality_estimate: Estimated number of unique keys (optional)
    
    Returns:
        (estimated_rows, risk_level)
    """
    if join_type == "cross":
        estimated = df1_rows * df2_rows
        risk = "CRITICAL" if estimated > 10_000_000 else "HIGH"
    elif join_type == "inner":
        if key_cardinality_estimate:
            estimated = min(df1_rows, df2_rows, key_cardinality_estimate)
        else:
            estimated = min(df1_rows, df2_rows) * 0.3  # Conservative estimate
        risk = "LOW"
    elif join_type == "left":
        estimated = df1_rows
        risk = "LOW"
    elif join_type == "outer":
        estimated = df1_rows + df2_rows
        risk = "MEDIUM"
    else:
        estimated = df1_rows + df2_rows
        risk = "MEDIUM"
    
    return int(estimated), risk


def merge_datasets_with_key(
    parquet_paths,
    join_columns,
    output_path=None,
    join_type="inner",
    add_source_column=True,
    file_names=None,
    progress_callback=None,
):
    """
    Merge multiple datasets based on specified join columns (key-based merge).
    
    Args:
        parquet_paths: List of parquet file paths to merge
        join_columns: List of column names to use as join keys
        output_path: Output parquet file path
        join_type: Type of join - 'inner', 'left', 'outer', 'cross'
        add_source_column: Whether to add source_file column
        file_names: List of source file names
        progress_callback: Progress callback function
    
    Returns:
        output_path: Path to merged parquet file
    """
    if not parquet_paths:
        raise ValueError("Tidak ada file Parquet yang diberikan untuk digabungkan.")
    
    if not join_columns:
        raise ValueError("Join columns harus ditentukan untuk key-based merge.")
    
    if len(parquet_paths) < 2:
        raise ValueError("Minimal 2 file diperlukan untuk key-based merge.")
    
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    if output_path is None:
        unique_id = uuid.uuid4().hex
        output_path = os.path.join(TEMP_DATA_DIR, f"merged_key_{unique_id}.parquet")
    
    _report_progress(progress_callback, "merge_start", 10.0, 
                   f"Menyiapkan key-based merge pada kolom: {', '.join(join_columns)}")
    
    try:
        # Load first dataset as base
        base_lf = pl.scan_parquet(parquet_paths[0])
        base_schema = base_lf.collect_schema()
        base_cols = list(base_schema.names())
        
        if add_source_column:
            base_fname = file_names[0] if file_names else os.path.basename(parquet_paths[0])
            base_lf = base_lf.with_columns(pl.lit(str(base_fname)).alias("source_file"))
        
        _report_progress(progress_callback, "loading_base", 20.0, 
                       f"Memuat dataset dasar: {os.path.basename(parquet_paths[0])}")
        
        # Merge with other datasets sequentially
        for i, (path, fname) in enumerate(zip(parquet_paths[1:], 
                                              file_names[1:] if file_names else 
                                              [os.path.basename(p) for p in parquet_paths[1:]]), 
                                      start=1):
            _report_progress(progress_callback, f"merge_step_{i}", 
                           30.0 + (i * 20.0 / len(parquet_paths)),
                           f"Menggabungkan dataset {i+1}/{len(parquet_paths)}: {fname}")
            
            current_lf = pl.scan_parquet(path)
            current_schema = current_lf.collect_schema()
            current_cols = list(current_schema.names())
            
            # Validate join columns exist
            is_valid, error_msg = validate_join_columns(base_cols, current_cols, join_columns)
            if not is_valid:
                raise ValueError(
                    f"Validasi join gagal untuk {fname}: {error_msg}"
                )
            
            if add_source_column:
                current_lf = current_lf.with_columns(pl.lit(str(fname)).alias("source_file"))
            
            # Perform join
            try:
                base_lf = base_lf.join(
                    current_lf,
                    on=join_columns,
                    how=join_type,
                    suffix=f"_ds{i}"
                )
            except Exception as e:
                logger.error(f"Key-based join failed for {fname}: {e}")
                raise ValueError(
                    f"Gagal menggabungkan {fname} dengan key-based join. "
                    f"Pastikan kolom join '{', '.join(join_columns)}' memiliki tipe data yang konsisten."
                ) from e
        
        _report_progress(progress_callback, "writing", 90.0, "Menulis hasil merge ke disk...")
        
        # Write merged result
        base_lf.sink_parquet(output_path, compression="zstd")
        
        _report_progress(progress_callback, "merge_done", 100.0, "Key-based merge selesai.")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Key-based merge failed: {e}")
        raise


def merge_multiple_raw_parquets(
    parquet_paths,
    output_path=None,
    add_source_column=True,
    file_names=None,
    progress_callback=None,
):
    """
    Merge multiple raw Parquet files into a single master Parquet file
    using Polars diagonal lazy concatenation (zero RAM spike).
    """
    if not parquet_paths:
        raise ValueError("Tidak ada file Parquet yang diberikan untuk digabungkan.")

    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    if output_path is None:
        unique_id = uuid.uuid4().hex
        output_path = os.path.join(TEMP_DATA_DIR, f"merged_raw_{unique_id}.parquet")

    _report_progress(progress_callback, "merge_start", 10.0, "Menyiapkan penggabungan dataset...")

    lfs = []
    file_names = file_names or [os.path.basename(p) for p in parquet_paths]
    for p, fname in zip(parquet_paths, file_names):
        if not os.path.exists(p):
            continue
        lf = pl.scan_parquet(p)
        if add_source_column:
            lf = lf.with_columns(pl.lit(str(fname)).alias("source_file"))
        lfs.append(lf)

    if not lfs:
        raise ValueError("Seluruh file Parquet yang akan digabungkan tidak ditemukan atau kosong.")

    _report_progress(progress_callback, "merging", 50.0, f"Menggabungkan {len(lfs)} dataset via Polars streaming...")

    merged_lf = pl.concat(lfs, how="diagonal")
    merged_lf.sink_parquet(output_path, compression="zstd")

    _report_progress(progress_callback, "merge_done", 90.0, "Penggabungan dataset selesai.")
    return output_path


def ingest_multiple_files_to_raw_parquet(
    uploaded_files,
    add_source_column=True,
    merge_mode="diagonal",
    join_columns=None,
    join_type="inner",
    progress_callback=None,
):
    """
    Ingest multiple uploaded files (CSV/Excel/Parquet/GZ) streaming to disk,
    then merge them into a single master raw Parquet file.
    
    Args:
        uploaded_files: List of uploaded file objects
        add_source_column: Whether to add source_file column
        merge_mode: 'diagonal' or 'key_based'
        join_columns: List of columns for key-based join
        join_type: Type of join (inner, left, outer, cross)
        progress_callback: Progress callback function
    
    Returns:
        (master_parquet_path, total_rows, schema_dict, summaries)
    """
    if not uploaded_files:
        raise ValueError("Tidak ada file yang diunggah.")

    total_size = sum(int(getattr(f, "size", 0) or 0) for f in uploaded_files)
    check_ingest_resources(total_size)  # warning ignored in internal path

    raw_paths = []
    file_names = []
    summaries = []
    total_files = len(uploaded_files)

    try:
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            fname = getattr(uploaded_file, "name", f"file_{idx}")
            ftype = normalize_upload_type("", fname)
            fsize = int(getattr(uploaded_file, "size", 0) or 0)

            check_file_size(fsize, ftype)

            def sub_callback(stage, pct, msg, _idx=idx, _fname=fname):
                overall_pct = ((_idx - 1) / total_files) * 75.0 + (pct / 100.0) * (75.0 / total_files)
                _report_progress(progress_callback, "sub_ingest", overall_pct, f"[{_idx}/{total_files}] {_fname}: {msg}")

            raw_path, rows, schema = ingest_file_to_raw_parquet(
                uploaded_file, ftype, progress_callback=sub_callback
            )
            raw_paths.append(raw_path)
            file_names.append(fname)
            summaries.append({
                "file_name": fname,
                "file_type": ftype,
                "size_bytes": fsize,
                "size_mb": fsize / (1024 * 1024),
                "rows": rows,
                "raw_path": raw_path,
                "columns": schema.get("columns", []),
            })

        # Merge based on mode
        if len(raw_paths) == 1 and not add_source_column:
            master_path = raw_paths[0]
        else:
            def merge_cb(stage, pct, msg):
                overall_pct = 75.0 + (pct / 100.0) * 23.0
                _report_progress(progress_callback, "merge", overall_pct, msg)

            if merge_mode == "key_based" and join_columns:
                # Key-based merge
                master_path = merge_datasets_with_key(
                    raw_paths,
                    join_columns=join_columns,
                    join_type=join_type,
                    add_source_column=add_source_column,
                    file_names=file_names,
                    progress_callback=merge_cb,
                )
            else:
                # Default diagonal concat
                master_path = merge_multiple_raw_parquets(
                    raw_paths,
                    add_source_column=add_source_column,
                    file_names=file_names,
                    progress_callback=merge_cb,
                )
            
            # Remove intermediate raw parquets if merged into new master
            for p in raw_paths:
                if p != master_path and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

        lf = pl.scan_parquet(master_path)
        total_rows = int(lf.select(pl.len()).collect().item())
        schema = lf.collect_schema()
        schema_dict = {
            "columns": list(schema.names()),
            "dtypes": {name: str(dtype) for name, dtype in schema.items()},
            "total_rows": total_rows,
        }

        merge_mode_str = "key-based" if merge_mode == "key_based" else "diagonal"
        _report_progress(
            progress_callback,
            "complete",
            100.0,
            f"Seluruh {total_files} dataset berhasil digabungkan ({merge_mode_str}, {total_rows:,} baris).",
        )
        return master_path, total_rows, schema_dict, summaries

    except Exception:
        for p in raw_paths:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass
        raise


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

def show_file_size_warning(file_size_gb, file_type=None):
    """
    Show warning for large files with format awareness
    """
    ft = (file_type or "").lower().lstrip(".")
    if ft in ("xlsx", "xls"):
        if file_size_gb > 0.08:  # > 80MB
            st.warning(f"⚠️ File Excel berukuran {file_size_gb*1024:.0f} MB terdeteksi. Streaming iterator aktif untuk menjaga stabilitas RAM.")
    elif ft == "parquet":
        if file_size_gb > 0.5:
            st.info(f"🚀 File Parquet terkompresi ({file_size_gb:.2f} GB) memuat data skala besar. ASTINA menggunakan lazy columnar streaming Polars untuk menjaga memori tetap optimal.")
    else:
        if file_size_gb > 1.0:
            st.warning(f"⚠️ File CSV besar terdeteksi ({file_size_gb:.2f} GB). Ingest streaming ke Parquet aktif untuk mengoptimalkan penggunaan memory.")
            st.info("💡 Tips: File diproses per partisi tanpa membebani RAM.")
        elif file_size_gb > 0.5:
            st.info(f"📊 File medium ({file_size_gb:.2f} GB) akan diproses dengan optimasi memory streaming.")

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