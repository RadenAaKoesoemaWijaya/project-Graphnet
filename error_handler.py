"""
Enhanced error handling module for ASTINA with actionable error messages
and troubleshooting tips.
"""
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

import traceback
import sys
from typing import Optional, Dict, Any
from functools import wraps
import logging

logger = logging.getLogger("graphnet.error_handler")

class ErrorContext:
    """Context information for better error messages"""
    
    ERROR_CONTEXTS = {
        'MemoryError': {
            'title': 'Memori Tidak Cukup',
            'tips': [
                '💡 Kurangi ukuran chunk processing',
                '💡 Gunakan sampling untuk data besar',
                '💡 Tutup aplikasi lain yang memakan memori',
                '💡 Coba gunakan format Parquet yang lebih efisien'
            ],
            'severity': 'error'
        },
        'ValueError': {
            'title': 'Nilai Data Tidak Valid',
            'tips': [
                '💡 Periksa format file dan pastikan sesuai',
                '💡 Validasi kolom data sebelum upload',
                '💡 Pastikan tidak ada nilai kosong di kolom kunci',
                '💡 Cek tipe data yang diharapkan'
            ],
            'severity': 'error'
        },
        'KeyError': {
            'title': 'Kolom Data Tidak Ditemukan',
            'tips': [
                '💡 Pastikan nama kolom sesuai dengan yang diharapkan',
                '💡 Cek case-sensitive nama kolom',
                '💡 Gunakan fitur mapping kolom jika tersedia',
                '💡 Lihat sampel data untuk nama kolom yang benar'
            ],
            'severity': 'error'
        },
        'FileNotFoundError': {
            'title': 'File Tidak Ditemukan',
            'tips': [
                '💡 Pastikan file path benar',
                '💡 Cek apakah file sudah diupload',
                '💡 Verifikasi file permissions',
                '💡 Coba upload ulang file'
            ],
            'severity': 'error'
        },
        'AttributeError': {
            'title': 'Atribut Tidak Tersedia',
            'tips': [
                '💡 Pastikan data sudah diproses dengan benar',
                '💡 Cek apakah model sudah dilatih',
                '💡 Verifikasi langkah-langkah preprocessing',
                '💡 Restart aplikasi jika error persist'
            ],
            'severity': 'warning'
        },
        'TypeError': {
            'title': 'Tipe Data Tidak Sesuai',
            'tips': [
                '💡 Pastikan tipe data kolom benar',
                '💡 Konversi tipe data sebelum processing',
                '💡 Cek dokumentasi untuk tipe data yang diharapkan',
                '💡 Gunakan fungsi konversi yang tersedia'
            ],
            'severity': 'error'
        },
        'ConnectionError': {
            'title': 'Koneksi Gagal',
            'tips': [
                '💡 Cek koneksi internet',
                '💡 Pastikan server tersedia',
                '💡 Coba lagi dalam beberapa saat',
                '💡 Gunakan mode offline jika tersedia'
            ],
            'severity': 'warning'
        },
        'TimeoutError': {
            'title': 'Waktu Proses Habis',
            'tips': [
                '💡 Kurangi ukuran data yang diproses',
                '💡 Tingkatkan timeout di konfigurasi',
                '💡 Gunakan processing yang lebih efisien',
                '💡 Coba lagi dengan data yang lebih kecil'
            ],
            'severity': 'warning'
        },
        'ImportError': {
            'title': 'Modul Tidak Ditemukan',
            'tips': [
                '💡 Install dependensi yang hilang: pip install -r requirements.txt',
                '💡 Pastikan virtual environment aktif',
                '💡 Cek versi Python yang kompatibel',
                '💡 Update pip: pip install --upgrade pip'
            ],
            'severity': 'error'
        },
        'RuntimeError': {
            'title': 'Runtime Error',
            'tips': [
                '💡 Cek log error untuk detail lebih lanjut',
                '💡 Pastikan konfigurasi sistem benar',
                '💡 Verifikasi input data',
                '💡 Hubungi support jika error persist'
            ],
            'severity': 'error'
        }
    }
    
    @classmethod
    def get_context(cls, error: Exception) -> Dict[str, Any]:
        """Get error context based on exception type"""
        error_type = type(error).__name__
        return cls.ERROR_CONTEXTS.get(error_type, {
            'title': 'Error Tidak Dikenal',
            'tips': [
                '💡 Cek log error untuk detail lebih lanjut',
                '💡 Pastikan input data valid',
                '💡 Coba restart aplikasi',
                '💡 Hubungi support dengan error message'
            ],
            'severity': 'error'
        })

def handle_error_with_context(
    error: Exception,
    context: Optional[str] = None,
    show_traceback: bool = False,
    log_error: bool = True
) -> None:
    """
    Display user-friendly error message with actionable troubleshooting tips.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        show_traceback: Whether to show full traceback (for debugging)
        log_error: Whether to log the error to the logging system
    """
    if log_error:
        logger.error(f"Error in {context if context else 'unknown context'}: {str(error)}", exc_info=True)
    
    error_context = ErrorContext.get_context(error)
    
    # Display error title
    if STREAMLIT_AVAILABLE and st:
        if error_context['severity'] == 'error':
            st.error(f"❌ {error_context['title']}")
        else:
            st.warning(f"⚠️ {error_context['title']}")
        
        # Display error message
        st.markdown(f"**Detail Error:** `{str(error)}`")
        
        # Display context if provided
        if context:
            st.info(f"📍 **Konteks:** {context}")
        
        # Display actionable tips
        st.markdown("**Solusi yang Disarankan:**")
        for tip in error_context['tips']:
            st.markdown(tip)
        
        # Show traceback for debugging if requested
        if show_traceback:
            with st.expander("🔍 Detail Teknis (Traceback)"):
                st.code(traceback.format_exc(), language='python')
    else:
        # Fallback for non-streamlit environments
        print(f"❌ {error_context['title']}")
        print(f"Detail Error: {str(error)}")
        if context:
            print(f"Konteks: {context}")
        print("Solusi yang Disarankan:")
        for tip in error_context['tips']:
            print(tip)
        if show_traceback:
            print("Traceback:")
            traceback.print_exc()

def safe_execute(
    context: str,
    show_traceback: bool = False,
    default_return: Any = None,
    log_error: bool = True
):
    """
    Decorator for safe execution with enhanced error handling.
    
    Args:
        context: Description of the operation being performed
        show_traceback: Whether to show full traceback on error
        default_return: Value to return if error occurs
        log_error: Whether to log the error
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error_with_context(e, context, show_traceback, log_error)
                return default_return
        return wrapper
    return decorator

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame before processing.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Raises:
        ValueError: If validation fails
    """
    if df is None:
        raise ValueError("DataFrame is None")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(df)}")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has only {len(df)} rows, minimum {min_rows} required")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")

def get_user_friendly_error_message(error: Exception) -> str:
    """
    Convert technical error message to user-friendly Indonesian message.
    
    Args:
        error: The exception to convert
        
    Returns:
        User-friendly error message in Indonesian
    """
    error_messages = {
        'MemoryError': 'Memori komputer tidak cukup untuk memproses data ini.',
        'ValueError': 'Format atau nilai data tidak sesuai dengan yang diharapkan.',
        'KeyError': 'Kolom data yang diperlukan tidak ditemukan.',
        'FileNotFoundError': 'File tidak dapat ditemukan atau belum diupload.',
        'AttributeError': 'Fitur atau fungsi yang diminta tidak tersedia.',
        'TypeError': 'Tipe data tidak sesuai dengan yang diharapkan.',
        'ConnectionError': 'Tidak dapat terhubung ke server atau layanan.',
        'TimeoutError': 'Proses memakan waktu terlalu lama dan dihentikan.',
        'ImportError': 'Modul atau library yang diperlukan tidak terinstall.',
    }
    
    error_type = type(error).__name__
    return error_messages.get(error_type, f'Terjadi error: {str(error)}')

# Import pandas for validation
try:
    import pandas as pd
except ImportError:
    pd = None
