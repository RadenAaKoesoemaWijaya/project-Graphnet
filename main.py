import streamlit as st
import logging
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# FIX: Windows asyncio ProactorEventLoop ConnectionResetError (Python 3.12+)
# Must be set BEFORE uvicorn/anyio creates the event loop.
# ProactorEventLoop has a known bug causing noisy unhandled exceptions when
# browsers close TCP connections: _ProactorBasePipeTransport._call_connection_lost
# → sock.shutdown(SHUT_RDWR) → [WinError 10054] Connection reset by remote host.
# SelectorEventLoop is stable for Streamlit's HTTP/WebSocket I/O workload.
# ─────────────────────────────────────────────────────────────────────────────
if sys.platform == "win32" and sys.version_info < (3, 14):
    import asyncio
    _set_policy = getattr(asyncio, "set_event_loop_policy", None)
    _policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if _set_policy and _policy_cls:
        _set_policy(_policy_cls())

from state_manager import navigate_to_page

# Setup logger
logger = logging.getLogger(__name__)

# Patch for Python 3.11.0rc1 missing get_int_max_str_digits which causes PyTorch error
if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
    sys.set_int_max_str_digits = lambda maxdigits: None

# Validate configuration and environment
try:
    from config_validator import run_all_validations
    is_valid, errors = run_all_validations()
    if not is_valid:
        st.error("❌ Configuration/Environment validation failed:")
        for error in errors:
            st.error(f"  - {error}")
        st.stop()
except ImportError:
    logger.warning("config_validator not found, skipping validation")

# Configure thread pool and streaming limits to prevent memory exhaustion and crashes
max_threads = str(min(4, max(1, (os.cpu_count() or 2) // 2)))
os.environ.setdefault("POLARS_MAX_THREADS", max_threads)
os.environ.setdefault("RAYON_NUM_THREADS", max_threads)
os.environ.setdefault("OMP_NUM_THREADS", max_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", max_threads)
os.environ.setdefault("MKL_NUM_THREADS", max_threads)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", max_threads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", max_threads)
os.environ.setdefault("POLARS_STREAMING_CHUNK_SIZE", "50000")

# Suppress Polars CPU feature check warnings
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

# Set page configuration FIRST, before any other Streamlit commands
st.set_page_config(page_title='ASTINA - Analisis Sistem Transaksi Identifikasi Nilai Anomali', layout='wide')

from ui_components import apply_custom_css, render_footer
# Apply Custom CSS
apply_custom_css()

# UI components imports
from ui.sidebar import render_sidebar
from ui.pages.home import show_home_page
from ui.pages.data_collection import show_data_collection_page
from ui.pages.training import show_training_page
from ui.pages.evaluation import show_evaluation_page
from ui.pages.detection import show_detection_page
from ui.pages.status import show_status_page
from ui.pages.settings import show_settings_page

from auth_manager import AuthManager, validate_production_auth_configuration

def main():
    try:
        validate_production_auth_configuration()
        import cloud_storage
        cloud_storage.validate_production_storage_configuration()
    except RuntimeError as exc:
        logger.critical("%s", exc)
        st.error(str(exc))
        st.stop()

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'
    
    # Initialize processing flags
    if 'is_processing' not in st.session_state:
        st.session_state['is_processing'] = False
    if 'processing_message' not in st.session_state:
        st.session_state['processing_message'] = ''
    if 'page_before_processing' not in st.session_state:
        st.session_state['page_before_processing'] = None

    # Check Authentication Gateway
    if not AuthManager.is_authenticated():
        AuthManager.render_login_page()
        render_footer()
        return

    # Render Sidebar
    render_sidebar()

    # Get current page safely
    current_page = st.session_state.get('page', 'home')

    # RBAC Access Guard
    if not AuthManager.can_access_page(current_page):
        st.error(f"⛔ **Akses Ditolak**: Peran Anda (`{AuthManager.get_current_role().upper()}`) tidak memiliki izin untuk membuka halaman `{current_page}`.")
        st.info("Silakan gunakan navigasi menu di sebelah kiri untuk mengakses modul yang diizinkan.")
        render_footer()
        return

    # Display appropriate page
    try:
        if current_page == 'home':
            show_home_page()
        elif current_page == 'collect':
            show_data_collection_page()
        elif current_page == 'train':
            show_training_page()
        elif current_page == 'evaluate':
            show_evaluation_page()
        elif current_page == 'detect':
            show_detection_page()
        elif current_page == 'status':
            show_status_page()
        elif current_page == 'settings':
            show_settings_page()
    except Exception as e:
        # Log error and show recovery options
        import traceback
        import uuid
        incident_id = f"ASTINA-{uuid.uuid4().hex[:8].upper()}"
        logger.error(f"[{incident_id}] Error displaying page '{current_page}': {e}", exc_info=True)

        env_name = os.getenv("ASTINA_ENVIRONMENT", os.getenv("ENVIRONMENT", "development")).strip().lower()
        is_production = env_name in {"production", "prod"}

        if is_production:
            st.error(f"❌ Terjadi kesalahan internal pada halaman ({current_page}).")
            st.warning(
                f"🛡️ **Mode Keamanan Produksi Aktif**: Detail teknis dilindungi. "
                f"ID Insiden: `{incident_id}`. Silakan hubungi administrator sistem atau tim teknis dengan menyertakan ID ini."
            )
        else:
            st.error(f"❌ Terjadi kesalahan pada halaman ({current_page}): {str(e)}")
            with st.expander("🔍 Detail Teknis Kesalahan (Traceback) - [Dev Mode]"):
                st.code(traceback.format_exc(), language='python')
        
        st.markdown("---")
        col_err1, col_err2 = st.columns(2)
        with col_err1:
            if st.button("🔁 Muat Ulang Halaman Ini", key="retry_current_page"):
                st.rerun()
        with col_err2:
            if st.button("🏠 Kembali ke Beranda", key="back_to_home_from_error"):
                navigate_to_page('home')
    finally:
        # Render application footer with copyright
        render_footer()

if __name__ == "__main__":
    main()
