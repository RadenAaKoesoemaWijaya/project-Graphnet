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
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from state_manager import navigate_to_page

# Setup logger
logger = logging.getLogger(__name__)

# Patch for Python 3.11.0rc1 missing get_int_max_str_digits which causes PyTorch error
if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
    sys.set_int_max_str_digits = lambda maxdigits: None

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

from auth_manager import AuthManager

def main():
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
    except Exception as e:
        # Log error and show recovery options
        import traceback
        logger.error(f"Error displaying page '{current_page}': {e}", exc_info=True)
        st.error(f"❌ Terjadi kesalahan pada halaman ({current_page}): {str(e)}")
        
        with st.expander("🔍 Detail Teknis Kesalahan (Traceback)"):
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
