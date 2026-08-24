import streamlit as st
import logging
import sys
import os
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

from ui_components import apply_custom_css
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
    
    # Render Sidebar
    render_sidebar()

    # Get current page safely
    current_page = st.session_state.get('page', 'home')
    
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
        logger.error(f"Error displaying page '{current_page}': {e}", exc_info=True)
        st.error(f"❌ Terjadi kesalahan pada halaman: {str(e)}")
        
        st.markdown("---")
        if st.button("🏠 Kembali ke Beranda"):
            navigate_to_page('home')

if __name__ == "__main__":
    main()
