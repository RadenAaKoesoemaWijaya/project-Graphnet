import streamlit as st
from model_registry import get_versions, load_model_version
from state_manager import navigate_to_page
from ui.utils import load_persisted_detector, get_gpu_status, get_gpu_status_display

MODEL_PREFIX = "models/fraud_detector"

def render_sidebar():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 10px; background-color: #FFD700; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #000000; margin: 0;">🛡️ ASTINA</h2>
        <p style="color: #000000; font-size: 0.8em; font-weight: bold; margin: 0;">ANALISIS SISTEM TRANSAKSI IDENTIFIKASI NILAI ANOMALI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Menu Utama")
    
    # Check if any long operation is in progress
    is_processing = st.session_state.get('is_processing', False)
    processing_message = st.session_state.get('processing_message', '')
    
    # Show processing indicator if active
    if is_processing:
        st.sidebar.warning(f"⏳ {processing_message}")
        st.sidebar.info("Menu dinonaktifkan selama pemrosesan. Silakan tunggu...")
    
    # Disable sidebar buttons if processing
    sidebar_disabled = is_processing
    
    if st.sidebar.button("🏠 Beranda", width='stretch', key="sidebar_home", disabled=sidebar_disabled):
        navigate_to_page('home')
    
    if st.sidebar.button("📂 Unggah Data", width='stretch', key="sidebar_collect", disabled=sidebar_disabled):
        navigate_to_page('collect')
    
    if st.sidebar.button("🎯 Pelatihan Model", width='stretch', key="sidebar_train", disabled=sidebar_disabled):
        navigate_to_page('train')
    
    if st.sidebar.button("📊 Evaluasi Model", width='stretch', key="sidebar_evaluate", disabled=sidebar_disabled):
        navigate_to_page('evaluate')
    
    if st.sidebar.button("🔍 Deteksi Anomali", width='stretch', key="sidebar_detect", disabled=sidebar_disabled):
        navigate_to_page('detect')
    
    if st.sidebar.button("🖥️ Status Sistem", width='stretch', key="sidebar_status", disabled=sidebar_disabled):
        navigate_to_page('status')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 💡 Status Sistem
    """)
    
    # Show status indicators in sidebar
    if 'df_processed_path' in st.session_state:
        st.sidebar.success("✅ Data Siap")
    else:
        st.sidebar.warning("⏳ Menunggu Data")
        
    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        st.sidebar.success("✅ Model Siap")
    else:
        st.sidebar.warning("⏳ Model Belum Dilatih")
    
    # GPU Status
    gpu_status_text = get_gpu_status_display()
    gpu_info = get_gpu_status()
    
    if gpu_info['cuda_available']:
        st.sidebar.success(gpu_status_text)
    else:
        st.sidebar.info(gpu_status_text)
    
    st.sidebar.markdown("---")
    st.sidebar.info("ASTINA v2.0 - Didukung pembelajaran mesin hibrida")
    
    # Footer
    st.sidebar.markdown("---")
    
    # Model Registry UI in sidebar
    st.sidebar.markdown("### 🗄️ Model Registry")
    versions = get_versions()
    if versions:
        version_names = [v['version'] for v in versions]
        # Sort newest first
        version_names.reverse()
        selected_version = st.sidebar.selectbox("Muat Versi Model", ["-- Pilih Versi --"] + version_names)
        if selected_version != "-- Pilih Versi --":
            if st.sidebar.button("Muat Model ini"):
                if load_model_version(selected_version, MODEL_PREFIX):
                    # Reload the detector into session state
                    loaded_det = load_persisted_detector()
                    if loaded_det:
                        st.sidebar.success(f"Berhasil memuat {selected_version}")
                    else:
                        st.sidebar.error("Model korup atau tidak lengkap.")
                else:
                    st.sidebar.error("Gagal memuat versi.")
    else:
        st.sidebar.info("Belum ada model yang tersimpan.")
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tentang ASTINA")
    st.sidebar.info("Sistem deteksi anomali transaksi asuransi kesehatan menggunakan pembelajaran mesin hibrida.")
