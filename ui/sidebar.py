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
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Status Pipeline & Sistem")

    # Compute pipeline readiness percentage
    has_data = 'df_processed_path' in st.session_state
    is_trained = st.session_state.get('model_trained', False)
    has_eval = 'evaluation_results' in st.session_state or 'eval_metrics' in st.session_state
    has_detected = 'fraud_results' in st.session_state or 'detection_results' in st.session_state

    progress_val = 0.1
    if has_data:
        progress_val = 0.35
    if is_trained:
        progress_val = 0.65
    if has_eval:
        progress_val = 0.85
    if has_detected:
        progress_val = 1.0

    st.sidebar.progress(progress_val, text=f"Kesiapan Pipeline: {int(progress_val * 100)}%")

    # Dataset Card
    meta = st.session_state.get('preprocessing_metadata', {})
    total_rows = meta.get('total_rows_processed') or st.session_state.get('raw_data_total_rows', 0)
    feature_cols = st.session_state.get('feature_columns', [])

    if has_data:
        data_status_badge = '<span style="color:#10B981; font-weight:700;">● SIAP</span>'
        row_str = f"{total_rows:,}" if total_rows > 0 else "-"
        feat_str = f"{len(feature_cols)}" if feature_cols else "-"
        data_engine_str = "Parquet Stream"
    else:
        data_status_badge = '<span style="color:#F59E0B; font-weight:700;">○ MENUNGGU</span>'
        row_str = "0"
        feat_str = "0"
        data_engine_str = "Menunggu Ingest"

    # Model Card
    if is_trained:
        model_status_badge = '<span style="color:#10B981; font-weight:700;">● AKTIF</span>'
        trained_det = st.session_state.get('trained_detector')
        model_name = getattr(trained_det, 'best_algorithm', None) or st.session_state.get('selected_model_type', 'Hybrid Ensemble')
        if isinstance(model_name, str):
            model_name = model_name.replace('_', ' ').title()
    else:
        model_status_badge = '<span style="color:#F59E0B; font-weight:700;">○ BELUM SIAP</span>'
        model_name = "-"

    # Hardware status
    gpu_status_text = get_gpu_status_display()
    gpu_info = get_gpu_status()
    hw_icon = "🚀" if gpu_info.get('cuda_available') else "💻"
    hw_name = gpu_info.get('device_name', 'CPU') if gpu_info.get('cuda_available') else "Multi-core CPU"

    # Render informative HTML status card
    sidebar_card_html = f"""
    <div class="sidebar-status-card">
        <div class="sidebar-status-header">
            <span>💾 Data Transaksi</span>
            {data_status_badge}
        </div>
        <div class="status-row">
            <span class="status-label">📈 Baris Data</span>
            <span class="status-val">{row_str}</span>
        </div>
        <div class="status-row">
            <span class="status-label">🔢 Fitur Terpilih</span>
            <span class="status-val">{feat_str}</span>
        </div>
        <div class="status-row">
            <span class="status-label">⚙️ Format Ingest</span>
            <span class="status-val">{data_engine_str}</span>
        </div>
    </div>

    <div class="sidebar-status-card">
        <div class="sidebar-status-header">
            <span>🎯 Model AI & ML</span>
            {model_status_badge}
        </div>
        <div class="status-row">
            <span class="status-label">🤖 Algoritma</span>
            <span class="status-val">{model_name[:18]}</span>
        </div>
        <div class="status-row">
            <span class="status-label">🧠 Copilot & RAG</span>
            <span class="status-val" style="color:#38BDF8;">Aktif</span>
        </div>
        <div class="status-row">
            <span class="status-label">🛡️ Audit Log</span>
            <span class="status-val" style="color:#34D399;">SHA-256</span>
        </div>
    </div>

    <div class="sidebar-status-card">
        <div class="sidebar-status-header">
            <span>🖥️ Hardware Engine</span>
            <span style="color:#38BDF8; font-weight:700;">{hw_icon}</span>
        </div>
        <div class="status-row">
            <span class="status-label">⚡ Akselerator</span>
            <span class="status-val">{hw_name[:16]}</span>
        </div>
        <div class="status-row">
            <span class="status-label">⚡ Engine Out-of-Core</span>
            <span class="status-val">Polars Lazy</span>
        </div>
    </div>
    """
    st.sidebar.markdown(sidebar_card_html, unsafe_allow_html=True)
    
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
    st.sidebar.caption("© 2026 TIM ASTINA INDONESIA. All rights reserved.")
