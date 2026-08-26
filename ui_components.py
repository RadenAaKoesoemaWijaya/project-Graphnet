import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app with optimized selectors"""
    st.markdown("""
<style>
    :root {
        --primary-color: #1E40AF;
        --secondary-color: #3B82F6;
        --bg-color: #F8FAFC;
        --text-color: #334155;
        --border-color: #E2E8F0;
        --success-color: #10B981;
        --error-color: #EF4444;
        --warning-color: #F59E0B;
        --info-color: #3B82F6;
    }

    /* Main background and text colors */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-color) !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em;
    }

    h1 {
        border-bottom: 2px solid var(--secondary-color);
        padding-bottom: 12px;
        margin-bottom: 24px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--border-color);
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-color);
    }

    /* Button styling */
    .stButton>button {
        background-color: #FFFFFF;
        color: var(--primary-color);
        border: 1px solid var(--primary-color);
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:hover {
        background-color: var(--secondary-color);
        color: #FFFFFF;
        border-color: var(--secondary-color);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
        font-size: 2.2rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--secondary-color);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.1);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 4px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #64748B;
        border-radius: 6px;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: bold;
        box-shadow: 0 4px 6px -1px rgba(30, 64, 175, 0.2);
    }

    /* Dataframe styling */
    .dataframe {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
        overflow: hidden;
    }

    .dataframe th {
        background-color: #F1F5F9 !important;
        color: var(--primary-color) !important;
        border-bottom: 2px solid var(--border-color) !important;
        font-weight: 600 !important;
    }

    .dataframe td {
        color: var(--text-color) !important;
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #F1F5F9 !important;
    }

    .dataframe tr:nth-child(even) td {
        background-color: var(--bg-color) !important;
    }

    /* Message styling */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid var(--success-color);
        color: #047857;
        border-radius: 8px;
    }

    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid var(--error-color);
        color: #B91C1C;
        border-radius: 8px;
    }

    .stWarning {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid var(--warning-color);
        color: #B45309;
        border-radius: 8px;
    }

    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid var(--info-color);
        color: #1D4ED8;
        border-radius: 8px;
    }

    /* Custom containers */
    .highlight-container {
        background-color: #FFFFFF;
        border-left: 4px solid var(--error-color);
        padding: 20px;
        margin: 15px 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: var(--text-color);
    }

    .results-container {
        background-color: #FFFFFF;
        border-left: 4px solid var(--success-color);
        padding: 20px;
        margin: 15px 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: var(--text-color);
    }

    /* Input label styling */
    .stSelectbox label, .stRadio label, .stTextInput label,
    .stNumberInput label, .stFileUploader label {
        color: var(--text-color) !important;
        font-weight: 600;
    }

    /* Top Navbar Status Banner */
    .astina-top-navbar {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        color: #F8FAFC;
    }

    .nav-brand-section {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .nav-brand-logo {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #0F172A;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 4px 8px;
        border-radius: 6px;
        letter-spacing: 0.5px;
    }

    .nav-brand-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #FFFFFF;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .nav-pills-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }

    .pill-success {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.3);
    }

    .pill-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border-color: rgba(245, 158, 11, 0.3);
    }

    .pill-info {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border-color: rgba(59, 130, 246, 0.3);
    }

    .pill-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border-color: rgba(168, 85, 247, 0.3);
    }

    .pill-neutral {
        background: rgba(148, 163, 184, 0.12);
        color: #94A3B8;
        border-color: rgba(148, 163, 184, 0.2);
    }

    .live-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }

    .live-dot-green {
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }

    .live-dot-amber {
        background-color: #F59E0B;
        box-shadow: 0 0 8px #F59E0B;
    }

    .live-dot-blue {
        background-color: #3B82F6;
        box-shadow: 0 0 8px #3B82F6;
    }

    /* Pipeline Step Tracker */
    .pipeline-bar-wrapper {
        width: 100%;
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 8px 14px;
        margin-top: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
    }

    .step-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        color: #64748B;
    }

    .step-item.active {
        color: #38BDF8;
        font-weight: 700;
    }

    .step-item.done {
        color: #34D399;
    }

    .step-badge {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.68rem;
        font-weight: 700;
        background: rgba(100, 116, 139, 0.3);
        color: #94A3B8;
    }

    .step-item.active .step-badge {
        background: #0284C7;
        color: #FFFFFF;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
    }

    .step-item.done .step-badge {
        background: #059669;
        color: #FFFFFF;
    }

    .step-arrow {
        color: rgba(148, 163, 184, 0.3);
        font-size: 0.7rem;
    }

    /* Sidebar Status Card */
    .sidebar-status-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
        color: #E2E8F0;
    }

    .sidebar-status-header {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        font-size: 0.78rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .status-row:last-child {
        border-bottom: none;
    }

    .status-label {
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .status-val {
        font-weight: 600;
        color: #F1F5F9;
    }
</style>
""", unsafe_allow_html=True)

def custom_container(content, container_type="highlight"):
    """Create a custom styled container for content"""
    if container_type == "highlight":
        st.markdown(f'<div class="highlight-container">{content}</div>', unsafe_allow_html=True)
    elif container_type == "results":
        st.markdown(f'<div class="results-container">{content}</div>', unsafe_allow_html=True)

def render_top_navbar_status():
    """
    Renders an informative, responsive top navbar status bar across all application pages.
    Displays live dataset status, trained model info, hardware engine accelerator,
    Copilot status, and an interactive 5-stage pipeline tracker.
    """
    current_page = st.session_state.get('page', 'home')
    
    # 1. Dataset Status
    has_data = 'df_processed_path' in st.session_state
    if has_data:
        meta = st.session_state.get('preprocessing_metadata', {})
        total_rows = meta.get('total_rows_processed') or st.session_state.get('raw_data_total_rows', 0)
        feature_cols = st.session_state.get('feature_columns', [])
        data_text = f"Data: {total_rows:,} Baris | {len(feature_cols)} Fitur" if total_rows > 0 else f"Data: {len(feature_cols)} Fitur Siap"
        data_pill = f'<span class="status-pill pill-success"><span class="live-dot live-dot-green"></span> 📊 {data_text}</span>'
    else:
        data_pill = '<span class="status-pill pill-neutral"><span class="live-dot live-dot-amber"></span> 📂 Data: Menunggu Upload</span>'

    # 2. Model Status
    is_trained = st.session_state.get('model_trained', False)
    if is_trained:
        trained_det = st.session_state.get('trained_detector')
        algo_name = getattr(trained_det, 'best_algorithm', None) or st.session_state.get('selected_model_type', 'Ensemble ML')
        if isinstance(algo_name, str):
            algo_name = algo_name.replace('_', ' ').title()
        model_pill = f'<span class="status-pill pill-success"><span class="live-dot live-dot-green"></span> 🎯 Model: {algo_name} (Siap)</span>'
    else:
        model_pill = '<span class="status-pill pill-warning"><span class="live-dot live-dot-amber"></span> 🎯 Model: Belum Dilatih</span>'

    # 3. Hardware / Compute Engine
    try:
        from ui.utils import get_gpu_status
        gpu_info = get_gpu_status()
        if gpu_info.get('cuda_available', False):
            gpu_name = gpu_info.get('device_name', 'CUDA GPU')
            vram = gpu_info.get('total_memory', 0)
            engine_pill = f'<span class="status-pill pill-purple">🚀 GPU: {gpu_name} ({vram:.1f}GB)</span>'
        else:
            engine_pill = '<span class="status-pill pill-info">💻 Engine: CPU Multi-core (Polars Lazy)</span>'
    except Exception:
        engine_pill = '<span class="status-pill pill-info">💻 Engine: CPU Multithreaded</span>'

    # 4. Agentic Copilot & Audit
    copilot_pill = '<span class="status-pill pill-primary" style="background: rgba(30, 64, 175, 0.2); color: #93C5FD; border-color: rgba(59, 130, 246, 0.3);">🧠 Copilot & RAG: Aktif</span>'
    audit_pill = '<span class="status-pill pill-neutral" style="font-size: 0.72rem;">🛡️ Audit Trail ON</span>'

    # 5. Pipeline Stages & Progress
    stages = [
        ('collect', '1', 'Unggah Data'),
        ('feature', '2', 'Praproses & Fitur'),
        ('train', '3', 'Pelatihan Model'),
        ('evaluate', '4', 'Evaluasi'),
        ('detect', '5', 'Deteksi Anomali'),
    ]

    # Map current_page to stage index
    stage_order = {'home': 0, 'collect': 1, 'train': 3, 'evaluate': 4, 'detect': 5, 'status': 0}
    current_idx = stage_order.get(current_page, 0)
    if has_data and current_idx == 1:
        current_idx = 2  # data uploaded -> at feature step

    steps_html = []
    for key, num, label in stages:
        step_num_int = int(num)
        if step_num_int < current_idx or (step_num_int == 1 and has_data) or (step_num_int == 2 and has_data) or (step_num_int == 3 and is_trained):
            status_class = "done"
            badge_content = "✓"
        elif step_num_int == current_idx:
            status_class = "active"
            badge_content = num
        else:
            status_class = "pending"
            badge_content = num

        step_html = f'<div class="step-item {status_class}"><span class="step-badge">{badge_content}</span><span>{label}</span></div>'
        steps_html.append(step_html)

    pipeline_tracker_html = '<span class="step-arrow">➔</span>'.join(steps_html)

    navbar_html = (
        '<div class="astina-top-navbar">'
        '<div class="nav-brand-section">'
        '<span class="nav-brand-logo">ASTINA</span>'
        '<span class="nav-brand-title">Analisis Transaksi & Fraud Intel</span>'
        '</div>'
        f'<div class="nav-pills-container">{data_pill} {model_pill} {engine_pill} {copilot_pill} {audit_pill}</div>'
        f'<div class="pipeline-bar-wrapper">{pipeline_tracker_html}</div>'
        '</div>'
    )
    st.markdown(navbar_html, unsafe_allow_html=True)

def render_footer():
    """
    Renders standard application footer with copyright notice at the bottom of every page.
    """
    footer_html = """
    <div style="margin-top: 60px; padding-top: 24px; padding-bottom: 24px; border-top: 1px solid rgba(226, 232, 240, 0.8); text-align: center; color: #64748B; font-size: 0.82rem;">
        <div style="margin-bottom: 6px; font-weight: 600; color: #334155; letter-spacing: 0.3px;">
            🛡️ ASTINA — Analisis Sistem Transaksi Identifikasi Nilai Anomali
        </div>
        <div style="font-weight: 500;">
            copyright@2026 TIM ASTINA INDONESIA. All rights reserved.
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


