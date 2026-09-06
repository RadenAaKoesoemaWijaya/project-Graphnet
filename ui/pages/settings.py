import streamlit as st
import os
from auth_manager import AuthManager
from state_manager import navigate_to_page
from ui.utils import get_gpu_status

def show_settings_page():
    """Display comprehensive settings page for LLM, Copilot, and system configuration"""
    
    # Check authentication
    current_user = AuthManager.get_current_user()
    user_role = current_user.get('role', 'viewer')
    
    # Page header
    st.title("⚙️ Pengaturan Sistem")
    st.markdown("Konfigurasi LLM, Agentic Copilot, dan parameter sistem ASTINA")
    
    # Role-based access warning
    if user_role == 'viewer':
        st.warning("⚠️ Anda login sebagai **Viewer**. Beberapa pengaturan mungkin tidak dapat dimodifikasi.")
    
    # Settings tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔌 LLM & Copilot", 
        "🗄️ Model Registry", 
        "📊 System Configuration", 
        "🔐 Security & Privacy"
    ])
    
    # TAB 1: LLM & Copilot Configuration
    with tab1:
        st.markdown("### 🔌 Konfigurasi LLM & Agentic Copilot")
        st.info("Konfigurasi ini akan berlaku untuk sesi ini. Untuk pengaturan permanen, gunakan environment variables.")
        
        # Provider Selection
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            provider_options = {
                "🧠 Heuristic Engine (Offline)": "heuristic",
                "🔵 Google Gemini": "gemini", 
                "🟢 OpenAI / Compatible": "openai",
                "🟠 Local Ollama": "ollama"
            }
            
            # Initialize session state if not exists
            if 'settings_provider_sel' not in st.session_state:
                st.session_state['settings_provider_sel'] = "🧠 Heuristic Engine (Offline)"
            
            provider_choice = st.selectbox(
                "LLM Engine Provider:",
                list(provider_options.keys()),
                index=list(provider_options.keys()).index(st.session_state['settings_provider_sel']),
                key="settings_provider",
                help="Pilih provider LLM untuk Agentic Copilot. Heuristic Engine bekerja offline tanpa API key."
            )
            st.session_state['settings_provider_sel'] = provider_choice
        
        with col2:
            if "Gemini" in provider_choice or "OpenAI" in provider_choice:
                if 'settings_api_key_val' not in st.session_state:
                    st.session_state['settings_api_key_val'] = ""
                
                api_key_input = st.text_input(
                    "API Key:",
                    value=st.session_state['settings_api_key_val'],
                    type="password",
                    key="settings_apikey",
                    help="Masukkan API key dari provider. Key akan disimpan di session state."
                )
                st.session_state['settings_api_key_val'] = api_key_input
                
                # API Key help text
                if "Gemini" in provider_choice:
                    st.markdown("""
                    <div style='padding:8px 12px;background:#eff6ff;border-left:3px solid #3b82f6;font-size:0.8rem;color:#1e40af;'>
                    💡 <b>Dapatkan API Key Google Gemini:</b><br>
                    1. Kunjungi <a href='https://aistudio.google.com/app/apikey' target='_blank'>Google AI Studio</a><br>
                    2. Create new API key<br>
                    3. Copy dan paste di atas
                    </div>
                    """, unsafe_allow_html=True)
                elif "OpenAI" in provider_choice:
                    st.markdown("""
                    <div style='padding:8px 12px;background:#eff6ff;border-left:3px solid #3b82f6;font-size:0.8rem;color:#1e40af;'>
                    💡 <b>Dapatkan API Key OpenAI:</b><br>
                    1. Kunjungi <a href='https://platform.openai.com/api-keys' target='_blank'>OpenAI Platform</a><br>
                    2. Create new secret key<br>
                    3. Copy dan paste di atas
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='padding:28px 12px;background:#f0fdf4;border-left:3px solid #22c55e;font-size:0.8rem;color:#166534;'>
                ✅ <b>Mode Offline:</b> Heuristic Engine / Local Ollama tidak memerlukan API key eksternal.
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'settings_auditor_val' not in st.session_state:
                st.session_state['settings_auditor_val'] = "Investigator Senior ASTINA"
            
            auditor_name = st.text_input(
                "Nama Auditor:",
                value=st.session_state['settings_auditor_val'],
                key="settings_auditor",
                help="Nama yang akan muncul di Berita Acara Pemeriksaan (BAP)"
            )
            st.session_state['settings_auditor_val'] = auditor_name
        
        # Advanced Configuration
        st.markdown("---")
        st.markdown("### 🔧 Konfigurasi Lanjutan")
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            if "Gemini" in provider_choice:
                if 'settings_gemini_model' not in st.session_state:
                    st.session_state['settings_gemini_model'] = "gemini-1.5-flash"
                
                model_choice = st.selectbox(
                    "Model Gemini:",
                    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
                    index=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"].index(
                        st.session_state['settings_gemini_model']
                    ),
                    key="settings_gemini_model",
                    help="Pilih model Gemini. Flash lebih cepat, Pro lebih akurat."
                )
                st.session_state['settings_gemini_model'] = model_choice
                
            elif "OpenAI" in provider_choice:
                if 'settings_openai_model' not in st.session_state:
                    st.session_state['settings_openai_model'] = "gpt-4o-mini"
                
                model_choice = st.text_input(
                    "Model Name:",
                    value=st.session_state['settings_openai_model'],
                    key="settings_openai_model",
                    help="Contoh: gpt-4o-mini, gpt-4o, gpt-3.5-turbo"
                )
                st.session_state['settings_openai_model'] = model_choice
                
            elif "Ollama" in provider_choice:
                if 'settings_ollama_model' not in st.session_state:
                    st.session_state['settings_ollama_model'] = "llama3"
                
                model_choice = st.text_input(
                    "Ollama Model Name:",
                    value=st.session_state['settings_ollama_model'],
                    key="settings_ollama_model",
                    help="Model yang tersedia di Ollama lokal (contoh: llama3, mistral, codellama)"
                )
                st.session_state['settings_ollama_model'] = model_choice
        
        with col_adv2:
            if "Ollama" in provider_choice:
                if 'settings_ollama_endpoint' not in st.session_state:
                    st.session_state['settings_ollama_endpoint'] = "http://localhost:11434/api/generate"
                
                endpoint_choice = st.text_input(
                    "Ollama Endpoint URL:",
                    value=st.session_state['settings_ollama_endpoint'],
                    key="settings_ollama_endpoint",
                    help="URL endpoint Ollama lokal (default: http://localhost:11434/api/generate)"
                )
                st.session_state['settings_ollama_endpoint'] = endpoint_choice
                
            elif "OpenAI" in provider_choice:
                if 'settings_openai_endpoint' not in st.session_state:
                    st.session_state['settings_openai_endpoint'] = ""
                
                endpoint_choice = st.text_input(
                    "Base URL / Custom Endpoint (Opsional):",
                    value=st.session_state['settings_openai_endpoint'],
                    placeholder="https://api.openai.com/v1/chat/completions",
                    key="settings_openai_endpoint",
                    help="Untuk OpenAI-compatible API atau custom endpoint"
                )
                st.session_state['settings_openai_endpoint'] = endpoint_choice
        
        # Connection Test
        st.markdown("---")
        col_test1, col_test2 = st.columns([1, 4])
        
        with col_test1:
            test_button = st.button("🔌 Test Koneksi", key="settings_test_conn", type="primary")
        
        with col_test2:
            if test_button and "Heuristic" not in provider_choice:
                with st.spinner("Menguji koneksi LLM..."):
                    try:
                        from agentic_copilot import AgenticInvestigatorCopilot
                        
                        provider_map = {
                            "🔵 Google Gemini": "gemini",
                            "🟢 OpenAI / Compatible": "openai", 
                            "🟠 Local Ollama": "ollama"
                        }
                        
                        test_engine = AgenticInvestigatorCopilot(
                            provider=provider_map.get(provider_choice, "heuristic"),
                            api_key=api_key_input if "Gemini" in provider_choice or "OpenAI" in provider_choice else None,
                            model_name=model_choice,
                            endpoint_url=endpoint_choice if endpoint_choice else None
                        )
                        
                        conn_result = test_engine.test_connection()
                        
                        if conn_result["ok"]:
                            st.success(f"✅ {conn_result['message']} — Provider: **{conn_result['provider']}**")
                        else:
                            st.error(f"❌ {conn_result['message']}")
                            st.info("💡 Jika koneksi gagal, sistem akan otomatis menggunakan Heuristic Engine sebagai fallback.")
                    except Exception as e:
                        st.error(f"❌ Error saat testing koneksi: {str(e)}")
            elif test_button and "Heuristic" in provider_choice:
                st.info("✅ Heuristic Engine (Offline) tidak memerlukan koneksi internet.")
        
        # Environment Variables Reference
        st.markdown("---")
        with st.expander("📖 Referensi Environment Variables"):
            st.markdown("""
            ### Environment Variables (Override UI Settings)
            
            Anda dapat menggunakan environment variables untuk pengaturan permanen yang mengoverride konfigurasi UI:
            
            | Variable | Default | Deskripsi |
            |-----------|---------|-----------|
            | `LLM_PROVIDER` | `heuristic` | Provider default (gemini/openai/ollama/heuristic) |
            | `LLM_MODEL_NAME` | `gemini-1.5-flash` | Model name default |
            | `LLM_ENDPOINT_URL` | `http://localhost:11434/api/generate` | Custom endpoint URL |
            | `GEMINI_API_KEY` | - | Google Gemini API Key |
            | `OPENAI_API_KEY` | - | OpenAI API Key |
            
            **Contoh Penggunaan (PowerShell):**
            ```powershell
            $env:GEMINI_API_KEY="your-api-key-here"
            $env:LLM_PROVIDER="gemini"
            python run.py
            ```
            
            **Contoh Penggunaan (Linux/macOS):**
            ```bash
            export GEMINI_API_KEY="your-api-key-here"
            export LLM_PROVIDER="gemini"
            python run.py
            ```
            """)
    
    # TAB 2: Model Registry
    with tab2:
        st.markdown("### 🗄️ Model Registry")
        st.info("Kelola versi model yang tersimpan dan load model untuk digunakan.")
        
        try:
            from model_registry import get_versions, load_model_version
            from ui.utils import load_persisted_detector
            
            versions = get_versions()
            
            if versions:
                st.success(f"✅ Terdapat {len(versions)} versi model tersimpan")
                
                version_df = []
                for v in versions:
                    version_df.append({
                        "Versi": v['version'],
                        "Tanggal": v.get('created_at', '-'),
                        "Algoritma": v.get('algorithm', '-'),
                        "Fitur": v.get('num_features', '-'),
                        "Path": v.get('path', '-')
                    })
                
                st.dataframe(version_df, width='stretch', hide_index=True)
                
                # Load model section
                st.markdown("---")
                st.markdown("### 📥 Load Model")
                
                version_names = [v['version'] for v in versions]
                version_names.reverse()  # Newest first
                
                selected_version = st.selectbox(
                    "Pilih Versi Model untuk Dimuat:",
                    ["-- Pilih Versi --"] + version_names,
                    key="settings_load_model"
                )
                
                if selected_version != "-- Pilih Versi --":
                    col_load1, col_load2 = st.columns(2)
                    with col_load1:
                        if st.button("📥 Muat Model Ini", key="settings_btn_load_model"):
                            with st.spinner("Memuat model..."):
                                MODEL_PREFIX = "models/fraud_detector"
                                if load_model_version(selected_version, MODEL_PREFIX):
                                    loaded_det = load_persisted_detector()
                                    if loaded_det:
                                        st.success(f"✅ Berhasil memuat model: {selected_version}")
                                        st.session_state['model_trained'] = True
                                    else:
                                        st.error("❌ Model korup atau tidak lengkap.")
                                else:
                                    st.error("❌ Gagal memuat versi model.")
                    
                    with col_load2:
                        if st.button("🗑️ Hapus Model Ini", key="settings_btn_delete_model"):
                            st.warning("⚠️ Fitur hapus model belum diimplementasikan.")
            else:
                st.info("ℹ️ Belum ada model yang tersimpan. Latih model terlebih dahulu di halaman Pelatihan.")
                
        except Exception as e:
            st.error(f"❌ Error mengakses model registry: {str(e)}")
    
    # TAB 3: System Configuration
    with tab3:
        st.markdown("### 📊 Konfigurasi Sistem")
        st.info("Konfigurasi parameter sistem dan performa.")
        
        # Hardware Status
        st.markdown("#### 💻 Status Hardware")
        gpu_info = get_gpu_status()
        
        hw_col1, hw_col2, hw_col3 = st.columns(3)
        with hw_col1:
            st.metric("PyTorch Version", gpu_info['torch_version'])
        with hw_col2:
            st.metric("Device", gpu_info['device_name'])
        with hw_col3:
            if gpu_info['cuda_available']:
                st.metric("GPU Status", "✅ Active")
            else:
                st.metric("GPU Status", "❌ Inactive")
        
        if gpu_info['cuda_available']:
            st.info(f"🚀 GPU CUDA Aktif: {gpu_info['device_name']} ({gpu_info['total_memory']:.1f} GB)")
        else:
            st.warning("💻 Menggunakan CPU Mode - Install PyTorch dengan CUDA untuk akselerasi GPU")
        
        st.markdown("---")
        
        # Memory Configuration
        st.markdown("#### 🧠 Konfigurasi Memori")
        
        try:
            from config import MEMORY_LIMIT_MB, CHUNK_SIZE, MAX_FILE_SIZE
            
            mem_col1, mem_col2, mem_col3 = st.columns(3)
            with mem_col1:
                st.metric("Memory Limit", f"{MEMORY_LIMIT_MB} MB")
            with mem_col2:
                st.metric("Chunk Size", f"{CHUNK_SIZE // (1024*1024)} MB")
            with mem_col3:
                st.metric("Max File Size", f"{MAX_FILE_SIZE // (1024*1024*1024)} GB")
            
            st.info("💡 Konfigurasi memori diatur di file `config.py`. Modifikasi memerlukan restart aplikasi.")
            
        except Exception as e:
            st.error(f"❌ Error membaca konfigurasi memori: {str(e)}")
        
        st.markdown("---")
        
        # File Upload Configuration
        st.markdown("#### 📁 Konfigurasi Upload File")
        
        upload_col1, upload_col2 = st.columns(2)
        with upload_col1:
            st.info("📂 Format yang didukung: CSV, Excel (.xlsx, .xls), Parquet")
        with upload_col2:
            st.info("📏 Ukuran maksimal: 3 GB (diproses per chunk)")
    
    # TAB 4: Security & Privacy
    with tab4:
        st.markdown("### 🔐 Security & Privacy")
        st.info("Pengaturan keamanan dan privasi data.")
        
        # Authentication Status
        st.markdown("#### 🔑 Status Autentikasi")
        
        auth_col1, auth_col2, auth_col3 = st.columns(3)
        with auth_col1:
            st.metric("Mode", "Production" if AuthManager.is_auth_enforced() else "Development")
        with auth_col2:
            st.metric("Role", user_role.upper())
        with auth_col3:
            st.metric("User", current_user.get('name', 'Unknown'))
        
        st.markdown("---")
        
        # PII Masking
        st.markdown("#### 🎭 Perlindungan Data Pribadi (PII)")
        
        pii_col1, pii_col2 = st.columns(2)
        with pii_col1:
            st.success("✅ PII Masking Aktif")
            st.caption("Data sensitif (NIK, nama, rekam medis) dilindungi sesuai UU PDP")
        with pii_col2:
            st.success("✅ Audit Trail SHA-256")
            st.caption("Seluruh aktivitas tercatat dengan hash kriptografis")
        
        st.markdown("---")
        
        # Audit Trail Status
        st.markdown("#### 📝 Status Audit Trail")
        
        try:
            from audit_trail import get_audit_trail
            audit_trail = get_audit_trail()
            recent_events = audit_trail.get_recent_events(5)
            
            if recent_events:
                st.success(f"✅ Audit Trail Aktif - {len(recent_events)} event tercatat")
                
                event_df = []
                for event in recent_events:
                    event_df.append({
                        "Waktu": event.get('timestamp', '-'),
                        "Event": event.get('event_type', '-'),
                        "Action": event.get('action', '-'),
                        "User": event.get('user', '-')
                    })
                
                st.dataframe(event_df, width='stretch', hide_index=True)
            else:
                st.info("ℹ️ Belum ada event audit tercatat")
                
        except Exception as e:
            st.error(f"❌ Error mengakses audit trail: {str(e)}")
        
        st.markdown("---")
        
        # Security Recommendations
        st.markdown("#### 🛡️ Rekomendasi Keamanan")
        
        st.markdown("""
        **Untuk Lingkungan Production:**
        
        1. **Aktifkan Mode Autentikasi:**
           ```powershell
           $env:AUTH_ENABLED="true"
           python run.py
           ```
        
        2. **Gunakan Environment Variables untuk API Key:**
           - Jangan hardcode API key di kode
           - Gunakan secret manager untuk deployment
        
        3. **Update Password Default:**
           - Ganti password default untuk semua role
           - Gunakan password yang kuat dan unik
        
        4. **Aktifkan HTTPS:**
           - Gunakan reverse proxy (nginx/Apache)
           - Configure SSL/TLS certificate
        
        5. **Limit Access:**
           - Gunakan firewall untuk membatasi akses
           - Implementasi IP whitelisting jika diperlukan
        """)
    
    # Apply Settings Button
    st.markdown("---")
    col_apply1, col_apply2, col_apply3 = st.columns([1, 1, 2])
    
    with col_apply1:
        if st.button("💾 Simpan Konfigurasi", key="settings_save", type="primary"):
            # Sync settings to session state for detection page
            st.session_state['copilot_provider_sel'] = st.session_state.get('settings_provider_sel', "🧠 Heuristic Engine (Offline)")
            st.session_state['copilot_api_key_val'] = st.session_state.get('settings_api_key_val', "")
            st.session_state['copilot_auditor_val'] = st.session_state.get('settings_auditor_val', "Investigator Senior ASTINA")
            
            # Model specific settings
            if "Gemini" in st.session_state.get('settings_provider_sel', ""):
                st.session_state['copilot_model_val'] = st.session_state.get('settings_gemini_model', "gemini-1.5-flash")
            elif "OpenAI" in st.session_state.get('settings_provider_sel', ""):
                st.session_state['copilot_model_val'] = st.session_state.get('settings_openai_model', "gpt-4o-mini")
            elif "Ollama" in st.session_state.get('settings_provider_sel', ""):
                st.session_state['copilot_model_val'] = st.session_state.get('settings_ollama_model', "llama3")
                st.session_state['copilot_endpoint_val'] = st.session_state.get('settings_ollama_endpoint', "http://localhost:11434/api/generate")
            
            st.success("✅ Konfigurasi berhasil disimpan dan disinkronkan ke halaman Deteksi!")
    
    with col_apply2:
        if st.button("🔄 Reset ke Default", key="settings_reset"):
            # Reset settings to defaults
            st.session_state['settings_provider_sel'] = "🧠 Heuristic Engine (Offline)"
            st.session_state['settings_api_key_val'] = ""
            st.session_state['settings_auditor_val'] = "Investigator Senior ASTINA"
            st.session_state['settings_gemini_model'] = "gemini-1.5-flash"
            st.session_state['settings_openai_model'] = "gpt-4o-mini"
            st.session_state['settings_ollama_model'] = "llama3"
            st.session_state['settings_ollama_endpoint'] = "http://localhost:11434/api/generate"
            
            st.success("✅ Konfigurasi di-reset ke default!")
            st.rerun()
    
    with col_apply3:
        st.caption("💡 Konfigurasi disimpan di session state dan akan berlaku selama sesi aktif.")