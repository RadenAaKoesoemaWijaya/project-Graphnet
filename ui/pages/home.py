import streamlit as st
import os
from state_manager import navigate_to_page

def persisted_model_artifacts_exist():
    """Check whether a persisted detector can be loaded from disk."""
    MODEL_PREFIX = "models/fraud_detector"
    required_files = [
        f"{MODEL_PREFIX}_params.json",
        f"{MODEL_PREFIX}_scaler.pkl"
    ]
    model_files = [
        f"{MODEL_PREFIX}_isolation_forest.pkl",
        f"{MODEL_PREFIX}_autoencoder.pt",
        f"{MODEL_PREFIX}_xgboost.pkl",
        f"{MODEL_PREFIX}_dbscan.pkl",
        f"{MODEL_PREFIX}_gnn.pt"
    ]
    return all(os.path.exists(path) for path in required_files) and any(
        os.path.exists(path) for path in model_files
    )

def show_home_page():
    st.title('ASTINA - Fraud Intelligence & Anomaly Detection')
    st.info("Mulai dari Unggah Data untuk memvalidasi dan menyiapkan dataset. Setelah itu latih model, evaluasi hasilnya, lalu jalankan Deteksi untuk menemukan klaim yang perlu ditinjau.")
    st.markdown("""
    ## 🛡️ Selamat Datang di ASTINA

    **ASTINA** adalah platform analitik risiko klaim untuk asuransi kesehatan yang menggabungkan **machine learning**, **business rule fraud detection**, dan **dashboard eksekutif** dalam satu alur kerja yang terintegrasi.

    Platform ini membantu tim fraud, audit, dan operasi klaim mengidentifikasi pola seperti **repeat billing**, **phantom service**, **upcoding**, **inflated bill**, **prolonged stay**, dan **medication/device fraud** dengan lebih cepat dan terdokumentasi.

    ### 🚀 Alur kerja utama

    1. **Unggah dan validasi data klaim** dalam format CSV/Parquet/Excel.
    2. **Praproses otomatis** untuk data besar, imputasi, validasi, dan feature alignment.
    3. **Pelatihan model hibrida** berbasis anomaly detection.
    4. **Deteksi risiko bisnis** lewat rule engine dan fuzzy matching.
    5. **Dashboard review** dengan executive summary, filter kategori, dan detail claim per klaim.
    6. **Export dan audit follow-up** untuk kebutuhan review manual atau operasional.

    ### ✨ Nilai bisnis ASTINA
    - **Prioritas risiko lebih jelas** untuk analisis claim yang memerlukan review cepat.
    - **Deteksi lebih kuat** karena menggabungkan model ML dan logika domain bisnis.
    - **Dashboard lebih siap stakeholder** untuk presentasi management dan audit.
    - **Workflow lebih terstruktur** untuk fraud analyst dan tim operasi.

    ### 🧩 Modul yang didukung
    - Repeat billing
    - Phantom service
    - Provider capacity validation
    - Duplicate payment / claim status validation
    - Upcoding dan unbundling
    - Inflated bill / cloning
    - Prolonged stay / readmission
    - Medication dan device fraud
    """)

    st.markdown("""
    <div style="margin-top: 2.2rem; margin-bottom: 1.2rem;">
        <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); padding: 4px 12px; border-radius: 9999px; font-size: 0.76rem; font-weight: 700; color: #1D4ED8; text-transform: uppercase; letter-spacing: 0.5px;">
            🧭 Alur Kerja Terintegrasi
        </div>
        <h3 style="margin-top: 10px; margin-bottom: 4px; font-size: 1.45rem; color: #0F172A; font-weight: 700;">Apa yang ingin Anda lakukan?</h3>
        <p style="color: #64748B; font-size: 0.92rem; margin-top: 0;">Pilih salah satu tahapan navigasi di bawah untuk memulai pemrosesan, pelatihan, atau investigasi klaim.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tampilkan opsi navigasi dengan kartu modern terstruktur
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="astina-action-card">
            <div class="action-card-top">
                <div class="action-card-icon-box">📂</div>
                <span class="action-card-badge">Tahap 01</span>
            </div>
            <div class="action-card-title">Unggah Data</div>
            <p class="action-card-desc">Upload dataset asuransi (CSV, Parquet, Excel), validasi skema 14 kolom, dan praproses otomatis.</p>
        </div>
        """, unsafe_allow_html=True)
        collect_button = st.button("🚀 Mulai Unggah", key="collect", use_container_width=True, type="primary")

    with col2:
        st.markdown("""
        <div class="astina-action-card card-accent-purple">
            <div class="action-card-top">
                <div class="action-card-icon-box icon-box-purple">🧠</div>
                <span class="action-card-badge badge-purple">Tahap 02</span>
            </div>
            <div class="action-card-title">Pelatihan Model</div>
            <p class="action-card-desc">Latih model ML Ensemble & Graph Neural Network (GNN) dengan profil komputasi adaptif.</p>
        </div>
        """, unsafe_allow_html=True)
        train_button = st.button("⚡ Latih Model", key="train", use_container_width=True)

    with col3:
        st.markdown("""
        <div class="astina-action-card card-accent-emerald">
            <div class="action-card-top">
                <div class="action-card-icon-box icon-box-emerald">🔍</div>
                <span class="action-card-badge badge-emerald">Tahap 03</span>
            </div>
            <div class="action-card-title">Deteksi & Review</div>
            <p class="action-card-desc">Jalankan scoring risiko hybrid, audit 9 aturan bisnis, dan investigasi via AI Copilot.</p>
        </div>
        """, unsafe_allow_html=True)
        detect_button = st.button("🎯 Deteksi Anomali", key="detect", use_container_width=True)

    with col4:
        st.markdown("""
        <div class="astina-action-card card-accent-slate">
            <div class="action-card-top">
                <div class="action-card-icon-box icon-box-slate">📊</div>
                <span class="action-card-badge badge-slate">Sistem</span>
            </div>
            <div class="action-card-title">Status Sistem</div>
            <p class="action-card-desc">Pantau kesehatan server, telemetri CPU/GPU, manajemen cache, dan audit trail SHA-256.</p>
        </div>
        """, unsafe_allow_html=True)
        status_button = st.button("📈 Cek Status", key="status", use_container_width=True)
    
    if collect_button:
        navigate_to_page('collect')
    
    if train_button:
        navigate_to_page('train')
    
    if status_button:
        navigate_to_page('status')
    
    if detect_button:
        # Cek apakah model tersimpan sudah ada
        models_exist = persisted_model_artifacts_exist()
        has_data = (
            'df_processed_path' in st.session_state
            or 'processed_data_hash' in st.session_state
        )
        
        if models_exist:
            navigate_to_page('detect')
        else:
            # Tampilkan pesan peringatan dengan opsi
            st.warning("⚠️ Persyaratan untuk deteksi anomali belum terpenuhi!")
            
            if not models_exist:
                st.info("🔧 Model belum dilatih. Anda perlu melatih model terlebih dahulu.")
                if st.button("Lanjut ke Pelatihan Model", key="train_from_detect"):
                    navigate_to_page('train')
            
            if not has_data:
                st.info("📊 Data transaksi belum diproses. Anda perlu mengunggah dan memproses data transaksi terlebih dahulu.")
                if st.button("Mulai dari Unggah Data Transaksi", key="collect_from_detect"):
                    navigate_to_page('collect')
            
            # Jika salah satu sudah ada, beri opsi yang sesuai
            if has_data and not models_exist:
                st.info("🎯 Gunakan data transaksi yang sudah diproses untuk melatih model")
                if st.button("Latih Model dengan Data Tersedia", key="train_with_existing"):
                    navigate_to_page('train')
    
    # Tampilkan informasi tambahan tentang aplikasi
    st.markdown("""
    ---
    ### Tentang Aplikasi Ini
    
    Aplikasi ini membantu Anda menganalisis transaksi asuransi, melakukan eksplorasi data, dan mendeteksi anomali menggunakan kombinasi beberapa algoritma pembelajaran mesin:
    
    🔍 **Isolation Forest**: Mendeteksi anomali dengan mengisolasi data yang berbeda secara statistik.
    
    🧠 **Autoencoder**: Mempelajari pola normal melalui pembelajaran mendalam dan mendeteksi deviasi melalui galat rekonstruksi.
    
    🚀 **XGBoost/LightGBM**: Algoritma gradient boosting untuk mengidentifikasi pola anomali kompleks pada data tabular.
    
    🕸️ **Graph Neural Network (GNN)**: Menganalisis fitur transaksi sekaligus struktur hubungan antar entitas untuk mendeteksi anomali berbasis jaringan.
    
    ### Metodologi
    
    Aplikasi ini menggunakan pendekatan **Hybrid Ensemble** yang menggabungkan kekuatan keempat algoritma tersebut:
    - **Isolation Forest** efektif untuk deteksi outlier awal.
    - **Autoencoder** menangkap anomali non-linear yang kompleks.
    - **XGBoost/LightGBM** memberikan prediksi berbasis pohon keputusan yang sangat akurat.
    - **GNN** memberikan dimensi baru dalam deteksi anomali dengan memetakan hubungan antar transaksi.
    
    Seluruh model dioptimalkan menggunakan **Optuna** untuk mencari konfigurasi terbaik, dan hasilnya dikombinasikan dengan bobot dinamis untuk memberikan skor anomali yang paling akurat dan dapat dipertanggungjawabkan.
    """)
