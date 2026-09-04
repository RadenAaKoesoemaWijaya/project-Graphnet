import streamlit as st
import os
import pandas as pd
from state_manager import navigate_to_page
from ui.utils import get_gpu_status, get_gpu_status_display
from auth_manager import AuthManager

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
    current_user = AuthManager.get_current_user()
    user_name = current_user.get('name', 'Analyst')
    user_role = current_user.get('role', 'viewer').upper()

    # --- 1. HERO HEADER ---
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #1E3A8A 100%); padding: 28px 26px; border-radius: 16px; margin-bottom: 22px; color: #FFFFFF; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(147, 197, 253, 0.35); padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; color: #93C5FD; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;">
                    🛡️ Fraud Intelligence & Graph Analytics Platform
                </div>
                <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.5px;">ASTINA</h1>
                <p style="margin: 6px 0 0 0; font-size: 1.0rem; color: #CBD5E1; max-width: 720px; line-height: 1.5;">
                    Platform analitik risiko klaim asuransi kesehatan berbasis <strong>Hybrid Machine Learning</strong>, <strong>Graph Neural Network (GNN)</strong>, dan <strong>9 Mesin Aturan Kecurangan Bisnis</strong> terintegrasi.
                </p>
            </div>
            <div style="background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 10px 16px; text-align: right; min-width: 170px;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Sesi Pengguna</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">👤 {user_name}</div>
                <div style="display: inline-block; background: #3B82F6; color: #FFFFFF; font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; margin-top: 4px;">
                    {user_role}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 2. REAL-TIME PIPELINE READINESS BAROMETER ---
    has_data = (
        'df_processed_path' in st.session_state
        or 'processed_data_hash' in st.session_state
    )
    meta = st.session_state.get('preprocessing_metadata', {})
    total_rows = meta.get('total_rows_processed') or st.session_state.get('raw_data_total_rows', 0)
    feature_cols = st.session_state.get('feature_columns', [])

    models_exist = persisted_model_artifacts_exist() or st.session_state.get('model_trained', False)
    trained_det = st.session_state.get('trained_detector')
    model_name = getattr(trained_det, 'best_algorithm', None) or st.session_state.get('selected_model_type', 'Hybrid Ensemble')
    if isinstance(model_name, str):
        model_name = model_name.replace('_', ' ').title()

    gpu_info = get_gpu_status()
    hw_icon = "🚀" if gpu_info.get('cuda_available') else "💻"
    hw_label = "GPU CUDA Aktif" if gpu_info.get('cuda_available') else "Multi-core CPU"
    hw_sub = gpu_info.get('device_name', 'Default CPU') if gpu_info.get('cuda_available') else "Polars Multi-thread"

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

    with b_col1:
        if has_data:
            st.metric("📁 Dataset Klaim", f"{total_rows:,} Baris" if total_rows > 0 else "Siap", delta=f"{len(feature_cols)} Fitur" if feature_cols else "Normal")
        else:
            st.metric("📁 Dataset Klaim", "Belum Diunggah", delta="Menunggu Ingest", delta_color="off")

    with b_col2:
        if models_exist:
            st.metric("🧠 Model Deteksi", model_name[:18], delta="Siap Skoring")
        else:
            st.metric("🧠 Model Deteksi", "Belum Dilatih", delta="Perlu Training", delta_color="off")

    with b_col3:
        st.metric(f"{hw_icon} Akselerasi Komputasi", hw_label, delta=hw_sub[:18])

    with b_col4:
        st.metric("🛡️ Integritas Audit", "SHA-256 Aktif", delta="Tamper-proof")

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # --- 3. 5-STAGE INTEGRATED ACTION WORKFLOW ---
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); padding: 3px 10px; border-radius: 9999px; font-size: 0.74rem; font-weight: 700; color: #1D4ED8; text-transform: uppercase; letter-spacing: 0.5px;">
            🧭 Alur Kerja Operasional Klaim
        </div>
        <h3 style="margin-top: 8px; margin-bottom: 4px; font-size: 1.35rem; color: #0F172A; font-weight: 700;">Tahapan Navigasi Terintegrasi</h3>
        <p style="color: #64748B; font-size: 0.88rem; margin: 0;">Jalankan tahapan sistem secara berurutan untuk menghasilkan skor risiko klaim yang akurat, terverifikasi, dan siap diaudit.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("""
        <div class="astina-action-card">
            <div class="action-card-top">
                <div class="action-card-icon-box">📂</div>
                <span class="action-card-badge">Tahap 01</span>
            </div>
            <div class="action-card-title">Unggah Data</div>
            <p class="action-card-desc">Ingest dataset klaim (CSV, Parquet, Excel hingga 3GB), evaluasi skema 14 kolom inti, dan praproses fitur otomatis.</p>
        </div>
        """, unsafe_allow_html=True)
        collect_button = st.button("🚀 Mulai Unggah", key="home_collect", use_container_width=True, type="primary")

    with col2:
        st.markdown("""
        <div class="astina-action-card card-accent-purple">
            <div class="action-card-top">
                <div class="action-card-icon-box icon-box-purple">🧠</div>
                <span class="action-card-badge badge-purple">Tahap 02</span>
            </div>
            <div class="action-card-title">Pelatihan Model</div>
            <p class="action-card-desc">Latih Hybrid Ensemble (Isolation Forest, Deep Autoencoder, XGBoost/LightGBM) & Graph Neural Network (GNN).</p>
        </div>
        """, unsafe_allow_html=True)
        train_button = st.button("⚡ Latih Model", key="home_train", use_container_width=True)

    with col3:
        st.markdown("""
        <div class="astina-action-card card-accent-amber">
            <div class="action-card-top">
                <div class="action-card-icon-box icon-box-amber">📊</div>
                <span class="action-card-badge badge-amber">Tahap 03</span>
            </div>
            <div class="action-card-title">Evaluasi Model</div>
            <p class="action-card-desc">Validasi performa model melalui ROC-AUC, Precision-Recall, Confusion Matrix, dan kalibrasi ambang batas risiko.</p>
        </div>
        """, unsafe_allow_html=True)
        evaluate_button = st.button("📈 Evaluasi Model", key="home_eval", use_container_width=True)

    with col4:
        st.markdown("""
        <div class="astina-action-card card-accent-emerald">
            <div class="action-card-top">
                <div class="action-card-icon-box icon-box-emerald">🎯</div>
                <span class="action-card-badge badge-emerald">Tahap 04</span>
            </div>
            <div class="action-card-title">Deteksi & Review</div>
            <p class="action-card-desc">Eksekusi skoring risiko hybrid, audit 9 aturan kecurangan bisnis, serta investigasi terpandu via AI Copilot.</p>
        </div>
        """, unsafe_allow_html=True)
        detect_button = st.button("🔍 Deteksi Anomali", key="home_detect", use_container_width=True)

    with col5:
        st.markdown("""
        <div class="astina-action-card card-accent-slate">
            <div class="action-card-top">
                <div class="action-card-icon-box icon-box-slate">🖥️</div>
                <span class="action-card-badge badge-slate">Sistem</span>
            </div>
            <div class="action-card-title">Status Sistem</div>
            <p class="action-card-desc">Pantau kesehatan runtime, telemetri RAM/GPU, manajemen cache, dan log audit trail transaksi berkriptografi SHA-256.</p>
        </div>
        """, unsafe_allow_html=True)
        status_button = st.button("⚙️ Cek Status", key="home_status", use_container_width=True)

    # Action Handlers
    if collect_button:
        navigate_to_page('collect')

    if train_button:
        navigate_to_page('train')

    if evaluate_button:
        if models_exist:
            navigate_to_page('evaluate')
        else:
            st.warning("⚠️ Model belum tersedia. Anda perlu melatih model di Tahap 02 terlebih dahulu sebelum melakukan evaluasi.")
            if st.button("➡️ Menuju Halaman Pelatihan Model", key="jump_train_from_eval"):
                navigate_to_page('train')

    if status_button:
        navigate_to_page('status')

    if detect_button:
        if models_exist:
            navigate_to_page('detect')
        else:
            st.warning("⚠️ Persyaratan untuk deteksi anomali belum terpenuhi!")
            st.info("🔧 Model belum dilatih atau dimuat dari registry. Silakan unggah data dan latih model terlebih dahulu.")
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                if st.button("📂 Unggah Data Transaksi", key="btn_collect_fallback", use_container_width=True):
                    navigate_to_page('collect')
            with sub_col2:
                if st.button("🧠 Latih Model AI", key="btn_train_fallback", use_container_width=True):
                    navigate_to_page('train')

    st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)

    # --- 4. INTERACTIVE KNOWLEDGE BASE & OPERATIONAL REFERENCE ---
    st.markdown("### 📚 Pusat Informasi & Panduan Operasional ASTINA")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📑 Panduan Skema 14 Kolom",
        "🛡️ Kamus 9 Aturan Kecurangan",
        "👥 Matriks Peran & Tindak Lanjut",
        "🧬 Arsitektur Hibrida AI"
    ])

    # --- TAB 1: 14 COLUMNS DATA DICTIONARY ---
    with tab1:
        st.markdown("""
        Sistem ASTINA memerlukan **14 kolom standar** untuk menjamin seluruh modul machine learning, topologi graf GNN, dan 9 mesin aturan bisnis bekerja dengan akurasi 100%.
        """)

        schema_data = [
            {"No": 1, "Nama Kolom": "claim_id", "Tipe": "String / Int", "Contoh": "CLM-01001", "Fungsi Utama": "Kunci unik klaim, audit trail SHA-256, pelacakan riwayat."},
            {"No": 2, "Nama Kolom": "patient_id", "Tipe": "String / Int", "Contoh": "PAT-00201", "Fungsi Utama": "Identitas peserta, deteksi repeat billing, node graf GNN."},
            {"No": 3, "Nama Kolom": "provider_id", "Tipe": "String / Int", "Contoh": "PROV-00011", "Fungsi Utama": "Identitas faskes/dokter, audit kapasitas, node relasi GNN."},
            {"No": 4, "Nama Kolom": "service_code", "Tipe": "String", "Contoh": "99213 / CPT", "Fungsi Utama": "Kode tindakan medis, audit unbundling & phantom service."},
            {"No": 5, "Nama Kolom": "diagnosis_code", "Tipe": "String", "Contoh": "J06.9, E11.9", "Fungsi Utama": "Kode ICD-10, validasi keselarasan klinis diagnosis-tindakan."},
            {"No": 6, "Nama Kolom": "billing_date", "Tipe": "Date (YYYY-MM-DD)", "Contoh": "2024-01-15", "Fungsi Utama": "Tanggal penagihan, deteksi repeat billing temporal < 30 hari."},
            {"No": 7, "Nama Kolom": "service_date", "Tipe": "Date (YYYY-MM-DD)", "Contoh": "2024-01-10", "Fungsi Utama": "Tanggal pelayanan, audit batas kapasitas harian faskes."},
            {"No": 8, "Nama Kolom": "billed_amount", "Tipe": "Float / Numerik", "Contoh": "15000000.0", "Fungsi Utama": "Nominal klaim diajukan, fitur dasar anomaly detection."},
            {"No": 9, "Nama Kolom": "paid_amount", "Tipe": "Float / Numerik", "Contoh": "12000000.0", "Fungsi Utama": "Nominal disetujui, perhitungan payment_ratio & inflated bill."},
            {"No": 10, "Nama Kolom": "allowed_amount", "Tipe": "Float / Numerik", "Contoh": "13500000.0", "Fungsi Utama": "Batas plafon tarif yang diperkenankan (allowance_ratio)."},
            {"No": 11, "Nama Kolom": "claim_status", "Tipe": "String", "Contoh": "APPROVED / PAID", "Fungsi Utama": "Status verifikasi, pencegahan duplicate payment."},
            {"No": 12, "Nama Kolom": "patient_age", "Tipe": "Integer", "Contoh": "45", "Fungsi Utama": "Usia pasien, feature engineering age_group & anomali demografis."},
            {"No": 13, "Nama Kolom": "length_of_stay", "Tipe": "Integer", "Contoh": "3 (0 jika rawat jalan)", "Fungsi Utama": "Lama hari rawat inap (LOS), deteksi prolonged stay outlier."},
            {"No": 14, "Nama Kolom": "quantity", "Tipe": "Integer", "Contoh": "10", "Fungsi Utama": "Jumlah obat / alkes, deteksi medication & device fraud."}
        ]
        df_schema = pd.DataFrame(schema_data)
        st.dataframe(df_schema, use_container_width=True, hide_index=True)

        st.info("💡 **Spesifikasi Engine Ingest**: ASTINA mendukung file hingga **3 GB** dalam format `.csv`, `.parquet`, `.xlsx`, dan `.json` dengan pemrosesan streaming non-blocking menggunakan engine Polars LazyFrame.")

    # --- TAB 2: FRAUD RULEBOOK ---
    with tab2:
        st.markdown("""
        ASTINA mengombinasikan machine learning dengan **9 Modul Aturan Bisnis (*Business Rule Engine*)** yang diselaraskan dengan pedoman verifikasi klinis dan regulasi pencegahan kecurangan jaminan kesehatan:
        """)

        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.markdown(r"""
            #### 1. 🔁 Repeat Billing
            Mendeteksi klaim berulang untuk pasien, tindakan medis, atau nominal yang sama/mirip dalam jendela waktu sempit ($\le 30$ hari) tanpa indikasi klinis valid.

            #### 2. 👻 Phantom Service
            Mengidentifikasi penagihan tindakan medis fiktif yang tidak pernah terjadi, tanggal tindakan tidak masuk akal, atau layanan rawat jalan yang diklaimkan di luar periode rawat inap.

            #### 3. ⏱️ Provider Capacity Anomaly
            Mengaudit total volume layanan medis yang dilakukan seorang dokter atau fasilitas kesehatan dalam satu hari. Sistem menandai jika kuantitas melampaui batas fisiologis kerja manusia normal.

            #### 4. 💰 Claim Status & Duplicate Payment
            Mendeteksi pengajuan kembali klaim yang sebelumnya telah berstatus `PAID` atau disetujui, mencegah potensi pembayaran ganda (*double disbursement*).

            #### 5. 🏷️ Upcoding & Unbundling
            Mendeteksi manipulasi penugasan kode diagnosis/tindakan ke tingkat keparahan yang lebih tinggi (*upcoding*) serta pemecahan satu paket tindakan medis terpadu menjadi beberapa klaim parsial terpisah (*unbundling*).
            """)

        with r_col2:
            st.markdown("""
            #### 6. 📈 Inflated Bill & Cloning
            Menandai tagihan dengan deviasi biaya ekstrem melebihi *standard benchmark* medis, serta pola rekam medis hasil penyalinan identik (*cloned medical records*).

            #### 7. 🏥 Length of Stay & Readmission
            Mendeteksi pasien yang dirawat inap melebihi rata-rata standar medis (*prolonged stay*) serta pasien yang sengaja dipulangkan terlalu dini lalu didaftarkan ulang dalam beberapa hari (*early readmission*).

            #### 8. 💊 Medication & Device Fraud
            Mengaudit pemberian dosis obat berlebih, peresepan obat di luar batas formularium nasional, serta klaim penggunaan implan atau alat medis sekali pakai secara berulang.

            #### 9. 🔤 Fuzzy Claim Matching
            Menggunakan algoritma *Levenshtein* dan *Jaro-Winkler distance* untuk mendeteksi kesamaan klaim terselubung yang sedikit diubah nama atau nomor tindakannya untuk mengecoh sistem.
            """)

        st.markdown(r"""
        ---
        **Formula Agregasi Skor Risiko Akhir**:
        $$\text{Final Risk Score} = 0.50(\text{Business Risk Score}) + 0.30(\text{ML Anomaly Score}) + 0.20(\text{Duplicate Payment Flag})$$
        """)

    # --- TAB 3: ROLES & ACTION MATRIX ---
    with tab3:
        st.markdown("#### 🚦 Matriks Ambang Batas Risiko & Rekomendasi Tindak Lanjut")
        matrix_data = [
            {
                "Tingkat Risiko": "🔴 TINGGI (High Risk)",
                "Rentang Skor": "Skor >= 0.65",
                "Indikasi": "Terdeteksi multi-anomali ML dan pelanggaran keras aturan bisnis fraud.",
                "Tindakan Operasional yang Disarankan": "HOLD KLAIM SEGERA. Tunda pencairan dana, lakukan audit berkas medis lengkap, dan tugaskan tim investigasi lapangan ke fasilitas kesehatan terkait."
            },
            {
                "Tingkat Risiko": "🟡 SEDANG (Medium Risk)",
                "Rentang Skor": "0.40 <= Skor < 0.65",
                "Indikasi": "Terdapat anomali biaya moderat atau indikasi pengulangan tindakan non-kritis.",
                "Tindakan Operasional yang Disarankan": "KLARIFIKASI DOKUMEN. Minta dokumen pendukung medis tambahan (resume medis / bukti laboratorium) kepada faskes sebelum persetujuan final."
            },
            {
                "Tingkat Risiko": "🟢 RENDAH (Low Risk)",
                "Rentang Skor": "Skor < 0.40",
                "Indikasi": "Pola transaksi normal, konsisten dengan riwayat historis klaim sejenis.",
                "Tindakan Operasional yang Disarankan": "PERSETUJUAN OTOMATIS (STP). Klaim dapat diproses langsung untuk pembayaran sesuai alur standar tanpa hambatan operasional."
            }
        ]
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 🔐 Matriks Hak Akses Peran Pengguna (RBAC)")
        rbac_data = [
            {"Peran": "👑 Admin", "Unggah Data": "✅ Penuh", "Pelatihan Model": "✅ Penuh", "Evaluasi Model": "✅ Penuh", "Deteksi Anomali": "✅ Penuh", "Status & Audit Log": "✅ Penuh + Hapus Cache"},
            {"Peran": "🔍 Auditor", "Unggah Data": "✅ Penuh", "Pelatihan Model": "👁️ Lihat Saja", "Evaluasi Model": "✅ Penuh", "Deteksi Anomali": "✅ Penuh", "Status & Audit Log": "✅ Audit Trail Penuh"},
            {"Peran": "🎯 Analyst", "Unggah Data": "✅ Penuh", "Pelatihan Model": "✅ Penuh", "Evaluasi Model": "✅ Penuh", "Deteksi Anomali": "✅ Penuh + AI Copilot", "Status & Audit Log": "👁️ Lihat Saja"},
            {"Peran": "👁️ Viewer", "Unggah Data": "❌ Akses Ditolak", "Pelatihan Model": "❌ Akses Ditolak", "Evaluasi Model": "👁️ Lihat Metrik", "Deteksi Anomali": "👁️ Lihat Hasil", "Status & Audit Log": "👁️ Ringkasan Status"}
        ]
        st.dataframe(pd.DataFrame(rbac_data), use_container_width=True, hide_index=True)

    # --- TAB 4: HYBRID AI ARCHITECTURE ---
    with tab4:
        st.markdown("""
        #### 🤖 Pendekatan Deteksi Multimodel ASTINA
        ASTINA tidak mengandalkan satu algoritma tunggal, melainkan menggabungkan kekuatan 4 pilar kecerdasan buatan:

        1. **🌲 Isolation Forest**:
           Mengidentifikasi outlier statistik global dengan cara memisahkan data anomali yang memiliki jarak pemotongan partisi (*isolation depth*) jauh lebih pendek daripada data normal.
        2. **🧠 PyTorch Deep Autoencoder**:
           Jaringan saraf tiruan non-linear yang mempelajari representasi normalitas transaksi. Klaim anomali terdeteksi saat *Reconstruction Loss* (galat rekonstruksi) melampaui batas ambang kuantil normal.
        3. **🚀 XGBoost / LightGBM Gradient Boosting**:
           Algoritma berbasis *decision trees* yang dioptimalkan dengan **Optuna Hyperparameter Tuning** untuk memprediksi probabilitas risiko pada data tabular berdimensi tinggi.
        4. **🕸️ Graph Neural Network (GNN - GATConv)**:
           Memodelkan relasi transaksi sebagai graf berbobot antar entitas (Pasien, Dokter, Faskes, Diagnosis). Menggunakan mekanisme *Graph Attention Network* dan *NeighborLoader* untuk membongkar sindikat kecurangan terorganisir (*fraud rings*).
        5. **🔍 Explainable AI (XAI) & Agentic Copilot**:
           Setiap anomali yang terdeteksi dilengkapi penjelasan lokal menggunakan **SHAP** dan **LIME** serta rekomendasi investigasi kontekstual dari **Local RAG & FAISS Vector Search**.
        """)

    st.markdown("---")
    st.caption("🛡️ **ASTINA Framework 2026** — Enterprise Health Insurance Fraud Intelligence. Terlindungi enkripsi SHA-256 dan kepatuhan regulasi PDP.")
