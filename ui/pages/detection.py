import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
from ui.utils import *
from state_manager import *
from repeat_billing_detector import RepeatBillingDetector
from fuzzy_claim_matcher import FuzzyClaimMatcher
from phantom_service_rules import PhantomServiceRuleEngine
from ui.utils import (
    generate_sample_claims_template,
    render_schema_readiness_card,
    TEMPLATE_CORE_COLUMNS,
    COLUMN_DESCRIPTIONS,
    COLUMN_RULE_DEPENDENCIES,
)
from agentic_copilot import AgenticInvestigatorCopilot, ClaimContextBuilder
from rag_engine import get_rag_knowledge_base
from ui_components import (
    lru_session_put,
    lru_session_get,
    rate_limit_check,
)
import shutil


def show_repeat_phantom_insights(df: pd.DataFrame):
    """Display repeat billing and phantom service findings in the detection page."""
    if df is None or df.empty:
        return

    st.subheader("🚨 Business Risk & Rule Violations")

    repeat_detector = RepeatBillingDetector(temporal_window_days=30, fuzzy_threshold=0.8)
    repeat_results = repeat_detector.detect_repeat_claims(df)

    engine = PhantomServiceRuleEngine()
    phantom_flags = []
    for idx, claim in df.iterrows():
        is_valid, violations = engine.validate_claim(claim.to_dict())
        if not is_valid:
            phantom_flags.append({
                "claim_id": claim.get("claim_id", idx),
                "service_code": claim.get("service_code", ""),
                "violations": "; ".join(violations),
            })
    phantom_df = pd.DataFrame(phantom_flags)

    total_claims = len(df)
    repeat_count = len(repeat_results)
    phantom_count = len(phantom_df)
    risk_score = round((repeat_count + phantom_count) / max(total_claims, 1) * 100, 2)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Klaim", f"{total_claims:,}")
    with col2:
        st.metric("Repeat Billing", f"{repeat_count:,}")
    with col3:
        st.metric("Phantom Service", f"{phantom_count:,}")
    with col4:
        st.metric("Risk Rate", f"{risk_score}%")

    col1, col2 = st.columns(2)
    with col1:
        if repeat_results.empty:
            st.success("✅ Tidak ditemukan pola repeat billing pada dataset ini.")
        else:
            st.warning(f"⚠️ Ditemukan {len(repeat_results)} potensi repeat billing")
            top_repeat = repeat_results.sort_values("risk_score", ascending=False).head(10)
            avail_cols = [c for c in ["first_claim_id", "repeat_claim_id", "time_gap_days", "similarity_score", "risk_score", "detection_reason"] if c in top_repeat.columns]
            st.dataframe(top_repeat[avail_cols], width='stretch')

    with col2:
        if phantom_df.empty:
            st.success("✅ Tidak ditemukan indikasi phantom service pada dataset ini.")
        else:
            st.warning(f"⚠️ Ditemukan {len(phantom_df)} potensi phantom service")
            st.dataframe(phantom_df.head(10), width='stretch')

    if not repeat_results.empty or not phantom_df.empty:
        st.info("📌 Kombinasi repeat billing + phantom service berfungsi sebagai sinyal prioritas audit untuk tim verifikator klaim.")


def _derive_risk_category(row: pd.Series) -> str:
    """Map a claim row to its dominant risk category."""
    flag_checks = [
        (row.get('repeat_billing_flag', 0) == 1, 'Repeat Billing'),
        (row.get('phantom_service_flag', 0) == 1, 'Phantom Service'),
        (row.get('provider_capacity_flag', 0) == 1, 'Provider Capacity'),
        (row.get('upcoding_unbundling_flag', 0) == 1, 'Upcoding'),
        (row.get('inflated_bill_cloning_flag', 0) == 1, 'Inflated Bill / Cloning'),
        (row.get('prolonged_stay_readmission_flag', 0) == 1, 'Prolonged Stay'),
        (row.get('medication_device_fraud_flag', 0) == 1, 'Medication / Device'),
        (row.get('duplicate_payment_flag', 0) == 1, 'Duplicate Payment'),
    ]
    for matched, label in flag_checks:
        if matched:
            return label
    if row.get('final_risk_flag', 0) == 1 or row.get('anomaly_prediction', 0) == 1:
        return 'Anomaly'
    return 'Normal'


def _build_safety_summary(df_result: pd.DataFrame, risk_summary: dict) -> list:
    """Assemble the executive summary cards for the UI."""
    total_claims = len(df_result)
    anomaly_claims = int(df_result.get('anomaly_prediction', pd.Series(0, index=df_result.index)).sum())
    high_risk_claims = int(risk_summary.get('high_risk_claims') or risk_summary.get('final_high_risk_claims') or 0)
    risk_cards = [
        ("Total Klaim", total_claims),
        ("Anomali", anomaly_claims),
        ("High Risk", high_risk_claims),
        ("Repeat Billing", int(risk_summary.get('repeat_billing_cases') or 0)),
        ("Phantom", int(risk_summary.get('phantom_service_cases') or 0)),
        ("Provider Capacity", int(risk_summary.get('provider_capacity_issues') or 0)),
        ("Duplicate Payment", int(risk_summary.get('duplicate_payment_claims') or 0)),
        ("Upcoding", int(risk_summary.get('upcoding_unbundling_cases') or 0)),
        ("Cloning", int(risk_summary.get('inflated_bill_cloning_cases') or 0)),
        ("Stay Risk", int(risk_summary.get('prolonged_stay_readmission_cases') or 0)),
        ("Med/Device", int(risk_summary.get('medication_device_fraud_cases') or 0)),
    ]
    return risk_cards


def _get_active_session_dataset():
    """Check and retrieve dataset available from earlier workflow steps."""
    for key in ['df_processed', 'data', 'train_df', 'raw_data_cache_sample', 'raw_data_cache_df']:
        candidate = st.session_state.get(key)
        if isinstance(candidate, pd.DataFrame) and not candidate.empty and len(candidate) >= 2:
            return candidate, f"Dataset Sesi Aktif ({key}: {len(candidate):,} baris)"
            
    if 'df_processed_path' in st.session_state and os.path.exists(st.session_state['df_processed_path']):
        try:
            df_disk = pd.read_parquet(st.session_state['df_processed_path'])
            if len(df_disk) >= 2:
                return df_disk, f"Dataset Praproses Disk ({len(df_disk):,} baris)"
        except Exception:
            pass
            
    return None, None


def show_detection_page():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetricContainer"] > div {
            background: rgba(255,255,255,0.78);
            border: 1px solid #dbe3f0;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }
        .stDataFrame {
            border: 1px solid #dfe7f3;
            border-radius: 12px;
            overflow: hidden;
        }
        .stExpander {
            border: 1px solid #dfe7f3;
            border-radius: 12px;
            background: rgba(255,255,255,0.78);
        }
        h1, h2, h3 {
            color: #0f172a;
        }
        .ready-card {
            background: #ffffff;
            border: 1px solid #c7d2fe;
            border-left: 6px solid #4f46e5;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔍 Deteksi Anomali Transaksi Klaim")
    st.caption("Deteksi komprehensif menggabungkan Machine Learning & Deep Learning (Isolation Forest, Autoencoder, XGBoost, GNN) dengan 9 Modul Business Rules & AI Investigator Copilot.")

    with st.expander("💡 Panduan Matriks Risiko & Tindak Lanjut Investigasi", expanded=False):
        st.markdown("""
        - 🔴 **Risiko Tinggi (Skor >= 0.65)**: **HOLD KLAIM SEGERA**. Tahan pencairan dana, lakukan audit berkas medis mendalam, dan jadwalkan verifikasi lapangan ke faskes terkait.
        - 🟡 **Risiko Sedang (0.40 <= Skor < 0.65)**: **KLARIFIKASI DOKUMEN**. Minta resume medis atau bukti penunjang (laboratorium/radiologi) kepada faskes sebelum persetujuan.
        - 🟢 **Risiko Rendah (Skor < 0.40)**: **PERSETUJUAN OTOMATIS (STP)**. Pola klaim normal, diproses pembayaran langsung sesuai alur standar.
        """)

    # ── 1. Model Loading ───────────────────────────────────────────────────────
    detector = load_persisted_detector()
    if detector is None:
        st.error("❌ Model belum tersedia. Silakan lakukan pelatihan model terlebih dahulu di halaman Pelatihan atau impor model.")
        logger.error("No detector available in detection page")
        if st.button("🚀 Ke Halaman Pelatihan Model", type="primary"):
            navigate_to_page('train')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if (torch is not None and getattr(torch, 'cuda', None) is not None) else 'cpu'
    training_features = (
        st.session_state.get('training_features')
        or getattr(detector, 'training_metadata', {}).get('training_features')
        or st.session_state.get('feature_columns', [])
    )
    feature_selection_method = st.session_state.get(
        'feature_selection_method',
        getattr(detector, 'training_metadata', {}).get('feature_selection_method', 'Model Tersimpan')
    )

    if not training_features:
        st.error("❌ **Metadata Fitur Pelatihan Tidak Ditemukan**: Model yang tersimpan tidak memiliki metadata nama kolom fitur yang valid. Silakan latih ulang model pada halaman Pelatihan agar skema fitur tersimpan lengkap.")
        if st.button("🚀 Ke Halaman Pelatihan Model", type="primary", key="goto_train_features_missing"):
            navigate_to_page('train')
        return

    # Model status bar
    with st.expander("ℹ️ Status & Metadata Model Champion Aktif", expanded=False):
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.metric("Fitur Model", f"{len(training_features)} Fitur")
        with c_m2:
            st.metric("Metode Seleksi", str(feature_selection_method))
        with c_m3:
            st.metric("Komputasi", str(device).upper())
        st.caption(f"Bobot Model: Isolation Forest ({detector.isolation_weight:.2f}), Autoencoder ({detector.autoencoder_weight:.2f}), XGBoost ({detector.xgboost_weight:.2f}), GNN ({getattr(detector, 'gnn_weight', 0.0):.2f})")

    st.markdown("---")

    # ── 2. Data Source Selector ───────────────────────────────────────────────
    st.subheader("📂 1. Pilih Sumber Data Klaim")
    
    session_df, session_label = _get_active_session_dataset()
    source_options = ["📤 Unggah File Baru (CSV / XLSX / XLS / Parquet)"]
    if session_df is not None:
        source_options.append(f"🔄 Gunakan {session_label}")
    source_options.append("🧪 Muat Dataset Sampel Demo (test_claims.csv)")

    selected_source = st.radio(
        "Pilih metode penyediaan data klaim:",
        source_options,
        index=0,
        horizontal=True,
        key="detection_source_radio"
    )

    raw_df = None
    source_description = ""

    if selected_source.startswith("📤 Unggah File Baru"):
        col_up1, col_up2 = st.columns([3, 1])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Unggah file dataset klaim asuransi:",
                type=["csv", "xlsx", "xls", "parquet"],
                help="Mendukung CSV, Excel (.xlsx, .xls), dan Parquet.",
                key="detection_file_uploader"
            )
        with col_up2:
            template_df = generate_sample_claims_template(n_rows=5)
            template_csv = template_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Unduh Template CSV",
                data=template_csv,
                file_name="astina_claim_template.csv",
                mime="text/csv",
                key="btn_download_template_detect",
                help="Unduh panduan struktur kolom standar klaim ASTINA"
            )

        if uploaded_file is not None:
            try:
                file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
                fmt_map = {"csv": "csv", "xlsx": "xlsx", "xls": "xls", "parquet": "parquet"}
                file_format: str = fmt_map.get(file_ext, file_ext) or "csv"
                raw_df = read_file_with_optimization(uploaded_file, file_format)

                # ── Post-read validation ──────────────────────────────────
                if raw_df is None or not isinstance(raw_df, pd.DataFrame):
                    raise ValueError("File tidak menghasilkan tabel data yang valid.")
                if raw_df.empty:
                    raise ValueError(
                        "File berhasil dibaca tetapi tidak mengandung baris data. "
                        "Pastikan terdapat minimal 1 baris data di bawah baris header."
                    )
                if raw_df.shape[1] == 1:
                    # Single-column almost always means wrong delimiter
                    col_sample = raw_df.columns[0]
                    for sep_char in (';', '\t', '|'):
                        if sep_char in col_sample:
                            st.warning(
                                f"⚠️ Terdeteksi hanya 1 kolom (`{col_sample[:40]}`). "
                                f"File CSV mungkin menggunakan pemisah `'{sep_char}'` bukan koma. "
                                "Simpan ulang file dengan pemisah koma (,) atau ubah ke format Excel/Parquet."
                            )
                            break

                source_description = f"File: {uploaded_file.name} ({len(raw_df):,} baris, {raw_df.shape[1]} kolom)"
                st.success(f"✅ File berhasil dimuat — **{len(raw_df):,} baris**, **{raw_df.shape[1]} kolom**.")

            except ValueError as e:
                st.error(f"❌ Format file tidak valid: {e}")
                logger.error("Detection upload ValueError: %s", e)
            except Exception as e:
                # Provide actionable guidance based on error type
                err_str = str(e).lower()
                if "codec" in err_str or "unicode" in err_str or "encoding" in err_str:
                    st.error(
                        "❌ **Error Encoding**: File menggunakan karakter non-UTF-8. "
                        "Simpan ulang file sebagai UTF-8 (di Excel: Simpan Sebagai → CSV UTF-8)."
                    )
                elif "memory" in err_str or "allocat" in err_str:
                    st.error(
                        "❌ **Error Memori**: File terlalu besar untuk dimuat langsung. "
                        "Gunakan format **Parquet** atau pecah file menjadi bagian lebih kecil (<500MB per file)."
                    )
                elif "password" in err_str or "crypt" in err_str:
                    st.error("❌ **File Terproteksi**: File Excel dilindungi kata sandi. Hapus proteksi terlebih dahulu.")
                elif "no sheet" in err_str or "worksheet" in err_str:
                    st.error("❌ **Sheet Tidak Ditemukan**: Pastikan file Excel memiliki sheet aktif dengan data klaim.")
                elif "zip" in err_str or "xlsx" in err_str:
                    st.error(
                        "❌ **File Excel Rusak**: File `.xlsx` tidak dapat dibuka. "
                        "Coba simpan ulang dari Excel atau ekspor ke CSV terlebih dahulu."
                    )
                else:
                    st.error(f"❌ Gagal membaca file: {e}")
                logger.error("Detection file read error: %s", e, exc_info=True)

    elif selected_source.startswith("🔄 Gunakan"):
        raw_df = session_df
        source_description = session_label

    elif selected_source.startswith("🧪 Muat Dataset Sampel"):
        try:
            if os.path.exists("test_claims.csv"):
                raw_df = pd.read_csv("test_claims.csv")
                source_description = f"Dataset Sampel Bawaan (test_claims.csv: {len(raw_df):,} baris)"
            else:
                raw_df = generate_sample_claims_template(n_rows=20)
                source_description = f"Dataset Sampel Sintetis ({len(raw_df):,} baris)"
        except Exception as e:
            st.error(f"❌ Gagal memuat dataset sampel: {str(e)}")

    # ── 3. Dataset Readiness & Execution Card ──────────────────────────────────
    if raw_df is None or len(raw_df) == 0:
        st.info("📤 Silakan pilih atau unggah dataset klaim di atas untuk memulai analisis deteksi anomali.")
        return

    if len(raw_df) < 2:
        st.warning("⚠️ Dataset harus memiliki minimal **2 baris data** agar analisis statistik, GNN, dan aturan temporal dapat berjalan dengan akurat.")
        return

    # Check if dataset changed to clear previous results
    current_df_signature = f"{len(raw_df)}_{len(raw_df.columns)}_{list(raw_df.columns[:3])}"
    if st.session_state.get('last_detection_signature') != current_df_signature:
        st.session_state['last_detection_signature'] = current_df_signature
        st.session_state.pop('detection_results', None)
        st.session_state.pop('detection_executed', None)
        st.session_state.pop('detection_processed_df', None)

    # Readiness Card UI
    st.markdown(
        f"""
        <div class="ready-card">
            <div style="font-size: 1.1rem; font-weight: 700; color: #1e1b4b; margin-bottom: 0.4rem;">
                📋 Dataset Siap Dieksekusi: <span style="color: #4f46e5;">{source_description}</span>
            </div>
            <div style="font-size: 0.9rem; color: #475569;">
                Total <b>{len(raw_df):,} klaim</b> dengan <b>{len(raw_df.columns)} kolom</b> terdeteksi dan tervalidasi.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Schema completeness inspection
    with st.expander("🩺 Periksa Kesiapan Skema Kolom Bisnis", expanded=False):
        render_schema_readiness_card(raw_df)
        st.dataframe(raw_df.head(5), width='stretch')

    # ── 4. Detection Configuration & Action Panel ─────────────────────────────
    st.subheader("⚙️ 2. Konfigurasi & Eksekusi Deteksi")
    
    cfg_col1, cfg_col2 = st.columns([2, 1])
    with cfg_col1:
        threshold = st.slider(
            "Ambang Batas Anomali (Anomaly Threshold):",
            min_value=0.10,
            max_value=0.95,
            value=st.session_state.get('detection_threshold', 0.50),
            step=0.05,
            help="Skor probabilitas gabungan di atas ambang ini akan diklasifikasikan sebagai anomali/potensi fraud."
        )
    with cfg_col2:
        enable_gnn_inf = st.checkbox(
            "Aktifkan Analisis Graf Relasi (GNN)",
            value=getattr(detector, "gnn_model", None) is not None,
            help="Menghubungkan pola klaim antar pasien dan provider dalam jaringan graf relasi."
        )

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        run_detection_clicked = st.button(
            "🚀 Jalankan Deteksi Anomali Multi-Algoritma",
            type="primary",
            key="btn_run_detection_primary"
        )
    with btn_col2:
        if st.button("🔄 Reset Hasil Deteksi", key="btn_reset_detection"):
            st.session_state.pop('detection_results', None)
            st.session_state.pop('detection_executed', None)
            st.rerun()

    # ── 5. Execution Pipeline ──────────────────────────────────────────────────
    if run_detection_clicked and isinstance(raw_df, pd.DataFrame):
        with st.spinner("⏳ Memproses data, merekonstruksi fitur inferensi, dan menjalankan multi-model ensemble..."):
            try:
                # 1. Preprocessing with cache
                if 'detection_processed_df' in st.session_state and st.session_state.get('last_detection_signature') == current_df_signature:
                    df_processed = st.session_state['detection_processed_df']
                    feature_columns_proc = st.session_state.get('detection_feature_columns', [])
                else:
                    df_processed, feature_columns_proc, _ = preprocess_insurance_claims_optimized(
                        raw_df,
                        enable_large_file_handling=st.session_state.get('enable_large_file_handling', True),
                        enable_outlier_detection=st.session_state.get('enable_outlier_detection', True),
                        enable_data_validation=st.session_state.get('enable_data_validation', True)
                    )
                    st.session_state['detection_processed_df'] = df_processed
                    st.session_state['detection_feature_columns'] = feature_columns_proc

                if not isinstance(df_processed, pd.DataFrame):
                    df_processed = pd.DataFrame(df_processed)

                # 2. Strict feature alignment with training metadata
                # Priority order for median imputation:
                #   1. session_state['feature_medians']  (loaded from persisted _params.json)
                #   2. detector.training_metadata['feature_medians']  (in-session training)
                #   3. Empty dict → falls back to column-level median or 0.0
                training_stats: dict = (
                    st.session_state.get('feature_medians')
                    or (getattr(detector, 'training_metadata', {}) or {}).get('feature_medians')
                    or {}
                )
                aligned_feature_df, alignment_summary = build_aligned_inference_features(
                    df_processed, training_features, training_stats=training_stats
                )

                # Store alignment summary for audit / expander display
                st.session_state['detection_alignment_summary'] = alignment_summary

                n_existing  = len(alignment_summary.get('existing_features', []))
                n_derived   = len(alignment_summary.get('derived_features', []))
                n_filled    = len(alignment_summary.get('filled_features', []))
                n_expected  = alignment_summary.get('expected_features', len(training_features))

                X = aligned_feature_df[training_features].values
                edge_index = None
                edge_type = None

                # 3. Optional GNN Graph construction
                if enable_gnn_inf and getattr(detector, "gnn_model", None) is not None and getattr(detector, "gnn_weight", 0) > 0:
                    graph_frame = df_processed.copy()
                    for feature_name in training_features:
                        graph_frame[feature_name] = aligned_feature_df[feature_name].to_numpy()
                    graph_metadata = getattr(detector, 'training_metadata', {}) or {}
                    graph_method = st.session_state.get('graph_method', graph_metadata.get('graph_method', 'star'))
                    graph_kwargs = {}
                    if graph_method == 'knn' and graph_metadata.get('graph_k'):
                        graph_kwargs['k'] = int(graph_metadata['graph_k'])
                    graph_result = create_claim_graph(
                        graph_frame,
                        training_features,
                        method=graph_method,
                        max_nodes=min(len(graph_frame), 20000),
                        **graph_kwargs,
                    )
                    if len(graph_result) == 3:
                        _, edge_index, edge_type = graph_result
                    else:
                        _, edge_index = graph_result

                # 4. Multi-model ensemble inference
                probabilities, individual_probs = detector.predict_anomaly_probability(
                    X, edge_index=edge_index, edge_type=edge_type, device=str(device)
                )
                predictions = (probabilities > threshold).astype(int)

                # 5. Build results DataFrame
                df_result = raw_df.copy()
                df_result['anomaly_probability'] = pd.Series(np.asarray(probabilities, dtype=np.float64), index=df_result.index)
                df_result['anomaly_prediction'] = pd.Series(np.asarray(predictions, dtype=np.int64), index=df_result.index)
                df_result['isolation_forest_score'] = pd.Series(np.asarray(individual_probs.get('isolation_forest', np.zeros(len(df_result))), dtype=np.float64), index=df_result.index)
                df_result['autoencoder_score'] = pd.Series(np.asarray(individual_probs.get('autoencoder', np.zeros(len(df_result))), dtype=np.float64), index=df_result.index)
                if 'dbscan' in individual_probs:
                    df_result['dbscan_score'] = pd.Series(np.asarray(individual_probs['dbscan'], dtype=np.float64), index=df_result.index)
                df_result['xgboost_score'] = pd.Series(np.asarray(individual_probs.get('xgboost', np.zeros(len(df_result))), dtype=np.float64), index=df_result.index)

                # 6. Integrated Claim Risk Pipeline (9 Business Rules + Composite Scoring)
                from fraud_risk_pipeline import run_integrated_claim_risk_pipeline
                df_risk, risk_summary = run_integrated_claim_risk_pipeline(df_result)
                df_result = df_risk
                df_result['business_risk_score'] = df_result.get('business_risk_score', pd.Series(0.0, index=df_result.index))
                df_result['final_risk_score'] = df_result.get('final_risk_score', df_result['anomaly_probability'])
                df_result['final_risk_flag'] = df_result.get('final_risk_flag', pd.Series((df_result['final_risk_score'] >= threshold).astype(int), index=df_result.index))

                # Store persistently in session_state
                st.session_state['detection_results'] = df_result
                st.session_state['detection_threshold'] = threshold
                st.session_state['risk_summary'] = risk_summary
                st.session_state['detection_executed'] = True

                # Feature alignment diagnostics banner & audit expander
                if n_filled > 0:
                    st.warning(
                        f"⚠️ **Feature Alignment**: {n_existing}/{n_expected} fitur ditemukan langsung, "
                        f"{n_derived} diturunkan otomatis, **{n_filled} diimputasi dengan median training**. "
                        f"Pertimbangkan menambahkan kolom yang hilang ke dataset untuk akurasi lebih tinggi."
                    )
                else:
                    st.success(
                        f"✅ **Deteksi anomali berhasil dieksekusi!** "
                        f"Semua {n_existing} fitur ditemukan langsung + {n_derived} fitur turunan. "
                        f"Hasil disajikan pada dashboard di bawah."
                    )

                with st.expander("🔍 Detail Penyelarasan Fitur Inferensi (Feature Alignment Audit)", expanded=False):
                    diag_c1, diag_c2, diag_c3 = st.columns(3)
                    with diag_c1:
                        st.markdown(f"**Fitur Eksisting ({n_existing}):**")
                        st.caption(", ".join(alignment_summary.get('existing_features', [])) or "-")
                    with diag_c2:
                        st.markdown(f"**Fitur Diturunkan ({n_derived}):**")
                        st.caption(", ".join(alignment_summary.get('derived_features', [])) or "-")
                    with diag_c3:
                        st.markdown(f"**Fitur Imputasi Median ({n_filled}):**")
                        st.caption(", ".join(alignment_summary.get('filled_features', [])) or "-")

            except Exception as err:
                st.error(f"❌ Terjadi kesalahan saat eksekusi deteksi: {str(err)}")
                logger.error(f"Detection execution pipeline error: {err}", exc_info=True)
                with st.expander("Detail Error (Traceback)"):
                    import traceback
                    st.code(traceback.format_exc())

    # ── 6. Persistent Results Dashboard (Tabs) ────────────────────────────────
    if st.session_state.get('detection_executed', False) and 'detection_results' in st.session_state:
        df_result = st.session_state['detection_results']
        risk_summary = st.session_state.get('risk_summary', {})
        threshold = st.session_state.get('detection_threshold', threshold)

        st.markdown("---")
        st.subheader("📊 3. Hasil & Analisis Deteksi Anomali")

        # Top 4 Metrics Summary
        total_claims = len(df_result)
        anomaly_claims = int(df_result['anomaly_prediction'].sum())
        anomaly_rate = anomaly_claims / max(total_claims, 1)
        high_risk_claims = int(df_result.get('final_risk_flag', 0).sum())

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Klaim Dianalisis", f"{total_claims:,}")
        with m2:
            st.metric("Klaim Anomali (ML)", f"{anomaly_claims:,}", delta=f"{anomaly_rate:.1%}", delta_color="inverse")
        with m3:
            st.metric("Klaim High Risk (Final)", f"{high_risk_claims:,}")
        with m4:
            st.metric("Ambang Batas", f"{threshold:.2f}")

        # Tabbed Layout
        tab_summary, tab_rules, tab_table, tab_copilot, tab_drift = st.tabs([
            "📊 Ringkasan & Visualisasi",
            "🚨 Business Risk & Rules",
            "📋 Fraud Review Table & Export",
            "🤖 AI Investigator Copilot & BAP",
            "📈 Concept Drift & Retraining"
        ])

        # ── TAB 1: Visualizations ──
        with tab_summary:
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                pred_series = pd.to_numeric(df_result['anomaly_prediction'], errors='coerce').fillna(0).astype(int)
                normal_count = int((pred_series == 0).sum())
                anomaly_count = int((pred_series == 1).sum())
                fig_pred = create_bar_chart(['Normal', 'Anomali'], [normal_count, anomaly_count],
                                           title='Distribusi Prediksi Anomali',
                                           labels={'x': 'Kategori', 'y': 'Jumlah'})
                fig_pred.update_layout(showlegend=False)
                st.plotly_chart(fig_pred, width='stretch')

            with v_col2:
                fig_hist = create_histogram_chart(df_result, 'anomaly_probability', nbins=40,
                                                 title='Distribusi Skor Anomali (Multi-Model Ensemble)')
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                                   annotation_text=f"Threshold: {threshold:.2f}")
                st.plotly_chart(fig_hist, width='stretch')

            st.markdown("#### 🛡️ Executive Risk Summary Panel")
            summary_cards = _build_safety_summary(df_result, risk_summary)
            summary_cols = st.columns(4)
            for idx, (label, value) in enumerate(summary_cards):
                with summary_cols[idx % 4]:
                    st.markdown(
                        f"<div style='border:1px solid #dfe3e8; border-radius:10px; padding:0.7rem; background:#f8fafc; text-align:center; margin-bottom:0.5rem;'>"
                        f"<div style='font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:0.04em;'>{label}</div>"
                        f"<div style='font-size:1.5rem; font-weight:700; color:#0f172a; margin-top:0.2rem;'>{value}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Category distribution chart
            category_chart_df = df_result.copy()
            if 'risk_category' not in category_chart_df.columns:
                category_chart_df['risk_category'] = category_chart_df.apply(_derive_risk_category, axis=1)
            category_counts = category_chart_df['risk_category'].value_counts().reset_index()
            category_counts.columns = ['Risk Category', 'Count']
            category_counts = category_counts[category_counts['Count'] > 0]

            if not category_counts.empty:
                st.markdown("#### 📌 Proporsi Risiko per Kategori")
                c_left, c_right = st.columns([1.5, 1])
                with c_left:
                    risk_fig = px.bar(
                        category_counts,
                        x='Risk Category',
                        y='Count',
                        color='Risk Category',
                        title='Jumlah Klaim per Kategori Risiko',
                        color_discrete_sequence=['#0f172a', '#2563eb', '#f59e0b', '#ef4444', '#8b5cf6', '#10b981', '#14b8a6', '#f97316']
                    )
                    risk_fig.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(risk_fig, width='stretch')
                with c_right:
                    pie_fig = px.pie(
                        category_counts,
                        names='Risk Category',
                        values='Count',
                        title='Proporsi Kategori Risiko',
                        color='Risk Category',
                        color_discrete_sequence=['#2563eb', '#f59e0b', '#ef4444', '#8b5cf6', '#10b981', '#14b8a6', '#f97316', '#64748b']
                    )
                    pie_fig.update_traces(textinfo='percent+label', hole=0.35)
                    st.plotly_chart(pie_fig, width='stretch')

        # ── TAB 2: Business Rules ──
        with tab_rules:
            show_repeat_phantom_insights(df_result)

        # ── TAB 3: Fraud Review Table ──
        with tab_table:
            st.subheader("📋 Tabel Audit Klaim Terfilter")
            
            display_df = df_result.copy()
            for col in ['final_risk_flag', 'business_risk_flag', 'phantom_service_flag', 'repeat_billing_flag', 'duplicate_payment_flag', 'anomaly_prediction',
                        'upcoding_unbundling_flag', 'inflated_bill_cloning_flag', 'prolonged_stay_readmission_flag', 'medication_device_fraud_flag']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].fillna(0).astype(int)

            if 'final_risk_flag' not in display_df.columns:
                display_df['final_risk_flag'] = (display_df.get('anomaly_prediction', 0).fillna(0).astype(int) > 0).astype(int)
            if 'business_risk_score' not in display_df.columns:
                display_df['business_risk_score'] = display_df.get('anomaly_probability', 0.0).astype(float)
            if 'final_risk_score' not in display_df.columns:
                display_df['final_risk_score'] = display_df['business_risk_score']
            if 'risk_category' not in display_df.columns:
                display_df['risk_category'] = display_df.apply(_derive_risk_category, axis=1)
            if 'severity' not in display_df.columns:
                display_df['severity'] = np.where(
                    display_df['final_risk_flag'] == 1,
                    'High',
                    np.where(display_df.get('anomaly_prediction', 0).astype(int) == 1, 'Medium', 'Low')
                )

            # Filters
            f_cols = st.columns(5)
            with f_cols[0]:
                risk_group = st.selectbox("Kategori Risiko:", ["Semua", "High Risk", "Normal", "Anomali", "Repeat Billing", "Phantom Service", "Provider Capacity", "Duplicate Payment", "Upcoding", "Inflated Bill / Cloning", "Prolonged Stay", "Medication / Device"], index=0, key="tbl_f_risk")
            with f_cols[1]:
                prov_opts = ["Semua"] + sorted(display_df['provider_id'].astype(str).dropna().unique().tolist()[:50]) if 'provider_id' in display_df.columns else ["Semua"]
                provider_filter = st.selectbox("Provider:", prov_opts, index=0, key="tbl_f_prov")
            with f_cols[2]:
                srv_opts = ["Semua"] + sorted(display_df['service_code'].astype(str).dropna().unique().tolist()[:50]) if 'service_code' in display_df.columns else ["Semua"]
                service_filter = st.selectbox("Service Code:", srv_opts, index=0, key="tbl_f_srv")
            with f_cols[3]:
                risk_band = st.selectbox("Severity:", ["Semua", "High", "Medium", "Low"], index=0, key="tbl_f_sev")
            with f_cols[4]:
                sort_order = st.selectbox("Urutan Skor:", ['Menurun (Tertinggi)', 'Menaik (Terendah)'], index=0, key="tbl_f_sort")

            # Apply filters
            if risk_group == "High Risk":
                display_df = display_df[display_df['final_risk_flag'] == 1]
            elif risk_group == "Normal":
                display_df = display_df[display_df['final_risk_flag'] == 0]
            elif risk_group == "Anomali":
                display_df = display_df[display_df.get('anomaly_prediction', 0).astype(int) == 1]
            elif risk_group != "Semua":
                display_df = display_df[display_df['risk_category'] == risk_group]

            if provider_filter != "Semua" and 'provider_id' in display_df.columns:
                display_df = display_df[display_df['provider_id'].astype(str).str.contains(str(provider_filter), case=False, na=False)]
            if service_filter != "Semua" and 'service_code' in display_df.columns:
                display_df = display_df[display_df['service_code'].astype(str).str.contains(str(service_filter), case=False, na=False)]
            if risk_band != "Semua":
                display_df = display_df[display_df['severity'] == risk_band]

            display_df = display_df.sort_values(by='final_risk_score', ascending=(sort_order == 'Menaik (Terendah)'))

            # Table display
            display_columns = [
                'claim_id', 'patient_id', 'provider_id', 'service_code', 'amount', 'billed_amount',
                'anomaly_probability', 'final_risk_score', 'severity', 'risk_category', 'final_risk_flag'
            ]
            for col in ['repeat_billing_flag', 'phantom_service_flag', 'duplicate_payment_flag', 'upcoding_unbundling_flag', 'inflated_bill_cloning_flag', 'prolonged_stay_readmission_flag', 'medication_device_fraud_flag']:
                if col in display_df.columns:
                    display_columns.append(col)

            display_columns = [col for col in display_columns if col in display_df.columns]

            t_info_col, t_limit_col = st.columns([3, 1])
            with t_limit_col:
                preview_limit = st.selectbox("Tampilkan baris:", [25, 50, 100, 250, 500], index=1, key="tbl_limit")
            with t_info_col:
                st.caption(f"Menampilkan **{min(len(display_df), preview_limit)}** dari total **{len(display_df):,}** klaim terfilter.")

            fraud_table = display_df[display_columns].head(preview_limit).copy()

            def highlight_fraud(row):
                sev = row.get('severity', 'Low')
                if sev == 'High':
                    return ['background-color: #fee2e2'] * len(row)
                if sev == 'Medium':
                    return ['background-color: #fff7ed'] * len(row)
                return [''] * len(row)

            st.dataframe(fraud_table.style.apply(highlight_fraud, axis=1), width='stretch')

            # Export button
            st.markdown("---")
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Seluruh Hasil Deteksi (CSV)",
                data=csv_data,
                file_name=f"hasil_deteksi_klaim_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_detection_results_csv",
                type="secondary"
            )

        # ── TAB 4: Copilot & BAP ──
        with tab_copilot:
            st.subheader("🤖 AI Investigator Copilot & Berkas BAP")
            st.caption("Gunakan Agentic AI & RAG Regulasi Medis (*Permenkes, INA-CBGs, FORNAS*) untuk menyusun Berita Acara Pemeriksaan (BAP) audit klaim secara otomatis.")

            # ── FIX #1: Use the full df_result, not the filtered display_df from Tab 3.
            # This ensures Tab 4 always shows all detected claims regardless of
            # whatever filter the user applied in Tab 3.
            copilot_base_df = df_result  # df_result is persisted in session_state above

            if copilot_base_df is None or copilot_base_df.empty:
                st.info("📤 Jalankan deteksi anomali terlebih dahulu untuk mengaktifkan Copilot.")
            elif 'claim_id' not in copilot_base_df.columns:
                st.warning("⚠️ Kolom `claim_id` tidak ditemukan pada dataset. Tambahkan kolom tersebut agar Copilot dapat memilih klaim secara spesifik.")
            else:
                # Sort: high-risk first so investigator sees most critical claims at the top
                _cop_df = copilot_base_df.copy()
                if 'final_risk_score' in _cop_df.columns:
                    _cop_df = _cop_df.sort_values('final_risk_score', ascending=False)

                claim_choices = (
                    ["-- Pilih Klaim untuk Diinvestigasi --"]
                    + _cop_df['claim_id'].astype(str).dropna().unique().tolist()
                )
                selected_claim = st.selectbox(
                    "Pilih nomor klaim (diurutkan: risiko tertinggi dulu):",
                    claim_choices,
                    index=0,
                    key="copilot_claim_select",
                )

                if selected_claim not in ("-- Pilih Klaim untuk Diinvestigasi --",):
                    selected_row = _cop_df[_cop_df['claim_id'].astype(str) == str(selected_claim)].iloc[0]

                    # ── CLAIM IDENTITY PANEL ──
                    severity_val = str(selected_row.get('severity', 'Low'))
                    risk_score_val = float(selected_row.get('final_risk_score', 0.0))
                    anomaly_prob_val = float(selected_row.get('anomaly_probability', 0.0))

                    sev_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}.get(severity_val, "#64748b")
                    sev_bg = {"High": "#fee2e2", "Medium": "#fff7ed", "Low": "#d1fae5"}.get(severity_val, "#f1f5f9")

                    st.markdown(f"""
                    <div style="background:#fff;border:1px solid #e2e8f0;border-left:5px solid {sev_color};border-radius:10px;padding:14px 18px;margin-bottom:14px;display:flex;flex-wrap:wrap;align-items:center;gap:10px;">
                        <div style="flex:1;min-width:220px;">
                            <div style="font-size:0.78rem;color:#64748b;font-weight:600;letter-spacing:0.4px;text-transform:uppercase;">Klaim Terpilih</div>
                            <div style="font-size:1.18rem;font-weight:800;color:#0f172a;font-family:monospace;">{selected_claim}</div>
                            <div style="font-size:0.8rem;color:#475569;margin-top:2px;">Provider: <b>{selected_row.get('provider_id','N/A')}</b> &nbsp;|&nbsp; Layanan: <code>{selected_row.get('service_code','N/A')}</code> &nbsp;|&nbsp; Diagnosis: <code>{selected_row.get('diagnosis_code','N/A')}</code></div>
                        </div>
                        <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
                            <div style="text-align:center;background:#eff6ff;border:1px solid #dbeafe;border-radius:8px;padding:8px 16px;">
                                <div style="font-size:0.7rem;color:#3b82f6;font-weight:700;text-transform:uppercase;">Risk Score</div>
                                <div style="font-size:1.5rem;font-weight:800;color:#1e40af;">{risk_score_val:.2f}</div>
                            </div>
                            <div style="text-align:center;background:#f5f3ff;border:1px solid #ede9fe;border-radius:8px;padding:8px 16px;">
                                <div style="font-size:0.7rem;color:#7c3aed;font-weight:700;text-transform:uppercase;">Anomaly Prob</div>
                                <div style="font-size:1.5rem;font-weight:800;color:#6d28d9;">{anomaly_prob_val:.2f}</div>
                            </div>
                            <div style="text-align:center;background:{sev_bg};border:1px solid {sev_color}33;border-radius:8px;padding:8px 16px;">
                                <div style="font-size:0.7rem;color:{sev_color};font-weight:700;text-transform:uppercase;">Severity</div>
                                <div style="font-size:1.2rem;font-weight:800;color:{sev_color};">{severity_val.upper()}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── VIOLATION INDICATORS ──
                    detail_reasons = []
                    flag_icon_map = {
                        'repeat_billing_flag':             ('🔁', 'Repeat Billing', 'Klaim berulang dalam jangka waktu dekat'),
                        'phantom_service_flag':            ('👻', 'Phantom Service', 'Tindakan/obat fiktif tanpa dasar medis'),
                        'duplicate_payment_flag':          ('💳', 'Duplicate Payment', 'Pembayaran ganda atas klaim yang sama'),
                        'upcoding_unbundling_flag':        ('📈', 'Upcoding / Unbundling', 'Penggelembungan kode atau pemecahan tagihan'),
                        'inflated_bill_cloning_flag':      ('🧬', 'Inflated Bill / Cloning', 'Kloning klaim atau biaya digelembungkan'),
                        'prolonged_stay_readmission_flag': ('🏥', 'Prolonged Stay / Readmission', 'Masa rawat tidak wajar atau rawat ulang anomali'),
                        'medication_device_fraud_flag':    ('💊', 'Medication / Device Fraud', 'Obat/alkes diberikan dengan kuantitas tidak lazim'),
                    }
                    for flag_col, (icon, short_label, desc) in flag_icon_map.items():
                        if flag_col in selected_row and pd.notna(selected_row.get(flag_col)) and int(selected_row.get(flag_col, 0)) == 1:
                            detail_reasons.append((icon, short_label, desc))

                    if detail_reasons:
                        badges_html = "".join([
                            f'<span style="display:inline-flex;align-items:center;gap:5px;background:#fee2e2;border:1px solid #fca5a5;color:#991b1b;'
                            f'border-radius:20px;padding:4px 12px;font-size:0.76rem;font-weight:700;margin:3px 4px 3px 0;">'
                            f'{icon} {lbl}</span>'
                            for icon, lbl, _ in detail_reasons
                        ])
                        st.markdown(
                            f'<div style="background:#fff5f5;border:1px solid #fee2e2;border-radius:8px;padding:10px 14px;margin-bottom:12px;">'
                            f'<div style="font-size:0.76rem;font-weight:700;color:#991b1b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">⚠️ Indikator Pelanggaran Terdeteksi</div>'
                            f'{badges_html}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("ℹ️ Tidak ada aturan eksplisit terpicu. Anomali terdeteksi murni oleh ensemble model ML (deviasi statistik multivariat).")

                    # ── COPILOT CONFIGURATION (SESSION PERSISTENT) ──
                    with st.expander("🛠️ Konfigurasi Copilot & LLM Engine", expanded=True):
                        # Ensure session-level defaults
                        if 'copilot_provider_sel' not in st.session_state:
                            st.session_state['copilot_provider_sel'] = "Heuristic Engine (Offline)"
                        if 'copilot_api_key_val' not in st.session_state:
                            st.session_state['copilot_api_key_val'] = ""
                        if 'copilot_auditor_val' not in st.session_state:
                            st.session_state['copilot_auditor_val'] = "Investigator Senior ASTINA"
                        if 'copilot_model_val' not in st.session_state:
                            st.session_state['copilot_model_val'] = ""
                        if 'copilot_endpoint_val' not in st.session_state:
                            st.session_state['copilot_endpoint_val'] = "http://localhost:11434/api/generate"

                        c_cfg1, c_cfg2, c_cfg3 = st.columns([1.5, 2, 1.5])
                        with c_cfg1:
                            provider_choice = st.selectbox(
                                "LLM Engine:",
                                ["Heuristic Engine (Offline)", "Google Gemini", "OpenAI / Compatible", "Local Ollama"],
                                index=["Heuristic Engine (Offline)", "Google Gemini", "OpenAI / Compatible", "Local Ollama"].index(
                                    st.session_state['copilot_provider_sel'] if st.session_state['copilot_provider_sel'] in ["Heuristic Engine (Offline)", "Google Gemini", "OpenAI / Compatible", "Local Ollama"] else "Heuristic Engine (Offline)"
                                ),
                                key="copilot_cfg_provider"
                            )
                            st.session_state['copilot_provider_sel'] = provider_choice

                        with c_cfg2:
                            if provider_choice in ["Google Gemini", "OpenAI / Compatible"]:
                                api_key_input = st.text_input(
                                    "API Key:",
                                    value=st.session_state['copilot_api_key_val'],
                                    type="password",
                                    key="copilot_cfg_apikey"
                                )
                                st.session_state['copilot_api_key_val'] = api_key_input
                            else:
                                api_key_input = ""
                                st.markdown(f"<div style='padding-top:28px;font-size:0.8rem;color:#64748b;'>🔌 Mode: <b>{'Heuristic (Offline)' if 'Heuristic' in provider_choice else 'Local Endpoint'}</b> — tidak memerlukan API key eksternal.</div>", unsafe_allow_html=True)

                        with c_cfg3:
                            auditor_name = st.text_input(
                                "Nama Verifikator/Auditor:",
                                value=st.session_state['copilot_auditor_val'],
                                key="copilot_cfg_auditor"
                            )
                            st.session_state['copilot_auditor_val'] = auditor_name

                        # Advanced Model & Endpoint tuning row
                        c_adv1, c_adv2 = st.columns(2)
                        model_name_choice = ""
                        endpoint_choice = ""
                        with c_adv1:
                            if provider_choice == "Google Gemini":
                                model_name_choice = st.selectbox(
                                    "Model Gemini:",
                                    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
                                    index=0,
                                    key="copilot_cfg_model_gemini"
                                )
                            elif provider_choice == "OpenAI / Compatible":
                                model_name_choice = st.text_input(
                                    "Model Name:",
                                    value="gpt-4o-mini",
                                    key="copilot_cfg_model_openai"
                                )
                            elif provider_choice == "Local Ollama":
                                model_name_choice = st.text_input(
                                    "Ollama Model Name:",
                                    value="llama3",
                                    key="copilot_cfg_model_ollama"
                                )
                        with c_adv2:
                            if provider_choice == "Local Ollama":
                                endpoint_choice = st.text_input(
                                    "Ollama Endpoint URL:",
                                    value=st.session_state['copilot_endpoint_val'],
                                    key="copilot_cfg_endpoint_ollama"
                                )
                                st.session_state['copilot_endpoint_val'] = endpoint_choice
                            elif provider_choice == "OpenAI / Compatible":
                                endpoint_choice = st.text_input(
                                    "Base URL / Custom Endpoint (Opsional):",
                                    value="",
                                    placeholder="https://api.openai.com/v1/chat/completions",
                                    key="copilot_cfg_endpoint_openai"
                                )

                        # ── FIX #2: Connection test button ────────────────────
                        if provider_choice != "Heuristic Engine (Offline)":
                            if st.button("🔌 Test Koneksi LLM", key="btn_test_llm_conn"):
                                _test_engine = AgenticInvestigatorCopilot(
                                    provider={"Google Gemini": "gemini", "OpenAI / Compatible": "openai", "Local Ollama": "ollama"}.get(provider_choice, "heuristic"),
                                    api_key=api_key_input,
                                    model_name=model_name_choice,
                                    endpoint_url=endpoint_choice if endpoint_choice else None,
                                )
                                with st.spinner("Menguji koneksi..."):
                                    _conn = _test_engine.test_connection()
                                if _conn["ok"]:
                                    st.success(f"{_conn['message']} — Provider: **{_conn['provider']}**")
                                else:
                                    st.error(f"{_conn['message']}")
                                    st.info("💡 Jika API Key salah atau habis kuota, BAP akan tetap dibuat menggunakan **Heuristic Engine (Offline)** sebagai fallback.")

                    provider_map = {
                        "Heuristic Engine (Offline)": "heuristic",
                        "Google Gemini": "gemini",
                        "OpenAI / Compatible": "openai",
                        "Local Ollama": "ollama"
                    }
                    copilot_engine = AgenticInvestigatorCopilot(
                        provider=provider_map.get(provider_choice, "heuristic"),
                        api_key=api_key_input,
                        model_name=model_name_choice,
                        endpoint_url=endpoint_choice if endpoint_choice else None
                    )

                    # ── DYNAMIC XAI & GNN CONTEXT EXTRACTION ──
                    # 1. Feature deviation (z-score relative to full result set)
                    feature_contributions = dict(st.session_state.get('shap_contributions', {}))
                    candidate_num_cols = ['billed_amount', 'paid_amount', 'amount', 'length_of_stay', 'procedure_count', 'medication_count', 'daily_claims_by_provider']
                    for num_col in candidate_num_cols:
                        if num_col in selected_row and pd.notna(selected_row[num_col]):
                            try:
                                c_val = float(selected_row[num_col])
                                if num_col in copilot_base_df.columns:
                                    s = pd.to_numeric(copilot_base_df[num_col], errors='coerce').dropna()
                                    if len(s) > 1 and s.std() > 0:
                                        z = (c_val - s.mean()) / (s.std() + 1e-6)
                                        if abs(z) >= 0.8:
                                            feature_contributions[num_col] = round(float(z), 3)
                            except Exception:
                                pass

                    # 2. GNN topology cluster extraction
                    gnn_clusters = []
                    prov_id = selected_row.get('provider_id')
                    if prov_id and str(prov_id) not in ('N/A', 'None', ''):
                        p_matches = copilot_base_df[copilot_base_df['provider_id'] == prov_id]
                        if len(p_matches) > 1:
                            susp_c = int((pd.to_numeric(p_matches.get('anomaly_prediction', 0), errors='coerce') == 1).sum())
                            gnn_clusters.append(f"Faskes {prov_id}: {len(p_matches)} klaim terhubung ({susp_c} anomali) dalam klaster audit")
                    diag_code = selected_row.get('diagnosis_code')
                    if diag_code and str(diag_code) not in ('N/A', 'None', ''):
                        d_matches = copilot_base_df[copilot_base_df['diagnosis_code'] == diag_code]
                        if len(d_matches) > 5:
                            gnn_clusters.append(f"Diagnosa {diag_code}: {len(d_matches)} episode klaim terdaftar pada periode berjalan")

                    claim_ctx = ClaimContextBuilder.build_sanitized_context(
                        claim_row=selected_row,
                        shap_contributions=feature_contributions,
                        gnn_neighbors=gnn_clusters,
                        mask_sensitive=True
                    )

                    # ── ACTION BUTTONS ──
                    btn_bap_col, btn_rag_col = st.columns(2)
                    with btn_bap_col:
                        btn_gen_bap = st.button("📑 Generate Berkas BAP & Resume Medis", key=f"btn_bap_{selected_claim}", type="primary")
                    with btn_rag_col:
                        btn_view_rag = st.button("⚖️ Cek Dasar Regulasi Terkait (RAG)", key=f"btn_rag_{selected_claim}")

                    dossier_key = f"generated_bap_{selected_claim}"
                    rag_key = f"rag_results_{selected_claim}"

                    if btn_gen_bap:
                        # ── P1-3: Rate limit BAP generation (cooldown 2s) ──
                        _allowed, _remain = rate_limit_check(f"bap_gen_{selected_claim}", cooldown_seconds=2.0)
                        if not _allowed:
                            st.warning(f"⏳ Tunggu {_remain:.1f} detik sebelum generate BAP ulang (rate limit anti-spam).")
                        else:
                            _engine_label = provider_choice
                            with st.spinner(f"🤖 Menyusun BAP menggunakan **{_engine_label}**... (maks. ~30 detik untuk cloud LLM)"):
                                dossier_res = copilot_engine.generate_investigation_dossier(
                                    context=claim_ctx,
                                    investigator_name=auditor_name
                                )
                                # ── P1-2: LRU-capped session storage (max 20 klaim aktif) ──
                                lru_session_put("bap", dossier_key, dossier_res)

                    if btn_view_rag:
                        _allowed, _remain = rate_limit_check(f"rag_lookup_{selected_claim}", cooldown_seconds=1.0)
                        if not _allowed:
                            st.warning(f"⏳ Tunggu {_remain:.1f} detik sebelum cek regulasi ulang.")
                        else:
                            with st.spinner("🔍 Mencari pasal regulasi terkait di RAG Knowledge Base..."):
                                rag_kb = get_rag_knowledge_base()
                                # ── FIX #4: build a richer query including claim-specific codes ──
                                _active = claim_ctx.get("active_rules", [])
                                _svc    = claim_ctx.get("service_code", "")
                                _diag   = claim_ctx.get("diagnosis_code", "")
                                _rag_query = (
                                    " ".join(_active) + f" {_svc} {_diag}"
                                    if _active
                                    else f"deviasi biaya outlier statistik kewajaran tarif {_svc} {_diag}"
                                )
                                matched_docs = rag_kb.retrieve(query=_rag_query.strip(), top_k=3)
                                lru_session_put("rag_results", rag_key, matched_docs)

                    # ── RAG RESULTS PANEL ──
                    matched_docs = lru_session_get("rag_results", rag_key, None)
                    if matched_docs is not None:
                        st.markdown("---")
                        st.markdown("#### 📚 Referensi Regulasi Terkait (RAG Knowledge Base)")
                        if matched_docs:
                            for i, doc in enumerate(matched_docs, 1):
                                score_val = doc.get('similarity_score', 0.0)
                                score_badge = f"<span style='background:#dbeafe;color:#1e40af;font-size:0.75rem;padding:2px 8px;border-radius:10px;font-weight:700;'>Skor Relevansi: {score_val:.2f}</span>"
                                with st.expander(f"📋 [{i}] {doc.get('title', 'Regulasi')} — {doc.get('category', '')}", expanded=(i == 1)):
                                    st.markdown(f"""
                                    <div style="margin-bottom:8px;">{score_badge} &nbsp; <span style="font-size:0.75rem;color:#64748b;">Tags: {', '.join(doc.get('tags', []))}</span></div>
                                    <div style="background:#f8fafc;border-left:4px solid #3b82f6;border-radius:0 8px 8px 0;padding:10px 14px;font-size:0.87rem;color:#334155;line-height:1.6;">
                                    {doc.get('content', 'Konten tidak tersedia.')}
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("Tidak ditemukan regulasi yang relevan untuk aturan dan kode tindakan klaim ini.")

                    # ── DOSSIER / BAP REPORT PANEL ──
                    dossier_data = lru_session_get("bap", dossier_key, None)
                    if dossier_data is not None:
                        st.markdown("---")

                        # ── FIX #3: Clear LLM vs Heuristic visual indicator ────
                        _prov_used = str(dossier_data.get("provider_used", "heuristic")).lower()
                        _is_llm    = "heuristic" not in _prov_used and "fallback" not in _prov_used
                        _is_fallback = "fallback" in _prov_used

                        if _is_llm:
                            st.success(f"✅ Dokumen BAP dihasilkan oleh **LLM: {dossier_data.get('provider_used', '').upper()}**")
                        elif _is_fallback:
                            st.warning(
                                f"⚠️ LLM tidak merespons — BAP dihasilkan oleh **Heuristic Engine (Fallback)**. "
                                "Periksa API Key atau koneksi internet, lalu coba generate ulang."
                            )
                        else:
                            st.info("🔧 Dokumen BAP dihasilkan oleh **Heuristic Engine (Offline)** — hasil deterministik tanpa LLM.")

                        # Metadata strip above the report
                        meta_provider = str(dossier_data.get("provider_used", "heuristic")).upper()
                        meta_hash = dossier_data.get("audit_hash", "N/A")
                        meta_generated = dossier_data.get("generated_at", pd.Timestamp.now().strftime('%d %B %Y %H:%M:%S WIB'))
                        meta_rules_count = len(dossier_data.get("active_rules", []))

                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:10px 10px 0 0;padding:10px 18px;display:flex;flex-wrap:wrap;gap:14px;align-items:center;">
                            <div style="color:#f8fafc;font-size:0.78rem;font-weight:700;letter-spacing:0.5px;flex:1;min-width:200px;">
                                🗂️ <span style="opacity:0.7;">Berkas BAP</span> &nbsp;|&nbsp; Nomor: <code style="background:rgba(255,255,255,0.1);padding:2px 7px;border-radius:4px;font-size:0.75rem;">{dossier_data.get('dossier_number', f"BAP/{selected_claim}/...")}</code>
                            </div>
                            <div style="display:flex;flex-wrap:wrap;gap:8px;">
                                <span style="background:{'rgba(16,185,129,0.25)' if _is_llm else 'rgba(59,130,246,0.2)'};color:{'#34d399' if _is_llm else '#93c5fd'};border:1px solid {'rgba(16,185,129,0.4)' if _is_llm else 'rgba(59,130,246,0.3)'};border-radius:12px;padding:2px 10px;font-size:0.72rem;font-weight:600;">{'🤖' if _is_llm else '🔧'} Engine: {meta_provider}</span>
                                <span style="background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3);border-radius:12px;padding:2px 10px;font-size:0.72rem;font-weight:600;">⚠️ {meta_rules_count} Aturan Terpicu</span>
                                <span style="background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);border-radius:12px;padding:2px 10px;font-size:0.72rem;font-weight:600;">🔐 Hash: {meta_hash}</span>
                                <span style="background:rgba(148,163,184,0.1);color:#94a3b8;border:1px solid rgba(148,163,184,0.2);border-radius:12px;padding:2px 10px;font-size:0.72rem;">🕐 {meta_generated}</span>
                            </div>
                        </div>
                        <div style="background:#ffffff;border:1px solid #e2e8f0;border-top:none;border-radius:0 0 10px 10px;padding:20px 24px;">
                        """, unsafe_allow_html=True)

                        st.markdown(dossier_data.get("dossier_text", ""))

                        st.markdown("</div>", unsafe_allow_html=True)

                        st.download_button(
                            label="📥 Unduh Dokumen BAP (.md)",
                            data=dossier_data.get("dossier_text", ""),
                            file_name=f"BAP_{selected_claim}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            key=f"dl_bap_md_{selected_claim}"
                        )

                    # ── INTERACTIVE MULTI-TURN Q&A ──
                    st.markdown("---")
                    st.markdown("##### 💬 Tanya Copilot tentang Klaim Ini:")

                    qa_history_key = f"copilot_qa_history_{selected_claim}"
                    # ── P1-2: LRU-capped storage ──
                    _qa_hist = lru_session_get("qa_history", qa_history_key, None)
                    if _qa_hist is None:
                        _qa_hist = []
                        lru_session_put("qa_history", qa_history_key, _qa_hist)

                    # ── FIX #5: Q&A history controls — limit + clear button ──
                    QA_MAX = 10
                    if _qa_hist:
                        _qa_hdr_col, _qa_clr_col = st.columns([5, 1])
                        with _qa_hdr_col:
                            st.caption(f"Riwayat percakapan: **{len(_qa_hist)}** pertanyaan (maks. {QA_MAX})")
                        with _qa_clr_col:
                            if st.button("🗑️ Hapus", key=f"clear_qa_{selected_claim}", help="Hapus seluruh riwayat percakapan untuk klaim ini"):
                                lru_session_put("qa_history", qa_history_key, [])
                                st.rerun()

                        # Show only last QA_MAX exchanges; oldest are silently dropped
                        visible_hist = _qa_hist[-QA_MAX:]
                        for q_item in visible_hist:
                            _q_provider = str(q_item.get('provider_used', '')).upper()
                            _prov_badge = (
                                f"&nbsp;<span style='background:#eef2ff;color:#3730a3;font-size:0.65rem;padding:1px 6px;border-radius:8px;font-weight:700;'>{_q_provider}</span>"
                                if _q_provider else ""
                            )
                            st.markdown(
                                f"<div style='background:#f1f5f9;border-radius:8px;padding:8px 12px;margin-bottom:6px;font-size:0.85rem;'><b>👤 Auditor:</b> {q_item['question']}</div>",
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f"<div style='background:#f0f9ff;border:1px solid #bae6fd;border-left:4px solid #0284c7;border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:12px;font-size:0.87rem;'>{_prov_badge}&nbsp;&nbsp;{q_item['answer']}</div>",
                                unsafe_allow_html=True
                            )

                    q_col1, q_col2 = st.columns([4, 1])
                    with q_col1:
                        user_question = st.text_input(
                            "Pertanyaan audit:",
                            placeholder="Contoh: Apakah biaya klaim ini wajar untuk diagnosis tersebut?",
                            key=f"q_input_{selected_claim}",
                            label_visibility="collapsed",
                        )
                    with q_col2:
                        ask_clicked = st.button("Tanyakan", key=f"q_btn_{selected_claim}")

                    if ask_clicked and user_question and user_question.strip():
                        # ── P1-3: Rate limit Q&A (cooldown 2s) ──
                        _allowed, _remain = rate_limit_check(f"qa_{selected_claim}", cooldown_seconds=2.0)
                        if not _allowed:
                            st.warning(f"⏳ Tunggu {_remain:.1f} detik sebelum bertanya lagi (anti-spam).")
                        else:
                            with st.spinner("🤖 Menganalisis respon audit..."):
                                ans = copilot_engine.answer_investigator_query(context=claim_ctx, user_question=user_question.strip())
                                _qa_hist = lru_session_get("qa_history", qa_history_key, [])
                                # ── QA item sekarang menyertakan provider_used ──
                                _qa_hist.append({
                                    "question": user_question.strip(),
                                    "answer": ans,
                                    "provider_used": str(copilot_engine.provider if hasattr(copilot_engine, 'provider') else provider_choice).lower()
                                })
                                # Trim to QA_MAX so session state doesn't grow unbounded
                                if len(_qa_hist) > QA_MAX:
                                    _qa_hist = _qa_hist[-QA_MAX:]
                                lru_session_put("qa_history", qa_history_key, _qa_hist)
                                st.rerun()

        # ── TAB 5: Concept Drift ──
        with tab_drift:
            st.subheader("📈 Analisis Concept Drift & Auto-Retraining Quality Gate")
            st.caption("Pemeriksaan apakah distribusi data klaim baru telah bergeser secara signifikan dari data saat model dilatih (Kolmogorov-Smirnov Test).")

            if st.button("🔍 Deteksi Concept Drift Sekarang", key="btn_drift_check_tab", type="primary"):
                with st.spinner("Menganalisis pergeseran distribusi fitur numerik..."):
                    try:
                        from model_explainer import ConceptDriftDetector, AdaptiveLearningManager
                        
                        training_stats = (getattr(detector, 'training_metadata', {}) or {}).get('feature_medians', {})

                        # 1. Siapkan Current Data (inferensi saat ini) yang sudah dialignkan fiturnya
                        source_new_df = st.session_state.get('detection_processed_df', df_result)
                        aligned_current_df, _ = build_aligned_inference_features(
                            source_new_df, training_features, training_stats=training_stats
                        )
                        current_features_df = aligned_current_df[training_features].apply(pd.to_numeric, errors='coerce').fillna(0)

                        # 2. Siapkan Baseline Reference Data
                        if 'train_df' in st.session_state and st.session_state['train_df'] is not None:
                            aligned_ref_df, _ = build_aligned_inference_features(
                                st.session_state['train_df'], training_features, training_stats=training_stats
                            )
                            reference_data = aligned_ref_df[training_features].apply(pd.to_numeric, errors='coerce').fillna(0)
                            eval_data = current_features_df
                        else:
                            # Fallback jika model dimuat tanpa train_df di memori:
                            if len(current_features_df) >= 200:
                                split_idx = max(int(len(current_features_df) * 0.3), 100)
                                reference_data = current_features_df.iloc[:split_idx]
                                eval_data = current_features_df.iloc[split_idx:]
                                st.info(f"ℹ️ Menggunakan {len(reference_data):,} sampel awal sebagai baseline referensi perbandingan.")
                            else:
                                reference_data = current_features_df
                                eval_data = current_features_df
                                st.info("ℹ️ Baseline menggunakan batch sampel inferensi yang tersedia.")

                        drift_detector = ConceptDriftDetector(
                            reference_data=reference_data,
                            feature_names=training_features,
                            threshold=0.05
                        )
                        
                        drift_detected, drift_report = drift_detector.detect_drift(
                            eval_data,
                            method='ks_test'
                        )

                        if drift_detected:
                            st.warning("⚠️ **Concept Drift Terdeteksi!** Distribusi data baru berbeda signifikan dari data pelatihan.")
                        else:
                            st.success("✅ **Tidak ada Concept Drift Terdeteksi.** Distribusi data baru konsisten dengan data pelatihan model.")

                        drift_detector.plot_drift_report(drift_report)
                        st.session_state['last_drift_detected'] = drift_detected

                    except Exception as drift_err:
                        st.error(f"Gagal mendeteksi concept drift: {str(drift_err)}")
                        logger.error(f"Concept drift error: {drift_err}", exc_info=True)
