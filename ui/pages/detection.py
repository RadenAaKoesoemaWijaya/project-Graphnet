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
            st.dataframe(top_repeat[avail_cols], use_container_width=True)

    with col2:
        if phantom_df.empty:
            st.success("✅ Tidak ditemukan indikasi phantom service pada dataset ini.")
        else:
            st.warning(f"⚠️ Ditemukan {len(phantom_df)} potensi phantom service")
            st.dataframe(phantom_df.head(10), use_container_width=True)

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
        st.error("❌ Metadata fitur training tidak ditemukan di model tersimpan.")
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
                file_format = fmt_map.get(file_ext, file_ext)
                raw_df = read_file_with_optimization(uploaded_file, file_format)
                source_description = f"File: {uploaded_file.name} ({len(raw_df):,} baris)"
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {str(e)}")
                logger.error(f"Detection file read error: {e}", exc_info=True)

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
        st.dataframe(raw_df.head(5), use_container_width=True)

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
            use_container_width=True,
            key="btn_run_detection_primary"
        )
    with btn_col2:
        if st.button("🔄 Reset Hasil Deteksi", use_container_width=True, key="btn_reset_detection"):
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
                training_stats: dict = (
                    getattr(detector, 'training_metadata', {}) or {}
                ).get('feature_medians', {})
                aligned_feature_df, alignment_summary = build_aligned_inference_features(
                    df_processed, training_features, training_stats=training_stats
                )

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
                    X, edge_index=edge_index, edge_type=edge_type, device=device
                )
                predictions = (probabilities > threshold).astype(int)

                # 5. Build results DataFrame
                df_result = raw_df.copy()
                df_result['anomaly_probability'] = pd.Series(probabilities, index=df_result.index, dtype=float)
                df_result['anomaly_prediction'] = pd.Series(predictions, index=df_result.index, dtype=int)
                df_result['isolation_forest_score'] = pd.Series(individual_probs.get('isolation_forest', np.zeros(len(df_result))), index=df_result.index, dtype=float)
                df_result['autoencoder_score'] = pd.Series(individual_probs.get('autoencoder', np.zeros(len(df_result))), index=df_result.index, dtype=float)
                if 'dbscan' in individual_probs:
                    df_result['dbscan_score'] = pd.Series(individual_probs['dbscan'], index=df_result.index, dtype=float)
                df_result['xgboost_score'] = pd.Series(individual_probs.get('xgboost', np.zeros(len(df_result))), index=df_result.index, dtype=float)

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
                st.success("✅ **Deteksi anomali berhasil dieksekusi!** Hasil disajikan pada dashboard di bawah.")

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
                st.plotly_chart(fig_pred, use_container_width=True)

            with v_col2:
                fig_hist = create_histogram_chart(df_result, 'anomaly_probability', nbins=40,
                                                 title='Distribusi Skor Anomali (Multi-Model Ensemble)')
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                                   annotation_text=f"Threshold: {threshold:.2f}")
                st.plotly_chart(fig_hist, use_container_width=True)

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
                    st.plotly_chart(risk_fig, use_container_width=True)
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
                    st.plotly_chart(pie_fig, use_container_width=True)

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

            st.dataframe(fraud_table.style.apply(highlight_fraud, axis=1), use_container_width=True)

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

            if not display_df.empty:
                claim_choices = ["-- Pilih Klaim untuk Diinvestigasi --"] + display_df['claim_id'].astype(str).dropna().unique().tolist() if 'claim_id' in display_df.columns else ["-- Data klaim tidak memiliki claim_id --"]
                selected_claim = st.selectbox("Pilih nomor klaim:", claim_choices, index=0, key="copilot_claim_select")

                if selected_claim != "-- Pilih Klaim untuk Diinvestigasi --" and selected_claim != "-- Data klaim tidak memiliki claim_id --":
                    selected_row = display_df[display_df['claim_id'].astype(str) == str(selected_claim)].iloc[0]

                    # Metrics for selected claim
                    sc_1, sc_2, sc_3, sc_4 = st.columns(4)
                    with sc_1:
                        st.metric("Final Risk Score", f"{float(selected_row.get('final_risk_score', 0.0)):.2f}")
                    with sc_2:
                        st.metric("Anomaly Probability", f"{float(selected_row.get('anomaly_probability', 0.0)):.2f}")
                    with sc_3:
                        st.metric("Severity", str(selected_row.get('severity', 'Low')))
                    with sc_4:
                        st.metric("Kategori", str(selected_row.get('risk_category', 'Normal')))

                    # Reasoning details
                    detail_reasons = []
                    for flag_col, label in [
                        ('repeat_billing_flag', 'Repeat Billing (Klaim berulang dalam jangka waktu dekat)'),
                        ('phantom_service_flag', 'Phantom Service (Tindakan/obat fiktif tanpa dasar medis)'),
                        ('duplicate_payment_flag', 'Duplicate Payment (Ganda pembayaran klaim)'),
                        ('upcoding_unbundling_flag', 'Upcoding / Unbundling (Pemecahan tagihan atau kenaikan kode diagnosis)'),
                        ('inflated_bill_cloning_flag', 'Inflated Bill / Cloning (Kloning klaim atau penggelembungan biaya)'),
                        ('prolonged_stay_readmission_flag', 'Prolonged Stay / Readmission (Masa rawat tidak wajar atau rawat ulang)'),
                        ('medication_device_fraud_flag', 'Medication / Device Fraud (Pemberian obat/alkes tidak lazim)'),
                    ]:
                        if flag_col in selected_row and pd.notna(selected_row.get(flag_col)) and int(selected_row.get(flag_col, 0)) == 1:
                            detail_reasons.append(label)
                    if not detail_reasons:
                        detail_reasons.append('Tidak ada aturan eksplisit terpicu; anomali statistik terdeteksi oleh ensemble ML.')

                    st.markdown("**Indikator Pelanggaran & Bukti:**")
                    st.write("• " + "\n• ".join(detail_reasons))

                    # Copilot Settings
                    st.markdown("---")
                    st.markdown("#### 🛠️ Konfigurasi Copilot")
                    c_cfg1, c_cfg2, c_cfg3 = st.columns([1.5, 2, 1.5])
                    with c_cfg1:
                        provider_choice = st.selectbox(
                            "LLM Engine:",
                            ["Heuristic Engine (Offline)", "Google Gemini", "OpenAI / Azure", "Local Ollama"],
                            key=f"c_prov_{selected_claim}"
                        )
                    with c_cfg2:
                        api_key_input = ""
                        if provider_choice in ["Google Gemini", "OpenAI / Azure"]:
                            api_key_input = st.text_input("API Key:", type="password", key=f"c_key_{selected_claim}")
                    with c_cfg3:
                        auditor_name = st.text_input("Nama Verifikator/Auditor:", value="Investigator Senior ASTINA", key=f"c_auditor_{selected_claim}")

                    provider_map = {
                        "Heuristic Engine (Offline)": "heuristic",
                        "Google Gemini": "gemini",
                        "OpenAI / Azure": "openai",
                        "Local Ollama": "ollama"
                    }
                    copilot_engine = AgenticInvestigatorCopilot(
                        provider=provider_map.get(provider_choice, "heuristic"),
                        api_key=api_key_input
                    )

                    claim_ctx = ClaimContextBuilder.build_sanitized_context(
                        claim_row=selected_row,
                        shap_contributions=st.session_state.get('shap_contributions', {}),
                        mask_sensitive=True
                    )

                    btn_bap_col, btn_rag_col = st.columns(2)
                    with btn_bap_col:
                        btn_gen_bap = st.button("📑 Generate Berkas BAP & Resume Medis", key=f"btn_bap_{selected_claim}", type="primary", use_container_width=True)
                    with btn_rag_col:
                        btn_view_rag = st.button("⚖️ Cek Dasar Regulasi Terkait (RAG)", key=f"btn_rag_{selected_claim}", use_container_width=True)

                    dossier_key = f"generated_bap_{selected_claim}"

                    if btn_gen_bap:
                        with st.spinner("🤖 Menyusun Berita Acara Pemeriksaan (BAP)..."):
                            dossier_res = copilot_engine.generate_investigation_dossier(
                                context=claim_ctx,
                                investigator_name=auditor_name
                            )
                            st.session_state[dossier_key] = dossier_res

                    if btn_view_rag:
                        with st.spinner("🔍 Mencari pasal regulasi terkait di RAG Knowledge Base..."):
                            rag_kb = get_rag_knowledge_base()
                            matched_docs = rag_kb.retrieve(
                                query=" ".join(claim_ctx.get("active_rules", [])) + f" {claim_ctx.get('service_code')} {claim_ctx.get('diagnosis_code')}",
                                top_k=3
                            )
                            st.markdown("##### 📚 Regulasi Medis Terkait:")
                            for doc in matched_docs:
                                st.info(f"**{doc['title']}** ({doc['category']})\n\n{doc['content']}")

                    if dossier_key in st.session_state:
                        dossier_data = st.session_state[dossier_key]
                        st.markdown("---")
                        st.markdown(dossier_data.get("dossier_text", ""))
                        st.download_button(
                            label="📥 Unduh Dokumen BAP (.md)",
                            data=dossier_data.get("dossier_text", ""),
                            file_name=f"BAP_{selected_claim}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            key=f"dl_bap_md_{selected_claim}"
                        )

                    # Interactive Q&A
                    st.markdown("---")
                    st.markdown("##### 💬 Tanya Copilot tentang Klaim Ini:")
                    q_col1, q_col2 = st.columns([4, 1])
                    with q_col1:
                        user_question = st.text_input("Pertanyaan audit:", placeholder="Contoh: Apakah biaya klaim ini wajar untuk diagnosis tersebut?", key=f"q_input_{selected_claim}", label_visibility="collapsed")
                    with q_col2:
                        ask_clicked = st.button("Tanyakan", key=f"q_btn_{selected_claim}", use_container_width=True)

                    if ask_clicked and user_question:
                        with st.spinner("🤖 Menganalisis respon audit..."):
                            ans = copilot_engine.answer_investigator_query(context=claim_ctx, user_question=user_question)
                            st.success(ans)

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
