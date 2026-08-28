import streamlit as st
import pandas as pd
import numpy as np
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

    st.markdown("---")
    st.subheader("🚨 Business Risk Dashboard")

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
        st.metric("Total Klaim", total_claims)
    with col2:
        st.metric("Repeat Billing", repeat_count)
    with col3:
        st.metric("Phantom Service", phantom_count)
    with col4:
        st.metric("Risk Rate", f"{risk_score}%")

    col1, col2 = st.columns(2)
    with col1:
        if repeat_results.empty:
            st.success("✅ Tidak ditemukan pola repeat billing pada data ini.")
        else:
            st.warning(f"⚠️ Ditemukan {len(repeat_results)} potensi repeat billing")
            top_repeat = repeat_results.sort_values("risk_score", ascending=False).head(5)
            st.dataframe(top_repeat[["first_claim_id", "repeat_claim_id", "time_gap_days", "similarity_score", "risk_score", "detection_reason"]], use_container_width=True)

    with col2:
        if phantom_df.empty:
            st.success("✅ Tidak ditemukan indikasi phantom service pada data ini.")
        else:
            st.warning(f"⚠️ Ditemukan {len(phantom_df)} potensi phantom service")
            st.dataframe(phantom_df, use_container_width=True)

    if not repeat_results.empty or not phantom_df.empty:
        st.info("📌 Kombinasi repeat billing + phantom service berfungsi sebagai sinyal prioritas review untuk tim fraud analyst.")


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
    high_risk_claims = int(risk_summary.get('high_risk_claims', risk_summary.get('final_high_risk_claims', 0)))
    risk_cards = [
        ("Total Klaim", total_claims),
        ("Anomali", anomaly_claims),
        ("High Risk", high_risk_claims),
        ("Repeat Billing", int(risk_summary.get('repeat_billing_cases', 0))),
        ("Phantom", int(risk_summary.get('phantom_service_cases', 0))),
        ("Provider Capacity", int(risk_summary.get('provider_capacity_issues', 0))),
        ("Duplicate Payment", int(risk_summary.get('duplicate_payment_claims', 0))),
        ("Upcoding", int(risk_summary.get('upcoding_unbundling_cases', 0))),
        ("Cloning", int(risk_summary.get('inflated_bill_cloning_cases', 0))),
        ("Stay Risk", int(risk_summary.get('prolonged_stay_readmission_cases', 0))),
        ("Med/Device", int(risk_summary.get('medication_device_fraud_cases', 0))),
    ]
    return risk_cards


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
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔍 Deteksi Anomali Transaksi")
    st.info("Deteksi menggabungkan skor model hasil training dengan rule-based detection seperti repeat billing, phantom service, kapasitas provider, duplicate payment, dan rule fraud lainnya.")

    # Upload model zip file
    st.subheader("📤 Impor Model (Opsional)")
    uploaded_zip = st.file_uploader(
        "Unggah file ZIP model (jika ingin menggunakan model yang dibagikan):",
        type=["zip"],
        help="Unggah file ZIP model yang telah diunduh dari halaman pelatihan"
    )
    if uploaded_zip is not None:
        try:
            import zipfile
            from io import BytesIO

            # Ensure models directory exists
            os.makedirs(os.path.dirname(MODEL_PREFIX), exist_ok=True)

            # Extract zip file
            with zipfile.ZipFile(BytesIO(uploaded_zip.read()), "r") as zip_ref:
                extraction_root = os.path.realpath(os.path.dirname(MODEL_PREFIX))
                for member in zip_ref.infolist():
                    member_path = os.path.realpath(os.path.join(extraction_root, member.filename))
                    if os.path.commonpath([extraction_root, member_path]) != extraction_root:
                        raise ValueError(f"Path ZIP tidak aman: {member.filename}")
                    if member.is_dir():
                        os.makedirs(member_path, exist_ok=True)
                        continue
                    os.makedirs(os.path.dirname(member_path), exist_ok=True)
                    with zip_ref.open(member) as source, open(member_path, "wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)

            st.success("✅ Model berhasil diekstrak! Memuat model...")
            # Clear session state to reload the new model
            for key in ['detector', 'model_trained', 'training_features', 'feature_selection_method', 'training_mode', 'training_label_column']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        except Exception as e:
            st.error(f"❌ Gagal mengekstrak model: {str(e)}")
            logger.error(f"Failed to extract model ZIP: {e}", exc_info=True)

    st.markdown("---")

    detector = load_persisted_detector()
    if detector is None:
        st.error("❌ Model belum tersedia. Silakan training model terlebih dahulu atau unggah model di atas.")
        logger.error("No detector available in detection page")
        if st.button("Kembali ke Training"):
            navigate_to_page('train')
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    training_features = (
        st.session_state.get('training_features')
        or getattr(detector, 'training_metadata', {}).get('training_features')
        or st.session_state.get('feature_columns', [])
    )
    feature_selection_method = st.session_state.get(
        'feature_selection_method',
        getattr(detector, 'training_metadata', {}).get('feature_selection_method', 'Tidak diketahui')
    )
    if not training_features:
        st.error("❌ Metadata fitur training tidak tersedia, sehingga deteksi tidak dapat dilanjutkan.")
        return
    st.success(f"✅ Model terlatih berhasil dimuat! Menggunakan {len(training_features)} fitur ({feature_selection_method})")  # type: ignore
    
    st.write(f"Perangkat komputasi: {device}")
    st.write(f"Bobot model - Isolation Forest: {detector.isolation_weight}, Autoencoder: {detector.autoencoder_weight}, XGBoost: {detector.xgboost_weight}")
    
    st.markdown("""
    ### Deteksi Anomali pada Data Transaksi
    
    Gunakan model yang telah dilatih untuk mendeteksi anomali pada data transaksi asuransi.
    """)
    
    # ── Template dataset download ────────────────────────────────────────────
    st.subheader("📋 Panduan Format Data & Template")
    with st.expander("📖 Kolom yang Diperlukan untuk Deteksi Optimal", expanded=False):
        st.markdown(
            "Dataset Anda harus menyertakan kolom-kolom berikut agar seluruh **9 Modul Aturan Bisnis** "
            "dan **GNN Graf Relasi** dapat berjalan secara penuh."
        )
        guide_rows = [
            {"Kolom": c,
             "Keterangan": COLUMN_DESCRIPTIONS.get(c, ""),
             "Modul yang Bergantung": ", ".join(COLUMN_RULE_DEPENDENCIES.get(c, ["-"]))}
            for c in TEMPLATE_CORE_COLUMNS
        ]
        st.dataframe(pd.DataFrame(guide_rows), use_container_width=True, hide_index=True)

    template_df  = generate_sample_claims_template(n_rows=5)
    template_csv = template_df.to_csv(index=False).encode("utf-8")
    col_tmpl1, col_tmpl2 = st.columns([1, 3])
    with col_tmpl1:
        st.download_button(
            label="⬇️ Unduh Template Dataset (CSV)",
            data=template_csv,
            file_name="astina_claim_template.csv",
            mime="text/csv",
            key="btn_download_astina_claim_template",
            help="Unduh template CSV dengan format kolom standar. Isi dengan data klaim Anda sebelum mengunggah.",
        )
    with col_tmpl2:
        st.caption(
            "Template berisi 5 baris contoh dengan format kolom lengkap. "
            "Anda dapat menambahkan baris baru atau menggantinya dengan data aktual."
        )

    st.markdown("---")

    # ── Batch dataset upload ─────────────────────────────────────────────────
    st.subheader("📂 Unggah Dataset Klaim (Batch)")
    st.info(
        "Deteksi anomali memerlukan **dataset batch** (lebih dari 1 baris) agar modul statistik, "
        "GNN, dan aturan temporal bekerja secara akurat. "
        "Format yang didukung: **CSV, XLSX, XLS, Parquet**."
    )

    if 'enable_large_file_handling' not in st.session_state:
        st.session_state['enable_large_file_handling'] = True
    st.checkbox(
        "Aktifkan pemrosesan bertahap untuk dataset besar",
        key='enable_large_file_handling',
        help="Disarankan aktif untuk mencegah error memori pada data besar."
    )

    uploaded_file = st.file_uploader(
        "Unggah file dataset klaim asuransi:",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Gunakan template yang diunduh di atas sebagai referensi format kolom.",
    )

    if uploaded_file is not None:
        try:
            # Detect format from file extension
            file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
            fmt_map   = {"csv": "csv", "xlsx": "excel", "xls": "excel", "parquet": "parquet"}
            file_format = fmt_map.get(file_ext, "csv")

            df_new = read_file_with_optimization(uploaded_file, file_format)
            if df_new is None or len(df_new) == 0:
                st.error("❌ File tidak dapat dibaca atau kosong.")
                return

            if len(df_new) < 2:
                st.error(
                    "❌ Dataset harus memiliki minimal **2 baris data** agar analisis statistik, "
                    "GNN, dan aturan temporal dapat berjalan dengan benar."
                )
                return

            st.success(f"✅ File berhasil diunggah: **{len(df_new):,} baris**, {len(df_new.columns)} kolom")

            # ── Schema Readiness Card ───────────────────────────────────────
            st.subheader("🩺 Kesiapan Skema Data")
            schema_info = render_schema_readiness_card(df_new)

            # Preprocess data
            df_processed, feature_columns, preprocessing_metadata_new = preprocess_insurance_claims_optimized(
                df_new,
                enable_large_file_handling=st.session_state['enable_large_file_handling'],
                enable_outlier_detection=st.session_state.get('enable_outlier_detection', True),
                enable_data_validation=st.session_state.get('enable_data_validation', True)
            )

            # --- CONCEPT DRIFT DETECTION ---
            # Compare distribution of new data with training data
            st.info("📊 Pemeriksaan Concept Drift: Membandingkan distribusi data baru dengan data latih...")

            if st.button("🔍 Deteksi Concept Drift", key="detect_drift"):
                with st.spinner("Menganalisis perubahan distribusi data..."):
                    try:
                        # Get reference data from training metadata if available
                        if 'train_df' in st.session_state:
                            reference_data = st.session_state['train_df'][training_features]
                        else:
                            st.warning("Data latih tidak tersedia untuk perbandingan. Menggunakan data baru sebagai baseline.")
                            reference_data = df_processed[training_features].head(1000)

                        # Initialize drift detector
                        drift_detector = ConceptDriftDetector(
                            reference_data=reference_data,
                            feature_names=training_features,
                            threshold=0.05
                        )

                        # Detect drift
                        drift_detected, drift_report = drift_detector.detect_drift(
                            df_processed[training_features],
                            method='ks_test'
                        )

                        # Display results
                        if drift_detected:
                            st.error("⚠️ Concept Drift Terdeteksi!")
                            st.warning("Distribusi data baru berbeda signifikan dari data latih.")

                            # QA Automated Retraining Option
                            st.markdown("### 🔄 Automated Retraining Pipeline (Optuna + Stratified K-Fold CV)")
                            auto_retrain = st.checkbox(
                                "Picu Retraining Otomatis & Evaluasi Champion vs Challenger",
                                value=True,
                                help="Memicu pelatihan ulang model baru dengan Optuna hyperparameter tuning dan Stratified K-Fold CV untuk adaptasi terhadap pola klaim baru."
                            )

                            if auto_retrain and st.button("🚀 Jalankan Auto-Retraining Sekarang", key="btn_auto_retrain"):
                                with st.spinner("Menjalankan retraining model penantang (Challenger) dengan Optuna & K-Fold CV..."):
                                    try:
                                        from model_explainer import AdaptiveLearningManager
                                        adaptive_mgr = AdaptiveLearningManager(detector=st.session_state.get('model'))

                                        retrain_res = drift_detector.check_and_trigger_retraining(
                                            new_data=df_processed[training_features],
                                            adaptive_manager=adaptive_mgr,
                                            min_drift_feature_pct=0.10,
                                            optuna_n_trials=15,
                                            optuna_timeout=90
                                        )

                                        if retrain_res.get('retraining_triggered'):
                                            res = retrain_res.get('retraining_result', {})
                                            if res.get('status') == 'promoted':
                                                st.success("✅ **Model Challenger Lolos QA Quality Gate & Berhasil Di-deploy!**")
                                                st.session_state['model'] = adaptive_mgr.detector
                                                st.session_state['adaptive_learning_manager'] = adaptive_mgr

                                                gate = res.get('gate_result', {})
                                                c_m  = gate.get('champion_metrics', {})
                                                ch_m = gate.get('challenger_metrics', {})

                                                m_col1, m_col2, m_col3 = st.columns(3)
                                                with m_col1:
                                                    st.metric("FPR Model", f"{ch_m.get('fpr', 0):.2%}", delta=f"{(ch_m.get('fpr', 0) - c_m.get('fpr', 0)):.2%}", delta_color="inverse")
                                                with m_col2:
                                                    st.metric("Recall Model", f"{ch_m.get('recall', 0):.2%}", delta=f"{(ch_m.get('recall', 0) - c_m.get('recall', 0)):.2%}")
                                                with m_col3:
                                                    st.metric("F1-Score", f"{ch_m.get('f1', 0):.2%}", delta=f"{(ch_m.get('f1', 0) - c_m.get('f1', 0)):.2%}")
                                            else:
                                                st.warning(f"⚠️ Model Challenger ditolak oleh Quality Gate: {res.get('message', 'Performa belum melampaui Champion')}")
                                    except Exception as err:
                                        st.error(f"Gagal mengeksekusi auto-retraining: {str(err)}")
                        else:
                            st.success("✅ Tidak ada Concept Drift Terdeteksi")
                            st.info("Distribusi data baru konsisten dengan data latih model.")

                        # Show drift report visualization
                        drift_detector.plot_drift_report(drift_report)

                        # Store drift detector in session
                        st.session_state['drift_detector'] = drift_detector
                        st.session_state['last_drift_detected'] = drift_detected

                    except Exception as e:
                        st.error(f"Error dalam deteksi concept drift: {str(e)}")
                        st.info("Deteksi concept drift memerlukan scipy. Install dengan: pip install scipy")

            # Show drift status if previously detected
            if 'last_drift_detected' in st.session_state:
                drift_status = "⚠️ Terdeteksi" if st.session_state['last_drift_detected'] else "✅ Tidak Terdeteksi"
                st.metric("Status Concept Drift Terakhir", drift_status)

            df = df_processed
            st.session_state['uploaded_data'] = df
            st.session_state['feature_columns_new'] = feature_columns
            st.session_state['preprocessing_metadata_new'] = preprocessing_metadata_new

        except Exception as e:
            st.error(f"Gagal memproses file: {str(e)}")
            logger.error(f"Detection page preprocessing error: {e}", exc_info=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Coba Upload Ulang", key="retry_upload"):
                    st.session_state.pop('uploaded_data', None)
                    st.session_state.pop('preprocessing_metadata_new', None)
                    st.rerun()
            with col2:
                if st.button("🏠 Kembali ke Beranda", key="detection_to_home"):
                    navigate_to_page('home')
            return

    else:
        st.info("📤 Silakan unggah file dataset klaim (CSV / XLSX / XLS / Parquet) untuk memulai deteksi.")
        return

    if 'uploaded_data' not in st.session_state:
        return

    df = st.session_state['uploaded_data']
    feature_columns = st.session_state.get('feature_columns_new', [])

    # ── Strict feature alignment ────────────────────────────────────────────
    st.subheader("🔧 Adaptasi Fitur untuk Deteksi")
    incoming_feature_count = len(feature_columns)
    # Retrieve training-time median stats from detector metadata (if available)
    training_stats: dict = (
        getattr(detector, 'training_metadata', {}) or {}
    ).get('feature_medians', {})
    aligned_feature_df, alignment_summary = build_aligned_inference_features(
        df, training_features, training_stats=training_stats
    )
    feature_columns = training_features

    comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
    with comparison_col1:
        st.metric("Fitur Training", alignment_summary['expected_features'])
    with comparison_col2:
        st.metric("Fitur Data Baru", incoming_feature_count)
    with comparison_col3:
        st.metric("Fitur Diisi 0", len(alignment_summary['filled_zero_features']))

    st.info(
        f"✅ Inferensi akan memakai tepat {len(feature_columns)} fitur sesuai training "
        f"({len(alignment_summary['existing_features'])} langsung, "
        f"{len(alignment_summary['derived_features'])} diturunkan, "
        f"{len(alignment_summary['filled_features'])} diimputasi dari statistik training)."
    )

    if alignment_summary['filled_zero_features']:
        st.warning(
            f"⚠️ Fitur yang tidak bisa direkonstruksi dan diisi 0: "
            f"{', '.join(alignment_summary['filled_zero_features'][:5])}"
            f"{'...' if len(alignment_summary['filled_zero_features']) > 5 else ''}"
        )

    with st.expander("📋 Pemetaan Fitur Training → Data Baru"):
        mapping_data = []
        for feat in training_features:
            if feat in alignment_summary['existing_features']:
                status = "✅ Langsung tersedia"
            elif feat in alignment_summary['derived_features']:
                status = "🛠️ Diturunkan otomatis"
            else:
                status = "⚠️ Diisi 0"
            mapping_data.append({
                'Fitur Training': feat,
                'Status': status
            })
        st.dataframe(pd.DataFrame(mapping_data), width='stretch')
    
    st.write(f"Jumlah fitur yang digunakan: {len(feature_columns)}")
    
    # Detection threshold
    st.subheader("⚙️ Konfigurasi Deteksi")
    
    threshold = st.slider("Threshold Deteksi Anomali:", 0.1, 0.9, 0.5, 0.05)
    
    # Run detection
    if st.button("🔍 Deteksi Anomali", type="primary"):
        with st.spinner("Mendeteksi anomali..."):
            # Prepare features
            X = aligned_feature_df[feature_columns].values
            edge_index = None
            edge_type = None
            if getattr(detector, "gnn_model", None) is not None and getattr(detector, "gnn_weight", 0) > 0:
                graph_frame = df.copy()
                for feature_name in feature_columns:
                    graph_frame[feature_name] = aligned_feature_df[feature_name].to_numpy()
                graph_metadata = getattr(detector, 'training_metadata', {}) or {}
                graph_method = st.session_state.get('graph_method', graph_metadata.get('graph_method', 'star'))
                graph_kwargs = {}
                if graph_method == 'knn' and graph_metadata.get('graph_k'):
                    graph_kwargs['k'] = int(graph_metadata['graph_k'])
                graph_result = create_claim_graph(
                    graph_frame,
                    feature_columns,
                    method=graph_method,
                    max_nodes=min(len(graph_frame), 20000),
                    **graph_kwargs,
                )
                if len(graph_result) == 3:
                    _, edge_index, edge_type = graph_result
                else:
                    _, edge_index = graph_result
            
            # Predict anomaly
            probabilities, individual_probs = detector.predict_anomaly_probability(
                X, edge_index=edge_index, edge_type=edge_type, device=device
            )  # type: ignore[union-attr]
            predictions = (probabilities > threshold).astype(int)
            
            # Add results to dataframe
            df_result = df.copy()
            df_result['anomaly_probability'] = probabilities
            df_result['anomaly_prediction'] = predictions
            df_result['isolation_forest_score'] = individual_probs['isolation_forest']
            df_result['autoencoder_score'] = individual_probs['autoencoder']
            if 'dbscan' in individual_probs:
                df_result['dbscan_score'] = individual_probs['dbscan']
            df_result['xgboost_score'] = individual_probs.get('xgboost', np.zeros(len(df_result)))

        from fraud_risk_pipeline import run_integrated_claim_risk_pipeline
        df_risk, risk_summary = run_integrated_claim_risk_pipeline(df_result)
        df_result = df_risk
        df_result['business_risk_score'] = df_result.get('business_risk_score', 0.0)
        df_result['final_risk_score'] = df_result.get('final_risk_score', df_result['anomaly_probability'])
        df_result['final_risk_flag'] = df_result.get('final_risk_flag', (df_result['final_risk_score'] >= 0.65).astype(int))

        st.session_state['detection_results'] = df_result
        st.session_state['detection_threshold'] = threshold
        st.session_state['risk_summary'] = risk_summary

        # Summary statistics
        total_claims = len(df_result)
        anomaly_claims = df_result['anomaly_prediction'].sum()
        anomaly_rate = anomaly_claims / total_claims
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transaksi", total_claims)
        with col2:
            st.metric("Transaksi Anomali", anomaly_claims)
        with col3:
            st.metric("Tingkat Anomali", f"{anomaly_rate:.2%}")
        with col4:
            st.metric("Ambang Deteksi", f"{threshold:.2f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution
            pred_counts = df_result['anomaly_prediction'].value_counts()
            fig = create_bar_chart(['Normal', 'Anomali'], pred_counts.values,
                       title='Distribusi Prediksi Anomali',
                       labels={'x': 'Kategori', 'y': 'Jumlah'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Probability distribution
            fig = create_histogram_chart(df_result, 'anomaly_probability', nbins=50,
                             title='Distribusi Skor Anomali')
            fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                         annotation_text=f"Ambang: {threshold}")
            st.plotly_chart(fig, width='stretch')
        
        show_repeat_phantom_insights(df_result)

        st.subheader("📊 Executive Summary Panel")
        risk_summary = st.session_state.get('risk_summary', {})
        summary_cards = _build_safety_summary(df_result, risk_summary)
        summary_cols = st.columns(3)
        for idx, (label, value) in enumerate(summary_cards):
            with summary_cols[idx % 3]:
                st.markdown(
                    f"<div style='border:1px solid #dfe3e8; border-radius:10px; padding:0.8rem; background:#f8fafc; text-align:center; margin-bottom:0.6rem;'>"
                    f"<div style='font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:0.04em;'>{label}</div>"
                    f"<div style='font-size:1.6rem; font-weight:700; color:#0f172a; margin-top:0.2rem;'>{value}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.subheader("� Distribusi Risiko per Kategori")
        category_chart_df = display_df.copy() if 'display_df' in locals() else df_result.copy()
        if 'risk_category' not in category_chart_df.columns:
            category_chart_df['risk_category'] = category_chart_df.apply(_derive_risk_category, axis=1)
        category_counts = category_chart_df['risk_category'].value_counts().reset_index()
        category_counts.columns = ['Risk Category', 'Count']
        category_counts = category_counts[category_counts['Count'] > 0]
        if not category_counts.empty:
            chart_col_left, chart_col_right = st.columns([1.4, 1])
            with chart_col_left:
                if len(category_counts) > 1:
                    risk_fig = px.bar(
                        category_counts,
                        x='Risk Category',
                        y='Count',
                        color='Risk Category',
                        title='Jumlah Claim per Kategori Risiko',
                        color_discrete_sequence=['#0f172a', '#2563eb', '#f59e0b', '#ef4444', '#8b5cf6', '#10b981', '#14b8a6', '#f97316']
                    )
                    risk_fig.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(risk_fig, use_container_width=True)
                else:
                    st.info("Data kategorisasi risiko masih terbatas untuk chart distribusi.")
            with chart_col_right:
                pie_fig = px.pie(
                    category_counts,
                    names='Risk Category',
                    values='Count',
                    title='Proporsi Risiko',
                    color='Risk Category',
                    color_discrete_sequence=['#2563eb', '#f59e0b', '#ef4444', '#8b5cf6', '#10b981', '#14b8a6', '#f97316', '#64748b']
                )
                pie_fig.update_traces(textinfo='percent+label', hole=0.35)
                st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("Belum ada kategori risiko yang aktif untuk chart distribusi.")

        st.markdown("---")
        st.subheader("�📋 Fraud Review Table")

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

        filters = st.columns(6)
        with filters[0]:
            risk_group = st.selectbox("Kategori risiko", ["Semua", "High Risk", "Normal", "Anomali", "Repeat Billing", "Phantom Service", "Provider Capacity", "Duplicate Payment", "Upcoding", "Inflated Bill / Cloning", "Prolonged Stay", "Medication / Device"], index=0)
        with filters[1]:
            provider_values = ["Semua"] + sorted(display_df['provider_id'].astype(str).dropna().unique().tolist()[:50])
            provider_filter = st.selectbox("Provider", provider_values, index=0)
        with filters[2]:
            service_values = ["Semua"] + sorted(display_df['service_code'].astype(str).dropna().unique().tolist()[:50])
            service_filter = st.selectbox("Service", service_values, index=0)
        with filters[3]:
            show_only_fraud = st.checkbox("Hanya claim berisiko", value=False)
        with filters[4]:
            risk_band = st.selectbox("Severity", ["Semua", "High", "Medium", "Low"], index=0)
        with filters[5]:
            sort_order = st.selectbox("Urutan", ['Menurun', 'Menaik'], index=0)

        if risk_group == "High Risk":
            display_df = display_df[display_df['final_risk_flag'] == 1]
        elif risk_group == "Normal":
            display_df = display_df[display_df['final_risk_flag'] == 0]
        elif risk_group == "Anomali":
            display_df = display_df[display_df.get('anomaly_prediction', 0).astype(int) == 1]
        elif risk_group != "Semua":
            display_df = display_df[display_df['risk_category'] == risk_group]

        if provider_filter != "Semua":
            display_df = display_df[display_df['provider_id'].astype(str).str.contains(str(provider_filter), case=False, na=False)]
        if service_filter != "Semua":
            display_df = display_df[display_df['service_code'].astype(str).str.contains(str(service_filter), case=False, na=False)]
        if show_only_fraud:
            display_df = display_df[(display_df['final_risk_flag'] == 1) | (display_df.get('anomaly_prediction', 0).astype(int) == 1)]
        if risk_band != "Semua":
            display_df = display_df[display_df['severity'] == risk_band]

        sort_candidates = [
            col for col in ['final_risk_score', 'business_risk_score', 'anomaly_probability', 'xgboost_score', 'isolation_forest_score', 'autoencoder_score', 'amount', 'billed_amount', 'patient_age', 'age']
            if col in display_df.columns
        ]
        sort_by = st.selectbox("Urutkan berdasarkan:", sort_candidates or ['final_risk_score'])
        display_df = display_df.sort_values(by=sort_by, ascending=(sort_order == 'Menaik'))

        display_columns = [
            'claim_id', 'patient_id', 'provider_id', 'service_code', 'amount', 'anomaly_probability',
            'final_risk_score', 'business_risk_score', 'severity', 'risk_category', 'final_risk_flag', 'anomaly_prediction'
        ]
        for col in ['repeat_billing_flag', 'phantom_service_flag', 'duplicate_payment_flag', 'upcoding_unbundling_flag', 'inflated_bill_cloning_flag', 'prolonged_stay_readmission_flag', 'medication_device_fraud_flag', 'status_message']:
            if col in display_df.columns:
                display_columns.append(col)

        display_columns = [col for col in display_columns if col in display_df.columns]
        
        col_table_info, col_table_limit = st.columns([3, 1])
        with col_table_limit:
            preview_limit = st.selectbox("Tampilkan baris:", [50, 100, 250, 500], index=1)
        with col_table_info:
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

        st.markdown("---")
        st.subheader("🔎 Detail Review Per Claim")

        if not display_df.empty:
            claim_choices = ["-- Pilih claim untuk detail --"] + display_df['claim_id'].astype(str).dropna().unique().tolist()
            selected_claim = st.selectbox("Pilih claim", claim_choices, index=0)
            if selected_claim != "-- Pilih claim untuk detail --":
                selected_row = display_df[display_df['claim_id'].astype(str) == str(selected_claim)].iloc[0]
                detail_reasons = []
                for flag_col, label in [
                    ('repeat_billing_flag', 'Repeat Billing'),
                    ('phantom_service_flag', 'Phantom Service'),
                    ('duplicate_payment_flag', 'Duplicate Payment'),
                    ('upcoding_unbundling_flag', 'Upcoding / Unbundling'),
                    ('inflated_bill_cloning_flag', 'Inflated Bill / Cloning'),
                    ('prolonged_stay_readmission_flag', 'Prolonged Stay / Readmission'),
                    ('medication_device_fraud_flag', 'Medication / Device Fraud'),
                ]:
                    if flag_col in selected_row and pd.notna(selected_row.get(flag_col)) and int(selected_row.get(flag_col, 0)) == 1:
                        detail_reasons.append(label)
                if not detail_reasons:
                    detail_reasons.append('No explicit rule triggered; anomaly score reviewed manually')

                evidence_items = []
                for field in ['provider_id', 'service_code', 'amount', 'anomaly_probability', 'final_risk_score', 'business_risk_score', 'severity', 'risk_category']:
                    if field in selected_row and pd.notna(selected_row.get(field)):
                        evidence_items.append({"Field": field, "Value": selected_row.get(field)})

                with st.expander(f"Detail claim {selected_claim}", expanded=True):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Overall Risk", f"{float(selected_row.get('final_risk_score', 0.0)):.2f}")
                    with col_b:
                        st.metric("Anomaly Score", f"{float(selected_row.get('anomaly_probability', 0.0)):.2f}")
                    with col_c:
                        st.metric("Severity", str(selected_row.get('severity', 'Low')))

                    st.subheader("Reasoning")
                    st.write("• " + "\n• ".join(detail_reasons))

                    st.subheader("Evidence")
                    st.dataframe(pd.DataFrame(evidence_items), use_container_width=True, hide_index=True)

                    st.subheader("Risk Flags")
                    flag_rows = []
                    for flag_col, label in [
                        ('repeat_billing_flag', 'Repeat Billing'),
                        ('phantom_service_flag', 'Phantom Service'),
                        ('duplicate_payment_flag', 'Duplicate Payment'),
                        ('upcoding_unbundling_flag', 'Upcoding / Unbundling'),
                        ('inflated_bill_cloning_flag', 'Inflated Bill / Cloning'),
                        ('prolonged_stay_readmission_flag', 'Prolonged Stay / Readmission'),
                        ('medication_device_fraud_flag', 'Medication / Device Fraud'),
                    ]:
                        flag_rows.append({"Category": label, "Flag": int(selected_row.get(flag_col, 0) or 0)})
                    st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)

                    # ── AGENTIC AI INVESTIGATOR COPILOT & BAP DRAFTING ────────
                    st.markdown("---")
                    st.subheader("🤖 AI Investigator Copilot & Case Dossier Generator")
                    st.caption(
                        "Gunakan Agentic AI berbasis RAG untuk menyusun **Berita Acara Pemeriksaan (BAP)**, "
                        "mengkaji dasar regulasi (*Permenkes, INA-CBGs, FORNAS*), dan merumuskan rekomendasi audit."
                    )

                    copilot_config_cols = st.columns([1.5, 2, 1.5])
                    with copilot_config_cols[0]:
                        provider_choice = st.selectbox(
                            "LLM Engine",
                            ["Heuristic Engine (Offline)", "Google Gemini", "OpenAI / Azure", "Local Ollama"],
                            key=f"copilot_provider_{selected_claim}"
                        )
                    with copilot_config_cols[1]:
                        api_key_input = ""
                        if provider_choice in ["Google Gemini", "OpenAI / Azure"]:
                            api_key_input = st.text_input(
                                "API Key",
                                type="password",
                                help="Masukkan API Key untuk LLM. Jika dikosongkan, Copilot otomatis menggunakan Heuristic Fallback.",
                                key=f"copilot_key_{selected_claim}"
                            )
                        elif provider_choice == "Local Ollama":
                            st.caption("Endpoint: `http://localhost:11434/api/generate` (Model: `llama3`)")
                    with copilot_config_cols[2]:
                        investigator_input = st.text_input(
                            "Nama Auditor",
                            value="Investigator Senior ASTINA",
                            key=f"copilot_auditor_{selected_claim}"
                        )

                    provider_map = {
                        "Heuristic Engine (Offline)": "heuristic",
                        "Google Gemini": "gemini",
                        "OpenAI / Azure": "openai",
                        "Local Ollama": "ollama"
                    }
                    selected_provider = provider_map.get(provider_choice, "heuristic")

                    copilot_engine = AgenticInvestigatorCopilot(
                        provider=selected_provider,
                        api_key=api_key_input,
                        investigator_name=investigator_input if 'investigator_name' in locals() else "Investigator Senior ASTINA"
                    )

                    # Build sanitized context (PII Masked)
                    claim_ctx = ClaimContextBuilder.build_sanitized_context(
                        claim_row=selected_row,
                        shap_contributions=st.session_state.get('shap_contributions', {}),
                        mask_sensitive=True
                    )

                    action_col1, action_col2 = st.columns(2)
                    with action_col1:
                        btn_gen_bap = st.button("📑 Generate Draf BAP & Resume Medis", key=f"btn_bap_{selected_claim}", type="primary", use_container_width=True)
                    with action_col2:
                        btn_view_rag = st.button("⚖️ Cek Dasar Regulasi Terkait (RAG)", key=f"btn_rag_{selected_claim}", use_container_width=True)

                    # Session key for generated dossier
                    dossier_key = f"generated_bap_{selected_claim}"

                    if btn_gen_bap:
                        with st.spinner("🤖 Menyusun Berita Acara Pemeriksaan (BAP) dengan Agentic AI & RAG..."):
                            dossier_res = copilot_engine.generate_investigation_dossier(
                                context=claim_ctx,
                                investigator_name=investigator_input
                            )
                            st.session_state[dossier_key] = dossier_res

                    if btn_view_rag:
                        with st.spinner("🔍 Mencari pasal regulasi relevan di RAG Knowledge Base..."):
                            rag_kb = get_rag_knowledge_base()
                            matched_docs = rag_kb.retrieve(
                                query=" ".join(claim_ctx.get("active_rules", [])) + f" {claim_ctx.get('service_code')} {claim_ctx.get('diagnosis_code')}",
                                top_k=3
                            )
                            st.markdown("#### 📚 Regulasi Terkait Kasus Ini (RAG Knowledge Base)")
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
                            key=f"dl_bap_{selected_claim}"
                        )

                    # Interactive Q&A for Investigator
                    st.markdown("##### 💬 Tanya Copilot tentang Klaim Ini")
                    q_col, btn_q_col = st.columns([4, 1])
                    with q_col:
                        custom_q = st.text_input(
                            "Pertanyaan investigasi:",
                            placeholder="Contoh: Apakah kombinasi diagnosis dan tindakan ini lazim untuk rawat jalan?",
                            key=f"custom_q_{selected_claim}",
                            label_visibility="collapsed"
                        )
                    with btn_q_col:
                        btn_ask = st.button("Tanyakan", key=f"btn_ask_{selected_claim}", use_container_width=True)

                    if btn_ask and custom_q:
                        with st.spinner("🤖 Menganalisis..."):
                            answer = copilot_engine.answer_investigator_query(
                                context=claim_ctx,
                                user_question=custom_q
                            )
                            st.success(answer)

        # Download results
        st.subheader("💾 Unduh Hasil")

        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="Unduh Hasil Deteksi (CSV)",
            data=csv_data,
            file_name=f"anomaly_detection_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Main application
