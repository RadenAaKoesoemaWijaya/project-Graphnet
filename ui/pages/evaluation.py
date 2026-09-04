import streamlit as st
import pandas as pd
import numpy as np
from ui.utils import *
from state_manager import *
from model_explainer import ModelExplainer, SUPPORTED_EXPLAINABILITY_MODELS

def show_evaluation_page():
    st.title("📊 Evaluasi Model")
    st.info("Evaluasi mengukur kualitas model pada data yang tidak digunakan untuk training. Gunakan metrik dan grafik ini untuk menilai apakah threshold deteksi sudah sesuai.")

    with st.expander("💡 Panduan Interpretasi Metrik & Dampak Bisnis", expanded=False):
        st.markdown("""
        - **ROC-AUC (Receiver Operating Characteristic)**: Mengukur ketepatan pemisahan klaim anomali vs wajar. Nilai > 0.85 menunjukkan kemampuan diskriminasi model yang sangat baik.
        - **Recall (Sensitivitas Fraud)**: Persentase fraud yang berhasil tertangkap. *Recall tinggi* sangat krusial dalam asuransi kesehatan guna meminimalkan klaim curang yang lolos pencairan (*False Negative*).
        - **Precision (Akurasi Prediksi)**: Ketepatan saat model memprediksi anomali. *Precision tinggi* mencegah klaim wajar tertahan review berlebihan (*False Positive*).
        - **Confusion Matrix**: Menampilkan tabulasi silang aktual vs prediksi untuk menentukan kalibrasi ambang batas (*threshold tuning*).
        """)

    detector = load_persisted_detector()
    if detector is None:
        st.error("❌ Model belum tersedia. Silakan training model terlebih dahulu.")
        logger.error("No detector available in evaluation page")
        if st.button("Kembali ke Training"):
            navigate_to_page('train')
        return

    training_features = (
        st.session_state.get('training_features')
        or getattr(detector, 'training_metadata', {}).get('training_features')
        or st.session_state.get('feature_columns', [])
    )
    if not training_features:
        st.error("❌ Metadata fitur training tidak tersedia.")
        return

    feature_selection_method = st.session_state.get(
        'feature_selection_method',
        getattr(detector, 'training_metadata', {}).get('feature_selection_method', 'Tidak diketahui')
    )
    training_mode = normalize_training_mode(st.session_state.get(
        'training_mode',
        getattr(detector, 'training_metadata', {}).get('training_mode', 'Tanpa supervisi')
    ))
    label_column = st.session_state.get(
        'training_label_column',
        getattr(detector, 'training_metadata', {}).get('label_column')
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    st.info(
        f"✅ Model siap dievaluasi dengan {len(training_features)} fitur "  # type: ignore
        f"({feature_selection_method}) pada mode {get_training_mode_label(training_mode)}."
    )
    st.write(f"Perangkat komputasi: {device}")

    st.subheader("📋 Data Evaluasi")
    if 'test_df' not in st.session_state:
        if 'df_processed' not in st.session_state and 'processed_data_hash' in st.session_state:
            file_hash = st.session_state['processed_data_hash']
            if hydrate_processed_data_reference_from_cache(file_hash):
                st.info("✅ Data berhasil dimuat dari cache!")
            else:
                st.error("❌ Data tidak ditemukan di cache. Silakan unggah dan praproses data terlebih dahulu.")
                if st.button("Kembali ke Unggah Data"):
                    navigate_to_page('collect')
                return
        elif 'df_processed_path' not in st.session_state:
            st.error("❌ Data belum tersedia. Silakan unggah dan praproses data terlebih dahulu.")
            if st.button("Kembali ke Unggah Data"):
                navigate_to_page('collect')
            return

        df_processed = get_df_processed()
        if df_processed is None:
            st.error("❌ Gagal memuat data hasil praproses untuk evaluasi.")
            return
        test_size = st.slider("Ukuran Test Set (%)", min_value=10, max_value=40, value=20, step=5)
        if st.button("🔄 Split Data untuk Evaluasi", key="split_eval_data"):
            try:
                stratify_data = None
                if training_mode == TRAINING_MODE_SUPERVISED and label_column in df_processed.columns:
                    label_series = pd.Series(df_processed[label_column]).fillna(0)
                    if label_series.nunique() == 2:
                        stratify_data = label_series

                train_df, test_df = train_test_split(
                    df_processed,
                    test_size=test_size / 100,
                    random_state=42,
                    stratify=stratify_data
                )
                st.session_state['train_df'] = train_df
                st.session_state['test_df'] = test_df
                st.success(f"✅ Data evaluasi siap: Data latih ({len(train_df)}), data uji ({len(test_df)})")
            except Exception as e:
                st.error(f"❌ Gagal membagi data evaluasi: {str(e)}")
                st.info("💡 Tips: Pastikan dataset memiliki cukup data dan label valid untuk stratified split.")
                # Clear invalid state
                st.session_state.pop('train_df', None)
                st.session_state.pop('test_df', None)

    if 'test_df' not in st.session_state:
        st.info("📝 Klik tombol split untuk menyiapkan data evaluasi.")
        return

    eval_df = st.session_state['test_df'].copy()
    X_eval_df, alignment_summary = build_aligned_inference_features(eval_df, training_features)
    st.session_state['X_eval_test'] = X_eval_df.values

    align_col1, align_col2, align_col3 = st.columns(3)
    with align_col1:
        st.metric("Fitur Training", alignment_summary['expected_features'])
    with align_col2:
        st.metric("Fitur Diturunkan", len(alignment_summary['derived_features']))
    with align_col3:
        st.metric("Fitur Diisi 0", len(alignment_summary['filled_zero_features']))

    if alignment_summary['filled_zero_features']:
        st.warning(
            f"⚠️ {len(alignment_summary['filled_zero_features'])} fitur tidak ditemukan dan diisi 0: "
            f"{', '.join(alignment_summary['filled_zero_features'][:5])}"
            f"{'...' if len(alignment_summary['filled_zero_features']) > 5 else ''}"
        )

    if st.button("🚀 Mulai Evaluasi Model", key="start_evaluation"):
        with st.spinner("Mengevaluasi model..."):
            edge_index = None
            edge_type = None
            if getattr(detector, "gnn_model", None) is not None and getattr(detector, "gnn_weight", 0) > 0:
                graph_frame = eval_df.copy()
                for feature_name in training_features:
                    graph_frame[feature_name] = X_eval_df[feature_name].to_numpy()
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
            probabilities, individual_probs = detector.predict_anomaly_probability(
                X_eval_df.values,
                edge_index=edge_index,
                edge_type=edge_type,
                device=device
            )
            predictions = (probabilities > 0.5).astype(int)

            result_df = eval_df.copy()
            result_df['anomaly_probability'] = probabilities
            result_df['anomaly_prediction'] = predictions

            st.session_state['eval_result_df'] = result_df
            st.session_state['eval_predictions'] = predictions
            st.session_state['eval_probabilities'] = probabilities
            st.session_state['individual_probs'] = individual_probs

            if training_mode == TRAINING_MODE_SUPERVISED and label_column in eval_df.columns:
                y_true = pd.Series(eval_df[label_column]).fillna(0).astype(int).values
                st.session_state['eval_y_true'] = y_true
            else:
                st.session_state['eval_y_true'] = None

            st.success("✅ Evaluasi selesai!")

    if 'eval_result_df' not in st.session_state:
        st.info("📝 Klik tombol evaluasi untuk melihat hasil.")
        return

    result_df = st.session_state['eval_result_df']
    test_predictions = st.session_state['eval_predictions']
    test_probabilities = st.session_state['eval_probabilities']
    individual_probs = st.session_state['individual_probs']
    y_true = st.session_state.get('eval_y_true')
    fraud_rate = float(np.mean(test_predictions))

    st.subheader("📊 Hasil Evaluasi")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transaksi", len(result_df))
    with col2:
        st.metric("Prediksi Anomali", int(np.sum(test_predictions)))
    with col3:
        st.metric("Tingkat Anomali", f"{fraud_rate:.2%}")
    with col4:
        st.metric("Rata-rata Probabilitas", f"{np.mean(test_probabilities):.3f}")

    fig = create_probability_distribution(result_df['anomaly_probability'],
                                        title="Distribusi Probabilitas Anomali",
                                        threshold=0.5)
    st.plotly_chart(fig, width='stretch')

    st.subheader("🔍 Performa Individual Algoritma")
    algorithm_rows = []
    for algorithm_name, probs in individual_probs.items():
        algorithm_rows.append({
            'Algoritma': algorithm_name,
            'Rata-rata Probabilitas': float(np.mean(probs)),
            'Simpangan Baku': float(np.std(probs)),
            'Tingkat Anomali @ 0.5': float(np.mean(np.array(probs) > 0.5))
        })
    algo_df = pd.DataFrame(algorithm_rows)
    st.dataframe(algo_df, width='stretch')
    
    # GNN-specific metrics if available
    if 'gnn' in individual_probs and hasattr(detector, 'gnn_model') and detector.gnn_model is not None:
        st.subheader("🕸️ Performa GNN Spesifik")
        gnn_probs = individual_probs['gnn']
        
        gnn_col1, gnn_col2, gnn_col3, gnn_col4 = st.columns(4)
        with gnn_col1:
            st.metric("GNN Mean Probability", f"{float(np.mean(gnn_probs)):.4f}")
        with gnn_col2:
            st.metric("GNN Std Probability", f"{float(np.std(gnn_probs)):.4f}")
        with gnn_col3:
            st.metric("GNN Min Probability", f"{float(np.min(gnn_probs)):.4f}")
        with gnn_col4:
            st.metric("GNN Max Probability", f"{float(np.max(gnn_probs)):.4f}")
        
        # GNN contribution to ensemble
        if detector.gnn_weight > 0:
            st.info(f"🎯 GNN contributes {detector.gnn_weight*100:.1f}% to the ensemble prediction")
        
        # GNN probability distribution
        with st.expander("📊 Distribusi Probabilitas GNN"):
            fig_gnn = create_probability_distribution(gnn_probs,
                                                      title="Distribusi Probabilitas GNN",
                                                      threshold=0.5)
            st.plotly_chart(fig_gnn, width='stretch')

    if y_true is not None and len(np.unique(y_true)) == 2:
        st.subheader("✅ Metrik Supervised (Ground Truth Nyata)")
        try:
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                accuracy = (test_predictions == y_true).mean()
                st.metric("Akurasi", f"{accuracy:.4f}")
            with metrics_col2:
                precision = precision_score(y_true, test_predictions, zero_division=0)
                st.metric("Presisi", f"{precision:.4f}")
            with metrics_col3:
                recall = recall_score(y_true, test_predictions, zero_division=0)
                st.metric("Recall", f"{recall:.4f}")
            with metrics_col4:
                f1 = f1_score(y_true, test_predictions, zero_division=0)
                st.metric("Skor F1", f"{f1:.4f}")

            try:
                roc_auc_value = roc_auc_score(y_true, test_probabilities)
                st.metric("ROC-AUC", f"{roc_auc_value:.4f}")
            except Exception as roc_error:
                st.warning(f"⚠️ Tidak dapat menghitung ROC-AUC: {str(roc_error)}")

            cm = confusion_matrix(y_true, test_predictions)
            cm_df = pd.DataFrame(cm, index=['Aktual 0', 'Aktual 1'], columns=['Pred 0', 'Pred 1'])
            st.dataframe(cm_df, width='stretch')

            report = classification_report(y_true, test_predictions, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Kelas'})
            st.dataframe(report_df, width='stretch')
        except Exception as metrics_error:
            st.error(f"❌ Error dalam perhitungan metrik supervisi: {str(metrics_error)}")
            st.info("Dataset mungkin tidak memiliki distribusi kelas yang valid untuk metrik ini.")
    else:
        st.subheader("🧪 Analisis Diagnostik")
        st.info(
            "Dataset evaluasi tidak memiliki label acuan yang valid, sehingga halaman ini "
            "menampilkan analisis diagnostik distribusi skor, bukan metrik supervisi seperti ROC-AUC."
        )

        diagnostic_stats = pd.DataFrame([
            {'Metrik': 'Rata-rata Probabilitas', 'Nilai': float(np.mean(test_probabilities))},
            {'Metrik': 'Median Probabilitas', 'Nilai': float(np.median(test_probabilities))},
            {'Metrik': 'Simpangan Baku Probabilitas', 'Nilai': float(np.std(test_probabilities))},
            {'Metrik': 'Probabilitas Minimum', 'Nilai': float(np.min(test_probabilities))},
            {'Metrik': 'Probabilitas Maksimum', 'Nilai': float(np.max(test_probabilities))}
        ])
        st.dataframe(diagnostic_stats, width='stretch')

    st.subheader("🎯 Analisis Threshold")
    threshold_rows = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        threshold_preds = (test_probabilities > threshold).astype(int)
        threshold_rows.append({
            'Ambang': round(float(threshold), 2),
            'Jumlah Anomali': int(np.sum(threshold_preds)),
            'Tingkat Anomali': float(np.mean(threshold_preds))
        })
    threshold_df = pd.DataFrame(threshold_rows)
    st.dataframe(threshold_df, width='stretch')

    st.subheader("🔬 Analisis Fitur")
    feature_impact = []
    for feature in training_features:
        feature_values = X_eval_df[feature].values
        if np.std(feature_values) > 0:
            correlation = np.corrcoef(feature_values, test_probabilities)[0, 1]
            correlation = 0.0 if np.isnan(correlation) else float(correlation)
        else:
            correlation = 0.0
        feature_impact.append({
            'Fitur': feature,
            'Korelasi dengan Anomali': correlation,
            'Nilai Rata-rata': float(np.mean(feature_values)),
            'Simpangan Baku': float(np.std(feature_values))
        })
    feature_df = pd.DataFrame(feature_impact).sort_values(
        'Korelasi dengan Anomali',
        key=lambda s: s.abs(),
        ascending=False
    )
    st.dataframe(feature_df.head(10), width='stretch')
    
    # Model Explainability Section
    st.markdown("---")
    st.subheader("🧠 Explainability AI (SHAP)")

    available_explainability_models = []
    for model_name in sorted(SUPPORTED_EXPLAINABILITY_MODELS):
        if model_name == 'isolation_forest' and getattr(detector, 'isolation_forest', None) is not None:
            available_explainability_models.append(model_name)
        elif model_name == 'xgboost' and getattr(detector, 'xgboost_model', None) is not None:
            available_explainability_models.append(model_name)

    explainability_compatible = bool(available_explainability_models)

    if not explainability_compatible:
        st.warning(
            "⚠️ Analisis Feature Importance hanya didukung untuk model Isolation Forest dan XGBoost. "
            "Latih ulang model dengan algoritma yang kompatibel untuk mengaktifkan tombol ini."
        )

    if st.button(
        "🔍 Analisis Feature Importance dengan SHAP",
        key="shap_analysis",
        disabled=not explainability_compatible,
        help="Tombol ini otomatis dinonaktifkan jika model yang dipilih tidak kompatibel dengan explainability."
    ):
        with st.spinner("Menginisialisasi SHAP explainer..."):
            try:
                # Initialize explainer
                explainer = ModelExplainer(detector=detector, feature_names=training_features)
                
                # Use training data as background
                if 'train_df' in st.session_state:
                    background_data = st.session_state['train_df'][training_features].values
                else:
                    # Use evaluation data as fallback
                    background_data = X_eval_df.values[:100]
                
                # Initialize explainers
                if explainer.initialize_explainers(background_data):
                    st.success("✅ SHAP explainers initialized successfully")
                    
                    # Show feature importance for available models
                    col_expl1, col_expl2 = st.columns(2)
                    
                    with col_expl1:
                        if 'isolation_forest' in explainer.explainers:
                            st.subheader("Isolation Forest Feature Importance")
                            explainer.plot_feature_importance('isolation_forest', X=X_eval_df.values, max_features=10)
                    
                    with col_expl2:
                        if 'xgboost' in explainer.explainers:
                            st.subheader("XGBoost Feature Importance")
                            explainer.plot_feature_importance('xgboost', X=X_eval_df.values, max_features=10)
                    
                    # SHAP summary plot
                    if 'isolation_forest' in explainer.explainers or 'xgboost' in explainer.explainers:
                        st.subheader("SHAP Summary Plot")
                        model_to_plot = 'isolation_forest' if 'isolation_forest' in explainer.explainers else 'xgboost'
                        explainer.plot_shap_summary(X_eval_df.values, model_to_plot, max_display=10)
                    
                    # Store explainer in session for later use
                    st.session_state['model_explainer'] = explainer
                    
            except Exception as e:
                st.error(f"Error in SHAP analysis: {str(e)}")
                st.info("SHAP analysis requires additional dependencies. Install with: pip install shap matplotlib")
    
    # Performance Monitoring Section
    st.markdown("---")
    st.subheader("📊 Monitoring Performa")
    
    # Initialize performance monitor if not exists
    if 'performance_monitor' not in st.session_state:
        st.session_state['performance_monitor'] = PerformanceMonitor()
    
    # Initialize adaptive learning manager if not exists
    if 'adaptive_learning_manager' not in st.session_state:
        st.session_state['adaptive_learning_manager'] = AdaptiveLearningManager(detector=detector)
    
    # Log current evaluation metrics
    if st.session_state['eval_y_true'] is not None:
        current_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc_value if 'roc_auc_value' in locals() else 0.0
        }
        st.session_state['performance_monitor'].log_performance(current_metrics)
        
        # Show current metrics
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        with col_perf1:
            st.metric("Akurasi Saat Ini", f"{accuracy:.4f}")
        with col_perf2:
            st.metric("F1 Score Saat Ini", f"{f1:.4f}")
        with col_perf3:
            trend = st.session_state['performance_monitor'].get_performance_trend('f1_score')
            trend_emoji = {'improving': '📈', 'degrading': '📉', 'stable': '➡️', 'insufficient_data': '❓'}.get(trend, '❓')
            st.metric("Trend F1 Score", f"{trend_emoji} {trend}")
        
        # Show performance history if available
        if len(st.session_state['performance_monitor'].performance_history) > 1:
            with st.expander("📈 Riwayat Performa"):
                st.session_state['performance_monitor'].plot_performance_history(['accuracy', 'precision', 'recall', 'f1_score'])
        
        # Check if retraining is needed
        adaptive_manager = st.session_state['adaptive_learning_manager']
        should_retrain, retrain_reason = adaptive_manager.should_retrain(current_metrics)
        
        if should_retrain:
            st.error(f"⚠️ {retrain_reason}")
            if st.button("🔄 Retrain Model Sekarang", key="retrain_now"):
                navigate_to_page('train')
        else:
            st.success(f"✅ {retrain_reason}")
    else:
        st.info("Monitoring performa tersedia untuk mode supervised dengan label valid")
    
    # Adaptive Learning Section
    st.markdown("---")
    st.subheader("🔄 Adaptive Learning & Feedback")
    
    adaptive_manager = st.session_state['adaptive_learning_manager']
    
    # Show feedback collection UI
    with st.expander("💬 Kumpulkan Feedback Prediksi"):
        st.write("Berikan feedback pada prediksi untuk meningkatkan akurasi model di masa depan")
        
        # Select a prediction to give feedback on
        if 'eval_result_df' in st.session_state:
            sample_predictions = st.session_state['eval_result_df'].head(10)
            selected_idx = st.selectbox(
                "Pilih prediksi untuk diberi feedback:",
                range(len(sample_predictions)),
                format_func=lambda x: f"Baris {x} - Probabilitas: {sample_predictions.iloc[x]['anomaly_probability']:.3f}"
            )
            
            col_fb1, col_fb2, col_fb3 = st.columns(3)
            with col_fb1:
                feedback_type = st.selectbox("Feedback:", ["correct", "incorrect", "uncertain"])
            with col_fb2:
                actual_label = st.selectbox("Label Aktual:", [0, 1])
            with col_fb3:
                if st.button("Kirim Feedback"):
                    adaptive_manager.collect_feedback(
                        prediction_id=f"eval_{selected_idx}",
                        features=X_eval_df.iloc[selected_idx].values,
                        prediction=sample_predictions.iloc[selected_idx]['anomaly_prediction'],
                        actual_label=actual_label,
                        feedback_type=feedback_type
                    )
                    st.success("✅ Feedback berhasil dikumpulkan")
        
        # Show feedback summary
        adaptive_manager.plot_feedback_summary()
    
    # Show retraining history
    with st.expander("📜 Riwayat Adaptive Learning"):
        adaptive_manager.plot_retraining_history()

    st.markdown("---")
    st.subheader("🚀 Langkah Selanjutnya")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Lanjut ke Deteksi Anomali", key="eval_to_detection", type="primary"):
            navigate_to_page('detect')
    with col2:
        if st.button("🧠 Kembali ke Training", key="eval_to_training"):
            navigate_to_page('train')

