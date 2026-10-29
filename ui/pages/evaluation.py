import streamlit as st
import pandas as pd
import numpy as np
from ui.utils import *
from state_manager import *
from model_explainer import ModelExplainer, SUPPORTED_EXPLAINABILITY_MODELS
from fraud_detection_metrics import (
    calculate_all_fraud_metrics,
    get_threshold_tradeoff_analysis,
    get_optimal_threshold
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
import logging
import io

logger = logging.getLogger(__name__)

def plotly_to_bytes(fig, format='png', width=1200, height=800, scale=2):
    """Convert a Plotly figure to bytes for PDF embedding."""
    try:
        img_bytes = fig.to_image(format=format, width=width, height=height, scale=scale)
        return img_bytes
    except Exception as e:
        logger.warning(f"Failed to convert plotly figure to bytes: {e}")
        return None

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
    preprocessing_metadata = st.session_state.get("preprocessing_metadata", {})
    model_preprocessing_contract = (
        getattr(detector, "training_metadata", {}) or {}
    ).get("preprocessing_contract_version")
    if (
        preprocessing_metadata.get("preprocessing_contract_version") != 1
        or model_preprocessing_contract != 1
    ):
        st.error(
            "Evaluasi hanya tersedia untuk dataset dan model dengan kontrak preprocessing "
            "train-only. Proses ulang data mentah dan latih ulang model sebelum evaluasi."
        )
        return
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
        if "__astina_partition" not in df_processed.columns:
            st.error(
                "Dataset evaluasi ini tidak memiliki split yang ditetapkan sebelum "
                "preprocessing. Unggah ulang data mentah dan jalankan preprocessing "
                "train-only sebelum evaluasi."
            )
            return
        else:
            try:
                # Get custom split proportions from session state or use defaults
                test_size = st.session_state.get('split_test_size', 0.2)
                validation_size = st.session_state.get('split_validation_size', 0.15)

                train_df, validation_df, test_df, stratify_label = split_processed_dataset_with_validation(
                    df_processed,
                    test_size=test_size,
                    validation_size=validation_size
                )
                st.session_state['train_df'] = train_df
                st.session_state['validation_df'] = validation_df
                st.session_state['test_df'] = test_df
                st.success(
                    f"✅ Partisi yang sama dari data mentah digunakan: latih ({len(train_df)}), "
                    f"validasi ({len(validation_df)}), test ({len(test_df)}); "
                    f"stratifikasi: {stratify_label or 'tidak tersedia'}"
                )
            except Exception as e:
                st.error(f"❌ Gagal menggunakan partisi data yang telah diproses: {str(e)}")
                st.session_state.pop('train_df', None)
                st.session_state.pop('validation_df', None)
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

    threshold_col1, threshold_col2 = st.columns([3, 1])
    with threshold_col1:
        eval_threshold = st.slider(
            "🎚️ Threshold Klasifikasi Anomali",
            min_value=0.10,
            max_value=0.95,
            value=float(st.session_state.get('eval_threshold', 0.5)),
            step=0.05,
            help="Untuk deteksi fraud, threshold rendah = recall tinggi (lebih banyak tertangkap tapi lebih banyak false positive). Threshold tinggi = precision tinggi."
        )
        st.session_state['eval_threshold'] = eval_threshold
    with threshold_col2:
        st.metric("Threshold Aktif", f"{eval_threshold:.2f}")
        validation_df = st.session_state.get("validation_df")
        validation_available = isinstance(validation_df, pd.DataFrame) and not validation_df.empty
        if not validation_available:
            st.warning(
                "Threshold tuning tidak tersedia: dataset validasi independen tidak ditemukan. "
                "Latih ulang model dengan pembagian train/validation/test."
            )
        if st.button(
            "🧠 Auto-Optimasi (ImbalanceHandler)",
            key="optimize_thr_btn",
            disabled=(
                training_mode != TRAINING_MODE_SUPERVISED
                or label_column not in eval_df.columns
                or not validation_available
            ),
        ):
            try:
                if hasattr(detector, 'imbalance_handler') and detector.imbalance_handler is not None:
                    X_validation, _ = build_aligned_inference_features(
                        validation_df, training_features
                    )
                    _X_temp = X_validation.values
                    _y_temp = pd.Series(validation_df[label_column]).fillna(0).astype(int).values
                    if len(np.unique(_y_temp)) != 2:
                        raise ValueError("Data validasi harus memiliki kedua kelas label.")
                    _prob_temp, _ = detector.predict_anomaly_probability(_X_temp, edge_index=None, edge_type=None, device=device)
                    best_thr, _ = detector.imbalance_handler.optimize_threshold(_y_temp, _prob_temp)
                    st.session_state['eval_threshold'] = float(best_thr)
                    st.success(f"✅ Threshold optimal: {best_thr:.3f}")
                    st.rerun()
                else:
                    st.warning("⚠️ ImbalanceHandler tidak tersedia pada detector.")
            except Exception as _e:
                st.error(f"⚠️ Gagal optimasi threshold: {str(_e)}")

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
            _active_thr = float(st.session_state.get('eval_threshold', 0.5))
            predictions = (probabilities > _active_thr).astype(int)

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
    _active_thr = float(st.session_state.get('eval_threshold', 0.5))
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
                                        threshold=_active_thr)
    st.plotly_chart(fig, width='stretch')

    st.subheader("🔍 Performa Individual Algoritma")
    algorithm_rows = []
    for algorithm_name, probs in individual_probs.items():
        algorithm_rows.append({
            'Algoritma': algorithm_name,
            'Rata-rata Probabilitas': float(np.mean(probs)),
            'Simpangan Baku': float(np.std(probs)),
            f'Tingkat Anomali @ {_active_thr:.2f}': float(np.mean(np.array(probs) > _active_thr))
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
                                                      threshold=_active_thr)
            st.plotly_chart(fig_gnn, width='stretch')

    _has_supervised_labels = (y_true is not None and len(np.unique(y_true)) == 2)

    if _has_supervised_labels:
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

        # Fraud Detection Specific Metrics
        st.markdown("---")
        st.subheader("🎯 Metrik Khusus Deteksi Fraud")

        with st.expander("💡 Mengapa Metrik Ini Penting?", expanded=False):
            st.markdown("""
            - **PR-AUC**: Lebih relevan untuk data imbalanced daripada ROC-AUC, fokus pada performa kelas positif (fraud)
            - **F2-Score**: Memberi bobot 2x pada recall - krusial untuk fraud karena missing fraud lebih mahal daripada false positive
            - **Specificity**: Persentase non-fraud yang benar diklasifikasikan
            - **False Negative Rate**: Persentase fraud yang lolos (critical metric untuk asuransi)
            - **Matthews Correlation Coefficient (MCC)**: Metrik yang lebih robust untuk data imbalanced
            """)

        try:
            fraud_metrics = calculate_all_fraud_metrics(y_true, test_predictions, test_probabilities)

            fraud_col1, fraud_col2, fraud_col3, fraud_col4 = st.columns(4)
            with fraud_col1:
                st.metric("PR-AUC", f"{fraud_metrics.get('pr_auc', 0):.4f}",
                         help="Precision-Recall AUC - lebih informatif untuk data imbalanced")
            with fraud_col2:
                st.metric("F2-Score", f"{fraud_metrics.get('f2_score', 0):.4f}",
                         help="Memberi bobot 2x pada recall - ideal untuk fraud detection")
            with fraud_col3:
                st.metric("Specificity", f"{fraud_metrics.get('specificity', 0):.4f}",
                         help="True Negative Rate - persentase non-fraud yang benar diklasifikasikan")
            with fraud_col4:
                st.metric("MCC", f"{fraud_metrics.get('mcc', 0):.4f}",
                         help="Matthews Correlation Coefficient - robust untuk data imbalanced")

            # Critical fraud metrics
            st.markdown("#### ⚠️ Metrik Kritis untuk Bisnis")
            fraud_crit_col1, fraud_crit_col2 = st.columns(2)
            with fraud_crit_col1:
                fnr = fraud_metrics.get('false_negative_rate', 0)
                st.metric("False Negative Rate", f"{fnr:.2%}",
                         delta_color="inverse",
                         help="Persentase fraud yang lolos - lebih rendah lebih baik")
            with fraud_crit_col2:
                fpr = fraud_metrics.get('false_positive_rate', 0)
                st.metric("False Positive Rate", f"{fpr:.2%}",
                         help="Persentase non-fraud yang salah diklasifikasikan")

            # Confusion matrix details
            st.markdown("#### 📊 Detail Confusion Matrix")
            cm_details = pd.DataFrame([
                {'Kategori': 'True Negatives (Non-fraud benar)', 'Jumlah': fraud_metrics.get('true_negatives', 0)},
                {'Kategori': 'False Positives (Non-fraud salah)', 'Jumlah': fraud_metrics.get('false_positives', 0)},
                {'Kategori': 'False Negatives (Fraud lolos)', 'Jumlah': fraud_metrics.get('false_negatives', 0)},
                {'Kategori': 'True Positives (Fraud tertangkap)', 'Jumlah': fraud_metrics.get('true_positives', 0)},
            ])
            st.dataframe(cm_details, width='stretch', hide_index=True)

        except Exception as fraud_metrics_error:
            st.error(f"❌ Error dalam perhitungan metrik fraud: {str(fraud_metrics_error)}")
            logger.error(f"Error calculating fraud metrics: {fraud_metrics_error}", exc_info=True)

        # Threshold Trade-off Analysis
        st.markdown("---")
        st.subheader("📈 Analisis Trade-off Threshold")

        with st.expander("💡 Cara Menggunakan Analisis Ini", expanded=False):
            st.markdown("""
            Analisis ini menunjukkan bagaimana precision dan recall berubah pada berbagai threshold.
            - **Threshold rendah**: Recall tinggi (lebih banyak fraud tertangkap) tapi precision rendah (banyak false positive)
            - **Threshold tinggi**: Precision tinggi (sedikit false positive) tapi recall rendah (banyak fraud lolos)
            - Pilih threshold berdasarkan prioritas bisnis Anda!
            """)

        try:
            tradeoff_data = get_threshold_tradeoff_analysis(y_true, test_probabilities)

            # Create trade-off plot
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig_tradeoff = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Precision vs Recall', 'F1 & F2 Scores vs Threshold',
                               'Precision vs Threshold', 'Recall vs Threshold'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # Precision vs Recall
            fig_tradeoff.add_trace(
                go.Scatter(x=tradeoff_data['recall'], y=tradeoff_data['precision'],
                          mode='lines+markers', name='Precision-Recall Curve',
                          line=dict(color='blue')),
                row=1, col=1
            )
            fig_tradeoff.add_trace(
                go.Scatter(x=[recall], y=[precision],
                          mode='markers', name='Current Threshold',
                          marker=dict(size=15, color='red', symbol='star')),
                row=1, col=1
            )

            # F1 & F2 vs Threshold
            fig_tradeoff.add_trace(
                go.Scatter(x=tradeoff_data['thresholds'], y=tradeoff_data['f1'],
                          mode='lines', name='F1-Score', line=dict(color='green')),
                row=1, col=2
            )
            fig_tradeoff.add_trace(
                go.Scatter(x=tradeoff_data['thresholds'], y=tradeoff_data['f2'],
                          mode='lines', name='F2-Score', line=dict(color='orange')),
                row=1, col=2
            )
            fig_tradeoff.add_vline(x=_active_thr, line_dash="dash", line_color="red",
                                   annotation_text="Current", row=1, col=2)

            # Precision vs Threshold
            fig_tradeoff.add_trace(
                go.Scatter(x=tradeoff_data['thresholds'], y=tradeoff_data['precision'],
                          mode='lines', name='Precision', line=dict(color='blue')),
                row=2, col=1
            )
            fig_tradeoff.add_vline(x=_active_thr, line_dash="dash", line_color="red",
                                   annotation_text="Current", row=2, col=1)

            # Recall vs Threshold
            fig_tradeoff.add_trace(
                go.Scatter(x=tradeoff_data['thresholds'], y=tradeoff_data['recall'],
                          mode='lines', name='Recall', line=dict(color='green')),
                row=2, col=2
            )
            fig_tradeoff.add_vline(x=_active_thr, line_dash="dash", line_color="red",
                                   annotation_text="Current", row=2, col=2)

            fig_tradeoff.update_xaxes(title_text="Recall", row=1, col=1)
            fig_tradeoff.update_yaxes(title_text="Precision", row=1, col=1)
            fig_tradeoff.update_xaxes(title_text="Threshold", row=1, col=2)
            fig_tradeoff.update_yaxes(title_text="Score", row=1, col=2)
            fig_tradeoff.update_xaxes(title_text="Threshold", row=2, col=1)
            fig_tradeoff.update_yaxes(title_text="Precision", row=2, col=1)
            fig_tradeoff.update_xaxes(title_text="Threshold", row=2, col=2)
            fig_tradeoff.update_yaxes(title_text="Recall", row=2, col=2)
            fig_tradeoff.update_layout(height=800, showlegend=True,
                                        title_text="Threshold Trade-off Analysis")

            st.plotly_chart(fig_tradeoff, width='stretch')

            # Optimal threshold recommendations
            st.markdown("#### 🎯 Rekomendasi Threshold Optimal")
            opt_f1_th, opt_f1_score = get_optimal_threshold(y_true, test_probabilities, metric='f1')
            opt_f2_th, opt_f2_score = get_optimal_threshold(y_true, test_probabilities, metric='f2')
            opt_prec_th, opt_prec_score = get_optimal_threshold(y_true, test_probabilities, metric='precision')
            opt_rec_th, opt_rec_score = get_optimal_threshold(y_true, test_probabilities, metric='recall')

            opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
            with opt_col1:
                st.metric("Optimal F1", f"{opt_f1_th:.3f}",
                         help=f"Threshold untuk F1-Score maksimal: {opt_f1_score:.4f}")
            with opt_col2:
                st.metric("Optimal F2", f"{opt_f2_th:.3f}",
                         help=f"Threshold untuk F2-Score maksimal: {opt_f2_score:.4f}")
            with opt_col3:
                st.metric("Optimal Precision", f"{opt_prec_th:.3f}",
                         help=f"Threshold untuk Precision maksimal: {opt_prec_score:.4f}")
            with opt_col4:
                st.metric("Optimal Recall", f"{opt_rec_th:.3f}",
                         help=f"Threshold untuk Recall maksimal: {opt_rec_score:.4f}")

            # Quick apply buttons
            st.markdown("#### ⚡ Terapkan Threshold Optimal")
            opt_btn_col1, opt_btn_col2, opt_btn_col3 = st.columns(3)
            with opt_btn_col1:
                if st.button("Gunakan Optimal F2 (Rekomendasi untuk Fraud)", key="apply_f2"):
                    st.session_state['eval_threshold'] = opt_f2_th
                    st.success(f"✅ Threshold diubah ke {opt_f2_th:.3f}")
                    st.rerun()
            with opt_btn_col2:
                if st.button("Gunakan Optimal F1 (Balance)", key="apply_f1"):
                    st.session_state['eval_threshold'] = opt_f1_th
                    st.success(f"✅ Threshold diubah ke {opt_f1_th:.3f}")
                    st.rerun()
            with opt_btn_col3:
                if st.button("Gunakan Optimal Precision (Minimalkan False Positive)", key="apply_prec"):
                    st.session_state['eval_threshold'] = opt_prec_th
                    st.success(f"✅ Threshold diubah ke {opt_prec_th:.3f}")
                    st.rerun()

        except Exception as tradeoff_error:
            st.error(f"❌ Error dalam analisis threshold trade-off: {str(tradeoff_error)}")
            logger.error(f"Error in threshold trade-off analysis: {tradeoff_error}", exc_info=True)
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
                    background_df = sample_dataframe_for_visualization(
                        st.session_state['train_df'],
                        max_rows=SHAP_BACKGROUND_SAMPLE_SIZE,
                    )
                    background_data = background_df[training_features].values
                else:
                    # Use evaluation data as fallback
                    background_data = sample_dataframe_for_visualization(
                        X_eval_df,
                        max_rows=SHAP_BACKGROUND_SAMPLE_SIZE,
                    ).values
                
                # Initialize explainers
                if explainer.initialize_explainers(background_data):
                    st.success("✅ SHAP explainers initialized successfully")
                    
                    # Show feature importance for available models
                    col_expl1, col_expl2 = st.columns(2)
                    
                    with col_expl1:
                        if 'isolation_forest' in explainer.explainers:
                            st.subheader("Isolation Forest Feature Importance")
                            explainer.plot_feature_importance(
                                'isolation_forest',
                                X=sample_dataframe_for_visualization(X_eval_df).values,
                                max_features=10,
                            )
                    
                    with col_expl2:
                        if 'xgboost' in explainer.explainers:
                            st.subheader("XGBoost Feature Importance")
                            explainer.plot_feature_importance(
                                'xgboost',
                                X=sample_dataframe_for_visualization(X_eval_df).values,
                                max_features=10,
                            )
                    
                    # SHAP summary plot
                    if 'isolation_forest' in explainer.explainers or 'xgboost' in explainer.explainers:
                        st.subheader("SHAP Summary Plot")
                        model_to_plot = 'isolation_forest' if 'isolation_forest' in explainer.explainers else 'xgboost'
                        explainer.plot_shap_summary(
                            sample_dataframe_for_visualization(X_eval_df).values,
                            model_to_plot,
                            max_display=10,
                        )
                    
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
    
    # Compute supervised metrics under consistent guard. Sentinel None means
    # "metric unavailable" — prevents logging misleading placeholders (e.g.
    # accuracy=1.0 from a tautology or precision=0.0 when no labels exist).
    _m_accuracy = None
    _m_precision = None
    _m_recall = None
    _m_f1 = None
    _m_roc_auc = None
    _m_f2_score = None
    _m_pr_auc = None
    _m_mcc = None

    if _has_supervised_labels:
        try:
            _m_accuracy = float((test_predictions == y_true).mean())
            _m_precision = float(precision_score(y_true, test_predictions, zero_division=0))
            _m_recall = float(recall_score(y_true, test_predictions, zero_division=0))
            _m_f1 = float(f1_score(y_true, test_predictions, zero_division=0))
            try:
                _m_roc_auc = float(roc_auc_score(y_true, test_probabilities))
            except Exception:
                _m_roc_auc = 0.0

            # Calculate fraud-specific metrics
            try:
                fraud_metrics = calculate_all_fraud_metrics(y_true, test_predictions, test_probabilities)
                _m_f2_score = fraud_metrics.get('f2_score', 0.0)
                _m_pr_auc = fraud_metrics.get('pr_auc', 0.0)
                _m_mcc = fraud_metrics.get('mcc', 0.0)
            except Exception:
                pass
        except Exception:
            pass

    _supervised_metrics_valid = (
        _m_accuracy is not None
        and _m_precision is not None
        and _m_recall is not None
        and _m_f1 is not None
    )

    if _supervised_metrics_valid:
        current_metrics = {
            'accuracy': _m_accuracy,
            'precision': _m_precision,
            'recall': _m_recall,
            'f1_score': _m_f1,
            'roc_auc': _m_roc_auc if _m_roc_auc is not None else 0.0,
            'f2_score': _m_f2_score if _m_f2_score is not None else 0.0,
            'pr_auc': _m_pr_auc if _m_pr_auc is not None else 0.0,
            'mcc': _m_mcc if _m_mcc is not None else 0.0,
        }
        st.session_state['performance_monitor'].log_performance(current_metrics)
        
        # Show current metrics
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        with col_perf1:
            st.metric("Akurasi Saat Ini", f"{_m_accuracy:.4f}")
        with col_perf2:
            st.metric("F1 Score Saat Ini", f"{_m_f1:.4f}")
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
        st.info("Monitoring performa tersedia untuk mode supervised dengan label valid (2 kelas unik).")
    
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
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔍 Lanjut ke Deteksi Anomali", key="eval_to_detection", type="primary"):
            navigate_to_page('detect')
    with col2:
        if st.button("🧠 Kembali ke Training", key="eval_to_training"):
            navigate_to_page('train')
    with col3:
        if st.button("📥 Unduh Laporan Evaluasi (PDF)", key="download_eval_pdf"):
            with st.spinner("Membuat laporan PDF..."):
                try:
                    from pdf_generator import PDFReportGenerator
                    import plotly.graph_objects as go
                    from sklearn.metrics import roc_curve, auc
                    import matplotlib.pyplot as plt
                    import io

                    # Generate PDF report for evaluation
                    generator = PDFReportGenerator()

                    # Prepare report data
                    report_data = {
                        'metadata': {
                            'generated_at': pd.Timestamp.now().strftime('%d %B %Y %H:%M:%S WIB'),
                            'language': 'Indonesia',
                            'report_type': 'Model Evaluation'
                        },
                        'executive_summary': f"""
                        Laporan evaluasi model deteksi anomali pada dataset test dengan {len(result_df)} sampel.
                        Threshold yang digunakan: {eval_threshold:.2f}.
                        Tingkat anomali terdeteksi: {fraud_rate:.2%}.
                        """,
                        'dataset_analysis': f"""
                        Dataset evaluasi terdiri dari {len(result_df)} sampel dengan {len(training_features)} fitur.
                        Rata-rata probabilitas anomali: {np.mean(test_probabilities):.4f}.
                        Simpangan baku probabilitas: {np.std(test_probabilities):.4f}.
                        """,
                        'feature_analysis': f"""
                        Fitur yang digunakan untuk evaluasi: {len(training_features)} fitur.
                        Fitur dengan korelasi tertinggi dengan anomali dapat dilihat pada tabel analisis fitur.
                        """,
                        'model_architecture': f"""
                        Model dievaluasi menggunakan {len(individual_probs)} algoritma ensemble.
                        Algoritma yang digunakan: {', '.join(individual_probs.keys())}.
                        """,
                        'training_process': """
                        Model telah dilatih sebelumnya menggunakan data training yang valid.
                        Evaluasi dilakukan pada data test yang tidak digunakan selama pelatihan.
                        """,
                        'performance_metrics': f"""
                        Tingkat anomali terdeteksi: {fraud_rate:.2%}.
                        Rata-rata probabilitas anomali: {np.mean(test_probabilities):.4f}.
                        Simpangan baku probabilitas: {np.std(test_probabilities):.4f}.
                        """,
                        'evaluation_metrics': {},
                        'explainability_analysis': """
                        Analisis explainability AI digunakan untuk memahami fitur yang paling berpengaruh
                        dalam prediksi model. SHAP dan LIME dapat digunakan untuk menjelaskan keputusan
                        model pada level individual.
                        """,
                        'recommendations': """
                        - Sesuaikan threshold berdasarkan trade-off antara precision dan recall.
                        - Gunakan analisis feature importance untuk memahami faktor utama anomali.
                        - Lakukan monitoring performa secara berkala untuk mendeteksi drift.
                        """
                    }

                    # Preserve the report's existing format while including the
                    # preprocessing and training context in the final evaluation PDF.
                    training_report = st.session_state.get('model_interpretation_report', {})
                    if not isinstance(training_report, dict):
                        training_report = {}
                    for section in (
                        'executive_summary',
                        'dataset_analysis',
                        'feature_analysis',
                        'model_architecture',
                        'training_process',
                        'performance_metrics',
                    ):
                        training_section = training_report.get(section)
                        if training_section:
                            report_data[section] = (
                                f"{training_section}\n\n### Hasil Evaluasi\n"
                                f"{report_data[section]}"
                            )

                    preprocessing_metadata = st.session_state.get(
                        'preprocessing_metadata', {}
                    )
                    preprocessing_detail_tables = {}
                    if preprocessing_metadata:
                        summary_fields = (
                            ('Jumlah baris diproses', 'total_rows_processed'),
                            ('Jumlah kolom awal', 'original_columns_count'),
                            ('Jumlah fitur akhir', 'final_features_count'),
                            ('Jumlah fitur numerik', 'numerical_columns_count'),
                            ('Jumlah kolom tanggal', 'date_columns_count'),
                            ('Jumlah kolom kategorikal', 'categorical_columns_count'),
                            ('Scope fit preprocessing', 'preprocessing_fit_scope'),
                            ('Versi kontrak preprocessing', 'preprocessing_contract_version'),
                            ('Waktu preprocessing', 'processed_at'),
                        )
                        preprocessing_lines = []
                        preprocessing_table = [['Informasi', 'Nilai']]
                        for label, key in summary_fields:
                            value = preprocessing_metadata.get(key)
                            if value is not None:
                                preprocessing_lines.append(f"- {label}: {value}")
                                preprocessing_table.append([label, str(value)])
                        report_data['preprocessing_analysis'] = (
                            "Ringkasan tahapan preprocessing:\n"
                            + ("\n".join(preprocessing_lines) if preprocessing_lines else
                               "Metadata preprocessing tersedia; rincian tercantum pada tabel lampiran.")
                        )
                        for metadata_key, table_name in (
                            ('fit_statistics', 'preprocessing_fit_statistics'),
                            ('missing_value_metadata', 'preprocessing_missing_values'),
                            ('enhanced_encoding_metadata', 'preprocessing_encoding_details'),
                            ('outlier_metadata', 'preprocessing_outlier_handling'),
                            ('validation_metadata', 'preprocessing_validation_details'),
                            ('duplicate_removal', 'preprocessing_duplicate_removal'),
                        ):
                            metadata_details = preprocessing_metadata.get(metadata_key)
                            if isinstance(metadata_details, dict) and metadata_details:
                                preprocessing_detail_tables[table_name] = [
                                    ['Kolom / Informasi', 'Rincian'],
                                    *[
                                        [str(key), str(value)]
                                        for key, value in metadata_details.items()
                                    ],
                                ]
                    else:
                        preprocessing_table = []

                    training_charts = training_report.get('charts', {})
                    training_tables = training_report.get('tables', {})
                    preprocessing_artifacts = st.session_state.get(
                        'report_artifacts', {}
                    )
                    charts = dict(training_charts) if isinstance(training_charts, dict) else {}
                    tables = dict(training_tables) if isinstance(training_tables, dict) else {}
                    if preprocessing_table:
                        tables['preprocessing_summary'] = preprocessing_table
                    tables.update(preprocessing_detail_tables)
                    if isinstance(preprocessing_artifacts, dict):
                        charts.update(preprocessing_artifacts.get('charts', {}))
                        tables.update(preprocessing_artifacts.get('tables', {}))
                        plotly_figures = preprocessing_artifacts.get('plotly_figures', {})
                        if isinstance(plotly_figures, dict):
                            import plotly.io as plotly_io
                            failed_chart_names = []
                            for chart_name, figure_json in plotly_figures.items():
                                if charts.get(chart_name):
                                    continue
                                try:
                                    figure = plotly_io.from_json(figure_json)
                                    charts[chart_name] = figure.to_image(
                                        format='png', width=1200, height=800, scale=2
                                    )
                                except Exception as chart_error:
                                    logger.warning(
                                        "Failed to render preprocessing chart %s for PDF: %s",
                                        chart_name,
                                        chart_error,
                                    )
                                    failed_chart_names.append(chart_name)
                            if failed_chart_names:
                                st.warning(
                                    "Sebagian grafik preprocessing tidak dapat dirender ke PDF: "
                                    + ", ".join(failed_chart_names)
                                )

                    algorithm_table = [[
                        'Algoritma',
                        'Rata-rata Probabilitas',
                        'Simpangan Baku',
                        f'Tingkat Anomali @ {_active_thr:.2f}',
                    ]]
                    for algorithm_name, probs in individual_probs.items():
                        algorithm_table.append([
                            algorithm_name,
                            f"{float(np.mean(probs)):.4f}",
                            f"{float(np.std(probs)):.4f}",
                            f"{float(np.mean(np.asarray(probs) > _active_thr)):.2%}",
                        ])
                    tables['evaluation_algorithm_performance'] = algorithm_table
                    tables['evaluation_prediction_summary'] = [
                        ['Ringkasan', 'Nilai'],
                        ['Jumlah sampel evaluasi', str(len(result_df))],
                        ['Jumlah prediksi anomali', str(int(np.sum(test_predictions)))],
                        ['Tingkat anomali', f'{fraud_rate:.2%}'],
                        ['Threshold evaluasi', f'{_active_thr:.4f}'],
                        ['Rata-rata probabilitas', f'{float(np.mean(test_probabilities)):.4f}'],
                        ['Simpangan baku probabilitas', f'{float(np.std(test_probabilities)):.4f}'],
                    ]

                    # Add supervised metrics if available
                    if y_true is not None and len(np.unique(y_true)) == 2:
                        report_data['evaluation_metrics'] = {
                            'Accuracy': float((test_predictions == y_true).mean()),
                            'Precision': precision_score(y_true, test_predictions, zero_division=0),
                            'Recall': recall_score(y_true, test_predictions, zero_division=0),
                            'F1-Score': f1_score(y_true, test_predictions, zero_division=0),
                        }
                        try:
                            report_data['evaluation_metrics']['ROC-AUC'] = roc_auc_score(y_true, test_probabilities)
                        except:
                            pass

                        cm = confusion_matrix(y_true, test_predictions)
                        tables['evaluation_confusion_matrix'] = [
                            ['Aktual / Prediksi', 'Prediksi 0', 'Prediksi 1'],
                            ['Aktual 0', str(int(cm[0, 0])), str(int(cm[0, 1]))],
                            ['Aktual 1', str(int(cm[1, 0])), str(int(cm[1, 1]))],
                        ]
                        class_report = classification_report(
                            y_true, test_predictions, output_dict=True, zero_division=0
                        )
                        tables['evaluation_classification_report'] = [
                            ['Kelas', 'Precision', 'Recall', 'F1-Score', 'Support'],
                            *[
                                [
                                    label,
                                    f"{values.get('precision', 0):.4f}",
                                    f"{values.get('recall', 0):.4f}",
                                    f"{values.get('f1-score', 0):.4f}",
                                    str(values.get('support', 0)),
                                ]
                                for label, values in class_report.items()
                                if isinstance(values, dict)
                            ],
                        ]

                        # Add fraud-specific metrics
                        try:
                            fraud_metrics = calculate_all_fraud_metrics(y_true, test_predictions, test_probabilities)
                            report_data['evaluation_metrics']['PR-AUC'] = fraud_metrics.get('pr_auc', 0.0)
                            report_data['evaluation_metrics']['F2-Score'] = fraud_metrics.get('f2_score', 0.0)
                            report_data['evaluation_metrics']['Specificity'] = fraud_metrics.get('specificity', 0.0)
                            report_data['evaluation_metrics']['False Negative Rate'] = fraud_metrics.get('false_negative_rate', 0.0)
                            report_data['evaluation_metrics']['False Positive Rate'] = fraud_metrics.get('false_positive_rate', 0.0)
                            report_data['evaluation_metrics']['Matthews Correlation Coefficient'] = fraud_metrics.get('mcc', 0.0)
                            tables['evaluation_fraud_confusion_details'] = [
                                ['Kategori', 'Jumlah'],
                                ['True Negatives (Non-fraud benar)', str(fraud_metrics.get('true_negatives', 0))],
                                ['False Positives (Non-fraud salah)', str(fraud_metrics.get('false_positives', 0))],
                                ['False Negatives (Fraud lolos)', str(fraud_metrics.get('false_negatives', 0))],
                                ['True Positives (Fraud tertangkap)', str(fraud_metrics.get('true_positives', 0))],
                            ]
                        except Exception as e:
                            logger.warning(f"Failed to add fraud metrics to PDF: {e}")

                    # Prepare charts
                    # Confusion Matrix chart
                    if y_true is not None and len(np.unique(y_true)) == 2:
                        try:
                            cm = confusion_matrix(y_true, test_predictions)
                            fig_cm = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=['Pred 0', 'Pred 1'],
                                y=['Aktual 0', 'Aktual 1'],
                                colorscale='Blues',
                                text=cm,
                                texttemplate="%{text}",
                                textfont={"size": 20}
                            ))
                            fig_cm.update_layout(
                                title='Confusion Matrix',
                                xaxis_title='Predicted',
                                yaxis_title='Actual'
                            )
                            charts['confusion_matrix'] = plotly_to_bytes(fig_cm)
                        except Exception as e:
                            logger.warning(f"Failed to create confusion matrix chart: {e}")

                    # ROC Curve chart
                    if y_true is not None and len(np.unique(y_true)) == 2:
                        try:
                            fpr, tpr, _ = roc_curve(y_true, test_probabilities)
                            fig_roc = go.Figure()
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name='ROC Curve',
                                line=dict(color='blue', width=2)
                            ))
                            fig_roc.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random Classifier',
                                line=dict(color='gray', width=1, dash='dash')
                            ))
                            fig_roc.update_layout(
                                title='ROC Curve',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate'
                            )
                            charts['roc_curve'] = plotly_to_bytes(fig_roc)
                        except Exception as e:
                            logger.warning(f"Failed to create ROC curve chart: {e}")

                    # Precision-Recall Curve chart
                    if y_true is not None and len(np.unique(y_true)) == 2:
                        try:
                            precision_vals, recall_vals, _ = precision_recall_curve(y_true, test_probabilities)
                            fig_pr = go.Figure()
                            fig_pr.add_trace(go.Scatter(
                                x=recall_vals, y=precision_vals,
                                mode='lines',
                                name='Precision-Recall Curve',
                                line=dict(color='green', width=2)
                            ))
                            fig_pr.update_layout(
                                title='Precision-Recall Curve',
                                xaxis_title='Recall',
                                yaxis_title='Precision'
                            )
                            charts['precision_recall_curve'] = plotly_to_bytes(fig_pr)
                        except Exception as e:
                            logger.warning(f"Failed to create precision-recall curve chart: {e}")

                    # Threshold Trade-off Analysis chart
                    if y_true is not None and len(np.unique(y_true)) == 2:
                        displayed_tradeoff_figure = locals().get('fig_tradeoff')
                        if displayed_tradeoff_figure is not None:
                            charts['threshold_tradeoff'] = plotly_to_bytes(
                                displayed_tradeoff_figure
                            )
                        else:
                            try:
                                tradeoff_data = get_threshold_tradeoff_analysis(
                                    y_true, test_probabilities
                                )
                                fig_tradeoff = go.Figure()
                                fig_tradeoff.add_trace(go.Scatter(
                                    x=tradeoff_data['thresholds'],
                                    y=tradeoff_data['precision'],
                                    mode='lines',
                                    name='Precision',
                                    line=dict(color='blue')
                                ))
                                fig_tradeoff.add_trace(go.Scatter(
                                    x=tradeoff_data['thresholds'],
                                    y=tradeoff_data['recall'],
                                    mode='lines',
                                    name='Recall',
                                    line=dict(color='green')
                                ))
                                fig_tradeoff.add_vline(
                                    x=eval_threshold,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Current ({eval_threshold:.2f})",
                                )
                                fig_tradeoff.update_layout(
                                    title='Threshold Trade-off Analysis',
                                    xaxis_title='Threshold',
                                    yaxis_title='Score'
                                )
                                charts['threshold_tradeoff'] = plotly_to_bytes(fig_tradeoff)
                            except Exception as e:
                                logger.warning(f"Failed to create threshold trade-off chart: {e}")

                    # Probability Distribution chart
                    try:
                        fig_dist = create_probability_distribution(
                            result_df['anomaly_probability'],
                            title="Distribusi Probabilitas Anomali",
                            threshold=eval_threshold
                        )
                        charts['probability_distribution'] = plotly_to_bytes(fig_dist)
                    except Exception as e:
                        logger.warning(f"Failed to create probability distribution chart: {e}")

                    if 'gnn' in individual_probs:
                        try:
                            fig_gnn = create_probability_distribution(
                                individual_probs['gnn'],
                                title="Distribusi Probabilitas GNN",
                                threshold=eval_threshold,
                            )
                            charts['gnn_probability_distribution'] = plotly_to_bytes(fig_gnn)
                        except Exception as e:
                            logger.warning(f"Failed to create GNN probability chart: {e}")

                    # Add SHAP charts if explainer is available
                    if 'model_explainer' in st.session_state:
                        try:
                            explainer = st.session_state['model_explainer']
                            X_sample = sample_dataframe_for_visualization(X_eval_df).values

                            # SHAP summary plot
                            if 'isolation_forest' in explainer.explainers or 'xgboost' in explainer.explainers:
                                try:
                                    import matplotlib
                                    matplotlib.use('Agg')
                                    import matplotlib.pyplot as plt

                                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                                    model_to_plot = 'isolation_forest' if 'isolation_forest' in explainer.explainers else 'xgboost'
                                    explainer.plot_shap_summary(X_sample, model_to_plot, max_display=10, ax=ax)
                                    buf = io.BytesIO()
                                    fig_shap.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                                    buf.seek(0)
                                    charts['shap_summary'] = buf.getvalue()
                                    plt.close(fig_shap)
                                except Exception as e:
                                    logger.warning(f"Failed to create SHAP summary chart: {e}")

                            # SHAP feature importance
                            if 'isolation_forest' in explainer.explainers or 'xgboost' in explainer.explainers:
                                try:
                                    fig_fi, ax = plt.subplots(figsize=(10, 6))
                                    model_to_plot = 'isolation_forest' if 'isolation_forest' in explainer.explainers else 'xgboost'
                                    explainer.plot_feature_importance(model_to_plot, X_sample, max_features=10, ax=ax)
                                    buf = io.BytesIO()
                                    fig_fi.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                                    buf.seek(0)
                                    charts['shap_feature_importance'] = buf.getvalue()
                                    plt.close(fig_fi)
                                except Exception as e:
                                    logger.warning(f"Failed to create SHAP feature importance chart: {e}")
                        except Exception as e:
                            logger.warning(f"Failed to add SHAP charts to PDF: {e}")

                    # Generate PDF
                    pdf_buffer = generator.generate_pdf(
                        report_data,
                        charts=charts,
                        tables=tables,
                        summary_only=True,
                    )

                    st.download_button(
                        label="⬇️ Klik untuk Mengunduh PDF",
                        data=pdf_buffer.getvalue(),
                        file_name=f"ASTINA_Model_Evaluation_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="download_eval_pdf_button"
                    )
                    st.success("✅ Laporan PDF berhasil dibuat!")

                except Exception as e:
                    st.error(f"❌ Gagal membuat PDF: {str(e)}")
                    logger.error(f"Error generating evaluation PDF: {e}", exc_info=True)
