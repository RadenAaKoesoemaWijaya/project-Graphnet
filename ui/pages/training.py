import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
try:
    import torch
except (ImportError, OSError, Exception):
    import sys
    sys.modules.pop('torch', None)
    torch = None
from ui.utils import *
from state_manager import *
def show_training_page():
    X_train = None
    edge_index = None
    st.title("Pelatihan Model Deteksi Anomali")
    st.info("Training mempelajari pola dataset yang sudah diproses. Pilih algoritma sesuai kebutuhan: model tabular untuk pola fitur dan GNN untuk hubungan antar klaim.")

    # Check if data is available
    if 'df_processed_path' not in st.session_state or 'feature_columns' not in st.session_state:
        st.error("❌ Data belum diproses. Silakan unggah dan praproses data terlebih dahulu.")
        if st.button("Kembali ke Unggah Data"):
            navigate_to_page('collect')
        return

    try:
        df_processed = get_df_processed()
        if df_processed is None:
            st.error("❌ Gagal memuat data hasil praproses!")
            logger.error("Failed to load processed data in training page")
            return
        if isinstance(df_processed, dict):
            if 'path' in df_processed:
                df_processed = pd.read_parquet(df_processed['path'])
            else:
                st.error("❌ Format data praproses tidak valid.")
                return
        if not isinstance(df_processed, pd.DataFrame):
            st.error("❌ Data praproses bukan DataFrame valid.")
            return
        feature_columns = st.session_state['feature_columns']

        # Use selected features if available, otherwise use all features
        if 'selected_features' in st.session_state:
            selected_features = st.session_state['selected_features']
            feature_selection_method = st.session_state.get('feature_selection_method', 'Tidak diketahui')
            original_count = st.session_state.get('original_feature_count', len(feature_columns))
            final_count = st.session_state.get('final_feature_count', len(selected_features))
        else:
            selected_features = feature_columns
            feature_selection_method = 'Semua Fitur (Bawaan)'
            original_count = len(feature_columns)
            final_count = len(feature_columns)

    except Exception as e:
        st.error(f"Gagal mengakses data: {str(e)}")
        logger.error(f"Error accessing data in training page: {e}", exc_info=True)
        return
    
    st.markdown("""
    ### Training Model Deteksi Anomali Transaksi Asuransi
    
    Pada halaman ini, Anda akan melatih model deteksi anomali menggunakan kombinasi tiga algoritma:
    - **Isolation Forest**: Untuk deteksi anomali awal
    - **Autoencoder**: Untuk mempelajari pola normal
    - **XGBoost**: Untuk menganalisis pola kompleks dengan gradient boosting
    """)
    
    # Feature Selection Summary
    st.subheader("🔧 Informasi Seleksi Fitur")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Metode Seleksi", feature_selection_method)
    
    with col2:
        st.metric("Fitur Awal", original_count)
    
    with col3:
        st.metric("Fitur Terpilih", final_count)
    
    with col4:
        reduction_pct = ((original_count - final_count) / original_count) * 100 if original_count > 0 else 0
        st.metric("Reduksi", f"{reduction_pct:.1f}%")
    
    if final_count < original_count:
        st.info(f"✅ Menggunakan {final_count} fitur yang dipilih dari {original_count} fitur tersedia")
    else:
        st.info(f"✅ Menggunakan semua {final_count} fitur yang tersedia")
    
    # Show selected features
    with st.expander("📋 Lihat Fitur yang Digunakan"):
        st.write(f"**Total {len(selected_features)} fitur:**")
        for i, feat in enumerate(selected_features, 1):
            st.write(f"{i}. {feat}")
    
    # Data preparation
    st.subheader("📊 Persiapan Data")

    if (
        st.session_state.pop('auto_split_after_preprocessing', False)
        and 'train_df' not in st.session_state
    ):
        try:
            train_df, test_df, stratify_label = split_processed_dataset(df_processed, test_size=0.2)
            st.session_state['train_df'] = train_df
            st.session_state['test_df'] = test_df
            if stratify_label:
                st.info(f"🔀 Pembagian data otomatis menggunakan stratified split berdasarkan kolom '{stratify_label}'")
            st.success(
                f"✅ Data hasil praproses langsung disiapkan untuk pelatihan: "
                f"data latih ({len(train_df)} baris), data uji ({len(test_df)} baris)"
            )
        except Exception as e:
            st.error(f"❌ Gagal membagi data otomatis: {str(e)}")
            st.info("📝 Silakan coba bagi data secara manual dengan tombol di bawah.")
            # Clear invalid state
            st.session_state.pop('train_df', None)
            st.session_state.pop('test_df', None)
    
    # Split data
    test_size = st.slider("Ukuran Data Uji:", 0.1, 0.4, 0.2, 0.05)
    
    if st.button("Bagi Data"):
        try:
            train_df, test_df, stratify_label = split_processed_dataset(df_processed, test_size=test_size)
            st.session_state['train_df'] = train_df
            st.session_state['test_df'] = test_df
            if stratify_label:
                st.info(f"🔀 Menggunakan stratified split berdasarkan kolom '{stratify_label}'")
            
            st.success(f"Data berhasil dibagi: Data latih ({len(train_df)} baris), data uji ({len(test_df)} baris)")
            
            # Show data distribution
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Latih:**")
                st.dataframe(train_df[selected_features].describe())
            
            with col2:
                st.write("**Data Uji:**")
                st.dataframe(test_df[selected_features].describe())
        except Exception as e:
            st.error(f"❌ Gagal membagi data: {str(e)}")
            st.info("💡 Tips: Pastikan dataset memiliki cukup data (minimal 10 baris) dan tidak terlalu imbalanced.")
            # Clear invalid state
            st.session_state.pop('train_df', None)
            st.session_state.pop('test_df', None)
    
    if 'train_df' not in st.session_state:
        st.info("Silakan bagi data terlebih dahulu.")
        return
    
    # Model configuration
    st.subheader("⚙️ Konfigurasi & Preset Pelatihan")

    # Hardware Detection Banner
    has_gpu = torch is not None and torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "CPU Komputasi Standar"
    
    if has_gpu:
        st.success(f"🚀 **Akselerator Hardware Terdeteksi:** {gpu_name} (Siap untuk Deep Learning & GNN cepat)")
    else:
        st.info(f"ℹ️ **Perangkat Komputasi:** {gpu_name} | *Catatan: Model neural (Autoencoder & GNN) akan berjalan di CPU.*")

    # Training Profile Presets
    preset_choice = st.radio(
        "Pilih Profil Pelatihan:",
        [
            "⚡ Mode Cepat (Tabular Fast) — [Rekomendasi Eksperimen Cepat ~10-30 detik]",
            "⚖️ Mode Seimbang (Balanced) — [Isolation Forest + Autoencoder ~1-2 menit]",
            "🧠 Mode Lengkap (Deep Graph Ensemble) — [Semua Algoritma + GNN + Optuna]",
            "🛠️ Mode Kustom (Konfigurasi Manual Sepenuhnya)"
        ],
        index=0,
        help="Pilih preset untuk konfigurasi otomatis atau sesuaikan secara manual."
    )

    is_fast_preset = "Mode Cepat" in preset_choice
    is_balanced_preset = "Mode Seimbang" in preset_choice
    is_deep_preset = "Mode Lengkap" in preset_choice
    is_custom_preset = "Mode Kustom" in preset_choice

    training_mode_options = {
        "Tanpa supervisi (Isolation Forest / Autoencoder / DBSCAN / GNN)": TRAINING_MODE_UNSUPERVISED,
        "Dengan supervisi (XGBoost/LightGBM/Random Forest/SVM)": TRAINING_MODE_SUPERVISED,
    }
    training_mode_label = st.radio(
        "Pilih mode pelatihan:",
        list(training_mode_options.keys()),
        index=0
    )
    training_mode = training_mode_options[training_mode_label]

    supervised_model_type = None
    label_column = None

    dbscan_eps = 0.5
    dbscan_min_samples = 5

    iso_weight = 0.0
    ae_weight = 0.0
    dbscan_weight = 0.0
    xgb_weight = 0.0
    gnn_weight = 0.0

    iso_contamination = 0.05
    iso_n_estimators = 50 if is_fast_preset else (50 if is_balanced_preset else 100)

    ae_encoding_dim = 32
    ae_hidden_dims = "64,48"
    ae_epochs = 20 if is_balanced_preset else (50 if is_deep_preset else 30)
    ae_batch_size = 1024
    ae_early_stopping = True
    ae_patience = 8
    ae_min_delta = 0.0001

    xgb_n_estimators = 100 if is_fast_preset else 200
    xgb_max_depth = 5 if is_fast_preset else 6
    xgb_learning_rate = 0.1
    xgb_extra_params = {}

    # Initialize GNN parameters
    algo_options = []
    gnn_hidden = 32 if not has_gpu else 64
    gnn_heads = 2 if not has_gpu else 4
    gnn_dropout = 0.2
    gnn_epochs = 30 if not has_gpu else 60
    graph_method = "star"
    graph_k = 5

    # Initialize supervised mode options with defaults
    enable_hyperparameter_tuning = False
    enable_cross_validation = False
    cv_folds = 5

    if training_mode == TRAINING_MODE_UNSUPERVISED:
        if is_fast_preset:
            default_algos = ["Isolation Forest"]
        elif is_balanced_preset:
            default_algos = ["Isolation Forest", "Autoencoder"]
        elif is_deep_preset:
            default_algos = ["Isolation Forest", "Autoencoder", "GNN"]
        else:
            default_algos = ["Isolation Forest", "Autoencoder"]

        algo_options = st.multiselect(
            "Pilih algoritma tanpa supervisi:",
            ["Isolation Forest", "Autoencoder", "DBSCAN", "GNN"],
            default=default_algos
        )

        if not algo_options:
            st.error("❌ Pilih minimal 1 algoritma.")
            return

        with st.expander("🛠️ Parameter Algoritma Detail", expanded=is_custom_preset):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if "Isolation Forest" in algo_options:
                    st.write("**Isolation Forest**")
                    iso_contamination = st.slider("Tingkat Kontaminasi:", 0.01, 0.2, 0.05, 0.01)
                    iso_n_estimators = st.slider("Jumlah Estimator:", 10, 200, iso_n_estimators, 10)

            with col2:
                if "Autoencoder" in algo_options:
                    st.write("**Autoencoder**")
                    ae_encoding_dim = st.slider("Dimensi Encoding:", 4, 128, 32, 4)
                    ae_hidden_dims = st.text_input("Layer Tersembunyi (pisahkan koma):", "64,48")
                    ae_epochs = st.slider("Epoch Pelatihan:", 5, 150, ae_epochs, 5)
                    ae_batch_size = st.slider("Ukuran Batch:", 32, 4096, 1024, 64)
                    ae_early_stopping = st.checkbox("Aktifkan Early Stopping", value=True)
                    if ae_early_stopping:
                        ae_patience = st.slider("Patience:", 1, 30, 8, 1)

            with col3:
                if "DBSCAN" in algo_options:
                    st.write("**DBSCAN**")
                    dbscan_eps = st.slider("eps:", 0.1, 10.0, 0.5, 0.1)
                    dbscan_min_samples = st.slider("min_samples:", 2, 100, 5, 1)

            with col4:
                if "GNN" in algo_options:
                    st.write("**Graph Neural Network**")
                    gnn_hidden = st.slider("Hidden Channels:", 16, 256, gnn_hidden, 16)
                    gnn_heads = st.slider("Heads:", 1, 8, gnn_heads, 1)
                    gnn_dropout = st.slider("Dropout:", 0.0, 0.8, 0.2, 0.05)
                    gnn_epochs = st.slider("Epochs GNN:", 10, 200, gnn_epochs, 10)
                    graph_method = st.selectbox("Graph Method:", ["star", "knn", "heterogeneous"])
                    if graph_method == "knn":
                        graph_k = st.slider("k for k-NN:", 2, 20, 5, 1)

        # Optuna Ensemble Weight Tuning Option
        default_optuna_val = True if is_deep_preset else False
        enable_optuna_ensemble_weights = st.checkbox(
            "⚡ Optimasi Bobot Ensemble Dinamis (Optuna FPR Minimizer)",
            value=default_optuna_val,
            help="Gunakan Optuna untuk mencari bobot ensemble optimal secara matematis guna meminimalkan False Positive Rate (FPR). Memerlukan waktu ekstra beberapa menit."
        )

        st.subheader("⚖️ Bobot Kombinasi (Unsupervised)")
        enabled = {
            'isolation_forest': ("Isolation Forest" in algo_options),
            'autoencoder': ("Autoencoder" in algo_options),
            'dbscan': ("DBSCAN" in algo_options),
            'gnn': ("GNN" in algo_options)
        }
        enabled_count = sum(1 for v in enabled.values() if v)

        if enabled_count == 1:
            iso_weight = 1.0 if enabled['isolation_forest'] else 0.0
            ae_weight = 1.0 if enabled['autoencoder'] else 0.0
            dbscan_weight = 1.0 if enabled['dbscan'] else 0.0
            gnn_weight = 1.0 if enabled['gnn'] else 0.0
            st.info("Hanya 1 algoritma dipilih: bobot otomatis = 1.0")
        else:
            w_col1, w_col2, w_col3, w_col4 = st.columns(4)
            with w_col1:
                iso_weight = st.slider("Bobot Isolation Forest:", 0.0, 1.0, 0.35 if enabled['isolation_forest'] else 0.0, 0.05, disabled=not enabled['isolation_forest'])
            with w_col2:
                ae_weight = st.slider("Bobot Autoencoder:", 0.0, 1.0, 0.35 if enabled['autoencoder'] else 0.0, 0.05, disabled=not enabled['autoencoder'])
            with w_col3:
                dbscan_weight = st.slider("Bobot DBSCAN:", 0.0, 1.0, 0.0, 0.05, disabled=not enabled['dbscan'])
            with w_col4:
                gnn_weight = st.slider("Bobot GNN:", 0.0, 1.0, 0.30 if enabled['gnn'] else 0.0, 0.05, disabled=not enabled['gnn'])

            total_weight = iso_weight + ae_weight + dbscan_weight + gnn_weight
            if total_weight <= 0:
                st.error("❌ Total bobot tidak boleh 0.")
                return

            iso_weight /= total_weight
            ae_weight /= total_weight
            dbscan_weight /= total_weight
            gnn_weight /= total_weight

        use_dynamic_weights = False

    else:
        supervised_model_type = st.selectbox(
            "Pilih algoritma dengan supervisi:",
            ["XGBoost", "LightGBM", "Random Forest", "SVM"]
        )

        label_candidates = [
            col for col in df_processed.columns
            if any(k in col.lower() for k in ['fraud', 'label', 'target', 'class'])
        ]
        if not label_candidates:
            label_candidates = df_processed.columns.tolist()
        label_column = st.selectbox("Pilih kolom label (0/1):", label_candidates)

        # Hyperparameter Tuning Option
        enable_hyperparameter_tuning = st.checkbox(
            "Aktifkan Hyperparameter Tuning (Optuna)",
            value=False,
            help="Gunakan Optuna untuk mencari hyperparameter terbaik secara otomatis. Proses ini akan memakan waktu lebih lama."
        )

        # Optuna Ensemble Weight Tuning Option
        enable_optuna_ensemble_weights = st.checkbox(
            "⚡ Optimasi Bobot Ensemble Dinamis (Optuna FPR Minimizer)",
            value=False,
            help="Gunakan Optuna untuk mencari bobot ensemble optimal secara matematis guna meminimalkan False Positive Rate (FPR)."
        )

        # Cross-Validation Option
        enable_cross_validation = st.checkbox(
            "Aktifkan Cross-Validation",
            value=False,
            help="Gunakan k-fold cross-validation untuk estimasi performa model yang lebih akurat."
        )
        
        if enable_cross_validation:
            cv_folds = st.slider("Jumlah Fold (k):", 2, 10, 5, 1)

        with st.expander("🛠️ Parameter Supervised Detail", expanded=is_custom_preset):
            if supervised_model_type in ["XGBoost", "LightGBM"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    xgb_n_estimators = st.slider("Jumlah Estimator:", 25, 500, xgb_n_estimators, 25)
                with col2:
                    xgb_max_depth = st.slider("Kedalaman Maksimum:", 2, 20, xgb_max_depth, 1)
                with col3:
                    xgb_learning_rate = st.slider("Laju Pembelajaran:", 0.001, 1.0, xgb_learning_rate, 0.001)
            elif supervised_model_type == "Random Forest":
                rf_col1, rf_col2 = st.columns(2)
                with rf_col1:
                    xgb_n_estimators = st.slider("Jumlah Pohon:", 25, 500, 200, 25)
                with rf_col2:
                    xgb_max_depth = st.slider("Kedalaman Maksimum:", 2, 50, 10, 1)
            else:
                svm_col1, svm_col2, svm_col3 = st.columns(3)
                with svm_col1:
                    xgb_extra_params['C'] = st.slider("C:", 0.0, 100.0, 1.0, 0.1)
                with svm_col2:
                    xgb_extra_params['kernel'] = st.selectbox("Kernel:", ["rbf", "linear", "poly", "sigmoid"])
                with svm_col3:
                    xgb_extra_params['gamma'] = st.selectbox("Gamma:", ["scale", "auto"])

        iso_weight = 0.0
        ae_weight = 0.0
        dbscan_weight = 0.0
        gnn_weight = 0.0
        xgb_weight = 1.0
        use_dynamic_weights = False

    # ----------------------------------------------------
    # DYNAMIC COMPLEXITY & HARDWARE ESTIMATOR (QA BADGE)
    # ----------------------------------------------------
    n_samples = len(st.session_state.get('train_df', [])) if 'train_df' in st.session_state else len(df_processed)
    
    # Calculate complexity score
    complexity_score = 0
    if training_mode == TRAINING_MODE_UNSUPERVISED:
        if "Isolation Forest" in algo_options:
            complexity_score += 1
        if "Autoencoder" in algo_options:
            complexity_score += 3 + (ae_epochs // 20)
        if "DBSCAN" in algo_options:
            complexity_score += 2
        if "GNN" in algo_options:
            complexity_score += 5 + (gnn_epochs // 15)
    else:
        complexity_score += 2
        if enable_cross_validation:
            complexity_score += 2 * cv_folds

    if enable_optuna_ensemble_weights:
        complexity_score += 6
    if enable_hyperparameter_tuning:
        complexity_score += 8

    # Apply GPU acceleration factor
    if has_gpu:
        complexity_score = max(1, int(complexity_score * 0.45))

    st.markdown("---")
    st.subheader("📊 Estimasi Beban Komputasi & Rekomendasi QA")

    est_col1, est_col2, est_col3 = st.columns([1.2, 1.2, 2.6])

    with est_col1:
        if complexity_score <= 4:
            st.success("🟢 **Beban: Ringan**\n\nEstimasi: **< 30 Detik**")
        elif complexity_score <= 10:
            st.warning("🟡 **Beban: Sedang**\n\nEstimasi: **1 – 3 Menit**")
        else:
            st.error("🔴 **Beban: Tinggi**\n\nEstimasi: **5 – 15+ Menit**")

    with est_col2:
        st.metric("Total Data Latih", f"{n_samples:,} baris")

    with est_col3:
        if complexity_score <= 4:
            st.info("💡 **Rekomendasi QA:** Konfigurasi sangat efisien dan responsif. Cocok untuk iterasi kilat dan uji coba fitur.")
        elif complexity_score <= 10:
            st.info("💡 **Rekomendasi QA:** Keseimbangan optimal antara waktu dan performa deteksi.")
        else:
            if not has_gpu:
                st.warning("⚠️ **Perhatian Hardware:** Anda menjalankan model neural / GNN / Optuna di **CPU**. Pelatihan akan membutuhkan waktu lebih lama. Pertimbangkan gunakan *Mode Cepat* jika membutuhkan hasil segera.")
            else:
                st.info("💡 **Rekomendasi QA:** Menggunakan GPU terakselerasi. Model ensemble siap dilatih dengan akurasi optimal.")

    xgboost_params = {
        'n_estimators': xgb_n_estimators,
        'max_depth': xgb_max_depth,
        'learning_rate': xgb_learning_rate,
        **xgb_extra_params
    }
    if supervised_model_type == "LightGBM":
        xgboost_params['model_type'] = 'lightgbm'
    elif supervised_model_type == "Random Forest":
        xgboost_params['model_type'] = 'random_forest'
    elif supervised_model_type == "SVM":
        xgboost_params['model_type'] = 'svm'
    else:
        xgboost_params['model_type'] = 'xgboost'
    
    # Training
    if st.button("🚀 Mulai Training", type="primary"):
        with st.spinner("Sedang melatih model..."):
            train_df = st.session_state['train_df']
            test_df = st.session_state['test_df']
            
            # Prepare features using selected features
            X_train = train_df[selected_features].values
            X_test = test_df[selected_features].values

            # Initialize y_train early for hyperparameter tuning and cross-validation
            y_train = None
            if training_mode == TRAINING_MODE_SUPERVISED:
                try:
                    y_train = train_df[label_column].values
                    y_train = pd.Series(y_train).fillna(0).values
                    unique_vals = pd.Series(y_train).dropna().unique().tolist()
                    if len(unique_vals) > 2:
                        st.error("❌ Label harus biner (0/1).")
                        return
                except Exception as e:
                    st.error(f"❌ Gagal membaca kolom label: {str(e)}")
                    return

            # Hyperparameter Tuning with Optuna (for supervised mode)
            if training_mode == TRAINING_MODE_SUPERVISED and enable_hyperparameter_tuning:
                try:
                    import optuna
                    from sklearn.model_selection import cross_val_score
                    
                    st.info("🔍 Memulai Hyperparameter Tuning dengan Optuna...")
                    
                    def objective(trial):
                        # Suggest hyperparameters based on model type
                        if supervised_model_type in ["XGBoost", "LightGBM"]:
                            n_estimators = trial.suggest_int('n_estimators', 50, 500)
                            max_depth = trial.suggest_int('max_depth', 3, 10)
                            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                            
                            params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'learning_rate': learning_rate,
                                'model_type': 'lightgbm' if supervised_model_type == "LightGBM" else 'xgboost'
                            }
                        elif supervised_model_type == "Random Forest":
                            n_estimators = trial.suggest_int('n_estimators', 50, 500)
                            max_depth = trial.suggest_int('max_depth', 2, 30)
                            
                            params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'model_type': 'random_forest'
                            }
                        else:  # SVM
                            C = trial.suggest_float('C', 0.1, 10.0)
                            kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
                            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
                            
                            params = {
                                'C': C,
                                'kernel': kernel,
                                'gamma': gamma,
                                'model_type': 'svm'
                            }
                        
                        # Create temporary detector for evaluation
                        temp_detector = CombinedAnomalyDetector(
                            isolation_forest_params={'contamination': 0.05, 'n_estimators': 50},
                            autoencoder_params={'encoding_dim': 32, 'epochs': 10, 'hidden_dims': [64], 'batch_size': 1024},
                            xgboost_params=params,
                            algorithms=['xgboost'],
                            use_dynamic_weights=False,
                            imbalance_config={'enabled': False}
                        )
                        
                        if y_train is None or X_train is None:
                            return 1.0
                        subset_size = min(len(X_train), 1000)
                        X_subset = X_train[:subset_size]
                        y_subset = y_train[:subset_size]
                        
                        try:
                            temp_detector.fit(X_subset, y_subset, device='cpu')
                            predictions = temp_detector.predict(X_subset)
                            score = -f1_score(y_subset, predictions, zero_division=0)
                        except Exception:
                            score = 1.0  # Penalize errors
                        
                        return score
                    
                    # Create study and optimize
                    study = optuna.create_study(direction='minimize')
                    study.optimize(objective, n_trials=10, timeout=300)  # 10 trials or 5 minutes max
                    
                    # Get best parameters
                    best_params = study.best_params
                    st.success(f"✅ Hyperparameter Tuning Selesai! Best F1-Score: {-study.best_value:.4f}")
                    st.info(f"Best Parameters: {best_params}")
                    
                    # Update parameters with best found
                    if supervised_model_type in ["XGBoost", "LightGBM"]:
                        xgb_n_estimators = best_params['n_estimators']
                        xgb_max_depth = best_params['max_depth']
                        xgb_learning_rate = best_params['learning_rate']
                    elif supervised_model_type == "Random Forest":
                        xgb_n_estimators = best_params['n_estimators']
                        xgb_max_depth = best_params['max_depth']
                    else:  # SVM
                        xgb_extra_params['C'] = best_params['C']
                        xgb_extra_params['kernel'] = best_params['kernel']
                        xgb_extra_params['gamma'] = best_params['gamma']
                    
                    # Update xgboost_params
                    xgboost_params = {
                        'n_estimators': xgb_n_estimators,
                        'max_depth': xgb_max_depth,
                        'learning_rate': xgb_learning_rate,
                        **xgb_extra_params
                    }
                    if supervised_model_type == "LightGBM":
                        xgboost_params['model_type'] = 'lightgbm'
                    elif supervised_model_type == "Random Forest":
                        xgboost_params['model_type'] = 'random_forest'
                    elif supervised_model_type == "SVM":
                        xgboost_params['model_type'] = 'svm'
                    else:
                        xgboost_params['model_type'] = 'xgboost'
                        
                except ImportError:
                    st.warning("⚠️ Optuna tidak terinstall. Menggunakan parameter manual.")
                except Exception as e:
                    st.warning(f"⚠️ Hyperparameter tuning gagal: {str(e)}. Menggunakan parameter manual.")
            
            # Cross-Validation for Supervised Mode
            if training_mode == TRAINING_MODE_SUPERVISED and enable_cross_validation:
                try:
                    from sklearn.model_selection import StratifiedKFold, cross_val_score
                    from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
                    
                    st.info("🔍 Memulai Cross-Validation...")
                    
                    # Create stratified k-fold
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    
                    # Temporary detector for CV
                    cv_detector = CombinedAnomalyDetector(
                        isolation_forest_params={'contamination': 0.05, 'n_estimators': 50},
                        autoencoder_params={'encoding_dim': 32, 'epochs': 10, 'hidden_dims': [64], 'batch_size': 1024},
                        xgboost_params=xgboost_params,
                        algorithms=['xgboost'],
                        use_dynamic_weights=False,
                        imbalance_config={'enabled': False}
                    )
                    
                    # Perform cross-validation
                    cv_scores = []
                    cv_f1_scores = []
                    cv_precision_scores = []
                    cv_recall_scores = []
                    
                    if y_train is not None and X_train is not None:
                        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                            
                            try:
                                cv_detector.fit(X_fold_train, y_fold_train, device='cpu')
                                fold_predictions = cv_detector.predict(X_fold_val)
                                
                                fold_f1 = f1_score(y_fold_val, fold_predictions, zero_division=0)
                                fold_precision = precision_score(y_fold_val, fold_predictions, zero_division=0)
                                fold_recall = recall_score(y_fold_val, fold_predictions, zero_division=0)
                                
                                cv_f1_scores.append(fold_f1)
                                cv_precision_scores.append(fold_precision)
                                cv_recall_scores.append(fold_recall)
                                
                                st.write(f"Fold {fold+1}/{cv_folds} - F1: {fold_f1:.4f}, Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}")
                            except Exception as e:
                                st.warning(f"Fold {fold+1} gagal: {str(e)}")
                    
                    # Display CV results
                    if cv_f1_scores:
                        st.success(f"✅ Cross-Validation Selesai!")
                        st.metric("Mean F1-Score", f"{np.mean(cv_f1_scores):.4f}")
                        st.metric("Std F1-Score", f"{np.std(cv_f1_scores):.4f}")
                        st.metric("Mean Precision", f"{np.mean(cv_precision_scores):.4f}")
                        st.metric("Mean Recall", f"{np.mean(cv_recall_scores):.4f}")
                        
                        # Plot CV scores
                        cv_df = pd.DataFrame({
                            'Fold': range(1, len(cv_f1_scores) + 1),
                            'F1-Score': cv_f1_scores,
                            'Precision': cv_precision_scores,
                            'Recall': cv_recall_scores
                        })
                        fig = px.line(cv_df, x='Fold', y=['F1-Score', 'Precision', 'Recall'],
                                     title=f'Cross-Validation Scores ({cv_folds}-Fold)',
                                     labels={'value': 'Score', 'variable': 'Metric'})
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.warning("⚠️ Cross-Validation gagal untuk semua folds.")
                        
                except Exception as e:
                    st.warning(f"⚠️ Cross-Validation gagal: {str(e)}. Melanjutkan tanpa CV.")
            
            # Prepare features for XGBoost and GNN
            st.write("Menyiapkan fitur untuk training...")
            edge_index = None
            node_features = None
            st.session_state.pop('edge_index', None)
            st.session_state.pop('edge_type', None)
            st.session_state.pop('graph_method', None)
            st.session_state.pop('graph_node_count', None)
            st.session_state.pop('graph_edge_count', None)
            
            # Build graph for GNN if selected
            if "GNN" in algo_options and training_mode == TRAINING_MODE_UNSUPERVISED:
                try:
                    from model import create_claim_graph
                    
                    # Skip GNN for very large datasets to avoid long graph construction time
                    if len(train_df) > 500000:
                        st.warning(f"⚠️ Dataset terlalu besar ({len(train_df):,} samples) untuk GNN graph construction. GNN akan di-skip untuk menghindari bottleneck.")
                        st.info("💡 Tips: Gunakan dataset yang lebih kecil (<500K samples) atau non-aktifkan GNN untuk dataset besar.")
                        edge_index = None
                        node_features = None
                    else:
                        with st.spinner("Membangun graph untuk GNN..."):
                            graph_result = create_claim_graph(
                                train_df, 
                                selected_features, 
                                method=graph_method,
                                **({"k": graph_k} if graph_method == "knn" else {})
                            )
                            
                            if isinstance(graph_result, tuple) and len(graph_result) == 3:
                                node_features, edge_index, edge_type = graph_result
                                st.session_state['edge_type'] = edge_type
                            else:
                                node_features, edge_index = graph_result
                                
                            st.session_state['edge_index'] = edge_index
                            st.session_state['graph_method'] = graph_method
                            st.session_state['graph_k'] = graph_k
                            st.session_state['graph_node_count'] = int(node_features.shape[0])
                            st.session_state['graph_edge_count'] = int(edge_index.shape[1])
                            st.success(f"Graph berhasil dibuat: {node_features.shape[0]} nodes, {edge_index.shape[1]} edges")
                except Exception as e:
                    st.warning(f"Gagal membangun graph: {str(e)}. Melanjutkan tanpa GNN.")
                    edge_index = None
                    node_features = None

            # Initialize combined detector
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            st.write(f"Perangkat komputasi: {device}")
            
            # Add imbalance handling configuration
            imbalance_config = {
                'enabled': True,
                'method': 'smote',
                'sampling_strategy': 'auto'
            }
            
            # Progress callback for Autoencoder
            def autoencoder_progress_callback(epoch, total_epochs, loss):
                progress = 0.2 + 0.6 * (epoch + 1) / total_epochs  # 20% to 80% progress
                try:
                    import json, os
                    os.makedirs("cache", exist_ok=True)
                    with open("cache/training_status.json", "w") as f:
                        json.dump({"status": "running", "progress": progress, "message": f"Training Autoencoder: Epoch {epoch+1}/{total_epochs}, Loss: {loss:.4f}"}, f)
                except Exception:
                    pass
                # No longer directly call Streamlit elements from background thread

            
            detector = CombinedAnomalyDetector(
                isolation_forest_params={
                    'contamination': iso_contamination,
                    'n_estimators': iso_n_estimators
                },
                autoencoder_params={
                    'encoding_dim': ae_encoding_dim,
                    'epochs': ae_epochs,
                    'hidden_dims': [int(x.strip()) for x in ae_hidden_dims.split(',') if x.strip()],
                    'batch_size': ae_batch_size,
                    'early_stopping_patience': ae_patience if ae_early_stopping else None,
                    'early_stopping_min_delta': ae_min_delta if ae_early_stopping else None,
                    'progress_callback': autoencoder_progress_callback
                },
                dbscan_params={
                    'eps': dbscan_eps,
                    'min_samples': dbscan_min_samples
                },
                gnn_params={
                    'hidden_channels': gnn_hidden,
                    'num_heads': gnn_heads,
                    'dropout': gnn_dropout,
                    'epochs': gnn_epochs
                },
                xgboost_params={
                    'n_estimators': xgb_n_estimators,
                    'max_depth': xgb_max_depth,
                    'learning_rate': xgb_learning_rate,
                    'model_type': 'xgboost' if supervised_model_type is None else (
                        'xgboost' if supervised_model_type == 'XGBoost' else
                        'lightgbm' if supervised_model_type == 'LightGBM' else
                        'random_forest' if supervised_model_type == 'Random Forest' else
                        'svm'
                    )
                },
                algorithms=(
                    [
                        a for a in [
                            'isolation_forest' if "Isolation Forest" in algo_options else None,
                            'autoencoder' if "Autoencoder" in algo_options else None,
                            'dbscan' if "DBSCAN" in algo_options else None,
                            'gnn' if "GNN" in algo_options else None
                        ] if a is not None
                    ] if training_mode == TRAINING_MODE_UNSUPERVISED else
                    ['xgboost']
                ),
                use_dynamic_weights=use_dynamic_weights,
                imbalance_config=imbalance_config
            )
            
            detector.isolation_weight = iso_weight
            detector.autoencoder_weight = ae_weight
            detector.dbscan_weight = dbscan_weight
            detector.gnn_weight = gnn_weight
            detector.xgboost_weight = xgb_weight
            st.write(
                f"Weights set: Isolation={detector.isolation_weight:.3f}, "
                f"Autoencoder={detector.autoencoder_weight:.3f}, "
                f"DBSCAN={detector.dbscan_weight:.3f}, "
                f"GNN={detector.gnn_weight:.3f}, "
                f"Supervised={detector.xgboost_weight:.3f}"
            )
            
            # Store detector in session state for background thread
            st.session_state['current_training_detector'] = detector
            
            # Save features and configs immediately
            st.session_state['current_training_features'] = selected_features
            st.session_state['current_training_mode'] = training_mode
            st.session_state['current_training_label_column'] = (
                label_column if training_mode == TRAINING_MODE_SUPERVISED else None
            )
            
            import threading
            import json
            import os
            
            def train_worker(detector_obj, X, e_idx, e_type, y, dev, mode, opt_ensemble, opt_hyper):
                try:
                    os.makedirs("cache", exist_ok=True)
                    with open("cache/training_status.json", "w") as f:
                        json.dump({"status": "running", "progress": 0.1, "message": "Memulai pelatihan model ensemble..."}, f)
                    
                    if mode == TRAINING_MODE_SUPERVISED:
                        detector_obj.fit(
                            X, edge_index=e_idx, edge_type=e_type, labels=y, device=dev,
                            optimize_hyperparams=opt_hyper,
                            optimize_ensemble_weights=opt_ensemble
                        )
                    else:
                        detector_obj.fit(
                            X, edge_index=e_idx, edge_type=e_type, labels=None, device=dev,
                            optimize_hyperparams=opt_hyper,
                            optimize_ensemble_weights=opt_ensemble
                        )
                        
                    with open("cache/training_status.json", "w") as f:
                        json.dump({"status": "completed", "progress": 1.0, "message": "Training selesai!"}, f)
                except Exception as e:
                    with open("cache/training_status.json", "w") as f:
                        json.dump({"status": "error", "progress": 0.0, "message": str(e)}, f)

            st.session_state['training_in_progress'] = True
            
            t = threading.Thread(
                target=train_worker, 
                args=(st.session_state['current_training_detector'], X_train, edge_index,
                      st.session_state.get('edge_type'),
                      y_train if training_mode == TRAINING_MODE_SUPERVISED else None,
                      device, training_mode,
                      enable_optuna_ensemble_weights,
                      enable_hyperparameter_tuning)
            )
            t.start()
            
            # Give it a tiny sleep so the thread creates the json file before rerun
            import time
            time.sleep(0.5)
            st.rerun()

    # Polling logic outside the button context
    if st.session_state.get('training_in_progress', False):
        import time
        import json
        
        st.warning("⚠️ Proses pelatihan sedang berjalan di background. Anda bisa membiarkan halaman ini terbuka.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with open("cache/training_status.json", "r") as f:
                status = json.load(f)
                
            progress_bar.progress(min(status.get("progress", 0.0), 1.0))
            status_text.text(status.get("message", "Memproses..."))
            
            if status.get("status") == "completed":
                st.session_state['training_in_progress'] = False
                
                # Finalize
                detector = st.session_state['current_training_detector']
                selected_features = st.session_state['current_training_features']
                training_mode = st.session_state['current_training_mode']
                label_column = st.session_state['current_training_label_column']
                
                st.session_state['detector'] = detector
                st.session_state['model_trained'] = True
                st.session_state['training_features'] = selected_features
                st.session_state['training_mode'] = training_mode
                st.session_state['training_label_column'] = label_column
                
                feature_selection_method = st.session_state.get('feature_selection_method', 'Semua Fitur (Bawaan)')
                
                detector.save_models(
                    MODEL_PREFIX,
                    training_metadata={
                        'training_features': selected_features,
                        'feature_selection_method': feature_selection_method,
                        'training_mode': training_mode,
                        'label_column': label_column,
                        'graph_method': st.session_state.get('graph_method', 'star'),
                        'graph_k': st.session_state.get('graph_k', graph_k if 'graph_k' in locals() else None),
                        'graph_node_count': st.session_state.get('graph_node_count', 0),
                        'graph_edge_count': st.session_state.get('graph_edge_count', 0),
                    }
                )
                
                version = save_model_version(MODEL_PREFIX)
                st.success(f"✅ Model training completed and saved as version {version}!")
                
                # We can remove the temp json
                import os
                if os.path.exists("cache/training_status.json"):
                    os.remove("cache/training_status.json")
                    
                # Clean up session state for UI flow
                st.session_state.pop('current_training_detector', None)
                
            elif status.get("status") == "error":
                st.session_state['training_in_progress'] = False
                st.error(f"❌ Error saat training: {status.get('message')}")
            else:
                # Still running
                time.sleep(2)
                st.rerun()
                
        except Exception as e:
            # File might not be created yet, retry
            time.sleep(2)
            st.rerun()

    if st.session_state.get('model_trained', False) and not st.session_state.get('training_in_progress', False):
        # We only show the download buttons and summary if it's completely done
        detector = st.session_state.get('detector')
        if detector:
            # Download model button
            st.subheader("💾 Unduh Model untuk Dibagikan")
            try:
                zip_buffer = zip_model_artifacts()
                st.download_button(
                    label="📥 Unduh Model (ZIP)",
                    data=zip_buffer,
                    file_name="fraud_detector_model.zip",
                    mime="application/zip",
                    help="Unduh semua file model dalam satu file ZIP untuk dibagikan atau digunakan nanti"
                )
                st.info("💡 Tips: Ekstrak file ZIP ini ke direktori `models/` di aplikasi untuk menggunakan model kembali.")
            except Exception as e:
                st.error(f"❌ Gagal membuat file ZIP: {str(e)}")
            
            # Show model summary
            st.subheader("📊 Model Summary")
            try:
                
                # Collect weights and statuses
                models_used = []
                weights = []
                statuses = []
                details = []
                
                if detector.isolation_weight > 0:
                    models_used.append("Isolation Forest")
                    weights.append(detector.isolation_weight)
                    statuses.append("✅ Trained")
                    details.append(f"Contamination: {iso_contamination}")
                    
                if detector.autoencoder_weight > 0:
                    models_used.append("Autoencoder")
                    weights.append(detector.autoencoder_weight)
                    statuses.append("✅ Trained")
                    details.append(f"Encoding Dim: {ae_encoding_dim}")
                    
                if detector.dbscan_weight > 0:
                    models_used.append("DBSCAN")
                    weights.append(detector.dbscan_weight)
                    statuses.append("✅ Trained")
                    details.append(f"eps: {dbscan_eps}")
                    
                if detector.gnn_weight > 0:
                    models_used.append("GNN")
                    if hasattr(detector, 'gnn_model') and detector.gnn_model is not None:
                        weights.append(detector.gnn_weight)
                        statuses.append("✅ Trained")
                        details.append(f"Hidden: {gnn_hidden if 'gnn_hidden' in locals() else 64}, Heads: {gnn_heads if 'gnn_heads' in locals() else 4}")
                    else:
                        weights.append(0)
                        statuses.append("⚠️ Skipped")
                        details.append("Model not initialized")
                    
                if detector.xgboost_weight > 0:
                    models_used.append("Supervised/XGBoost")
                    if hasattr(detector, 'xgboost_model') and detector.xgboost_model is not None:
                        weights.append(detector.xgboost_weight)
                        statuses.append("✅ Trained")
                        details.append(f"Estimators: {xgb_n_estimators}")
                    else:
                        weights.append(0)
                        statuses.append("⚠️ Skipped")
                        details.append("Model not initialized")
                        
                # Create a row of Metric Cards
                metrics_cols = st.columns(len(models_used) if models_used else 1)
                for i, model_name in enumerate(models_used):
                    with metrics_cols[i]:
                        st.metric(model_name, statuses[i])
                        st.caption(details[i])
                        st.caption(f"**Weight:** {weights[i]:.3f}")
                        
                # Create Pie Chart for Weights
                if sum(weights) > 0:
                    import plotly.express as px
                    fig = px.pie(
                        names=models_used, 
                        values=weights, 
                        title="⚖️ Distribusi Bobot Ensemble Model",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, width='stretch')
                
                # Show Optuna Ensemble Weight Optimization Results
                if getattr(detector, 'weight_optimization_results', None):
                    opt_res = detector.weight_optimization_results
                    cmp = opt_res.get('metric_comparison', {})
                    def_m = cmp.get('default', {})
                    opt_m = cmp.get('optimized', {})
                    fpr_red = cmp.get('fpr_reduction_pct', 0.0)
                    
                    st.success(f"⚡ **Optuna Ensemble Weights Applied!** Berhasil mereduksi False Positive Rate sebesar **{fpr_red:.1f}%**")
                    with st.expander("📊 Detail Evaluasi Optimasi Bobot Ensemble (Optuna vs Default)", expanded=True):
                        opt_c1, opt_c2, opt_c3, opt_c4 = st.columns(4)
                        with opt_c1:
                            st.metric("False Positive Rate (FPR)", f"{opt_m.get('fpr', 0):.2%}", delta=f"{-fpr_red:.1f}%", delta_color="inverse")
                        with opt_c2:
                            st.metric("Precision (Presisi)", f"{opt_m.get('precision', 0):.2%}", delta=f"{(opt_m.get('precision', 0) - def_m.get('precision', 0)):.2%}")
                        with opt_c3:
                            st.metric("Recall (Sensitivitas)", f"{opt_m.get('recall', 0):.2%}", delta=f"{(opt_m.get('recall', 0) - def_m.get('recall', 0)):.2%}")
                        with opt_c4:
                            st.metric("F1-Score / F0.5", f"{opt_m.get('f1', 0):.2%}", delta=f"{(opt_m.get('f1', 0) - def_m.get('f1', 0)):.2%}")
                        
                        st.caption(f"Optimasi diselesaikan dalam {opt_res.get('n_trials_completed', 0)} trials menggunakan Stratified K-Fold Cross-Validation.")
                elif use_dynamic_weights:
                    st.success("🎯 **Dynamic Weights Applied**: Weights were automatically optimized based on your data characteristics!")
                    
                    # Show weight optimization details
                    train_df_current = st.session_state.get('train_df')
                    current_sample_count = len(train_df_current) if train_df_current is not None else len(df_processed)
                    with st.expander("🔍 Weight Optimization Details"):
                        st.write(f"""
                        **Final Optimized Weights:**
                        - **Isolation Forest**: {detector.isolation_weight:.3f} ({detector.isolation_weight*100:.1f}%)
                        - **Autoencoder**: {detector.autoencoder_weight:.3f} ({detector.autoencoder_weight*100:.1f}%)
                        - **XGBoost**: {detector.xgboost_weight:.3f} ({detector.xgboost_weight*100:.1f}%)
                        
                        **Data Characteristics Analysis:**
                        - Data Size: {current_sample_count} samples
                        - Feature Count: {len(selected_features)} features
                        - XGBoost Features: {'Ready' if current_sample_count > 1 else 'Insufficient Data'}
                        """)
                else:
                    st.info("🎛️ **Manual Weights Applied**: Using manually configured weights.")
                
                # Graph Visualization if GNN was trained
                current_edge_index = st.session_state.get('edge_index', edge_index)
                if detector.gnn_weight > 0 and hasattr(detector, 'gnn_model') and detector.gnn_model is not None and current_edge_index is not None:
                    st.markdown("---")
                    st.subheader("🕸️ Visualisasi Graph Network")

                    try:
                        import networkx as nx
                        import plotly.graph_objects as go

                        # Score the same graph nodes that are rendered. Evaluation
                        # rows are not interchangeable with training graph nodes.
                        graph_scores = None
                        try:
                            _, graph_individual_probs = detector.predict_anomaly_probability(
                                node_features,
                                edge_index=current_edge_index,
                                edge_type=st.session_state.get('edge_type'),
                                device=device,
                            )
                            graph_scores = np.asarray(graph_individual_probs.get('gnn', []), dtype=float)
                            if graph_scores.size != node_features.shape[0]:
                                graph_scores = None
                        except Exception as score_error:
                            logger.warning("GNN visualization scoring failed: %s", score_error)

                        # Render a relevant graph slice while preserving stable node IDs.
                        total_edges = int(current_edge_index.shape[1])
                        requested_nodes = int(st.number_input(
                            "Jumlah node untuk visualisasi",
                            min_value=1,
                            max_value=max(1, int(node_features.shape[0]) if node_features is not None else total_edges),
                            value=min(300, max(1, int(node_features.shape[0]) if node_features is not None else total_edges)),
                            step=50,
                            key="gnn_visualization_node_limit",
                        ))
                        max_nodes = requested_nodes
                        edge_list = current_edge_index.t().tolist()
                        edge_type_values = None
                        current_edge_type = st.session_state.get('edge_type')
                        if current_edge_type is not None:
                            edge_type_values = np.asarray(current_edge_type).reshape(-1).tolist()

                        # Create networkx graph
                        G = nx.Graph()

                        if graph_scores is not None:
                            selected_nodes = set(np.argsort(graph_scores)[-max_nodes:].tolist())
                        else:
                            selected_nodes = set(range(min(max_nodes, int(node_features.shape[0]))))
                        G.add_nodes_from(selected_nodes)
                        selected_edges = []
                        for edge_index_position, edge in enumerate(edge_list):
                            if edge[0] in selected_nodes and edge[1] in selected_nodes:
                                G.add_edge(edge[0], edge[1])
                                selected_edges.append((edge_index_position, edge))

                        # Check if graph has nodes
                        if len(G.nodes()) == 0:
                            st.warning("⚠️ Graph tidak memiliki nodes untuk divisualisasikan.")
                        else:
                            # Get layout
                            pos = nx.spring_layout(G, seed=42)

                            # Prepare relation-specific edge traces. A
                            # heterogeneous graph keeps provider/patient/
                            # diagnosis edges visually distinguishable.
                            relation_colors = {
                                0: ('Provider', '#2563eb'),
                                1: ('Patient', '#10b981'),
                                2: ('Diagnosis', '#f59e0b'),
                            }
                            edge_traces = []
                            relation_groups = {}
                            for edge_position, edge in selected_edges:
                                relation_id = (
                                    edge_type_values[edge_position]
                                    if edge_type_values is not None and edge_position < len(edge_type_values)
                                    else None
                                )
                                relation_groups.setdefault(relation_id, []).append(edge)
                            for relation_id, relation_edges in relation_groups.items():
                                edge_x = []
                                edge_y = []
                                for edge in relation_edges:
                                    x0, y0 = pos[edge[0]]
                                    x1, y1 = pos[edge[1]]
                                    edge_x.extend([x0, x1, None])
                                    edge_y.extend([y0, y1, None])
                                label, color = relation_colors.get(
                                    relation_id, ('Relation', '#888888')
                                )
                                edge_traces.append(go.Scatter(
                                    x=edge_x, y=edge_y,
                                    line=dict(width=0.7, color=color),
                                    hoverinfo='none',
                                    mode='lines',
                                    name=label,
                                ))

                            # Prepare node traces with anomaly coloring
                            node_x = []
                            node_y = []
                            node_colors = []
                            node_text = []

                            # Get anomaly predictions for coloring
                            try:
                                eval_result_df = st.session_state.get('eval_result_df')
                                if eval_result_df is not None and 'anomaly_probability' in eval_result_df.columns:
                                    if 'node_id' in eval_result_df.columns:
                                        anomaly_probs = dict(zip(
                                            eval_result_df['node_id'],
                                            eval_result_df['anomaly_probability'],
                                        ))
                                    else:
                                        anomaly_probs = dict(enumerate(eval_result_df['anomaly_probability'].values))
                                    for node in G.nodes():
                                        x, y = pos[node]
                                        node_x.append(x)
                                        node_y.append(y)
                                        # Color based on anomaly probability
                                        if graph_scores is not None and node < len(graph_scores):
                                            prob = float(graph_scores[node])
                                            node_colors.append(prob)
                                            node_text.append(f"Node {node}<br>GNN Probability: {prob:.3f}")
                                        elif node in anomaly_probs:
                                            prob = anomaly_probs[node]
                                            node_colors.append(prob)
                                            node_text.append(f"Node {node}<br>Anomaly Prob: {prob:.3f}")
                                        else:
                                            node_colors.append(0.5)
                                            node_text.append(f"Node {node}")
                                else:
                                    for node in G.nodes():
                                        x, y = pos[node]
                                        node_x.append(x)
                                        node_y.append(y)
                                        node_colors.append(0.5)
                                        node_text.append(f"Node {node}")
                            except Exception:
                                # Fallback to simple coloring
                                for node in G.nodes():
                                    x, y = pos[node]
                                    node_x.append(x)
                                    node_y.append(y)
                                    node_colors.append(0.5)
                                    node_text.append(f"Node {node}")

                            node_trace = go.Scatter(
                                x=node_x, y=node_y,
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(
                                    size=8,
                                    color=node_colors,
                                    colorscale='RdYlBu_r',  # Red for high anomaly, blue for low
                                    cmin=0,
                                    cmax=1,
                                    showscale=True,
                                    colorbar=dict(title="Anomaly Probability"),
                                    line=dict(width=1, color='blue')
                                ),
                                text=node_text
                            )

                            # Create figure
                            fig = go.Figure(data=edge_traces + [node_trace],
                                           layout=go.Layout(
                                               title=f'Graph Network Visualization ({len(G.nodes())} nodes, {len(G.edges())} edges)',
                                               showlegend=edge_type_values is not None,
                                               hovermode='closest',
                                               margin=dict(b=20, l=20, r=20, t=40),
                                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                               height=600
                                           ))

                            st.plotly_chart(fig, width='stretch')
                            st.info(f"💡 Visualisasi menampilkan subset graph ({max_nodes} nodes) untuk performa. Graph asli memiliki {current_edge_index.shape[1]} edges. Warna node menunjukkan probabilitas anomali (merah = tinggi, biru = rendah).")

                    except Exception as viz_error:
                        st.warning(f"⚠️ Gagal memvisualisasikan graph: {str(viz_error)}")
                        st.info("Visualisasi graph membutuhkan library networkx. Install dengan: pip install networkx")
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                logger.error(f"Training error: {e}", exc_info=True)
                
                # Provide recovery options
                st.markdown("---")
                st.subheader("🔧 Opsi Pemulihan")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Coba Lagi", key="retry_training"):
                        st.info("Silakan konfigurasi ulang parameter dan coba training lagi.")
                
                with col2:
                    if st.button("📊 Bagi Ulang Data", key="resplit_data"):
                        # Clear train/test split to force re-split using batch update
                        batch_state_update({
                            'train_df': None,
                            'test_df': None
                        })
                
                with col3:
                    if st.button("🏠 Kembali ke Beranda", key="training_to_home"):
                        navigate_to_page('home')
                
                # Show debug information in expandable section
                with st.expander("🔍 Debug Information"):
                    st.write("**Error Type:**", type(e).__name__)
                    st.write("**Error Message:**", str(e))
                    st.write("**Session State Keys:**", list(st.session_state.keys()))
                    
                    # Check critical state variables
                    critical_keys = ['train_df', 'test_df', 'selected_features', 'df_processed_path']
                    st.write("**Critical State Status:**")
                    for key in critical_keys:
                        status = "✅ Ada" if key in st.session_state else "❌ Tidak ada"
                        st.write(f"  - {key}: {status}")
                
                return
            
            # Navigation to next step
            st.markdown("---")
            st.subheader("🚀 Langkah Selanjutnya")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Evaluasi Model", key="evaluate_model", type="secondary"):
                    navigate_to_page('evaluate')
            
            with col2:
                if st.button("🔍 Lanjut ke Deteksi Anomali", key="proceed_to_detection", type="primary"):
                    navigate_to_page('detect')

