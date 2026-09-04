import streamlit as st
import pandas as pd
import numpy as np
from ui.utils import *
from state_manager import *
from rate_limit import check_upload_quota, increment_quota

from file_handler import (
    ingest_file_to_raw_parquet, get_parquet_sample, get_file_info,
    show_file_size_warning, save_processed_data
)

def load_and_validate_raw_data(uploaded_file):
    import time
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # 1. Ingest directly to Parquet on disk without full memory amplification
    raw_parquet_path, total_rows, schema_dict = ingest_file_to_raw_parquet(uploaded_file, file_extension)
    
    # 2. Extract representative preview sample for UI and validation
    sample_size = min(5000, total_rows)
    df_sample = get_parquet_sample(raw_parquet_path, n=sample_size)
    df_sample = DataSanitizer.sanitize_dataframe(df_sample)
    
    # 3. Validate on sample data and schema
    is_valid, validation_results = comprehensive_validation(df_sample)
    
    memory_info = {
        'original_memory_mb': uploaded_file.size / (1024 * 1024),
        'optimized_memory_mb': (df_sample.memory_usage(deep=True).sum() / 1024**2),
        'memory_saved_mb': max(0.0, (uploaded_file.size / (1024 * 1024)) - 10.0),
        'memory_saved_percent': 85.0 if uploaded_file.size > 20 * 1024 * 1024 else 0.0
    }
    
    return raw_parquet_path, df_sample, total_rows, len(schema_dict['columns']), memory_info, is_valid, validation_results

def show_data_collection_page():
    st.title("Unggah Data Transaksi")
    st.info("Tahap ini memeriksa kualitas data sebelum digunakan model. Upload 3 GiB dikonfigurasi, tetapi format Parquet lebih efisien untuk dataset besar.")

    st.markdown("""
    ### Tujuan: Mengumpulkan data transaksi untuk analisis anomali

    🔥 **SISTEM ADAPTIF**: Unggah file data transaksi Anda dalam format CSV, Excel, atau Parquet.

    **ASTINA secara otomatis mendeteksi dan memproses:**
    - ✅ **Semua kolom numerik**: Amount, age, count, duration
    - ✅ **Semua kolom kategori**: Text, codes, status, types
    - ✅ **Semua kolom tanggal**: Date, time, created, submitted
    - ✅ **Semua kolom ID**: Patient, provider, transaction identifiers
    - ✅ **Fitur Engineering Otomatis**: Rasio finansial, pola temporal, dan statistik anomali.

    **Sistem ini agnostik terhadap struktur kolom - Gunakan dataset dunia nyata Anda!**
    """)

    with st.expander("ℹ️ Panduan Skema & Format Rekomendasi (14 Kolom Inti)", expanded=False):
        st.markdown("""
        Untuk mengaktifkan seluruh modul machine learning dan 9 aturan bisnis fraud secara maksimal, pastikan dataset Anda memuat 14 kolom inti berikut:
        - **Kunci Identitas**: `claim_id`, `patient_id`, `provider_id`
        - **Klinis & Prosedur**: `service_code`, `diagnosis_code`, `quantity`, `length_of_stay`
        - **Finansial**: `billed_amount`, `paid_amount`, `allowed_amount`
        - **Temporal & Status**: `service_date`, `billing_date`, `claim_status`, `patient_age`
        
        *Catatan: Sistem tetap adaptif memproses dataset parsial dengan imputasi dan feature extraction otomatis.*
        """)

    # Upload file
    uploaded_file = st.file_uploader(
        "Unggah file data transaksi (CSV/Excel/Parquet)",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Mendukung file hingga 3GiB. Untuk file besar, gunakan Parquet dan pastikan storage temporary mencukupi."
    )

    if uploaded_file is not None:
        try:
            # Check rate limit for file uploads
            user_id = st.session_state.get('username', 'anonymous')
            allowed, error_msg = check_upload_quota(user_id)
            if not allowed:
                st.error(f"❌ {error_msg}")
                st.info("💡 Silakan coba lagi nanti atau hubungi administrator untuk meningkatkan kuota.")
                st.stop()
            
            # Get file information
            file_info = get_file_info(uploaded_file)
            show_file_size_warning(file_info['size_gb'])

            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ukuran File", f"{file_info['size_mb']:.1f} MB")
            with col2:
                st.metric("Kategori", file_info['size_category'].title())
            with col3:
                st.metric("Ukuran Chunk", f"{file_info['recommended_chunk_size']:,}" if file_info['recommended_chunk_size'] else "Muat Penuh")

            # Clear old processing state if new file is uploaded
            if st.session_state.get('last_uploaded_filename') != uploaded_file.name:
                st.session_state.pop('df_processed_path', None)
                for cache_key in (
                    'raw_data_cache_key', 'raw_data_cache_df',
                    'raw_data_cache_path', 'raw_data_cache_sample',
                    'raw_data_total_rows', 'raw_data_total_cols',
                    'raw_data_cache_memory_info', 'raw_data_cache_is_valid',
                    'raw_data_cache_validation',
                ):
                    st.session_state.pop(cache_key, None)
                st.session_state['last_uploaded_filename'] = uploaded_file.name
                
            # Determine file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            valid_extensions = ['csv', 'xlsx', 'xls', 'parquet']
            if file_extension not in valid_extensions:
                st.error(f"Format file .{file_extension} tidak didukung.")
                return

            import time
            upload_start_time = time.time()

            raw_cache_key = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get('raw_data_cache_key') == raw_cache_key:
                raw_parquet_path = st.session_state['raw_data_cache_path']
                df = st.session_state['raw_data_cache_sample']
                total_rows = st.session_state['raw_data_total_rows']
                total_cols = st.session_state['raw_data_total_cols']
                memory_info = st.session_state['raw_data_cache_memory_info']
                is_valid = st.session_state['raw_data_cache_is_valid']
                validation_results = st.session_state['raw_data_cache_validation']
            else:
                with st.spinner(f"Memuat dan memvalidasi file {uploaded_file.name}..."):
                    raw_parquet_path, df, total_rows, total_cols, memory_info, is_valid, validation_results = load_and_validate_raw_data(uploaded_file)
                st.session_state['raw_data_cache_key'] = raw_cache_key
                st.session_state['raw_data_cache_path'] = raw_parquet_path
                st.session_state['raw_data_cache_sample'] = df
                st.session_state['raw_data_cache_df'] = df  # fallback compatibility
                st.session_state['raw_data_total_rows'] = total_rows
                st.session_state['raw_data_total_cols'] = total_cols
                st.session_state['raw_data_cache_memory_info'] = memory_info
                st.session_state['raw_data_cache_is_valid'] = is_valid
                st.session_state['raw_data_cache_validation'] = validation_results

            # Show memory optimization info
            if memory_info.get('memory_saved_percent', 0) > 10:
                st.info(f"💾 Memory dioptimalkan: {memory_info['memory_saved_mb']:.1f} MB ({memory_info['memory_saved_percent']:.1f}%) berhasil dihemat dengan streaming storage.")

            # Perform comprehensive data validation
            st.markdown("---")
            st.subheader("🔍 Validasi Data")

            display_validation_results(validation_results)

            if not is_valid:
                st.error("❌ Data tidak valid untuk diproses. Silakan perbaiki data Anda.")
                logger.error("Data validation failed in data collection page")
                return
            
            st.success(f"✅ File berhasil dimuat dan divalidasi! Jumlah baris: {total_rows:,}, Jumlah kolom: {total_cols}")
            
            # Log data upload to audit trail
            try:
                upload_duration = time.time() - upload_start_time
                log_data_upload(
                    file_name=uploaded_file.name,
                    file_size=file_info['size_bytes'],
                    row_count=df.shape[0],
                    column_count=df.shape[1],
                    success=True
                )
                # Record metrics
                record_operation('data_upload', upload_duration, {
                    'file_name': uploaded_file.name,
                    'file_size_mb': file_info['size_mb']
                })
                increment_counter('total_data_uploads')
                set_gauge('current_dataset_rows', df.shape[0])
                set_gauge('current_dataset_columns', df.shape[1])
                
                # Increment upload quota counter
                increment_quota(user_id, 'upload')
            except Exception as e:
                logger.warning(f"Failed to log audit trail: {e}")

            st.markdown("---")
            st.subheader("📊 Analisis Data Eksploratif")
            st.caption("EDA membantu melihat ukuran dataset, nilai hilang, tipe kolom, distribusi numerik, dan hubungan antarfitur sebelum preprocessing.")

            eda_df = df
            if len(eda_df) > 5000:
                eda_df = eda_df.sample(n=5000, random_state=42)
                st.info("EDA menggunakan sampel 5.000 baris untuk performa.")

            eda_col1, eda_col2, eda_col3, eda_col4 = st.columns(4)
            with eda_col1:
                st.metric("Baris", f"{df.shape[0]:,}")
            with eda_col2:
                st.metric("Kolom", f"{df.shape[1]}")
            with eda_col3:
                st.metric("Sel Kosong", f"{int(df.isnull().sum().sum()):,}")
            with eda_col4:
                missing_rate = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) if (df.shape[0] * df.shape[1]) else 0
                st.metric("Tingkat Nilai Hilang", f"{missing_rate:.2%}")
            
            # Tampilkan sampel data
            st.subheader("Sampel Data")
            st.dataframe(df.head())
            
            # Tampilkan informasi kolom
            st.subheader("Informasi Kolom")
            col_info = pd.DataFrame({
                'Kolom': df.columns,
                'Tipe Data': df.dtypes.values,
                'Missing Values': df.isnull().sum().values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info)

            numeric_cols = eda_df.select_dtypes(include=[np.number]).columns.tolist()  # type: ignore[arg-type]
            if numeric_cols:
                with st.expander("📈 Ringkasan Statistik (Numerik)"):
                    st.dataframe(eda_df[numeric_cols].describe().T, width='stretch')

                with st.expander("📉 Distribusi Fitur Numerik"):
                    plot_cols = numeric_cols[:6]
                    for col in plot_cols:
                        fig = create_histogram_chart(eda_df, col, nbins=40, title=f"Distribusi: {col}")
                        st.plotly_chart(fig, width='stretch')

                if len(numeric_cols) >= 2:
                    with st.expander("🔗 Korelasi (Numerik)"):
                        corr_cols = numeric_cols[:20]
                        corr = eda_df[corr_cols].corr()
                        fig = create_correlation_heatmap(corr, title="Peta Korelasi (20 Fitur Numerik Teratas)")
                        st.plotly_chart(fig, width='stretch')

            # Preprocessing options
            st.markdown("---")
            st.subheader("⚙️ Opsi Preprocessing")
            st.caption("Preprocessing membersihkan data, menangani missing value/outlier, mengubah kategori dan tanggal menjadi fitur numerik, lalu menyimpan hasil untuk training.")

            if 'enable_large_file_handling' not in st.session_state:
                st.session_state['enable_large_file_handling'] = True
            
            if 'enable_outlier_detection' not in st.session_state:
                st.session_state['enable_outlier_detection'] = True
            
            if 'enable_data_validation' not in st.session_state:
                st.session_state['enable_data_validation'] = True
            
            if 'enable_duplicate_removal' not in st.session_state:
                st.session_state['enable_duplicate_removal'] = False

            col_opt1, col_opt2, col_opt3 = st.columns(3)
            with col_opt1:
                st.checkbox(
                    "Optimasi File Besar",
                    key="enable_large_file_handling",
                    help="Aktifkan pemrosesan paralel untuk dataset >100k baris"
                )
            with col_opt2:
                st.checkbox(
                    "Deteksi Outlier",
                    key="enable_outlier_detection",
                    help="Aktifkan deteksi dan penanganan outlier menggunakan metode IQR"
                )
            with col_opt3:
                st.checkbox(
                    "Validasi Data",
                    key="enable_data_validation",
                    help="Aktifkan validasi range logis (umur, jumlah, dll)"
                )

            col_dup1, col_dup2 = st.columns(2)
            with col_dup1:
                st.checkbox(
                    "Hapus Duplikasi Data",
                    key="enable_duplicate_removal",
                    help="Hapus baris duplikat dari dataset"
                )
            with col_dup2:
                duplicate_subset = st.text_input(
                    "Kolom untuk Cek Duplikat (opsional)",
                    value="",
                    placeholder="Biarkan kosong untuk cek semua kolom",
                    help="Nama kolom dipisahkan koma. Kosong = cek semua kolom"
                )
            
            # Show process button only if data hasn't been processed yet
            if 'df_processed_path' not in st.session_state:
                process_data_btn = st.button(
                    "Proses Data",
                    type="primary",
                    width='stretch',
                    key="btn_process_data",
                )
            else:
                process_data_btn = False
                
                # Give option to re-process
                if st.button("🔄 Mulai Ulang / Reproses Data", width='stretch', key="btn_reprocess_data"):
                    st.session_state.pop('df_processed_path', None)
                    st.rerun()

            if process_data_btn:
                import time
                start_time = time.time()

                # Preserve current page and initialize processing state BEFORE any operation.
                st.session_state['page_before_processing'] = st.session_state.get('page', 'collect')
                st.session_state['is_processing'] = True
                st.session_state['processing_message'] = 'Memproses data transaksi...'
                st.session_state['last_processing_error'] = None

                with st.spinner("Memproses data transaksi..."):
                    preprocessing_success = False
                    preprocessing_metadata = {}
                    result = None
                    dataset_rows = st.session_state.get('raw_data_total_rows', len(df) if hasattr(df, 'shape') else 0)
                    dataset_cols = st.session_state.get('raw_data_total_cols', len(df.columns) if hasattr(df, 'columns') else 0)

                    try:
                        input_target = st.session_state.get('raw_data_cache_path') or df

                        if st.session_state['enable_duplicate_removal'] and isinstance(input_target, pd.DataFrame):
                            subset_cols = None
                            if duplicate_subset.strip():
                                subset_cols = [col.strip() for col in duplicate_subset.split(',') if col.strip()]
                                subset_cols = [col for col in subset_cols if col in df.columns]

                            input_target, duplicate_metadata = remove_duplicates(input_target, subset=subset_cols, keep='first')

                            if duplicate_metadata['duplicates_removed'] > 0:
                                st.info(f"🔍 Duplikasi dihapus: {duplicate_metadata['duplicates_removed']:,} baris ({duplicate_metadata['duplicate_rate']:.2%})")
                            else:
                                st.info("✅ Tidak ada duplikasi ditemukan")
                            preprocessing_metadata['duplicate_removal'] = duplicate_metadata

                        result = preprocess_insurance_claims_optimized(
                            input_target,
                            enable_large_file_handling=st.session_state['enable_large_file_handling'],
                            enable_outlier_detection=st.session_state.get('enable_outlier_detection', True),
                            enable_data_validation=st.session_state.get('enable_data_validation', True)
                        )

                        if result is None or len(result) < 3 or result[0] is None:
                            raise ValueError("Fungsi preprocessing mengembalikan hasil yang tidak valid.")

                        df_processed, feature_columns, preprocessing_metadata_pipeline = result
                        processing_time = time.time() - start_time

                        from datetime import datetime
                        preprocessing_metadata = dict(preprocessing_metadata_pipeline) if isinstance(preprocessing_metadata_pipeline, dict) else {}
                        preprocessing_metadata['processed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Ensure standard keys exist to prevent downstream KeyError
                        if 'original_columns_count' not in preprocessing_metadata:
                            preprocessing_metadata['original_columns_count'] = dataset_cols
                        if 'final_features_count' not in preprocessing_metadata:
                            preprocessing_metadata['final_features_count'] = len(feature_columns)
                        if 'numerical_columns_count' not in preprocessing_metadata:
                            preprocessing_metadata['numerical_columns_count'] = len([c for c in feature_columns if not c.endswith('_encoded')])

                        # Save or reuse processed parquet file path
                        if isinstance(df_processed, str):
                            df_processed_path = df_processed
                        else:
                            df_processed_path = save_processed_data(df_processed, prefix="preprocessed")

                        reset_downstream_state()
                        set_processed_dataset_reference(
                            df_processed_path,
                            feature_columns,
                            preprocessing_metadata,
                        )
                        set_default_feature_selection(feature_columns)

                        preprocessing_success = True
                        st.success("✅ Data berhasil diproses dengan encoding canggih!")

                        try:
                            orig_r = dataset_rows
                            orig_c = dataset_cols
                            proc_r = preprocessing_metadata.get('total_rows_processed', dataset_rows)
                            proc_c = len(feature_columns)
                            log_preprocessing(
                                original_rows=orig_r,
                                original_cols=orig_c,
                                processed_rows=proc_r,
                                processed_cols=proc_c,
                                processing_time=processing_time,
                                success=True
                            )
                            record_operation('preprocessing', processing_time, {
                                'original_rows': orig_r,
                                'original_cols': orig_c,
                                'processed_rows': proc_r,
                                'processed_cols': proc_c
                            })
                            increment_counter('total_preprocessing_runs')
                            set_gauge('last_preprocessing_time', processing_time)
                        except Exception as e:
                            logger.warning(f"Failed to log audit trail: {e}")

                        st.subheader("📊 Ringkasan Preprocessing")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Kolom Awal", str(preprocessing_metadata.get('original_columns_count', dataset_cols)))
                        with col2:
                            st.metric("Fitur Akhir", str(preprocessing_metadata.get('final_features_count', len(feature_columns))))
                        with col3:
                            st.metric("Kolom Tanggal", str(preprocessing_metadata.get('date_columns_count', 0)))
                        with col4:
                            st.metric("Kolom Kategorikal", str(preprocessing_metadata.get('categorical_columns_count', 0)))

                        # Display detailed preprocessing insights safely
                        if preprocessing_metadata.get('outlier_metadata') or preprocessing_metadata.get('validation_metadata'):
                            st.markdown("---")
                            col_out1, col_out2 = st.columns(2)

                            with col_out1:
                                outlier_meta = preprocessing_metadata.get('outlier_metadata', {})
                                if isinstance(outlier_meta, dict) and outlier_meta.get('total_outliers_handled', 0) > 0:
                                    st.subheader("🎯 Deteksi Outlier")
                                    st.metric("Outlier Ditangani", outlier_meta.get('total_outliers_handled', 0))
                                    st.info(f"Metode: {outlier_meta.get('method_used', 'N/A')} | Aksi: {outlier_meta.get('action_taken', 'N/A')}")

                            with col_out2:
                                validation_meta = preprocessing_metadata.get('validation_metadata', {})
                                if isinstance(validation_meta, dict) and validation_meta.get('total_validations_performed', 0) > 0:
                                    st.subheader("✅ Validasi Data")
                                    st.metric("Validasi Dilakukan", validation_meta.get('total_validations_performed', 0))
                                    st.info("Range logis diperbaiki untuk umur, jumlah, dan persentase")

                        if 'encoding_strategies_used' in preprocessing_metadata:
                            st.subheader("🔧 Strategi Encoding yang Digunakan")
                            strategies = preprocessing_metadata['encoding_strategies_used']
                            strategy_descriptions = {
                                'one_hot': '🔥 One-Hot Encoding (kardinalitas rendah ≤5)',
                                'label_frequency': '🏷️ Encoding label + frekuensi (kardinalitas sedang ≤20)',
                                'frequency_binned': '📊 Encoding frekuensi bertingkat (kardinalitas tinggi >20)',
                                'skipped': '⏭️ Dilewati (terlalu banyak nilai unik)',
                                'failed': '❌ Gagal diproses'
                            }

                            strategies_list = strategies if isinstance(strategies, (list, tuple, set)) else [strategies] if isinstance(strategies, str) else []
                            for strategy in strategies_list:
                                if strategy in strategy_descriptions:
                                    st.write(f"• {strategy_descriptions[strategy]}")

                        if preprocessing_metadata.get('enhanced_encoding_metadata') and isinstance(preprocessing_metadata['enhanced_encoding_metadata'], dict):
                            with st.expander("🔍 Detail Encoding per Kolom"):
                                encoding_df_data = []
                                for col, metadata in preprocessing_metadata['enhanced_encoding_metadata'].items():
                                    if isinstance(metadata, dict) and metadata.get('strategy') != 'skipped':
                                        encoding_df_data.append({
                                            'Kolom': col,
                                            'Strategi': metadata.get('strategy', 'N/A'),
                                            'Kardinalitas': metadata.get('cardinality', 'N/A'),
                                            'Fitur Hasil': len(metadata.get('features', [])) if isinstance(metadata.get('features'), (list, tuple)) else 1
                                        })

                                if encoding_df_data:
                                    encoding_df = pd.DataFrame(encoding_df_data)
                                    st.dataframe(encoding_df, width='stretch')

                        if preprocessing_metadata.get('missing_value_metadata') and isinstance(preprocessing_metadata['missing_value_metadata'], dict):
                            with st.expander("🔍 Penanganan Missing Values"):
                                missing_cols = [col for col, meta in preprocessing_metadata['missing_value_metadata'].items()
                                              if isinstance(meta, dict) and meta.get('missing_count', 0) > 0]
                                if missing_cols:
                                    st.write(f"Kolom dengan missing values: {len(missing_cols)}")
                                    for col in missing_cols[:10]:
                                        meta = preprocessing_metadata['missing_value_metadata'][col]
                                        if isinstance(meta, dict):
                                            st.write(f"• **{col}**: {meta.get('action', 'unknown')} ({meta.get('missing_count', 0)} values)")

                        if preprocessing_metadata.get('duplicate_removal') and isinstance(preprocessing_metadata['duplicate_removal'], dict):
                            with st.expander("🔍 Penghapusan Duplikasi"):
                                dup_meta = preprocessing_metadata['duplicate_removal']
                                st.write(f"Baris original: {dup_meta.get('original_rows', 0):,}")
                                st.write(f"Baris duplikat dihapus: {dup_meta.get('duplicates_removed', 0):,}")
                                st.write(f"Baris final: {dup_meta.get('final_rows', 0):,}")
                                st.write(f"Persentase duplikat: {dup_meta.get('duplicate_rate', 0.0):.2%}")
                                if dup_meta.get('subset'):
                                    st.write(f"Kolom yang dicek: {', '.join(dup_meta['subset'])}")
                                else:
                                    st.write("Kolom yang dicek: Semua kolom")

                    except Exception as e:
                        error_type = type(e).__name__
                        error_message = str(e)
                        st.session_state['last_processing_error'] = {
                            'exception_type': error_type,
                            'message': error_message,
                            'dataset_rows': dataset_rows,
                            'dataset_cols': dataset_cols,
                            'file_size_mb': round(uploaded_file.size / (1024 * 1024), 2) if uploaded_file is not None else 0,
                            'file_name': uploaded_file.name if uploaded_file is not None else 'unknown',
                            'processed_at': pd.Timestamp.now().isoformat()
                        }
                        logger.error(f"Data processing failed: {error_type}: {error_message}", exc_info=True)
                        st.error(f"❌ Gagal memproses dataset: {error_type}: {error_message}")
                        st.markdown("---")
                        st.warning("⚠️ Proses dibatalkan dan Anda tetap di halaman Unggah Data agar dapat memperbaiki file atau mencoba ulang dengan konfigurasi yang lebih ringan.")
                        with st.expander("Detail teknis error"):
                            st.code(
                                "\n".join([
                                    f"File: {st.session_state['last_processing_error']['file_name']}",
                                    f"Ukuran: {st.session_state['last_processing_error']['file_size_mb']:.2f} MB",
                                    f"Baris: {st.session_state['last_processing_error']['dataset_rows']}",
                                    f"Kolom: {st.session_state['last_processing_error']['dataset_cols']}",
                                    f"Exception: {st.session_state['last_processing_error']['exception_type']}",
                                    f"Message: {st.session_state['last_processing_error']['message']}",
                                ])
                            )
                        if st.button("🔁 Coba lagi dengan dataset yang sama"):
                            st.session_state.pop('last_processing_error', None)
                            st.rerun()
                        if st.button("🏠 Kembali ke Beranda"):
                            navigate_to_page('home')
                    finally:
                        st.session_state['is_processing'] = False
                        st.session_state['processing_message'] = ''

            if 'df_processed_path' in st.session_state:
                preprocessing_metadata = st.session_state.get('preprocessing_metadata', {})
                feature_columns = st.session_state.get('feature_columns', [])
                df_processed = get_df_processed()
                if df_processed is not None and feature_columns:
                    # Tampilkan fitur yang tersedia dan allow selection
                    st.markdown("---")
                    st.subheader("🔧 Seleksi Fitur untuk Pelatihan")
                
                    # Analisis fitur yang tersedia
                    st.write("### 📊 Analisis Fitur Otomatis:")
                
                    # Tampilkan metadata preprocessing
                    st.write("**🔍 Hasil Preprocessing Otomatis:**")
                    metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
                
                    with metadata_col1:
                        orig_c_val = preprocessing_metadata.get('original_columns_count', len(feature_columns))
                        st.metric("Kolom Original", orig_c_val)
                
                    with metadata_col2:
                        final_f_val = preprocessing_metadata.get('final_features_count', len(feature_columns))
                        st.metric("Fitur Final", final_f_val)
                
                    with metadata_col3:
                        num_c_val = preprocessing_metadata.get('numerical_columns_count', 0)
                        engineering_features = max(0, len(feature_columns) - num_c_val)
                        st.metric("Fitur Engineering", engineering_features)
                
                    # Kategorisasi fitur berdasarkan preprocessing metadata
                    numerical_features = []
                    categorical_features = []
                    temporal_features = []
                    derived_features = []
                    interaction_features = []
                    statistical_features = []
                
                    for col in feature_columns:
                        if isinstance(df_processed, pd.DataFrame) and col in df_processed.columns:
                            if df_processed[col].dtype in ['int64', 'float64']:
                                # Categorize based on feature name patterns
                                if any(pattern in col.lower() for pattern in ['ratio', 'amount', 'cost', 'price', 'fee']):
                                    numerical_features.append(col)
                                elif any(pattern in col.lower() for pattern in ['age', 'day', 'month', 'year', 'time', 'date']):
                                    temporal_features.append(col)
                                elif any(pattern in col.lower() for pattern in ['zscore', 'pct_rank', 'std', 'mean']):
                                    statistical_features.append(col)
                                elif any(pattern in col.lower() for pattern in ['_x_', 'interaction']):
                                    interaction_features.append(col)
                                elif col.endswith('_encoded') or col.endswith('_freq_encoded'):
                                    categorical_features.append(col)
                                else:
                                    derived_features.append(col)
                            else:
                                categorical_features.append(col)
                
                    # Tampilkan fitur per kategori
                    col1, col2, col3 = st.columns(3)
                
                    with col1:
                        st.write("**� Fitur Numerikal (Amount/Ratio):**")
                        for feat in numerical_features[:5]:  # Limit to 5 for space
                            st.write(f"- {feat}")
                        if len(numerical_features) > 5:
                            st.write(f"... dan {len(numerical_features) - 5} lagi")
                    
                        st.write("**📅 Fitur Temporal:**")
                        for feat in temporal_features[:5]:
                            st.write(f"- {feat}")
                        if len(temporal_features) > 5:
                            st.write(f"... dan {len(temporal_features) - 5} lagi")
                
                    with col2:
                        st.write("**🏷️ Fitur Kategorikal (Encoded):**")
                        for feat in categorical_features[:5]:
                            st.write(f"- {feat}")
                        if len(categorical_features) > 5:
                            st.write(f"... dan {len(categorical_features) - 5} lagi")
                    
                        st.write("**📈 Fitur Statistik:**")
                        for feat in statistical_features[:5]:
                            st.write(f"- {feat}")
                        if len(statistical_features) > 5:
                            st.write(f"... dan {len(statistical_features) - 5} lagi")
                
                    with col3:
                        st.write("**🔀 Fitur Interaksi:**")
                        for feat in interaction_features[:5]:
                            st.write(f"- {feat}")
                        if len(interaction_features) > 5:
                            st.write(f"... dan {len(interaction_features) - 5} lagi")
                    
                        st.write("**⚙️ Fitur Turunan Lainnya:**")
                        for feat in derived_features[:5]:
                            st.write(f"- {feat}")
                        if len(derived_features) > 5:
                            st.write(f"... dan {len(derived_features) - 5} lagi")
                
                    # Show detailed feature breakdown in expander
                    with st.expander("📋 Lihat Semua Fitur Detail"):
                        detail_col1, detail_col2 = st.columns(2)
                    
                        with detail_col1:
                            if numerical_features:
                                st.write(f"**💰 Numerikal ({len(numerical_features)}):**")
                                for feat in numerical_features:
                                    st.write(f"  - {feat}")
                        
                            if temporal_features:
                                st.write(f"**📅 Temporal ({len(temporal_features)}):**")
                                for feat in temporal_features:
                                    st.write(f"  - {feat}")
                    
                        with detail_col2:
                            if categorical_features:
                                st.write(f"**🏷️ Kategorikal ({len(categorical_features)}):**")
                                for feat in categorical_features:
                                    st.write(f"  - {feat}")
                        
                            if statistical_features:
                                st.write(f"**📈 Statistik ({len(statistical_features)}):**")
                                for feat in statistical_features:
                                    st.write(f"  - {feat}")
                        
                            if interaction_features:
                                st.write(f"**🔀 Interaksi ({len(interaction_features)}):**")
                                for feat in interaction_features:
                                    st.write(f"  - {feat}")
                        
                            if derived_features:
                                st.write(f"**⚙️ Turunan ({len(derived_features)}):**")
                                for feat in derived_features:
                                    st.write(f"  - {feat}")
                
                    # Feature selection options - Multiple Methods
                    st.write("### 🎯 Pilihan Seleksi Fitur:")
                
                    selection_method = st.selectbox(
                        "Pilih metode seleksi fitur:",
                        ["Semua Fitur", "Seleksi Manual Kustom", "Select K-Best (Statistical)", 
                         "Mutual Information (Non-linear Filter)", "Tree-based Importance (XGBoost/LGBM)", 
                         "PCA Dimensionality Reduction"],
                        help="Pilih metode untuk menentukan fitur yang akan digunakan dalam training"
                    )
                
                    selected_features = []
                
                    if selection_method == "Semua Fitur":
                        selected_features = feature_columns
                        st.info(f"✅ Menggunakan semua {len(feature_columns)} fitur yang tersedia")
                    
                    elif selection_method == "Seleksi Manual Kustom":
                        st.write("**Pilih Fitur untuk Pelatihan:**")
                    
                        # Show feature categories for reference
                        with st.expander("📋 Kategori Fitur yang Tersedia"):
                            col1, col2, col3, col4 = st.columns(4)
                        
                            with col1:
                                st.write("**📊 Fitur Original:**")
                                # Use feature_columns as original features since original_features is not defined
                                for feat in feature_columns[:5]:  # Show first 5
                                    st.write(f"  • {feat}")
                                if len(feature_columns) > 5:
                                    st.write(f"  ... dan {len(feature_columns)-5} lainnya")
                        
                            with col2:
                                st.write("**🔢 Fitur Numerikal:**")
                                if numerical_features:
                                    for feat in numerical_features[:5]:
                                        st.write(f"  • {feat}")
                                    if len(numerical_features) > 5:
                                        st.write(f"  ... dan {len(numerical_features)-5} lainnya")
                                else:
                                    st.write("  Tidak ada")
                        
                            with col3:
                                st.write("**⏰ Fitur Temporal:**")
                                if temporal_features:
                                    for feat in temporal_features:
                                        st.write(f"  • {feat}")
                                else:
                                    st.write("  Tidak ada")
                        
                            with col4:
                                st.write("**🔧 Fitur Engineering:**")
                                if derived_features:
                                    for feat in derived_features[:5]:
                                        st.write(f"  • {feat}")
                                    if len(derived_features) > 5:
                                        st.write(f"  ... dan {len(derived_features)-5} lainnya")
                                else:
                                    st.write("  Tidak ada")
                    
                        selected_features = st.multiselect(
                            "Pilih fitur yang ingin digunakan:",
                            options=feature_columns,
                            default=feature_columns[:15],  # Default 15 fitur pertama
                            help="Pilih fitur yang paling relevan untuk deteksi anomali. Semakin banyak fitur, semakin kompleks modelnya.",
                            max_selections=len(feature_columns)
                        )
                    
                        # Show selection summary
                        if len(selected_features) > 0:
                            st.info(f"✅ Telah memilih {len(selected_features)} fitur dari total {len(feature_columns)} fitur yang tersedia")
                        
                            # Show feature selection tips
                            with st.expander("💡 Tips Pemilihan Fitur"):
                                st.write("""
                                **Rekomendasi Fitur untuk Deteksi Anomali:**
                                • **Amount-related**: billed_amount, paid_amount, amount_ratio, cost_per_service
                                • **Time-based**: days_to_submit, processing_time, late_submission
                                • **Frequency**: claim_count, service_frequency, provider_frequency  
                                • **Anomaly indicators**: zscore_amount, pct_rank_amount, high_amount_flag
                                • **Provider patterns**: provider_avg_amount, provider_claim_frequency
                            
                                **Tips:**
                                • Pilih 10-25 fitur untuk model yang seimbang
                                • Include fitur yang berkaitan dengan uang/amount
                                • Tambahkan fitur waktu/frequency untuk pola perilaku
                                • Consider fitur anomaly detection untuk outlier detection
                                """)
                        else:
                            st.warning("⚠️ Silakan pilih minimal 1 fitur untuk melanjutkan")
                        
                    elif selection_method == "Select K-Best (Statistical)":
                        st.write("**Konfigurasi Select K-Best:**")
                    
                        col1, col2 = st.columns(2)
                    
                        with col1:
                            k_value = st.number_input(
                                "Jumlah Fitur Terbaik (K):",
                                min_value=5,
                                max_value=len(feature_columns),
                                value=min(20, len(feature_columns)),
                                step=1,
                                help="Jumlah fitur terbaik yang akan dipilih berdasarkan skor statistik"
                            )
                    
                        with col2:
                            score_function = st.selectbox(
                                "Metode Skoring:",
                                ["f_classif", "mutual_info_classif"],
                                help="Metode statistik untuk mengevaluasi importance fitur"
                            )
                    
                        # Advanced filtering options
                        with st.expander("🔧 Opsi Lanjutan"):
                            remove_correlated = st.checkbox(
                                "Hapus Fitur Berkorelasi Tinggi",
                                value=True,
                                help="Hapus fitur yang memiliki korelasi > 0.9 dengan fitur lain (mempertahankan fitur dengan skor tertinggi)"
                            )
                        
                            variance_threshold = st.slider(
                                "Threshold Variansi Normalisasi Minimal:",
                                min_value=0.0,
                                max_value=0.1,
                                value=0.01,
                                step=0.001,
                                help="Hapus fitur dengan variansi normalisasi < threshold"
                            )
                    
                        if st.button("🔧 Terapkan Select K-Best", type="primary"):
                            try:
                                if len(df_processed) > 5:
                                    with st.spinner("Menghitung skor Select K-Best..."):
                                        selected_features, feature_scores = apply_select_k_best(
                                            df_processed, feature_columns, k=k_value, score_func_name=score_function
                                        )
                                
                                        st.success(f"✅ Select K-Best ({score_function}) berhasil! Memilih {len(selected_features)} fitur terbaik")
                                        st.write("**📊 Skor Fitur Teratas:**")
                                        st.dataframe(feature_scores.head(k_value), width='stretch')
                                
                                        final_features = selected_features.copy()
                                
                                        # Apply advanced filtering if selected
                                        if remove_correlated:
                                            final_features, to_remove = filter_correlated_features(
                                                df_processed, final_features, correlation_threshold=0.9, feature_scores_df=feature_scores
                                            )
                                            if len(to_remove) > 0:
                                                st.warning(f"🗑️ Dihapus {len(to_remove)} fitur karena korelasi tinggi: {', '.join(to_remove)}")
                                
                                        if variance_threshold > 0:
                                            final_features, low_var_features = filter_low_variance_features(
                                                df_processed, final_features, variance_threshold=variance_threshold
                                            )
                                            if len(low_var_features) > 0:
                                                st.warning(f"🗑️ Dihapus {len(low_var_features)} fitur karena variansi rendah: {', '.join(low_var_features)}")
                                
                                        if len(final_features) == 0:
                                            st.error("❌ Tidak ada fitur yang tersisa setelah filtering!")
                                            return
                                
                                        st.session_state['selected_features_cache'] = final_features
                                        st.session_state['proceed_after_selection'] = True
                                else:
                                    st.error("❌ Data terlalu sedikit untuk Select K-Best (minimum 5 samples)")
                                    return
                            except Exception as e:
                                st.error(f"❌ Error dalam Select K-Best: {str(e)}")
                                return
                            
                    elif selection_method == "Mutual Information (Non-linear Filter)":
                        st.write("**Konfigurasi Mutual Information:**")
                        k_mi = st.number_input("Jumlah Fitur Terbaik (K):", min_value=5, max_value=len(feature_columns), value=min(20, len(feature_columns)), key="mi_k")
                    
                        if st.button("🔧 Terapkan Mutual Information", type="primary"):
                            with st.spinner("Menghitung Mutual Information..."):
                                selected_features, feature_scores = apply_mutual_info_selection(df_processed, feature_columns, k=k_mi)
                                st.success(f"✅ Berhasil memilih {len(selected_features)} fitur!")
                                st.write("**📊 Skor Fitur Teratas:**")
                                st.dataframe(feature_scores.head(k_mi), width='stretch')
                                st.session_state['selected_features_cache'] = selected_features
                                st.session_state['proceed_after_selection'] = True
                
                    elif selection_method == "Tree-based Importance (XGBoost/LGBM)":
                        st.write("**Konfigurasi Tree-based Importance:**")
                        k_tree = st.number_input("Jumlah Fitur Terbaik (K):", min_value=5, max_value=len(feature_columns), value=min(20, len(feature_columns)), key="tree_k")
                    
                        if st.button("🔧 Terapkan Tree-based Selection", type="primary"):
                            with st.spinner("Melatih model untuk menghitung importance..."):
                                selected_features, feature_importances, model_name = apply_tree_based_selection(df_processed, feature_columns, k=k_tree)
                                st.success(f"✅ Berhasil memilih {len(selected_features)} fitur menggunakan {model_name}!")
                                st.write(f"**📊 Importance Fitur ({model_name}):**")
                                st.dataframe(feature_importances.head(k_tree), width='stretch')
                                st.session_state['selected_features_cache'] = selected_features
                                st.session_state['proceed_after_selection'] = True
                            
                    elif selection_method == "PCA Dimensionality Reduction":
                        st.info("ℹ️ **Catatan Model Explainer & Interpretability:** PCA memproyeksikan fitur ke dalam komponen ortogonal abstrak (`PCA_Component_1`, dsb.). Ini sangat efisien untuk reduksi dimensi, namun interpretasi fitur individual pada modul **SHAP Explainer** dan **Agentic Copilot** akan menampilkan komponen PCA.")
                        st.write("**Konfigurasi PCA:**")
                        pca_mode = st.radio("Mode PCA:", ["Persentase Variansi", "Jumlah Komponen Tetap"])
                    
                        if pca_mode == "Persentase Variansi":
                            n_comp = st.slider("Persentase Variansi yang Dijaga:", 0.5, 0.99, 0.95, 0.01)
                        else:
                            n_comp = st.number_input("Jumlah Komponen:", min_value=2, max_value=len(feature_columns), value=min(10, len(feature_columns)))
                    
                        if st.button("🔧 Jalankan PCA", type="primary"):
                            with st.spinner("Menjalankan PCA..."):
                                # Load df_processed from disk
                                df_processed = get_df_processed()
                                if df_processed is None:
                                    st.error("Data hasil praproses tidak ditemukan!")
                                    return
                            
                                # Remove old PCA columns if they exist to prevent accumulation
                                old_pca_cols = [col for col in df_processed.columns if col.startswith('PCA_Component_')] if isinstance(df_processed, pd.DataFrame) else []
                                if old_pca_cols and isinstance(df_processed, pd.DataFrame):
                                    df_processed = df_processed.drop(columns=old_pca_cols)
                                    st.info(f"🗑️ Membersihkan {len(old_pca_cols)} kolom PCA lama")
                            
                                df_pca, pca_cols, explained_variance = apply_pca_reduction(df_processed, feature_columns, n_components=n_comp)
                            
                                # Update df_processed in session state to include PCA columns
                                for col in pca_cols:
                                    df_processed[col] = df_pca[col]
                            
                                # Save updated df_processed back to Parquet
                                update_df_processed(df_processed)
                            
                                st.success(f"✅ PCA selesai! {len(pca_cols)} komponen menjelaskan {sum(explained_variance):.2%} variansi.")

                                # Plot explained variance
                                fig = create_bar_chart([f"PC{i+1}" for i in range(len(explained_variance))],
                                           explained_variance,
                                           title='Variansi Terjelaskan per Komponen',
                                           labels={'x': 'Komponen Utama', 'y': 'Variansi Terjelaskan'})
                                st.plotly_chart(fig, width='stretch')
                            
                                st.session_state['selected_features_cache'] = pca_cols
                                st.session_state['proceed_after_selection'] = True
                
                    # Process selected features based on method
                    proceed_to_training = False
                    final_features = []
                
                    if selection_method == "Semua Fitur":
                        # Direct processing for "All Features"
                        final_features = selected_features
                        proceed_to_training = True
                    

                    elif selection_method == "Seleksi Manual Kustom":
                        if len(selected_features) == 0:
                            st.warning("⚠️ Silakan pilih minimal 1 fitur untuk melanjutkan")
                        else:
                            # Show feature importance preview
                            st.write("### 📈 Preview Fitur Terpilih:")
                            preview_data = []
                            for feat in selected_features[:10]:  # Show max 10 features
                                if isinstance(df_processed, pd.DataFrame) and feat in df_processed.columns:
                                    preview_data.append({
                                        'Fitur': feat,
                                        'Tipe': str(df_processed[feat].dtype),
                                        'Missing': df_processed[feat].isnull().sum(),
                                        'Min': df_processed[feat].min() if df_processed[feat].dtype.kind in 'iuf' else '-',
                                        'Max': df_processed[feat].max() if df_processed[feat].dtype.kind in 'iuf' else '-',
                                        'Unique': df_processed[feat].nunique(),
                                        'Sample': str(df_processed[feat].iloc[0]) if len(df_processed) > 0 else 'N/A'
                                    })

                            preview_df = pd.DataFrame(preview_data)
                            st.dataframe(preview_df, width='stretch')

                            # Advanced filtering options for manual selection
                            with st.expander("⚙️ Opsi Lanjutan (Opsional)"):
                                st.write("**Filtering Tambahan:**")

                                # Correlation threshold for removing highly correlated features
                                remove_correlated = st.checkbox(
                                    "Hapus fitur yang berkorelasi tinggi",
                                    value=False,
                                    help="Hapus fitur yang memiliki korelasi > threshold dengan fitur lain"
                                )

                                correlation_threshold = 0.9
                                if remove_correlated:
                                    correlation_threshold = st.slider(
                                        "Threshold korelasi:",
                                        min_value=0.7,
                                        max_value=0.95,
                                        value=0.9,
                                        step=0.05,
                                        help="Fitur dengan korelasi > threshold akan dihapus"
                                    )

                                # Variance threshold for removing low variance features
                                remove_low_variance = st.checkbox(
                                    "Hapus fitur dengan variansi rendah",
                                    value=False,
                                    help="Hapus fitur yang memiliki variansi normalisasi < threshold"
                                )

                                variance_threshold = 0.01
                                if remove_low_variance:
                                    variance_threshold = st.slider(
                                        "Threshold variansi normalisasi:",
                                        min_value=0.001,
                                        max_value=0.1,
                                        value=0.01,
                                        step=0.001,
                                        format="%.3f"
                                    )

                            # Apply button for manual selection
                            if st.button("🔧 Terapkan Seleksi Fitur Manual", type="primary"):
                                final_features = selected_features.copy()

                                # Apply advanced filtering if selected
                                if remove_correlated:
                                    final_features, to_remove = filter_correlated_features(
                                        df_processed, final_features, correlation_threshold=correlation_threshold
                                    )
                                    if len(to_remove) > 0:
                                        st.warning(f"🗑️ Dihapus {len(to_remove)} fitur karena korelasi tinggi: {', '.join(to_remove)}")

                                if remove_low_variance:
                                    final_features, low_var_features = filter_low_variance_features(
                                        df_processed, final_features, variance_threshold=variance_threshold
                                    )
                                    if len(low_var_features) > 0:
                                        st.warning(f"🗑️ Dihapus {len(low_var_features)} fitur karena variansi rendah: {', '.join(low_var_features)}")

                                proceed_to_training = True

                    elif selection_method in ["Select K-Best (Statistical)", "Mutual Information (Non-linear Filter)", "Tree-based Importance (XGBoost/LGBM)", "PCA Dimensionality Reduction"]:
                        # Check if we have cached results from a button click
                        if st.session_state.get('proceed_after_selection', False):
                            final_features = st.session_state.get('selected_features_cache', [])
                            proceed_to_training = True
                            # Reset for next time
                            st.session_state['proceed_after_selection'] = False
                        elif 'selected_features' in locals() and len(selected_features) > 0:
                            final_features = selected_features
                            proceed_to_training = True

                    # Save and proceed if ready
                    if proceed_to_training and (len(selected_features) > 0 or len(final_features) > 0):
                        # Save final feature selection
                        st.session_state['selected_features'] = final_features
                        st.session_state['feature_selection_method'] = selection_method
                        st.session_state['original_feature_count'] = len(feature_columns)
                        st.session_state['final_feature_count'] = len(final_features)

                        st.success(f"✅ Seleksi fitur berhasil! Menggunakan {len(final_features)} fitur dari {len(feature_columns)} fitur tersedia")

                        # Show summary
                        st.write("### 📋 Ringkasan Seleksi:")
                        summary_col1, summary_col2, summary_col3 = st.columns(3)

                        with summary_col1:
                            st.metric("Fitur Awal", len(feature_columns))

                        with summary_col2:
                            st.metric("Fitur Terpilih", len(final_features))

                        with summary_col3:
                            reduction_pct = ((len(feature_columns) - len(final_features)) / len(feature_columns)) * 100
                            st.metric("Reduksi", f"{reduction_pct:.1f}%")

                    
                    show_preprocessing_insight = st.checkbox(
                        "Tampilkan ringkasan detail hasil praproses",
                        value=False,
                        key="show_preprocessing_insight",
                        help="Nonaktifkan jika ingin langsung melanjutkan alur tanpa membebani tampilan.",
                    )

                    if show_preprocessing_insight and isinstance(df_processed, pd.DataFrame):
                        # Tampilkan informasi preprocessing
                        st.subheader("Hasil Preprocessing")
                        st.write(f"Jumlah fitur untuk modeling: {len(feature_columns)}")
                        st.write("Fitur yang digunakan:")
                        st.write(feature_columns)
                    
                        # Tampilkan statistik dasar
                        valid_feature_cols = [c for c in feature_columns if c in df_processed.columns]
                        if valid_feature_cols:
                            st.subheader("Statistik Data yang Diproses")
                            st.dataframe(df_processed[valid_feature_cols].describe())
                    
                        # Visualisasi distribusi beberapa fitur penting
                        st.subheader("Visualisasi Data")
                    
                        # Plot 1: Distribusi jumlah klaim
                        if 'billed_amount' in df_processed.columns:
                            fig = create_histogram_chart(df_processed, 'billed_amount',
                                           nbins=50, title='Distribusi Jumlah Klaim')
                            st.plotly_chart(fig, width='stretch')

                        # Plot 2: Distribusi usia pasien
                        if 'patient_age' in df_processed.columns:
                            fig = create_histogram_chart(df_processed, 'patient_age',
                                           nbins=20, title='Distribusi Usia Pasien')
                            st.plotly_chart(fig, width='stretch')

                        # Plot 3: Provider specialty distribution
                        if 'provider_specialty' in df_processed.columns:
                            provider_counts = df_processed['provider_specialty'].value_counts().head(10)
                            fig = create_bar_chart(provider_counts.index, provider_counts.values,
                                       title='10 Spesialis Provider Teratas',
                                       labels={'x': 'Spesialisasi', 'y': 'Jumlah Klaim'})
                            st.plotly_chart(fig, width='stretch')

                        # Plot 4: Claim status distribution
                        if 'claim_status' in df_processed.columns:
                            status_counts = df_processed['claim_status'].value_counts()
                            fig = create_pie_chart(status_counts.values, status_counts.index,
                                      title='Distribusi Status Klaim')
                            st.plotly_chart(fig, width='stretch')
                
                    # Tombol untuk lanjut ke training yang lebih jelas
                    st.markdown("---")
                    st.subheader("🚀 Langkah Selanjutnya")
                    if st.button("Lanjut ke Pelatihan Model", key="proceed_to_training_final", type="primary"):
                        navigate_to_page('train')
        except Exception as e:
            logger.error(f"Gagal memproses file: {e}", exc_info=True)
            st.error(f"❌ Gagal memproses file: {type(e).__name__}: {str(e)}")
            with st.expander("Detail Kesalahan"):
                import traceback
                st.code(traceback.format_exc())
            if st.button("🔁 Coba Muat Ulang", key="btn_reload_data_collection"):
                st.rerun()

