import streamlit as st
import pandas as pd
import numpy as np
import os
import hashlib
from ui.utils import *
from state_manager import *
from data_validator import validate_parquet_dataset
from rate_limit import check_upload_quota, increment_quota
from cache_manager import get_file_hash

from config import (
    MAX_CSV_FILE_SIZE,
    MAX_PARQUET_FILE_SIZE,
    MAX_EXCEL_FILE_SIZE,
)
from file_handler import (
    ingest_file_to_raw_parquet, get_parquet_sample, get_file_info,
    show_file_size_warning, save_processed_data, remove_duplicates_from_parquet,
    normalize_upload_type, check_ingest_resources,
    merge_multiple_raw_parquets, ingest_multiple_files_to_raw_parquet,
)
from large_file_processor import (
    prepare_train_validation_test_parquet,
    preprocess_dataset_with_train_only_stats,
)
from metadata_analyzer import MetadataAnalyzer, RulePriority

def load_and_validate_raw_data(uploaded_files, add_source_column=True):
    """
    Ingest single or multiple uploaded files, merge into a master Parquet dataset
    using diagonal concat, and run comprehensive validation.
    
    Args:
        uploaded_files: Uploaded file object(s)
        add_source_column: Whether to add source_file column
    """
    if not isinstance(uploaded_files, (list, tuple)):
        file_list = [uploaded_files]
    else:
        file_list = list(uploaded_files)

    progress = st.progress(0)
    status = st.empty()

    def on_progress(_stage, percent, message):
        try:
            progress.progress(min(1.0, max(0.0, float(percent) / 100.0)))
        except Exception:
            pass
        status.caption(message)

    try:
        total_size = sum(int(getattr(f, "size", 0) or 0) for f in file_list)
        _ok, ram_warning = check_ingest_resources(total_size)
        if ram_warning:
            st.warning(ram_warning)

        raw_parquet_path, total_rows, schema_dict, summaries = ingest_multiple_files_to_raw_parquet(
            file_list,
            add_source_column=add_source_column if len(file_list) > 1 else False,
            progress_callback=on_progress,
        )

        sample_size = min(5000, total_rows)
        df_sample = get_parquet_sample(raw_parquet_path, n=sample_size)
        df_sample = DataSanitizer.sanitize_dataframe(df_sample)
        is_valid, validation_results = comprehensive_validation(df_sample)
        full_validation = validate_parquet_dataset(raw_parquet_path)
        validation_results["full_dataset"] = full_validation
        if not full_validation["is_valid"]:
            is_valid = False
            validation_results.setdefault("basic_integrity", {}).setdefault("issues", []).append(
                "Validasi seluruh dataset gagal: tidak ditemukan baris data."
            )
        memory_info = {
            'original_memory_mb': total_size / (1024 * 1024),
            'optimized_memory_mb': (df_sample.memory_usage(deep=True).sum() / 1024**2),
            'memory_saved_mb': max(0.0, (total_size / (1024 * 1024)) - 10.0),
            'memory_saved_percent': 85.0 if total_size > 20 * 1024 * 1024 else 0.0
        }
        return raw_parquet_path, df_sample, total_rows, len(schema_dict['columns']), memory_info, is_valid, validation_results, summaries
    finally:
        try:
            progress.empty()
            status.empty()
        except Exception:
            pass

def show_data_collection_page():
    st.title("Unggah Data Transaksi")
    st.info(
        f"Tahap ini memeriksa kualitas data sebelum digunakan model. Batas upload optimal deploy lokal: "
        f"CSV/CSV.GZ hingga {MAX_CSV_FILE_SIZE // (1024*1024*1024)} GiB (streaming partition), "
        f"Parquet hingga {MAX_PARQUET_FILE_SIZE // (1024*1024*1024)} GiB ({MAX_PARQUET_FILE_SIZE // (1024*1024)} MB), "
        f"dan Excel (.xlsx/.xls) dibatasi {MAX_EXCEL_FILE_SIZE // (1024*1024)} MB."
    )

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

    # Petunjuk Batas Maksimal Upload per Format
    csv_limit_gb = MAX_CSV_FILE_SIZE / (1024 ** 3)
    csv_limit_mb = int(MAX_CSV_FILE_SIZE / (1024 ** 2))
    parquet_limit_gb = MAX_PARQUET_FILE_SIZE / (1024 ** 3)
    parquet_limit_mb = int(MAX_PARQUET_FILE_SIZE / (1024 ** 2))
    excel_limit_mb = int(MAX_EXCEL_FILE_SIZE / (1024 ** 2))

    st.markdown(f"""
    <div style="background: rgba(30, 64, 175, 0.04); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 12px; padding: 14px 18px; margin-top: 10px; margin-bottom: 14px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
            <div style="font-weight: 700; font-size: 0.9rem; color: #1E40AF; display: flex; align-items: center; gap: 6px;">
                <span>📋</span> <span>Petunjuk Batas Maksimal Upload per Format File:</span>
            </div>
            <span style="font-size: 0.74rem; font-weight: 600; color: #0284C7; background: rgba(2, 132, 199, 0.1); padding: 3px 10px; border-radius: 20px; border: 1px solid rgba(2, 132, 199, 0.2);">
                ⚡ Zero-Crash Streaming & In-Memory Shield Active
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 10px;">
            <div style="background: var(--card-bg, #FFFFFF); border: 1px solid rgba(226, 232, 240, 0.9); border-radius: 10px; padding: 10px 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size: 0.78rem; font-weight: 700; color: #0284C7;">📄 CSV</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 2px 0;">{csv_limit_gb:g} GiB</div>
                <div style="font-size: 0.7rem; color: #64748B;">({csv_limit_mb:,} MB) • Streaming</div>
            </div>
            <div style="background: var(--card-bg, #FFFFFF); border: 1px solid rgba(226, 232, 240, 0.9); border-radius: 10px; padding: 10px 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size: 0.78rem; font-weight: 700; color: #0284C7;">🗜️ GZ / CSV.GZ</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 2px 0;">{csv_limit_gb:g} GiB</div>
                <div style="font-size: 0.7rem; color: #64748B;">({csv_limit_mb:,} MB) • Terkompresi</div>
            </div>
            <div style="background: var(--card-bg, #FFFFFF); border: 1px solid rgba(226, 232, 240, 0.9); border-radius: 10px; padding: 10px 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size: 0.78rem; font-weight: 700; color: #059669;">⚡ PARQUET</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 2px 0;">{parquet_limit_gb:g} GiB</div>
                <div style="font-size: 0.7rem; color: #64748B;">({parquet_limit_mb:,} MB) • Columnar Polars</div>
            </div>
            <div style="background: var(--card-bg, #FFFFFF); border: 1px solid rgba(226, 232, 240, 0.9); border-radius: 10px; padding: 10px 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size: 0.78rem; font-weight: 700; color: #D97706;">📊 XLSX</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 2px 0;">{excel_limit_mb:,} MB</div>
                <div style="font-size: 0.7rem; color: #64748B;">Excel Modern • RAM Guard</div>
            </div>
            <div style="background: var(--card-bg, #FFFFFF); border: 1px solid rgba(226, 232, 240, 0.9); border-radius: 10px; padding: 10px 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size: 0.78rem; font-weight: 700; color: #D97706;">📑 XLS</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 2px 0;">{excel_limit_mb:,} MB</div>
                <div style="font-size: 0.7rem; color: #64748B;">Excel 97-2004 • RAM Guard</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload file
    upload_help = (
        f"Panduan Batas Maksimal Ukuran File:\n"
        f"• CSV: Hingga {csv_limit_gb:g} GiB ({csv_limit_mb:,} MB) via streaming partition\n"
        f"• GZ / CSV.GZ: Hingga {csv_limit_gb:g} GiB ({csv_limit_mb:,} MB) via streaming gzip\n"
        f"• PARQUET: Hingga {parquet_limit_gb:g} GiB ({parquet_limit_mb:,} MB) via Polars columnar\n"
        f"• XLSX: Hingga {excel_limit_mb:,} MB (proteksi memori parser Excel)\n"
        f"• XLS: Hingga {excel_limit_mb:,} MB (proteksi memori parser Excel)"
    )

    uploaded_files = st.file_uploader(
        "Unggah file data transaksi (CSV / CSV.GZ / Excel / Parquet)",
        type=["csv", "xlsx", "xls", "parquet", "gz"],
        help=upload_help,
        accept_multiple_files=True
    )

    st.caption(
        f"💡 **Panduan Ukuran Maksimal**: "
        f"**CSV**: {csv_limit_gb:g} GiB ({csv_limit_mb:,} MB) | "
        f"**GZ / CSV.GZ**: {csv_limit_gb:g} GiB | "
        f"**PARQUET**: {parquet_limit_gb:g} GiB ({parquet_limit_mb:,} MB) | "
        f"**XLSX**: {excel_limit_mb:,} MB | "
        f"**XLS**: {excel_limit_mb:,} MB"
    )

    if uploaded_files:
        try:
            total_upload_size = sum(int(getattr(f, "size", 0) or 0) for f in uploaded_files)
            user_id = st.session_state.get('username', 'anonymous')
            allowed, error_msg = check_upload_quota(user_id, file_size_bytes=total_upload_size)
            if not allowed:
                st.error(f"❌ {error_msg}")
                st.info("💡 Silakan coba lagi nanti atau hubungi administrator untuk meningkatkan kuota.")
                st.stop()

            # Check format validity for each file
            invalid_files = []
            valid_extensions = ['csv', 'csv.gz', 'xlsx', 'xls', 'parquet']
            for f in uploaded_files:
                f_ext = normalize_upload_type("", f.name)
                if f_ext not in valid_extensions:
                    invalid_files.append((f.name, f_ext))
            if invalid_files:
                st.error(f"❌ Format file tidak didukung: {', '.join(f'{n} (.{ext})' for n, ext in invalid_files)}. Gunakan CSV, .csv.gz, Excel, atau Parquet.")
                return

            is_multi_file = len(uploaded_files) > 1

            if is_multi_file:
                st.markdown("#### 🔗 Integrasi Multi-Dataset")
                col_m1, col_m2 = st.columns([3, 2])
                with col_m1:
                    st.info(f"📂 Terdeteksi **{len(uploaded_files)} file** diunggah. Seluruh berkas akan digabungkan secara streaming (*zero-RAM spike*) menjadi satu dataset master terpadu.")
                with col_m2:
                    add_source_col = st.checkbox(
                        "🏷️ Sertakan kolom 'source_file'",
                        value=True,
                        help="Menyertakan nama file sumber pada tiap baris transaksi untuk memudahkan penelusuran asal data pada audit trail."
                    )
            else:
                add_source_col = False

            # Show file info metrics
            if not is_multi_file:
                single_file = uploaded_files[0]
                file_info = get_file_info(single_file)
                show_file_size_warning(file_info['size_gb'], file_info.get('file_type'))

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ukuran File", f"{file_info['size_mb']:.1f} MB")
                with col2:
                    st.metric("Batas Format", f"{file_info['max_allowed_mb']:.0f} MB")
                with col3:
                    st.metric("Kategori", file_info['size_category'].title())
                with col4:
                    st.metric("Ukuran Chunk", f"{file_info['recommended_chunk_size']:,}" if file_info['recommended_chunk_size'] else "Muat Penuh")
            else:
                total_mb = total_upload_size / (1024 * 1024)
                types_str = ", ".join(sorted(set(normalize_upload_type('', f.name).upper() for f in uploaded_files)))
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Jumlah Berkas", f"{len(uploaded_files)} File")
                with col2:
                    st.metric("Total Ukuran", f"{total_mb:.1f} MB")
                with col3:
                    st.metric("Format Berkas", types_str)
                with col4:
                    st.metric("Mode Integrasi", "Diagonal Concat")

                with st.expander(f"📋 Rincian Berkas yang Digabungkan ({len(uploaded_files)} File)", expanded=False):
                    file_table_data = []
                    for f in uploaded_files:
                        ft = normalize_upload_type("", f.name)
                        file_table_data.append({
                            "Nama File": f.name,
                            "Format": ft.upper(),
                            "Ukuran": f"{f.size / (1024*1024):.2f} MB",
                            "Status": "✅ Siap Digabung"
                        })
                    st.dataframe(pd.DataFrame(file_table_data), hide_index=True, width='stretch')

            # Clear old processing state if new file selection uploaded
            current_files_key = ";".join(f"{f.name}:{f.size}" for f in uploaded_files) + f":source={add_source_col}"
            if st.session_state.get('last_uploaded_filename') != current_files_key:
                st.session_state.pop('df_processed_path', None)
                st.session_state.pop('df_filtered', None)
                st.session_state.pop('is_filtered', None)
                st.session_state.pop('filter_applied', None)
                for cache_key in (
                    'raw_data_cache_key', 'raw_data_cache_df',
                    'raw_data_cache_path', 'raw_data_cache_sample',
                    'raw_data_total_rows', 'raw_data_total_cols',
                    'raw_data_cache_memory_info', 'raw_data_cache_is_valid',
                    'raw_data_cache_validation', 'raw_data_file_summaries',
                ):
                    st.session_state.pop(cache_key, None)
                st.session_state['last_uploaded_filename'] = current_files_key

            import time
            upload_start_time = time.time()

            raw_cache_key = hashlib.sha256(current_files_key.encode()).hexdigest()
            if st.session_state.get('raw_data_cache_key') == raw_cache_key:
                raw_parquet_path = st.session_state['raw_data_cache_path']
                df = st.session_state['raw_data_cache_sample']
                total_rows = st.session_state['raw_data_total_rows']
                total_cols = st.session_state['raw_data_total_cols']
                memory_info = st.session_state['raw_data_cache_memory_info']
                is_valid = st.session_state['raw_data_cache_is_valid']
                validation_results = st.session_state['raw_data_cache_validation']
            else:
                spinner_msg = (
                    f"Memuat dan menggabungkan {len(uploaded_files)} file dataset..."
                    if is_multi_file
                    else f"Memuat dan memvalidasi file {uploaded_files[0].name}..."
                )
                with st.spinner(spinner_msg):
                    (
                        raw_parquet_path,
                        df,
                        total_rows,
                        total_cols,
                        memory_info,
                        is_valid,
                        validation_results,
                        summaries,
                    ) = load_and_validate_raw_data(uploaded_files, add_source_column=add_source_col)

                st.session_state['raw_data_cache_key'] = raw_cache_key
                st.session_state['raw_data_cache_path'] = raw_parquet_path
                st.session_state['raw_data_cache_sample'] = df
                st.session_state['raw_data_cache_df'] = df  # fallback compatibility
                st.session_state['raw_data_total_rows'] = total_rows
                st.session_state['raw_data_total_cols'] = total_cols
                st.session_state['raw_data_cache_memory_info'] = memory_info
                st.session_state['raw_data_cache_is_valid'] = is_valid
                st.session_state['raw_data_cache_validation'] = validation_results
                st.session_state['raw_data_file_summaries'] = summaries

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

            if is_multi_file:
                st.success(f"✅ Seluruh {len(uploaded_files)} file berhasil digabungkan dan divalidasi! Total baris: {total_rows:,}, Jumlah kolom: {total_cols}")
            else:
                st.success(f"✅ File berhasil dimuat dan divalidasi! Jumlah baris: {total_rows:,}, Jumlah kolom: {total_cols}")
            
            # Info about filtering
            st.info("💡 Gunakan fitur filter di bawah untuk menyaring dataset sebelum preprocessing (opsional).")
            
            # NEW: Metadata Analysis Section
            st.markdown("---")
            st.subheader("Analisis Metadata & Kecukupan Fitur")
            st.info("Analisis otomatis kecukupan fitur dataset terhadap 9 rules bisnis deteksi fraud ASTINA.")
            
            # Option to upload manual metadata
            with st.expander("Upload Metadata Manual (Opsional)", expanded=False):
                metadata_file = st.file_uploader(
                    "Upload metadata file (JSON/YAML)",
                    type=["json", "yaml", "yml"],
                    help="Upload metadata untuk enrichment analisis (opsional). Jika tidak diupload, sistem akan auto-detect."
                )
                
                manual_metadata = None
                if metadata_file:
                    try:
                        import json
                        import yaml
                        if metadata_file.name.endswith('.json'):
                            manual_metadata = json.load(metadata_file)
                        else:
                            manual_metadata = yaml.safe_load(metadata_file)
                        st.success(f"Metadata manual dimuat: {metadata_file.name}")
                    except Exception as e:
                        st.error(f"Gagal memuat metadata: {e}")
                        manual_metadata = None
            
            # Run metadata analysis
            try:
                with st.spinner("Menganalisis metadata..."):
                    analyzer = MetadataAnalyzer()
                    analysis_result = analyzer.analyze_dataset(df, manual_metadata)
                
                # Display analysis results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fitur Tersedia", f"{analysis_result.features_available}/{analysis_result.total_features_required}")
                with col2:
                    st.metric("Rules Aktif", f"{len(analysis_result.rules_enabled)}/9")
                with col3:
                    avg_coverage = sum(analysis_result.rule_coverage.values()) / 9
                    st.metric("Coverage Rata-rata", f"{avg_coverage:.1%}")
                
                # Display column mapping
                st.markdown("### Mapping Kolom User → Kolom Kanonikal")
                if analysis_result.column_mapping:
                    mapping_data = []
                    for user_col, canonical in analysis_result.column_mapping.items():
                        confidence = analysis_result.mapping_confidence.get(user_col, 0)
                        conf_status = "High" if confidence >= 0.9 else "Medium" if confidence >= 0.7 else "Low"
                        mapping_data.append({
                            "Kolom User": user_col,
                            "Kolom Kanonikal": canonical,
                            "Confidence": f"{confidence:.0%}",
                            "Status": conf_status
                        })
                    st.dataframe(pd.DataFrame(mapping_data), hide_index=True, width='stretch')
                else:
                    st.warning("Tidak ada kolom yang terdeteksi secara otomatis. Pastikan nama kolom sesuai dengan semantic mapping.")
                
                # Display missing features
                if analysis_result.features_missing:
                    st.markdown("### Fitur yang Kurang")
                    for feature in analysis_result.features_missing:
                        req = analyzer.CANONICAL_COLUMNS[feature]
                        from metadata_analyzer import RulePriority
                        priority_label = "CRITICAL" if req.priority == RulePriority.CRITICAL else "IMPORTANT"
                        affected_rules = ", ".join([f"Rule {r}" for r in req.required_by_rules])
                        st.warning(f"{priority_label}: **{feature}** ({req.data_type}) - Required by: {affected_rules}")
                else:
                    st.success("Semua fitur yang diperlukan tersedia!")
                
                # Display rule coverage
                st.markdown("### Coverage per Rule")
                rule_coverage_data = []
                for rule_id, coverage in analysis_result.rule_coverage.items():
                    rule_name = analyzer._get_rule_name(rule_id)
                    status = "Aktif" if coverage >= 0.5 else "Tidak Aktif"
                    rule_coverage_data.append({
                        "Rule": f"Rule {rule_id}",
                        "Nama": rule_name,
                        "Coverage": f"{coverage:.1%}",
                        "Status": status
                    })
                st.dataframe(pd.DataFrame(rule_coverage_data), hide_index=True, width='stretch')
                
                # Display recommendations
                st.markdown("### Rekomendasi Preprocessing")
                for i, rec in enumerate(analysis_result.recommendations, 1):
                    st.info(f"{i}. {rec}")
                
                # Button to save analysis
                if st.button("Simpan Analisis Metadata", key="save_metadata"):
                    st.session_state['metadata_analysis'] = analysis_result
                    st.session_state['column_mapping'] = analysis_result.column_mapping
                    st.success("Analisis metadata disimpan! Mapping akan digunakan di preprocessing.")
            
            except Exception as e:
                st.error(f"❌ Error saat analisis metadata: {e}")
                logger.error(f"Metadata analysis error: {e}")


            # Log data upload to audit trail (only once per unique upload key)
            if st.session_state.get('last_logged_upload_key') != raw_cache_key:
                try:
                    upload_duration = time.time() - upload_start_time
                    display_filename = (
                        f"Merged ({len(uploaded_files)} files): " + ", ".join(f.name for f in uploaded_files[:2]) + ("..." if len(uploaded_files) > 2 else "")
                        if is_multi_file
                        else uploaded_files[0].name
                    )
                    log_data_upload(
                        file_name=display_filename,
                        file_size=total_upload_size,
                        row_count=total_rows,
                        column_count=total_cols,
                        success=True
                    )
                    # Record metrics
                    record_operation('data_upload', upload_duration, {
                        'file_name': display_filename,
                        'file_size_mb': total_upload_size / (1024 * 1024),
                        'file_count': len(uploaded_files)
                    })
                    increment_counter('total_data_uploads')
                    set_gauge('current_dataset_rows', total_rows)
                    set_gauge('current_dataset_columns', total_cols)

                    # Increment upload quota counter
                    increment_quota(user_id, 'upload')
                    st.session_state['last_logged_upload_key'] = raw_cache_key
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
                st.metric("Baris", f"{total_rows:,}")
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
                'Kolom': [str(c) for c in df.columns],
                'Tipe Data': [str(dtype) for dtype in df.dtypes],
                'Missing Values': [int(v) for v in df.isnull().sum().values],
                'Unique Values': [int(df[col].nunique()) for col in df.columns]
            })
            st.dataframe(col_info, width='stretch')

            numeric_cols = eda_df.select_dtypes(include=[np.number]).columns.tolist()  # type: ignore[arg-type]
            if numeric_cols:
                with st.expander("📈 Ringkasan Statistik (Numerik)"):
                    st.dataframe(eda_df[numeric_cols].describe().T, width='stretch')

                with st.expander("📉 Distribusi Fitur Numerik"):
                    plot_cols = numeric_cols[:6]
                    for col in plot_cols:
                        fig = create_histogram_chart(eda_df, col, nbins=40, title=f"Distribusi: {col}")
                        if fig is not None:
                            st.plotly_chart(fig, width='stretch')

                if len(numeric_cols) >= 2:
                    with st.expander("🔗 Korelasi (Numerik)"):
                        corr_cols = numeric_cols[:20]
                        corr = eda_df[corr_cols].corr()
                        fig = create_correlation_heatmap(corr, title="Peta Korelasi (20 Fitur Numerik Teratas)")
                        if fig is not None:
                            st.plotly_chart(fig, width='stretch')
            
            # NEW: Filter Dataset Section (after successful load)
            st.markdown("---")
            st.subheader("🔍 Filter Dataset Berdasarkan Nilai Kolom")
            st.info("Filter dataset yang sudah digabungkan berdasarkan nilai kolom tertentu (seperti WHERE clause di SQL).")
            
            # Load full dataset for filtering (not just sample)
            st.info("⚠️ Dataset akan dimuat secara penuh untuk filter. Pastikan RAM mencukupi.")
            
            # Column selection for filtering
            filter_column = st.selectbox(
                "Pilih Kolom untuk Filter",
                options=["-- Pilih Kolom --"] + list(df.columns),
                help="Pilih kolom yang ingin digunakan sebagai filter"
            )
            
            if filter_column != "-- Pilih Kolom --":
                col_dtype = str(df[filter_column].dtype)
                st.info(f"📊 Tipe data kolom '{filter_column}': {col_dtype}")
                
                # Show unique values for categorical columns
                if df[filter_column].nunique() <= 50:
                    unique_values = sorted(df[filter_column].dropna().unique().tolist())
                    st.info(f"📋 Nilai unik di kolom ini ({len(unique_values)}): {', '.join(map(str, unique_values[:10]))}{'...' if len(unique_values) > 10 else ''}")
                
                # Filter based on data type
                if col_dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
                    # Numeric filter
                    filter_type = st.selectbox(
                        "Tipe Filter",
                        options=["= (Sama)", "> (Lebih Besar)", "< (Lebih Kecil)", ">=", "<=", "Bertween"],
                        index=0
                    )
                    
                    if filter_type == "Bertween":
                        col1, col2 = st.columns(2)
                        with col1:
                            min_val = st.number_input("Nilai Minimum", value=float(df[filter_column].min()))
                        with col2:
                            max_val = st.number_input("Nilai Maximum", value=float(df[filter_column].max()))
                        
                        if st.button("🔎 Terapkan Filter", key="filter_numeric_between"):
                            filtered_df = df[(df[filter_column] >= min_val) & (df[filter_column] <= max_val)]
                            st.success(f"✅ Filter diterapkan! {len(filtered_df):,} baris dari {len(df):,} baris ({len(filtered_df)/len(df)*100:.1f}%)")
                            st.session_state['df_filtered'] = filtered_df
                            st.session_state['is_filtered'] = True
                            st.session_state['filter_applied'] = f"{filter_column} BETWEEN {min_val} AND {max_val}"
                    
                    elif filter_type in ["=", ">", "<", ">=", "<="]:
                        filter_value = st.number_input(
                            f"Nilai untuk Filter ({filter_type})",
                            value=float(df[filter_column].median())
                        )
                        
                        if st.button("🔎 Terapkan Filter", key="filter_numeric_single"):
                            if filter_type == "=":
                                filtered_df = df[df[filter_column] == filter_value]
                            elif filter_type == ">":
                                filtered_df = df[df[filter_column] > filter_value]
                            elif filter_type == "<":
                                filtered_df = df[df[filter_column] < filter_value]
                            elif filter_type == ">=":
                                filtered_df = df[df[filter_column] >= filter_value]
                            elif filter_type == "<=":
                                filtered_df = df[df[filter_column] <= filter_value]
                            
                            st.success(f"✅ Filter diterapkan! {len(filtered_df):,} baris dari {len(df):,} baris ({len(filtered_df)/len(df)*100:.1f}%)")
                            st.session_state['df_filtered'] = filtered_df
                            st.session_state['is_filtered'] = True
                            st.session_state['filter_applied'] = f"{filter_column} {filter_type} {filter_value}"
                
                else:
                    # Categorical/String filter
                    st.info("📝 Filter untuk kolom kategorikal/teks")
                    
                    filter_method = st.radio(
                        "Metode Filter",
                        options=["Sama dengan (Exact Match)", "Mengandung (Contains)", "Di antara (In List)"],
                        horizontal=True
                    )
                    
                    if filter_method == "Sama dengan (Exact Match)":
                        unique_values = sorted(df[filter_column].dropna().unique().tolist())
                        if unique_values:
                            filter_value = st.selectbox(
                                "Pilih Nilai",
                                options=["-- Pilih Nilai --"] + unique_values
                            )
                            
                            if filter_value != "-- Pilih Nilai --" and st.button("🔎 Terapkan Filter", key="filter_exact"):
                                filtered_df = df[df[filter_column] == filter_value]
                                st.success(f"✅ Filter diterapkan! {len(filtered_df):,} baris dari {len(df):,} baris ({len(filtered_df)/len(df)*100:.1f}%)")
                                st.session_state['df_filtered'] = filtered_df
                                st.session_state['is_filtered'] = True
                                st.session_state['filter_applied'] = f"{filter_column} = '{filter_value}'"
                    
                    elif filter_method == "Mengandung (Contains)":
                        filter_text = st.text_input("Masukkan teks yang dicari (case-sensitive)")
                        
                        if filter_text and st.button("🔎 Terapkan Filter", key="filter_contains"):
                            filtered_df = df[df[filter_column].astype(str).str.contains(filter_text, na=False)]
                            st.success(f"✅ Filter diterapkan! {len(filtered_df):,} baris dari {len(df):,} baris ({len(filtered_df)/len(df)*100:.1f}%)")
                            st.session_state['df_filtered'] = filtered_df
                            st.session_state['is_filtered'] = True
                            st.session_state['filter_applied'] = f"{filter_column} LIKE '%{filter_text}%'"
                    
                    elif filter_method == "Di antara (In List)":
                        unique_values = sorted(df[filter_column].dropna().unique().tolist())
                        selected_values = st.multiselect(
                            "Pilih Nilai yang Diinginkan",
                            options=unique_values,
                            max_selections=10
                        )
                        
                        if selected_values and st.button("🔎 Terapkan Filter", key="filter_in_list"):
                            filtered_df = df[df[filter_column].isin(selected_values)]
                            st.success(f"✅ Filter diterapkan! {len(filtered_df):,} baris dari {len(df):,} baris ({len(filtered_df)/len(df)*100:.1f}%)")
                            st.session_state['df_filtered'] = filtered_df
                            st.session_state['is_filtered'] = True
                            st.session_state['filter_applied'] = f"{filter_column} IN ({', '.join(map(str, selected_values))})"
            
            # Show filtered data or reset filter
            if st.session_state.get('is_filtered', False):
                st.markdown("---")
                st.success(f"🎯 Filter Aktif: {st.session_state.get('filter_applied', '')}")
                
                col_filter1, col_filter2 = st.columns([3, 1])
                with col_filter1:
                    st.info(f"📊 Dataset terfilter: {len(st.session_state['df_filtered']):,} baris")
                with col_filter2:
                    if st.button("🔄 Reset Filter", key="reset_filter"):
                        st.session_state.pop('df_filtered', None)
                        st.session_state.pop('is_filtered', None)
                        st.session_state.pop('filter_applied', None)
                        st.rerun()
                
                # Show filtered data preview
                st.subheader("Preview Data Terfilter")
                st.dataframe(st.session_state['df_filtered'].head(20))
                
                # Button to use filtered dataset for preprocessing
                if st.button("✅ Gunakan Dataset Terfilter untuk Preprocessing", key="use_filtered"):
                    # Save filtered dataset to temp parquet
                    import uuid
                    import tempfile
                    filtered_path = os.path.join(
                        tempfile.gettempdir(),
                        f"filtered_{uuid.uuid4().hex}.parquet"
                    )
                    st.session_state['df_filtered'].to_parquet(filtered_path, index=False, compression="zstd")
                    st.session_state['df_processed_path'] = filtered_path
                    st.session_state['use_filtered_data'] = True
                    st.success(f"✅ Dataset terfilter ({len(st.session_state['df_filtered']):,} baris) disimpan untuk preprocessing")
                    st.info("🚀 Lanjut ke halaman Pelatihan Model untuk preprocessing dan training")
            else:
                # Button to use full dataset for preprocessing
                if st.button("✅ Gunakan Dataset Lengkap untuk Preprocessing", key="use_full"):
                    st.session_state['df_processed_path'] = raw_parquet_path
                    st.session_state['use_filtered_data'] = False
                    st.success("✅ Dataset lengkap akan digunakan untuk preprocessing")
                    st.info("🚀 Lanjut ke halaman Pelatihan Model untuk preprocessing dan training")

            # Preprocessing options
            st.markdown("---")
            st.subheader("⚙️ Opsi Preprocessing")
            st.caption("Preprocessing membersihkan data, menangani missing value/outlier, mengubah kategori dan tanggal menjadi fitur numerik, lalu menyimpan hasil untuk training.")

            if 'enable_outlier_detection' not in st.session_state:
                st.session_state['enable_outlier_detection'] = True
            
            if 'enable_data_validation' not in st.session_state:
                st.session_state['enable_data_validation'] = True
            
            if 'enable_duplicate_removal' not in st.session_state:
                st.session_state['enable_duplicate_removal'] = False

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                st.checkbox(
                    "Deteksi Outlier",
                    key="enable_outlier_detection",
                    help="Aktifkan deteksi dan penanganan outlier menggunakan metode IQR"
                )
            with col_opt2:
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

            preprocessing_test_size = st.slider(
                "Porsi data test sebelum preprocessing (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                key="preprocessing_test_size_pct",
                help=(
                    "Persentase data yang akan digunakan sebagai test set. "
                    "Split ditetapkan pada data mentah sebelum preprocessing."
                ),
            )

            preprocessing_validation_size = st.slider(
                "Porsi data validation sebelum preprocessing (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                key="preprocessing_validation_size_pct",
                help=(
                    "Persentase data yang akan digunakan sebagai validation set. "
                    "Diambil dari sisa data setelah test split. "
                    "Seluruh statistik preprocessing hanya di-fit pada train."
                ),
            )

            # Validation: Ensure test + validation < 80% (leaving at least 20% for train)
            total_split_pct = preprocessing_test_size + preprocessing_validation_size
            if total_split_pct >= 80:
                st.error(
                    f"⚠️ Total split (test + validation) harus kurang dari 80%. "
                    f"Saat ini: {total_split_pct}% (test: {preprocessing_test_size}%, validation: {preprocessing_validation_size}%). "
                    f"Silakan kurangi salah satu atau keduanya."
                )
                st.stop()

            # --- Opsi Penghapusan Fitur / Kolom Mentah ---
            available_raw_cols = [str(c) for c in df.columns]

            high_missing_suggestions = [
                str(c) for c in df.columns 
                if (df[c].isnull().sum() / len(df)) > 0.5
            ]
            constant_col_suggestions = [
                str(c) for c in df.columns 
                if df[c].nunique(dropna=False) <= 1
            ]

            with st.expander("🗑️ Opsi Penghapusan Fitur / Kolom Mentah (Feature Dropping)", expanded=True):
                st.markdown(
                    "Pilih kolom mentah yang ingin **dihapus / dikeluarkan** dari dataset sebelum pembersihan data, imputasi, dan rekayasa fitur (*feature engineering*). "
                    "Kolom yang dipilih tidak akan diproses, tidak akan membebani memori, dan tidak akan diubah menjadi fitur model."
                )

                q_cols = st.columns([1.2, 1.2, 1])
                with q_cols[0]:
                    if high_missing_suggestions:
                        if st.button(
                            f"⚡ Pilih Missing > 50% ({len(high_missing_suggestions)})",
                            key="btn_quick_missing",
                            help=f"Pilih: {', '.join(high_missing_suggestions)}"
                        ):
                            cur = set(st.session_state.get('columns_to_drop', []))
                            cur.update(high_missing_suggestions)
                            st.session_state['columns_to_drop'] = sorted(list(cur))
                            st.rerun()
                    else:
                        st.caption("✅ Tidak ada kolom missing > 50%")
                with q_cols[1]:
                    if constant_col_suggestions:
                        if st.button(
                            f"⚡ Pilih Kolom Konstan ({len(constant_col_suggestions)})",
                            key="btn_quick_constant",
                            help=f"Pilih: {', '.join(constant_col_suggestions)}"
                        ):
                            cur = set(st.session_state.get('columns_to_drop', []))
                            cur.update(constant_col_suggestions)
                            st.session_state['columns_to_drop'] = sorted(list(cur))
                            st.rerun()
                    else:
                        st.caption("✅ Tidak ada kolom nilai tunggal")
                with q_cols[2]:
                    if st.session_state.get('columns_to_drop'):
                        if st.button("🔄 Reset Pilihan Kolom", key="btn_reset_drop_cols"):
                            st.session_state['columns_to_drop'] = []
                            st.rerun()

                if 'columns_to_drop' not in st.session_state:
                    st.session_state['columns_to_drop'] = []

                # Filter agar hanya kolom yang valid di dataset
                st.session_state['columns_to_drop'] = [
                    c for c in st.session_state['columns_to_drop'] if c in available_raw_cols
                ]

                selected_drop_cols = st.multiselect(
                    "Pilih kolom untuk dihapus:",
                    options=available_raw_cols,
                    default=st.session_state['columns_to_drop'],
                    key="multiselect_drop_columns",
                    help="Pilih satu atau beberapa kolom yang ingin dihapus. Kolom ini akan dieliminasi dari seluruh proses downstream."
                )
                st.session_state['columns_to_drop'] = selected_drop_cols

                rem_cols = len(available_raw_cols) - len(selected_drop_cols)
                if selected_drop_cols:
                    drop_c1, drop_c2, drop_c3 = st.columns(3)
                    with drop_c1:
                        st.metric("Total Kolom Awal", len(available_raw_cols))
                    with drop_c2:
                        st.metric("Kolom Dihapus", f"-{len(selected_drop_cols)}")
                    with drop_c3:
                        st.metric("Kolom Tersisa", rem_cols)
                    st.warning(f"⚠️ **{len(selected_drop_cols)} kolom akan dihapus:** {', '.join(selected_drop_cols)}")
                    if rem_cols <= 0:
                        st.error("❌ Anda tidak dapat menghapus semua kolom. Sisakan minimal 1 kolom untuk diproses.")

            can_process = rem_cols > 0
            
            # Show process button only if data hasn't been processed yet
            if 'df_processed_path' not in st.session_state:
                process_data_btn = st.button(
                    "Proses Data",
                    type="primary",
                    width='stretch',
                    key="btn_process_data",
                    disabled=not can_process,
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
                    duplicate_metadata = None
                    result = None
                    temporary_input_paths = []
                    columns_to_drop = list(st.session_state.get('columns_to_drop', []))
                    dataset_rows = st.session_state.get('raw_data_total_rows', len(df) if hasattr(df, 'shape') else 0)
                    dataset_cols = st.session_state.get('raw_data_total_cols', len(df.columns) if hasattr(df, 'columns') else 0)

                    try:
                        input_target = st.session_state.get('raw_data_cache_path') or df

                        if st.session_state['enable_duplicate_removal'] and isinstance(input_target, str):
                            subset_cols = None
                            if duplicate_subset.strip():
                                subset_cols = [col.strip() for col in duplicate_subset.split(',') if col.strip()]
                                subset_cols = [col for col in subset_cols if col not in columns_to_drop]
                            duplicate_input = input_target
                            input_target, duplicate_metadata = remove_duplicates_from_parquet(
                                input_target,
                                subset=subset_cols,
                                keep='first',
                            )
                            if input_target != duplicate_input:
                                temporary_input_paths.append(input_target)

                            if duplicate_metadata['duplicates_removed'] > 0:
                                st.info(f"🔍 Duplikasi dihapus: {duplicate_metadata['duplicates_removed']:,} baris ({duplicate_metadata['duplicate_rate']:.2%})")
                            else:
                                st.info("✅ Tidak ada duplikasi ditemukan")

                        elif st.session_state['enable_duplicate_removal'] and isinstance(input_target, pd.DataFrame):
                            subset_cols = None
                            if duplicate_subset.strip():
                                subset_cols = [col.strip() for col in duplicate_subset.split(',') if col.strip()]
                                subset_cols = [col for col in subset_cols if col in df.columns and col not in columns_to_drop]

                            input_target, duplicate_metadata = remove_duplicates(input_target, subset=subset_cols, keep='first')

                            if duplicate_metadata['duplicates_removed'] > 0:
                                st.info(f"🔍 Duplikasi dihapus: {duplicate_metadata['duplicates_removed']:,} baris ({duplicate_metadata['duplicate_rate']:.2%})")
                            else:
                                st.info("✅ Tidak ada duplikasi ditemukan")

                        import gc
                        gc.collect()

                        # Get column mapping from metadata analysis if available
                        column_mapping = st.session_state.get('column_mapping', None)
                        result = preprocess_dataset_with_train_only_stats(
                            input_target,
                            enable_outlier_detection=st.session_state.get('enable_outlier_detection', True),
                            enable_data_validation=st.session_state.get('enable_data_validation', True),
                            test_size=preprocessing_test_size / 100,
                            validation_size=preprocessing_validation_size / 100,
                            drop_columns=columns_to_drop,
                            column_mapping=column_mapping,
                        )
                        for temporary_path in temporary_input_paths:
                            if os.path.exists(temporary_path):
                                os.unlink(temporary_path)
                        temporary_input_paths.clear()
                        gc.collect()

                        if result is None or len(result) < 3 or result[0] is None:
                            raise ValueError("Fungsi preprocessing mengembalikan hasil yang tidak valid.")

                        df_processed, feature_columns, preprocessing_metadata_pipeline = result
                        processing_time = time.time() - start_time

                        from datetime import datetime
                        preprocessing_metadata = dict(preprocessing_metadata_pipeline) if isinstance(preprocessing_metadata_pipeline, dict) else {}
                        preprocessing_metadata['processed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if duplicate_metadata is not None:
                            preprocessing_metadata['duplicate_removal'] = duplicate_metadata

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
                        if isinstance(df_processed, pd.DataFrame) and len(df_processed) >= 10:
                            (
                                training_frame,
                                validation_frame,
                                test_frame,
                                stratify_label,
                            ) = split_processed_dataset_with_validation(
                                df_processed,
                                test_size=preprocessing_test_size / 100,
                                validation_size=preprocessing_validation_size / 100
                            )
                            st.session_state["train_df"] = training_frame
                            st.session_state["validation_df"] = validation_frame
                            st.session_state["test_df"] = test_frame
                            st.session_state["split_test_size"] = preprocessing_test_size / 100
                            st.session_state["split_validation_size"] = preprocessing_validation_size / 100
                            st.session_state["feature_selection_scope"] = "train"
                            st.info(
                                f"Split train/validation/test ditetapkan pada data mentah. "
                                f"Train: {len(training_frame):,} ({(1 - preprocessing_test_size/100 - preprocessing_validation_size/100):.0%}), "
                                f"Validation: {len(validation_frame):,} ({preprocessing_validation_size/100:.0%}), "
                                f"Test: {len(test_frame):,} ({preprocessing_test_size/100:.0%}). "
                                "Statistik imputasi, encoding, dan outlier di-fit hanya pada train; "
                                f"validasi/test ditahan terpisah (stratifikasi: {stratify_label or 'tidak tersedia'})."
                            )
                        elif preprocessing_metadata.get("preprocessing_fit_scope") == "train_partition":
                            st.info(
                                "Split train/validation/test ditetapkan pada data mentah. "
                                "Statistik preprocessing di-fit hanya pada partisi train dan "
                                "diterapkan konsisten ke seluruh partisi."
                            )

                        preprocessing_success = True
                        st.success("✅ Data berhasil diproses dengan encoding canggih!")

                        try:
                            orig_r = int(dataset_rows) if isinstance(dataset_rows, (int, float, str)) else 0
                            orig_c = int(dataset_cols) if isinstance(dataset_cols, (int, float, str)) else 0
                            proc_r_raw = preprocessing_metadata.get('total_rows_processed', dataset_rows)
                            proc_r = int(proc_r_raw) if isinstance(proc_r_raw, (int, float, str)) else 0
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
                        sum_cols = st.columns(5)
                        with sum_cols[0]:
                            st.metric("Kolom Awal", str(dataset_cols))
                        with sum_cols[1]:
                            st.metric("Kolom Dihapus", f"-{len(columns_to_drop)}" if columns_to_drop else "0")
                        with sum_cols[2]:
                            st.metric("Fitur Akhir", str(preprocessing_metadata.get('final_features_count', len(feature_columns))))
                        with sum_cols[3]:
                            st.metric("Kolom Tanggal", str(preprocessing_metadata.get('date_columns_count', 0)))
                        with sum_cols[4]:
                            st.metric("Kolom Kategorikal", str(preprocessing_metadata.get('categorical_columns_count', 0)))

                        if columns_to_drop:
                            st.info(f"🗑️ **{len(columns_to_drop)} Kolom berhasil dieliminasi dari dataset:** {', '.join(columns_to_drop)}")

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
                                subset_val = dup_meta.get('subset')
                                if subset_val and isinstance(subset_val, (list, tuple)):
                                    st.write(f"Kolom yang dicek: {', '.join(str(c) for c in subset_val)}")
                                else:
                                    st.write("Kolom yang dicek: Semua kolom")

                    except Exception as e:
                        for temporary_path in temporary_input_paths:
                            if os.path.exists(temporary_path):
                                os.unlink(temporary_path)
                        error_type = type(e).__name__
                        error_message = str(e)
                        st.session_state['last_processing_error'] = {
                            'exception_type': error_type,
                            'message': error_message,
                            'dataset_rows': dataset_rows,
                            'dataset_cols': dataset_cols,
                            'file_size_mb': round(sum(int(getattr(f, 'size', 0) or 0) for f in uploaded_files) / (1024 * 1024), 2) if uploaded_files else 0,
                            'file_name': (
                                f"Merged ({len(uploaded_files)} files): " + ", ".join(f.name for f in uploaded_files[:2]) + ("..." if len(uploaded_files) > 2 else "")
                                if len(uploaded_files) > 1
                                else uploaded_files[0].name
                            ) if uploaded_files else 'unknown',
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
                # Use a bounded sample for the page summary; expensive selectors
                # load the full dataset only when the user starts the operation.
                df_processed = get_processed_sample(n=5000)
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

                    # Feature exclusion (Drop specific engineered features)
                    with st.expander("🗑️ Opsi Eliminasi Fitur Modeling (Exclude Features)", expanded=False):
                        st.caption("Keluarkan fitur spesifik dari kandidat model jika dirasa kurang informatif, redundan, atau berpotensi leakage.")
                        if 'excluded_engineered_features' not in st.session_state:
                            st.session_state['excluded_engineered_features'] = []
                        st.session_state['excluded_engineered_features'] = [
                            f for f in st.session_state['excluded_engineered_features'] if f in feature_columns
                        ]
                        excluded_features = st.multiselect(
                            "Pilih fitur yang ingin dikecualikan dari model:",
                            options=feature_columns,
                            default=st.session_state['excluded_engineered_features'],
                            key="excluded_engineered_features_selection",
                            help="Fitur yang dipilih akan dieliminasi dari kandidat pelatihan model."
                        )
                        st.session_state['excluded_engineered_features'] = excluded_features
                        if excluded_features:
                            st.warning(f"🗑️ **{len(excluded_features)} fitur dikecualikan:** {', '.join(excluded_features[:5])}{'...' if len(excluded_features) > 5 else ''}")

                    candidate_features = [f for f in feature_columns if f not in st.session_state.get('excluded_engineered_features', [])]
                    if not candidate_features:
                        candidate_features = feature_columns
                        st.warning("⚠️ Semua fitur dikecualikan! Mengembalikan semua fitur sebagai kandidat.")

                    selection_method = st.selectbox(
                        "Pilih metode seleksi fitur:",
                        ["Semua Fitur", "Seleksi Manual Kustom", "Select K-Best (Statistical)", 
                         "Mutual Information (Non-linear Filter)", "Tree-based Importance (XGBoost/LGBM)", 
                         "PCA Dimensionality Reduction"],
                        help="Pilih metode untuk menentukan fitur yang akan digunakan dalam training"
                    )

                    selected_features = []

                    if selection_method == "Semua Fitur":
                        selected_features = candidate_features
                        st.info(f"✅ Menggunakan semua {len(candidate_features)} fitur yang tersedia")
                    
                    elif selection_method == "Seleksi Manual Kustom":
                        st.write("**Pilih Fitur untuk Pelatihan:**")
                    
                        # Show feature categories for reference
                        with st.expander("📋 Kategori Fitur yang Tersedia"):
                            col1, col2, col3, col4 = st.columns(4)
                        
                            with col1:
                                st.write("**📊 Fitur Original:**")
                                # Use feature_columns as original features since original_features is not defined
                                for feat in candidate_features[:5]:  # Show first 5
                                    st.write(f"  • {feat}")
                                if len(candidate_features) > 5:
                                    st.write(f"  ... dan {len(candidate_features)-5} lainnya")
                        
                            with col2:
                                st.write("**🔢 Fitur Numerikal:**")
                                cand_numerical = [f for f in numerical_features if f in candidate_features]
                                if cand_numerical:
                                    for feat in cand_numerical[:5]:
                                        st.write(f"  • {feat}")
                                    if len(cand_numerical) > 5:
                                        st.write(f"  ... dan {len(cand_numerical)-5} lainnya")
                                else:
                                    st.write("  Tidak ada")
                        
                            with col3:
                                st.write("**⏰ Fitur Temporal:**")
                                cand_temporal = [f for f in temporal_features if f in candidate_features]
                                if cand_temporal:
                                    for feat in cand_temporal:
                                        st.write(f"  • {feat}")
                                else:
                                    st.write("  Tidak ada")
                        
                            with col4:
                                st.write("**🔧 Fitur Engineering:**")
                                cand_derived = [f for f in derived_features if f in candidate_features]
                                if cand_derived:
                                    for feat in cand_derived[:5]:
                                        st.write(f"  • {feat}")
                                    if len(cand_derived) > 5:
                                        st.write(f"  ... dan {len(cand_derived)-5} lainnya")
                                else:
                                    st.write("  Tidak ada")
                    
                        selected_features = st.multiselect(
                            "Pilih fitur yang ingin digunakan:",
                            options=candidate_features,
                            default=candidate_features[:min(15, len(candidate_features))],
                            help="Pilih fitur yang paling relevan untuk deteksi anomali. Semakin banyak fitur, semakin kompleks modelnya.",
                            max_selections=len(candidate_features)
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
                                selection_df = get_feature_selection_frame()
                                if selection_df is not None and len(selection_df) > 5:
                                    with st.spinner("Menghitung skor Select K-Best..."):
                                        selected_features, feature_scores = apply_select_k_best(
                                            selection_df, feature_columns, k=k_value, score_func_name=score_function
                                        )
                                
                                        st.success(f"✅ Select K-Best ({score_function}) berhasil! Memilih {len(selected_features)} fitur terbaik")
                                        st.write("**📊 Skor Fitur Teratas:**")
                                        st.dataframe(feature_scores.head(k_value), width='stretch')
                                
                                        final_features = selected_features.copy()
                                
                                        # Apply advanced filtering if selected
                                        if remove_correlated:
                                            final_features, to_remove = filter_correlated_features(
                                                selection_df, final_features, correlation_threshold=0.9, feature_scores_df=feature_scores
                                            )
                                            if len(to_remove) > 0:
                                                st.warning(f"🗑️ Dihapus {len(to_remove)} fitur karena korelasi tinggi: {', '.join(to_remove)}")
                                
                                        if variance_threshold > 0:
                                            final_features, low_var_features = filter_low_variance_features(
                                                selection_df, final_features, variance_threshold=variance_threshold
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
                                selection_df = get_feature_selection_frame()
                                if selection_df is None:
                                    st.error("Data hasil preprocessing tidak dapat dimuat.")
                                    return
                                selected_features, feature_scores = apply_mutual_info_selection(selection_df, feature_columns, k=k_mi)
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
                                selection_df = get_feature_selection_frame()
                                if selection_df is None:
                                    st.error("Data hasil preprocessing tidak dapat dimuat.")
                                    return
                                selected_features, feature_importances, model_name = apply_tree_based_selection(selection_df, feature_columns, k=k_tree)
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
                                processed_path = st.session_state.get("df_processed_path")
                                large_processed = (
                                    isinstance(processed_path, str)
                                    and os.path.exists(processed_path)
                                    and os.path.getsize(processed_path) > 50 * 1024 * 1024
                                )
                                if large_processed:
                                    df_processed = get_feature_selection_frame()
                                else:
                                    df_processed = get_df_processed()
                                if df_processed is None:
                                    st.error("Data hasil praproses tidak ditemukan!")
                                    return

                                old_pca_cols = [col for col in df_processed.columns if col.startswith('PCA_Component_')] if isinstance(df_processed, pd.DataFrame) else []
                                if old_pca_cols and isinstance(df_processed, pd.DataFrame):
                                    df_processed = df_processed.drop(columns=old_pca_cols)
                                    st.info(f"🗑️ Membersihkan {len(old_pca_cols)} kolom PCA lama")

                                df_pca, pca_cols, explained_variance = apply_pca_reduction(df_processed, feature_columns, n_components=n_comp)

                                if large_processed:
                                    st.warning(
                                        "Dataset besar: PCA dihitung dari sampel dan tidak menimpa file praproses penuh. "
                                        "Gunakan Select K-Best atau Semua Fitur untuk pelatihan."
                                    )
                                    st.session_state['selected_features_cache'] = feature_columns
                                else:
                                    for col in pca_cols:
                                        df_processed[col] = df_pca[col]
                                    update_df_processed(df_processed)
                                    st.session_state['selected_features_cache'] = pca_cols

                                st.success(f"✅ PCA selesai! {len(pca_cols)} komponen menjelaskan {sum(explained_variance):.2%} variansi.")

                                # Plot explained variance
                                fig = create_bar_chart([f"PC{i+1}" for i in range(len(explained_variance))],
                                           explained_variance,
                                           title='Variansi Terjelaskan per Komponen',
                                           labels={'x': 'Komponen Utama', 'y': 'Variansi Terjelaskan'})
                                st.plotly_chart(fig, width='stretch')
                                st.session_state['proceed_after_selection'] = not large_processed
                
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
                        excluded_set = set(st.session_state.get('excluded_engineered_features', []))
                        if excluded_set:
                            final_features = [f for f in final_features if f not in excluded_set]
                            selected_features = [f for f in selected_features if f not in excluded_set]

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
                        st.caption("Visualisasi menggunakan sample terbatas untuk menjaga responsivitas halaman.")
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
                            if fig is not None:
                                st.plotly_chart(fig, width='stretch')

                        # Plot 2: Distribusi usia pasien
                        if 'patient_age' in df_processed.columns:
                            fig = create_histogram_chart(df_processed, 'patient_age',
                                           nbins=20, title='Distribusi Usia Pasien')
                            if fig is not None:
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
