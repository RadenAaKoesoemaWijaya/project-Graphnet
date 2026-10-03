import polars as pl
import pandas as pd
import numpy as np
import streamlit as st
import gc
import os
import uuid
import logging
from config import TEMP_DATA_DIR, LARGE_DATASET_CONFIG

logger = logging.getLogger("graphnet.large_file_processor")


def _is_target_column(column_name):
    name = column_name.lower()
    return any(token in name for token in ("fraud", "label", "target", "class"))


def compute_global_preprocessing_stats(parquet_path: str, enable_outlier_detection: bool = True, enable_data_validation: bool = True):
    """
    Pass 1: Extract global statistics, quantiles, missing rates, and categorical frequencies
    using Polars Lazy streaming without loading the entire dataset into memory.
    """
    lf = pl.scan_parquet(parquet_path)
    schema = lf.collect_schema()
    
    partition_column = "__astina_partition"
    fit_lf = lf
    if partition_column in schema.names():
        fit_lf = lf.filter(pl.col(partition_column) == "train")

    total_rows = fit_lf.select(pl.len()).collect().item()
    if total_rows == 0:
        raise ValueError("Partisi training kosong; statistik preprocessing tidak dapat di-fit.")

    original_columns = list(schema.names())
    
    # Classify columns
    date_columns = []
    numerical_columns = []
    categorical_columns = []
    
    for col, dtype in schema.items():
        if col in {partition_column, "__astina_split_row_id"}:
            continue
        if _is_target_column(col):
            continue
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'submitted', 'timestamp']) or dtype in [pl.Date, pl.Datetime]:
            date_columns.append(col)
        elif dtype.is_numeric():
            numerical_columns.append(col)
        else:
            categorical_columns.append(col)

    # Global aggregations for numerical columns
    num_aggs = []
    for col in numerical_columns:
        num_aggs.extend([
            pl.col(col).null_count().alias(f"{col}__nulls"),
            pl.col(col).median().alias(f"{col}__median"),
            pl.col(col).mean().alias(f"{col}__mean"),
            pl.col(col).std().alias(f"{col}__std"),
            pl.col(col).min().alias(f"{col}__min"),
            pl.col(col).max().alias(f"{col}__max"),
            pl.col(col).quantile(0.25).alias(f"{col}__q25"),
            pl.col(col).quantile(0.75).alias(f"{col}__q75"),
            pl.col(col).quantile(0.90).alias(f"{col}__q90"),
            pl.col(col).quantile(0.10).alias(f"{col}__q10"),
        ])

    num_stats = {}
    if num_aggs:
        stats_df = fit_lf.select(num_aggs).collect()
        for col in numerical_columns:
            null_count = stats_df[f"{col}__nulls"][0]
            median_val = stats_df[f"{col}__median"][0]
            mean_val = stats_df[f"{col}__mean"][0]
            std_val = stats_df[f"{col}__std"][0]
            min_val = stats_df[f"{col}__min"][0]
            max_val = stats_df[f"{col}__max"][0]
            q25 = stats_df[f"{col}__q25"][0]
            q75 = stats_df[f"{col}__q75"][0]
            q90 = stats_df[f"{col}__q90"][0]
            q10 = stats_df[f"{col}__q10"][0]

            # Replace NaNs in stats if column was entirely null
            median_val = float(median_val) if median_val is not None and not np.isnan(median_val) else 0.0
            mean_val = float(mean_val) if mean_val is not None and not np.isnan(mean_val) else 0.0
            std_val = float(std_val) if std_val is not None and not np.isnan(std_val) else 0.0
            min_val = float(min_val) if min_val is not None and not np.isnan(min_val) else 0.0
            max_val = float(max_val) if max_val is not None and not np.isnan(max_val) else 0.0
            q25 = float(q25) if q25 is not None and not np.isnan(q25) else 0.0
            q75 = float(q75) if q75 is not None and not np.isnan(q75) else 0.0

            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            num_stats[col] = {
                'null_count': null_count,
                'missing_rate': null_count / total_rows,
                'median': median_val,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'q25': q25,
                'q75': q75,
                'q90': float(q90) if q90 is not None and not np.isnan(q90) else max_val,
                'q10': float(q10) if q10 is not None and not np.isnan(q10) else min_val,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

    # Extract categorical stats & frequency maps (Top categories only)
    cat_stats = {}
    for col in categorical_columns:
        try:
            cardinality = fit_lf.select(pl.col(col).n_unique()).collect().item()
            top_cats = (
                fit_lf.group_by(pl.col(col))
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .head(50)
                .collect()
            )
            freq_dict = {
                str(row[col]): int(row["count"])
                for row in top_cats.iter_rows(named=True)
                if row[col] is not None
            }
            cat_stats[col] = {
                'cardinality': int(cardinality or 0),
                'freq_map': freq_dict,
                'top_categories': list(freq_dict.keys())[:5] if cardinality <= 5 else []
            }
        except Exception as e:
            logger.warning(f"Failed to extract categorical stats for {col}: {e}")
            cat_stats[col] = {'cardinality': 0, 'freq_map': {}, 'top_categories': []}

    # Filter numerical features
    id_keywords = ['id', 'identifier', 'key', 'code']
    final_features = []
    for col, st_info in num_stats.items():
        if _is_target_column(col):
            continue
        if any(keyword in col.lower() for keyword in id_keywords):
            if not col.endswith('_encoded'):
                continue
        if st_info['missing_rate'] > 0.5:
            continue
        if st_info['std'] < 1e-6:
            continue
        final_features.append(col)

    stats = {
        'total_rows': total_rows,
        'fit_scope': 'train_partition' if partition_column in schema.names() else 'full_dataset',
        'original_columns': original_columns,
        'date_columns': date_columns,
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'num_stats': num_stats,
        'cat_stats': cat_stats,
        'initial_features': final_features
    }
    return stats

def stream_transform_to_parquet(input_parquet_path: str, output_parquet_path: str | None = None, stats: dict | None = None,
                                enable_outlier_detection: bool = True, enable_data_validation: bool = True):
    """
    Pass 2: Apply lazy transformation expressions and stream directly to output Parquet on disk.
    Peak memory remains bounded regardless of input file size.
    """
    if stats is None:
        stats = compute_global_preprocessing_stats(
            input_parquet_path, 
            enable_outlier_detection=enable_outlier_detection,
            enable_data_validation=enable_data_validation
        )

    if output_parquet_path is None:
        # Ensure TEMP_DATA_DIR exists before attempting to write
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)
        output_parquet_path = os.path.join(
            TEMP_DATA_DIR,
            f"preprocessed_{uuid.uuid4().hex}.parquet"
        )
    else:
        # Ensure the parent directory of the output path exists
        output_dir = os.path.dirname(output_parquet_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    lf = pl.scan_parquet(input_parquet_path)
    total_rows = int(lf.select(pl.len()).collect().item())
    transform_exprs = []
    final_features = list(stats['initial_features'])
    encoding_metadata = {}

    # 1. Date Features
    for col in stats['date_columns']:
        try:
            # Parse to date/datetime if string
            date_expr = pl.col(col)
            # Try parsing if string dtype
            date_col_typed = (
                pl.when(date_expr.is_not_null())
                .then(date_expr.cast(pl.Utf8).str.to_datetime(strict=False))
                .otherwise(None)
            )
            transform_exprs.extend([
                date_col_typed.dt.weekday().fill_null(0).cast(pl.Int8).alias(f"{col}_day_of_week"),
                date_col_typed.dt.month().fill_null(1).cast(pl.Int8).alias(f"{col}_month"),
                date_col_typed.dt.year().fill_null(2020).cast(pl.Int32).alias(f"{col}_year"),
                date_col_typed.dt.day().fill_null(1).cast(pl.Int8).alias(f"{col}_day"),
            ])
            final_features.extend([f"{col}_day_of_week", f"{col}_month", f"{col}_year", f"{col}_day"])
        except Exception as e:
            logger.warning(f"Could not construct lazy date features for {col}: {e}")

    # 2. Outlier Capping & Missing Imputation for Numerical Columns
    for col in stats['numerical_columns']:
        st_col = stats['num_stats'].get(col, {})
        median_val = st_col.get('median', 0.0)
        
        expr = pl.col(col).fill_null(median_val)
        if enable_outlier_detection and 'lower_bound' in st_col and 'upper_bound' in st_col:
            lower = st_col['lower_bound']
            upper = st_col['upper_bound']
            if upper > lower:
                expr = expr.clip(lower, upper)
        
        transform_exprs.append(expr.cast(pl.Float32).alias(col))

    # 3. Categorical Encodings
    for col, c_info in stats['cat_stats'].items():
        cardinality = c_info.get('cardinality', 0)
        freq_map = c_info.get('freq_map', {})
        top_cats = c_info.get('top_categories', [])

        if cardinality <= 5 and top_cats:
            # One-hot encoding
            encoded_names = []
            for cat in top_cats[1:]:  # drop first
                sanitized_cat = str(cat).replace(" ", "_").replace("-", "_")
                dummy_name = f"{col}_{sanitized_cat}"
                transform_exprs.append(
                    pl.when(pl.col(col).cast(pl.Utf8) == str(cat))
                    .then(1)
                    .otherwise(0)
                    .cast(pl.Int8)
                    .alias(dummy_name)
                )
                encoded_names.append(dummy_name)
                final_features.append(dummy_name)
            encoding_metadata[col] = {'strategy': 'one_hot', 'features': encoded_names, 'cardinality': cardinality}

        elif freq_map:
            # Frequency encoding via replace / mapping
            freq_name = f"{col}_freq_encoded"
            # Build when-then branch for top categories
            when_expr = None
            for cat_val, count_val in list(freq_map.items())[:30]:
                cond = pl.col(col).cast(pl.Utf8) == str(cat_val)
                if when_expr is None:
                    when_expr = pl.when(cond).then(count_val)
                else:
                    when_expr = when_expr.when(cond).then(count_val)
            
            if when_expr is not None:
                freq_expr = when_expr.otherwise(0).cast(pl.Float32).alias(freq_name)
                transform_exprs.append(freq_expr)
                final_features.append(freq_name)
                encoding_metadata[col] = {'strategy': 'frequency', 'features': [freq_name], 'cardinality': cardinality}

    # 4. Financial & Amount Ratios (Top 5 amounts)
    amount_cols = [c for c in stats['numerical_columns'] if any(k in c.lower() for k in ['amount', 'cost', 'price', 'fee', 'charge'])][:5]
    if len(amount_cols) >= 2:
        for i, c1 in enumerate(amount_cols):
            for c2 in amount_cols[i+1:i+3]:
                ratio_name = f"{c1}_to_{c2}_ratio"
                ratio_expr = (
                    pl.col(c1) / (pl.col(c2) + 1e-6)
                ).fill_nan(1.0).fill_null(1.0).clip(0.0, 100.0).cast(pl.Float32).alias(ratio_name)
                transform_exprs.append(ratio_expr)
                final_features.append(ratio_name)

    # 5. Age squared and high value flags
    age_cols = [c for c in stats['numerical_columns'] if any(k in c.lower() for k in ['age', 'years', 'old'])]
    for c in age_cols:
        sq_name = f"{c}_squared"
        transform_exprs.append((pl.col(c) ** 2).cast(pl.Float32).alias(sq_name))
        final_features.append(sq_name)

    # Apply all transforms lazily
    lf_transformed = lf.with_columns(transform_exprs)
    
    # Deduplicate final feature names preserving order
    seen_features = set()
    dedup_final_features = []
    for f in final_features:
        if f not in seen_features:
            seen_features.add(f)
            dedup_final_features.append(f)

    # Execute streaming sink to disk
    # Ensure the output directory exists — the OS may have cleaned the temp
    # folder between runs, causing Polars sink_parquet to raise FileNotFoundError.
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    logger.info(f"Streaming transformed dataset to {output_parquet_path}...")
    lf_transformed.sink_parquet(output_parquet_path, compression="zstd")

    preprocessing_metadata = {
        'original_columns': stats['original_columns'],
        'original_columns_count': len(stats['original_columns']),
        'total_rows_processed': total_rows,
        'preprocessing_fit_scope': stats.get('fit_scope', 'full_dataset'),
        'preprocessing_fit_rows': int(stats['total_rows']),
        'final_features_count': len(dedup_final_features),
        'final_features': dedup_final_features,
        'date_columns_count': len(stats['date_columns']),
        'categorical_columns_count': len(stats['categorical_columns']),
        'numerical_columns_count': len(stats['numerical_columns']),
        'enhanced_encoding_metadata': encoding_metadata,
        'outlier_metadata': {'outlier_detection_enabled': enable_outlier_detection},
        'validation_metadata': {'data_validation_enabled': enable_data_validation},
        'processing_method': 'polars_lazy_streaming_pipeline'
    }

    return output_parquet_path, dedup_final_features, preprocessing_metadata

def preprocess_large_dataset(df_or_path, chunk_size=50000, progress_bar=True, enable_outlier_detection=True, enable_data_validation=True):
    """
    Optimized streaming preprocessing for large datasets with bounded memory consumption.
    Accepts either a Parquet file path (str) or a pandas DataFrame.
    """
    temp_input_created = False
    
    if isinstance(df_or_path, str) and os.path.exists(df_or_path):
        parquet_input_path = df_or_path
    else:
        # Ensure TEMP_DATA_DIR exists before attempting to write
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)
        # Convert in-memory DataFrame to temporary Parquet to stream out-of-core
        parquet_input_path = os.path.join(TEMP_DATA_DIR, f"temp_input_{uuid.uuid4().hex}.parquet")
        if isinstance(df_or_path, pd.DataFrame):
            df_or_path.to_parquet(parquet_input_path, index=False, compression="snappy")
        elif isinstance(df_or_path, pl.DataFrame):
            df_or_path.write_parquet(parquet_input_path, compression="snappy")
        else:
            raise ValueError("Input must be a valid Parquet path or DataFrame.")
        temp_input_created = True

    try:
        stats = compute_global_preprocessing_stats(
            parquet_input_path,
            enable_outlier_detection=enable_outlier_detection,
            enable_data_validation=enable_data_validation
        )
        
        # Ensure TEMP_DATA_DIR exists before attempting to write
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)
        output_parquet_path = os.path.join(TEMP_DATA_DIR, f"preprocessed_{uuid.uuid4().hex}.parquet")
        out_path, final_features, metadata = stream_transform_to_parquet(
            parquet_input_path,
            output_parquet_path=output_parquet_path,
            stats=stats,
            enable_outlier_detection=enable_outlier_detection,
            enable_data_validation=enable_data_validation
        )
        
        # If input was a small DataFrame and caller expects a DataFrame back, check size
        if temp_input_created and stats['total_rows'] <= 50000:
            df_processed = pd.read_parquet(out_path)
            return df_processed, final_features, metadata
        
        # For large datasets, return the parquet path reference
        # If caller expects (df, features, metadata), we provide a lightweight proxy or df when needed
        return out_path, final_features, metadata

    finally:
        if temp_input_created and os.path.exists(parquet_input_path):
            try:
                os.unlink(parquet_input_path)
            except Exception:
                pass

def optimize_categorical_encoding_large(df, categorical_columns, sample_size=10000):
    """
    Backwards-compatible fallback for categorical encoding on smaller in-memory DataFrames
    """
    encoding_metadata = {}
    low_cardinality_cols = []
    new_features = []
    
    for col in categorical_columns:
        try:
            sample_data = df[col].sample(min(sample_size, len(df)))
            cardinality = sample_data.nunique()
            
            if cardinality <= 5:
                low_cardinality_cols.append(col)
                encoding_metadata[col] = {'strategy': 'one_hot', 'cardinality': cardinality}
            elif cardinality <= 50:
                new_features.append(pd.Series(pd.factorize(df[col])[0], name=f'{col}_encoded', index=df.index))
                freq_map = sample_data.value_counts()
                new_features.append(df[col].map(freq_map).fillna(0).rename(f'{col}_freq_encoded'))
                encoding_metadata[col] = {
                    'strategy': 'label_frequency',
                    'features': [f'{col}_encoded', f'{col}_freq_encoded'],
                    'cardinality': cardinality
                }
            else:
                freq_map = sample_data.value_counts()
                new_features.append(df[col].map(freq_map).fillna(0).rename(f'{col}_freq_encoded'))
                encoding_metadata[col] = {
                    'strategy': 'frequency_only',
                    'features': [f'{col}_freq_encoded'],
                    'cardinality': cardinality
                }
        except Exception as e:
            encoding_metadata[col] = {'strategy': 'failed', 'error': str(e)}
            continue
            
    if new_features:
        df = pd.concat([df] + new_features, axis=1)
        
    if low_cardinality_cols:
        df = pd.get_dummies(df, columns=low_cardinality_cols, drop_first=True)
        for col in low_cardinality_cols:
            encoded_features = [c for c in df.columns if c.startswith(f"{col}_")]
            encoding_metadata[col]['features'] = encoded_features
    
    return df, encoding_metadata

def smart_data_type_optimization(df, memory_threshold=0.8):
    """
    Smart data type optimization for DataFrame
    """
    if isinstance(df, str) or not isinstance(df, pd.DataFrame):
        return df
        
    current_memory_shallow = df.memory_usage(deep=False).sum() / (1024**3)
    if current_memory_shallow < 0.5:
        return df
    
    for col in df.select_dtypes(include=[np.int64]).columns:  # type: ignore[arg-type]
        c_min = df[col].min()
        c_max = df[col].max()
        if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)

    for col in df.select_dtypes(include=[np.float64]).columns:  # type: ignore[arg-type]
        df[col] = df[col].astype(np.float32)

    return df


def prepare_train_validation_test_parquet(input_data, test_size=0.2, validation_size=0.15):
    """Write a deterministic row-level split marker before fitting preprocessing statistics."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("Ukuran test dan validation harus berada di antara 0 dan 1.")
    if test_size + validation_size >= 1:
        raise ValueError("Gabungan ukuran test dan validation harus kurang dari 1.")

    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    source_is_temporary = False
    retain_temporary_source = False
    if isinstance(input_data, pd.DataFrame):
        source_path = os.path.join(TEMP_DATA_DIR, f"raw_split_source_{uuid.uuid4().hex}.parquet")
        input_data.to_parquet(source_path, index=False, compression="zstd")
        source_is_temporary = True
    elif isinstance(input_data, str) and os.path.exists(input_data):
        source_path = input_data
    else:
        raise ValueError("Input split harus berupa DataFrame atau path Parquet yang tersedia.")

    partition_column = "__astina_partition"
    row_id_column = "__astina_split_row_id"
    output_path = os.path.join(TEMP_DATA_DIR, f"partitioned_raw_{uuid.uuid4().hex}.parquet")
    try:
        lf = pl.scan_parquet(source_path)
        schema = lf.collect_schema()
        names = set(schema.names())
        if partition_column in names:
            allowed = {"train", "validation", "test"}
            invalid = (
                lf.select(pl.col(partition_column).drop_nulls().unique())
                .collect()
                .to_series()
                .to_list()
            )
            if not set(invalid).issubset(allowed):
                raise ValueError("Kolom partisi ASTINA berisi nilai yang tidak dikenal.")
            retain_temporary_source = source_is_temporary
            return source_path, source_is_temporary
        if row_id_column in names:
            raise ValueError(f"Kolom internal {row_id_column} sudah ada pada sumber data.")

        total_rows = int(lf.select(pl.len()).collect().item())
        if total_rows < 10:
            raise ValueError("Dataset minimal 10 baris diperlukan untuk train/validation/test.")

        indexed = lf.with_row_index(row_id_column)
        hash_columns = [pl.col(row_id_column)]
        label_candidates = [
            name for name in schema.names()
            if any(key in name.lower() for key in ("fraud", "label", "target", "class"))
        ]
        for candidate in label_candidates:
            unique_values = (
                lf.select(pl.col(candidate).drop_nulls().unique().limit(3))
                .collect()
                .to_series()
                .to_list()
            )
            if len(unique_values) == 2:
                hash_columns.append(pl.col(candidate))
                break
        bucket = pl.struct(hash_columns).hash(seed=42) % 10000
        train_boundary = int(round((1.0 - test_size - validation_size * (1.0 - test_size)) * 10000))
        validation_boundary = int(round((1.0 - test_size) * 10000))
        partition_expr = (
            pl.when(bucket < train_boundary)
            .then(pl.lit("train"))
            .when(bucket < validation_boundary)
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test"))
            .alias(partition_column)
        )
        indexed.with_columns(partition_expr).sink_parquet(output_path, compression="zstd")

        counts = {
            row[partition_column]: int(row["len"])
            for row in (
                pl.scan_parquet(output_path)
                .group_by(partition_column)
                .len()
                .collect()
                .iter_rows(named=True)
            )
        }
        if any(counts.get(name, 0) == 0 for name in ("train", "validation", "test")):
            raise ValueError(
                "Split deterministik menghasilkan partisi kosong; dataset terlalu kecil atau tidak seimbang."
            )
        return output_path, True
    except Exception:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise
    finally:
        if source_is_temporary and not retain_temporary_source and os.path.exists(source_path):
            os.unlink(source_path)


def preprocess_dataset_with_train_only_stats(
    input_data,
    *,
    enable_outlier_detection=True,
    enable_data_validation=True,
    test_size=0.2,
    validation_size=0.15,
):
    """Fit preprocessing statistics on train rows and transform all rows consistently."""
    partitioned_path, owns_partitioned_path = prepare_train_validation_test_parquet(
        input_data,
        test_size=test_size,
        validation_size=validation_size,
    )
    output_path = os.path.join(TEMP_DATA_DIR, f"preprocessed_train_fit_{uuid.uuid4().hex}.parquet")
    try:
        stats = compute_global_preprocessing_stats(
            partitioned_path,
            enable_outlier_detection=enable_outlier_detection,
            enable_data_validation=enable_data_validation,
        )
        transformed_path, feature_columns, metadata = stream_transform_to_parquet(
            partitioned_path,
            output_parquet_path=output_path,
            stats=stats,
            enable_outlier_detection=enable_outlier_detection,
            enable_data_validation=enable_data_validation,
        )
        metadata["preprocessing_contract_version"] = 1
        metadata["preprocessing_fit_scope"] = "train_partition"
        metadata["split_column"] = "__astina_partition"
        metadata["split_row_id_column"] = "__astina_split_row_id"
        metadata["split_counts"] = {
            row["__astina_partition"]: int(row["len"])
            for row in (
                pl.scan_parquet(partitioned_path)
                .group_by("__astina_partition")
                .len()
                .collect()
                .iter_rows(named=True)
            )
        }
        metadata["requested_split"] = {
            "test_fraction": float(test_size),
            "validation_fraction_of_remaining": float(validation_size),
        }
        metadata["fit_statistics"] = {
            "total_rows": int(stats["total_rows"]),
            "fit_scope": stats["fit_scope"],
            "original_columns": stats["original_columns"],
            "date_columns": stats["date_columns"],
            "numerical_columns": stats["numerical_columns"],
            "categorical_columns": stats["categorical_columns"],
            "num_stats": stats["num_stats"],
            "cat_stats": stats["cat_stats"],
            "initial_features": stats["initial_features"],
            "enable_outlier_detection": bool(enable_outlier_detection),
        }
        return transformed_path, feature_columns, metadata
    except Exception:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise
    finally:
        if owns_partitioned_path and os.path.exists(partitioned_path):
            os.unlink(partitioned_path)


def transform_dataframe_with_fitted_stats(df, stats):
    """Apply a persisted training preprocessing contract to one inference DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Inference input harus berupa pandas DataFrame.")
    if not isinstance(stats, dict) or not stats.get("initial_features"):
        raise ValueError("Statistik preprocessing training tidak valid atau tidak lengkap.")

    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    input_path = os.path.join(TEMP_DATA_DIR, f"inference_raw_{uuid.uuid4().hex}.parquet")
    output_path = os.path.join(TEMP_DATA_DIR, f"inference_transformed_{uuid.uuid4().hex}.parquet")
    try:
        inference_df = df.copy()
        required_columns = set(stats.get("original_columns", []))
        required_columns.update(stats.get("numerical_columns", []))
        required_columns.update(stats.get("categorical_columns", []))
        required_columns.update(stats.get("date_columns", []))
        for column in required_columns.difference(inference_df.columns):
            if column not in {"__astina_partition", "__astina_split_row_id"}:
                inference_df[column] = np.nan
        inference_df.to_parquet(input_path, index=False, compression="zstd")
        transformed_path, _, _ = stream_transform_to_parquet(
            input_path,
            output_parquet_path=output_path,
            stats=stats,
            enable_outlier_detection=bool(stats.get("enable_outlier_detection", True)),
        )
        return pd.read_parquet(transformed_path)
    finally:
        for path in (input_path, output_path):
            if os.path.exists(path):
                os.unlink(path)