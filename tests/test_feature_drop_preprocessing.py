import os
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from large_file_processor import (
    compute_global_preprocessing_stats,
    prepare_train_validation_test_parquet,
    preprocess_dataset_with_train_only_stats,
)
from preprocessing_optimized import preprocess_insurance_claims_optimized


def _make_sample_parquet(path, rows=120):
    df = pd.DataFrame({
        "claim_id": [f"C{i}" for i in range(rows)],
        "billed_amount": [float(i * 100 + 50) for i in range(rows)],
        "paid_amount": [float(i * 80 + 30) for i in range(rows)],
        "patient_age": [20 + (i % 50) for i in range(rows)],
        "claim_status": ["PAID" if i % 2 == 0 else "DENIED" for i in range(rows)],
        "unwanted_column_1": [f"junk_{i}" for i in range(rows)],
        "unwanted_column_2": [999.0 for _ in range(rows)],
        "fraud_label": [i % 2 for i in range(rows)],
    })
    df.to_parquet(path, index=False)
    return df


def test_col_info_arrow_serialization():
    """Verify that col_info serialization with string dtypes never fails Arrow serialization."""
    df = pd.DataFrame({
        "col_int": [1, 2, 3],
        "col_float": [1.5, 2.5, 3.5],
        "col_str": ["a", "b", "c"],
    })
    col_info = pd.DataFrame({
        "Kolom": [str(c) for c in df.columns],
        "Tipe Data": [str(dtype) for dtype in df.dtypes],
        "Missing Values": [int(v) for v in df.isnull().sum().values],
        "Unique Values": [int(df[col].nunique()) for col in df.columns],
    })
    table = pa.Table.from_pandas(col_info)
    assert table is not None
    assert "Tipe Data" in table.column_names
    assert len(table) == 3


def test_compute_stats_with_drop_columns(tmp_path):
    source_path = str(tmp_path / "data.parquet")
    _make_sample_parquet(source_path)

    stats = compute_global_preprocessing_stats(
        source_path,
        drop_columns=["unwanted_column_1", "unwanted_column_2"],
    )
    assert "unwanted_column_1" not in stats["original_columns"]
    assert "unwanted_column_2" not in stats["original_columns"]
    assert "unwanted_column_1" not in stats["categorical_columns"]
    assert "unwanted_column_2" not in stats["numerical_columns"]
    assert "unwanted_column_1" in stats["dropped_columns"]
    assert "unwanted_column_2" in stats["dropped_columns"]


def test_prepare_partitioned_with_drop_columns(tmp_path):
    source_path = str(tmp_path / "data.parquet")
    _make_sample_parquet(source_path)

    partitioned_path, owned = prepare_train_validation_test_parquet(
        source_path,
        drop_columns=["unwanted_column_1", "unwanted_column_2"],
    )
    try:
        schema = pl.scan_parquet(partitioned_path).collect_schema()
        assert "unwanted_column_1" not in schema.names()
        assert "unwanted_column_2" not in schema.names()
        assert "__astina_partition" in schema.names()
        assert "billed_amount" in schema.names()
    finally:
        if owned and os.path.exists(partitioned_path):
            os.unlink(partitioned_path)


def test_preprocess_with_train_only_stats_drop_columns(tmp_path):
    source_path = str(tmp_path / "data.parquet")
    _make_sample_parquet(source_path)

    processed_path, features, metadata = preprocess_dataset_with_train_only_stats(
        source_path,
        drop_columns=["unwanted_column_1", "unwanted_column_2"],
    )
    try:
        assert "unwanted_column_1" not in features
        assert "unwanted_column_2" not in features
        assert "unwanted_column_1" in metadata["dropped_columns"]
        assert "unwanted_column_2" in metadata["dropped_columns"]
        assert metadata["dropped_columns_count"] == 2

        processed_schema = pl.scan_parquet(processed_path).collect_schema()
        assert "unwanted_column_1" not in processed_schema.names()
        assert "unwanted_column_2" not in processed_schema.names()
    finally:
        if processed_path and os.path.exists(processed_path):
            os.unlink(processed_path)


def test_preprocess_insurance_claims_optimized_drop_columns():
    df = pd.DataFrame({
        "claim_id": ["C1", "C2", "C3", "C4"],
        "billed_amount": [100.0, 200.0, 300.0, 400.0],
        "paid_amount": [90.0, 180.0, 270.0, 360.0],
        "col_to_drop": ["x", "y", "z", "w"],
    })
    df_proc, features, meta = preprocess_insurance_claims_optimized(
        df,
        enable_large_file_handling=False,
        drop_columns=["col_to_drop"],
    )
    assert "col_to_drop" not in df_proc.columns
    assert "col_to_drop" not in features
    assert "col_to_drop" in meta["dropped_columns"]
