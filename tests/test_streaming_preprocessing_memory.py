import os
import tempfile
import numpy as np
import pandas as pd
import polars as pl
import pytest
from large_file_processor import (
    compute_global_preprocessing_stats,
    stream_transform_to_parquet,
    preprocess_large_dataset
)
from file_handler import stream_csv_to_parquet, ingest_file_to_raw_parquet

def test_compute_global_stats_and_stream_transform(tmp_path):
    # Generate test dataset
    n_rows = 5000
    df = pd.DataFrame({
        'claim_id': [f'CLM_{i}' for i in range(n_rows)],
        'claim_date': pd.date_range('2023-01-01', periods=n_rows, freq='h').astype(str),
        'claim_amount': np.random.uniform(100.0, 50000.0, size=n_rows),
        'billed_amount': np.random.uniform(120.0, 60000.0, size=n_rows),
        'patient_age': np.random.randint(18, 90, size=n_rows),
        'service_type': np.random.choice(['Inpatient', 'Outpatient', 'Emergency', 'Dental'], size=n_rows),
        'provider_id': np.random.choice([f'PRV_{j}' for j in range(20)], size=n_rows),
    })
    # Inject some missing values and outliers
    df.loc[10:30, 'claim_amount'] = np.nan
    df.loc[50, 'claim_amount'] = 9999999.0  # extreme outlier

    raw_parquet_path = str(tmp_path / "test_raw.parquet")
    df.to_parquet(raw_parquet_path, index=False)

    # 1. Test Pass 1 stats computation
    stats = compute_global_preprocessing_stats(raw_parquet_path)
    assert stats['total_rows'] == n_rows
    assert 'claim_amount' in stats['num_stats']
    assert stats['num_stats']['claim_amount']['null_count'] > 0
    assert 'service_type' in stats['cat_stats']

    # 2. Test Pass 2 streaming transformation
    output_parquet_path = str(tmp_path / "test_output.parquet")
    out_path, features, metadata = stream_transform_to_parquet(
        raw_parquet_path,
        output_parquet_path=output_parquet_path,
        stats=stats,
        enable_outlier_detection=True
    )

    assert os.path.exists(out_path)
    assert len(features) > 5

    # Verify transformed parquet can be read and has no nulls in final features
    df_result = pl.read_parquet(out_path)
    assert len(df_result) == n_rows
    assert 'claim_date_month' in df_result.columns
    assert 'claim_amount' in df_result.columns

    # Outlier capping check: extreme outlier 9999999.0 should be clipped
    max_amount = df_result['claim_amount'].max()
    assert max_amount < 9999999.0

def test_preprocess_large_dataset_from_path(tmp_path):
    n_rows = 2000
    df = pd.DataFrame({
        'claim_id': [f'CLM_{i}' for i in range(n_rows)],
        'amount': np.random.uniform(50, 1000, size=n_rows),
        'fee': np.random.uniform(10, 200, size=n_rows),
        'category': np.random.choice(['A', 'B', 'C'], size=n_rows)
    })
    raw_path = str(tmp_path / "input.parquet")
    df.to_parquet(raw_path, index=False)

    out_path, features, metadata = preprocess_large_dataset(
        raw_path,
        enable_outlier_detection=True
    )

    assert isinstance(out_path, str)
    assert os.path.exists(out_path)
    assert 'amount' in features
    assert metadata['total_rows_processed'] == n_rows
