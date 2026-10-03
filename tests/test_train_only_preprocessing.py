import json
import os

import pandas as pd
import polars as pl

from large_file_processor import (
    compute_global_preprocessing_stats,
    prepare_train_validation_test_parquet,
    preprocess_dataset_with_train_only_stats,
    transform_dataframe_with_fitted_stats,
)
from state_manager import split_processed_dataset_with_validation


def _write_claims(path, rows=300):
    pd.DataFrame(
        {
            "claim_id": [f"C{i}" for i in range(rows)],
            "amount": [float(i % 80 + 10) for i in range(rows)],
            "claim_status": ["PAID" if i % 2 else "PENDING" for i in range(rows)],
            "fraud_label": [i % 2 for i in range(rows)],
        }
    ).to_parquet(path, index=False)


def test_preprocessing_stats_are_fitted_only_on_train_partition(tmp_path):
    source_path = str(tmp_path / "claims.parquet")
    _write_claims(source_path)
    partitioned_path = None
    processed_path = None
    try:
        partitioned_path, owned = prepare_train_validation_test_parquet(source_path)
        assert owned is True

        raw = pl.read_parquet(partitioned_path)
        held_out_id = (
            raw.filter(pl.col("__astina_partition") != "train")
            .select("__astina_split_row_id")
            .item(0, 0)
        )
        modified = raw.with_columns(
            pl.when(pl.col("__astina_split_row_id") == held_out_id)
            .then(1_000_000_000.0)
            .otherwise(pl.col("amount"))
            .alias("amount")
        )
        modified_path = str(tmp_path / "modified.parquet")
        modified.write_parquet(modified_path)

        original_stats = compute_global_preprocessing_stats(partitioned_path)
        modified_stats = compute_global_preprocessing_stats(modified_path)
        assert original_stats["fit_scope"] == "train_partition"
        assert original_stats["num_stats"]["amount"] == modified_stats["num_stats"]["amount"]
        assert original_stats["cat_stats"]["claim_status"] == modified_stats["cat_stats"]["claim_status"]
        assert "fraud_label" not in original_stats["num_stats"]
        assert "fraud_label" not in original_stats["cat_stats"]

        processed_path, features, metadata = preprocess_dataset_with_train_only_stats(
            modified_path
        )
        assert metadata["preprocessing_fit_scope"] == "train_partition"
        assert metadata["preprocessing_fit_rows"] == metadata["split_counts"]["train"]
        assert "fraud_label" not in features
        json.dumps(metadata["fit_statistics"])

        processed = pl.read_parquet(processed_path)
        transformed_value = processed.filter(
            pl.col("__astina_split_row_id") == held_out_id
        )["amount"].item()
        fitted_upper_bound = metadata["fit_statistics"]["num_stats"]["amount"]["upper_bound"]
        assert transformed_value == fitted_upper_bound

        held_out_raw = raw.filter(pl.col("__astina_partition") != "train").drop(
            "__astina_partition", "__astina_split_row_id"
        ).to_pandas()
        inference_result = transform_dataframe_with_fitted_stats(
            held_out_raw, metadata["fit_statistics"]
        )
        assert set(metadata["fit_statistics"]["initial_features"]).issubset(features)
        assert set(features).issubset(inference_result.columns)
        assert "fraud_label" not in features
        assert len(inference_result) == len(held_out_raw)
        inference_row = inference_result.loc[
            inference_result["claim_id"] == held_out_raw.iloc[0]["claim_id"]
        ].iloc[0]
        assert inference_row["amount"] == held_out_raw.iloc[0]["amount"]

        frame = processed.to_pandas()
        train, validation, test, _ = split_processed_dataset_with_validation(frame)
        assert len(train) == metadata["split_counts"]["train"]
        assert len(validation) == metadata["split_counts"]["validation"]
        assert len(test) == metadata["split_counts"]["test"]
        assert set(train["claim_id"]).isdisjoint(validation["claim_id"])
        assert set(train["claim_id"]).isdisjoint(test["claim_id"])
        assert set(validation["claim_id"]).isdisjoint(test["claim_id"])
    finally:
        for path in (partitioned_path, processed_path):
            if path and os.path.exists(path):
                os.unlink(path)
