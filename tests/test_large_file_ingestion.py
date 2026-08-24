import pandas as pd

from file_handler import read_large_csv, stream_csv_to_parquet


def test_stream_csv_to_parquet_preserves_rows_and_columns(tmp_path):
    source = tmp_path / "claims.csv"
    expected = pd.DataFrame(
        {
            "claim_id": ["C1", "C2", "C3"],
            "amount": [10.5, 20.0, 30.25],
        }
    )
    expected.to_csv(source, index=False)

    output, row_count = stream_csv_to_parquet(
        str(source),
        output_path=str(tmp_path / "claims.parquet"),
        chunk_size=2,
        progress_bar=False,
    )

    actual = pd.read_parquet(output)
    assert row_count == len(expected)
    assert list(actual.columns) == list(expected.columns)
    assert actual.to_dict("records") == expected.to_dict("records")


def test_large_csv_reader_does_not_require_concat(monkeypatch, tmp_path):
    source = tmp_path / "claims.csv"
    expected = pd.DataFrame({"claim_id": ["C1", "C2"], "amount": [1, 2]})
    expected.to_csv(source, index=False)

    def fail_concat(*args, **kwargs):
        raise AssertionError("large CSV ingestion must not use pandas.concat")

    monkeypatch.setattr(pd, "concat", fail_concat)
    actual = read_large_csv(str(source), chunk_size=1, progress_bar=False)

    assert actual.shape == expected.shape
    assert actual["claim_id"].tolist() == expected["claim_id"].tolist()