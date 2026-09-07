import pandas as pd

from cache_manager import get_file_hash
from file_handler import read_large_csv, remove_duplicates_from_parquet, stream_csv_to_parquet


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


def test_remove_duplicates_from_parquet_preserves_order_without_pandas(tmp_path):
    source = tmp_path / "claims.parquet"
    expected = pd.DataFrame({
        "claim_id": ["C1", "C1", "C2"],
        "amount": [10, 10, 20],
    })
    expected.to_parquet(source, index=False)

    output, metadata = remove_duplicates_from_parquet(
        str(source),
        output_path=str(tmp_path / "deduplicated.parquet"),
        subset=["claim_id"],
    )

    actual = pd.read_parquet(output)
    assert actual.to_dict("records") == [
        {"claim_id": "C1", "amount": 10},
        {"claim_id": "C2", "amount": 20},
    ]
    assert metadata["duplicates_removed"] == 1


def test_upload_cache_hash_distinguishes_same_name_and_size():
    class UploadedFile:
        def __init__(self, payload):
            import io

            self._buffer = io.BytesIO(payload)
            self.name = "claims.csv"
            self.size = len(payload)

        def seek(self, position):
            return self._buffer.seek(position)

        def tell(self):
            return self._buffer.tell()

        def read(self, size=-1):
            return self._buffer.read(size)

    first = UploadedFile(b"claim_id,amount\nC1,10\n")
    second = UploadedFile(b"claim_id,amount\nC2,20\n")
    assert first.size == second.size
    assert get_file_hash(first) != get_file_hash(second)