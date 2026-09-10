import pandas as pd
import pytest

from cache_manager import get_file_hash
from config import MAX_EXCEL_FILE_SIZE
from file_handler import (
    ingest_file_to_raw_parquet,
    read_file_with_optimization,
    read_large_csv,
    remove_duplicates_from_parquet,
    stream_csv_to_parquet,
)


class UploadedBytes:
    def __init__(self, payload, name="claims.xlsx", reported_size=None):
        import io

        self._buffer = io.BytesIO(payload)
        self.name = name
        self.size = len(payload) if reported_size is None else reported_size

    def seek(self, position):
        return self._buffer.seek(position)

    def tell(self):
        return self._buffer.tell()

    def read(self, size=-1):
        return self._buffer.read(size)


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


def test_excel_upload_is_buffered_once_and_converted_to_parquet(tmp_path):
    source_df = pd.DataFrame({
        "claim_id": ["C1", "C2"],
        "amount": [100.5, 250.0],
        "status": ["PAID", "PENDING"],
    })
    excel_buffer = __import__("io").BytesIO()
    source_df.to_excel(excel_buffer, index=False, engine="openpyxl")
    uploaded_file = UploadedBytes(excel_buffer.getvalue())

    output, row_count, schema = ingest_file_to_raw_parquet(uploaded_file, "xlsx")

    actual = pd.read_parquet(output)
    assert row_count == 2
    assert schema["columns"] == ["claim_id", "amount", "status"]
    assert actual["claim_id"].tolist() == ["C1", "C2"]


def test_excel_upload_over_limit_fails_before_parsing():
    uploaded_file = UploadedBytes(
        b"not-a-real-workbook",
        reported_size=MAX_EXCEL_FILE_SIZE + 1,
    )

    with pytest.raises(ValueError, match="dibatasi 100MB"):
        read_file_with_optimization(uploaded_file, "xlsx")


def test_sniff_semicolon_csv_streams_to_parquet(tmp_path):
    from file_handler import sniff_csv_dialect

    source = tmp_path / "claims.csv"
    source.write_text("claim_id;amount\nC1;10.5\nC2;20.0\n", encoding="utf-8")
    dialect = sniff_csv_dialect(str(source))
    assert dialect["separator"] == ";"

    output, row_count = stream_csv_to_parquet(
        str(source),
        output_path=str(tmp_path / "claims.parquet"),
        chunk_size=2,
        progress_bar=False,
    )
    actual = pd.read_parquet(output)
    assert row_count == 2
    assert list(actual.columns) == ["claim_id", "amount"]
    assert actual["claim_id"].tolist() == ["C1", "C2"]


def test_csv_gz_ingest_to_parquet(tmp_path):
    import gzip

    payload = b"claim_id,amount\nC1,11\nC2,22\n"
    gz_bytes = gzip.compress(payload)
    uploaded_file = UploadedBytes(gz_bytes, name="claims.csv.gz")

    output, row_count, schema = ingest_file_to_raw_parquet(uploaded_file, "gz")
    actual = pd.read_parquet(output)
    assert row_count == 2
    assert schema["columns"] == ["claim_id", "amount"]
    assert actual["claim_id"].tolist() == ["C1", "C2"]


def test_parquet_ingest_writes_once(tmp_path):
    source_df = pd.DataFrame({"claim_id": ["C1", "C2"], "amount": [1.5, 2.5]})
    buffer = __import__("io").BytesIO()
    source_df.to_parquet(buffer, index=False)
    uploaded_file = UploadedBytes(buffer.getvalue(), name="claims.parquet")

    output, row_count, schema = ingest_file_to_raw_parquet(uploaded_file, "parquet")
    actual = pd.read_parquet(output)
    assert row_count == 2
    assert actual["claim_id"].tolist() == ["C1", "C2"]
    assert schema["total_rows"] == 2


def test_check_ingest_resources_rejects_insufficient_disk(monkeypatch):
    from file_handler import check_ingest_resources

    class Usage:
        free = 1024
        used = 1
        total = 2048

    monkeypatch.setattr("file_handler.shutil.disk_usage", lambda _path: Usage())
    with pytest.raises(ValueError, match="Ruang disk"):
        check_ingest_resources(100 * 1024 * 1024)


def test_latin1_csv_ingest(tmp_path):
    source = tmp_path / "claims.csv"
    source.write_bytes("claim_id,note\nC1,caf\xe9\n".encode("latin-1"))
    output, row_count = stream_csv_to_parquet(
        str(source),
        output_path=str(tmp_path / "latin.parquet"),
        progress_bar=False,
    )
    actual = pd.read_parquet(output)
    assert row_count == 1
    assert "caf" in str(actual["note"].iloc[0]).lower()