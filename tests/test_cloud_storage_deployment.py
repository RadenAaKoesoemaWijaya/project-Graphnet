import pytest

import cloud_storage


def test_production_storage_requires_bucket(monkeypatch):
    monkeypatch.setenv("ASTINA_ENVIRONMENT", "production")
    monkeypatch.delenv("GOOGLE_CLOUD_BUCKET", raising=False)

    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_BUCKET"):
        cloud_storage.validate_production_storage_configuration()


def test_production_storage_checks_write_access(monkeypatch):
    events = []

    class FakeBlob:
        def upload_from_string(self, payload):
            events.append(("upload", payload))

        def delete(self):
            events.append(("delete",))

    class FakeBucket:
        def blob(self, name):
            events.append(("blob", name))
            return FakeBlob()

    class FakeClient:
        def bucket(self, name):
            assert name == "astina-models"
            return FakeBucket()

    monkeypatch.setenv("ASTINA_ENVIRONMENT", "production")
    monkeypatch.setenv("GOOGLE_CLOUD_BUCKET", "astina-models")
    monkeypatch.setenv("GOOGLE_CLOUD_BUCKET_PREFIX", "production/models")
    monkeypatch.setattr(cloud_storage, "_GCS_AVAILABLE", True)
    monkeypatch.setattr(cloud_storage, "_client", FakeClient)

    cloud_storage.validate_production_storage_configuration()

    assert events[0][0] == "blob"
    assert events[0][1].startswith("production/models/.healthchecks/")
    assert events[1:] == [("upload", b""), ("delete",)]


def test_production_model_persistence_fails_if_upload_is_incomplete(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model")
    monkeypatch.setenv("ASTINA_ENVIRONMENT", "production")
    monkeypatch.setattr(cloud_storage, "is_enabled", lambda: True)
    monkeypatch.setattr(cloud_storage, "upload_files", lambda paths: 0)

    with pytest.raises(RuntimeError, match="Persistensi model tidak lengkap"):
        cloud_storage.sync_artefacts_after_save([str(model_path)])
