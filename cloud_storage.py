"""Cloud Storage adapter for ASTINA model artefacts.

The default ``CombinedAnomalyDetector.save_models`` / ``load_models`` write
to the local filesystem. On Cloud Run the only writable directory is
``/tmp`` and any local model is lost on instance restart. This module
adds an opt-in shim that mirrors the local files to Google Cloud Storage
when two environment variables are present:

* ``GOOGLE_CLOUD_BUCKET`` — bucket name (without ``gs://`` prefix).
* ``GOOGLE_CLOUD_BUCKET_PREFIX`` — optional key prefix, defaults to
  ``models``. Useful for separating dev/staging/prod paths.

If either variable is missing the adapter becomes a no-op and all
operations stay local. The adapter is also robust to ``google-cloud-
storage`` not being installed at runtime — it logs a single warning and
falls back to local-only.
"""
import io
import logging
import os
import shutil
import tempfile
from typing import Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("graphnet.storage")


_GCS_AVAILABLE: Optional[bool] = None


def _gcs_available() -> bool:
    """Lazy probe for the optional dependency."""
    global _GCS_AVAILABLE
    if _GCS_AVAILABLE is None:
        try:
            import google.cloud.storage  # noqa: F401
            _GCS_AVAILABLE = True
        except Exception as e:  # pragma: no cover - import guard
            logger.warning(
                "google-cloud-storage not available (%s). "
                "Models will stay on local filesystem only.",
                e,
            )
            _GCS_AVAILABLE = False
    return _GCS_AVAILABLE


def _bucket_name() -> Optional[str]:
    return os.environ.get("GOOGLE_CLOUD_BUCKET") or None


def _prefix() -> str:
    return (os.environ.get("GOOGLE_CLOUD_BUCKET_PREFIX") or "models").strip("/")


def is_enabled() -> bool:
    """``True`` when GCS mirroring is active for this process."""
    return bool(_bucket_name()) and _gcs_available()


def _client():
    """Return an authenticated GCS client (uses ADC on Cloud Run)."""
    import google.cloud.storage as gcs
    return gcs.Client()


def upload_files(local_paths: Iterable[str], destination_prefix: Optional[str] = None) -> int:
    """Upload files to GCS, preserving the local sub-path under the prefix.

    Returns the number of files uploaded. Silently no-ops when the
    adapter is not enabled.
    """
    if not is_enabled():
        return 0
    bucket = _client().bucket(_bucket_name())
    prefix = (destination_prefix or _prefix()).strip("/")
    
    # Helper function for single file upload
    def upload_single_file(local):
        if not local or not os.path.exists(local):
            return 0
        # Preserve the basename under the prefix; collisions overwrite.
        key = f"{prefix}/{os.path.basename(local)}" if not prefix else f"{prefix}/{os.path.basename(local)}"
        try:
            blob = bucket.blob(key)
            blob.upload_from_filename(local)
            return 1
        except Exception as e:
            logger.warning("GCS upload failed for %s: %s", local, e)
            return 0
    
    # Use parallel uploads for multiple files
    local_paths_list = list(local_paths)
    if len(local_paths_list) <= 1:
        # Sequential for single file
        n = upload_single_file(local_paths_list[0]) if local_paths_list else 0
    else:
        # Parallel for multiple files (max 4 workers to avoid overwhelming GCS)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(upload_single_file, local_paths_list))
        n = sum(results)
    
    if n:
        logger.info("Mirrored %d artefact(s) to gs://%s/%s", n, _bucket_name(), prefix)
    return n


def download_files(names: Iterable[str], local_dir: str,
                   source_prefix: Optional[str] = None) -> List[str]:
    """Download the given basenames from GCS into ``local_dir``.

    Returns the list of local paths that were successfully materialised.
    """
    if not is_enabled():
        return []
    os.makedirs(local_dir, exist_ok=True)
    bucket = _client().bucket(_bucket_name())
    prefix = (source_prefix or _prefix()).strip("/")
    paths: List[str] = []
    for name in names:
        key = f"{prefix}/{os.path.basename(name)}" if prefix else os.path.basename(name)
        local = os.path.join(local_dir, os.path.basename(name))
        try:
            blob = bucket.blob(key)
            if not blob.exists():
                continue
            blob.download_to_filename(local)
            paths.append(local)
        except Exception as e:
            logger.warning("GCS download failed for %s: %s", key, e)
    return paths


def sync_artefacts_after_save(local_paths: Iterable[str]) -> None:
    """Convenience hook: upload after a successful local save."""
    if not is_enabled():
        return
    upload_files(local_paths)


def ensure_artefacts_loaded(local_dir: str, basenames: Iterable[str]) -> None:
    """Download any missing basenames into ``local_dir`` before load."""
    if not is_enabled():
        return
    os.makedirs(local_dir, exist_ok=True)
    needed = []
    for name in basenames:
        local = os.path.join(local_dir, name)
        if not os.path.exists(local):
            needed.append(name)
    if needed:
        download_files(needed, local_dir)


def upload_large_dataset_stream(local_file_path: str, blob_name: Optional[str] = None) -> Optional[str]:
    """Upload large dataset to GCS bucket using chunked streaming (useful for 2GB+ files)."""
    if not is_enabled():
        return None
    try:
        bucket = _client().bucket(_bucket_name())
        blob_key = f"datasets/{blob_name or os.path.basename(local_file_path)}"
        blob = bucket.blob(blob_key)
        # Use 16MB chunk size for high-throughput upload
        blob.chunk_size = 16 * 1024 * 1024
        blob.upload_from_filename(local_file_path)
        gcs_uri = f"gs://{_bucket_name()}/{blob_key}"
        logger.info("Uploaded large dataset to %s", gcs_uri)
        return gcs_uri
    except Exception as e:
        logger.warning("GCS dataset stream upload failed for %s: %s", local_file_path, e)
        return None


def generate_signed_upload_url(blob_name: str, expiration_minutes: int = 30) -> Optional[str]:
    """Generate a GCS Resumable Signed URL for direct client-to-bucket upload."""
    if not is_enabled():
        return None
    try:
        import datetime
        bucket = _client().bucket(_bucket_name())
        blob_key = f"direct_uploads/{blob_name}"
        blob = bucket.blob(blob_key)
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=expiration_minutes),
            method="PUT",
            content_type="application/octet-stream"
        )
        return url
    except Exception as e:
        logger.warning("Failed to generate GCS signed upload URL: %s", e)
        return None

