from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("astina.claim_status")

CLAIM_KEY_COLUMNS = ("patient_id", "provider_id", "service_code")


class ClaimStatusValidator:
    """Validate claim payment status against historical records with safe DB fallback."""

    def __init__(self, db_connection=None, timeout_seconds: float = 5.0, max_retries: int = 3):
        self.db_connection = db_connection
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def check_duplicate_payment(self, claim: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]], str]:
        if not claim:
            return False, [], "Claim kosong."

        if self.db_connection is None:
            return False, [], "Validasi pembayaran tidak tersedia: koneksi database belum dikonfigurasi."
        history = self._query_claim_history(claim)
        return self.evaluate_duplicate_payment(claim, history)

    @staticmethod
    def claim_key(claim: Dict[str, Any]) -> Tuple[str, str, str] | None:
        values = tuple(str(claim.get(column) or "").strip() for column in CLAIM_KEY_COLUMNS)
        return values if all(values) else None

    @staticmethod
    def evaluate_duplicate_payment(
        claim: Dict[str, Any],
        history: Iterable[Dict[str, Any]],
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        history = list(history)
        if not history:
            return False, [], "Tidak ada riwayat pembayaran yang cocok."

        current_claim_id = str(claim.get("claim_id", "")).strip()
        comparable = [
            item for item in history
            if not current_claim_id or str(item.get("claim_id", "")).strip() != current_claim_id
        ]
        paid_matches = [
            item for item in comparable
            if str(item.get("status", "")).upper() == "PAID"
        ]
        if paid_matches:
            return True, paid_matches, "Riwayat pembayaran PAID ditemukan untuk klaim serupa."
        return False, comparable, "Riwayat klaim ditemukan, tetapi belum ada pembayaran PAID duplikat."

    def get_claim_history_batch(
        self,
        claims: Iterable[Dict[str, Any]],
        batch_size: int = 200,
    ) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
        """Fetch matching histories in bounded queries instead of one query per claim."""
        if batch_size < 1:
            raise ValueError("batch_size harus lebih besar dari 0.")
        if self.db_connection is None:
            return {}

        keys = list(dict.fromkeys(
            key for claim in claims if (key := self.claim_key(claim)) is not None
        ))
        histories: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {
            key: [] for key in keys
        }
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start:start + batch_size]
            row_placeholders = ", ".join(["(%s, %s, %s)"] * len(batch_keys))
            query = (
                "SELECT claim_id, patient_id, provider_id, service_code, status "
                "FROM claim_history WHERE (patient_id, provider_id, service_code) "
                f"IN ({row_placeholders})"
            )
            parameters = tuple(value for key in batch_keys for value in key)
            try:
                result = self.db_connection.execute(query, parameters)
                rows = result.fetchall() if callable(getattr(result, "fetchall", None)) else list(result or [])
                description = getattr(result, "description", None)
                column_names = [
                    getattr(column, "name", None) or column[0]
                    for column in description
                ] if description else None
                for row in rows:
                    if isinstance(row, dict):
                        record = row
                    elif hasattr(row, "_mapping"):
                        record = dict(row._mapping)
                    elif column_names:
                        record = dict(zip(column_names, row))
                    else:
                        raise TypeError(
                            "Koneksi database harus mengembalikan mapping atau cursor "
                            "dengan metadata kolom."
                        )
                    key = self.claim_key(record)
                    if key in histories:
                        histories[key].append(record)
            except Exception:
                logger.exception("Batch query riwayat pembayaran gagal.")
                raise
        return histories

    def check_pending_claims(self, claim: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]], str]:
        history = self._query_claim_history(claim)
        pending = [item for item in history if item.get("status") == "PENDING"]
        if pending:
            return True, pending, "Terdapat klaim pending yang mirip."
        return False, [], "Tidak ada klaim pending yang cocok."

    def _query_claim_history(self, claim: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.db_connection is None:
            return []

        key = self.claim_key(claim)
        if key is None:
            return []
        return self.get_claim_history_batch([claim]).get(key, [])
