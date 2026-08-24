from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("astina.claim_status")


class ClaimStatusValidator:
    """Validate claim payment status against historical records with safe DB fallback."""

    def __init__(self, db_connection=None, timeout_seconds: float = 5.0, max_retries: int = 3):
        self.db_connection = db_connection
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def check_duplicate_payment(self, claim: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]], str]:
        if not claim:
            return False, [], "Claim kosong."

        history = self._query_claim_history(claim)
        if not history:
            if self.db_connection is None:
                return False, [], "Validasi pembayaran tidak tersedia: koneksi database belum dikonfigurasi."
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

    def check_pending_claims(self, claim: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]], str]:
        history = self._query_claim_history(claim)
        pending = [item for item in history if item.get("status") == "PENDING"]
        if pending:
            return True, pending, "Terdapat klaim pending yang mirip."
        return False, [], "Tidak ada klaim pending yang cocok."

    def _query_claim_history(self, claim: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.db_connection is None:
            return []

        try:
            patient_id = claim.get("patient_id")
            provider_id = claim.get("provider_id")
            service_code = claim.get("service_code")
            if not patient_id and not provider_id and not service_code:
                return []

            result = self.db_connection.execute(
                "SELECT claim_id, patient_id, provider_id, service_code, status FROM claim_history WHERE patient_id = %s AND provider_id = %s AND service_code = %s",
                (patient_id, provider_id, service_code),
            )
            return list(result or [])
        except Exception as exc:  # pragma: no cover
            logger.warning("Database query failed in claim status validator: %s", exc)
            return []
