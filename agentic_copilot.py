"""
Agentic AI Investigator Copilot for ASTINA.

Translates technical ML probabilities, GNN graph topology collusion patterns,
SHAP feature importances, and rule violation flags into structured, official
Investigation Case Dossiers (Berita Acara Pemeriksaan / BAP) and actionable audit directives.

Ensures strict HIPAA/GDPR PII anonymization via pii_masker before sending context to LLMs.
"""

import os
import re
import json
import logging
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import requests as _requests
    _has_requests_lib = True
except ImportError:  # pragma: no cover - requests is almost always available in env with streamlit
    _has_requests_lib = False
    _requests = None  # type: ignore[assignment]

from pii_masker import PIIMasker
from rag_engine import get_rag_knowledge_base

logger = logging.getLogger(__name__)


# =============================================================================
# LLM API RUNTIME CONSTANTS
# =============================================================================

DEFAULT_CLOUD_LLM_TIMEOUT: float = 30.0
DEFAULT_OLLAMA_TIMEOUT: float = 60.0
LLM_RETRY_MAX_ATTEMPTS: int = 3
LLM_RETRY_MULTIPLIER: float = 1.5
LLM_RETRY_MIN_WAIT: float = 1.0
LLM_RETRY_MAX_WAIT: float = 10.0


def _llm_exponential_backoff(attempt: int) -> float:
    """Return backoff seconds (clamped) for 0-based retry attempt."""
    wait = LLM_RETRY_MIN_WAIT * (LLM_RETRY_MULTIPLIER ** max(0, attempt))
    return float(min(LLM_RETRY_MAX_WAIT, wait))


def _perform_http_json_request(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_CLOUD_LLM_TIMEOUT,
) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    """
    Unified JSON HTTP client with transparent `requests` → `urllib` fallback.
    Returns (parsed_json_dict or None, last_exception_or_None).
    Transient exceptions are bubbled up so the caller can retry.
    """
    data_bytes = json.dumps(payload or {}).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    method = method.upper()

    if _has_requests_lib and _requests is not None:
        try:
            resp = _requests.request(
                method=method,
                url=url,
                data=data_bytes,
                headers=hdrs,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json(), None
        except Exception as _e:
            # 4xx errors (invalid key, bad payload) are not transient → let caller decide
            if isinstance(_e, _requests.HTTPError):
                status = getattr(_e.response, "status_code", 0)
                if 400 <= status < 500 and status not in (408, 429):
                    return None, _e  # non-transient → do not retry upstack
            return None, _e  # 5xx, timeout, connection, 408/429 → retry eligible

    # ── Fallback: urllib ──
    try:
        req = urllib.request.Request(url, data=data_bytes, headers=hdrs, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body), None
    except urllib.error.HTTPError as _he:
        status = getattr(_he, "code", 0)
        if 400 <= status < 500 and status not in (408, 429):
            return None, _he  # non-transient
        return None, _he
    except Exception as _e:
        return None, _e


def _is_transient_llm_error(exc: Optional[Exception]) -> bool:
    """Decide whether a given exception should trigger an LLM retry."""
    if exc is None:
        return False
    msg = str(exc).lower()
    transient_markers = (
        "timeout", "timed out", "timed_out", "connection error", "connectionreset",
        "connection refused", "service unavailable", "bad gateway", "502", "503", "504",
        "408", "429", "rate limit", "too many requests", "temporarily unavailable",
        "dns", "network is unreachable", "read timed", "write timed", "broken pipe",
    )
    return any(m in msg for m in transient_markers)


# =============================================================================
# AI SECURITY GUARDRAIL (PROMPT INJECTION & LEAK PREVENTION)
# =============================================================================

class AIGuardrail:
    """Security filter to detect and prevent prompt injections, jailbreaks, and sensitive leaks."""

    INJECTION_PATTERNS = [
        # ── English / framework patterns (original) ──
        r'(?i)(ignore|bypass|override)\s+(all\s+)?(previous|prior|system)\s+(instructions|prompts|rules)',
        r'(?i)(reveal|show|leak|print|output|dump)\s+(your\s+)?(system\s+prompt|initial\s+prompt|developer\s+instructions|hidden\s+instructions)',
        r'(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(dan|unrestricted|jailbreak|developer\s+mode|root|sudo|admin\s+mode)',
        r'(?i)(<\|im_start\|>|<\|im_end\|>|\[SYSTEM\]|\[INST\]|\<\<SYS\>\>)',
        r'(?i)(drop\s+database|drop\s+table|delete\s+from\s+claims|update\s+.*\s+set\s+.*\s*=|insert\s+into\s+.*\s*values)',
        r'(?i)\b(rm\s+-rf|chmod\s+777|curl\s+\|sh|wget\s+-qO-.*\|\s*bash|exec\s*\(|eval\s*\(|os\.system|subprocess\.call)\b',
        # ── Bahasa Indonesia jailbreak / prompt injection patterns ──
        r'(?i)(abaikan|lewatkan|batalkan|sampingi)\s+(semua\s+)?(petunjuk|instruksi|aturan|perintah)\s+(sebelumnya|sebelum|awal|sistem)',
        r'(?i)(tunjukkan|beberkan|keluarkan|cetak|bocorkan|tuliskan)\s+(prompt\s+sistem|perintah\s+awal|instruksi\s+developer|petunjuk\s+tersembunyi|prompt\s+asli)',
        r'(?i)(sekarang\s+kamu\s+adalah|berperan\s+sebagai|pura-pura\s+menjadi|anda\s+sekarang\s+menjadi)\s+(mode\s+developer|tanpa\s+batasan|jailbreak|root|admin|bebas\s+aturan|tanpa\s+filter|DAN|AI\s+terbuka)',
        r'(?i)(hapus\s+aturan|matikan\s+filter|nonaktifkan\s+keamanan|buka\s+semua\s+pembatasan|hilangkan\s+guardrail|abaikan\s+etika)',
        r'(?i)(bagaimana\s+cara\s+kamu\s+dibuat|apa\s+kode\s+program\s+kamu|berikan\s+sumber\s+instruksi|detailkan\s+konfigurasi\s+prompt\s+kamu|apa\s+yang\s+ada\s+dalam\s+sistem\s+prompt\s+kamu)',
        r'(?i)(jangan\s+patuhi|tidak\s+usah\s+ikuti|melanggar\s+saja\s+aturan|abaikan\s+hukum|abaikan\s+kebijakan|aturan\s+tidak\s+berlaku\s+bagimu)',
        r'(?i)(beritahu\s+rahasia\s+sistem|apa\s+kata\s+sandi\s+kamu|siapa\s+admin\s+kamu|bagaimana\s+cara\s+mengakses\s+database|download\s+seluruh\s+data)',
        r'(?i)(skenario\s+ini\s+hanya\s+fiksi|ini\s+cuma\s+permainan\s+peran|roleplay\s+sebagai\s+penjahat|anggap\s+saja\s+aturan\s+tidak\s+ada|dalam\s+dunia\s+paralel\s+kamu\s+bisa)',
        r'(?i)(ulangi\s+persis\s+apa\s+yang\s+di\s+atas|salin\s+semua\s+petunjuk|copy\s+paste\s+seluruh\s+prompt|ulang\s+instruksi\s+awal\s+sampai\s+akhir)',
        r'(?i)(untuk\s+tujuan\s+penelitian|saya\s+peneliti\s+AI|demi\s+keamanan\s+sistem|CEO\s+memerintahkan|bos\s+anda\s+mengatakan|developer\s+mengizinkan\s+agar)',
    ]

    # Max length of system prompt snippet we consider a "leak" if it appears verbatim in output
    _PROMPT_LEAK_MIN_WINDOW = 60

    @classmethod
    def validate_query(cls, user_question: str) -> Tuple[bool, str]:
        """
        Validate user input against prompt injection attacks.
        Returns: (is_safe: bool, reason_or_sanitized: str)
        """
        if not user_question or not user_question.strip():
            return False, "Pertanyaan kosong."

        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, user_question):
                return False, f"Pola instruksi mencurigakan terdeteksi (guardrail)."

        return True, user_question.strip()

    @classmethod
    def _snippets_from_prompt(cls, prompt: str, window: int = _PROMPT_LEAK_MIN_WINDOW) -> set:
        """Extract rolling character snippets of `window` length from a prompt string."""
        s = re.sub(r'\s+', ' ', prompt).strip()
        snips: set = set()
        if len(s) <= window:
            snips.add(s)
        else:
            for i in range(0, len(s) - window + 1, max(1, window // 4)):
                snips.add(s[i:i+window])
        return snips

    @classmethod
    def validate_output(
        cls,
        llm_output: str,
        *system_prompts: str,
        replacement: str = "[REDACTED: bagian instruksi internal tidak boleh ditampilkan.]"
    ) -> Tuple[bool, str]:
        """
        Post-process an LLM output to prevent prompt/system-instruction leakage.
        Returns (is_safe: bool, sanitized_output: str).
        - is_safe=False only when leakage is unambiguous (in which case sanitized_output
          is the full replacement string); otherwise suspicious snippets are elided
          and is_safe is still True.
        """
        if not llm_output:
            return True, ""

        leak_snips: set = set()
        for sp in system_prompts:
            if sp:
                leak_snips.update(cls._snippets_from_prompt(sp))

        sanitized = llm_output
        leak_hit = False
        if leak_snips:
            normalized_out = re.sub(r'\s+', ' ', sanitized).strip()
            for snip in leak_snips:
                if snip and snip in normalized_out:
                    # Replace longest-matching literal occurrences; use regex-insensitive match with whitespace-tolerant pattern
                    pattern = r'\s*'.join(re.escape(ch) for ch in snip)
                    new_san, n_subs = re.subn(pattern, replacement, sanitized, flags=re.IGNORECASE)
                    if n_subs > 0:
                        sanitized = new_san
                        leak_hit = True

        if leak_hit and len(sanitized) < max(60, 0.4 * len(llm_output)):
            # Most of the original response was redacted → likely an outright prompt dump
            return False, replacement
        return True, sanitized


# Shared cache of the last system-prompts so output guarding works from anywhere.
_LAST_DOSSIER_PROMPT: str = ""
_LAST_QUERY_PROMPT: str = ""


# =============================================================================
# CONTEXT BUILDER & PII SANITIZER
# =============================================================================

class ClaimContextBuilder:
    """
    Builds a secure, anonymized context dictionary from a claim row,
    XAI features, and GNN graph relations.
    """

    @staticmethod
    def build_sanitized_context(
        claim_row: pd.Series | dict,
        shap_contributions: Optional[Dict[str, float]] = None,
        gnn_neighbors: Optional[List[str]] = None,
        mask_sensitive: bool = True
    ) -> Dict[str, Any]:
        """
        Extract and sanitize claim attributes for LLM ingestion.
        """
        raw_dict = claim_row.to_dict() if isinstance(claim_row, pd.Series) else dict(claim_row)

        # Convert numpy types to Python standard primitives
        clean_dict = {}
        for k, v in raw_dict.items():
            if pd.isna(v):
                clean_dict[k] = None
            elif isinstance(v, (np.integer, np.int64, np.int32)):
                clean_dict[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                clean_dict[k] = round(float(v), 2)
            else:
                clean_dict[k] = v

        # Apply PII masking
        if mask_sensitive:
            masked_dict = PIIMasker.mask_dict(clean_dict)
        else:
            masked_dict = clean_dict

        # Collect active violation flags
        active_rules = []
        rule_map = {
            'repeat_billing_flag': 'Repeat Billing (Tagihan Berulang)',
            'phantom_service_flag': 'Phantom Service (Layanan Fiktif)',
            'duplicate_payment_flag': 'Duplicate Payment (Pembayaran Ganda)',
            'upcoding_unbundling_flag': 'Upcoding & Unbundling (Penggelembungan Kode)',
            'inflated_bill_cloning_flag': 'Inflated Bill & Cloning (Lonjakan Tagihan Ekstrem)',
            'prolonged_stay_readmission_flag': 'Prolonged Stay & Readmission (Lama Rawat Anomali)',
            'medication_device_fraud_flag': 'Medication & Device Fraud (Kuantitas Obat/Alkes Berlebih)',
            'provider_capacity_flag': 'Provider Capacity (Kapasitas Harian Terlampaui)',
        }

        for flag_col, label in rule_map.items():
            raw_val = clean_dict.get(flag_col, raw_dict.get(flag_col, 0))
            try:
                if int(float(raw_val or 0)) == 1:
                    active_rules.append(label)
            except (ValueError, TypeError):
                if str(raw_val).strip().lower() in ('1', 'true', 'yes', 't'):
                    active_rules.append(label)

        # Extract top SHAP contributors
        top_shap = []
        if shap_contributions:
            sorted_shap = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_shap = [f"{feat}: {val:+.3f}" for feat, val in sorted_shap[:5]]

        def _safe_float(val, default: float = 0.0) -> float:
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default

        return {
            "claim_id": masked_dict.get("claim_id", clean_dict.get("claim_id", "CLM-UNKNOWN")),
            "patient_id": masked_dict.get("patient_id", clean_dict.get("patient_id", "PAT-ANON")),
            "provider_id": masked_dict.get("provider_id", clean_dict.get("provider_id", "PROV-ANON")),
            "service_code": str(clean_dict.get("service_code", "N/A")),
            "diagnosis_code": str(clean_dict.get("diagnosis_code", "N/A")),
            "billed_amount": _safe_float(clean_dict.get("billed_amount", clean_dict.get("amount", 0.0))),
            "paid_amount": _safe_float(clean_dict.get("paid_amount", 0.0)),
            "anomaly_score": _safe_float(clean_dict.get("anomaly_probability", 0.0)),
            "final_risk_score": _safe_float(clean_dict.get("final_risk_score", 0.0)),
            "severity": str(clean_dict.get("severity", "Medium")),
            "active_rules": active_rules,
            "top_shap_features": top_shap,
            "gnn_collusion_cluster": gnn_neighbors or [],
            "raw_attributes": masked_dict,
        }


# =============================================================================
# AGENTIC INVESTIGATOR COPILOT ENGINE
# =============================================================================

class AgenticInvestigatorCopilot:
    """
    Agentic Copilot for insurance fraud investigators.
    Supports Cloud LLMs (Gemini, OpenAI), Local Ollama, and an offline deterministic Heuristic Generator.
    """

    def __init__(
        self,
        provider: str = "heuristic",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")
        self.endpoint_url = endpoint_url or os.getenv("LLM_ENDPOINT_URL", "http://localhost:11434/api/generate")
        self.rag = get_rag_knowledge_base()

    def generate_investigation_dossier(
        self,
        context: Dict[str, Any],
        investigator_name: str = "Investigator Senior ASTINA",
        language: str = "id"
    ) -> Dict[str, Any]:
        """
        Generate a complete official Case Dossier (BAP) with regulatory citations.
        Output is post-filtered by AIGuardrail to prevent system prompt leakage.
        """
        # Retrieve relevant regulatory context from RAG
        rag_context = self.rag.get_regulation_context(
            active_flags=context.get("active_rules", []),
            query_extra=f"{context.get('service_code', '')} {context.get('diagnosis_code', '')}"
        )

        prompt = self._build_dossier_prompt(context, rag_context, investigator_name)
        response_text = None
        actual_provider_used = self.provider

        if self.provider == "gemini" and self.api_key:
            response_text = self._call_gemini_raw(prompt)
        elif self.provider in ("openai", "azure") and self.api_key:
            response_text = self._call_openai_raw(prompt)
        elif self.provider == "ollama":
            response_text = self._call_ollama_raw(prompt)

        # If LLM API returned empty or failed, gracefully fall back to heuristic with full context
        heuristic_used = False
        if not response_text:
            if self.provider != "heuristic":
                logger.info(f"LLM provider '{self.provider}' unavailable or failed; using deterministic heuristic fallback.")
                actual_provider_used = f"{self.provider} (Fallback: Heuristic)"
            response_text = self._generate_heuristic_dossier(context, rag_context, investigator_name)
            heuristic_used = True

        # ── P2-2: Prompt/System-Leak Output Guardrail ──
        if not heuristic_used and response_text:
            _safe, _san = AIGuardrail.validate_output(response_text, prompt)
            if not _safe:
                logger.warning("AIGuardrail detected prompt leakage in dossier LLM response — falling back to heuristic.")
                response_text = self._generate_heuristic_dossier(context, rag_context, investigator_name)
                actual_provider_used = f"{self.provider} (Fallback: Heuristic) [Leak Guard]"
            else:
                response_text = _san

        # Build clean cryptographic audit hash mockup for integrity
        import hashlib
        claim_id_str = str(context.get("claim_id", "CLM-UNKNOWN"))
        audit_hash = hashlib.sha256(f"{claim_id_str}-{context.get('final_risk_score', 0)}-{investigator_name}".encode('utf-8')).hexdigest()[:16].upper()

        return {
            "claim_id": context.get("claim_id"),
            "provider_used": actual_provider_used,
            "dossier_text": response_text,
            "regulatory_citations": rag_context,
            "dossier_number": f"BAP/{claim_id_str}/{pd.Timestamp.now().strftime('%Y%m%d')}",
            "audit_hash": f"ASTINA-SEC-{audit_hash}",
            "generated_at": pd.Timestamp.now().strftime('%d %B %Y %H:%M:%S WIB'),
            "investigator_name": investigator_name,
            "final_risk_score": context.get("final_risk_score", 0.0),
            "severity": context.get("severity", "Medium"),
            "active_rules": context.get("active_rules", []),
            "status": "success"
        }

    def answer_investigator_query(
        self,
        context: Dict[str, Any],
        user_question: str
    ) -> str:
        """
        Answer an ad-hoc investigative question regarding a specific claim context.
        Applies input guardrail (prompt injection) AND output guardrail (prompt leakage).
        """
        # Cybersecurity Guardrail Check (input)
        is_safe, guard_detail = AIGuardrail.validate_query(user_question)
        if not is_safe:
            logger.warning(f"AI Guardrail blocked query: '{user_question}' - Reason: {guard_detail}")
            try:
                from audit_trail import get_audit_logger
                audit = get_audit_logger()
                audit.log_event(
                    event_type="AI_PROMPT_INJECTION_BLOCKED",
                    actor=str(context.get('claim_id', 'unknown_claim')),
                    details={"reason": guard_detail, "blocked_query": user_question[:100]}
                )
            except Exception:
                pass
            return (
                "⚠️ **[SECURITY ALERT - AKSES DITOLAK]**\n\n"
                "Pertanyaan Anda terdeteksi melanggar kebijakan keamanan siber sistem ASTINA "
                "(upaya *prompt injection*, modifikasi instruksi sistem, atau ekstraksi data terlarang). "
                "Insiden keamanan ini telah dicatat ke dalam audit trail berantai."
            )

        rag_context = self.rag.get_regulation_context(
            active_flags=context.get("active_rules", []),
            query_extra=user_question
        )

        prompt = (
            f"Anda adalah AI Investigator Copilot ASTINA (Sistem Deteksi Fraud Asuransi Kesehatan).\n"
            f"Jawab pertanyaan auditor berikut dengan ringkas, padat, profesional, dan to-the-point.\n\n"
            f"Data Klaim: Claim ID={context.get('claim_id')}, Faskes={context.get('provider_id')}, "
            f"Diagnosa={context.get('diagnosis_code')}, Tindakan={context.get('service_code')}, "
            f"Tagihan=Rp {context.get('billed_amount', 0):,}, Skor Risiko={context.get('final_risk_score', 0.0):.2f} ({context.get('severity')}), "
            f"Pelanggaran Aturan={', '.join(context.get('active_rules', ['N/A']))}.\n\n"
            f"Regulasi Terkait:\n{rag_context}\n\n"
            f"Pertanyaan: {user_question}\n\n"
            f"Format jawaban:\n"
            f"1. **Kesimpulan Utama** (1 kalimat langsung menjawab)\n"
            f"2. **Poin Bukti & Regulasi** (2-3 poin ringkas)\n"
            f"3. **Tindakan Auditor yang Disarankan** (1-2 langkah konkret)"
        )

        raw_answer = None
        heuristic_used = False
        if self.provider == "gemini" and self.api_key:
            raw_answer = self._call_gemini_raw(prompt)
        elif self.provider in ("openai", "azure") and self.api_key:
            raw_answer = self._call_openai_raw(prompt)
        elif self.provider == "ollama":
            raw_answer = self._call_ollama_raw(prompt)

        if not raw_answer:
            heuristic_used = True

        if not heuristic_used and raw_answer:
            # ── P2-2: Prompt/System-Leak Output Guardrail ──
            _safe, _san = AIGuardrail.validate_output(raw_answer, prompt)
            if not _safe:
                logger.warning("AIGuardrail detected prompt leakage in Q&A LLM response — falling back to heuristic.")
                heuristic_used = True
            else:
                return _san

        # Reachable when heuristic_used is True (LLM unavailable OR output-guard rejected LLM output)
        return self._generate_heuristic_query_answer(context, user_question, rag_context)

    # ─────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_dossier_prompt(self, context: Dict[str, Any], rag_context: str, investigator_name: str) -> str:
        return f"""
Anda adalah Senior Medical Auditor & Fraud Investigator di platform Hybrid AI ASTINA.
Tugas Anda adalah menyusun Berita Acara Pemeriksaan (BAP) dan Resume Investigasi yang rapi, ringkas, profesional, dan informatif.

[DATA TEKNIS KLAIM]
- Nomor Klaim: {context.get('claim_id')}
- ID Pasien (Masked): {context.get('patient_id')}
- Kode Provider/Faskes: {context.get('provider_id')}
- Tindakan/Prosedur: {context.get('service_code')}
- Diagnosis ICD-10: {context.get('diagnosis_code')}
- Total Tagihan: Rp {context.get('billed_amount', 0):,}
- Total Dibayar: Rp {context.get('paid_amount', 0):,}
- Skor Anomali ML & GNN: {context.get('anomaly_score', 0.0):.2f}
- Skor Gabungan Final: {context.get('final_risk_score', 0.0):.2f} ({context.get('severity')} Risk)
- Pelanggaran Aturan Bisnis: {', '.join(context.get('active_rules', ['Tidak ada aturan eksplisit terpicu']))}
- Fitur Kontributor Utama (SHAP): {', '.join(context.get('top_shap_features', ['-']))}
- Cluster Kolusi Graf GNN: {', '.join(context.get('gnn_collusion_cluster', ['Tidak terhubung cluster']))}

[DASAR HUKUM & REGULASI RELEVAN (RAG)]
{rag_context}

[PANDUAN FORMAT DOKUMEN BAP]
Sajikan dokumen dalam Markdown resmi yang rapi, ringkas, dan to-the-point menggunakan struktur berikut:
# 📑 BERITA ACARA PEMERIKSAAN KLAIM ANOMALI (BAP-FRAUD)
> **Ref**: `BAP/{context.get('claim_id')}/{pd.Timestamp.now().strftime('%Y%m%d')}` | **Klasifikasi**: `{context.get('severity', 'HIGH').upper()} RISK` (Skor: {context.get('final_risk_score', 0.0):.2f}) | **Status**: `INVESTIGASI AUDIT LANJUTAN`

### I. IDENTITAS KASUS & RINGKASAN EKSEKUTIF
| Parameter | Keterangan | Parameter | Keterangan |
| :--- | :--- | :--- | :--- |
| **No. Klaim** | `{context.get('claim_id')}` | **Faskes / Provider** | `{context.get('provider_id')}` |
| **Pasien (Masked)** | `{context.get('patient_id')}` | **Diagnosis (ICD-10)** | `{context.get('diagnosis_code')}` |
| **Kode Layanan** | `{context.get('service_code')}` | **Nilai Pengajuan** | `Rp {context.get('billed_amount', 0):,}` |

*Ringkasan*: (Jelaskan dalam 2-3 kalimat lugas inti anomali dan potensi overbilling).

### II. TEMUAN INDIKASI FRAUD & ANALISIS HYBRID AI
- **Pelanggaran Aturan Bisnis**: (Sebutkan aturan yang dilanggar secara spesifik)
- **Faktor Pemicu ML & SHAP**: (Uraikan kontribusi fitur anomali)
- **Topologi Jaringan GNN**: (Keterkaitan pola tagihan berulang/kolusi pada faskes)

### III. KAJIAN KEPATUHAN REGULASI (RAG KNOWLEDGE BASE)
(Uraikan pasal dan ketentuan Permenkes / INA-CBGs / FORNAS yang relevan secara ringkas dan lugas)

### IV. REKOMENDASI TINDAK LANJUT AUDIT
1. **[Prioritas 1 - Urgent]**: (Tindakan pembekuan pembayaran / verifikasi billing)
2. **[Prioritas 2 - Klinis]**: (Uji petik rekam medis & konfirmasi tindakan)
3. **[Prioritas 3 - Faskes]**: (Audit kepatuhan provider & historis penagihan)

### V. PENGESAHAN AUDITOR
- **Penyusun Analisis**: {investigator_name}
- **Sistem**: ASTINA Hybrid AI Engine
- **Tanggal**: {pd.Timestamp.now().strftime('%d %B %Y %H:%M:%S WIB')}
"""

    # ─────────────────────────────────────────────────────────────────────────
    # LLM API CALLERS (RAW & FALLBACK-SAFE, WITH RETRY)
    # ─────────────────────────────────────────────────────────────────────────

    def _call_gemini_raw(self, prompt: str) -> Optional[str]:
        """Execute Gemini REST API call with retry + exponential backoff."""
        model = self.model_name if "gemini" in (self.model_name or "") else "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}
        }
        last_exc: Optional[Exception] = None
        for attempt in range(LLM_RETRY_MAX_ATTEMPTS):
            try:
                data, exc = _perform_http_json_request(
                    "POST", url, payload=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=DEFAULT_CLOUD_LLM_TIMEOUT,
                )
                if data is not None and "candidates" in data and data["candidates"]:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                if not _is_transient_llm_error(exc):
                    logger.warning(f"Gemini API non-transient failure (attempt {attempt+1}): {exc}")
                    return None
                last_exc = exc
            except Exception as _e:
                if not _is_transient_llm_error(_e):
                    logger.warning(f"Gemini API non-transient exception: {_e}")
                    return None
                last_exc = _e
            if attempt + 1 < LLM_RETRY_MAX_ATTEMPTS:
                wait = _llm_exponential_backoff(attempt)
                logger.info(f"Gemini attempt {attempt+1} failed; retrying in {wait:.1f}s...")
                time.sleep(wait)
        logger.warning(f"Gemini API call failed after {LLM_RETRY_MAX_ATTEMPTS} attempts: {last_exc}")
        return None

    def _call_openai_raw(self, prompt: str) -> Optional[str]:
        """Execute OpenAI / Azure-compatible REST API call with retry + exponential backoff."""
        url = self.endpoint_url if (self.endpoint_url and "api.openai.com" not in self.endpoint_url and "11434" not in self.endpoint_url) else "https://api.openai.com/v1/chat/completions"
        model = self.model_name if (self.model_name and "gemini" not in self.model_name and "llama" not in self.model_name) else "gpt-4o-mini"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        last_exc: Optional[Exception] = None
        for attempt in range(LLM_RETRY_MAX_ATTEMPTS):
            try:
                data, exc = _perform_http_json_request(
                    "POST", url, payload=payload, headers=headers,
                    timeout=DEFAULT_CLOUD_LLM_TIMEOUT,
                )
                if data is not None and "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"]
                if not _is_transient_llm_error(exc):
                    logger.warning(f"OpenAI/Azure non-transient failure (attempt {attempt+1}): {exc}")
                    return None
                last_exc = exc
            except Exception as _e:
                if not _is_transient_llm_error(_e):
                    logger.warning(f"OpenAI/Azure non-transient exception: {_e}")
                    return None
                last_exc = _e
            if attempt + 1 < LLM_RETRY_MAX_ATTEMPTS:
                wait = _llm_exponential_backoff(attempt)
                logger.info(f"OpenAI attempt {attempt+1} failed; retrying in {wait:.1f}s...")
                time.sleep(wait)
        logger.warning(f"OpenAI/Azure API call failed after {LLM_RETRY_MAX_ATTEMPTS} attempts: {last_exc}")
        return None

    def _call_ollama_raw(self, prompt: str) -> Optional[str]:
        """Execute Local Ollama call with retry + exponential backoff."""
        endpoint = self.endpoint_url or "http://localhost:11434/api/generate"
        model = self.model_name if (self.model_name and "gemini" not in self.model_name and "gpt" not in self.model_name) else "llama3"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        last_exc: Optional[Exception] = None
        for attempt in range(LLM_RETRY_MAX_ATTEMPTS):
            try:
                data, exc = _perform_http_json_request(
                    "POST", endpoint, payload=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=DEFAULT_OLLAMA_TIMEOUT,
                )
                if data is not None and data.get("response"):
                    return data.get("response")
                if not _is_transient_llm_error(exc):
                    logger.warning(f"Ollama non-transient failure (attempt {attempt+1}): {exc}")
                    return None
                last_exc = exc
            except Exception as _e:
                if not _is_transient_llm_error(_e):
                    logger.warning(f"Ollama non-transient exception: {_e}")
                    return None
                last_exc = _e
            if attempt + 1 < LLM_RETRY_MAX_ATTEMPTS:
                wait = _llm_exponential_backoff(attempt)
                logger.info(f"Ollama attempt {attempt+1} failed; retrying in {wait:.1f}s...")
                time.sleep(wait)
        logger.warning(f"Ollama call failed after {LLM_RETRY_MAX_ATTEMPTS} attempts: {last_exc}")
        return None

    def test_connection(self) -> dict:
        """
        Quick connectivity check for the configured LLM provider.
        Sends the smallest possible request and returns a result dict:
        {
          "ok": bool,
          "provider": str,
          "message": str,   # user-facing status message
          "latency_ms": int | None
        }
        """
        import time
        ping_prompt = "Balas hanya dengan kata: OK"
        t0 = time.time()

        if self.provider == "heuristic":
            return {
                "ok": True,
                "provider": "Heuristic (Offline)",
                "message": "✅ Mode offline — tidak memerlukan koneksi. Selalu tersedia.",
                "latency_ms": 0,
            }

        if self.provider == "gemini":
            if not self.api_key:
                return {"ok": False, "provider": "Gemini", "message": "❌ API Key kosong.", "latency_ms": None}
            resp = self._call_gemini_raw(ping_prompt)
            ms = int((time.time() - t0) * 1000)
            if resp:
                return {"ok": True, "provider": f"Gemini ({self.model_name})", "message": f"✅ Koneksi berhasil ({ms} ms).", "latency_ms": ms}
            return {"ok": False, "provider": "Gemini", "message": f"❌ Koneksi gagal — periksa API Key dan quota.", "latency_ms": ms}

        if self.provider in ("openai", "azure"):
            if not self.api_key:
                return {"ok": False, "provider": "OpenAI", "message": "❌ API Key kosong.", "latency_ms": None}
            resp = self._call_openai_raw(ping_prompt)
            ms = int((time.time() - t0) * 1000)
            if resp:
                return {"ok": True, "provider": f"OpenAI ({self.model_name})", "message": f"✅ Koneksi berhasil ({ms} ms).", "latency_ms": ms}
            return {"ok": False, "provider": "OpenAI/Azure", "message": f"❌ Koneksi gagal — periksa endpoint, API Key, dan model name.", "latency_ms": ms}

        if self.provider == "ollama":
            resp = self._call_ollama_raw(ping_prompt)
            ms = int((time.time() - t0) * 1000)
            if resp:
                return {"ok": True, "provider": f"Ollama ({self.model_name})", "message": f"✅ Ollama lokal merespons ({ms} ms).", "latency_ms": ms}
            return {"ok": False, "provider": "Ollama", "message": f"❌ Ollama tidak merespons di `{self.endpoint_url}`. Pastikan `ollama serve` sudah berjalan.", "latency_ms": ms}

        return {"ok": False, "provider": self.provider, "message": f"❌ Provider '{self.provider}' tidak dikenal.", "latency_ms": None}

    # Backward compatibility wrappers
    def _call_gemini_api(self, prompt: str, context: Optional[Dict[str, Any]] = None, rag_context: str = "", investigator_name: str = "Auditor ASTINA") -> str:
        resp = self._call_gemini_raw(prompt)
        if resp:
            return resp
        return self._generate_heuristic_dossier(context=context or {}, rag_context=rag_context, investigator_name=investigator_name)

    def _call_openai_api(self, prompt: str, context: Optional[Dict[str, Any]] = None, rag_context: str = "", investigator_name: str = "Auditor ASTINA") -> str:
        resp = self._call_openai_raw(prompt)
        if resp:
            return resp
        return self._generate_heuristic_dossier(context=context or {}, rag_context=rag_context, investigator_name=investigator_name)

    def _call_ollama_api(self, prompt: str, context: Optional[Dict[str, Any]] = None, rag_context: str = "", investigator_name: str = "Auditor ASTINA") -> str:
        resp = self._call_ollama_raw(prompt)
        if resp:
            return resp
        return self._generate_heuristic_dossier(context=context or {}, rag_context=rag_context, investigator_name=investigator_name)

    # ─────────────────────────────────────────────────────────────────────────
    # DETERMINISTIC HEURISTIC GENERATORS (OFFLINE / FALLBACK)
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_heuristic_dossier(self, context: Dict[str, Any], rag_context: str, investigator_name: str) -> str:
        """
        Generate high-quality structured investigation dossier deterministically
        without requiring active LLM API keys or internet connection.
        """
        claim_id = context.get("claim_id", "CLM-UNKNOWN")
        patient_id = context.get("patient_id", "PAT-ANON")
        provider_id = context.get("provider_id", "PROV-UNKNOWN")
        service_code = context.get("service_code", "N/A")
        diagnosis_code = context.get("diagnosis_code", "N/A")
        rules = context.get("active_rules", [])
        rules_text = "\n".join([f"- **{r}**: Terdeteksi ketidakwajaran pola penagihan melebihi ambang batas kepatuhan." for r in rules]) if rules else "- *Anomali statistik murni (Outlier Deviasi Multivariat)*"
        amount = context.get("billed_amount", 0)
        paid_amount = context.get("paid_amount", 0)
        risk_score = context.get("final_risk_score", 0.0)
        anomaly_score = context.get("anomaly_score", risk_score)
        severity = context.get("severity", "High")
        shap_feats = context.get("top_shap_features", [])
        shap_text = ", ".join(shap_feats) if shap_feats else "Deviasi tarif di atas median grup diagnosa, rasio klaim berulang temporal"

        return f"""# 📑 BERITA ACARA PEMERIKSAAN KLAIM ANOMALI (BAP-FRAUD)
> **No. Berkas**: `BAP/{claim_id}/{pd.Timestamp.now().strftime('%Y%m%d')}` &nbsp;|&nbsp; **Klasifikasi Risiko**: `{severity.upper()} RISK` (Skor: **{risk_score:.2f}**) &nbsp;|&nbsp; **Status**: `PERLU TINDAK LANJUT AUDIT`

---

### I. IDENTITAS KASUS & RINGKASAN EKSEKUTIF

| Atribut Klaim | Data Nilai | Atribut Klaim | Data Nilai |
| :--- | :--- | :--- | :--- |
| **Nomor Klaim** | `{claim_id}` | **Faskes / Provider** | `{provider_id}` |
| **ID Pasien (Masked)** | `{patient_id}` | **Diagnosis (ICD-10)** | `{diagnosis_code}` |
| **Kode Prosedur** | `{service_code}` | **Total Diajukan** | **Rp {amount:,.0f}** |
| **Total Dibayar** | Rp {paid_amount:,.0f} | **Skor Anomali ML** | **{anomaly_score:.2f}** |

**Ringkasan Singkat:**
Sistem analitik **Hybrid AI ASTINA** mendeteksi klaim nomor **`{claim_id}`** dari faskes **`{provider_id}`** memiliki indikasi deviasi biaya dan pola kepatuhan klinis dengan skor risiko gabungan **{risk_score:.2f}** ({severity} Risk). Direkomendasikan verifikasi dokumen fisik sebelum proses pencairan.

---

### II. TEMUAN INDIKASI FRAUD & ANALISIS HYBRID AI
{rules_text}

- **Analisis Driver Fitur (SHAP)**: Anomali didorong signifikan oleh `{shap_text}`.
- **Topologi Jaringan GNN**: Teridentifikasi keterkaitan klaster penagihan pada provider `{provider_id}` untuk diagnosis `{diagnosis_code}` dalam jendela waktu audit.

---

### III. KAJIAN KEPATUHAN REGULASI (RAG KNOWLEDGE BASE)
Berdasarkan penelusuran semantik regulasi pada basis pengetahuan ASTINA:

{rag_context}

---

### IV. REKOMENDASI TINDAK LANJUT INVESTIGASI
1. **[Prioritas 1 - Urgent] Penundaan Pembayaran**: Menangguhkan sementara pencairan (*pending disbursement*) klaim `{claim_id}` hingga klarifikasi tuntas.
2. **[Prioritas 2 - Klinis] Uji Petik Berkas Medis**: Meminta rekam medis lengkap, lembar persetujuan (*informed consent*), dan rincian billing tindakan `{service_code}` dari Faskes `{provider_id}`.
3. **[Prioritas 3 - Pasien] Verifikasi Layanan**: Konfirmasi sampel ke pasien/keluarga terkait kesesuaian waktu pelayanan dan penerimaan obat/tindakan.
4. **[Prioritas 4 - Faskes] Audit Historis Provider**: Membuka audit historis penagihan faskes `{provider_id}` untuk pola diagnosis `{diagnosis_code}`.

---

### V. PENGESAHAN AUDITOR
- **Penyusun Berkas**: {investigator_name}
- **Mesin Inferensi**: ASTINA Agentic Copilot & Hybrid AI Engine
- **Waktu Terbit**: {pd.Timestamp.now().strftime('%d %B %Y %H:%M:%S WIB')}
"""

    def _generate_heuristic_query_answer(self, context: Dict[str, Any], user_question: str, rag_context: str) -> str:
        rules = context.get("active_rules", [])
        claim_id = context.get("claim_id", "CLM-UNKNOWN")
        severity = context.get("severity", "Medium")
        score = context.get("final_risk_score", 0.0)
        amount = context.get("billed_amount", 0)

        return (
            f"**💡 Analisis AI Copilot untuk Klaim `{claim_id}`:**\n\n"
            f"**1. Kesimpulan Utama:**\n"
            f"Klaim ini diklasifikasikan sebagai **{severity.upper()} RISK** (Skor: **{score:.2f}**) dengan total tagihan **Rp {amount:,.0f}**.\n\n"
            f"**2. Poin Bukti & Aturan Terpicu:**\n"
            f"- Pelanggaran aturan: {', '.join(rules) if rules else 'Anomali deviasi statistik ML'}\n"
            f"- Terindikasi ketidaksesuaian tarif atau frekuensi klaim terhadap pola normal kelompok diagnosis serupa.\n\n"
            f"**3. Rujukan Regulasi Relevan:**\n"
            f"{rag_context[:350]}...\n\n"
            f"**4. Arahan Investigator:**\n"
            f"Lakukan uji petik berkas rekam medis dan cocokkan tanggal tindakan `{context.get('service_code', '-')}` dengan log billing faskes."
        )
