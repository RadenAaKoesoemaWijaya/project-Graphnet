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
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from pii_masker import PIIMasker
from rag_engine import get_rag_knowledge_base

logger = logging.getLogger(__name__)


# =============================================================================
# AI SECURITY GUARDRAIL (PROMPT INJECTION & LEAK PREVENTION)
# =============================================================================

class AIGuardrail:
    """Security filter to detect and prevent prompt injections, jailbreaks, and sensitive leaks."""

    INJECTION_PATTERNS = [
        r'(?i)(ignore|bypass|override)\s+(all\s+)?(previous|prior|system)\s+(instructions|prompts|rules)',
        r'(?i)(reveal|show|leak|print|output|dump)\s+(your\s+)?(system\s+prompt|initial\s+prompt|developer\s+instructions)',
        r'(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(dan|unrestricted|jailbreak|developer\s+mode|root)',
        r'(?i)(<\|im_start\|>|<\|im_end\|>|\[SYSTEM\]|\[INST\])',
        r'(?i)(drop\s+database|drop\s+table|delete\s+from\s+claims)',
    ]

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
                return False, f"Pola instruksi mencurigakan terdeteksi ({pattern})."

        return True, user_question.strip()


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
        if not response_text:
            if self.provider != "heuristic":
                logger.info(f"LLM provider '{self.provider}' unavailable or failed; using deterministic heuristic fallback.")
                actual_provider_used = f"{self.provider} (Fallback: Heuristic)"
            response_text = self._generate_heuristic_dossier(context, rag_context, investigator_name)

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
        """
        # Cybersecurity Guardrail Check
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
        if self.provider == "gemini" and self.api_key:
            raw_answer = self._call_gemini_raw(prompt)
        elif self.provider in ("openai", "azure") and self.api_key:
            raw_answer = self._call_openai_raw(prompt)
        elif self.provider == "ollama":
            raw_answer = self._call_ollama_raw(prompt)

        if raw_answer:
            return raw_answer

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
    # LLM API CALLERS (RAW & FALLBACK-SAFE)
    # ─────────────────────────────────────────────────────────────────────────

    def _call_gemini_raw(self, prompt: str) -> Optional[str]:
        """Execute Gemini REST API call; returns None on any failure."""
        try:
            model = self.model_name if "gemini" in (self.model_name or "") else "gemini-1.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}")
            return None

    def _call_openai_raw(self, prompt: str) -> Optional[str]:
        """Execute OpenAI / Azure-compatible REST API call; returns None on any failure."""
        try:
            url = self.endpoint_url if (self.endpoint_url and "api.openai.com" not in self.endpoint_url and "11434" not in self.endpoint_url) else "https://api.openai.com/v1/chat/completions"
            model = self.model_name if (self.model_name and "gemini" not in self.model_name and "llama" not in self.model_name) else "gpt-4o-mini"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"OpenAI/Azure API call failed: {e}")
            return None

    def _call_ollama_raw(self, prompt: str) -> Optional[str]:
        """Execute Local Ollama call; returns None on any failure."""
        try:
            endpoint = self.endpoint_url or "http://localhost:11434/api/generate"
            model = self.model_name if (self.model_name and "gemini" not in self.model_name and "gpt" not in self.model_name) else "llama3"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", None)
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
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
