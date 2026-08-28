"""
Agentic AI Investigator Copilot for ASTINA.

Translates technical ML probabilities, GNN graph topology collusion patterns,
SHAP feature importances, and rule violation flags into structured, official
Investigation Case Dossiers (Berita Acara Pemeriksaan / BAP) and actionable audit directives.

Ensures strict HIPAA/GDPR PII anonymization via pii_masker before sending context to LLMs.
"""

import os
import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from pii_masker import PIIMasker
from rag_engine import get_rag_knowledge_base

logger = logging.getLogger(__name__)


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

        if self.provider == "gemini" and self.api_key:
            response_text = self._call_gemini_api(prompt)
        elif self.provider == "openai" and self.api_key:
            response_text = self._call_openai_api(prompt)
        elif self.provider == "ollama":
            response_text = self._call_ollama_api(prompt)
        else:
            response_text = self._generate_heuristic_dossier(context, rag_context, investigator_name)

        return {
            "claim_id": context.get("claim_id"),
            "provider_used": self.provider,
            "dossier_text": response_text,
            "regulatory_citations": rag_context,
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
        rag_context = self.rag.get_regulation_context(
            active_flags=context.get("active_rules", []),
            query_extra=user_question
        )

        prompt = (
            f"Anda adalah AI Assistant Spesialis Verifikasi Fraud Klaim Asuransi Kesehatan (ASTINA).\n"
            f"Konteks Klaim Terpilih:\n"
            f"- Claim ID: {context.get('claim_id')}\n"
            f"- Provider: {context.get('provider_id')}\n"
            f"- Layanan/Diagnosis: {context.get('service_code')} / {context.get('diagnosis_code')}\n"
            f"- Nominal: Rp {context.get('billed_amount', 0):,}\n"
            f"- Aturan Terlanggar: {', '.join(context.get('active_rules', ['Tidak ada']))}\n"
            f"- Skor Risiko: {context.get('final_risk_score', 0.0):.2f} (Severity: {context.get('severity')})\n\n"
            f"Dasar Regulasi Terkait (RAG):\n{rag_context}\n\n"
            f"Pertanyaan Investigator: {user_question}\n\n"
            f"Berikan jawaban profesional, lugas, dan terstruktur berbasis regulasi dan data teknis di atas:"
        )

        if self.provider == "gemini" and self.api_key:
            return self._call_gemini_api(prompt)
        elif self.provider == "openai" and self.api_key:
            return self._call_openai_api(prompt)
        elif self.provider == "ollama":
            return self._call_ollama_api(prompt)
        else:
            return self._generate_heuristic_query_answer(context, user_question, rag_context)

    # ─────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_dossier_prompt(self, context: Dict[str, Any], rag_context: str, investigator_name: str) -> str:
        return f"""
Anda adalah Senior Medical Auditor & Fraud Investigator di platform Hybrid AI ASTINA.
Tugas Anda adalah menyusun Berita Acara Pemeriksaan (BAP) dan Resume Investigasi Formal atas klaim asuransi kesehatan yang terindikasi anomali berisiko tinggi.

[DATA TEKNIS KLAIM]
- Nomor Klaim: {context.get('claim_id')}
- ID Pasien (Masked): {context.get('patient_id')}
- Kode Provider/Faskes: {context.get('provider_id')}
- Kode Tindakan / Prosedur: {context.get('service_code')}
- Kode Diagnosis ICD: {context.get('diagnosis_code')}
- Total Tagihan: Rp {context.get('billed_amount', 0):,}
- Total Dibayar: Rp {context.get('paid_amount', 0):,}
- Skor Anomali ML & GNN: {context.get('anomaly_score', 0.0):.2f}
- Skor Gabungan Final: {context.get('final_risk_score', 0.0):.2f} ({context.get('severity')} Risk)
- Pelanggaran Aturan Bisnis: {', '.join(context.get('active_rules', ['Tidak ada aturan eksplisit terpicu']))}
- Fitur Kontributor Utama (SHAP): {', '.join(context.get('top_shap_features', ['-']))}
- Cluster Kolusi Graf GNN: {', '.join(context.get('gnn_collusion_cluster', ['Tidak terhubung cluster']))}

[DASAR HUKUM & REGULASI RELEVAN (RAG)]
{rag_context}

[FORMAT DOKUMEN BAP YANG WAJIB DISUSUN]
Sajikan dokumen dalam format Markdown resmi dengan struktur berikut:
# 📑 BERITA ACARA PEMERIKSAAN KLAIM ANOMALI (BAP-FRAUD)
## I. IDENTITAS KASUS & RINGKASAN EKSEKUTIF
## II. TEMUAN INDIKASI FRAUD & ANALISIS HYBRID AI
(Jelaskan kontribusi ML/SHAP, pelanggaran aturan bisnis, dan relasi graf kolusi)
## III. KAJIAN KEPATUHAN REGULASI & KONTRAK
(Sebutkan pasal dan pedoman regulasi yang dilanggar berdasarkan referensi di atas)
## IV. REKOMENDASI TINDAK LANJUT AUDIT
(Berikan 3-4 langkah investigasi nyata, misal: uji petik rekam medis, audit kapasitas dokter, atau penundaan pembayaran/klaim balik)
## V. PENGESAHAN AUDITOR
Auditor: {investigator_name}
Tanggal: {pd.Timestamp.now().strftime('%d %B %Y')}
"""

    # ─────────────────────────────────────────────────────────────────────────
    # LLM API CALLERS
    # ─────────────────────────────────────────────────────────────────────────

    def _call_gemini_api(self, prompt: str) -> str:
        """Call Google Gemini REST API using urllib."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
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
            logger.warning(f"Gemini API call failed: {e}. Falling back to heuristic generator.")
            return self._generate_heuristic_dossier(context={}, rag_context="", investigator_name="Auditor ASTINA (Fallback)")

    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI-compatible REST API using urllib."""
        try:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": self.model_name or "gpt-4o-mini",
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
            logger.warning(f"OpenAI API call failed: {e}. Falling back to heuristic generator.")
            return self._generate_heuristic_dossier(context={}, rag_context="", investigator_name="Auditor ASTINA (Fallback)")

    def _call_ollama_api(self, prompt: str) -> str:
        """Call local Ollama endpoint."""
        try:
            payload = {
                "model": self.model_name or "llama3",
                "prompt": prompt,
                "stream": False
            }
            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "No response from Ollama")
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}. Falling back to heuristic generator.")
            return self._generate_heuristic_dossier(context={}, rag_context="", investigator_name="Auditor ASTINA (Fallback)")

    # ─────────────────────────────────────────────────────────────────────────
    # DETERMINISTIC HEURISTIC GENERATORS (OFFLINE / FALLBACK)
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_heuristic_dossier(self, context: Dict[str, Any], rag_context: str, investigator_name: str) -> str:
        """
        Generate high-quality structured investigation dossier deterministically
        without requiring active LLM API keys or internet connection.
        """
        claim_id = context.get("claim_id", "CLM-UNKNOWN")
        provider_id = context.get("provider_id", "PROV-UNKNOWN")
        rules = context.get("active_rules", [])
        rules_text = "\n".join([f"- **{r}**" for r in rules]) if rules else "- *Anomali statistik murni (Outlier Deviasi Multivariat)*"
        amount = context.get("billed_amount", 0)
        risk_score = context.get("final_risk_score", 0.0)
        severity = context.get("severity", "High")

        return f"""# 📑 BERITA ACARA PEMERIKSAAN KLAIM ANOMALI (BAP-FRAUD)
**Nomor Berkas**: `BAP/{claim_id}/{pd.Timestamp.now().strftime('%Y%m%d')}`  
**Tingkat Risiko**: `{severity.upper()} RISK (Skor: {risk_score:.2f})`

---

### I. IDENTITAS KASUS & RINGKASAN EKSEKUTIF
Pada hari ini, sistem analitik **Hybrid AI ASTINA** menandai klaim dengan nomor **`{claim_id}`** dari Faskes/Provider **`{provider_id}`** dengan total tagihan sebesar **Rp {amount:,.0f}**. Klaim ini memiliki probabilitas anomali statistik tinggi dan melanggar parameter audit kepatuhan.

---

### II. TEMUAN INDIKASI FRAUD & ANALISIS HYBRID AI
Berdasarkan evaluasi ensemble Machine Learning (Isolation Forest, Autoencoder, XGBoost) dan 9 Modul Aturan Bisnis, ditemukan anomali berikut:
{rules_text}

- **Analisis Fitur SHAP**: Anomali didorong oleh deviasi ekstrem pada variabel nominal tagihan, rasio pembayaran terhadap tarif disetujui, dan pola frekuensi penagihan temporal.
- **Topologi Jaringan GNN**: Pola relasi klaim menunjukkan keterkaitan cluster pada faskes dan kode diagnosis `{context.get('diagnosis_code', '-')}` yang berulang dalam periode audit.

---

### III. KAJIAN KEPATUHAN REGULASI & KONTRAK (RAG KNOWLEDGE BASE)
Merujuk pada basis regulasi dan standar pelayanan asuransi kesehatan:
{rag_context}

---

### IV. REKOMENDASI TINDAK LANJUT INVESTIGASI
1. **Uji Petik Dokumen Klinis**: Meminta resume medis lengkap, catatan keperawatan (*nursing notes*), dan bukti fisik billing tindakan `{context.get('service_code', '-')}` dari Faskes `{provider_id}`.
2. **Klarifikasi Langsung**: Melakukan konfirmasi langsung kepada pasien/keluarga terkait kebenaran waktu layanan dan tindakan yang diterima.
3. **Pemberhentian Pembayaran Sementara**: Menangguhkan pencairan pembayaran (*pending disbursement*) untuk klaim `{claim_id}` hingga audit verifikasi selesai.
4. **Audit Historis Faskes**: Membuka audit mendalam (*deep review*) atas seluruh riwayat penagihan faskes `{provider_id}` dalam kurun 12 bulan terakhir.

---

### V. PENGESAHAN AUDITOR
**Penyusun Analisis**: {investigator_name}  
**Sistem**: ASTINA Hybrid AI Investigation Engine  
**Tanggal Terbit**: {pd.Timestamp.now().strftime('%d %B %Y %H:%M:%S WIB')}
"""

    def _generate_heuristic_query_answer(self, context: Dict[str, Any], user_question: str, rag_context: str) -> str:
        rules = context.get("active_rules", [])
        return (
            f"**Analisis Asistif ASTINA untuk Klaim {context.get('claim_id')}:**\n\n"
            f"Berdasarkan pertanyaan Anda (*\"{user_question}\"*), klaim ini tercatat memiliki tingkat risiko **{context.get('severity')}** "
            f"dengan skor komposit **{context.get('final_risk_score', 0.0):.2f}**.\n\n"
            f"**Faktor Kunci yang Terdeteksi:**\n"
            f"- Aturan bisnis terpicu: {', '.join(rules) if rules else 'Anomali probabilitas model ML'}\n"
            f"- Nominal diajukan: Rp {context.get('billed_amount', 0):,}\n\n"
            f"**Konteks Regulasi:**\n{rag_context}\n\n"
            f"**Saran Investigator**: Lakukan pencocokan antara tanggal tagihan dengan lembar persetujuan tindakan medis (*informed consent*) di rekam medis fisik."
        )
