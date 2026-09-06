"""
Local RAG (Retrieval-Augmented Generation) Knowledge Base Engine for ASTINA.

Indexed Regulatory Knowledge:
- Peraturan Menteri Kesehatan (Permenkes) tentang Pencegahan & Penanganan Kecurangan (Fraud) JKN
- Pedoman Verifikasi Klaim Asuransi Kesehatan & Kaidah INA-CBGs
- Batasan Kuantitas & Dosis Formularium Nasional (FORNAS)
- Panduan Clinical Pathway & Batas Wajar Rawat Inap (Length of Stay)
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss as _faiss
    _faiss_available = True
except ImportError:
    _faiss_available = False
    _faiss = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# =============================================================================
# SHARED CONSTANTS
# =============================================================================

INDONESIAN_STOPWORDS = sorted({
    "adalah","akan","akhir","akhirnya","aku","antara","apa","apakah","atau",
    "bahwa","banyak","bapak","beberapa","belum","berada","berbagai","berikan","berikut","bersama",
    "besar","bisa","bukan","buat","cukup","dan","dapat","dari","datang","dengan",
    "demikian","dia","diri","dokter","dua","dulu","empat","hal","hanya","hari",
    "hingga","ia","ibu","jadi","jika","jika","juga","jumlah","kalau","kami",
    "kamu","kasus","kecil","ke","keluar","kena","kepada","kini","kita","kota",
    "kurang","lagi","lain","lalu","lama","lebih","luar","maka","malam","masih",
    "mau","melalui","memiliki","menjadi","menurut","menyatakan","merupakan","meskipun","milik","mereka",
    "min","mungkin","mulai","nah","naik","namun","new","niat","no","nomor",
    "nyata","oleh","orang","pada","paling","para","pasti","pedoman","per","perlu",
    "pertama","perusahaan","pihak","program","pukul","saja","salah","sama","sampai","sangat",
    "satu","sedang","sedikit","see","sejak","sekarang","selalu","selama","seluruh","sementara",
    "sendiri","seorang","seperti","sering","sesuai","setelah","setiap","sisi","soal","sudah",
    "supaya","tahu","tak","tanpa","tanya","telah","tempat","tentang","terhadap","termasuk",
    "tersebut","terus","tetap","tiap","tidak","tidaklah","tiga","tinggal","tuju","turun",
    "untuk","usah","waduh","wah","walau","waktu","wanita","yaitu","yang",
    "dalam","oleh","karena","itu","ini","jika","apabila","maka","yaitu","adapun",
    "tersebut","suatu","bagi","ia","siap","manakah","bagaimanakah","bilamana","dimanakah","sedapat",
})

MIN_RAG_SIMILARITY_THRESHOLD = 0.06

# =============================================================================
# DEFAULT REGULATORY & CLINICAL KNOWLEDGE BASE
# =============================================================================

DEFAULT_KNOWLEDGE_DOCUMENTS = [
    {
        "id": "REG-001",
        "title": "Permenkes No. 16 Tahun 2019 - Pencegahan dan Penanganan Kecurangan JKN",
        "category": "Regulasi Anti-Fraud",
        "content": (
            "Pasal 3 & 4: Tindakan kecurangan (fraud) dalam Program JKN mencakup: "
            "1. Pengajuan klaim fiktif (phantom billing/service), yaitu klaim atas tindakan/obat yang tidak pernah diberikan. "
            "2. Penggelembungan tagihan (inflated bills), yaitu membebankan biaya lebih tinggi dari tarif riil. "
            "3. Pemecahan klaim tunggal menjadi beberapa tagihan terpisah (unbundling/fragmentasi). "
            "4. Penagihan berulang atas tindakan/episode yang sama (repeat billing). "
            "5. Klaim palsu atas layanan yang seharusnya satu paket tindakan (upcoding)."
        ),
        "tags": ["phantom_service", "repeat_billing", "inflated_bill", "unbundling", "upcoding", "fraud_jkn"]
    },
    {
        "id": "REG-002",
        "title": "Pedoman Penagihan Klaim Berulang (Repeat Billing & Readmission Rule)",
        "category": "Kaidah Verifikasi Klaim",
        "content": (
            "Klaim rawat jalan lanjutan atau rawat inap berulang untuk pasien yang sama dengan diagnosis primer "
            "identik dalam rentang waktu < 30 hari tanpa indikasi kegawatdaruratan baru dianggap sebagai satu episode "
            "perawatan terintegrasi. Penagihan dua kali atau lebih dalam jendela waktu 30 hari tanpa justifikasi medis "
            "merupakan pelanggaran Repeat Billing dan berpotensi menimbulkan pembayaran ganda (Duplicate Payment)."
        ),
        "tags": ["repeat_billing", "duplicate_payment", "readmission", "30_day_window"]
    },
    {
        "id": "REG-003",
        "title": "Standar Verifikasi Phantom Service & Kapasitas Tenaga Medis",
        "category": "Standar Pelayanan Medis",
        "content": (
            "1. Tanggal layanan medis wajib berada dalam rentang tanggal admisi dan discharge pasien. Layanan di luar "
            "tanggal rawat diklasifikasikan sebagai Phantom Service. "
            "2. Kapasitas harian dokter spesialis dibatasi maksimal melayani 30-40 pasien rawat jalan per hari atau "
            "tindakan bedah terencana maksimal 4-5 prosedur per hari. Klaim melebihi batas fisik logis tanpa tim "
            "pendamping terindikasi Phantom Capacity Fraud."
        ),
        "tags": ["phantom_service", "provider_capacity", "service_date", "over_utilization"]
    },
    {
        "id": "REG-004",
        "title": "Kaidah Koding INA-CBGs & Pencegahan Upcoding",
        "category": "Kaidah Koding Medis",
        "content": (
            "Kode diagnosis utama harus mencerminkan kondisi medis paling dominan yang menyebabkan pasien dirawat. "
            "Pencantuman diagnosis sekunder kompleks (komplikasi/komorbiditas) yang tidak didukung bukti resume medis, "
            "hasil laboratorium, atau intervensi terapi spesifik demi menaikkan level tarif INA-CBGs (Severity Level II/III) "
            "dikategorikan sebagai Upcoding ilegal dan wajib ditolak/direklasifikasi."
        ),
        "tags": ["upcoding", "ina_cbgs", "severity_level", "coding_fraud"]
    },
    {
        "id": "REG-005",
        "title": "Batasan Formularium Nasional (FORNAS) & Pemakaian Obat/Alkes",
        "category": "Farmasi & Alkes",
        "content": (
            "1. Peresepan obat harus sesuai dengan restriksi diagnosis, dosis harian maksimal, dan lama pemberian FORNAS. "
            "2. Klaim kuantitas obat melebihi 30 hari untuk kondisi kronis stabil, atau peresepan obat antibiotik lini tinggi "
            "tanpa kultur mikrobiologi terindikasi Medication Overutilization Fraud. "
            "3. Penggunaan implan dan alat kesehatan sekali pakai (single-use) tidak boleh ditagihkan berulang atau dikloning."
        ),
        "tags": ["medication_device_fraud", "fornas", "quantity_limit", "cloning"]
    },
    {
        "id": "REG-006",
        "title": "Standar Evaluasi Lama Rawat Inap (Length of Stay) & Readmisi",
        "category": "Kaidah Rawat Inap",
        "content": (
            "Setiap kelompok tarif INA-CBGs memiliki Standar Deviasi Lama Rawat (ALOS). Pasien yang dirawat melebihi "
            "2 kali ALOS tanpa komplikasi terdokumentasi (Prolonged Stay) atau pasien yang dipulangkan sebelum sembuh "
            "lalu dimasukkan kembali dalam < 48 jam (Premature Discharge & Rapid Readmission) wajib menjalani audit rekam medis mendalam."
        ),
        "tags": ["prolonged_stay", "readmission", "alos", "length_of_stay"]
    },
    {
        "id": "REG-007",
        "title": "Pedoman Audit Klaim Deviasi Biaya & Rasio Ekstrem (Outlier Statistik ML)",
        "category": "Kaidah Audit Deviasi Biaya & Kewajaran Tarif",
        "content": (
            "1. Klaim dengan nilai pengajuan melebihi 2 standar deviasi (+2 SD) atau persentil 95 dari tarif median "
            "kelompok diagnosis/tindakan serupa tanpa komorbiditas tercatat wajib menjalani verifikasi rincian tagihan "
            "(itemized bill audit). "
            "2. Ketidaksesuaian rasio antara biaya pengajuan (billed amount) dan tarif rujukan/klaim historis faskes "
            "mengindikasikan risiko penggelembungan biaya (inflated claim) atau distorsi billing faskes. "
            "Auditor wajib memverifikasi log tindakan sebelum menyetujui klaim."
        ),
        "tags": ["outlier", "deviasi_biaya", "statistical_anomaly", "rasio_tarif", "inflated_bill", "multivariat"]
    },
    {
        "id": "REG-008",
        "title": "Kaidah Kesesuaian Klinis Diagnosis (ICD-10) & Tindakan Medis",
        "category": "Standar Kesesuaian Koding Klinis",
        "content": (
            "Kesesuaian antara kode prosedur tindakan medis dan diagnosis utama ICD-10 merupakan syarat mutlak eligibilitas klaim. "
            "Tindakan atau prosedur kompleks yang ditagihkan untuk diagnosis ringan atau tidak berkorelasi langsung tanpa "
            "justifikasi klinis dalam resume medis diklasifikasikan sebagai Inappropriate Clinical Utilization. "
            "Auditor berwenang menunda pembayaran dan meminta lembar bukti tindakan dari DPJP."
        ),
        "tags": ["icd_10", "cpt", "service_code", "diagnosis_code", "kesesuaian_klinis", "dpjp"]
    }
]


class LocalRAGKnowledgeBase:
    """
    Lightweight, deterministic Local RAG Engine using FAISS / TF-IDF Vectorization.
    Allows instant offline semantic similarity matching against healthcare regulations.
    """

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None):
        self.documents = documents or list(DEFAULT_KNOWLEDGE_DOCUMENTS)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=INDONESIAN_STOPWORDS,
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
        )
        self.doc_texts = [
            f"{doc['title']} {doc['category']} {doc['content']} {' '.join(doc.get('tags', []))}"
            for doc in self.documents
        ]
        self._build_index()

    def rebuild_index(self) -> None:
        """Rebuild TF-IDF vectors and FAISS index after documents are mutated."""
        self.doc_texts = [
            f"{doc['title']} {doc['category']} {doc['content']} {' '.join(doc.get('tags', []))}"
            for doc in self.documents
        ]
        self._build_index()

    def add_document(self, doc: Dict[str, Any]) -> None:
        """
        Append a new regulatory/clinical document to the knowledge base and
        rebuild the vector index. Document must contain keys: id, title,
        category, content, tags (list).
        """
        if not isinstance(doc, dict):
            raise TypeError("doc must be a dict with keys: id,title,category,content,tags")
        for _req in ("id", "title", "category", "content"):
            if _req not in doc:
                raise ValueError(f"Document is missing required key '{_req}'")
        doc.setdefault("tags", [])
        if any(existing.get("id") == doc.get("id") for existing in self.documents):
            logger.warning(f"Document id={doc['id']} already exists — overwriting.")
            self.documents = [d for d in self.documents if d.get("id") != doc["id"]]
        self.documents.append(doc)
        self.rebuild_index()

    def add_documents_from_json(self, json_path: str, overwrite: bool = False) -> int:
        """
        Load additional documents from a JSON file. The file should contain a
        JSON array of document dicts. Returns the number of documents added.
        If `overwrite=True` the existing knowledge base is replaced first.
        """
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Knowledge JSON not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as _f:
            data = json.load(_f)
        if not isinstance(data, list):
            raise ValueError("JSON knowledge base root must be a list of documents.")
        if overwrite:
            self.documents = []
        _count_before = len(self.documents)
        for doc in data:
            self.add_document(doc)
        return len(self.documents) - _count_before

    def save_knowledge_base(self, json_path: str) -> None:
        """Persist current documents to a JSON file (so additions survive restarts)."""
        _dir = os.path.dirname(os.path.abspath(json_path))
        if _dir and not os.path.isdir(_dir):
            os.makedirs(_dir, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as _f:
            json.dump(self.documents, _f, ensure_ascii=False, indent=2)

    @classmethod
    def load_knowledge_base(cls, json_path: str) -> "LocalRAGKnowledgeBase":
        """Instantiate a LocalRAGKnowledgeBase from a persisted JSON file."""
        if not os.path.isfile(json_path):
            raise FileNotFoundError(json_path)
        with open(json_path, "r", encoding="utf-8") as _f:
            docs = json.load(_f)
        return cls(documents=docs)

    def _build_index(self):
        """Build TF-IDF matrix and FAISS index."""
        try:
            tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts).toarray().astype(np.float32)
            # Normalize for cosine similarity
            norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True) + 1e-10
            normalized_matrix = tfidf_matrix / norms

            self.matrix = normalized_matrix

            if FAISS_AVAILABLE:
                dim = normalized_matrix.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(normalized_matrix)
            else:
                self.index = None
                logger.info("FAISS not available; using cosine dot-product fallback.")
        except Exception as e:
            logger.error(f"Error building RAG index: {e}", exc_info=True)
            self.matrix = np.zeros((len(self.documents), 10), dtype=np.float32)
            self.index = None

    def retrieve(self, query: str, top_k: int = 2, min_similarity: float = MIN_RAG_SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most relevant regulatory documents for a given query string.
        Documents whose similarity score falls below `min_similarity` are dropped
        so low-confidence matches never mislead the downstream LLM or auditor.

        As a safety fallback for small knowledge bases or short/ambiguous queries,
        if *no* document survives the threshold filter, the single highest-scoring
        document is returned regardless of score — this prevents the downstream
        ``get_regulation_context`` from ever returning an empty string on KB that
        only has a handful of seeded documents.
        """
        if not query or not query.strip():
            return list(self.documents[:top_k])

        try:
            query_vec = self.vectorizer.transform([query]).toarray().astype(np.float32)
            norm = np.linalg.norm(query_vec) + 1e-10
            query_vec = query_vec / norm

            raw_scores: Optional[np.ndarray] = None
            best_idx: int = -1
            best_score: float = -1.0

            if FAISS_AVAILABLE and self.index is not None:
                scores, indices = self.index.search(query_vec, min(top_k * 2, len(self.documents)))
                retrieved_docs: List[Dict[str, Any]] = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.documents):
                        continue
                    score_f = float(score)
                    if score_f > best_score:
                        best_score = score_f
                        best_idx = int(idx)
                    if score_f < min_similarity:
                        continue
                    doc = dict(self.documents[idx])
                    doc["similarity_score"] = score_f
                    retrieved_docs.append(doc)
                    if len(retrieved_docs) >= top_k:
                        break
            else:
                # Cosine similarity fallback
                sims = np.dot(self.matrix, query_vec.T).flatten()
                raw_scores = sims
                # argsort descending, then filter by threshold and cap to top_k
                sorted_idx = list(np.argsort(-sims))
                retrieved_docs = []
                for idx in sorted_idx:
                    score_f = float(sims[idx])
                    if score_f < min_similarity:
                        break
                    doc = dict(self.documents[int(idx)])
                    doc["similarity_score"] = score_f
                    retrieved_docs.append(doc)
                    if len(retrieved_docs) >= top_k:
                        break

            # ── Fallback: Nothing passed threshold → return the single best doc ──
            if not retrieved_docs and len(self.documents) > 0:
                best_idx_arr: int = 0
                best_score_out: float = 0.0
                if raw_scores is not None:
                    best_idx_arr = int(np.argmax(raw_scores))
                    best_score_out = float(raw_scores[best_idx_arr])
                elif best_idx >= 0:
                    best_idx_arr = best_idx
                    best_score_out = best_score if best_score >= 0 else 0.0
                else:
                    best_idx_arr = 0
                    best_score_out = 0.0
                doc = dict(self.documents[best_idx_arr])
                doc["similarity_score"] = float(best_score_out)
                retrieved_docs.append(doc)

            return retrieved_docs

        except Exception as e:
            logger.error(f"Error in RAG retrieval: {e}", exc_info=True)
            return list(self.documents[:top_k])

    def get_regulation_context(self, active_flags: List[str], query_extra: str = "") -> str:
        """
        Synthesize formatted regulatory text to inject into the LLM Investigator Copilot prompt.
        """
        if active_flags:
            search_terms = " ".join(active_flags) + " " + query_extra
        else:
            # When no deterministic rule triggered, combine the generic ML-outlier
            # keywords (to pull statistical/clinical-audit documents) with the
            # claim's own service_code + diagnosis_code for clinical specificity.
            base_outlier_terms = "deviasi biaya anomali statistik outlier multivariat kewajaran tarif kesesuaian klinis"
            search_terms = f"{base_outlier_terms} {query_extra}".strip()
        
        matched_docs = self.retrieve(search_terms, top_k=2)

        if not matched_docs:
            return "Tidak ditemukan referensi regulasi spesifik."

        context_lines = []
        for i, doc in enumerate(matched_docs, 1):
            score_str = f" (Relevansi: {doc.get('similarity_score', 0.0):.2f})" if "similarity_score" in doc else ""
            context_lines.append(
                f"[{i}] {doc['title']} ({doc['category']}){score_str}:\n{doc['content']}"
            )
        return "\n\n".join(context_lines)


# Singleton instance
_rag_instance: Optional[LocalRAGKnowledgeBase] = None

def get_rag_knowledge_base() -> LocalRAGKnowledgeBase:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LocalRAGKnowledgeBase()
    return _rag_instance
