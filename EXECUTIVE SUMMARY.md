# ASTINA — Executive Summary
**Analisis Sistem Transaksi Identifikasi Nilai Anomali**
*Enterprise AI Platform for Healthcare Insurance Fraud Detection & Clinical-Financial Integrity*

---

## 1. Executive Problem Statement

Kecurangan klaim asuransi kesehatan (Healthcare Insurance Fraud, Waste, and Abuse / FWA) menimbulkan kerugian triliunan rupiah setiap tahun bagi penyelenggara jaminan sosial kesehatan dan asuransi swasta. Pola kecurangan modern tidak lagi bersifat acak, melainkan terorganisir rapi dalam bentuk:
1. **Sindikat Medis Terstruktur (*Fraud Rings*)**: Kolusi faskes, oknum dokter, dan peserta fiktif.
2. **Klaim Fiktif (*Phantom Services*)**: Penagihan tindakan atau obat yang tidak pernah diberikan.
3. **Pemisahan & Penggelembungan Tagihan (*Upcoding & Unbundling*)**: Manipulasi kode tarif INA-CBGs / ICD-10.
4. **Klaim Berulang (*Repeat Billing*)**: Penagihan berulang atas kunjungan medis atau tindakan yang sama.

Metode audit manual atau aturan statis konvensional (rule-based) terbukti lambat, mudah dihindari (*evasion*), dan menghasilkan tingkat alarm palsu (*false positive*) yang sangat tinggi.

---

## 2. The ASTINA Solution

**ASTINA** hadir sebagai solusi intelijen audit generasi baru yang menggabungkan:
- **Hybrid AI Detection Engine**: Memadukan kekuatan statistik Machine Learning (Isolation Forest, Deep Autoencoder, XGBoost) dengan **Graph Neural Network (GNN)** untuk membongkar jaringan kolusi relasional faskes dan pasien.
- **Deterministic 9-Module Business Rules**: Memvalidasi kepatuhan aturan medis dan regulasi BPJS/Kemenkes secara seketika (*real-time*) dengan akselerasi Polars/PyArrow.
- **Agentic AI Investigator Copilot & RAG**: Membantu auditor menghasilkan Berita Acara Pemeriksaan (BAP) formal berbasis sitasi aturan ICD-10/CPT secara otomatis dalam hitungan detik.
- **Enterprise Data Governance**: Kepatuhan penuh terhadap **UU Perlindungan Data Pribadi (UU PDP No. 27/2022)** dan **HIPAA Security Rule** melalui Dynamic PII Masking dan Audit Trail Kriptografis SHA-256.

---

## 3. Business Impact & ROI

| Parameter Kinerja | Metode Konvensional | Platform ASTINA | Peningkatan Nilai |
| :--- | :--- | :--- | :---: |
| **Kecepatan Analisis Klaim** | 3 - 7 hari / batch klaim | < 2 menit / 100.000 klaim | **> 98% Lebih Cepat** |
| **Deteksi Sindikat Kolusi** | Sangat sulit (manual cross-check) | Otomatis via GNN Graph Ring | **Pola Kompleks Terdeteksi** |
| **Akurasi & Presisi Audit** | Banyak False Positive | Terkalibrasi Hybrid Noisy-OR | **Reduksi Beban Verifikator 60%** |
| **Kepatuhan Regulasi** | Rentan kebocoran NIK/Medis | Role-Based Dynamic PII Masking | **100% Kepatuhan UU PDP** |
| **Akuntabilitas Forensik** | Log teks biasa yang mudah diubah | Tamper-Proof SHA-256 Audit Trail | **Bukti Hukum Terverifikasi** |

---

## 4. Production Readiness Status

- **Status Saat Ini**: 🟢 **Staging & Pilot Ready (v2.4.0)**
- **Audit Kesiapan Internal**:
  - Security: **PASS** (PBKDF2-HMAC-SHA256, RBAC, PII Masking, Rate Limiting).
  - Data Scalability: **PASS** (Streaming disk-buffering hingga 3GB per file dengan RAM < 1GB).
  - Machine Learning & GNN: **PASS** (Contract isolation, memory bounded subgraph).
  - Infrastructure: **PASS** (Docker multi-stage, Google Cloud Run & Cloud Storage ready).
- **Rekomendasi Penerapan Operasional**:
  Dapat segera dioperasikan untuk tim auditor investigasi dalam mode *single-team / staging review*, dengan jalur migrasi bertahap menuju arsitektur *distributed worker* untuk pemrosesan transaksi massal nasional.
