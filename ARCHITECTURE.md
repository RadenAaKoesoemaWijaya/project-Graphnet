# ASTINA Enterprise Architecture Documentation
**Analisis Sistem Transaksi Identifikasi Nilai Anomali (ASTINA)**
*Version: 2.4.0 (Enterprise Production Architecture)*
*Standards Compliance: UU No. 27/2022 (UU PDP), HIPAA Security Rule, OWASP Top 10 2021*

---

## 1. System Overview

ASTINA adalah platform analitik dan investigasi anomali transaksi klaim asuransi kesehatan berbasis **Hybrid AI Enterprise**. Platform ini memadukan kekuatan **Machine Learning Ensemble** berbobot dinamis dengan **9 Modul Aturan Bisnis Medis Tervektorisasi**, **Agentic AI Investigator Copilot bertenaga RAG (FAISS)**, serta **Audit Trail Kriptografis SHA-256**.

```
+-----------------------------------------------------------------------------------+
|                                  USER / CLIENT                                    |
|             (Web Browser: Streamlit UI / Headless CLI Batch Automation)           |
+-----------------------------------------------------------------------------------+
                                         |
                                  HTTPS / WSS (TLS 1.3)
                                         v
+-----------------------------------------------------------------------------------+
|                           SECURITY & AUTH GATEWAY                                 |
|  - PBKDF2-HMAC-SHA256 Auth & Dynamic Salt Generation                              |
|  - RBAC Enforcement (Admin, Auditor, Analyst, Viewer)                             |
|  - Brute-Force Lockout (5 attempts / 15 mins) & Quota Rate Limiter                |
|  - Dynamic PII Masking (NIK, Nama, Rekam Medis) & Audit Mode                      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        INGESTION & HARMONIZATION ENGINE                           |
|  - Resilient Multi-Format (CSV, CSV.GZ, Parquet, XLSX) Streaming Parser          |
|  - Disk-Buffering Chunk Ingestion (< 1 GB RAM Footprint)                          |
|  - Semantic Schema Harmonizer & Indonesian Aliasing                               |
|  - Deterministic Feature Derivation (LOS, Bill Ratio, Age)                        |
|  - Preprocessing Isolation Contract (Version 1, Train-Only Partition Fit)         |
+-----------------------------------------------------------------------------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
+---------------------------------------+ +-----------------------------------------+
|        HYBRID DETECTION ENGINE        | |       RULE-BASED BUSINESS ENGINE        |
| - Isolation Forest (Statistical)      | | 1. Repeat Billing (30-day temporal)     |
| - Autoencoder (Deep Reconstruction)   | | 2. Phantom Service (Service consistency)|
| - XGBoost / LightGBM / CatBoost       | | 3. Provider Capacity (Surge threshold)  |
| - Graph Neural Network (PyG GATConv)  | | 4. Duplicate Payment & Settled Claims   |
| - Noisy-OR Dynamic Score Ensembling   | | 5. Upcoding & Unbundling Rules          |
+---------------------------------------+ | 6. Inflated Bill & Cloned Records       |
                    |                     | 7. Prolonged Stay & Readmission         |
                    |                     | 8. Medication & Device Quantity Markups |
                    |                     | 9. Fuzzy Claim Levenshtein Matching     |
                    |                     | * Circuit Breaker Weight Renormalization|
                    |                     +-----------------------------------------+
                    +--------------------+--------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         INVESTIGATION & REASONING LAYER                           |
|  - Explainable AI (SHAP TreeExplainer & LIME Local Interpreters)                  |
|  - Agentic Copilot with Zero-Wipeout Fallback (Gemini / OpenAI / Local Heuristic) |
|  - FAISS RAG Medical Regulatory Knowledge Base (ICD-10, CPT, Kemenkes / BPJS)    |
|  - Automated BAP (Berita Acara Pemeriksaan) Markdown Generation                   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         DATA PERSISTENCE & AUDIT TRAIL                            |
|  - Multi-Tier Parquet Cache & Hash Signature Verification                         |
|  - SHA-256 Tamper-Proof Audit Trail (HMAC Hash Chaining)                          |
|  - Google Cloud Storage (GCS) Model Artifact Mirroring                            |
|  - Google Cloud Logging / WORM Storage Integration                                |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Pillars

### 2.1 Resilient Streaming Ingestion & Memory Management
- **Disk-Buffering Architecture**: Membaca dataset skala gigabyte dalam bongkahan (chunks) disk 8MB tanpa memicu lonjakan memori (RAM spike) di container RAM 2-16 GB.
- **Circuit Breaker Data**: Mencegah crash jika kolom opsional tidak ditemukan; bobot dinormalkan kembali ke 1.0 (100%) secara deterministik.
- **Zero Preprocessing Leakage**: Transformer statistik (Mean, Median, RobustScaler, OneHotEncoder) dihitung eksklusif pada partisi *train* (`__astina_partition == 'train'`) dan diterapkan ke partisi *test* via metadata artefak.

### 2.2 Hybrid Machine Learning & GNN Architecture
1. **Isolation Forest**: Menangkap anomali distribusi spasial multivariat tanpa supervisi.
2. **Deep Autoencoder (PyTorch)**: Mengukur error rekonstruksi non-linear fitur tagihan dan diagnosis.
3. **Gradient Boosting (XGBoost / LightGBM / CatBoost)**: Klasifikasi terawasi atau semi-terawasi dengan Optuna hyperparameter optimization.
4. **Graph Neural Network (PyTorch Geometric `GATConv`)**:
   - Memodelkan relasi sindikat penipuan (*fraud rings*) antara Pasien, Dokter, dan Faskes (Fasilitas Kesehatan).
   - Dilengkapi *subgraph bounding* (≤300 nodes, ≤5.000 edges) guna memastikan visualisasi dan inferensi stabil.

### 2.3 Security, Privacy & Compliance (UU PDP & HIPAA)
- **KDF Password Hashing**: PBKDF2-HMAC-SHA256 dengan 100.000 iterasi dan proteksi *timing attack*.
- **Role-Based Access Control (RBAC)**:
  - `Admin`: Akses penuh konfigurasi sistem, model, kuota, dan audit log.
  - `Auditor`: Akses deteksi, investigasi Copilot, ekspor, dan Audit Mode unmasking.
  - `Analyst`: Akses ingesti, praproses, dan pelatihan model; PII otomatis ter-masking.
  - `Viewer`: Mode read-only untuk dashboard eksekutif dan status sistem.
- **Dynamic PII Masking Engine**: NIK, Nama Pasien, Nomor Rekam Medis disamarkan secara dinamis. Setiap pembukaan masking oleh Auditor/Admin tercatat di audit trail forensik sebagai event `PII_UNMASK`.
- **Fail-Closed Production Gate**: Sistem menolak startup di lingkungan produksi jika variabel lingkungan keamanan belum dikonfigurasi secara lengkap dan kuat (kredensial ≥ 12 karakter).

---

## 3. Deployment Topology & Scalability Notes

- **Current Runtime Container**: Docker Multi-Stage (`python:3.12-slim`), dieksekusi dengan non-root user `appuser` (UID 1000).
- **Process Concurrency Architecture**:
  - Di lingkungan cloud container (Google Cloud Run / AWS ECS / Fly.io), aplikasi saat ini di-lock pada `_MAX_INSTANCES: 1` dan `_CONCURRENCY: 1` karena session state Streamlit dan kunci pelatihan model (`_TRAINING_JOB_LOCK`) bersifat *process-local*.
  - **Horizontal Scaling Roadmap**: Untuk deployment multi-instance horisontal, state sesi perlu dimigrasikan ke Redis Session Store, dan inferensi/training didelegasikan ke arsitektur asinkron (Celery Worker / Cloud Run Jobs).

---

## 4. Disaster Recovery & Auditability
- **Cryptographic Audit Trail**: Rantai blok log JSON di-hash berantai SHA-256. Setiap modifikasi berkas audit akan terdeteksi sebagai anomali integritas oleh skrip verifikasi `verify_audit_trail.py`.
- **WORM Storage Sync**: Sinkronisasi periodik berkas log dan model ke Google Cloud Storage bucket dengan retensi *Write-Once-Read-Many* (WORM).
