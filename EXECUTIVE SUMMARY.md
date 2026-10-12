# ASTINA — Executive Summary: AI-Powered Insurance Fraud Detection

## 📌 Executive Overview

**ASTINA** adalah sistem pertahanan dan audit cerdas tingkat enterprise yang dirancang untuk mendeteksi kecurangan (*fraud*), penyalahgunaan (*abuse*), dan anomali pada klaim asuransi kesehatan (BPJS/JKN & Asuransi Komersial).

Dengan mengintegrasikan **Machine Learning Ensemble canggih (Isolation Forest, Deep Autoencoder, XGBoost, Graph Neural Network)**, **9 Modul Aturan Bisnis Deterministik**, serta **Agentic Copilot bertenaga RAG (Retrieval-Augmented Generation)**, ASTINA mengubah proses investigasi klaim konvensional yang manual dan lambat menjadi investigasi otomatis yang presisi, terukur, dan transparan.

---

## 🎯 Nilai Bisnis & Dampak Finansial (ROI)

1. **Efisiensi Finansial & Pencegahan Kebocoran Dana**:
   - Mencegah pembayaran klaim palsu seperti *Repeat Billing*, *Phantom Service*, *Upcoding*, dan *Inflated Billing* secara preventif sebelum dana dicairkan.
2. **Peningkatan Produktivitas Investigator**:
   - Mengurangi waktu audit manual hingga 70% melalui *Review Table* interaktif berperingkat risiko dan *Auto-Generated Berita Acara Pemeriksaan (BAP)*.
3. **Deteksi Sindikat Terorganisir (Fraud Rings)**:
   - Membongkar pola kolusi tersembunyi antara fasilitas kesehatan (faskes), dokter, dan peserta menggunakan *Graph Neural Network (GNN)* yang tidak dapat dideteksi oleh sistem berbasis aturan tradisional.
4. **Kepatuhan Hukum & Perlindungan Data**:
   - Memenuhi penuh ketentuan **UU No. 27 Tahun 2022 tentang Perlindungan Data Pribadi (UU PDP)** melalui *Role-Based PII Dynamic Masking* dan rantai audit kriptografis SHA-256 yang *tamper-proof*.

---

## 🛡️ Kesiapan Operasional & Keamanan (Enterprise-Ready)

| Dimensi | Kapabilitas ASTINA |
| :--- | :--- |
| **Keamanan Autentikasi** | PBKDF2-HMAC-SHA256 (100.000 iterasi) + Account Lockout & Brute-Force Shield (5 percobaan gagal / 15 menit jeda). |
| **Proteksi Kuota & Anti-DoS** | Role-Based Rate Limiting & Quota Management (upload hingga 3GB, kuota pelatihan/inferensi deterministik per peran) untuk mencegah kehabisan sumber daya komputasi. |
| **Privasi Pasien** | Penyamaran otomatis NIK, Nama, dan No Peserta bagi analis/viewer; mode audit resmi dengan jejak forensik bagi auditor. |
| **Integritas Forensik** | Cryptographic SHA-256 hash chaining dengan sanitasi metrik deterministik dan sinkronisasi otomatis ke Google Cloud Storage WORM Bucket. |
| **Skalabilitas Data & Type Safety** | Streaming ingestion Parquet (<1GB RAM footprint) dengan parser Polars/Pandas bertipe eksplisit dan bebas ambiguitas runtime, mampu menangani jutaan baris data klaim tanpa resiko out-of-memory. |
| **Infrastruktur Fleksibel** | Siap dijalankan di Local Workstation (CPU/GPU CUDA), Docker Desktop terisolasi, maupun Serverless Google Cloud Run. |

---

## 🏁 Kesimpulan

ASTINA telah memenuhi seluruh kriteria kesiapan produksi (*Production Readiness Gate*), validasi *zero-regression* dan *zero-type-error* pada seluruh alur *ingestion*, *rate limiting*, dan *audit trail*, serta standar tata kelola kepatuhan data medis modern. Sistem ini beroperasi stabil dan optimal sebagai garda terdepan pengawasan klaim asuransi kesehatan enterprise.
