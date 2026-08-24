# ASTINA — AI-Powered Insurance Fraud & Anomaly Detection

**ASTINA** adalah platform analitik dan investigasi fraud klaim asuransi kesehatan berbasis **Hybrid AI** yang menggabungkan kekuatan **Machine Learning Ensemble** (Isolation Forest, Autoencoder, XGBoost, GNN) dengan **Rule-Based Business Engine** (9 modul aturan audit klaim). 

Aplikasi ini dilengkapi antarmuka interaktif berbasis **Streamlit**, mendukung pemrosesan dataset besar secara efisien, serta menyediakan Explainable AI (XAI) untuk transparansi keputusan investigasi.

---

## 🚀 Fitur Utama

- **Hybrid Detection Engine**: Menggabungkan probabilitas statistik/anomali ML dengan validasi kepatuhan aturan bisnis asuransi.
- **9 Modul Aturan Bisnis Fraud**:
  1. *Repeat Billing*: Deteksi klaim berulang dalam rentang waktu singkat.
  2. *Phantom Service*: Deteksi tindakan medis yang tidak wajar/fiktif.
  3. *Provider Capacity*: Validasi kapasitas harian dan over-utilization faskes/dokter.
  4. *Claim Status & Duplicate Payment*: Validasi duplikasi pembayaran dan inkonsistensi administratif.
  5. *Upcoding & Unbundling*: Deteksi penaikan tarif tindakan atau pemecahan paket tagihan.
  6. *Inflated Bill & Cloning*: Deteksi lonjakan tarif ekstrem dan duplikasi tagihan mirip.
  7. *Length of Stay & Readmission*: Evaluasi lama rawat inap dan pola readmisi tidak wajar.
  8. *Medication & Device Fraud*: Validasi kuantitas dan harga satuan obat/alkes.
  9. *Fuzzy Claim Matching*: Pencocokan kemiripan klaim non-identik berbasis kemiripan teks & atribut.
- **Explainable AI (SHAP & LIME)**: Penjelasan kontribusi fitur untuk model Tree-based (XGBoost) dan Isolation Forest.
- **Large Dataset Optimization**: Chunked processing, streaming ingestion, dan optimasi tipe data berbasis Polars/PyArrow.
- **Multi-Environment Ready**: Siap dijalankan di Local (Windows/Linux/macOS), Docker Desktop, dan Serverless Google Cloud Run.

---

## 📁 Struktur Repositori

```text
project-Graphnet-main/
├── main.py                     # Entry point aplikasi Streamlit
├── run.py                      # Production/Local runtime launcher
├── config.py                   # Konfigurasi global, limit memori, & rules parameter
├── fraud_risk_pipeline.py      # Pipeline integrasi scoring & agregasi risiko
├── preprocessing_optimized.py  # Pipeline preprocessing & feature engineering
├── large_file_processor.py     # Pemrosesan dataset besar berbasis chunk
├── file_handler.py             # IO file handler (CSV/Excel/Parquet streaming)
├── model.py                    # Implementasi ensemble model ML & GNN
├── model_registry.py           # Metadata & registry model tersimpan
├── model_explainer.py          # Modul Explainability (SHAP/LIME)
├── ui/                         # Komponen & modul halaman Streamlit
│   ├── pages/                  # Halaman aplikasi (home, detection, training, dll.)
│   ├── sidebar.py              # Navigasi & status perangkat (GPU/CPU)
│   └── utils.py                # Visual helper & komponen UI
├── tests/                      # Suite pengujian otomatis (Pytest)
│   └── test_detection_modules.py
├── .cloudrun/                  # Skrip & konfigurasi deployment Cloud Run
├── Dockerfile                  # Multi-stage Dockerfile teroptimasi
├── docker-compose.yml          # Konfigurasi orkestrasi Docker Desktop
├── requirements.txt            # Dependensi Python terverifikasi (Python 3.11 - 3.13)
└── README.md                   # Dokumentasi teknis & operasional
```

---

## ⚙️ Persyaratan Sistem & Dependensi

- **Python**: Versi `3.11` s/d `3.13` (Direkomendasikan Python 3.11 atau 3.13).
- **Hardware Minimum**: 4 Core CPU, 8 GB RAM (16 GB RAM disarankan untuk dataset besar).
- **Akselerasi Opsional**: GPU NVIDIA (CUDA 11.8 / 12.x) atau AMD GPU (ROCm) untuk akselerasi PyTorch/GNN.
- **Docker**: Docker Desktop versi terbaru dengan Docker Compose v2.

Semua dependensi inti telah disesuaikan dan dikunci pada `requirements.txt`:
`streamlit`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `joblib`, `torch`, `torch-geometric`, `imbalanced-learn`, `plotly`, `xgboost`, `lightgbm`, `catboost`, `polars`, `pyarrow`, `optuna`, `hdbscan`, `faiss-cpu`, `psutil`, `shap`, `lime`, dan `google-cloud-storage`.

---

## 🛠️ Panduan Menjalankan Aplikasi

### Opsi 1: Local Environment (PowerShell / Bash)

1. **Clone repositori dan masuk ke direktori proyek**:
   ```bash
  cd c:\project-Graphnet
   ```

2. **Buat dan aktifkan Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
      py -3.13 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
      python3.13 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install semua dependensi**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi**:
   - Menggunakan Streamlit langsung:
     ```bash
     streamlit run main.py
     ```
   - Atau menggunakan launcher runtime:
     ```bash
     python run.py
     ```

5. **Akses Dashboard**:
   Buka browser pada [http://localhost:8501](http://localhost:8501).

---

### Opsi 2: Docker Desktop (Rekomendasi untuk Kontainerisasi)

Aplikasi telah dilengkapi **Multi-stage Dockerfile** dan **Docker Compose** yang mengisolasi dependensi, mengoptimalkan ukuran image, serta menjalankan proses sebagai user non-root aman (`appuser`).

#### Menggunakan Docker Compose (Paling Cepat & Lengkap)

1. **Pastikan Docker Desktop sudah berjalan**.
2. **Jalankan build dan container**:
   ```bash
   docker-compose up --build -d
   ```
3. **Periksa status container & log**:
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```
4. **Buka aplikasi di browser**:
   Akses [http://localhost:8501](http://localhost:8501).
5. **Menghentikan container**:
   ```bash
   docker-compose down
   ```

*Catatan: Folder `./cache` dan `./models` otomatis di-mount ke host agar model terlatih dan cache analisis tidak hilang saat container di-restart.*

#### Menggunakan Docker CLI Standar

```bash
# 1. Build Image
docker build -t astina-app .

# 2. Jalankan Container dengan Volume Persistensi
docker run -d \
  --name astina-app \
  -p 8501:8501 \
  -v ${PWD}/cache:/app/cache \
  -v ${PWD}/models:/app/models \
  astina-app
```

---

### Opsi 3: Deployment ke Google Cloud Run

ASTINA mendukung continuous deployment ke Cloud Run:

- **Windows (PowerShell)**:
  ```powershell
  .\.cloudrun\deploy.ps1
  ```
- **Linux / macOS**:
  ```bash
  chmod +x .cloudrun/deploy.sh
  ./.cloudrun/deploy.sh
  ```

*Untuk persistensi model di Cloud Run, set `GOOGLE_CLOUD_BUCKET=nama-bucket-gcs-anda` dan gunakan service account khusus yang memiliki akses Storage Object Admin/Creator sesuai kebijakan keamanan. Deployment default bersifat privat (`--no-allow-unauthenticated`).*

---

## ✅ Status Readiness dan Batas Operasional

Pembaruan yang sudah diterapkan:

- Batas upload aplikasi, Docker, Compose, dan Cloud Run diselaraskan ke `3072 MiB`.
- `.streamlit/config.toml` menetapkan `maxUploadSize` dan `maxMessageSize` secara global, termasuk saat menjalankan `streamlit run main.py` langsung.
- Konfigurasi CORS/XSRF dibuat kompatibel agar tidak menimbulkan override keamanan saat startup.
- CSV besar ditulis ke Parquet secara batch sebelum dimaterialisasi, sehingga tidak lagi menahan seluruh list chunk untuk `pd.concat`.
- Graph star, heterogeneous, dan k-NN memiliki batas node/edge deterministik.
- Visualisasi GNN memakai `node_id` untuk pemetaan probabilitas dan jumlah node tampilan dapat dipilih.
- Training GNN menyediakan mode `NeighborLoader` mini-batch dengan loss pada seed nodes serta fallback kompatibel jika backend sampler PyG tidak tersedia.
- Pipeline rule menggunakan stable `_astina_row_id` sehingga hasil tetap benar setelah sorting, reset index, dan perubahan chunk.
- Evaluasi business rules dijalankan pada dataset global, sehingga pasangan atau group yang melintasi batas chunk tidak hilang.
- Fuzzy matching dipetakan berdasarkan stable row ID, provider capacity berdasarkan tanggal kalender, dan duplicate payment mengecualikan klaim saat ini.
- Evaluation dan detection meneruskan graph ke GNN saat model GNN aktif; medication/device juga menangani kasus quantity delivered nol.
- Visualisasi GNN membersihkan state graph lama, menyimpan metode graph, menghitung score pada node graph yang sama, memilih node paling relevan, mempertahankan isolated node, dan memakai color scale tetap 0-1.
- Visualisasi heterogeneous graph mempertahankan `edge_type` dan membedakan relasi Provider, Patient, dan Diagnosis dengan warna serta legend.
- Training GNN heterogeneous meneruskan one-hot `edge_type` sebagai `edge_attr` ke `GATConv`; checkpoint menyimpan `edge_dim` agar arsitektur relation-aware dapat dipulihkan.
- Startup tidak lagi menjalankan `st.set_page_config()` dua kali.
- Restore artefak model Cloud Storage menggunakan nama file yang sesuai dengan prefix model.
- Restore GNN memulihkan parameter arsitektur, termasuk jumlah layer, dropout, head, dan hidden channel.
- Impor model ZIP memvalidasi path dan mengekstrak file secara streaming untuk mencegah path traversal.
- Test suite tervalidasi: `33 passed`; `pip check` tidak menemukan dependency rusak.

### Penilaian Bottleneck

Status saat ini: **layak untuk localhost dan Docker pada dataset kecil/menengah; layak untuk Cloud Run setelah konfigurasi IAM/GCS dilengkapi; belum bebas bottleneck untuk dataset 3 GiB end-to-end.**

Batas yang masih perlu diperhatikan:

1. Halaman upload masih mengembalikan DataFrame penuh agar kompatibel dengan preprocessing dan EDA lama. File 3 GiB dapat membutuhkan RAM beberapa kali ukuran file.
2. Preprocessing dan train/test split masih membuat representasi DataFrame tambahan.
3. GNN mini-batch membutuhkan backend sampler PyG pada environment deployment. Tanpa backend tersebut, aplikasi fallback ke full-batch dan dapat mengalami tekanan memory.
4. Cloud Run filesystem bersifat ephemeral. Dataset, cache, registry, audit fallback, dan model harus disimpan ke GCS/database bila diperlukan lintas restart atau instance.
5. Upload 3 GiB production sebaiknya menggunakan resumable/direct upload ke GCS, bukan mengirim seluruh file melalui request Streamlit.
6. Satu job training besar per instance direkomendasikan (`concurrency=1`) agar beberapa job tidak berebut memory.
7. Cache DataFrame upload memakai session-level cache satu salinan, bukan serialisasi `st.cache_data` yang dapat menggandakan peak memory.

### Validasi Sebelum Production

```powershell
# Aktifkan environment yang kompatibel
.\.venv\Scripts\Activate.ps1

# Dependency, import, dan test
python -m pip check
python -m pytest -q
python -c "import main; print('startup imports ok')"

# Docker
docker compose config
docker compose up --build -d
docker compose ps
```

Acceptance test production wajib mencakup upload 3 GiB resumable, peak memory, disk temporary, restart worker, concurrent users, restore model dari GCS, training GNN sampled, dan pencocokan `node_id` dengan anomaly probability.

---

## 🔍 Alur Kerja & Formula Risiko

```text
[Input Data Klaim (CSV/XLSX)]
       │
       ▼
[1. Preprocessing & Alignment] ─── Validasi schema, normalisasi tanggal & tarif
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[2. ML Anomaly Scoring]               [3. Rule-Based Fraud Detection]
   • Isolation Forest                    • 9 Business Rule Modules
   • Autoencoder                         • Fuzzy Similarity Engine
   • XGBoost / GNN                       • Provider & Service Validation
       │                                         │
       └────────────────────┬────────────────────┘
                            │
                            ▼
               [4. Risk Score Aggregation]
                            │
                            ▼
         [5. Executive Dashboard & Investigation]
```

### Formula Skor Risiko Bisnis

Skor risiko bisnis (`business_risk_score`) dihitung secara deterministik:
$$\text{Business Risk} = 0.40(R_{\text{repeat}}) + 0.20(R_{\text{phantom}}) + 0.15(R_{\text{capacity}}) + 0.15(R_{\text{fuzzy}}) + 0.10(R_{\text{other}})$$

Skor ini kemudian dikombinasikan dengan skor anomali Machine Learning untuk menghasilkan **Overall Combined Risk Score** (0.00 - 1.00) dan klasifikasi severity: **Low**, **Medium**, atau **High Risk**.

---

## 🔧 Konfigurasi Lingkungan (`.env`)

Konfigurasi opsional dapat disetel melalui file `.env` di direktori utama:

| Variabel | Default | Keterangan |
| :--- | :--- | :--- |
| `PORT` | `8501` | Port listen aplikasi Streamlit |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | `3072` | Batas maksimum upload dataset (MiB) |
| `ASTINA_LOG_FORMAT` | `json` | Format logging (`json` / `text`) |
| `GOOGLE_CLOUD_BUCKET` | *(Opsional)* | Nama GCS Bucket untuk sinkronisasi model |
| `OPTUNA_N_JOBS` | `4` | Jumlah thread paralel optimasi hyperparameter |
| `CV_N_JOBS` | `4` | Jumlah thread paralel Cross Validation |

---

## 🧪 Pengujian & Validasi Kualitas

Aplikasi dilengkapi suite pengujian otomatis untuk memverifikasi fungsionalitas seluruh modul deteksi dan explainability:

```powershell
# Jalankan seluruh test suite dengan Pytest
python -m pytest -q
```

Hasil verifikasi memastikan:
- ✅ Seluruh 9 modul deteksi fraud berfungsi normal pada berbagai tipe data.
- ✅ Penanganan data kosong, missing values, dan format tidak standar berjalan tanpa crash.
- ✅ SHAP Feature Importance terlindungi UI Guard saat model non-kompatibel dipilih.
- ✅ Graph GNN menghormati batas node/edge dan mempertahankan ID node yang valid.

Training GNN pada graph besar mendukung `NeighborLoader` melalui parameter
`gnn_params`: `use_neighbor_sampling`, `sampling_threshold_nodes`,
`batch_size`, dan `num_neighbors`. Jika backend sampler PyG tidak tersedia,
aplikasi menggunakan fallback full-batch dan mencatat peringatan di log.

---

## 🩺 Troubleshooting

- **Port 8501 bentrok / sudah dipakai**:
  ```powershell
  streamlit run main.py --server.port 8502
  ```
- **Error PyTorch / CUDA di Local**:
  Pastikan versi PyTorch sesuai dengan versi CUDA driver Anda. Untuk mode CPU murni, instalasi standar `requirements.txt` langsung siap digunakan.
- **Docker Desktop permission / volume mount**:
  Pastikan folder `cache/` dan `models/` ada di root project sebelum menjalankan `docker-compose up`.
- **Dataset Besar Out of Memory**:
  Gunakan format Parquet. Ingestion CSV besar berjalan per chunk, tetapi training GNN tetap membutuhkan strategi sampling graph dan resource produksi yang memadai.

---

## 📄 Lisensi

Proyek ini dirancang untuk audit dan pengawasan klaim asuransi kesehatan dengan lisensi MIT / Enterprise Internal Policy.
#   p r o j e c t - G r a p h n e t  
 