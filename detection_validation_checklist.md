# Validation Checklist untuk Deteksi Data Baru

## ✅ Prasyarat untuk Deteksi Berhasil

### 1. Model Training Sudah Selesai
- [ ] Model sudah dilatih dengan data training
- [ ] Model tersimpan di folder `models/`
- [ ] `training_metadata` tersimpan lengkap

### 2. Fitur Data Baru Minimal
**Wajib ada minimal 1 kolom dari setiap kategori:**
- [ ] Identifikasi: `claim_id`, `patient_id`, atau `provider_id`
- [ ] Layanan: `service_code` atau kolom prosedur medis
- [ ] Nominal: `billed_amount` atau kolom jumlah tagihan
- [ ] Waktu: `billing_date`, `service_date`, atau kolom tanggal

### 3. Konsistensi Skema Data
**Pastikan data baru memiliki skema yang mirip dengan training:**
- [ ] Kolom numerik (jika ada di training)
- [ ] Kolom kategorikal (jika ada di training)
- [ ] Kolom tanggal dalam format yang valid
- [ ] Tidak ada kolom yang sepenuhnya kosong

### 4. Ukuran Data
- [ ] Minimal 2 baris data untuk deteksi
- [ ] Jika > 100MB, gunakan format Parquet untuk performa optimal

## 🔍 Cara Verifikasi Deteksi Berhasil

### Langkah 1: Cek Status Model
1. Buka halaman **Settings** → tab **🗄️ Model Registry**
2. Pastikan model tersedia dan statusnya "trained"
3. Cek jumlah fitur yang digunakan dalam training

### Langkah 2: Upload Data Baru
1. Buka halaman **Deteksi**
2. Pilih "📤 Unggah File Baru"
3. Upload dataset baru (CSV/Excel/Parquet)
4. Periksa pesan "Dataset Siap Dieksekusi"

### Langkah 3: Jalankan Deteksi
1. Klik "🚀 Jalankan Deteksi Anomali Multi-Algoritma"
2. Tunggu proses selesai
3. Periksa hasil di tab "📊 Ringkasan & Visualisasi"

### Langkah 4: Verifikasi Hasil
**Indikator deteksi berhasil:**
- ✅ Tidak ada error pada eksekusi
- ✅ Metrics ditampilkan (Total Klaim, Anomali, High Risk)
- ✅ Tab Business Risk & Rules menampilkan hasil
- ✅ Tab Fraud Review Table berisi data lengkap

**Indikator potensi masalah:**
- ❌ Error "Feature tidak ditemukan"
- ❌ Error "Metadata fitur tidak tersedia"
- ❌ Hasil kosong atau semua prediksi 0/1
- ❌ Warning tentang feature alignment

## ⚠️ Troubleshooting Umum

### Masalah: "Metadata Fitur Tidak Ditemukan"
**Penyebab**: Model lama tidak menyimpan metadata fitur

**Solusi**:
1. Latih ulang model dengan versi terbaru
2. Pastikan training menyimpan `training_metadata`

### Masalah: "Feature Alignment Gagal"
**Penyebab**: Kolom data baru sangat berbeda dengan training

**Solusi**:
1. Gunakan Semantic Schema Harmonizer untuk mapping kolom
2. Pastikan kolom penting ada di data baru
3. Jika perlu, latih ulang model dengan data yang lebih representatif

### Masalah: "Hasil Tidak Akurat"
**Penyebab**: Data baru memiliki distribusi yang sangat berbeda

**Solusi**:
1. Cek Concept Drift di tab "📈 Concept Drift & Retraining"
2. Retrain model dengan data baru jika drift terdeteksi
3. Gunakan adaptive threshold untuk menyesuaikan

## 🎯 Best Practices

### Untuk Deteksi Data Baru:
1. **Gunakan format yang sama** dengan data training
2. **Pastikan preprocessing konsisten** (encoding, scaling)
3. **Review feature alignment** di expander detail
4. **Validasi hasil dengan sampling manual**
5. **Gunakan threshold yang sesuai** dengan konteks bisnis

### Untuk Production:
1. **Batch processing** untuk dataset besar
2. **Schedule retraining** berkala (mingguan/bulanan)
3. **Monitor concept drift** secara rutin
4. **Validasi hasil** dengan auditor senior
5. **Maintain audit trail** untuk semua deteksi

## 📊 Contoh Workflow Deteksi yang Benar

### Scenario 1: Data Mingguan Baru
```
1. Upload klaim minggu ini (format sama dengan training)
2. Jalankan deteksi dengan threshold 0.5
3. Review 50 klaim high risk
4. Generate BAP untuk top 10 klaim
5. Export hasil untuk audit
```

### Scenario 2: Data Baru dengan Skema Berbeda
```
1. Upload data baru
2. Semantic Schema Harmonizer mapping kolom otomatis
3. Review gap analysis di metadata
4. Preprocessing dengan stats training
5. Jalankan deteksi
6. Validasi hasil manual
```

### Scenario 3: Large Dataset (>1GB)
```
1. Convert ke Parquet untuk optimalisasi
2. Upload file Parquet
3. Sistem otomatis sampling untuk interaktif
4. Jalankan deteksi pada sampel
5. Catat bahwa hasil hanya mencakup sampel dan bukan audit lengkap
6. Jangan menyatakan audit seluruh file didukung sampai jalur inference batch penuh tersedia
```

## 🔒 Security & Compliance

- Data baru diproses dengan PII masking sebelum LLM
- Audit trail mencatat semua deteksi
- RBAC membatasi akses ke hasil deteksi
- Hasil deteksi tidak disimpan permanen di session state

---

**Last Updated**: 2026-10-25
