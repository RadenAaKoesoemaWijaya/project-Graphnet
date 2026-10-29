"""
Generator dataset dummy deteksi fraud klaim asuransi kesehatan (ASTINA)
Menghasilkan 400 baris data dummy bervariasi yang merepresentasikan seluruh jenis fraud:
1. Repeat Billing (Klaim Berulang dalam Temporal Window)
2. Phantom Service (Layanan Fiktif: Kode Tidak Valid, Usia Tidak Logis, Frekuensi Ekstrem)
3. Provider Capacity Violation (Pelanggaran Batas Kapasitas Faskes per Hari)
4. Upcoding & Unbundling (Peningkatan Kode Diagnosa/Tindakan & Mismatch Ekstrem)
5. Inflated Bill & Cloning (Penggelembungan Biaya & Duplikasi/Kloning Klaim)
6. Prolonged Length of Stay & Suspicious Readmission (Lama Rawat Inap Berlebih & Readmisi Cepat)
7. Medication & Device Billing Fraud (Diskrepansi Kuantitas, Mark-up Harga Obat/Alkes, Kalkulasi Salah)
8. Duplicate Payment (Klaim Berulang dari Pembayaran yang Sudah Selesai/Paid)
9. Collusion & Network Anomaly (Sindikat Pasien dan Provider Fiktif)
10. Legitimate Control Claims (Klaim Normal Rawat Jalan, Rawat Inap, & Pasien Kronis Ter-whitelist)
"""
import csv
import os
import random
from datetime import datetime, timedelta

random.seed(42)

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_dummy_fraud_detection.csv")

FIELDNAMES = [
    "claim_id",
    "patient_id",
    "provider_id",
    "service_code",
    "procedure_code",
    "diagnosis_code",
    "billing_date",
    "service_date",
    "admission_date",
    "discharge_date",
    "billed_amount",
    "paid_amount",
    "allowed_amount",
    "claim_status",
    "patient_age",
    "length_of_stay",
    "quantity",
    "quantity_billed",
    "quantity_delivered",
    "unit_price",
    "item_code",
    "item_name",
    "provider_type",
    "facility_type",
    "patient_class",
    "patient_segment",
    "patient_province",
    "patient_city",
    "provider_province",
    "provider_city",
    "membership_status",
    "fraud_type",
    "is_fraud"
]

rows = []
claim_counter = 1

def make_claim_id(idx):
    return f"CLM-2026-{idx:04d}"

# ==============================================================================
# 1. REPEAT BILLING (36 baris / 18 pasang klaim)
# ==============================================================================
# Ciri: Pasien sama, Provider sama, Kode Layanan sama, Tanggal selisih 2-10 hari (<30 hari),
# Biaya identik / selisih < 5%, Diagnosa akut non-kronis (A01, K29, B02, R10, J06).
repeat_configs = [
    ("P-REP-01", "PRV-RS-01", "CT001", "CT001", "A01", "2026-05-02", 4, 2500000.0, 2500000.0, 42),
    ("P-REP-02", "PRV-RS-01", "XR001", "XR001", "K29", "2026-05-03", 5, 450000.0, 450000.0, 35),
    ("P-REP-03", "PRV-RS-02", "US001", "US001", "R10", "2026-05-05", 3, 650000.0, 640000.0, 29),
    ("P-REP-04", "PRV-RS-03", "BT001", "BT001", "B02", "2026-05-08", 6, 350000.0, 350000.0, 50),
    ("P-REP-05", "PRV-RS-02", "CONS001", "CONS001", "J06", "2026-05-10", 2, 250000.0, 250000.0, 23),
    ("P-REP-06", "PRV-RS-04", "CT001", "CT001", "S06", "2026-05-12", 7, 2800000.0, 2800000.0, 47),
    ("P-REP-07", "PRV-RS-01", "LAB001", "LAB001", "A01", "2026-05-14", 4, 400000.0, 395000.0, 38),
    ("P-REP-08", "PRV-RS-03", "XR001", "XR001", "M54", "2026-05-16", 5, 480000.0, 480000.0, 55),
    ("P-REP-09", "PRV-RS-04", "US001", "US001", "K29", "2026-05-18", 3, 620000.0, 620000.0, 31),
    ("P-REP-10", "PRV-RS-02", "CT001", "CT001", "R10", "2026-05-20", 8, 2600000.0, 2580000.0, 60),
    ("P-REP-11", "PRV-RS-05", "BT001", "BT001", "A01", "2026-05-22", 4, 320000.0, 320000.0, 27),
    ("P-REP-12", "PRV-RS-01", "CONS001", "CONS001", "J06", "2026-05-24", 2, 275000.0, 275000.0, 40),
    ("P-REP-13", "PRV-RS-03", "XR001", "XR001", "B02", "2026-05-26", 6, 460000.0, 460000.0, 33),
    ("P-REP-14", "PRV-RS-05", "CT001", "CT001", "S06", "2026-05-28", 5, 2700000.0, 2700000.0, 49),
    ("P-REP-15", "PRV-RS-02", "LAB001", "LAB001", "K29", "2026-06-01", 3, 420000.0, 415000.0, 52),
    ("P-REP-16", "PRV-RS-04", "US001", "US001", "R10", "2026-06-03", 7, 630000.0, 630000.0, 36),
    ("P-REP-17", "PRV-RS-01", "BT001", "BT001", "A01", "2026-06-05", 4, 340000.0, 340000.0, 44),
    ("P-REP-18", "PRV-RS-03", "CONS001", "CONS001", "J06", "2026-06-07", 3, 260000.0, 260000.0, 28),
]

for pat, prv, srv, prc, diag, d1_str, gap, amt1, amt2, age in repeat_configs:
    d1 = datetime.strptime(d1_str, "%Y-%m-%d")
    d2 = d1 + timedelta(days=gap)
    d2_str = d2.strftime("%Y-%m-%d")
    
    # Claim 1
    cid1 = make_claim_id(claim_counter)
    claim_counter += 1
    rows.append({
        "claim_id": cid1, "patient_id": pat, "provider_id": prv,
        "service_code": srv, "procedure_code": prc, "diagnosis_code": diag,
        "billing_date": d1_str, "service_date": d1_str, "admission_date": d1_str, "discharge_date": d1_str,
        "billed_amount": amt1, "paid_amount": round(amt1 * 0.85, 2), "allowed_amount": round(amt1 * 0.90, 2),
        "claim_status": "Settled", "patient_age": age, "length_of_stay": 0, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt1, "item_code": f"SRV-{srv}", "item_name": f"Layanan {srv}",
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 2", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DKI Jakarta", "patient_city": "Jakarta Selatan", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Selatan",
        "membership_status": "Aktif", "fraud_type": "Repeat Billing", "is_fraud": 1
    })
    
    # Claim 2 (Repeat)
    cid2 = make_claim_id(claim_counter)
    claim_counter += 1
    rows.append({
        "claim_id": cid2, "patient_id": pat, "provider_id": prv,
        "service_code": srv, "procedure_code": prc, "diagnosis_code": diag,
        "billing_date": d2_str, "service_date": d2_str, "admission_date": d2_str, "discharge_date": d2_str,
        "billed_amount": amt2, "paid_amount": round(amt2 * 0.85, 2), "allowed_amount": round(amt2 * 0.90, 2),
        "claim_status": "Settled", "patient_age": age, "length_of_stay": 0, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt2, "item_code": f"SRV-{srv}", "item_name": f"Layanan {srv}",
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 2", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DKI Jakarta", "patient_city": "Jakarta Selatan", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Selatan",
        "membership_status": "Aktif", "fraud_type": "Repeat Billing", "is_fraud": 1
    })

# ==============================================================================
# 2. PHANTOM SERVICE (28 baris)
# ==============================================================================
# 2A: Kode layanan tidak valid / fiktif (8 baris)
invalid_codes = ["BAD999", "PHAN01", "GHOST01", "UNK999", "FAK001", "NON002", "PSEUDO1", "NIL009"]
for i, inv_code in enumerate(invalid_codes):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = f"2026-05-{10+i:02d}"
    rows.append({
        "claim_id": cid, "patient_id": f"P-PHAN-INV{i+1:02d}", "provider_id": "PRV-RS-PHANTOM",
        "service_code": inv_code, "procedure_code": inv_code, "diagnosis_code": "R10",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": 1850000.0, "paid_amount": 1572500.0, "allowed_amount": 1665000.0,
        "claim_status": "Under Review", "patient_age": 30 + i, "length_of_stay": 0, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": 1850000.0, "item_code": inv_code, "item_name": "Tindakan Fiktif Tanpa Standar",
        "provider_type": "Klinik Pratama", "facility_type": "Klinik", "patient_class": "Kelas 3", "patient_segment": "PBPU Mandiri",
        "patient_province": "Jawa Barat", "patient_city": "Bogor", "provider_province": "Jawa Barat", "provider_city": "Bogor",
        "membership_status": "Aktif", "fraud_type": "Phantom Service", "is_fraud": 1
    })

# 2B: Usia tidak logis (CT Scan untuk bayi umur 0 bulan / 0 tahun) (6 baris)
for i in range(6):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = f"2026-05-{18+i:02d}"
    rows.append({
        "claim_id": cid, "patient_id": f"P-INFANT-{i+1:02d}", "provider_id": "PRV-RS-02",
        "service_code": "CT001", "procedure_code": "CT001", "diagnosis_code": "P07",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": 2800000.0, "paid_amount": 2380000.0, "allowed_amount": 2520000.0,
        "claim_status": "Pending", "patient_age": 0, "length_of_stay": 0, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": 2800000.0, "item_code": "SRV-CT001", "item_name": "CT Scan Kepala",
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 1", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DKI Jakarta", "patient_city": "Jakarta Pusat", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Pusat",
        "membership_status": "Aktif", "fraud_type": "Phantom Service", "is_fraud": 1
    })

# 2C: Frekuensi harian tidak wajar (14 klaim konsultasi pada hari yang sama untuk 1 pasien) (14 baris)
for i in range(14):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = "2026-05-15"
    rows.append({
        "claim_id": cid, "patient_id": "P-PHAN-FREQ01", "provider_id": "PRV-RS-03",
        "service_code": "CONS001", "procedure_code": "CONS001", "diagnosis_code": "R51",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": 250000.0, "paid_amount": 212500.0, "allowed_amount": 225000.0,
        "claim_status": "Pending", "patient_age": 45, "length_of_stay": 0, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": 250000.0, "item_code": "SRV-CONS001", "item_name": "Konsultasi Dokter Umum",
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 3", "patient_segment": "PBPU Mandiri",
        "patient_province": "Banten", "patient_city": "Tangerang", "provider_province": "Banten", "provider_city": "Tangerang",
        "membership_status": "Aktif", "fraud_type": "Phantom Service", "is_fraud": 1
    })

# ==============================================================================
# 3. PROVIDER CAPACITY VIOLATION (26 baris)
# ==============================================================================
# 3A: PRV-CAP-SURG01 melakukan 3 operasi (SUR001) pada 2026-06-01 (Maksimal 1 operasi / hari, durasi 180m*3 = 540m > 480m) (3 baris)
for i in range(3):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = "2026-06-01"
    rows.append({
        "claim_id": cid, "patient_id": f"P-SURG-CAP-{i+1:02d}", "provider_id": "PRV-CAP-SURG01",
        "service_code": "SUR001", "procedure_code": "SUR001", "diagnosis_code": "K35",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": 14500000.0, "paid_amount": 12325000.0, "allowed_amount": 13050000.0,
        "claim_status": "Settled", "patient_age": 28 + i * 5, "length_of_stay": 1, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": 14500000.0, "item_code": "SRV-SUR001", "item_name": "Tindakan Bedah Mayor",
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 1", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DKI Jakarta", "patient_city": "Jakarta Barat", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Barat",
        "membership_status": "Aktif", "fraud_type": "Provider Capacity", "is_fraud": 1
    })

# 3B: PRV-CAP-CT01 melakukan 5 CT Scan (CT001) pada 2026-06-02 (Maksimal 2 / hari) (5 baris)
for i in range(5):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = "2026-06-02"
    rows.append({
        "claim_id": cid, "patient_id": f"P-CT-CAP-{i+1:02d}", "provider_id": "PRV-CAP-CT01",
        "service_code": "CT001", "procedure_code": "CT001", "diagnosis_code": "S06",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": 2750000.0, "paid_amount": 2337500.0, "allowed_amount": 2475000.0,
        "claim_status": "Settled", "patient_age": 35 + i * 4, "length_of_stay": 0, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": 2750000.0, "item_code": "SRV-CT001", "item_name": "CT Scan Kepala Non-Kontras",
        "provider_type": "Klinik Utama", "facility_type": "Klinik", "patient_class": "Kelas 2", "patient_segment": "PBI APBN",
        "patient_province": "Jawa Barat", "patient_city": "Depok", "provider_province": "Jawa Barat", "provider_city": "Depok",
        "membership_status": "Aktif", "fraud_type": "Provider Capacity", "is_fraud": 1
    })

# 3C: PRV-CAP-CONS01 melakukan 18 Konsultasi (CONS001) pada 2026-06-03 (Maksimal 15 / hari) (18 baris)
for i in range(18):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = "2026-06-03"
    rows.append({
        "claim_id": cid, "patient_id": f"P-CONS-CAP-{i+1:02d}", "provider_id": "PRV-CAP-CONS01",
        "service_code": "CONS001", "procedure_code": "CONS001", "diagnosis_code": "A09",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": 250000.0, "paid_amount": 212500.0, "allowed_amount": 225000.0,
        "claim_status": "Settled", "patient_age": 20 + i * 2, "length_of_stay": 0, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": 250000.0, "item_code": "SRV-CONS001", "item_name": "Konsultasi Spesialis Penyakit Dalam",
        "provider_type": "Klinik Pratama", "facility_type": "Klinik", "patient_class": "Kelas 3", "patient_segment": "PBPU Mandiri",
        "patient_province": "DKI Jakarta", "patient_city": "Jakarta Timur", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Timur",
        "membership_status": "Aktif", "fraud_type": "Provider Capacity", "is_fraud": 1
    })

# ==============================================================================
# 4. UPCODING & UNBUNDLING (25 baris)
# ==============================================================================
# 4A: Diagnosis and Procedure code identical / duplicated (10 baris)
upcoding_pairs = [
    ("A01", "A01", 12500000.0),
    ("K29", "K29", 11200000.0),
    ("I10", "I10", 9800000.0),
    ("B02", "B02", 10500000.0),
    ("M54", "M54", 11800000.0),
    ("J06", "J06", 8900000.0),
    ("R10", "R10", 13500000.0),
    ("E11", "E11", 10200000.0),
    ("S06", "S06", 14800000.0),
    ("D01", "D01", 12000000.0),
]
for i, (d_code, p_code, amt) in enumerate(upcoding_pairs):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = f"2026-06-{10+i:02d}"
    rows.append({
        "claim_id": cid, "patient_id": f"P-UPCODE-A{i+1:02d}", "provider_id": "PRV-RS-UPCODE",
        "service_code": "SUR001", "procedure_code": p_code, "diagnosis_code": d_code,
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": amt, "paid_amount": round(amt * 0.85, 2), "allowed_amount": round(amt * 0.90, 2),
        "claim_status": "Under Review", "patient_age": 42 + i, "length_of_stay": 1, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt, "item_code": "SRV-SUR001", "item_name": "Tindakan Prosedur Upcoding",
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 1", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "Jawa Timur", "patient_city": "Surabaya", "provider_province": "Jawa Timur", "provider_city": "Surabaya",
        "membership_status": "Aktif", "fraud_type": "Upcoding", "is_fraud": 1
    })

# 4B: Diagnosa ringan dipasangkan dengan prosedur bedah mahal (15 baris)
mild_diag_pairs = [
    ("J00", "SUR001", 28500000.0, "Common Cold / Flu Ringan"),
    ("R51", "SUR001", 31000000.0, "Sakit Kepala / Cephalgia"),
    ("L20", "SUR001", 27000000.0, "Dermatitis Atopik"),
    ("K30", "SUR001", 29500000.0, "Dispepsia Ringan"),
    ("R05", "SUR001", 26500000.0, "Batuk Akut"),
    ("M79", "SUR001", 33000000.0, "Myalgia / Nyeri Otot"),
    ("H10", "SUR001", 28000000.0, "Konjungtivitis"),
    ("J02", "SUR001", 30500000.0, "Faringitis Akut"),
    ("K59", "SUR001", 32000000.0, "Konstipasi"),
    ("R42", "SUR001", 29000000.0, "Pusing / Vertigo Ringan"),
    ("L02", "SUR001", 34500000.0, "Kutaneus Abses Kecil"),
    ("N39", "SUR001", 27500000.0, "Infeksi Saluran Kemih Ringan"),
    ("T14", "SUR001", 35000000.0, "Cedera Ringan Terbuka"),
    ("K21", "SUR001", 31500000.0, "Refluks Gastroesofagus (GERD)"),
    ("R11", "SUR001", 28500000.0, "Mual dan Muntah"),
]
for i, (d_code, p_code, amt, diag_name) in enumerate(mild_diag_pairs):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = f"2026-06-{15 + (i % 12):02d}"
    rows.append({
        "claim_id": cid, "patient_id": f"P-UPCODE-B{i+1:02d}", "provider_id": "PRV-RS-UPCODE",
        "service_code": "SUR001", "procedure_code": p_code, "diagnosis_code": d_code,
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": amt, "paid_amount": round(amt * 0.85, 2), "allowed_amount": round(amt * 0.90, 2),
        "claim_status": "Settled", "patient_age": 30 + i * 2, "length_of_stay": 2, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt, "item_code": "SRV-SUR001", "item_name": f"Operasi Mayor untuk {diag_name}",
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 1", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "Jawa Timur", "patient_city": "Surabaya", "provider_province": "Jawa Timur", "provider_city": "Surabaya",
        "membership_status": "Aktif", "fraud_type": "Upcoding", "is_fraud": 1
    })

# ==============================================================================
# 5. INFLATED BILL & CLONING (26 baris)
# ==============================================================================
# 5A: Exact Cloned Records (3 set kloning, masing-masing 4 klaim identik pasien/provider/tanggal) (12 baris)
clone_groups = [
    ("P-CLON-01", "PRV-CLON-01", "2026-06-15", "K29", "US001", 4500000.0, 4),
    ("P-CLON-02", "PRV-CLON-02", "2026-06-18", "A01", "CT001", 5200000.0, 4),
    ("P-CLON-03", "PRV-CLON-01", "2026-06-20", "R10", "XR001", 3800000.0, 4),
]
for pat, prv, dt_str, diag, srv, amt, count in clone_groups:
    for c_idx in range(count):
        cid = make_claim_id(claim_counter)
        claim_counter += 1
        rows.append({
            "claim_id": cid, "patient_id": pat, "provider_id": prv,
            "service_code": srv, "procedure_code": srv, "diagnosis_code": diag,
            "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
            "billed_amount": amt, "paid_amount": round(amt * 0.85, 2), "allowed_amount": round(amt * 0.90, 2),
            "claim_status": "Settled", "patient_age": 48, "length_of_stay": 0, "quantity": 1,
            "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt, "item_code": f"SRV-{srv}", "item_name": f"Klaim Kloning {srv}",
            "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 2", "patient_segment": "Pekerja Penerima Upah",
            "patient_province": "Sumatera Utara", "patient_city": "Medan", "provider_province": "Sumatera Utara", "provider_city": "Medan",
            "membership_status": "Aktif", "fraud_type": "Inflated Bill / Cloning", "is_fraud": 1
        })

# 5B: Extreme Inflated Amount Outliers (Biaya sangat janggal untuk rawat jalan sederhana) (14 baris)
inflated_outliers = [
    ("P-INF-01", "CONS001", "A09", 48000000.0),
    ("P-INF-02", "XR001", "M54", 55000000.0),
    ("P-INF-03", "BT001", "K29", 42000000.0),
    ("P-INF-04", "US001", "R10", 62000000.0),
    ("P-INF-05", "CONS001", "J06", 49000000.0),
    ("P-INF-06", "XR001", "S06", 58000000.0),
    ("P-INF-07", "BT001", "A01", 45000000.0),
    ("P-INF-08", "LAB001", "D01", 68000000.0),
    ("P-INF-09", "CONS001", "R51", 52000000.0),
    ("P-INF-10", "US001", "K30", 61000000.0),
    ("P-INF-11", "BT001", "B02", 47000000.0),
    ("P-INF-12", "XR001", "M79", 54000000.0),
    ("P-INF-13", "CONS001", "J00", 50000000.0),
    ("P-INF-14", "LAB001", "R11", 65000000.0),
]
for i, (pat, srv, diag, amt) in enumerate(inflated_outliers):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = f"2026-06-{12 + (i % 15):02d}"
    rows.append({
        "claim_id": cid, "patient_id": pat, "provider_id": "PRV-INFLATE-MAX",
        "service_code": srv, "procedure_code": srv, "diagnosis_code": diag,
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": amt, "paid_amount": round(amt * 0.85, 2), "allowed_amount": round(amt * 0.90, 2),
        "claim_status": "Settled", "patient_age": 38 + i, "length_of_stay": 0, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt, "item_code": f"SRV-{srv}", "item_name": f"Tagihan Gelembung {srv}",
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 1", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DKI Jakarta", "patient_city": "Jakarta Selatan", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Selatan",
        "membership_status": "Aktif", "fraud_type": "Inflated Bill / Cloning", "is_fraud": 1
    })

# ==============================================================================
# 6. PROLONGED LENGTH OF STAY & READMISSION (24 baris)
# ==============================================================================
# 6A: Prolonged Stay Extreme Outliers (Lama rawat inap 42 - 58 hari untuk kasus umum) (10 baris)
prolonged_stays = [
    ("P-LOS-01", "PRV-HOSP-STAY", "K29", "2026-04-01", 44, 26000000.0),
    ("P-LOS-02", "PRV-HOSP-STAY", "A01", "2026-04-03", 48, 28500000.0),
    ("P-LOS-03", "PRV-HOSP-STAY", "A09", "2026-04-05", 42, 25000000.0),
    ("P-LOS-04", "PRV-HOSP-STAY", "J06", "2026-04-08", 50, 31000000.0),
    ("P-LOS-05", "PRV-HOSP-STAY", "R10", "2026-04-10", 46, 27500000.0),
    ("P-LOS-06", "PRV-HOSP-STAY", "K30", "2026-04-12", 52, 33000000.0),
    ("P-LOS-07", "PRV-HOSP-STAY", "B02", "2026-04-15", 45, 29000000.0),
    ("P-LOS-08", "PRV-HOSP-STAY", "R51", "2026-04-18", 55, 34500000.0),
    ("P-LOS-09", "PRV-HOSP-STAY", "M54", "2026-04-20", 43, 26500000.0),
    ("P-LOS-10", "PRV-HOSP-STAY", "A01", "2026-04-22", 58, 36000000.0),
]
for pat, prv, diag, d_adm_str, los, amt in prolonged_stays:
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    adm_dt = datetime.strptime(d_adm_str, "%Y-%m-%d")
    disch_dt = adm_dt + timedelta(days=los)
    disch_str = disch_dt.strftime("%Y-%m-%d")
    rows.append({
        "claim_id": cid, "patient_id": pat, "provider_id": prv,
        "service_code": "SUR001", "procedure_code": "SUR001", "diagnosis_code": diag,
        "billing_date": d_adm_str, "service_date": d_adm_str, "admission_date": d_adm_str, "discharge_date": disch_str,
        "billed_amount": amt, "paid_amount": round(amt * 0.85, 2), "allowed_amount": round(amt * 0.90, 2),
        "claim_status": "Settled", "patient_age": 50, "length_of_stay": los, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt, "item_code": "SRV-INPATIENT", "item_name": f"Rawat Inap Berkepanjangan {los} Hari",
        "provider_type": "RSUD Kelas B", "facility_type": "Rumah Sakit", "patient_class": "Kelas 2", "patient_segment": "PBI APBN",
        "patient_province": "Jawa Tengah", "patient_city": "Semarang", "provider_province": "Jawa Tengah", "provider_city": "Semarang",
        "membership_status": "Aktif", "fraud_type": "Prolonged Stay", "is_fraud": 1
    })

# 6B: Suspicious Readmission (Pasien dipulangkan dan masuk kembali dalam 2-5 hari dengan diagnosa persis sama) (14 baris / 7 pasang)
readmission_pairs = [
    ("P-READM-01", "PRV-HOSP-READM", "D01", "2026-05-01", 3, 2, "2026-05-06", 3, 5500000.0, 5800000.0),
    ("P-READM-02", "PRV-HOSP-READM", "K29", "2026-05-05", 2, 3, "2026-05-10", 3, 4800000.0, 5100000.0),
    ("P-READM-03", "PRV-HOSP-READM", "A01", "2026-05-08", 3, 2, "2026-05-13", 4, 6200000.0, 6500000.0),
    ("P-READM-04", "PRV-HOSP-READM", "R10", "2026-05-12", 2, 4, "2026-05-18", 3, 5100000.0, 5300000.0),
    ("P-READM-05", "PRV-HOSP-READM", "B02", "2026-05-15", 3, 2, "2026-05-20", 3, 5900000.0, 6100000.0),
    ("P-READM-06", "PRV-HOSP-READM", "J06", "2026-05-18", 2, 3, "2026-05-23", 2, 4700000.0, 4900000.0),
    ("P-READM-07", "PRV-HOSP-READM", "S06", "2026-05-22", 4, 2, "2026-05-28", 3, 7200000.0, 7500000.0),
]
for pat, prv, diag, adm1, los1, gap, adm2, los2, amt1, amt2 in readmission_pairs:
    disch1 = (datetime.strptime(adm1, "%Y-%m-%d") + timedelta(days=los1)).strftime("%Y-%m-%d")
    disch2 = (datetime.strptime(adm2, "%Y-%m-%d") + timedelta(days=los2)).strftime("%Y-%m-%d")
    
    # Kunjungan 1
    cid1 = make_claim_id(claim_counter)
    claim_counter += 1
    rows.append({
        "claim_id": cid1, "patient_id": pat, "provider_id": prv,
        "service_code": "SUR001", "procedure_code": "SUR001", "diagnosis_code": diag,
        "billing_date": adm1, "service_date": adm1, "admission_date": adm1, "discharge_date": disch1,
        "billed_amount": amt1, "paid_amount": round(amt1 * 0.85, 2), "allowed_amount": round(amt1 * 0.90, 2),
        "claim_status": "Settled", "patient_age": 45, "length_of_stay": los1, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt1, "item_code": "SRV-INPATIENT", "item_name": "Rawat Inap Episode 1",
        "provider_type": "RSUD Kelas B", "facility_type": "Rumah Sakit", "patient_class": "Kelas 2", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DI Yogyakarta", "patient_city": "Sleman", "provider_province": "DI Yogyakarta", "provider_city": "Sleman",
        "membership_status": "Aktif", "fraud_type": "Prolonged Stay", "is_fraud": 1
    })
    # Kunjungan 2 (Readmisi mencurigakan)
    cid2 = make_claim_id(claim_counter)
    claim_counter += 1
    rows.append({
        "claim_id": cid2, "patient_id": pat, "provider_id": prv,
        "service_code": "SUR001", "procedure_code": "SUR001", "diagnosis_code": diag,
        "billing_date": adm2, "service_date": adm2, "admission_date": adm2, "discharge_date": disch2,
        "billed_amount": amt2, "paid_amount": round(amt2 * 0.85, 2), "allowed_amount": round(amt2 * 0.90, 2),
        "claim_status": "Settled", "patient_age": 45, "length_of_stay": los2, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt2, "item_code": "SRV-INPATIENT", "item_name": "Readmisi Singkat Diagnosa Sama",
        "provider_type": "RSUD Kelas B", "facility_type": "Rumah Sakit", "patient_class": "Kelas 2", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DI Yogyakarta", "patient_city": "Sleman", "provider_province": "DI Yogyakarta", "provider_city": "Sleman",
        "membership_status": "Aktif", "fraud_type": "Prolonged Stay", "is_fraud": 1
    })

# ==============================================================================
# 7. MEDICATION & DEVICE FRAUD (24 baris)
# ==============================================================================
# 7A: Diskrepansi Kuantitas: quantity_billed jauh melebihi quantity_delivered (> 30%) (8 baris)
qty_discrepancy = [
    ("P-MED-01", 30, 5, 85000.0, "MED-ALB20", "Human Albumin 20% 100ml"),
    ("P-MED-02", 25, 4, 120000.0, "MED-MER1G", "Meropenem 1g Injeksi"),
    ("P-MED-03", 20, 3, 95000.0, "MED-ENOX40", "Enoxaparin 40mg/0.4ml"),
    ("P-MED-04", 40, 8, 75000.0, "DEV-CATH01", "Foley Catheter 3-way Silicone"),
    ("P-MED-05", 35, 6, 110000.0, "DEV-SYRINGE", "Infusion Pump Set Disposable"),
    ("P-MED-06", 28, 5, 80000.0, "MED-CEF1G", "Ceftriaxone 1g Injeksi"),
    ("P-MED-07", 50, 10, 65000.0, "DEV-CVP01", "Central Venous Catheter 3-Lumen"),
    ("P-MED-08", 22, 4, 130000.0, "MED-OMEP40", "Omeprazole 40mg Injeksi"),
]
for i, (pat, q_bill, q_del, u_prc, code, name) in enumerate(qty_discrepancy):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    amt = q_bill * u_prc
    dt_str = f"2026-06-{10+i:02d}"
    rows.append({
        "claim_id": cid, "patient_id": pat, "provider_id": "PRV-PHARM-FRAUD",
        "service_code": "LAB001", "procedure_code": "LAB001", "diagnosis_code": "A01",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": amt, "paid_amount": round(amt * 0.85, 2), "allowed_amount": round(amt * 0.90, 2),
        "claim_status": "Settled", "patient_age": 52, "length_of_stay": 0, "quantity": q_bill,
        "quantity_billed": q_bill, "quantity_delivered": q_del, "unit_price": u_prc, "item_code": code, "item_name": name,
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 1", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DKI Jakarta", "patient_city": "Jakarta Barat", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Barat",
        "membership_status": "Aktif", "fraud_type": "Medication / Device", "is_fraud": 1
    })

# 7B: Mark-up Harga Ekstrem (Unit price > 2.5x median) (8 baris)
price_markup = [
    ("P-MED-09", 10, 10, 850000.0, 8500000.0, "MED-GEN01", "Paracetamol Infus 1000mg Mark-up"),
    ("P-MED-10", 8, 8, 920000.0, 7360000.0, "MED-GEN02", "Ranitidine Injeksi Mark-up"),
    ("P-MED-11", 12, 12, 780000.0, 9360000.0, "MED-GEN03", "Ketorolac 30mg Injeksi Mark-up"),
    ("P-MED-12", 6, 6, 950000.0, 5700000.0, "MED-GEN04", "Ondansetron 8mg Injeksi Mark-up"),
    ("P-MED-13", 10, 10, 880000.0, 8800000.0, "DEV-GEN01", "Spuit 50cc Perfusion Mark-up"),
    ("P-MED-14", 5, 5, 990000.0, 4950000.0, "DEV-GEN02", "Nasal Cannula Dewasa Mark-up"),
    ("P-MED-15", 8, 8, 820000.0, 6560000.0, "MED-GEN05", "Metronidazole Infus Mark-up"),
    ("P-MED-16", 15, 15, 750000.0, 11250000.0, "DEV-GEN03", "Blood Transfusion Set Mark-up"),
]
for i, (pat, q_bill, q_del, u_prc, amt, code, name) in enumerate(price_markup):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = f"2026-06-{18+i:02d}"
    rows.append({
        "claim_id": cid, "patient_id": pat, "provider_id": "PRV-PHARM-FRAUD",
        "service_code": "LAB001", "procedure_code": "LAB001", "diagnosis_code": "K29",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": amt, "paid_amount": round(amt * 0.85, 2), "allowed_amount": round(amt * 0.90, 2),
        "claim_status": "Settled", "patient_age": 44, "length_of_stay": 0, "quantity": q_bill,
        "quantity_billed": q_bill, "quantity_delivered": q_del, "unit_price": u_prc, "item_code": code, "item_name": name,
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 2", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DKI Jakarta", "patient_city": "Jakarta Barat", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Barat",
        "membership_status": "Aktif", "fraud_type": "Medication / Device", "is_fraud": 1
    })

# 7C: Inconsistent Claim Amount Calculation (Amount billed != quantity_billed * unit_price, selisih > 40%) (8 baris)
calc_inconsistent = [
    ("P-MED-17", 5, 5, 100000.0, 4800000.0, "MED-INS01", "Insulin Pen Inconsistent Charge"),
    ("P-MED-18", 4, 4, 150000.0, 5200000.0, "MED-HEPA01", "Hepatitis Vaccine Inconsistent Charge"),
    ("P-MED-19", 6, 6, 80000.0, 3900000.0, "DEV-STENT01", "Stent Accessories Inconsistent Charge"),
    ("P-MED-20", 3, 3, 200000.0, 4500000.0, "MED-CHEMO01", "Antiemetic Inconsistent Charge"),
    ("P-MED-21", 5, 5, 120000.0, 4200000.0, "MED-DIAL01", "Dialysis Solution Inconsistent Charge"),
    ("P-MED-22", 4, 4, 90000.0, 3600000.0, "DEV-FILTER01", "Blood Filter Inconsistent Charge"),
    ("P-MED-23", 8, 8, 70000.0, 4600000.0, "MED-ANTIB01", "Ciprofloxacin Inconsistent Charge"),
    ("P-MED-24", 5, 5, 110000.0, 4100000.0, "DEV-ELECT01", "ECG Electrodes Inconsistent Charge"),
]
for i, (pat, q_bill, q_del, u_prc, amt, code, name) in enumerate(calc_inconsistent):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = f"2026-06-{24 + (i % 6):02d}"
    rows.append({
        "claim_id": cid, "patient_id": pat, "provider_id": "PRV-PHARM-FRAUD",
        "service_code": "LAB001", "procedure_code": "LAB001", "diagnosis_code": "E11",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": amt, "paid_amount": round(amt * 0.85, 2), "allowed_amount": round(amt * 0.90, 2),
        "claim_status": "Settled", "patient_age": 58, "length_of_stay": 0, "quantity": q_bill,
        "quantity_billed": q_bill, "quantity_delivered": q_del, "unit_price": u_prc, "item_code": code, "item_name": name,
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 2", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DKI Jakarta", "patient_city": "Jakarta Barat", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Barat",
        "membership_status": "Aktif", "fraud_type": "Medication / Device", "is_fraud": 1
    })

# ==============================================================================
# 8. DUPLICATE PAYMENT (15 baris)
# ==============================================================================
# Klaim dengan status "Paid", di mana klaim duplikat diajukan kembali ke sistem
for i in range(15):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    dt_str = f"2026-05-{10 + i:02d}"
    amt = 1750000.0 + (i * 50000.0)
    rows.append({
        "claim_id": cid, "patient_id": f"P-DUP-PAY-{i+1:02d}", "provider_id": "PRV-DUP-PAY",
        "service_code": "CT001", "procedure_code": "CT001", "diagnosis_code": "A01",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": amt, "paid_amount": amt, "allowed_amount": amt,
        "claim_status": "Paid", "patient_age": 36 + i, "length_of_stay": 0, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt, "item_code": "SRV-CT001", "item_name": "CT Scan Status Sudah Terbayar",
        "provider_type": "RS Swasta", "facility_type": "Rumah Sakit", "patient_class": "Kelas 2", "patient_segment": "Pekerja Penerima Upah",
        "patient_province": "DKI Jakarta", "patient_city": "Jakarta Selatan", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Selatan",
        "membership_status": "Aktif", "fraud_type": "Duplicate Payment", "is_fraud": 1
    })

# ==============================================================================
# 9. COLLUSION & NETWORK GRAPH ANOMALY (25 baris)
# ==============================================================================
# Jaringan sindikat 8 pasien yang selalu berkunjung ke faskes nakal PRV-ROGUE-NET01
# dalam frekuensi rapat dengan tagihan berulang dan klaim terkoordinasi
rogue_patients = [f"P-COL-{i+1:02d}" for i in range(8)]
for i in range(25):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    pat = rogue_patients[i % 8]
    srv = ["SUR001", "CT001", "US001"][i % 3]
    amt = 8500000.0 if srv == "SUR001" else (2600000.0 if srv == "CT001" else 650000.0)
    dt_str = f"2026-06-{1 + (i % 28):02d}"
    rows.append({
        "claim_id": cid, "patient_id": pat, "provider_id": "PRV-ROGUE-NET01",
        "service_code": srv, "procedure_code": srv, "diagnosis_code": "K29",
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": dt_str,
        "billed_amount": amt, "paid_amount": round(amt * 0.85, 2), "allowed_amount": round(amt * 0.90, 2),
        "claim_status": "Settled", "patient_age": 30 + (i % 15), "length_of_stay": 1 if srv == "SUR001" else 0, "quantity": 1,
        "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt, "item_code": f"SRV-{srv}", "item_name": f"Layanan Jaringan Kolusi {srv}",
        "provider_type": "Klinik Utama", "facility_type": "Klinik", "patient_class": "Kelas 2", "patient_segment": "PBPU Mandiri",
        "patient_province": "Jawa Barat", "patient_city": "Bekasi", "provider_province": "Jawa Barat", "provider_city": "Bekasi",
        "membership_status": "Aktif", "fraud_type": "Collusion / Graph Network", "is_fraud": 1
    })

# ==============================================================================
# 10. LEGITIMATE / NORMAL BASELINE CONTROL CLAIMS
# Sisa baris untuk mencapai tepat 400 baris (400 - jumlah baris saat ini)
# ==============================================================================
current_total = len(rows)
remaining = 400 - current_total

# Pembagian data normal:
# A. Kasus Kronis Ter-whitelist (Diabetes E11, Hipertensi I10, Ginjal N18) -> uji whitelist suppression (~45 baris)
# B. Kasus Rawat Jalan Standar (~80 baris)
# C. Kasus Rawat Inap Standar (LOS 2-4 hari) (~remaining - 125 baris)

# 10A: Kasus Kronis Ter-whitelist (45 baris / 15 pasien x 3 kunjungan berjarak ~30 hari)
chronic_configs = [
    ("P-CHRON-01", "PRV-NORM-01", "BT001", "E11", "Diabetes Melitus Tipe 2", 55),
    ("P-CHRON-02", "PRV-NORM-02", "CONS001", "E11", "Diabetes Melitus Tipe 2", 60),
    ("P-CHRON-03", "PRV-NORM-03", "LAB001", "E11", "Diabetes Melitus Tipe 2", 58),
    ("P-CHRON-04", "PRV-NORM-01", "CONS001", "I10", "Hipertensi Esensial", 62),
    ("P-CHRON-05", "PRV-NORM-02", "BT001", "I10", "Hipertensi Esensial", 65),
    ("P-CHRON-06", "PRV-NORM-04", "CONS001", "I10", "Hipertensi Esensial", 59),
    ("P-CHRON-07", "PRV-NORM-01", "LAB001", "N18", "Gagal Ginjal Kronis", 54),
    ("P-CHRON-08", "PRV-NORM-03", "CONS001", "N18", "Gagal Ginjal Kronis", 61),
    ("P-CHRON-09", "PRV-NORM-04", "LAB001", "N18", "Gagal Ginjal Kronis", 67),
    ("P-CHRON-10", "PRV-NORM-02", "BT001", "E10", "Diabetes Melitus Tipe 1", 32),
    ("P-CHRON-11", "PRV-NORM-05", "CONS001", "J45", "Asma Bronkial Rutin", 41),
    ("P-CHRON-12", "PRV-NORM-01", "US001", "M54", "Nyeri Punggung Kronis", 53),
    ("P-CHRON-13", "PRV-NORM-03", "XR001", "I11", "Penyakit Jantung Hipertensi", 68),
    ("P-CHRON-14", "PRV-NORM-04", "CONS001", "J46", "PPOK Kronis", 64),
    ("P-CHRON-15", "PRV-NORM-05", "LAB001", "N18", "Gagal Ginjal Kronis Rutin", 57),
]

for pat, prv, srv, diag, name, age in chronic_configs:
    base_dates = ["2026-04-10", "2026-05-12", "2026-06-15"]
    amt = 250000.0 if srv == "CONS001" else (350000.0 if srv == "BT001" else (400000.0 if srv == "LAB001" else 550000.0))
    for d_str in base_dates:
        cid = make_claim_id(claim_counter)
        claim_counter += 1
        rows.append({
            "claim_id": cid, "patient_id": pat, "provider_id": prv,
            "service_code": srv, "procedure_code": srv, "diagnosis_code": diag,
            "billing_date": d_str, "service_date": d_str, "admission_date": d_str, "discharge_date": d_str,
            "billed_amount": amt, "paid_amount": round(amt * 0.85, 2), "allowed_amount": round(amt * 0.90, 2),
            "claim_status": "Settled", "patient_age": age, "length_of_stay": 0, "quantity": 1,
            "quantity_billed": 1, "quantity_delivered": 1, "unit_price": amt, "item_code": f"SRV-{srv}", "item_name": f"Kontrol Rutin {name}",
            "provider_type": "RSUD Kelas B", "facility_type": "Rumah Sakit", "patient_class": "Kelas 2", "patient_segment": "PBI APBN",
            "patient_province": "DKI Jakarta", "patient_city": "Jakarta Timur", "provider_province": "DKI Jakarta", "provider_city": "Jakarta Timur",
            "membership_status": "Aktif", "fraud_type": "Normal / Legitimate (Whitelisted Chronic)", "is_fraud": 0
        })

# 10B & 10C: Rawat Jalan dan Rawat Inap Normal Sesuai Benchmark
current_total = len(rows)
remaining = 400 - current_total

normal_services = [
    ("CONS001", "CONS001", "Konsultasi Dokter Spesialis", 200000.0, 350000.0, 0),
    ("BT001", "BT001", "Pemeriksaan Darah Rutin", 150000.0, 280000.0, 0),
    ("XR001", "XR001", "Rontgen Thorax Standar", 250000.0, 420000.0, 0),
    ("US001", "US001", "USG Abdomen Standar", 350000.0, 600000.0, 0),
    ("LAB001", "LAB001", "Pemeriksaan Kimia Darah", 220000.0, 380000.0, 0),
    ("SUR001", "SUR001", "Rawat Inap Bedah Appendectomy", 4500000.0, 7500000.0, 3),
    ("SUR001", "SUR001", "Rawat Inap Bedah Hernia Repair", 5000000.0, 8000000.0, 4),
    ("CONS001", "CONS001", "Rawat Inap Penyakit Dalam Typhoid", 3200000.0, 5200000.0, 3),
    ("CONS001", "CONS001", "Rawat Inap Gastroenteritis Akut", 2800000.0, 4600000.0, 2),
]

normal_diags = ["A01", "A09", "K29", "J06", "R10", "K35", "K40", "M54", "J00", "R51", "B02", "D01"]
normal_cities = [
    ("DKI Jakarta", "Jakarta Selatan"),
    ("DKI Jakarta", "Jakarta Timur"),
    ("Jawa Barat", "Bandung"),
    ("Jawa Barat", "Bekasi"),
    ("Jawa Tengah", "Semarang"),
    ("DI Yogyakarta", "Yogyakarta"),
    ("Jawa Timur", "Surabaya"),
    ("Banten", "Tangerang"),
    ("Sumatera Utara", "Medan"),
]

for idx in range(remaining):
    cid = make_claim_id(claim_counter)
    claim_counter += 1
    
    pat_id = f"P-NORM-{idx+1:04d}"
    prv_id = f"PRV-NORM-{(idx % 8) + 1:02d}"
    
    srv_tuple = normal_services[idx % len(normal_services)]
    srv_code, prc_code, srv_name, min_amt, max_amt, los = srv_tuple
    
    diag_code = normal_diags[idx % len(normal_diags)]
    prov, city = normal_cities[idx % len(normal_cities)]
    
    # Tanggal menyebar dari April - Juli 2026
    month = 4 + (idx % 4)
    day = 1 + (idx % 28)
    dt_str = f"2026-{month:02d}-{day:02d}"
    
    base_dt = datetime.strptime(dt_str, "%Y-%m-%d")
    disch_str = (base_dt + timedelta(days=los)).strftime("%Y-%m-%d")
    
    # Biaya normal sesuai rentang
    amt = round(random.uniform(min_amt, max_amt), -3)
    paid = round(amt * 0.85, 2)
    allowed = round(amt * 0.90, 2)
    
    status = "Settled" if (idx % 5 != 0) else "Pending"
    age = random.randint(18, 72)
    
    u_price = amt if los == 0 else round(amt / max(los, 1), 2)
    qty = 1 if los == 0 else los
    
    rows.append({
        "claim_id": cid, "patient_id": pat_id, "provider_id": prv_id,
        "service_code": srv_code, "procedure_code": prc_code, "diagnosis_code": diag_code,
        "billing_date": dt_str, "service_date": dt_str, "admission_date": dt_str, "discharge_date": disch_str,
        "billed_amount": amt, "paid_amount": paid, "allowed_amount": allowed,
        "claim_status": status, "patient_age": age, "length_of_stay": los, "quantity": qty,
        "quantity_billed": qty, "quantity_delivered": qty, "unit_price": u_price, "item_code": f"SRV-{srv_code}", "item_name": srv_name,
        "provider_type": "RSUD Kelas B" if los > 0 else "Klinik Pratama", "facility_type": "Rumah Sakit" if los > 0 else "Klinik",
        "patient_class": "Kelas 2" if los > 0 else "Rawat Jalan", "patient_segment": "Pekerja Penerima Upah" if idx % 2 == 0 else "PBI APBN",
        "patient_province": prov, "patient_city": city, "provider_province": prov, "provider_city": city,
        "membership_status": "Aktif", "fraud_type": "Normal / Legitimate", "is_fraud": 0
    })

# Tulis ke CSV
with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Berhasil menghasilkan dataset dummy: {OUTPUT_FILE}")
print(f"Total baris: {len(rows)}")

# Ringkasan per kategori fraud
from collections import Counter
type_counts = Counter(r["fraud_type"] for r in rows)
print("\nDistribusi Kategori:")
for f_type, count in type_counts.most_common():
    print(f" - {f_type}: {count} baris ({count/len(rows)*100:.1f}%)")
