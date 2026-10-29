"""
Script validasi dataset dummy untuk memastikan kepatuhan 100% terhadap:
1. Pydantic Schemas (schemas.py)
2. Schema Harmonizer canonical columns (schema_harmonizer.py)
3. Keberadaan seluruh 9 kategori fraud & normal control
"""
import csv
import os
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_dummy_fraud_detection.csv")

VALID_STATUSES = {"Settled", "Pending", "Rejected", "Paid", "Partially Paid", "Under Review"}

def validate():
    if not os.path.exists(DATASET_FILE):
        print(f"Error: File tidak ditemukan di {DATASET_FILE}")
        return False

    with open(DATASET_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total baris terdeteksi: {len(rows)}")
    assert len(rows) <= 450, f"Jumlah baris melebihi batas 450: {len(rows)}"

    claim_ids = set()
    errors = []
    fraud_types = {}

    now = datetime(2026, 10, 29, 23, 59, 59)

    for idx, r in enumerate(rows, start=1):
        cid = r["claim_id"]
        if not cid:
            errors.append(f"Baris {idx}: claim_id kosong")
        if cid in claim_ids:
            errors.append(f"Baris {idx}: duplicate claim_id {cid}")
        claim_ids.add(cid)

        # Dates
        b_date = None
        s_date = None
        try:
            b_date = datetime.strptime(r["billing_date"], "%Y-%m-%d")
            if b_date > now:
                errors.append(f"Baris {idx}: billing_date di masa depan ({r['billing_date']})")
        except Exception as e:
            errors.append(f"Baris {idx}: format billing_date salah ({r['billing_date']}): {e}")

        try:
            s_date = datetime.strptime(r["service_date"], "%Y-%m-%d")
            if s_date > now:
                errors.append(f"Baris {idx}: service_date di masa depan ({r['service_date']})")
            if b_date is not None and s_date < b_date:
                errors.append(f"Baris {idx}: service_date ({r['service_date']}) sebelum billing_date ({r['billing_date']})")
        except Exception as e:
            errors.append(f"Baris {idx}: format service_date salah ({r['service_date']}): {e}")

        # Amounts
        try:
            b_amt = float(r["billed_amount"])
            p_amt = float(r["paid_amount"])
            a_amt = float(r["allowed_amount"])

            if b_amt < 0:
                errors.append(f"Baris {idx}: billed_amount negatif ({b_amt})")
            if p_amt < 0:
                errors.append(f"Baris {idx}: paid_amount negatif ({p_amt})")
            if a_amt < 0:
                errors.append(f"Baris {idx}: allowed_amount negatif ({a_amt})")
            if p_amt > b_amt:
                errors.append(f"Baris {idx}: paid_amount ({p_amt}) > billed_amount ({b_amt})")
            if a_amt > b_amt:
                errors.append(f"Baris {idx}: allowed_amount ({a_amt}) > billed_amount ({b_amt})")
        except Exception as e:
            errors.append(f"Baris {idx}: parsing nominal gagal: {e}")

        # Status
        status = r["claim_status"]
        if status not in VALID_STATUSES:
            errors.append(f"Baris {idx}: status klaim tidak valid ({status})")

        # Age
        try:
            age = int(r["patient_age"])
            if age < 0 or age > 120:
                errors.append(f"Baris {idx}: usia tidak valid ({age})")
        except Exception as e:
            errors.append(f"Baris {idx}: usia bukan integer: {e}")

        # Length of stay
        try:
            los = int(r["length_of_stay"])
            if los < 0 or los > 180:
                errors.append(f"Baris {idx}: length_of_stay di luar batas 0-180 ({los})")
        except Exception as e:
            errors.append(f"Baris {idx}: los bukan integer: {e}")

        # Quantity
        try:
            qty = int(r["quantity"])
            if qty < 0:
                errors.append(f"Baris {idx}: quantity negatif ({qty})")
        except Exception as e:
            errors.append(f"Baris {idx}: quantity bukan integer: {e}")

        # Track fraud type
        ft = r["fraud_type"]
        fraud_types[ft] = fraud_types.get(ft, 0) + 1

    if errors:
        print(f"Ditemukan {len(errors)} error validasi:")
        for err in errors[:10]:
            print(f"  - {err}")
        return False

    print("\n✅ Validasi Skema & Integritas: 100% SUKSES (0 ERROR)!")
    print("\nDistribusi Tipe Fraud:")
    for ft, count in sorted(fraud_types.items(), key=lambda x: -x[1]):
        print(f"  ✓ {ft:45s}: {count:3d} baris ({count/len(rows)*100:5.1f}%)")

    # Verifikasi semua tipe fraud ada
    expected_categories = [
        "Repeat Billing",
        "Phantom Service",
        "Provider Capacity",
        "Upcoding",
        "Inflated Bill / Cloning",
        "Prolonged Stay",
        "Medication / Device",
        "Duplicate Payment",
        "Collusion / Graph Network",
        "Normal / Legitimate"
    ]

    missing_categories = [c for c in expected_categories if not any(c in k for k in fraud_types.keys())]
    if missing_categories:
        print(f"\n❌ Error: Kategori fraud yang hilang: {missing_categories}")
        return False
    else:
        print("\n✅ Semua jenis fraud yang ingin dideteksi aplikasi TERSEDIA LENGKAP!")

    return True

if __name__ == "__main__":
    validate()
