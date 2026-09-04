"""
Schema Harmonizer module for ASTINA.

Provides:
1. Semantic column aliasing (Indonesian, English, and healthcare standards).
2. Deterministic feature derivation (amounts, dates, LOS, ratios).
3. Metadata provenance tagging to protect against false positives.
4. Business rule readiness evaluation (Circuit Breaker prerequisites check).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("astina.schema_harmonizer")

# Canonical aliases mapping: Canonical Name -> List of potential aliases (lowercase, trimmed)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "claim_id": [
        "no_klaim", "id_klaim", "claim_no", "nomor_klaim", "claim_number",
        "claimid", "claim", "no_tagihan", "id_transaksi", "nomor_transaksi",
    ],
    "patient_id": [
        "no_peserta", "nomor_kartu", "id_pasien", "no_kartu", "patient_number",
        "patientid", "nik", "patient", "id_peserta", "nomor_peserta", "no_rekam_medis", "no_rm",
    ],
    "provider_id": [
        "kode_faskes", "id_provider", "kode_rs", "hospital_id", "provider_code",
        "providerid", "provider", "kode_provider", "id_faskes", "faskes_id",
        "nama_faskes", "kode_klinik",
    ],
    "service_code": [
        "kode_tindakan", "kode_prosedur", "procedure_code", "kode_layanan",
        "servicecode", "service", "cpt", "cpt_code", "kode_tarif", "tindakan",
    ],
    "diagnosis_code": [
        "diagnosa", "kode_icd", "icd10", "icd_10", "diagnosa_utama",
        "diagnosis", "diagnosiscode", "primary_diagnosis", "icd_code",
        "kode_diagnosa", "diagnosa_primer",
    ],
    "billing_date": [
        "tgl_klaim", "tgl_tagihan", "tanggal_klaim", "tgl_billing",
        "billingdate", "claim_date", "bill_date", "tanggal_tagihan",
        "tgl_pengajuan", "tanggal_pengajuan",
    ],
    "service_date": [
        "tgl_pelayanan", "tgl_masuk", "tanggal_layanan", "tgl_tindakan",
        "servicedate", "date_of_service", "treatment_date", "tanggal_masuk",
        "tgl_rawat", "tanggal_tindakan",
    ],
    "billed_amount": [
        "amount", "biaya_tagihan", "total_tagihan", "biaya_klaim",
        "nominal_klaim", "billedamount", "tagihan", "biaya",
        "total_biaya", "tarif_tagihan", "total_klaim",
    ],
    "paid_amount": [
        "biaya_dibayar", "nominal_bayar", "jumlah_bayar", "total_bayar",
        "paidamount", "dibayar", "tarif_riil", "realisasi_bayar",
    ],
    "allowed_amount": [
        "biaya_disetujui", "nominal_setuju", "plafon", "allowedamount",
        "disetujui", "tarif_disetujui", "klaim_disetujui",
    ],
    "claim_status": [
        "status", "status_klaim", "status_pembayaran", "claimstatus",
        "status_tagihan", "approval_status",
    ],
    "patient_age": [
        "umur", "usia", "umur_pasien", "patientage", "age",
        "usia_pasien", "tahun_usia",
    ],
    "length_of_stay": [
        "lama_rawat", "hari_rawat", "los", "lengthofstay",
        "lama_inap", "lama_hari_rawat", "durasi_rawat",
    ],
    "quantity": [
        "jumlah", "banyaknya", "qty", "volume",
        "quantity_billed", "kuantitas", "jumlah_tindakan", "unit",
    ],
    "admission_date": [
        "tgl_masuk_rs", "tanggal_masuk_rs", "tgl_admisi", "admissiondate",
        "adm_date", "tgl_checkin",
    ],
    "discharge_date": [
        "tgl_keluar_rs", "tanggal_keluar_rs", "tgl_pulang", "dischargedate",
        "disch_date", "tgl_checkout",
    ],
}

_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in COLUMN_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower().strip()] = canonical


def _clean_numeric_series(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Robust numeric cleaner that strips currency symbols, commas, and handles invalid strings."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    
    # Process string representation: remove spaces, $, Rp, commas
    cleaned = series.astype(str).str.replace(r"[^\d.-]", "", regex=True)
    cleaned = cleaned.mask(cleaned.str.strip() == "", np.nan)
    return pd.to_numeric(cleaned, errors="coerce").fillna(default)


class SchemaHarmonizer:
    """Intelligent schema harmonizer, alias resolver, and deterministic feature synthesizer."""

    @staticmethod
    def resolve_aliases(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Detect and resolve column aliases, mapping them to canonical names.
        Returns the updated DataFrame and a mapping of {original_col: canonical_col}.
        """
        if df is None or df.empty:
            return df.copy() if df is not None else pd.DataFrame(), {}

        clean = df.copy()
        resolved: Dict[str, str] = {}

        for col in clean.columns:
            col_lower = f"{col}".lower().strip()
            if col in COLUMN_ALIASES:
                continue

            canonical = _ALIAS_TO_CANONICAL.get(col_lower)
            if canonical and canonical not in clean.columns:
                clean[canonical] = clean[col]
                resolved[col] = canonical
                logger.info(f"Harmonized alias: '{col}' -> '{canonical}'")

        return clean, resolved

    @staticmethod
    def harmonize_claims_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive harmonization:
        1. Resolves aliases.
        2. Injects stable identifier `_astina_row_id`.
        3. Performs safe deterministic derivations (dates, LOS, amounts, ratios).
        4. Injects safe heuristic defaults with provenance tags.

        Returns:
            (harmonized_df, harmonization_report)
        """
        if df is None:
            raise ValueError("DataFrame klaim tidak boleh None")

        canonical_core = [
            "claim_id", "patient_id", "provider_id", "service_code",
            "diagnosis_code", "billing_date", "service_date", "amount",
            "billed_amount", "paid_amount", "allowed_amount", "claim_status",
            "patient_age", "length_of_stay", "quantity"
        ]

        if df.empty:
            clean = df.copy()
            for col in canonical_core:
                if col not in clean.columns:
                    if col in ["billing_date", "service_date"]:
                        clean[col] = pd.Series(dtype="datetime64[ns]")
                    elif col in ["amount", "billed_amount", "paid_amount", "allowed_amount"]:
                        clean[col] = pd.Series(dtype="float64")
                    elif col in ["patient_age", "length_of_stay", "quantity"]:
                        clean[col] = pd.Series(dtype="int64")
                    else:
                        clean[col] = pd.Series(dtype="object")
            return clean, {
                "resolved_aliases": {},
                "derived_columns": [],
                "imputed_columns": [],
            }

        clean, resolved_aliases = SchemaHarmonizer.resolve_aliases(df)
        derived_cols: List[str] = []
        imputed_cols: List[str] = []

        # 1. Stable row ID
        if "_astina_row_id" not in clean.columns:
            clean["_astina_row_id"] = pd.Series(np.arange(len(clean), dtype=np.int64), index=clean.index)

        # 2. Synchronize amount <-> billed_amount
        has_billed = "billed_amount" in clean.columns and clean["billed_amount"].notna().any()
        has_amount = "amount" in clean.columns and clean["amount"].notna().any()

        if has_billed and not has_amount:
            clean["amount"] = _clean_numeric_series(clean["billed_amount"])
            clean["billed_amount"] = clean["amount"]
            derived_cols.append("amount")
        elif has_amount and not has_billed:
            clean["amount"] = _clean_numeric_series(clean["amount"])
            clean["billed_amount"] = clean["amount"]
            derived_cols.append("billed_amount")
        elif "amount" in clean.columns or "billed_amount" in clean.columns:
            if "amount" in clean.columns:
                clean["amount"] = _clean_numeric_series(clean["amount"])
            if "billed_amount" in clean.columns:
                clean["billed_amount"] = _clean_numeric_series(clean["billed_amount"])
            if "amount" not in clean.columns:
                clean["amount"] = clean["billed_amount"]
                derived_cols.append("amount")
            if "billed_amount" not in clean.columns:
                clean["billed_amount"] = clean["amount"]
                derived_cols.append("billed_amount")
        else:
            clean["amount"] = 0.0
            clean["billed_amount"] = 0.0
            imputed_cols.extend(["amount", "billed_amount"])

        # 3. Synchronize Temporal Fields: billing_date & service_date
        has_billing = "billing_date" in clean.columns and clean["billing_date"].notna().any()
        has_service = "service_date" in clean.columns and clean["service_date"].notna().any()

        if has_billing and not has_service:
            clean["billing_date"] = pd.to_datetime(clean["billing_date"], errors="coerce")
            clean["service_date"] = clean["billing_date"].copy()
            derived_cols.append("service_date")
        elif has_service and not has_billing:
            clean["service_date"] = pd.to_datetime(clean["service_date"], errors="coerce")
            clean["billing_date"] = clean["service_date"].copy()
            derived_cols.append("billing_date")
        else:
            if "billing_date" not in clean.columns:
                clean["billing_date"] = pd.NaT
                imputed_cols.append("billing_date")
            else:
                clean["billing_date"] = pd.to_datetime(clean["billing_date"], errors="coerce")

            if "service_date" not in clean.columns:
                clean["service_date"] = pd.NaT
                imputed_cols.append("service_date")
            else:
                clean["service_date"] = pd.to_datetime(clean["service_date"], errors="coerce")

        # 4. Length of Stay (LOS), admission_date, discharge_date
        has_adm = "admission_date" in clean.columns and clean["admission_date"].notna().any()
        has_disch = "discharge_date" in clean.columns and clean["discharge_date"].notna().any()
        has_los = "length_of_stay" in clean.columns and clean["length_of_stay"].notna().any()

        if has_adm and has_disch:
            clean["admission_date"] = pd.to_datetime(clean["admission_date"], errors="coerce")
            clean["discharge_date"] = pd.to_datetime(clean["discharge_date"], errors="coerce")
            if not has_los:
                date_delta = clean["discharge_date"] - clean["admission_date"]
                clean["length_of_stay"] = (date_delta / np.timedelta64(1, "D")).fillna(0).clip(lower=0).astype(int)
                derived_cols.append("length_of_stay")
            else:
                clean["length_of_stay"] = _clean_numeric_series(clean["length_of_stay"]).astype(int)
        elif has_los:
            clean["length_of_stay"] = _clean_numeric_series(clean["length_of_stay"]).astype(int)
            base_date = clean["service_date"] if clean["service_date"].notna().any() else clean["billing_date"]
            if base_date.notna().any():
                if "admission_date" not in clean.columns:
                    clean["admission_date"] = base_date
                    derived_cols.append("admission_date")
                if "discharge_date" not in clean.columns:
                    clean["discharge_date"] = base_date + pd.to_timedelta(clean["length_of_stay"], unit="D")
                    derived_cols.append("discharge_date")
        else:
            if "length_of_stay" not in clean.columns:
                clean["length_of_stay"] = 0
                imputed_cols.append("length_of_stay")

        # 5. Financial ratios
        if "paid_amount" in clean.columns and clean["paid_amount"].notna().any():
            clean["paid_amount"] = _clean_numeric_series(clean["paid_amount"])
            if "payment_ratio" not in clean.columns:
                clean["payment_ratio"] = (clean["paid_amount"] / clean["billed_amount"].replace(0, np.nan)).fillna(0.0).clip(0, 5.0)
                derived_cols.append("payment_ratio")
        else:
            clean["paid_amount"] = clean["billed_amount"] * 0.85
            imputed_cols.append("paid_amount")

        if "allowed_amount" in clean.columns and clean["allowed_amount"].notna().any():
            clean["allowed_amount"] = _clean_numeric_series(clean["allowed_amount"])
            if "allowance_ratio" not in clean.columns:
                clean["allowance_ratio"] = (clean["allowed_amount"] / clean["billed_amount"].replace(0, np.nan)).fillna(0.0).clip(0, 5.0)
                derived_cols.append("allowance_ratio")
        else:
            clean["allowed_amount"] = clean["billed_amount"] * 0.90
            imputed_cols.append("allowed_amount")

        # 6. Entity & Categorical Defaults (Safe defaults, avoiding fabricated facts)
        if "claim_id" not in clean.columns or clean["claim_id"].isna().all():
            clean["claim_id"] = [f"CLM-SYNTH-{i:06d}" for i in range(len(clean))]
            imputed_cols.append("claim_id")
        else:
            clean["claim_id"] = clean["claim_id"].fillna("").astype(str)

        for col in ["patient_id", "provider_id", "service_code", "diagnosis_code"]:
            if col not in clean.columns:
                clean[col] = ""
                imputed_cols.append(col)
            else:
                clean[col] = clean[col].fillna("").astype(str)

        if "claim_status" not in clean.columns or clean["claim_status"].isna().all():
            clean["claim_status"] = "APPROVED"
            imputed_cols.append("claim_status")
        else:
            clean["claim_status"] = clean["claim_status"].fillna("APPROVED").astype(str)

        if "quantity" not in clean.columns or clean["quantity"].isna().all():
            clean["quantity"] = 1
            imputed_cols.append("quantity")
        else:
            clean["quantity"] = _clean_numeric_series(clean["quantity"], default=1.0).astype(int)

        if "patient_age" not in clean.columns or clean["patient_age"].isna().all():
            clean["patient_age"] = 45
            imputed_cols.append("patient_age")
        else:
            clean["patient_age"] = _clean_numeric_series(clean["patient_age"], default=45.0).astype(int)

        # Store provenance metadata internally in DataFrame attrs
        clean.attrs["_resolved_aliases"] = resolved_aliases
        clean.attrs["_derived_columns"] = list(set(derived_cols))
        clean.attrs["_imputed_columns"] = list(set(imputed_cols))

        report = {
            "resolved_aliases": resolved_aliases,
            "derived_columns": list(set(derived_cols)),
            "imputed_columns": list(set(imputed_cols)),
        }
        return clean, report

    @staticmethod
    def evaluate_rule_readiness(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates the readiness of the 9 Business Rule modules on the current DataFrame.
        Each rule is checked for mandatory prerequisites. If prerequisites are not met,
        the rule is marked as SKIPPED so the Circuit Breaker can re-normalize weights.
        """
        if df is None or df.empty:
            return {}

        imputed_cols = set(df.attrs.get("_imputed_columns", []))
        derived_cols = set(df.attrs.get("_derived_columns", []))

        def is_really_present(col: str) -> bool:
            if col not in df.columns:
                return False
            # If wholly imputed as empty string or NaT, it's not clinically or operationally present
            if col in imputed_cols and (col in ["patient_id", "provider_id", "service_code", "diagnosis_code", "billing_date", "service_date"]):
                return False
            series = df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                return bool(series.notna().any())
            if pd.api.types.is_numeric_dtype(series):
                return bool((series > 0).any() or series.notna().any())
            str_series = series.astype(str).str.strip()
            return bool(((str_series != "") & (str_series != "nan") & (str_series != "None")).any())

        rule_specs = {
            "repeat_billing": {
                "name": "Repeat Billing",
                "required": ["patient_id", "provider_id", "service_code", "billing_date", "amount"],
                "default_weight": 0.40,
            },
            "phantom_service": {
                "name": "Phantom Service",
                "required": ["service_code"],
                "default_weight": 0.20,
            },
            "provider_capacity": {
                "name": "Provider Capacity",
                "required": ["provider_id", "service_date", "service_code"],
                "default_weight": 0.15,
            },
            "fuzzy_similarity": {
                "name": "Fuzzy Claim Matching",
                "required": ["patient_id", "billing_date"],
                "default_weight": 0.15,
            },
            "upcoding_unbundling": {
                "name": "Upcoding & Unbundling",
                "required": ["claim_id", "patient_id", "provider_id", "amount", "diagnosis_code"],
                "default_weight": 0.025,
            },
            "inflated_bill_cloning": {
                "name": "Inflated Bill & Cloning",
                "required": ["claim_id", "patient_id", "provider_id", "amount"],
                "default_weight": 0.025,
            },
            "prolonged_stay_readmission": {
                "name": "Length of Stay & Readmission",
                "required": ["claim_id", "patient_id", "provider_id", "length_of_stay"],
                "default_weight": 0.025,
            },
            "medication_device_fraud": {
                "name": "Medication & Device Fraud",
                "required": ["claim_id", "patient_id", "provider_id", "amount", "quantity"],
                "default_weight": 0.025,
            },
            "duplicate_payment": {
                "name": "Duplicate Payment",
                "required": ["claim_id", "claim_status"],
                "default_weight": 0.10,
            },
        }

        readiness: Dict[str, Dict[str, Any]] = {}
        for rule_id, spec in rule_specs.items():
            missing = [c for c in spec["required"] if not is_really_present(c)]
            derived = [c for c in spec["required"] if c in derived_cols]

            if missing:
                status = "SKIPPED"
                ready = False
                reason = f"Kolom wajib tidak tersedia: {', '.join(missing)}"
            elif derived:
                status = "DERIVED"
                ready = True
                reason = f"Siap berjalan dengan kolom hasil derivasi: {', '.join(derived)}"
            else:
                status = "READY"
                ready = True
                reason = "Seluruh kolom wajib tersedia penuh"

            readiness[rule_id] = {
                "name": spec["name"],
                "ready": ready,
                "status": status,
                "required_columns": spec["required"],
                "missing_columns": missing,
                "derived_columns": derived,
                "default_weight": spec["default_weight"],
                "reason": reason,
            }

        return readiness
