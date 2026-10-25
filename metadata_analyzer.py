"""
Metadata Analyzer untuk analisis kecukupan fitur terhadap 9 rules bisnis ASTINA.
Module ini menyediakan analisis gap antara dataset yang diupload dengan fitur yang diperlukan
untuk menjalankan 9 rules bisnis deteksi fraud.
"""

from typing import Dict, List, Optional, Set
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger("astina.metadata_analyzer")


class RulePriority(Enum):
    """Priority level untuk fitur yang diperlukan."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ColumnRequirement:
    """Requirement untuk sebuah kolom kanonikal."""
    canonical_name: str
    data_type: str
    required_by_rules: List[int] = field(default_factory=list)
    priority: RulePriority = RulePriority.MEDIUM
    nullable: bool = False
    unique: bool = False
    description: str = ""


@dataclass
class MetadataAnalysisResult:
    """Hasil analisis metadata dataset."""
    total_features_required: int
    features_available: int
    features_missing: List[str]
    rules_enabled: List[int]
    rules_disabled: List[int]
    rule_coverage: Dict[int, float]  # rule_id -> coverage_percentage
    recommendations: List[str]
    column_mapping: Dict[str, str]  # user_column -> canonical_column
    mapping_confidence: Dict[str, float]  # user_column -> confidence_score


class MetadataAnalyzer:
    """Analyzer untuk metadata dataset terhadap 9 rules bisnis ASTINA."""
    
    # 14 Kolom kanonikal ASTINA dengan requirements
    CANONICAL_COLUMNS: Dict[str, ColumnRequirement] = {
        "claim_id": ColumnRequirement(
            canonical_name="claim_id",
            data_type="string",
            required_by_rules=[1, 4, 6, 9],
            priority=RulePriority.CRITICAL,
            nullable=False,
            unique=True,
            description="ID unik klaim"
        ),
        "patient_id": ColumnRequirement(
            canonical_name="patient_id",
            data_type="string",
            required_by_rules=[1, 6, 7, 9],
            priority=RulePriority.CRITICAL,
            nullable=False,
            unique=False,
            description="ID unik pasien"
        ),
        "provider_id": ColumnRequirement(
            canonical_name="provider_id",
            data_type="string",
            required_by_rules=[1, 2, 3, 6, 7, 9],
            priority=RulePriority.CRITICAL,
            nullable=False,
            unique=False,
            description="Kode faskes/dokter"
        ),
        "service_code": ColumnRequirement(
            canonical_name="service_code",
            data_type="string",
            required_by_rules=[2, 3, 5, 8, 9],
            priority=RulePriority.CRITICAL,
            nullable=False,
            unique=False,
            description="Kode prosedur medis"
        ),
        "diagnosis_code": ColumnRequirement(
            canonical_name="diagnosis_code",
            data_type="string",
            required_by_rules=[2, 5, 7, 8],
            priority=RulePriority.HIGH,
            nullable=True,
            unique=False,
            description="Kode diagnosis ICD-10"
        ),
        "billing_date": ColumnRequirement(
            canonical_name="billing_date",
            data_type="date",
            required_by_rules=[1, 3, 7],
            priority=RulePriority.CRITICAL,
            nullable=False,
            unique=False,
            description="Tanggal penagihan"
        ),
        "service_date": ColumnRequirement(
            canonical_name="service_date",
            data_type="date",
            required_by_rules=[2, 3, 7],
            priority=RulePriority.CRITICAL,
            nullable=False,
            unique=False,
            description="Tanggal tindakan medis"
        ),
        "billed_amount": ColumnRequirement(
            canonical_name="billed_amount",
            data_type="float",
            required_by_rules=[4, 5, 6, 9],
            priority=RulePriority.CRITICAL,
            nullable=False,
            unique=False,
            description="Nominal tagihan"
        ),
        "paid_amount": ColumnRequirement(
            canonical_name="paid_amount",
            data_type="float",
            required_by_rules=[4, 6],
            priority=RulePriority.CRITICAL,
            nullable=True,
            unique=False,
            description="Nominal yang dibayar"
        ),
        "allowed_amount": ColumnRequirement(
            canonical_name="allowed_amount",
            data_type="float",
            required_by_rules=[4, 5],
            priority=RulePriority.HIGH,
            nullable=True,
            unique=False,
            description="Nominal yang disetujui"
        ),
        "claim_status": ColumnRequirement(
            canonical_name="claim_status",
            data_type="string",
            required_by_rules=[4],
            priority=RulePriority.CRITICAL,
            nullable=False,
            unique=False,
            description="Status klaim"
        ),
        "patient_age": ColumnRequirement(
            canonical_name="patient_age",
            data_type="integer",
            required_by_rules=[7],
            priority=RulePriority.HIGH,
            nullable=True,
            unique=False,
            description="Usia pasien"
        ),
        "length_of_stay": ColumnRequirement(
            canonical_name="length_of_stay",
            data_type="integer",
            required_by_rules=[7],
            priority=RulePriority.HIGH,
            nullable=True,
            unique=False,
            description="Lama rawat inap (hari)"
        ),
        "quantity": ColumnRequirement(
            canonical_name="quantity",
            data_type="integer",
            required_by_rules=[3, 8],
            priority=RulePriority.HIGH,
            nullable=True,
            unique=False,
            description="Kuantitas obat/alkes"
        ),
    }
    
    # Semantic harmonizer mapping (simplified version)
    SEMANTIC_MAPPING: Dict[str, str] = {
        # claim_id variations
        "no_klaim": "claim_id",
        "claim_id": "claim_id",
        "claimid": "claim_id",
        "klaim_id": "claim_id",
        "nomor_klaim": "claim_id",
        "claim_number": "claim_id",
        
        # patient_id variations
        "no_peserta": "patient_id",
        "patient_id": "patient_id",
        "patientid": "patient_id",
        "peserta_id": "patient_id",
        "nomor_peserta": "patient_id",
        "patient_number": "patient_id",
        "member_id": "patient_id",
        
        # provider_id variations
        "kode_faskes": "provider_id",
        "provider_id": "provider_id",
        "providerid": "provider_id",
        "faskes_id": "provider_id",
        "kode_provider": "provider_id",
        "provider_code": "provider_id",
        "hospital_id": "provider_id",
        "clinic_id": "provider_id",
        
        # service_code variations
        "kode_tindakan": "service_code",
        "service_code": "service_code",
        "servicecode": "service_code",
        "tindakan_kode": "service_code",
        "procedure_code": "service_code",
        "procedurecode": "service_code",
        "cpt_code": "service_code",
        
        # diagnosis_code variations
        "diagnosis": "diagnosis_code",
        "diagnosis_code": "diagnosis_code",
        "diagnosiscode": "diagnosis_code",
        "kode_diagnosis": "diagnosis_code",
        "icd_code": "diagnosis_code",
        "icd10": "diagnosis_code",
        
        # billing_date variations
        "tanggal_tagihan": "billing_date",
        "billing_date": "billing_date",
        "billingdate": "billing_date",
        "tgl_tagihan": "billing_date",
        "invoice_date": "billing_date",
        
        # service_date variations
        "tanggal_tindakan": "service_date",
        "service_date": "service_date",
        "servicedate": "service_date",
        "tgl_tindakan": "service_date",
        "procedure_date": "service_date",
        
        # billed_amount variations
        "biaya_tagihan": "billed_amount",
        "billed_amount": "billed_amount",
        "billedamount": "billed_amount",
        "nominal_tagihan": "billed_amount",
        "claim_amount": "billed_amount",
        "invoice_amount": "billed_amount",
        
        # paid_amount variations
        "biaya_dibayar": "paid_amount",
        "paid_amount": "paid_amount",
        "paidamount": "paid_amount",
        "nominal_dibayar": "paid_amount",
        "payment_amount": "paid_amount",
        
        # allowed_amount variations
        "biaya_disetujui": "allowed_amount",
        "allowed_amount": "allowed_amount",
        "allowedamount": "allowed_amount",
        "nominal_disetujui": "allowed_amount",
        "approved_amount": "allowed_amount",
        
        # claim_status variations
        "status_klaim": "claim_status",
        "claim_status": "claim_status",
        "claimstatus": "claim_status",
        "status": "claim_status",
        
        # patient_age variations
        "usia": "patient_age",
        "patient_age": "patient_age",
        "patientage": "patient_age",
        "age": "patient_age",
        
        # length_of_stay variations
        "lama_rawat": "length_of_stay",
        "length_of_stay": "length_of_stay",
        "lengthofstay": "length_of_stay",
        "los": "length_of_stay",
        "durasi_rawat": "length_of_stay",
        
        # quantity variations
        "jumlah": "quantity",
        "quantity": "quantity",
        "qty": "quantity",
        "kuantitas": "quantity",
    }
    
    def __init__(self):
        """Initialize MetadataAnalyzer."""
        self.logger = logging.getLogger("astina.metadata_analyzer")
    
    def analyze_dataset(
        self,
        df: pd.DataFrame,
        manual_metadata: Optional[Dict] = None
    ) -> MetadataAnalysisResult:
        """
        Analisis dataset terhadap 9 rules bisnis.
        
        Args:
            df: Dataset pandas DataFrame
            manual_metadata: Metadata manual (opsional) untuk enrichment
        
        Returns:
            MetadataAnalysisResult dengan gap analysis dan rekomendasi
        """
        try:
            user_columns = list(df.columns)
            column_mapping = {}
            mapping_confidence = {}
            
            # Step 1: Auto-detect mapping menggunakan semantic harmonizer
            for user_col in user_columns:
                canonical, confidence = self._detect_canonical_column(user_col)
                if canonical:
                    column_mapping[user_col] = canonical
                    mapping_confidence[user_col] = confidence
            
            # Step 2: Enrich dengan manual metadata jika ada
            if manual_metadata:
                for col_info in manual_metadata.get("columns", []):
                    user_col = col_info.get("user_column_name")
                    canonical = col_info.get("canonical_column")
                    if canonical and user_col in user_columns:
                        column_mapping[user_col] = canonical
                        mapping_confidence[user_col] = 1.0  # Manual mapping = 100% confidence
            
            # Step 3: Analisis gap
            available_canonical = set(column_mapping.values())
            missing_canonical = set(self.CANONICAL_COLUMNS.keys()) - available_canonical
            
            # Step 4: Analisis coverage per rule
            rule_coverage = {}
            for rule_id in range(1, 10):
                required_cols = self._get_columns_for_rule(rule_id)
                available_for_rule = [col for col in required_cols if col in available_canonical]
                coverage = len(available_for_rule) / len(required_cols) if required_cols else 0
                rule_coverage[rule_id] = coverage
            
            # Step 5: Tentukan rules yang bisa diaktifkan
            rules_enabled = [rid for rid, cov in rule_coverage.items() if cov >= 0.5]
            rules_disabled = [rid for rid, cov in rule_coverage.items() if cov < 0.5]
            
            # Step 6: Generate rekomendasi
            recommendations = self._generate_recommendations(
                missing_canonical,
                rule_coverage,
                user_columns,
                available_canonical
            )
            
            self.logger.info(f"Metadata analysis completed: {len(available_canonical)}/{len(self.CANONICAL_COLUMNS)} features available")
            
            return MetadataAnalysisResult(
                total_features_required=len(self.CANONICAL_COLUMNS),
                features_available=len(available_canonical),
                features_missing=list(missing_canonical),
                rules_enabled=rules_enabled,
                rules_disabled=rules_disabled,
                rule_coverage=rule_coverage,
                recommendations=recommendations,
                column_mapping=column_mapping,
                mapping_confidence=mapping_confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error during metadata analysis: {e}")
            raise
    
    def _detect_canonical_column(self, user_column: str) -> tuple[Optional[str], float]:
        """
        Detect canonical column name using semantic harmonizer.
        
        Returns:
            (canonical_column_name, confidence_score)
        """
        user_col_lower = user_column.lower().strip()
        
        # Exact match
        if user_col_lower in self.SEMANTIC_MAPPING:
            return self.SEMANTIC_MAPPING[user_col_lower], 1.0
        
        # Fuzzy match (contains)
        for key, value in self.SEMANTIC_MAPPING.items():
            if key in user_col_lower or user_col_lower in key:
                return value, 0.8
        
        # No match
        return None, 0.0
    
    def _get_columns_for_rule(self, rule_id: int) -> List[str]:
        """Get canonical columns required for a specific rule."""
        rule_columns = {
            1: ["claim_id", "patient_id", "billing_date"],
            2: ["service_code", "provider_id"],
            3: ["provider_id", "service_date"],
            4: ["claim_id", "claim_status", "paid_amount"],
            5: ["service_code", "billed_amount"],
            6: ["billed_amount", "claim_id"],
            7: ["patient_id", "service_date", "length_of_stay"],
            8: ["service_code", "quantity"],
            9: ["claim_id", "patient_id", "provider_id"],
        }
        return rule_columns.get(rule_id, [])
    
    def _generate_recommendations(
        self,
        missing_columns: Set[str],
        rule_coverage: Dict[int, float],
        user_columns: List[str],
        available_canonical: Set[str]
    ) -> List[str]:
        """Generate rekomendasi preprocessing."""
        recommendations = []
        
        # Critical features missing
        critical_missing = [
            col for col in missing_columns
            if self.CANONICAL_COLUMNS[col].priority == RulePriority.CRITICAL
        ]
        if critical_missing:
            affected_rules = self._get_affected_rules(critical_missing)
            recommendations.append(
                f"CRITICAL: Fitur wajib yang kurang: {', '.join(critical_missing)}. "
                f"Rules yang terdampak: {affected_rules}. "
                f"Pertimbangkan untuk menambahkan kolom ini ke dataset Anda."
            )
        
        # High priority features missing
        high_missing = [
            col for col in missing_columns
            if self.CANONICAL_COLUMNS[col].priority == RulePriority.HIGH
        ]
        if high_missing:
            affected_rules = self._get_affected_rules(high_missing)
            recommendations.append(
                f"IMPORTANT: Fitur yang disarankan: {', '.join(high_missing)}. "
                f"Rules yang terdampak: {affected_rules}. "
                f"Menambahkan fitur ini akan meningkatkan akurasi deteksi."
            )
        
        # Rules with low coverage
        low_coverage_rules = [
            f"Rule {rid} ({self._get_rule_name(rid)})"
            for rid, cov in rule_coverage.items()
            if cov < 0.5
        ]
        if low_coverage_rules:
            recommendations.append(
                f"WARNING: Rules yang tidak dapat diaktifkan penuh: {', '.join(low_coverage_rules)}. "
                f"Pertimbangkan untuk menambahkan fitur yang diperlukan atau disable rules tersebut."
            )
        
        # Data type recommendations
        recommendations.append(
            f"INFO: Pastikan tipe data sesuai: tanggal untuk kolom tanggal, "
            f"numerik untuk amount, string untuk ID. "
            f"Semantic harmonizer akan otomatis mengubah nama kolom yang sesuai."
        )
        
        # Coverage summary
        coverage_pct = len(available_canonical) / len(self.CANONICAL_COLUMNS) * 100
        if coverage_pct >= 80:
            recommendations.append(
                f"SUCCESS: Dataset Anda memiliki {coverage_pct:.0f}% fitur yang diperlukan. "
                f"Siap untuk preprocessing dan training."
            )
        elif coverage_pct >= 50:
            recommendations.append(
                f"WARNING: Dataset Anda memiliki {coverage_pct:.0f}% fitur yang diperlukan. "
                f"Beberapa rules mungkin tidak berjalan optimal."
            )
        else:
            recommendations.append(
                f"ERROR: Dataset Anda hanya memiliki {coverage_pct:.0f}% fitur yang diperlukan. "
                f"Sangat disarankan untuk menambahkan kolom yang kurang."
            )
        
        return recommendations
    
    def _get_affected_rules(self, columns: List[str]) -> str:
        """Get list of rules affected by missing columns."""
        affected_rules = set()
        for col in columns:
            req = self.CANONICAL_COLUMNS[col]
            affected_rules.update(req.required_by_rules)
        return ", ".join([f"Rule {r}" for r in sorted(affected_rules)])
    
    def _get_rule_name(self, rule_id: int) -> str:
        """Get human-readable rule name."""
        rule_names = {
            1: "Repeat Billing",
            2: "Phantom Service",
            3: "Provider Capacity",
            4: "Claim Status & Duplicate Payment",
            5: "Upcoding & Unbundling",
            6: "Inflated Bill & Cloning",
            7: "Length of Stay & Readmission",
            8: "Medication & Device Fraud",
            9: "Fuzzy Claim Matching",
        }
        return rule_names.get(rule_id, f"Rule {rule_id}")
    
    def get_rule_requirements(self, rule_id: int) -> Dict[str, ColumnRequirement]:
        """Get all column requirements for a specific rule."""
        required_cols = self._get_columns_for_rule(rule_id)
        return {
            col: self.CANONICAL_COLUMNS[col]
            for col in required_cols
            if col in self.CANONICAL_COLUMNS
        }
    
    def get_all_requirements(self) -> Dict[str, ColumnRequirement]:
        """Get all canonical column requirements."""
        return self.CANONICAL_COLUMNS.copy()
