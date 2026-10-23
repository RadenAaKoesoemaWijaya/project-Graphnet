"""
Model Interpretation Service for ASTINA

Generates comprehensive narrative explanations of ML model training results
using generative AI, with PDF export capability for audit and documentation.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

try:
    from agentic_copilot import AgenticInvestigatorCopilot, ClaimContextBuilder
    _has_copilot = True
except ImportError:
    _has_copilot = False
    AgenticInvestigatorCopilot = None
    ClaimContextBuilder = None

logger = logging.getLogger(__name__)


class ModelTrainingContext:
    """Captures complete training context for interpretation."""

    def __init__(self):
        self.dataset_info = {}
        self.preprocessing_info = {}
        self.feature_info = {}
        self.model_config = {}
        self.training_metrics = {}
        self.algorithm_details = {}
        self.hyperparameters = {}
        self.timing_info = {}
        self.validation_results = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM context."""
        return {
            "dataset": self.dataset_info,
            "preprocessing": self.preprocessing_info,
            "features": self.feature_info,
            "model_config": self.model_config,
            "metrics": self.training_metrics,
            "algorithms": self.algorithm_details,
            "hyperparameters": self.hyperparameters,
            "timing": self.timing_info,
            "validation": self.validation_results
        }


class ModelInterpreter:
    """
    Generates narrative explanations of model training results
    using generative AI or heuristic fallback.
    """

    def __init__(self, provider: str = "heuristic"):
        self.provider = provider
        if _has_copilot:
            self.copilot = AgenticInvestigatorCopilot(provider=provider)
        else:
            self.copilot = None
            logger.warning("AgenticInvestigatorCopilot not available, using heuristic-only mode")

    def capture_training_context(
        self,
        df_processed: pd.DataFrame,
        feature_columns: List[str],
        selected_features: List[str],
        training_mode: str,
        algo_options: List[str],
        hyperparameters: Dict[str, Any],
        detector: Any,
        training_metadata: Dict[str, Any]
    ) -> ModelTrainingContext:
        """Capture complete training context from session state and detector."""
        ctx = ModelTrainingContext()

        # Dataset information
        ctx.dataset_info = {
            "total_samples": len(df_processed),
            "features_total": len(feature_columns),
            "features_selected": len(selected_features),
            "feature_reduction_pct": ((len(feature_columns) - len(selected_features)) / len(feature_columns) * 100) if feature_columns else 0,
            "has_partition_column": "__astina_partition" in df_processed.columns,
            "preprocessing_version": training_metadata.get("preprocessing_contract_version", "unknown"),
            "fit_scope": training_metadata.get("preprocessing_fit_scope", "unknown")
        }

        # Preprocessing information
        ctx.preprocessing_info = {
            "method": training_metadata.get("feature_selection_method", "Unknown"),
            "original_count": training_metadata.get("original_feature_count", len(feature_columns)),
            "final_count": training_metadata.get("final_feature_count", len(selected_features)),
            "fit_statistics": training_metadata.get("fit_statistics", {})
        }

        # Feature information
        ctx.feature_info = {
            "total_features": feature_columns,
            "selected_features": selected_features,
            "selection_method": training_metadata.get("feature_selection_method", "All Features")
        }

        # Model configuration
        ctx.model_config = {
            "training_mode": training_mode,
            "algorithms_used": algo_options,
            "ensemble_strategy": "weighted_ensemble" if len(algo_options) > 1 else "single_model"
        }

        # Extract training metrics from detector
        if detector:
            ctx.training_metrics = {
                "isolation_weight": getattr(detector, 'isolation_weight', 0.0),
                "autoencoder_weight": getattr(detector, 'autoencoder_weight', 0.0),
                "dbscan_weight": getattr(detector, 'dbscan_weight', 0.0),
                "gnn_weight": getattr(detector, 'gnn_weight', 0.0),
                "xgboost_weight": getattr(detector, 'xgboost_weight', 0.0),
                "has_isolation_forest": hasattr(detector, 'isolation_model') and detector.isolation_model is not None,
                "has_autoencoder": hasattr(detector, 'autoencoder_model') and detector.autoencoder_model is not None,
                "has_gnn": hasattr(detector, 'gnn_model') and detector.gnn_model is not None,
                "has_xgboost": hasattr(detector, 'xgboost_model') and detector.xgboost_model is not None
            }

            # Algorithm details
            ctx.algorithm_details = {}
            if "Isolation Forest" in algo_options:
                ctx.algorithm_details["isolation_forest"] = {
                    "type": "unsupervised",
                    "mechanism": "anomaly detection via isolation trees",
                    "weight": ctx.training_metrics["isolation_weight"]
                }
            if "Autoencoder" in algo_options:
                ctx.algorithm_details["autoencoder"] = {
                    "type": "unservised",
                    "mechanism": "reconstruction error based anomaly detection",
                    "weight": ctx.training_metrics["autoencoder_weight"]
                }
            if "GNN" in algo_options:
                ctx.algorithm_details["gnn"] = {
                    "type": "graph-based",
                    "mechanism": "anomaly detection via graph neural networks",
                    "weight": ctx.training_metrics["gnn_weight"],
                    "graph_method": training_metadata.get("graph_method", "star")
                }
            if "XGBoost" in algo_options or training_mode == "supervised":
                ctx.algorithm_details["xgboost"] = {
                    "type": "supervised",
                    "mechanism": "gradient boosting for classification",
                    "weight": ctx.training_metrics["xgboost_weight"]
                }

        # Hyperparameters
        ctx.hyperparameters = hyperparameters

        # Validation results (if available)
        if hasattr(detector, 'weight_optimization_results') and detector.weight_optimization_results:
            opt_res = detector.weight_optimization_results
            ctx.validation_results = {
                "optuna_used": True,
                "fpr_reduction": opt_res.get('metric_comparison', {}).get('fpr_reduction_pct', 0.0),
                "optimized_metrics": opt_res.get('metric_comparison', {}).get('optimized', {}),
                "default_metrics": opt_res.get('metric_comparison', {}).get('default', {}),
                "n_trials": opt_res.get('n_trials_completed', 0)
            }

        return ctx

    def generate_training_report(
        self,
        context: ModelTrainingContext,
        language: str = "id"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive narrative report of model training.
        Returns dict with report sections for PDF generation.
        """
        report = {
            "metadata": {
                "generated_at": datetime.now().strftime("%d %B %Y %H:%M:%S WIB"),
                "language": language,
                "interpreter_provider": self.provider
            },
            "executive_summary": self._generate_executive_summary(context),
            "dataset_analysis": self._generate_dataset_analysis(context),
            "feature_analysis": self._generate_feature_analysis(context),
            "model_architecture": self._generate_model_architecture(context),
            "training_process": self._generate_training_process(context),
            "performance_metrics": self._generate_performance_metrics(context),
            "recommendations": self._generate_recommendations(context)
        }

        return report

    def _generate_executive_summary(self, ctx: ModelTrainingContext) -> str:
        """Generate executive summary using LLM or heuristic."""
        if self.copilot and self.provider != "heuristic":
            try:
                prompt = self._build_executive_summary_prompt(ctx)
                response = self.copilot._call_gemini_raw(prompt) if self.provider == "gemini" else None
                if response:
                    return response
            except Exception as e:
                logger.warning(f"LLM executive summary failed: {e}, using heuristic")

        # Heuristic fallback
        dataset_size = ctx.dataset_info["total_samples"]
        n_features = ctx.dataset_info["features_selected"]
        n_algos = len(ctx.model_config["algorithms_used"])
        mode = ctx.model_config["training_mode"]

        summary = f"""
## Ringkasan Eksekutif

Model deteksi anomali ASTINA telah berhasil dilatih menggunakan dataset berisi {dataset_size:,} sampel
dengan {n_features} fitur yang telah dipilih secara selektif.

**Konfigurasi Pelatihan:**
- Mode pelatihan: {mode}
- Algoritma yang digunakan: {', '.join(ctx.model_config['algorithms_used'])}
- Jumlah algoritma: {n_algos}
- Strategi ensemble: {ctx.model_config['ensemble_strategy']}

**Hasil Utama:**
- Reduksi fitur: {ctx.dataset_info['feature_reduction_pct']:.1f}%
- Metode seleksi fitur: {ctx.preprocessing_info['method']}
- Status preprocessing: Versi {ctx.dataset_info['preprocessing_version']} dengan scope {ctx.dataset_info['fit_scope']}

Model ini siap digunakan untuk deteksi anomali transaksi asuransi dengan kombinasi
pendekatan {mode} yang menggabungkan keunggulan masing-masing algoritma dalam ensemble.
"""
        return summary

    def _generate_dataset_analysis(self, ctx: ModelTrainingContext) -> str:
        """Generate dataset analysis section."""
        return f"""
## Analisis Dataset

**Ukuran Dataset:**
- Total sampel: {ctx.dataset_info['total_samples']:,}
- Fitur awal: {ctx.dataset_info['features_total']}
- Fitur terpilih: {ctx.dataset_info['features_selected']}
- Reduksi fitur: {ctx.dataset_info['feature_reduction_pct']:.1f}%

**Preprocessing:**
- Metode seleksi: {ctx.preprocessing_info['method']}
- Versi preprocessing: {ctx.dataset_info['preprocessing_version']}
- Scope fit: {ctx.dataset_info['fit_scope']}
- Kolom partisi tersedia: {'Ya' if ctx.dataset_info['has_partition_column'] else 'Tidak'}

**Fitur yang Digunakan:**
{self._format_feature_list(ctx.feature_info['selected_features'])}
"""

    def _generate_feature_analysis(self, ctx: ModelTrainingContext) -> str:
        """Generate feature analysis section."""
        selected = ctx.feature_info['selected_features']
        total = ctx.feature_info['total_features']

        return f"""
## Analisis Fitur

**Seleksi Fitur:**
- Metode: {ctx.feature_info['selection_method']}
- Dari {total} fitur awal dipilih {len(selected)} fitur
- Efisiensi: {((total - len(selected)) / total * 100):.1f}% fitur dieliminasi

**Fitur yang Digunakan:**
{self._format_feature_list(selected, max_display=20)}

**Rasional Seleksi:**
Metode seleksi fitur yang digunakan ({ctx.feature_info['selection_method']}) bertujuan untuk:
1. Mengurangi dimensi untuk meningkatkan efisiensi komputasi
2. Menghilangkan fitur yang tidak relevan atau redundant
3. Meningkatkan akurasi model dengan fokus pada fitur yang paling informatif
4. Mencegah overfitting dengan mengurangi kompleksitas model
"""

    def _generate_model_architecture(self, ctx: ModelTrainingContext) -> str:
        """Generate model architecture description."""
        sections = []
        sections.append("## Arsitektur Model\n")

        sections.append(f"**Strategi Ensemble:** {ctx.model_config['ensemble_strategy']}\n")
        sections.append(f"**Mode Pelatihan:** {ctx.model_config['training_mode']}\n")
        sections.append(f"**Algoritma yang Digunakan:** {', '.join(ctx.model_config['algorithms_used'])}\n\n")

        sections.append("### Detail Algoritma:\n")
        for algo_name, details in ctx.algorithm_details.items():
            sections.append(f"**{algo_name.replace('_', ' ').title()}**")
            sections.append(f"- Tipe: {details['type']}")
            sections.append(f"- Mekanisme: {details['mechanism']}")
            sections.append(f"- Bobot dalam ensemble: {details['weight']:.3f}\n")

        sections.append("### Distribusi Bobot Ensemble:\n")
        for algo_name, details in ctx.algorithm_details.items():
            sections.append(f"- {algo_name}: {details['weight']:.1%}")

        return "\n".join(sections)

    def _generate_training_process(self, ctx: ModelTrainingContext) -> str:
        """Generate training process description."""
        sections = []
        sections.append("## Proses Pelatihan\n")

        sections.append("### Konfigurasi Hyperparameter:\n")
        for param, value in ctx.hyperparameters.items():
            if isinstance(value, (int, float)):
                sections.append(f"- {param}: {value}")
            elif isinstance(value, list):
                sections.append(f"- {param}: {', '.join(map(str, value[:5]))}...")
            else:
                sections.append(f"- {param}: {value}")

        sections.append("\n### Alur Pelatihan:\n")
        sections.append("1. **Data Preparation**: Dataset dibagi menjadi train/validation/test partitions")
        sections.append("2. **Feature Engineering**: Fitur dipilih dan diproses sesuai metode seleksi")
        sections.append("3. **Model Training**: Setiap algoritma dilatih secara independen")
        sections.append("4. **Ensemble Combination**: Hasil prediksi digabungkan dengan bobot yang ditentukan")
        sections.append("5. **Optimization**: Hyperparameter dan bobot ensemble dioptimasi (jika diaktifkan)")

        if ctx.validation_results.get("optuna_used"):
            sections.append("\n### Optimasi dengan Optuna:\n")
            sections.append(f"- Jumlah trials: {ctx.validation_results['n_trials']}")
            sections.append(f"- Reduksi FPR: {ctx.validation_results['fpr_reduction']:.1f}%")
            sections.append("- Metrik yang dioptimasi: False Positive Rate")

        return "\n".join(sections)

    def _generate_performance_metrics(self, ctx: ModelTrainingContext) -> str:
        """Generate performance metrics section."""
        sections = []
        sections.append("## Metrik Performa\n")

        sections.append("### Status Model:\n")
        metrics = ctx.training_metrics
        sections.append(f"- Isolation Forest: {'✅ Terlatih' if metrics['has_isolation_forest'] else '❌ Tidak terlatih'}")
        sections.append(f"- Autoencoder: {'✅ Terlatih' if metrics['has_autoencoder'] else '❌ Tidak terlatih'}")
        sections.append(f"- GNN: {'✅ Terlatih' if metrics['has_gnn'] else '❌ Tidak terlatih'}")
        sections.append(f"- XGBoost: {'✅ Terlatih' if metrics['has_xgboost'] else '❌ Tidak terlatih'}")

        sections.append("\n### Bobot Ensemble:\n")
        sections.append(f"- Isolation Forest: {metrics['isolation_weight']:.3f}")
        sections.append(f"- Autoencoder: {metrics['autoencoder_weight']:.3f}")
        sections.append(f"- DBSCAN: {metrics['dbscan_weight']:.3f}")
        sections.append(f"- GNN: {metrics['gnn_weight']:.3f}")
        sections.append(f"- XGBoost: {metrics['xgboost_weight']:.3f}")

        if ctx.validation_results.get("optuna_used"):
            sections.append("\n### Hasil Optimasi Optuna:\n")
            opt_metrics = ctx.validation_results['optimized_metrics']
            sections.append(f"- False Positive Rate: {opt_metrics.get('fpr', 0):.2%}")
            sections.append(f"- Precision: {opt_metrics.get('precision', 0):.2%}")
            sections.append(f"- Recall: {opt_metrics.get('recall', 0):.2%}")
            sections.append(f"- F1-Score: {opt_metrics.get('f1', 0):.2%}")

        return "\n".join(sections)

    def _generate_recommendations(self, ctx: ModelTrainingContext) -> str:
        """Generate recommendations section."""
        recommendations = []

        recommendations.append("## Rekomendasi\n")

        # Dataset recommendations
        if ctx.dataset_info['total_samples'] < 1000:
            recommendations.append("### Dataset:")
            recommendations.append("- Pertimbangkan untuk menambah ukuran dataset untuk meningkatkan generalisasi model")
            recommendations.append("- Dataset kecil mungkin menyebabkan overfitting")

        # Feature recommendations
        if ctx.dataset_info['feature_reduction_pct'] > 50:
            recommendations.append("### Fitur:")
            recommendations.append("- Reduksi fitur yang signifikan (>50%) - verifikasi bahwa fitur penting tidak tereliminasi")
            recommendations.append("- Pertimbangkan untuk menggunakan metode seleksi fitur yang lebih konservatif")

        # Model recommendations
        n_algos = len(ctx.model_config['algorithms_used'])
        if n_algos == 1:
            recommendations.append("### Model:")
            recommendations.append("- Hanya menggunakan satu algoritma - pertimbangkan ensemble untuk robustness")
        elif n_algos > 3:
            recommendations.append("### Model:")
            recommendations.append("- Banyak algoritma digunakan - pastikan konfigurasi hyperparameter optimal untuk masing-masing")

        # Performance recommendations
        if ctx.validation_results.get("optuna_used"):
            fpr_red = ctx.validation_results['fpr_reduction']
            if fpr_red < 10:
                recommendations.append("### Optimasi:")
                recommendations.append("- Optimasi bobot ensemble hanya memberikan sedikit peningkatan (<10%)")
                recommendations.append("- Pertimbangkan untuk menggunakan bobot manual atau meningkatkan jumlah trials")
            else:
                recommendations.append("### Optimasi:")
                recommendations.append(f"- Optimasi bobot ensemble berhasil mereduksi FPR sebesar {fpr_red:.1f}%")
                recommendations.append("- Konfigurasi saat ini optimal untuk produksi")

        recommendations.append("\n### Umum:")
        recommendations.append("- Lakukan evaluasi rutin dengan data baru untuk memantau drift")
        recommendations.append("- Pertimbangkan untuk retraining secara berkala (misalnya bulanan)")
        recommendations.append("- Monitor false positive rate di produksi untuk menyesuaikan threshold")

        return "\n".join(recommendations)

    def _build_executive_summary_prompt(self, ctx: ModelTrainingContext) -> str:
        """Build prompt for LLM executive summary generation."""
        context_json = json.dumps(ctx.to_dict(), indent=2, default=str)

        prompt = f"""
Anda adalah ahli machine learning senior yang berspesialisasi dalam interpretasi model deteksi anomali untuk industri asuransi.

Buat ringkasan eksekutif yang komprehensif dan profesional dalam Bahasa Indonesia untuk hasil pelatihan model berikut:

KONTEKS PELATIHAN:
{context_json}

Ringkasan harus mencakup:
1. Ukuran dan karakteristik dataset
2. Strategi seleksi fitur dan dampaknya
3. Arsitektur model ensemble yang digunakan
4. Algoritma yang dilatih dan kontribusinya
5. Hasil optimasi (jika ada)
6. Status kesiapan model untuk produksi

Gunakan bahasa formal, profesional, dan mudah dipahami oleh stakeholder bisnis maupun teknis.
"""

        return prompt

    def _format_feature_list(self, features: List[str], max_display: int = 50) -> str:
        """Format feature list for display."""
        if len(features) <= max_display:
            return ", ".join(features)
        else:
            return ", ".join(features[:max_display]) + f"\n... dan {len(features) - max_display} fitur lainnya"
