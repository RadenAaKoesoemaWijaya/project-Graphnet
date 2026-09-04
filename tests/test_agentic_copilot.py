import unittest
import pandas as pd
from rag_engine import LocalRAGKnowledgeBase, get_rag_knowledge_base
from agentic_copilot import ClaimContextBuilder, AgenticInvestigatorCopilot


class TestAgenticCopilotAndRAG(unittest.TestCase):
    def setUp(self):
        self.rag = get_rag_knowledge_base()
        self.copilot = AgenticInvestigatorCopilot(provider="heuristic")

    def test_rag_knowledge_retrieval(self):
        # Query repeat billing regulations
        docs = self.rag.retrieve("repeat billing tagihan ganda 30 hari", top_k=2)
        self.assertTrue(len(docs) > 0)
        # Check that relevant tags or title matched
        titles = [d['title'] for d in docs]
        self.assertTrue(any("Repeat Billing" in t or "Permenkes" in t for t in titles))

    def test_rag_regulation_context_formatting(self):
        context_str = self.rag.get_regulation_context(
            active_flags=["repeat_billing", "phantom_service"],
            query_extra="99213"
        )
        self.assertIsInstance(context_str, str)
        self.assertIn("Permenkes", context_str)

    def test_claim_context_builder_pii_sanitization(self):
        sample_row = pd.Series({
            "claim_id": "CLM-999888",
            "patient_id": "PAT-SECRET-123",
            "provider_id": "PROV-HOSP-01",
            "service_code": "99214",
            "diagnosis_code": "I10",
            "billed_amount": 7500000.0,
            "paid_amount": 6000000.0,
            "anomaly_probability": 0.85,
            "final_risk_score": 0.92,
            "severity": "High",
            "repeat_billing_flag": 1,
            "phantom_service_flag": 0,
            "medication_device_fraud_flag": 1,
            "prolonged_stay_readmission_flag": 1,
        })

        shap_dict = {"billed_amount": 0.42, "payment_ratio": -0.15}
        ctx = ClaimContextBuilder.build_sanitized_context(
            claim_row=sample_row,
            shap_contributions=shap_dict,
            mask_sensitive=True
        )

        self.assertEqual(ctx["claim_id"], "***888")
        self.assertEqual(ctx["severity"], "High")
        self.assertIn("Repeat Billing (Tagihan Berulang)", ctx["active_rules"])
        self.assertIn("Medication & Device Fraud (Kuantitas Obat/Alkes Berlebih)", ctx["active_rules"])
        self.assertIn("Prolonged Stay & Readmission (Lama Rawat Anomali)", ctx["active_rules"])
        self.assertTrue(len(ctx["top_shap_features"]) > 0)
        self.assertIn("billed_amount: +0.420", ctx["top_shap_features"][0])
        self.assertEqual(ctx["billed_amount"], 7500000.0)

    def test_dossier_generation_heuristic(self):
        sample_context = {
            "claim_id": "CLM-TEST-001",
            "patient_id": "PAT-***-001",
            "provider_id": "PROV-001",
            "service_code": "43239",
            "diagnosis_code": "K21.0",
            "billed_amount": 12000000,
            "paid_amount": 10000000,
            "anomaly_score": 0.88,
            "final_risk_score": 0.95,
            "severity": "High",
            "active_rules": ["Phantom Service (Layanan Fiktif)", "Repeat Billing (Tagihan Berulang)"],
            "top_shap_features": ["billed_amount: +0.650"],
            "gnn_collusion_cluster": ["PROV-001 connected to 12 suspicious claims"]
        }

        result = self.copilot.generate_investigation_dossier(
            context=sample_context,
            investigator_name="Auditor QA"
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["claim_id"], "CLM-TEST-001")
        dossier_text = result["dossier_text"]
        self.assertIn("BERITA ACARA PEMERIKSAAN KLAIM ANOMALI", dossier_text)
        self.assertIn("Auditor QA", dossier_text)
        self.assertIn("CLM-TEST-001", dossier_text)
        self.assertIn("Phantom Service", dossier_text)

    def test_investigator_query_answering(self):
        sample_context = {
            "claim_id": "CLM-TEST-002",
            "patient_id": "PAT-***-002",
            "provider_id": "PROV-002",
            "service_code": "99213",
            "diagnosis_code": "J06.9",
            "billed_amount": 3500000,
            "final_risk_score": 0.78,
            "severity": "Medium",
            "active_rules": ["Upcoding & Unbundling (Penggelembungan Kode)"]
        }

    def test_rag_statistical_outlier_retrieval(self):
        # Query when no rules are active (pure statistical ML outlier)
        context_str = self.rag.get_regulation_context(
            active_flags=[],
            query_extra="99214 E11.9"
        )
        self.assertIsInstance(context_str, str)
        self.assertTrue("Deviasi Biaya" in context_str or "Kesesuaian Klinis" in context_str or "Permenkes" in context_str)

    def test_dossier_generation_api_failure_fallback_preserves_context(self):
        # Test that cloud LLM failure falls back to heuristic WITH full context preserved (no CLM-UNKNOWN)
        copilot_cloud = AgenticInvestigatorCopilot(
            provider="gemini",
            api_key="INVALID_TEST_KEY"
        )
        sample_context = {
            "claim_id": "CLM-REAL-999",
            "patient_id": "PAT-MASKED-777",
            "provider_id": "PROV-HOSP-ABC",
            "service_code": "43239",
            "diagnosis_code": "K21.0",
            "billed_amount": 15000000,
            "paid_amount": 12000000,
            "anomaly_score": 0.91,
            "final_risk_score": 0.94,
            "severity": "High",
            "active_rules": ["Repeat Billing (Tagihan Berulang)"],
            "top_shap_features": ["billed_amount: +1.240"],
            "gnn_collusion_cluster": ["Faskes PROV-HOSP-ABC: 5 klaim terhubung"]
        }

        result = copilot_cloud.generate_investigation_dossier(
            context=sample_context,
            investigator_name="Auditor Fallback Test"
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["claim_id"], "CLM-REAL-999")
        self.assertIn("Fallback: Heuristic", result["provider_used"])
        dossier_text = result["dossier_text"]
        self.assertIn("CLM-REAL-999", dossier_text)
        self.assertNotIn("CLM-UNKNOWN", dossier_text)
        self.assertIn("Rp 15,000,000", dossier_text)
        self.assertIn("Auditor Fallback Test", dossier_text)

    def test_query_answering_api_failure_fallback_returns_answer_not_dossier(self):
        # When cloud API fails, query answer must return concise Q&A answer, NOT a full BAP document
        copilot_cloud = AgenticInvestigatorCopilot(
            provider="openai",
            api_key="INVALID_TEST_KEY"
        )
        sample_context = {
            "claim_id": "CLM-QUERY-555",
            "patient_id": "PAT-MASKED-555",
            "provider_id": "PROV-XYZ",
            "service_code": "99213",
            "diagnosis_code": "J06.9",
            "billed_amount": 3500000,
            "final_risk_score": 0.82,
            "severity": "Medium",
            "active_rules": ["Upcoding & Unbundling (Penggelembungan Kode)"]
        }

        answer = copilot_cloud.answer_investigator_query(
            context=sample_context,
            user_question="Apakah ada indikasi tagihan berlebih?"
        )

        self.assertIsInstance(answer, str)
        self.assertIn("CLM-QUERY-555", answer)
        self.assertIn("Kesimpulan Utama", answer)
        # Verify it did not accidentally return full BAP header
        self.assertNotIn("# 📑 BERITA ACARA PEMERIKSAAN KLAIM ANOMALI", answer)

    def test_claim_context_with_gnn_and_feature_deviations(self):
        sample_row = pd.Series({
            "claim_id": "CLM-GNN-111",
            "patient_id": "PAT-111",
            "provider_id": "PROV-111",
            "service_code": "99213",
            "diagnosis_code": "I10",
            "billed_amount": 5000000,
            "final_risk_score": 0.88,
            "severity": "High"
        })
        deviations = {"billed_amount": +1.52, "length_of_stay": +0.85}
        clusters = ["Faskes PROV-111 memiliki 8 klaim anomali terhubung"]

        ctx = ClaimContextBuilder.build_sanitized_context(
            claim_row=sample_row,
            shap_contributions=deviations,
            gnn_neighbors=clusters,
            mask_sensitive=True
        )

        self.assertEqual(len(ctx["top_shap_features"]), 2)
        self.assertIn("billed_amount: +1.520", ctx["top_shap_features"][0])
        self.assertEqual(ctx["gnn_collusion_cluster"], clusters)


if __name__ == '__main__':
    unittest.main()
