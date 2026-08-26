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
        self.assertTrue(len(ctx["top_shap_features"]) > 0)
        self.assertIn("billed_amount: +0.420", ctx["top_shap_features"][0])

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

        answer = self.copilot.answer_investigator_query(
            context=sample_context,
            user_question="Apakah tarif ini wajar untuk diagnosis ISPA?"
        )
        self.assertIsInstance(answer, str)
        self.assertIn("CLM-TEST-002", answer)


if __name__ == '__main__':
    unittest.main()
