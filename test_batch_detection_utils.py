import unittest
import pandas as pd
import numpy as np
from ui.utils import (
    generate_sample_claims_template,
    TEMPLATE_CORE_COLUMNS,
    build_aligned_inference_features,
    _derive_inference_feature,
)

class TestBatchDetectionUtilities(unittest.TestCase):
    def test_generate_sample_claims_template(self):
        df = generate_sample_claims_template(n_rows=5)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        for col in TEMPLATE_CORE_COLUMNS:
            self.assertIn(col, df.columns)

    def test_build_aligned_inference_features_with_stats(self):
        # Create dummy batch df with only some features
        df = pd.DataFrame({
            "billed_amount": [1000000.0, 2000000.0, 3000000.0],
            "paid_amount": [900000.0, 1800000.0, 2700000.0],
            "patient_age": [30, 45, 60]
        })
        
        training_features = [
            "billed_amount", 
            "billed_amount_high", 
            "payment_ratio", 
            "patient_age_group_encoded",
            "missing_feature_x"
        ]
        
        training_stats = {
            "missing_feature_x": 42.0
        }
        
        aligned_df, summary = build_aligned_inference_features(
            df, training_features, training_stats=training_stats
        )
        
        self.assertEqual(list(aligned_df.columns), training_features)
        self.assertEqual(summary['expected_features'], 5)
        self.assertIn("billed_amount", summary['existing_features'])
        self.assertIn("billed_amount_high", summary['derived_features'])
        self.assertIn("payment_ratio", summary['derived_features'])
        self.assertIn("patient_age_group_encoded", summary['derived_features'])
        self.assertIn("missing_feature_x", summary['filled_features'])
        self.assertEqual(aligned_df["missing_feature_x"].iloc[0], 42.0)

    def test_derive_inference_feature_fallbacks(self):
        df = pd.DataFrame({
            "amount": [100, 200, 300],
            "duration": [1, 2, 3]
        })
        # Test ratio derivation
        series, source = _derive_inference_feature(df, "amount_to_duration_ratio")
        self.assertEqual(source, "derived")
        self.assertAlmostEqual(series.iloc[0], 100.0, places=4)

if __name__ == '__main__':
    unittest.main()
