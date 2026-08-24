#!/usr/bin/env python3
"""
Test feature importance computation with sample dataset
Verifies the isolation_forest feature importance fix works correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

def test_feature_importance():
    """Test feature importance with sample data"""
    
    # Load sample dataset
    sample_file = Path("sample_healthcare_claims_500.csv")
    if not sample_file.exists():
        print(f"❌ Sample dataset not found: {sample_file}")
        print("   Run: python generate_sample_dataset.py")
        return False
    
    print(f"📊 Loading sample dataset: {sample_file}")
    df = pd.read_csv(sample_file)
    print(f"   ✓ Loaded {len(df)} rows × {len(df.columns)} columns")
    
    # Prepare features (exclude non-numeric columns and target columns)
    exclude_cols = ['claim_id', 'patient_id', 'provider_id', 'report_date', 
                   'claim_date', 'diagnosis_main', 'diagnosis_secondary', 'procedure_codes',
                   'Y_FLAG_REPEAT', 'Y_FLAG_PHANTOM', 'Y_FLAG_BOTH']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
    X = df[feature_cols].fillna(0).values
    
    print(f"✓ Selected {len(feature_cols)} numeric features")
    print(f"✓ Feature shape: {X.shape}")
    
    # Import detector and explainer
    try:
        from model import CombinedAnomalyDetector
        from model_explainer import ModelExplainer
        
        print("\n🔧 Initializing CombinedAnomalyDetector...")
        detector = CombinedAnomalyDetector(
            random_state=42,
            verbose=False
        )
        
        print("🎓 Training on sample data...")
        detector.fit(X, fit_isolation_forest=True, fit_xgboost=False)  # Quick test with IF only
        print("✓ Detector trained successfully")
        
        print("\n🧠 Initializing ModelExplainer...")
        explainer = ModelExplainer(
            detector=detector,
            feature_names=feature_cols
        )
        
        print("🔍 Initializing explainers...")
        if not explainer.initialize_explainers(X[:100]):  # Use first 100 samples
            print("❌ Failed to initialize explainers")
            return False
        print("✓ Explainers initialized")
        
        print("\n📈 Computing feature importance for Isolation Forest...")
        importance_df = explainer.get_feature_importance('isolation_forest', X=X)
        
        if importance_df is None:
            print("❌ Feature importance returned None")
            return False
        
        print(f"✓ Feature importance computed: {len(importance_df)} features")
        print("\nTop 10 Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # Verify dataframe structure
        assert 'feature' in importance_df.columns, "Missing 'feature' column"
        assert 'importance' in importance_df.columns, "Missing 'importance' column"
        assert len(importance_df) > 0, "Feature importance is empty"
        assert importance_df['importance'].sum() > 0, "Importance scores sum to zero"
        
        print("\n✅ All feature importance tests PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Ensure model.py and model_explainer.py are in the workspace")
        raise
    except Exception as e:
        print(f"❌ Error during feature importance computation: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    success = test_feature_importance()
    sys.exit(0 if success else 1)
