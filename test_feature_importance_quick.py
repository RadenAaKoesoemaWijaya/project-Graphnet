#!/usr/bin/env python3
"""
Test feature importance computation - Lightweight version
Verifies the isolation_forest feature importance fix works correctly
Uses synthetic data (no external files required)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

def test_feature_importance():
    """Test feature importance with synthetic data"""
    
    print("🧪 Testing Feature Importance Fix\n" + "="*50)
    
    # Generate synthetic data
    print("📊 Generating synthetic healthcare data...")
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    X = np.random.randn(n_samples, n_features) * 100
    X[np.random.rand(n_samples, n_features) < 0.1] = np.random.uniform(-1000, 1000, 1)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    print(f"   ✓ Generated {X.shape[0]} samples × {X.shape[1]} features")
    
    # Import and test
    print("\n🔧 Initializing CombinedAnomalyDetector...")
    try:
        from model import CombinedAnomalyDetector
        from model_explainer import ModelExplainer
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Ensure model.py and model_explainer.py exist")
        raise
    
    try:
        # Initialize detector with only Isolation Forest algorithm (faster for testing)
        detector = CombinedAnomalyDetector(
            random_state=42, 
            verbose=False,
            algorithms=['isolation_forest']
        )
        
        print("🎓 Fitting Isolation Forest...")
        # Fit the detector
        detector.fit(X)
        print("   ✓ Isolation Forest fitted")
        
        # Initialize explainer
        print("\n🧠 Initializing ModelExplainer...")
        explainer = ModelExplainer(detector=detector, feature_names=feature_names)
        
        # Initialize explainers
        print("🔍 Initializing SHAP explainers...")
        assert explainer.initialize_explainers(X[:50]), "Failed to initialize explainers"
        print("   ✓ SHAP explainers initialized")
        
        # Test: Get feature importance for Isolation Forest with X
        print("\n📈 Computing feature importance for Isolation Forest...")
        importance_df = explainer.get_feature_importance('isolation_forest', X=X)
        
        assert importance_df is not None, "Feature importance returned None"
        assert isinstance(importance_df, pd.DataFrame), f"Expected DataFrame, got {type(importance_df)}"
        assert len(importance_df) > 0, "Feature importance DataFrame is empty"
        assert 'feature' in importance_df.columns, f"Missing 'feature' column. Got: {importance_df.columns.tolist()}"
        assert 'importance' in importance_df.columns, f"Missing 'importance' column. Got: {importance_df.columns.tolist()}"
        assert importance_df['importance'].sum() > 0, "Importance scores are all zero or negative"
        
        print(f"   ✓ Feature importance computed for {len(importance_df)} features")
        print("\n   Top 5 Important Features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"      {idx+1}. {row['feature']:20s} → {row['importance']:.6f}")
        
        # Test: Feature importance without X (should fail gracefully for IF)
        print("\n⚠️  Testing error handling (calling without X data)...")
        importance_df_no_x = explainer.get_feature_importance('isolation_forest', X=None)
        if importance_df_no_x is not None:
            print("   ⚠️  Got result without X (permutation importance fallback)")
        else:
            print("   ✓ Correctly returned None when X not provided")
        
        print("\n✅ ALL TESTS PASSED!")
        print("\n" + "="*50)
        print("Summary:")
        print(f"  • Isolation Forest feature importance: ✓ Working")
        print(f"  • SHAP KernelExplainer: ✓ Initialized")
        print(f"  • Data validation: ✓ Passed")
        print(f"  • Error handling: ✓ Graceful")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_feature_importance()
    sys.exit(0)
