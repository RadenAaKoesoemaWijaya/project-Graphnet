"""
Model Explainability Module for ASTINA
Provides SHAP-based explanations for anomaly detection models
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available for statistical tests")

try:
    from alibi_detect import CD
    ALIBI_DETECT_AVAILABLE = True
except ImportError:
    ALIBI_DETECT_AVAILABLE = False
    # alibi-detect is optional for advanced drift detection, not required

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")


SUPPORTED_EXPLAINABILITY_MODELS = {
    'isolation_forest',
    'xgboost'
}


class ModelExplainer:
    """
    Model explainability using SHAP for tree-based models and LIME for local explanations
    """
    
    def __init__(self, detector=None, feature_names=None):
        """
        Initialize model explainer
        
        Args:
            detector: CombinedAnomalyDetector instance
            feature_names: List of feature names
        """
        self.detector = detector
        self.feature_names = feature_names
        self.explainers = {}
        self.shap_values = {}
        self.lime_explainer = None
        
    def initialize_explainers(self, X_background, sample_size=100):
        """
        Initialize SHAP explainers for available models
        
        Args:
            X_background: Background data for SHAP (sample of training data)
            sample_size: Number of samples to use for background
        """
        if not SHAP_AVAILABLE:
            st.error("SHAP not available. Install with: pip install shap")
            return False
            
        if self.detector is None:
            st.error("No detector provided for explanation")
            return False
            
        try:
            # Sample background data for efficiency
            if len(X_background) > sample_size:
                X_sample = X_background[:sample_size]
            else:
                X_sample = X_background
                
            # Initialize Isolation Forest explainer
            if hasattr(self.detector, 'isolation_forest') and self.detector.isolation_forest is not None:
                try:
                    # Use KernelExplainer for Isolation Forest (not TreeExplainer)
                    # Define a prediction function that returns anomaly scores
                    def if_predict(X):
                        return self.detector.isolation_forest.decision_function(X)
                    
                    self.explainers['isolation_forest'] = shap.KernelExplainer(
                        if_predict,
                        data=X_sample,
                        link="identity"
                    )
                    st.success("✅ SHAP explainer initialized for Isolation Forest")
                except Exception as e:
                    st.warning(f"Could not initialize SHAP for Isolation Forest: {str(e)}")
                    # Store for fallback permutation importance
                    self.explainers['isolation_forest'] = 'permutation'
            
            # Initialize XGBoost explainer
            if hasattr(self.detector, 'xgboost_model') and self.detector.xgboost_model is not None:
                try:
                    self.explainers['xgboost'] = shap.TreeExplainer(
                        self.detector.xgboost_model,
                        data=X_sample
                    )
                    st.success("✅ SHAP explainer initialized for XGBoost")
                except Exception as e:
                    st.warning(f"Could not initialize SHAP for XGBoost: {str(e)}")
            
            # Initialize LIME explainer for local explanations
            if LIME_AVAILABLE:
                try:
                    self.lime_explainer = LimeTabularExplainer(
                        X_sample,
                        feature_names=self.feature_names,
                        class_names=['Normal', 'Anomaly'],
                        mode='classification',
                        discretize_continuous=True
                    )
                    st.success("✅ LIME explainer initialized for local explanations")
                except Exception as e:
                    st.warning(f"Could not initialize LIME: {str(e)}")
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing SHAP explainers: {str(e)}")
            return False
    
    def explain_prediction(self, X, model_name='isolation_forest', max_display=10):
        """
        Get SHAP values for a single prediction
        
        Args:
            X: Input features (single instance)
            model_name: Name of model to explain
            max_display: Maximum number of features to display
            
        Returns:
            shap_values: SHAP values for the prediction
            expected_value: Base value for the prediction
        """
        if not SHAP_AVAILABLE:
            return None, None
            
        if model_name not in self.explainers:
            st.error(f"Explainer for {model_name} not initialized")
            return None, None
            
        try:
            explainer = self.explainers[model_name]
            
            # Ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
                
            shap_values = explainer.shap_values(X)
            expected_value = explainer.expected_value
            
            return shap_values, expected_value
            
        except Exception as e:
            st.error(f"Error computing SHAP values: {str(e)}")
            return None, None

    def get_feature_importance(self, model_name='isolation_forest', X=None, max_samples=500):
        """
        Get global feature importance using SHAP or Permutation Importance.

        A sampling budget is enforced so explainability stays tractable on large datasets.
        """
        if model_name not in SUPPORTED_EXPLAINABILITY_MODELS:
            st.warning(
                f"Model '{model_name}' tidak didukung untuk explainability. "
                f"Model yang tersedia saat ini: {', '.join(sorted(SUPPORTED_EXPLAINABILITY_MODELS))}. "
                "Silakan latih ulang model dengan algoritma yang kompatibel untuk analisis fitur."
            )
            return None

        if model_name not in self.explainers:
            st.warning(
                f"Model '{model_name}' belum dilatih atau tidak tersedia untuk explainability. "
                "Pastikan algoritma tersebut aktif saat training sebelum menjalankan analisis fitur."
            )
            return None

        try:
            explainer = self.explainers[model_name]

            # Handle permutation importance fallback
            if explainer == 'permutation':
                if X is None:
                    st.warning(
                        f"Analisis feature importance untuk '{model_name}' memerlukan data evaluasi. "
                        "Tambahkan data X saat memanggil fitur ini atau latih model yang kompatibel terlebih dahulu."
                    )
                    return None
                X_eval = X[:max_samples] if len(X) > max_samples else X
                return self._compute_permutation_importance(model_name, X_eval)

            # Use SHAP for feature importance
            if X is not None:
                X_eval = X[:max_samples] if len(X) > max_samples else X
                shap_values = explainer.shap_values(X_eval)

                # Calculate mean absolute SHAP values as importance
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                importance = np.mean(np.abs(shap_values), axis=0)
            else:
                # For tree-based models without X, try direct feature importance.
                if model_name == 'xgboost' and hasattr(self.detector, 'xgboost_model') and hasattr(self.detector.xgboost_model, 'feature_importances_'):
                    importance = self.detector.xgboost_model.feature_importances_
                else:
                    st.warning(
                        f"Model '{model_name}' tidak memiliki data yang cukup untuk feature importance. "
                        "Masukkan data evaluasi (X) atau gunakan model yang sesuai untuk explainability."
                    )
                    return None

            importance_df = pd.DataFrame({
                'feature': self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            }).sort_values('importance', ascending=False)

            return importance_df

        except Exception as e:
            st.warning(f"Error computing feature importance: {str(e)}")
            st.info("Pastikan model yang dipilih kompatibel dengan explainability dan data evaluasi sudah tersedia.")
            return None
    
    def _compute_permutation_importance(self, model_name, X):
        """
        Compute permutation importance as fallback for models without SHAP support
        
        Args:
            model_name: Name of model
            X: Data to use for importance computation
            
        Returns:
            importance_df: DataFrame with feature importance
        """
        try:
            from sklearn.inspection import permutation_importance
            
            if model_name == 'isolation_forest':
                model = self.detector.isolation_forest
                # Use decision function as score (anomaly score)
                def score_func(y_true, y_pred):
                    return -np.mean((y_pred == -1).astype(int))  # Negative for minimization
                
                perm_importance = permutation_importance(
                    model, X, model.predict(X),
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                st.error(f"Permutation importance not supported for {model_name}")
                return None
            
            importance = perm_importance.importances_mean
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            st.error(f"Error computing permutation importance: {str(e)}")
            return None
    
    def plot_feature_importance(self, model_name='isolation_forest', X=None, max_features=15):
        """
        Plot feature importance using SHAP or Permutation Importance
        
        Args:
            model_name: Name of model to visualize
            max_features: Maximum number of features to show
        """
        if not SHAP_AVAILABLE:
            st.error("SHAP not available")
            return
            
        importance_df = self.get_feature_importance(model_name, X=X)
        # Limit to top features
        importance_df = importance_df.head(max_features)
        
        # Create plot
        import plotly.express as px
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Feature Importance ({model_name})',
            labels={'importance': 'Mean |SHAP Value|', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_shap_summary(self, X, model_name='isolation_forest', max_display=10):
        """
        Plot SHAP summary plot
        
        Args:
            X: Data to visualize
            model_name: Name of model to visualize
            max_display: Maximum number of features to display
        """
        if not SHAP_AVAILABLE:
            st.error("SHAP not available")
            return
            
        if model_name not in self.explainers:
            st.error(f"Explainer for {model_name} not initialized")
            return
            
        try:
            # Sample data for efficiency
            if len(X) > 1000:
                X_sample = X[:1000]
            else:
                X_sample = X
                
            explainer = self.explainers[model_name]
            shap_values = explainer.shap_values(X_sample)
            
            # Create matplotlib figure
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            
            st.pyplot(plt.gcf())
            plt.close()
            
        except Exception as e:
            st.error(f"Error creating SHAP summary plot: {str(e)}")
    
    def explain_with_lime(self, X, num_features=10):
        """
        Explain a single prediction using LIME
        
        Args:
            X: Input features (single instance)
            num_features: Number of features to show in explanation
            
        Returns:
            explanation: LIME explanation object
        """
        if not LIME_AVAILABLE:
            st.error("LIME not available. Install with: pip install lime")
            return None
            
        if self.lime_explainer is None:
            st.error("LIME explainer not initialized")
            return None
            
        try:
            # Ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
                
            explanation = self.lime_explainer.explain_instance(
                X[0],
                self.detector.predict_proba if hasattr(self.detector, 'predict_proba') else None,
                num_features=num_features
            )
            
            return explanation
            
        except Exception as e:
            st.error(f"Error computing LIME explanation: {str(e)}")
            return None
    
    def plot_lime_explanation(self, explanation):
        """
        Plot LIME explanation
        
        Args:
            explanation: LIME explanation object
        """
        if explanation is None:
            return
            
        try:
            import matplotlib.pyplot as plt
            
            fig = explanation.as_pyplot_figure()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Error plotting LIME explanation: {str(e)}")


class ConceptDriftDetector:
    """
    Concept drift detection using statistical methods
    """
    
    def __init__(self, reference_data=None, feature_names=None, threshold=0.05):
        """
        Initialize concept drift detector
        
        Args:
            reference_data: Reference (baseline) data
            feature_names: List of feature names
            threshold: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.threshold = threshold
        self.drift_detected = False
        self.drifted_features = []
        
    def set_reference(self, reference_data):
        """
        Set reference data for drift detection
        
        Args:
            reference_data: Baseline data
        """
        self.reference_data = reference_data
        
    def detect_drift(self, new_data, method='ks_test'):
        """
        Detect concept drift using statistical tests
        
        Args:
            new_data: New data to compare against reference
            method: Method to use ('ks_test', 'psi', 'both')
            
        Returns:
            drift_detected: Whether drift was detected
            drift_report: Dictionary with drift details
        """
        if self.reference_data is None:
            st.error("Reference data not set. Call set_reference() first.")
            return False, {}
            
        if not SCIPY_AVAILABLE:
            st.error("scipy not available for statistical tests")
            return False, {}
            
        drift_report = {
            'method': method,
            'threshold': self.threshold,
            'features': {},
            'overall_drift': False
        }
        
        try:
            # Ensure both datasets have same columns
            common_cols = set(self.reference_data.columns) & set(new_data.columns)
            if len(common_cols) == 0:
                st.error("No common features between reference and new data")
                return False, drift_report
                
            common_cols = list(common_cols)
            
            for feature in common_cols:
                ref_col = self.reference_data[feature].dropna()
                new_col = new_data[feature].dropna()
                
                if len(ref_col) == 0 or len(new_col) == 0:
                    continue
                    
                feature_report = {'drift': False, 'p_value': 1.0, 'statistic': 0.0}
                
                if method in ['ks_test', 'both']:
                    # Kolmogorov-Smirnov test
                    ks_stat, p_value = stats.ks_2samp(ref_col, new_col)
                    feature_report['p_value'] = p_value
                    feature_report['statistic'] = ks_stat
                    feature_report['drift'] = p_value < self.threshold
                    
                    drift_report['features'][feature] = feature_report
                    
                    if feature_report['drift']:
                        drift_report['overall_drift'] = True
                        self.drifted_features.append(feature)
            
            self.drift_detected = drift_report['overall_drift']
            
            return self.drift_detected, drift_report
            
        except Exception as e:
            st.error(f"Error detecting concept drift: {str(e)}")
            return False, drift_report
    
    def plot_drift_report(self, drift_report):
        """
        Visualize drift detection results
        
        Args:
            drift_report: Drift detection report
        """
        if not drift_report['features']:
            st.info("No features analyzed for drift")
            return
            
        # Create DataFrame for visualization
        drift_df = pd.DataFrame([
            {
                'feature': feature,
                'p_value': report['p_value'],
                'statistic': report['statistic'],
                'drift': report['drift']
            }
            for feature, report in drift_report['features'].items()
        ])
        
        if len(drift_df) == 0:
            st.info("No drift detected")
            return
            
        # Sort by p-value
        drift_df = drift_df.sort_values('p_value')
        
        # Plot
        import plotly.express as px
        
        fig = px.bar(
            drift_df,
            x='p_value',
            y='feature',
            orientation='h',
            title='Concept Drift Detection Results',
            labels={'p_value': 'P-Value', 'feature': 'Feature'},
            color='drift',
            color_discrete_map={True: 'red', False: 'green'}
        )
        
        fig.add_vline(x=self.threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Threshold ({self.threshold})")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary
        drifted_count = drift_df['drift'].sum()
        st.metric("Features with Drift", f"{drifted_count}/{len(drift_df)}")
        
        if drifted_count > 0:
            st.warning(f"⚠️ Concept drift detected in {drifted_count} features")
            st.write("Drifted features:", drift_df[drift_df['drift']]['feature'].tolist())

    def check_and_trigger_retraining(self, new_data, adaptive_manager=None, min_drift_feature_pct=0.20, retrain_callback=None, **retrain_kwargs):
        """Automatically detect drift and trigger model retraining if drift ratio exceeds threshold.
        
        Args:
            new_data: Incoming claims batch / new dataset to evaluate.
            adaptive_manager: AdaptiveLearningManager instance (optional).
            min_drift_feature_pct: Minimum proportion of drifted features to trigger retraining (e.g. 0.20 = 20%).
            retrain_callback: Optional callable func(new_data, **retrain_kwargs) -> new_detector.
            retrain_kwargs: Additional arguments passed to retraining pipeline.
            
        Returns:
            Dictionary with drift detection metrics, trigger status, and retraining outcome.
        """
        drift_detected, drift_report = self.detect_drift(new_data)
        
        features_dict = drift_report.get('features', {})
        total_feats = len(features_dict)
        drifted_feats = [f for f, rep in features_dict.items() if rep.get('drift', False)]
        drift_ratio = len(drifted_feats) / total_feats if total_feats > 0 else 0.0
        
        should_retrain = drift_detected and (drift_ratio >= min_drift_feature_pct)
        outcome = {
            'drift_detected': bool(drift_detected),
            'drift_ratio': float(drift_ratio),
            'drifted_features': drifted_feats,
            'total_features': total_feats,
            'retraining_triggered': bool(should_retrain),
            'retraining_result': None,
            'reason': f"Drift in {len(drifted_feats)}/{total_feats} ({drift_ratio:.1%}) features exceeding {min_drift_feature_pct:.1%} threshold." if should_retrain else "Drift within acceptable tolerance."
        }
        
        if should_retrain:
            logger.info("Concept Drift Trigger Activated: %s", outcome['reason'])
            if retrain_callback is not None:
                try:
                    outcome['retraining_result'] = retrain_callback(new_data, **retrain_kwargs)
                except Exception as e:
                    outcome['retraining_result'] = {'status': 'error', 'message': str(e)}
            elif adaptive_manager is not None:
                try:
                    outcome['retraining_result'] = adaptive_manager.trigger_automated_retraining(
                        train_data=new_data,
                        reason=outcome['reason'],
                        **retrain_kwargs
                    )
                except Exception as e:
                    outcome['retraining_result'] = {'status': 'error', 'message': str(e)}
                    
        return outcome


class PerformanceMonitor:
    """
    Monitor model performance over time
    """
    
    def __init__(self):
        """Initialize performance monitor"""
        self.performance_history = []
        self.current_metrics = {}
        
    def log_performance(self, metrics, timestamp=None):
        """
        Log performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
            timestamp: Timestamp of the measurement
        """
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.now()
            
        entry = {
            'timestamp': timestamp,
            **metrics
        }
        
        self.performance_history.append(entry)
        self.current_metrics = metrics
        
    def get_performance_trend(self, metric_name, window=10):
        """
        Get performance trend for a specific metric
        
        Args:
            metric_name: Name of the metric
            window: Number of recent entries to consider
            
        Returns:
            trend: Performance trend ('improving', 'degrading', 'stable')
        """
        if len(self.performance_history) < window:
            return 'insufficient_data'
            
        recent_entries = self.performance_history[-window:]
        values = [entry.get(metric_name) for entry in recent_entries if metric_name in entry]
        
        if len(values) < 2:
            return 'insufficient_data'
            
        # Calculate trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        
        change = (avg_second - avg_first) / avg_first if avg_first != 0 else 0
        
        if change > 0.05:
            return 'improving'
        elif change < -0.05:
            return 'degrading'
        else:
            return 'stable'
    
    def plot_performance_history(self, metric_names=None):
        """
        Plot performance history
        
        Args:
            metric_names: List of metrics to plot (if None, plot all)
        """
        if not self.performance_history:
            st.info("No performance data available")
            return
            
        # Create DataFrame
        df = pd.DataFrame(self.performance_history)
        
        if metric_names is None:
            # Plot numeric columns only
            metric_names = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove timestamp column
            metric_names = [m for m in metric_names if m != 'timestamp']
        
        if not metric_names:
            st.info("No numeric metrics to plot")
            return
        
        # Plot each metric
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=len(metric_names),
            cols=1,
            subplot_titles=metric_names,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metric_names):
            if metric in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric
                    ),
                    row=i+1,
                    col=1
                )
        
        fig.update_layout(height=200*len(metric_names), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


class AdaptiveLearningManager:
    """
    Manage adaptive learning with automatic retraining triggers and feedback collection
    """
    
    def __init__(self, detector=None, retraining_threshold=0.1, performance_window=5):
        """
        Initialize adaptive learning manager
        
        Args:
            detector: CombinedAnomalyDetector instance
            retraining_threshold: Performance degradation threshold to trigger retraining
            performance_window: Number of evaluations to consider for trend
        """
        self.detector = detector
        self.retraining_threshold = retraining_threshold
        self.performance_window = performance_window
        self.feedback_history = []
        self.retraining_history = []
        self.performance_monitor = PerformanceMonitor()
        
    def should_retrain(self, current_performance):
        """
        Determine if model should be retrained based on performance degradation
        
        Args:
            current_performance: Current performance metrics
            
        Returns:
            should_retrain: Boolean indicating if retraining is needed
            reason: Reason for retraining decision
        """
        if len(self.performance_monitor.performance_history) < self.performance_window:
            return False, "Insufficient performance history"
        
        # Get trend for key metrics
        f1_trend = self.performance_monitor.get_performance_trend('f1_score', self.performance_window)
        
        if f1_trend == 'degrading':
            # Calculate degradation amount
            recent_metrics = [entry.get('f1_score', 0) for entry in self.performance_monitor.performance_history[-self.performance_window:]]
            if len(recent_metrics) >= 2:
                degradation = (recent_metrics[0] - recent_metrics[-1]) / recent_metrics[0] if recent_metrics[0] > 0 else 0
                if degradation > self.retraining_threshold:
                    return True, f"F1 score degraded by {degradation:.2%} (threshold: {self.retraining_threshold:.2%})"
        
        return False, "Performance stable"
    
    def collect_feedback(self, prediction_id, features, prediction, actual_label=None, feedback_type=None):
        """
        Collect feedback on predictions
        
        Args:
            prediction_id: Unique identifier for the prediction
            features: Input features
            prediction: Model prediction
            actual_label: Actual label (if available)
            feedback_type: Type of feedback (correct/incorrect/uncertain)
        """
        feedback_entry = {
            'prediction_id': prediction_id,
            'timestamp': pd.Timestamp.now(),
            'features': features,
            'prediction': prediction,
            'actual_label': actual_label,
            'feedback_type': feedback_type
        }
        
        self.feedback_history.append(feedback_entry)
    
    def get_feedback_summary(self):
        """
        Get summary of collected feedback
        
        Returns:
            summary: Dictionary with feedback statistics
        """
        if not self.feedback_history:
            return {'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0.0}
        
        total = len(self.feedback_history)
        correct = sum(1 for f in self.feedback_history if f['feedback_type'] == 'correct')
        incorrect = sum(1 for f in self.feedback_history if f['feedback_type'] == 'incorrect')
        
        return {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def log_retraining(self, reason, performance_before, performance_after):
        """
        Log retraining event
        
        Args:
            reason: Reason for retraining
            performance_before: Performance metrics before retraining
            performance_after: Performance metrics after retraining
        """
        retraining_entry = {
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'performance_before': performance_before,
            'performance_after': performance_after
        }
        
        self.retraining_history.append(retraining_entry)
    
    def get_retraining_history(self):
        """
        Get retraining history
        
        Returns:
            history: List of retraining events
        """
        return self.retraining_history
    
    def plot_retraining_history(self):
        """
        Visualize retraining history
        """
        if not self.retraining_history:
            st.info("No retraining history available")
            return
        
        # Create DataFrame
        history_df = pd.DataFrame(self.retraining_history)
        
        # Show summary
        st.metric("Total Retraining Events", len(self.retraining_history))
        
        # Show recent retrainings
        with st.expander("📜 Riwayat Retraining"):
            for i, entry in enumerate(self.retraining_history[-5:]):
                st.write(f"**{entry['timestamp']}**")
                st.write(f"Reason: {entry['reason']}")
                st.write(f"Performance Before: {entry['performance_before']}")
                st.write(f"Performance After: {entry['performance_after']}")
                st.markdown("---")
    
    def plot_feedback_summary(self):
        """
        Visualize feedback summary
        """
        summary = self.get_feedback_summary()
        
        col_fb1, col_fb2, col_fb3 = st.columns(3)
        with col_fb1:
            st.metric("Total Feedback", summary['total'])
        with col_fb2:
            st.metric("Correct Predictions", summary['correct'])
        with col_fb3:
            st.metric("Accuracy", f"{summary['accuracy']:.2%}")
        
        if summary['total'] > 0:
            import plotly.express as px
            
            feedback_df = pd.DataFrame([
                {'Type': 'Correct', 'Count': summary['correct']},
                {'Type': 'Incorrect', 'Count': summary['incorrect']}
            ])
            
            fig = px.pie(feedback_df, values='Count', names='Type', title='Feedback Distribution')
            st.plotly_chart(fig, use_container_width=True)

    def evaluate_champion_challenger(self, champion_detector, challenger_detector, validation_data,
                                      validation_labels=None, max_fpr_increase=0.01, min_recall_retention=0.95):
        """Senior QA Champion vs Challenger Quality Gate: Evaluates candidate model before production deployment.
        
        Args:
            champion_detector: Existing active CombinedAnomalyDetector instance.
            challenger_detector: Retrained candidate CombinedAnomalyDetector instance.
            validation_data: Dataset for comparison.
            validation_labels: Binary labels (pseudo-labels generated if None).
            max_fpr_increase: Maximum allowed FPR degradation (e.g. 0.01 = +1% FPR allowed).
            min_recall_retention: Minimum recall retention factor (e.g. 0.95 = 95% of champion recall).
            
        Returns:
            Dictionary with gate decision ('promote' or 'reject'), metrics comparison, and reasons.
        """
        try:
            # Generate predictions from both models
            champ_scores = champion_detector.predict_anomaly_probability(validation_data)
            chal_scores = challenger_detector.predict_anomaly_probability(validation_data)
            
            if isinstance(champ_scores, tuple):
                champ_scores = champ_scores[0]
            if isinstance(chal_scores, tuple):
                chal_scores = chal_scores[0]
                
            n_samples = len(validation_data)
            if validation_labels is None:
                # Use high-confidence consensus as validation target
                combined_ref = (champ_scores + chal_scores) / 2.0
                thresh = np.percentile(combined_ref, 95)
                y_eval = (combined_ref >= thresh).astype(int)
            else:
                y_eval = np.asarray(validation_labels, dtype=int)
                
            def _calc_qa_metrics(scores, y_true):
                preds = (scores >= 0.5).astype(int)
                tp = np.sum((preds == 1) & (y_true == 1))
                tn = np.sum((preds == 0) & (y_true == 0))
                fp = np.sum((preds == 1) & (y_true == 0))
                fn = np.sum((preds == 0) & (y_true == 1))
                
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                return {'fpr': float(fpr), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}
                
            champ_metrics = _calc_qa_metrics(champ_scores, y_eval)
            chal_metrics = _calc_qa_metrics(chal_scores, y_eval)
            
            # QA Evaluation Rules
            passed_fpr = chal_metrics['fpr'] <= (champ_metrics['fpr'] + max_fpr_increase)
            passed_recall = chal_metrics['recall'] >= (champ_metrics['recall'] * min_recall_retention)
            passed_f1 = chal_metrics['f1'] >= (champ_metrics['f1'] * 0.90)
            
            promoted = passed_fpr and (passed_recall or chal_metrics['f1'] > champ_metrics['f1'])
            
            decision = "promote" if promoted else "reject"
            reasons = []
            if not passed_fpr:
                reasons.append(f"FPR exceeded threshold: Challenger FPR={chal_metrics['fpr']:.2%}, Champion={champ_metrics['fpr']:.2%}")
            if not passed_recall:
                reasons.append(f"Recall dropped below retention threshold: Challenger Recall={chal_metrics['recall']:.2%}, Champion={champ_metrics['recall']:.2%}")
            if promoted:
                reasons.append("Challenger satisfies all QA stability and FPR constraints.")
                
            return {
                'decision': decision,
                'promoted': bool(promoted),
                'champion_metrics': champ_metrics,
                'challenger_metrics': chal_metrics,
                'reasons': reasons,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            return {
                'decision': 'error',
                'promoted': False,
                'error': str(e)
            }

    def trigger_automated_retraining(self, train_data, train_labels=None, edge_index=None,
                                     device='cpu', optuna_n_trials=15, optuna_timeout=120,
                                     lambda_fpr=0.5, reason="Concept Drift Detected"):
        """Execute automated retraining pipeline with Optuna tuning and Champion-Challenger validation.
        
        Args:
            train_data: Training features dataframe / matrix.
            train_labels: Target labels (optional).
            edge_index: Graph edge index (optional).
            device: Computing device ('cpu' or 'cuda').
            optuna_n_trials: Optuna trials count.
            optuna_timeout: Optuna timeout in seconds.
            lambda_fpr: False positive penalty weight.
            reason: Text trigger reason.
            
        Returns:
            Dictionary detailing retraining execution and deployment outcome.
        """
        import copy
        from model import CombinedAnomalyDetector
        
        logger.info("Triggering automated retraining pipeline. Reason: %s", reason)
        
        # Instantiate Challenger model using current detector architecture
        challenger = CombinedAnomalyDetector(
            algorithms=list(self.detector.algorithms) if self.detector else None,
            use_dynamic_weights=True
        )
        
        try:
            # Fit challenger with Optuna hyperparameter + Ensemble Weight Tuning enabled
            challenger.fit(
                features=train_data.values if hasattr(train_data, 'values') else train_data,
                labels=train_labels,
                edge_index=edge_index,
                device=device,
                optimize_hyperparams=True,
                optuna_n_trials=optuna_n_trials,
                optuna_timeout=optuna_timeout,
                optimize_ensemble_weights=True,
                lambda_fpr=lambda_fpr
            )
            
            # If Champion exists, run Champion-Challenger Quality Gate
            if self.detector is not None:
                gate_result = self.evaluate_champion_challenger(
                    champion_detector=self.detector,
                    challenger_detector=challenger,
                    validation_data=train_data.values if hasattr(train_data, 'values') else train_data,
                    validation_labels=train_labels
                )
                
                if gate_result.get('promoted', False):
                    old_detector = self.detector
                    self.detector = challenger
                    self.log_retraining(
                        reason=reason,
                        performance_before=gate_result.get('champion_metrics', {}),
                        performance_after=gate_result.get('challenger_metrics', {})
                    )
                    logger.info("Challenger successfully promoted to Champion! Status: DEPLOYED")
                    return {
                        'status': 'promoted',
                        'message': 'Challenger model passed QA Gate and has been deployed.',
                        'gate_result': gate_result,
                        'weights': {
                            'isolation': challenger.isolation_weight,
                            'autoencoder': challenger.autoencoder_weight,
                            'xgboost': challenger.xgboost_weight,
                            'gnn': challenger.gnn_weight
                        }
                    }
                else:
                    logger.warning("Challenger failed QA Gate. Retained Champion model. Reasons: %s", gate_result.get('reasons'))
                    return {
                        'status': 'rejected',
                        'message': 'Challenger failed QA quality gate. Retained champion model.',
                        'gate_result': gate_result
                    }
            else:
                self.detector = challenger
                return {'status': 'deployed_initial', 'message': 'Challenger deployed as initial champion model.'}
                
        except Exception as e:
            logger.error("Automated retraining pipeline failed: %s", e)
            return {'status': 'error', 'message': str(e)}

