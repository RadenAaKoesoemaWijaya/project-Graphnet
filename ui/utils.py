import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from state_manager import navigate_to_page

# Setup logger
logger = logging.getLogger(__name__)

# Patch for Python 3.11.0rc1 missing get_int_max_str_digits which causes PyTorch error
if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
    sys.set_int_max_str_digits = lambda maxdigits: None

# Suppress Polars CPU feature check warnings
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    sys.modules.pop('torch', None)
    sys.modules.pop('torch_geometric', None)
    torch = None
    F = None
    Data = None
    TORCH_AVAILABLE = False
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import ClaimAnomalyXGBoostModel, ClaimAnomalyAutoencoder, CombinedAnomalyDetector, analyze_anomaly_networks, create_claim_graph
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
import re
from file_handler import read_file_with_optimization, get_file_info, show_file_size_warning, optimize_dataframe_memory, save_processed_data, load_processed_data, cleanup_temp_data
from cache_manager import get_cache_path
from config import MAX_FILE_SIZE, LARGE_DATASET_CONFIG
from preprocessing_optimized import preprocess_insurance_claims_optimized, apply_mutual_info_selection, apply_tree_based_selection, apply_pca_reduction, remove_duplicates
import json
from model_registry import save_model_version, get_versions, load_model_version
from ui_components import apply_custom_css, custom_container
from model_explainer import ModelExplainer, ConceptDriftDetector, PerformanceMonitor, AdaptiveLearningManager
from error_handler import handle_error_with_context, safe_execute, validate_dataframe
from data_validator import DataSanitizer, DataValidator, comprehensive_validation, display_validation_results
from audit_trail import get_audit_trail, log_data_upload, log_preprocessing, log_model_training, log_anomaly_detection
from enhanced_metrics import get_metrics_collector, record_operation, increment_counter, set_gauge
import system_status

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

TRAINING_MODE_UNSUPERVISED = "unsupervised"
TRAINING_MODE_SUPERVISED = "supervised"

# Cached Plotly chart generators for performance optimization
@st.cache_data(ttl=300, max_entries=20)
def create_histogram_chart(df, column, nbins=40, title=None):
    """Create cached histogram chart"""
    if title is None:
        title = f"Distribusi: {column}"
    return px.histogram(df, x=column, nbins=nbins, title=title)

@st.cache_data(ttl=300, max_entries=20)
def create_correlation_heatmap(corr_matrix, title="Peta Korelasi"):
    """Create cached correlation heatmap"""
    return px.imshow(corr_matrix, title=title)

@st.cache_data(ttl=300, max_entries=20)
def create_bar_chart(x_values, y_values, title=None, labels=None):
    """Create cached bar chart"""
    if labels is None:
        labels = {'x': 'x', 'y': 'y'}
    if title is None:
        title = "Bar Chart"
    return px.bar(x=x_values, y=y_values, title=title, labels=labels)

@st.cache_data(ttl=300, max_entries=20)
def create_pie_chart(values, names, title="Pie Chart"):
    """Create cached pie chart"""
    return px.pie(values=values, names=names, title=title)

@st.cache_data(ttl=300, max_entries=20)
def create_probability_distribution(probabilities, title="Distribusi Probabilitas", threshold=0.5):
    """Create cached probability distribution chart"""
    fig = px.histogram(
        x=probabilities,
        nbins=50,
        title=title,
        labels={'x': 'Probability', 'y': 'Count'}
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Ambang = {threshold}")
    return fig

def get_df_processed():
    """Helper function to get df_processed from session state path"""
    if 'df_processed_path' not in st.session_state:
        return None
    try:
        result = load_processed_data(st.session_state['df_processed_path'])
        # Handle lazy loading dict response
        if isinstance(result, dict):
            if result.get('lazy'):
                # Load full data if lazy loaded
                return pd.read_parquet(result['path'])
            return result
        return result
    except Exception as e:
        st.error(f"Gagal memuat data hasil praproses: {e}")
        return None

def update_df_processed(new_df):
    """Helper function to update df_processed and save to Parquet"""
    new_path = save_processed_data(new_df, prefix="preprocessed")
    st.session_state['df_processed_path'] = new_path
    return new_df

def normalize_training_mode(raw_mode):
    """Normalize persisted or UI training mode values to internal codes."""
    raw = str(raw_mode or "").strip().lower()
    if raw in {
        TRAINING_MODE_SUPERVISED,
        "supervised",
        "dengan supervisi",
        "dengan supervisi (xgboost/lightgbm/random forest/svm)",
    }:
        return TRAINING_MODE_SUPERVISED
    return TRAINING_MODE_UNSUPERVISED

def get_training_mode_label(raw_mode):
    """Human-friendly label for training mode."""
    mode = normalize_training_mode(raw_mode)
    if mode == TRAINING_MODE_SUPERVISED:
        return "Dengan supervisi"
    return "Tanpa supervisi"

def set_processed_dataset_reference(file_path, feature_columns, preprocessing_metadata):
    """Persist references to the latest processed dataset without duplicating the dataframe in session."""
    st.session_state['df_processed_path'] = file_path
    st.session_state['feature_columns'] = feature_columns
    st.session_state['preprocessing_metadata'] = preprocessing_metadata

def hydrate_processed_data_reference_from_cache(file_hash):
    """Attempts to load the processed dataset reference from cache and populate the session state."""
    parquet_path, final_features, metadata = get_cache_path(file_hash)
    if parquet_path is not None:
        set_processed_dataset_reference(parquet_path, final_features, metadata)
        return True
    return False

def reset_downstream_state():
    """Clear data-dependent state so a newly processed dataset starts from a clean pipeline."""
    keys_to_clear = [
        'train_df',
        'test_df',
        'selected_features',
        'selected_features_cache',
        'feature_selection_method',
        'original_feature_count',
        'final_feature_count',
        'proceed_after_selection',
        'eval_result_df',
        'eval_predictions',
        'eval_probabilities',
        'eval_y_true',
        'individual_probs',
        'detection_results',
        'detection_threshold',
        'training_features',
        'training_mode',
        'training_label_column',
        'detector',
        'model_trained',
        'X_eval_test',  # Add this key that's used in evaluation
        'uploaded_data',  # Clear uploaded data from detection page
        'preprocessing_metadata_new',  # Clear new preprocessing metadata
        'last_drift_detected',  # Clear drift detection status
        'drift_detector',  # Clear drift detector instance
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    
    logger.info("Downstream state reset completed")

def validate_session_state(required_keys, context=""):
    """
    Validate that required session state keys exist and are not None.
    Returns tuple of (is_valid, missing_keys)
    """
    missing_keys = []
    for key in required_keys:
        if key not in st.session_state or st.session_state[key] is None:
            missing_keys.append(key)
    
    if missing_keys:
        logger.warning(f"Missing required session state keys in {context}: {missing_keys}")
    
    return len(missing_keys) == 0, missing_keys

def safe_rerun():
    """Safely rerun the Streamlit app with error handling."""
    try:
        st.rerun()
    except Exception as e:
        logger.error(f"Failed to rerun app: {e}")
        st.error("❌ Gagal me-refresh halaman. Silakan refresh browser secara manual.")

def batch_state_update(state_updates, rerun=True):
    """
    Batch multiple state updates together and optionally rerun once.
    This reduces unnecessary reruns when multiple state changes occur.
    
    Args:
        state_updates: Dictionary of {key: value} state updates
        rerun: Whether to rerun after applying updates (default True)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Apply all state updates
        for key, value in state_updates.items():
            if value is None:
                st.session_state.pop(key, None)
            else:
                st.session_state[key] = value
        
        logger.debug(f"Batch state update applied: {list(state_updates.keys())}")
        
        # Rerun once if requested
        if rerun:
            safe_rerun()
        
        return True
    except Exception as e:
        logger.error(f"Failed to batch state update: {e}")
        return False

def handle_critical_error(error_message, context="", recovery_options=None):
    """
    Handle critical errors with user-friendly messages and recovery options.
    
    Args:
        error_message: The error message to display
        context: Additional context about where the error occurred
        recovery_options: List of tuples (button_text, action_key, action_description)
    """
    logger.error(f"Critical error in {context}: {error_message}")
    
    st.error(f"❌ {error_message}")
    
    if context:
        st.info(f"📍 Lokasi: {context}")
    
    if recovery_options is None:
        recovery_options = [
            ("🔄 Coba Lagi", "retry", "Coba ulang operasi yang gagal"),
            ("🏠 Kembali ke Beranda", "home", "Kembali ke halaman utama"),
        ]
    
    st.markdown("---")
    st.subheader("🔧 Opsi Pemulihan")
    
    cols = st.columns(len(recovery_options))
    for i, (button_text, action_key, action_desc) in enumerate(recovery_options):
        with cols[i]:
            if st.button(button_text, key=f"recovery_{action_key}"):
                if action_key == "home":
                    navigate_to_page('home')
                elif action_key == "retry":
                    st.info("Silakan coba lagi dengan parameter yang berbeda.")
                elif action_key == "resplit":
                    st.session_state.pop('train_df', None)
                    st.session_state.pop('test_df', None)
                    safe_rerun()

def cleanup_large_session_state(threshold_mb=100):
    """
    Periodically clean up large DataFrames from session state to prevent memory bloat.
    
    Args:
        threshold_mb: Memory threshold in MB for considering a DataFrame as "large"
    """
    large_keys = ['train_df', 'test_df', 'df_processed', 'uploaded_data', 'eval_result_df']
    
    for key in large_keys:
        if key in st.session_state:
            data = st.session_state[key]
            if hasattr(data, 'memory_usage'):
                try:
                    memory_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
                    if memory_mb > threshold_mb:
                        logger.info(f"Cleaning up large session state key: {key} ({memory_mb:.2f}MB)")
                        # Save to temporary parquet file if needed
                        temp_key = f"{key}_path"
                        if temp_key not in st.session_state:
                            try:
                                import tempfile
                                import uuid
                                temp_dir = tempfile.gettempdir()
                                temp_path = os.path.join(temp_dir, f"astina_{key}_{uuid.uuid4()}.parquet")
                                data.to_parquet(temp_path, index=False, compression='snappy')
                                st.session_state[temp_key] = temp_path
                                del st.session_state[key]
                                logger.info(f"Saved {key} to temporary file: {temp_path}")
                            except Exception as e:
                                logger.warning(f"Failed to save {key} to temporary file: {e}")
                                # Keep in memory if save fails
                except Exception as e:
                    logger.warning(f"Failed to check memory usage for {key}: {e}")

def get_df_from_session_or_temp(key):
    """
    Load DataFrame from session state or temporary file if it was cleaned up.
    
    Args:
        key: Session state key to retrieve
    
    Returns:
        DataFrame or None if not available
    """
    if key in st.session_state:
        return st.session_state[key]
    
    # Check if there's a temporary file path
    temp_key = f"{key}_path"
    if temp_key in st.session_state:
        try:
            temp_path = st.session_state[temp_key]
            if os.path.exists(temp_path):
                df = pd.read_parquet(temp_path)
                # Reload into session state for faster access
                st.session_state[key] = df
                logger.info(f"Loaded {key} from temporary file: {temp_path}")
                return df
        except Exception as e:
            logger.error(f"Failed to load {key} from temporary file: {e}")
    
    return None

def set_default_feature_selection(feature_columns, method="Semua Fitur (Otomatis)"):
    """Use all processed features as the default selection for downstream training."""
    features = list(feature_columns)
    st.session_state['selected_features'] = features
    st.session_state['feature_selection_method'] = method
    st.session_state['original_feature_count'] = len(features)
    st.session_state['final_feature_count'] = len(features)

def split_processed_dataset(df_processed, test_size=0.2):
    """Split processed data once and reuse the same logic for manual and automatic transitions."""
    try:
        # Validate input data
        if df_processed is None or len(df_processed) == 0:
            raise ValueError("Dataset kosong atau tidak valid")
        
        if len(df_processed) < 10:
            raise ValueError(f"Dataset terlalu kecil ({len(df_processed)} baris). Minimal 10 baris diperlukan untuk pembagian data.")
        
        stratify_col = None
        stratify_label = None
        label_candidates = [
            col for col in df_processed.columns
            if any(k in col.lower() for k in ['fraud', 'label', 'target', 'class'])
        ]
        if label_candidates:
            candidate = label_candidates[0]
            try:
                if df_processed[candidate].nunique() == 2:
                    stratify_col = df_processed[candidate]
                    stratify_label = candidate
            except Exception as e:
                logger.warning(f"Gagal memeriksa kolom label {candidate}: {e}")
        
        # Attempt stratified split if label column is available
        try:
            train_df, test_df = train_test_split(
                df_processed,
                test_size=test_size,
                random_state=42,
                stratify=stratify_col,
            )
        except ValueError as e:
            # Fallback to non-stratified split if stratification fails
            logger.warning(f"Stratified split gagal: {e}. Menggunakan regular split.")
            train_df, test_df = train_test_split(
                df_processed,
                test_size=test_size,
                random_state=42,
            )
        
        # Validate split results
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Pembagian data menghasilkan dataset kosong")
        
        return train_df, test_df, stratify_label
    
    except Exception as e:
        logger.error(f"Error dalam split_processed_dataset: {e}")
        raise

def make_json_serializable(obj):
    """Convert non-serializable objects to serializable types"""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, 'dtype') and pd.api.types.is_object_dtype(obj.dtype):
        return str(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

MODEL_PREFIX = "models/fraud_detector"

def zip_model_artifacts():
    """Zip all model artifacts into a single file for sharing.
    Returns a BytesIO buffer containing the zip file.
    """
    import zipfile
    from io import BytesIO
    zip_buffer = BytesIO()
    
    # List of all model files (both required and optional
    model_files = [
        f"{MODEL_PREFIX}_params.json",
        f"{MODEL_PREFIX}_scaler.pkl",
        f"{MODEL_PREFIX}_imputer.pkl",
        f"{MODEL_PREFIX}_isolation_forest.pkl",
        f"{MODEL_PREFIX}_autoencoder.pt",
        f"{MODEL_PREFIX}_xgboost.pkl",
        f"{MODEL_PREFIX}_dbscan.pkl",
        f"{MODEL_PREFIX}_gnn.pt"
    ]
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in model_files:
            if os.path.exists(file_path):
                # Add file to zip with just the filename (without directory path)
                zip_file.write(file_path, arcname=os.path.basename(file_path))
    
    zip_buffer.seek(0)
    return zip_buffer

def persisted_model_artifacts_exist():
    """Check whether a persisted detector can be loaded from disk."""
    required_files = [
        f"{MODEL_PREFIX}_params.json",
        f"{MODEL_PREFIX}_scaler.pkl"
    ]
    model_files = [
        f"{MODEL_PREFIX}_isolation_forest.pkl",
        f"{MODEL_PREFIX}_autoencoder.pt",
        f"{MODEL_PREFIX}_xgboost.pkl",
        f"{MODEL_PREFIX}_dbscan.pkl",
        f"{MODEL_PREFIX}_gnn.pt"
    ]
    return all(os.path.exists(path) for path in required_files) and any(
        os.path.exists(path) for path in model_files
    )

def load_persisted_detector():
    """Load the trained detector and training metadata from disk when session state is empty."""
    if 'detector' in st.session_state:
        return st.session_state['detector']

    if not persisted_model_artifacts_exist():
        return None

    try:
        with open(f"{MODEL_PREFIX}_params.json", 'r') as f:
            params = json.load(f)

        training_metadata = params.get('training_metadata', {})
        training_features = (
            training_metadata.get('training_features')
            or st.session_state.get('training_features')
            or st.session_state.get('feature_columns')
        )
        if not training_features:
            raise ValueError("Metadata fitur training tidak tersedia di model yang tersimpan.")

        detector = CombinedAnomalyDetector(
            autoencoder_params=params.get('autoencoder_params', {}),
            dbscan_params=params.get('dbscan_params', {}),
            xgboost_params=params.get('xgboost_params', {}),
            gnn_params=params.get('gnn_params', {}),
            algorithms=params.get('algorithms', ['isolation_forest', 'autoencoder', 'xgboost']),
            use_dynamic_weights=False
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        detector.load_models(
            MODEL_PREFIX,
            num_features=len(training_features),  # type: ignore
            num_classes=2,
            device=device
        )

        st.session_state['detector'] = detector
        st.session_state['model_trained'] = True
        st.session_state['training_features'] = training_features
        st.session_state['feature_selection_method'] = training_metadata.get(
            'feature_selection_method',
            st.session_state.get('feature_selection_method', 'Persisted Model')
        )
        st.session_state['training_mode'] = normalize_training_mode(
            training_metadata.get(
                'training_mode',
                st.session_state.get('training_mode', TRAINING_MODE_UNSUPERVISED)
            )
        )
        if training_metadata.get('label_column'):
            st.session_state['training_label_column'] = training_metadata['label_column']

        return detector
    except Exception as e:
        st.error(f"❌ Gagal memuat model tersimpan: {str(e)}")
        return None

def _derive_inference_feature(df, feature_name):
    """Derive a training feature from incoming data when possible."""
    if feature_name in df.columns:
        return df[feature_name], 'existing'

    if feature_name.endswith('_missing'):
        original_col = feature_name[:-8]
        if original_col in df.columns:
            return df[original_col].isnull().astype(np.int8), 'derived'

    if feature_name.endswith('_encoded'):
        original_col = feature_name[:-8]
        if original_col in df.columns:
            encoded = pd.Series(pd.factorize(df[original_col].fillna('Unknown'))[0], index=df.index)
            return encoded, 'derived'

    if feature_name.endswith('_freq_encoded'):
        original_col = feature_name[:-13]
        if original_col in df.columns:
            freq_map = df[original_col].fillna('Unknown').value_counts()
            return df[original_col].fillna('Unknown').map(freq_map).fillna(0), 'derived'

    if feature_name.endswith('_freq_binned'):
        original_col = feature_name[:-12]
        if original_col in df.columns:
            freq_map = df[original_col].fillna('Unknown').value_counts()
            freq_values = df[original_col].fillna('Unknown').map(freq_map).fillna(0)
            if freq_map.nunique() > 1:
                freq_bins = pd.qcut(freq_map, q=min(10, freq_map.nunique()), labels=False, duplicates='drop')
                freq_binned_map = freq_map.groupby(freq_bins).mean().to_dict()
                return freq_values.map(freq_binned_map).fillna(0), 'derived'
            return freq_values, 'derived'

    if feature_name.endswith('_zscore'):
        original_col = feature_name[:-7]
        if original_col in df.columns:
            series: pd.Series = pd.Series(pd.to_numeric(df[original_col], errors='coerce'), dtype=float)
            std = series.std()
            if pd.notna(std) and std > 0:
                return (series - series.mean()) / std, 'derived'
            return pd.Series(0.0, index=df.index), 'derived'

    if feature_name.endswith('_pct_rank'):
        original_col = feature_name[:-9]
        if original_col in df.columns:
            return pd.Series(pd.to_numeric(df[original_col], errors='coerce'), dtype=float).rank(pct=True), 'derived'

    if feature_name.endswith('_squared'):
        original_col = feature_name[:-8]
        if original_col in df.columns:
            series = pd.Series(pd.to_numeric(df[original_col], errors='coerce'), dtype=float).fillna(0)
            return series ** 2, 'derived'

    if feature_name.endswith('_log'):
        original_col = feature_name[:-4]
        if original_col in df.columns:
            series = pd.Series(pd.to_numeric(df[original_col], errors='coerce'), dtype=float).fillna(0)
            return np.log1p(series.clip(lower=0)), 'derived'

    if feature_name.endswith('_high'):
        original_col = feature_name[:-5]
        if original_col in df.columns:
            series = pd.Series(pd.to_numeric(df[original_col], errors='coerce'), dtype=float)
            threshold = series.quantile(0.75)
            return (series > threshold).astype(np.int8), 'derived'

    if feature_name.endswith('_late'):
        original_col = feature_name[:-5]
        if original_col in df.columns:
            series = pd.Series(pd.to_numeric(df[original_col], errors='coerce'), dtype=float)
            return (series > series.quantile(0.9)).astype(np.int8), 'derived'

    if feature_name.endswith('_quick'):
        original_col = feature_name[:-6]
        if original_col in df.columns:
            series = pd.Series(pd.to_numeric(df[original_col], errors='coerce'), dtype=float)
            return (series < series.quantile(0.1)).astype(np.int8), 'derived'

    if feature_name.endswith('_group_encoded'):
        original_col = feature_name[:-14]
        if original_col in df.columns:
            groups = pd.cut(
                pd.to_numeric(df[original_col], errors='coerce'),  # type: ignore
                bins=[0, 18, 35, 50, 65, 100],
                labels=['0-18', '19-35', '36-50', '51-65', '65+']
            )
            return pd.Series(pd.factorize(groups)[0], index=df.index), 'derived'

    if feature_name.endswith('_day_of_week'):
        original_col = feature_name[:-12]
        if original_col in df.columns:
            dates = pd.Series(pd.to_datetime(df[original_col], errors='coerce'))
            return dates.dt.dayofweek, 'derived'

    if feature_name.endswith('_month'):
        original_col = feature_name[:-6]
        if original_col in df.columns:
            dates = pd.Series(pd.to_datetime(df[original_col], errors='coerce'))
            return dates.dt.month, 'derived'

    if feature_name.endswith('_year'):
        original_col = feature_name[:-5]
        if original_col in df.columns:
            dates = pd.Series(pd.to_datetime(df[original_col], errors='coerce'))
            return dates.dt.year, 'derived'

    return None, 'error'


def get_gpu_status():
    """
    Get GPU status information.
    Returns a dictionary with GPU configuration details.
    """
    try:
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': None,
            'device_name': 'CPU',
            'torch_version': torch.__version__,
            'compute_capability': None,
            'total_memory': 0,
        }
        
        if torch.cuda.is_available():
            try:
                gpu_info['current_device'] = torch.cuda.current_device()
                gpu_info['device_name'] = torch.cuda.get_device_name(0)
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_info['compute_capability'] = f"{gpu_props.major}.{gpu_props.minor}"
                gpu_info['total_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
            except Exception as e:
                logger.warning(f"Error getting GPU details: {e}")
        
        return gpu_info
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        return {
            'cuda_available': False,
            'device_count': 0,
            'current_device': None,
            'device_name': 'CPU',
            'torch_version': torch.__version__ if hasattr(torch, '__version__') else 'Unknown',
            'compute_capability': None,
            'total_memory': 0,
        }


def get_gpu_status_display():
    """
    Get formatted GPU status string for UI display.
    """
    gpu_info = get_gpu_status()
    
    if gpu_info['cuda_available']:
        memory_info = f" ({gpu_info['total_memory']:.1f}GB)" if gpu_info['total_memory'] > 0 else ""
        device_info = f" [Compute {gpu_info['compute_capability']}]" if gpu_info['compute_capability'] else ""
        return f"🚀 GPU: {gpu_info['device_name']}{memory_info}{device_info}"
    else:
        # Check if it's ROCm or standard CPU
        if 'rocm' in gpu_info['torch_version'].lower():
            return "🔴 GPU (ROCm): No device detected"
        return "💻 Engine: CPU Multithreaded"
      
def _derive_inference_feature(df, feature_name, training_stats: dict | None = None):
    """
    Derive a feature from available columns for inference.

    Returns a (series, source) tuple where source is one of:
        'existing'  – column present in df as-is
        'derived'   – computed from related columns
        'filled'    – imputed with a domain-neutral default (not just 0)

    Parameters
    ----------
    df : pd.DataFrame
        Incoming batch dataset.
    feature_name : str
        Name of the feature expected by the trained model.
    training_stats : dict, optional
        Median / mean statistics saved from training (from metadata.json).
        Keys are feature names; values are numeric medians. Used for imputation
        so that missing features receive a realistic default value rather than 0.
    """
    if training_stats is None:
        training_stats = {}

    # ── 1. Column exists directly ──────────────────────────────────────────────
    if feature_name in df.columns:
        return df[feature_name], 'existing'

    # ── 2. Suffix-based derivations ───────────────────────────────────────────
    if feature_name.endswith('_high'):
        original_col = feature_name[:-5]
        if original_col in df.columns:
            s = pd.to_numeric(df[original_col], errors='coerce').astype(float)
            return (s > s.quantile(0.9)).astype(np.int8), 'derived'

    if feature_name.endswith('_very_high'):
        original_col = feature_name[:-10]
        if original_col in df.columns:
            s = pd.to_numeric(df[original_col], errors='coerce').astype(float)
            return (s > s.quantile(0.9)).astype(np.int8), 'derived'

    if feature_name.endswith('_late'):
        original_col = feature_name[:-5]
        if original_col in df.columns:
            s = pd.to_numeric(df[original_col], errors='coerce').astype(float)
            return (s > s.quantile(0.9)).astype(np.int8), 'derived'

    if feature_name.endswith('_quick'):
        original_col = feature_name[:-6]
        if original_col in df.columns:
            s = pd.to_numeric(df[original_col], errors='coerce').astype(float)
            return (s < s.quantile(0.1)).astype(np.int8), 'derived'

    if feature_name.endswith('_zscore'):
        original_col = feature_name[:-7]
        if original_col in df.columns:
            s = pd.to_numeric(df[original_col], errors='coerce').astype(float)
            std = s.std()
            if std > 0:
                return ((s - s.mean()) / std), 'derived'

    if feature_name.endswith('_group_encoded'):
        original_col = feature_name[:-14]
        if original_col in df.columns:
            groups = pd.cut(
                pd.to_numeric(df[original_col], errors='coerce'),
                bins=[0, 18, 35, 50, 65, 100],
                labels=['0-18', '19-35', '36-50', '51-65', '65+']
            )
            return pd.Series(pd.factorize(groups)[0], index=df.index), 'derived'

    if feature_name.endswith('_day_of_week'):
        original_col = feature_name[:-12]
        if original_col in df.columns:
            return pd.to_datetime(df[original_col], errors='coerce').dt.dayofweek, 'derived'

    if feature_name.endswith('_month'):
        original_col = feature_name[:-6]
        if original_col in df.columns:
            return pd.to_datetime(df[original_col], errors='coerce').dt.month, 'derived'

    if feature_name.endswith('_year'):
        original_col = feature_name[:-5]
        if original_col in df.columns:
            return pd.to_datetime(df[original_col], errors='coerce').dt.year, 'derived'

    if feature_name.endswith('_quarter'):
        original_col = feature_name[:-8]
        if original_col in df.columns:
            return pd.to_datetime(df[original_col], errors='coerce').dt.quarter, 'derived'

    if '_to_' in feature_name and feature_name.endswith('_ratio'):
        base_name = feature_name[:-6]
        parts = base_name.split('_to_')
        if len(parts) == 2 and parts[0] in df.columns and parts[1] in df.columns:
            left = pd.to_numeric(df[parts[0]], errors='coerce').astype(float)
            right = pd.to_numeric(df[parts[1]], errors='coerce').astype(float)
            ratio = left / (right + 1e-8)
            return ratio.replace([np.inf, -np.inf], np.nan), 'derived'

    # ── 3. Known domain-specific engineered features ──────────────────────────
    # high_amount_quick_submit: requires at least one amount-like and time-like column
    if feature_name == 'high_amount_quick_submit':
        amount_candidates = [c for c in df.columns if any(k in c.lower() for k in ('amount', 'billed', 'cost', 'paid'))]
        time_candidates   = [c for c in df.columns if any(k in c.lower() for k in ('days', 'time', 'duration', 'gap'))]
        if amount_candidates and time_candidates:
            a = pd.to_numeric(df[amount_candidates[0]], errors='coerce').astype(float)
            t = pd.to_numeric(df[time_candidates[0]],  errors='coerce').astype(float)
            return ((a > a.quantile(0.75)) & (t < t.quantile(0.25))).astype(np.int8), 'derived'

    if feature_name == 'payment_ratio':
        billed = next((c for c in df.columns if 'billed' in c.lower()), None)
        paid   = next((c for c in df.columns if 'paid'   in c.lower()), None)
        if billed and paid:
            b = pd.to_numeric(df[billed], errors='coerce').astype(float)
            p = pd.to_numeric(df[paid],   errors='coerce').astype(float)
            ratio = (p / (b + 1e-8)).clip(0.0, 1.0)
            return ratio.replace([np.inf, -np.inf], np.nan), 'derived'

    if feature_name == 'allowance_ratio':
        billed   = next((c for c in df.columns if 'billed'  in c.lower()), None)
        allowed  = next((c for c in df.columns if 'allowed' in c.lower()), None)
        if billed and allowed:
            b = pd.to_numeric(df[billed],  errors='coerce').astype(float)
            a = pd.to_numeric(df[allowed], errors='coerce').astype(float)
            ratio = (a / (b + 1e-8)).clip(0.0, 1.0)
            return ratio.replace([np.inf, -np.inf], np.nan), 'derived'

    # ── 4. Fallback: impute using training median, else 0 ─────────────────────
    fill_val = training_stats.get(feature_name, 0.0)
    return pd.Series(float(fill_val), index=df.index), 'filled'


def build_aligned_inference_features(df, training_features, training_stats: dict | None = None):
    """
    Build an inference matrix that exactly matches the training feature list and order.

    Missing features are derived via known engineering rules when possible,
    or imputed using stored training medians (from metadata.json) rather
    than silent zeros.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed input batch dataset.
    training_features : list[str]
        Ordered list of feature names the model was trained on.
    training_stats : dict, optional
        Median statistics per feature from training. Used for domain-neutral imputation.

    Returns
    -------
    aligned_df : pd.DataFrame  (float32, same index as df)
    summary    : dict with keys:
        - expected_features   : int
        - existing_features   : list[str]
        - derived_features    : list[str]
        - filled_features     : list[str]   (previously 'filled_zero_features')
        - fill_values_used    : dict[str, float]
    """
    if training_stats is None:
        training_stats = {}

    aligned_df       = pd.DataFrame(index=df.index)
    existing_features = []
    derived_features  = []
    filled_features   = []
    fill_values_used  = {}

    for feature_name in training_features:
        series, source = _derive_inference_feature(df, feature_name, training_stats)

        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=df.index)
        else:
            series = series.reindex(df.index)

        if not pd.api.types.is_numeric_dtype(series):
            series = pd.Series(pd.factorize(series.fillna('Unknown'))[0], index=df.index)

        series = pd.to_numeric(series, errors='coerce').astype(float)

        if series.isnull().any():
            if source == 'existing':
                # For existing columns, use the column's own median
                fill_value = series.median() if not series.dropna().empty else training_stats.get(feature_name, 0.0)
            else:
                # For engineered/filled features use training median or domain neutral value
                fill_value = training_stats.get(feature_name, 0.0)
            series = series.fillna(fill_value)
        else:
            fill_value = None

        aligned_df[feature_name] = series.astype(np.float32)

        if source == 'existing':
            existing_features.append(feature_name)
        elif source == 'derived':
            derived_features.append(feature_name)
        else:
            filled_features.append(feature_name)
            fill_values_used[feature_name] = fill_value if fill_value is not None else training_stats.get(feature_name, 0.0)

    summary = {
        'expected_features':    len(training_features),  # type: ignore
        'existing_features':    existing_features,
        'derived_features':     derived_features,
        'filled_features':      filled_features,
        # Legacy alias kept for backward compatibility
        'filled_zero_features': filled_features,
        'fill_values_used':     fill_values_used,
    }
    return aligned_df, summary


# ─────────────────────────────────────────────────────────────────────────────
# DATASET TEMPLATE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

# Core columns required for full operation of all 9 business rule modules and GNN
TEMPLATE_CORE_COLUMNS = [
    "claim_id",
    "patient_id",
    "provider_id",
    "service_code",
    "diagnosis_code",
    "billing_date",
    "service_date",
    "billed_amount",
    "paid_amount",
    "allowed_amount",
    "claim_status",
    "patient_age",
    "length_of_stay",
    "quantity",
]

# Human-readable description for each column — shown in the schema readiness card
COLUMN_DESCRIPTIONS = {
    "claim_id":       "Identifikasi unik klaim (String/Integer)",
    "patient_id":     "Identifikasi unik pasien (diperlukan untuk Repeat Billing & Fuzzy Match)",
    "provider_id":    "Kode dokter/faskes (diperlukan untuk Provider Capacity & Graf GNN)",
    "service_code":   "Kode prosedur/tindakan medis (diperlukan untuk Phantom Service & Upcoding)",
    "diagnosis_code": "Kode diagnosis ICD (diperlukan untuk deteksi Phantom Service & GNN)",
    "billing_date":   "Tanggal penagihan dalam format YYYY-MM-DD (diperlukan untuk aturan temporal)",
    "service_date":   "Tanggal layanan diberikan (YYYY-MM-DD)",
    "billed_amount":  "Total nominal ditagihkan dalam Rupiah (Float)",
    "paid_amount":    "Nominal yang dibayarkan (Float) — digunakan untuk payment_ratio",
    "allowed_amount": "Nominal yang disetujui (Float) — digunakan untuk allowance_ratio",
    "claim_status":   "Status klaim: APPROVED / PENDING / REJECTED",
    "patient_age":    "Usia pasien dalam tahun (Integer)",
    "length_of_stay": "Lama rawat inap dalam hari (Integer, isi 0 untuk rawat jalan)",
    "quantity":       "Jumlah unit tindakan/obat/alkes yang diklaim (Integer)",
}

# Which modul / rule depends on each column
COLUMN_RULE_DEPENDENCIES = {
    "claim_id":       ["Audit Trail", "Duplicate Payment"],
    "patient_id":     ["Repeat Billing", "Fuzzy Claim Matching"],
    "provider_id":    ["Provider Capacity", "GNN Graf Relasi"],
    "service_code":   ["Phantom Service", "Upcoding & Unbundling"],
    "diagnosis_code": ["Phantom Service", "GNN Graf Relasi"],
    "billing_date":   ["Repeat Billing (30-day window)", "high_amount_quick_submit"],
    "service_date":   ["Provider Capacity", "Length of Stay"],
    "billed_amount":  ["ML Ensemble (Feature: amount)", "payment_ratio", "allowance_ratio"],
    "paid_amount":    ["payment_ratio", "Inflated Bill & Cloning"],
    "allowed_amount": ["allowance_ratio"],
    "claim_status":   ["Duplicate Payment & Status Check"],
    "patient_age":    ["Feature Engineering: age_group_encoded"],
    "length_of_stay": ["Length of Stay & Readmission"],
    "quantity":       ["Medication & Device Fraud"],
}


def generate_sample_claims_template(n_rows: int = 5) -> pd.DataFrame:
    """
    Generate a sample insurance claims DataFrame with all core columns.

    The template contains realistic sample values to guide users in preparing
    their data for batch anomaly detection.

    Parameters
    ----------
    n_rows : int
        Number of sample rows to include. Default is 5.

    Returns
    -------
    pd.DataFrame
        Template DataFrame with TEMPLATE_CORE_COLUMNS as columns.
    """
    import random
    import datetime

    random.seed(42)
    base_date = datetime.date(2024, 1, 15)
    statuses  = ["APPROVED", "PENDING", "REJECTED", "APPROVED", "APPROVED"]
    services  = ["99213", "99214", "71046", "80053", "43239"]
    diagnoses = ["J06.9", "E11.9", "I10", "Z00.00", "K21.0"]

    rows = []
    for i in range(n_rows):
        billing_dt  = base_date + datetime.timedelta(days=i * 7)
        service_dt  = billing_dt - datetime.timedelta(days=random.randint(0, 5))
        billed      = round(random.uniform(500_000, 10_000_000), 0)
        paid        = round(billed * random.uniform(0.60, 1.00), 0)
        allowed     = round(billed * random.uniform(0.70, 1.00), 0)
        rows.append({
            "claim_id":       f"CLM-{1000 + i:05d}",
            "patient_id":     f"PAT-{200 + (i % 3):04d}",
            "provider_id":    f"PROV-{10 + (i % 2):03d}",
            "service_code":   services[i % len(services)],
            "diagnosis_code": diagnoses[i % len(diagnoses)],
            "billing_date":   billing_dt.strftime("%Y-%m-%d"),
            "service_date":   service_dt.strftime("%Y-%m-%d"),
            "billed_amount":  billed,
            "paid_amount":    paid,
            "allowed_amount": allowed,
            "claim_status":   statuses[i % len(statuses)],
            "patient_age":    random.randint(20, 75),
            "length_of_stay": random.randint(0, 7),
            "quantity":       random.randint(1, 5),
        })

    return pd.DataFrame(rows)


def render_schema_readiness_card(df: pd.DataFrame) -> dict:
    """
    Analyse an uploaded batch dataset against TEMPLATE_CORE_COLUMNS and render
    a visual schema readiness card in the Streamlit UI.

    Displays:
    - A green/amber/red banner summarising overall schema completeness.
    - A per-column table indicating availability and which rules are affected.

    Parameters
    ----------
    df : pd.DataFrame
        The uploaded/preprocessed batch dataset.

    Returns
    -------
    dict  with keys:
        - 'complete_pct'   : float (0–100)
        - 'present'        : list[str]
        - 'missing'        : list[str]
        - 'affected_rules' : list[str]   (unique rules that cannot run)
    """
    present = [c for c in TEMPLATE_CORE_COLUMNS if c in df.columns]
    missing = [c for c in TEMPLATE_CORE_COLUMNS if c not in df.columns]
    complete_pct = len(present) / len(TEMPLATE_CORE_COLUMNS) * 100

    # Collect rules that cannot run
    affected_rules: list[str] = []
    for col in missing:
        affected_rules.extend(COLUMN_RULE_DEPENDENCIES.get(col, []))
    affected_rules = list(dict.fromkeys(affected_rules))  # deduplicate preserving order

    # ── Banner ──────────────────────────────────────────────────────────────
    if complete_pct == 100:
        st.success(
            f"✅ **Skema Data Lengkap 100%** — Seluruh {len(TEMPLATE_CORE_COLUMNS)} kolom inti tersedia. "
            "Semua 9 modul aturan bisnis dan GNN aktif penuh."
        )
    elif complete_pct >= 70:
        st.warning(
            f"⚠️ **Skema Data {complete_pct:.0f}%** — {len(missing)} kolom inti tidak ditemukan. "
            f"Modul berikut mungkin tidak aktif: *{', '.join(affected_rules[:4])}*"
            + (" dan lainnya." if len(affected_rules) > 4 else ".")
        )
    else:
        st.error(
            f"❌ **Skema Data Tidak Memadai ({complete_pct:.0f}%)** — {len(missing)} kolom penting tidak ada. "
            "Akurasi deteksi akan sangat terdegradasi. Unduh template dan sesuaikan dataset Anda."
        )

    # ── Per-column status table ──────────────────────────────────────────────
    table_rows = []
    for col in TEMPLATE_CORE_COLUMNS:
        status = "✅ Ada" if col in df.columns else "❌ Tidak ada"
        rules  = ", ".join(COLUMN_RULE_DEPENDENCIES.get(col, ["-"]))
        table_rows.append({
            "Kolom": col,
            "Status": status,
            "Keterangan": COLUMN_DESCRIPTIONS.get(col, ""),
            "Modul yang Bergantung": rules,
        })

    with st.expander("📋 Rincian Kelengkapan Kolom Data", expanded=(complete_pct < 100)):
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
        )


    return {
        "complete_pct":   complete_pct,
        "present":        present,
        "missing":        missing,
        "affected_rules": affected_rules,
    }
