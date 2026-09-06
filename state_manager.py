import streamlit as st
import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from file_handler import save_processed_data, load_processed_data
from cache_manager import get_cache_path

logger = logging.getLogger(__name__)

TRAINING_MODE_UNSUPERVISED = "unsupervised"
TRAINING_MODE_SUPERVISED = "supervised"
VALID_PAGES = {"home", "collect", "train", "evaluate", "detect", "status", "settings"}


def navigate_to_page(page_name, *, rerun=True):
    """Safely switch pages without creating rerun loops on each button click."""
    page_key = str(page_name or "home").strip().lower()
    if page_key not in VALID_PAGES:
        raise ValueError(f"Halaman tidak valid: {page_name}")

    current_page = st.session_state.get("page")
    st.session_state["page"] = page_key

    if rerun and current_page != page_key:
        try:
            st.rerun()
        except Exception as exc:
            logger.warning(f"Rerun halaman gagal saat navigasi ke '{page_key}': {exc}")

    return page_key


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

def get_processed_lazy():
    """Helper function to get Polars LazyFrame for out-of-core queries"""
    import polars as pl
    if 'df_processed_path' not in st.session_state:
        return None
    path = st.session_state['df_processed_path']
    if not os.path.exists(path):
        return None
    return pl.scan_parquet(path)

def get_processed_sample(n=1000):
    """Helper function to get a lightweight head sample dataframe"""
    lazy_lf = get_processed_lazy()
    if lazy_lf is None:
        return None
    return lazy_lf.head(n).collect().to_pandas()

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
                                os.makedirs(temp_dir, exist_ok=True)
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
