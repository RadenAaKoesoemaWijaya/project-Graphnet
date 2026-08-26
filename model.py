import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):  # pragma: no cover - optional deep learning dependency / Windows DLL issues
    import sys
    sys.modules.pop('torch', None)
    sys.modules.pop('torch_geometric', None)
    TORCH_AVAILABLE = False

if not TORCH_AVAILABLE:
    class _DummyModule:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def parameters(self):
            return []
        def eval(self):
            return self
        def train(self):
            return self
        def to(self, *args, **kwargs):
            return self

    class _DummyNN:
        Module = _DummyModule
        ModuleList = list
        Sequential = list
        Linear = _DummyModule
        ReLU = _DummyModule
        BatchNorm1d = _DummyModule
        Dropout = _DummyModule
        CrossEntropyLoss = _DummyModule
        MSELoss = _DummyModule

    class _DummyTensor(np.ndarray):
        def __new__(cls, input_array, dtype=None):
            arr = np.asarray(input_array, dtype=dtype)
            return arr.view(cls)
        def t(self):
            return _DummyTensor(self.T)
        def contiguous(self):
            return self
        def to(self, *args, **kwargs):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return np.asarray(self)
        def dim(self):
            return self.ndim
        def size(self, dim=None):
            if dim is None:
                return self.shape
            return self.shape[dim]

    class _DummyDevice:
        def __init__(self, type_str='cpu'):
            self.type = str(type_str)
        def __str__(self):
            return self.type
        def __repr__(self):
            return f"device(type='{self.type}')"

    class _DummyTorch:
        nn = _DummyNN
        Tensor = _DummyTensor
        device = _DummyDevice

        @staticmethod
        def FloatTensor(*args, **kwargs):
            if len(args) == 1:
                return _DummyTensor(args[0], dtype=np.float32)
            return _DummyTensor(args, dtype=np.float32)

        @staticmethod
        def LongTensor(*args, **kwargs):
            if len(args) == 1:
                return _DummyTensor(args[0], dtype=np.int64)
            return _DummyTensor(args, dtype=np.int64)

        @staticmethod
        def as_tensor(x, *args, **kwargs):
            return _DummyTensor(np.asarray(x))

        @staticmethod
        def manual_seed(seed):
            pass

        class cuda:
            @staticmethod
            def is_available():
                return False
            @staticmethod
            def manual_seed_all(seed):
                pass
            @staticmethod
            def get_device_properties(*args):
                return None
            @staticmethod
            def get_device_name(*args):
                return "CPU"

        class backends:
            class cudnn:
                deterministic = True
                benchmark = False

    torch = _DummyTorch()
    nn = torch.nn
    F = None
    GATConv = _DummyModule
    global_add_pool = None
    global_mean_pool = None
    Data = _DummyModule


import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, auc, precision_score, recall_score
import joblib
import json
import os
import logging
import inspect
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import streamlit as st

# QW6: module-level logger — replaced print() calls throughout the model
# so that logs can be redirected to a file or aggregated by Docker / CI.
# R3: the formatter (plain or JSON) is driven by the ASTINA_LOG_FORMAT
# env var through logging_config.configure_logging().
from logging_config import configure_logging
configure_logging()
logger = logging.getLogger("graphnet.model")

def get_optimal_batch_size(device, num_samples, feature_dim, default_batch=1024):
    """
    Calculate optimal batch size based on available memory and data characteristics.
    
    Args:
        device: torch.device (cpu or cuda)
        num_samples: Number of samples in dataset
        feature_dim: Number of features per sample
        default_batch: Default batch size to fall back to
    
    Returns:
        Optimal batch size for the given configuration
    """
    try:
        if device.type == 'cuda' and torch.cuda.is_available():
            # GPU-based calculation
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # Reserve 2GB for overhead and other operations
            available_memory = (gpu_memory - 2 * 1024**3) * 0.8
            # Each sample uses feature_dim * 4 bytes (float32)
            sample_size_bytes = feature_dim * 4
            max_batch = int(available_memory / sample_size_bytes)
            # Cap at reasonable limits
            optimal_batch = min(max_batch, num_samples, 8192)
            # Ensure minimum batch size
            optimal_batch = max(optimal_batch, 32)
            logger.info(f"GPU memory: {gpu_memory/1024**3:.2f}GB, Optimal batch size: {optimal_batch}")
            return optimal_batch
        else:
            # CPU-based calculation - use larger batches but respect system memory
            try:
                import psutil
                available_ram = psutil.virtual_memory().available
                # Reserve 4GB for system and other processes
                available_memory = (available_ram - 4 * 1024**3) * 0.6
                sample_size_bytes = feature_dim * 4
                max_batch = int(available_memory / sample_size_bytes)
                # CPU can handle larger batches
                optimal_batch = min(max_batch, num_samples, 16384)
                optimal_batch = max(optimal_batch, 64)
                logger.info(f"CPU available RAM: {available_ram/1024**3:.2f}GB, Optimal batch size: {optimal_batch}")
                return optimal_batch
            except ImportError:
                # psutil not available, use conservative default
                logger.warning("psutil not available, using default batch size")
                return min(default_batch, num_samples)
    except Exception as e:
        logger.warning(f"Failed to calculate optimal batch size, using default: {e}")
        return min(default_batch, num_samples)

def get_adaptive_gnn_threshold(device, feature_dim, num_nodes):
    """
    Calculate adaptive threshold for using NeighborLoader in GNN inference.
    
    Args:
        device: torch.device (cpu or cuda)
        feature_dim: Number of features per node
        num_nodes: Number of nodes in graph
    
    Returns:
        Adaptive threshold for using NeighborLoader
    """
    try:
        if device is not None and getattr(device, 'type', None) == 'cuda' and torch is not None and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # Conservative threshold for GPUs to prevent OOM
            if gpu_memory < 4 * 1024**3:  # < 4GB
                return 3000
            elif gpu_memory < 8 * 1024**3:  # < 8GB
                return 5000
            else:  # >= 8GB
                return 8000
        else:
            # CPU: conservative threshold
            try:
                import psutil
                available_ram = psutil.virtual_memory().available
                if available_ram < 8 * 1024**3:  # < 8GB RAM
                    return 2000
                elif available_ram < 16 * 1024**3:  # < 16GB RAM
                    return 3000
                else:  # >= 16GB RAM
                    return 5000
            except (ImportError, Exception):
                # psutil not available, use conservative default
                return 3000
    except Exception as e:
        logger.warning(f"Failed to calculate adaptive GNN threshold, using default: {e}")
        return 3000  # Conservative default

def fast_rank_normalize(arr, threshold=10000):
    """
    Fast rank normalization with approximation for large arrays.
    Uses percentile-based approximation for large arrays to avoid O(n log n) sorting.
    
    Args:
        arr: Input array to normalize
        threshold: Array size threshold for using approximation
    
    Returns:
        Normalized array in [0, 1] range
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    
    if n == 0:
        return arr
    
    # For small arrays, use exact rank normalization
    if n <= threshold:
        return exact_rank_normalize(arr)
    
    # For large arrays, use percentile-based approximation
    try:
        # Compute percentiles at regular intervals
        num_percentiles = min(1000, n // 100)  # Adaptive number of percentiles
        percentiles = np.linspace(0, 100, num_percentiles)
        percentile_values = np.percentile(arr, percentiles)
        
        # Map each value to its approximate rank using interpolation
        # This is O(n log k) where k is number of percentiles, much faster than O(n log n)
        ranks = np.interp(arr, percentile_values, percentiles * n / 100)
        
        # Normalize to [0, 1]
        norm = (ranks - 0.5) / n
        norm = np.clip(norm, 0, 1)
        
        if np.allclose(norm, norm[0]):
            return arr  # cannot rank, return original
        
        logger.debug(f"Used fast rank normalization for array of size {n}")
        return norm
    except Exception as e:
        logger.warning(f"Fast rank normalization failed, falling back to exact: {e}")
        return exact_rank_normalize(arr)

def exact_rank_normalize(arr):
    """
    Exact rank normalization using full sort operation.
    
    Args:
        arr: Input array to normalize
    
    Returns:
        Normalized array in [0, 1] range
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n == 0:
        return arr
    
    # Average ranks break ties fairly.
    order = np.argsort(arr, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    
    # Average duplicates: group identical values.
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    avg_ranks = np.zeros_like(ranks)
    for i, cnt in enumerate(counts):
        if cnt == 1:
            avg_ranks[inv == i] = ranks[inv == i]
        else:
            avg_ranks[inv == i] = ranks[inv == i].mean()
    
    denom = max(1.0, float(n))
    norm = (avg_ranks - 0.5) / denom  # in (0, 1)
    
    if np.allclose(norm, norm[0]):
        return arr  # cannot rank, return original
    
    return norm

class ProgressiveFeatureScaler:
    """
    Progressive feature scaler that caches scaled features to avoid redundant computation.
    This is especially useful for ensemble models where the same features are scaled multiple times.
    """
    def __init__(self, imputer=None, scaler=None):
        self.imputer = imputer if imputer is not None else SimpleImputer(strategy='median')
        self.scaler = scaler if scaler is not None else StandardScaler()
        self._features_hash = None
        self._features_imputed = None
        self._features_scaled = None
    
    def get_scaled_features(self, features, force_refresh=False):
        """
        Get scaled features, using cached values if available and features haven't changed.
        
        Args:
            features: Input features to scale
            force_refresh: Force recomputation even if cached
        
        Returns:
            Scaled features
        """
        # Compute hash of features to detect changes
        import hashlib
        features_bytes = features.tobytes() if hasattr(features, 'tobytes') else features.data.tobytes()
        current_hash = hashlib.md5(features_bytes).hexdigest()
        
        # Return cached if available and not forced to refresh
        if not force_refresh and self._features_hash == current_hash and self._features_scaled is not None:
            logger.debug("Using cached scaled features")
            return self._features_scaled
        
        # Compute scaled features
        logger.debug("Computing scaled features (cache miss or forced refresh)")
        self._features_imputed = self.imputer.fit_transform(features)
        self._features_scaled = self.scaler.fit_transform(self._features_imputed)
        self._features_hash = current_hash
        
        return self._features_scaled
    
    def get_imputed_features(self, features, force_refresh=False):
        """
        Get imputed features (before scaling).
        
        Args:
            features: Input features to impute
            force_refresh: Force recomputation even if cached
        
        Returns:
            Imputed features
        """
        import hashlib
        features_bytes = features.tobytes() if hasattr(features, 'tobytes') else features.data.tobytes()
        current_hash = hashlib.md5(features_bytes).hexdigest()
        
        if not force_refresh and self._features_hash == current_hash and self._features_imputed is not None:
            logger.debug("Using cached imputed features")
            return self._features_imputed
        
        logger.debug("Computing imputed features (cache miss or forced refresh)")
        self._features_imputed = self.imputer.fit_transform(features)
        self._features_scaled = self.scaler.fit_transform(self._features_imputed)
        self._features_hash = current_hash
        
        return self._features_imputed
    
    def clear_cache(self):
        """Clear cached features"""
        self._features_hash = None
        self._features_imputed = None
        self._features_scaled = None
        logger.debug("Feature scaler cache cleared")

# HDBSCAN import (faster than DBSCAN for big data)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: HDBSCAN not available. Install with: pip install hdbscan. Falling back to DBSCAN.")

# Imbalance handling imports
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")

# XGBoost/LightGBM imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# CatBoost import (excellent for categorical features)
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

try:
    from optuna.trial import Trial
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    Trial = None
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")


def _set_global_seeds(seed: int = 42):
    """Set seeds across Python, NumPy, and PyTorch for reproducible results."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    os.environ['PYTHONHASHSEED'] = str(seed)

class InsuranceAnomalyGNNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64, num_heads=8, dropout=0.2, 
                 num_layers=2, pooling='add', residual=False, batch_norm=False,
                 edge_dim=None):
        super(InsuranceAnomalyGNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        self.residual = residual
        self.batch_norm = batch_norm
        self.edge_dim = edge_dim
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(GATConv(num_features, hidden_channels, heads=num_heads, dropout=dropout, edge_dim=edge_dim))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Hidden layers
        for i in range(1, num_layers):
            if i == num_layers - 1:  # Last GAT layer
                self.gat_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout, edge_dim=edge_dim))
                if batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            else:
                self.gat_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout, edge_dim=edge_dim))
                if batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Determine the size of the output from the last GAT layer
        # If num_layers == 1, the output is hidden_channels * num_heads
        # Otherwise, if the last layer is applied, it depends on whether it's the last layer
        if num_layers == 1:
            final_hidden_size = hidden_channels * num_heads
        else:
            final_hidden_size = hidden_channels  # Last layer has heads=1
        
        # Output layers
        self.lin1 = nn.Linear(final_hidden_size, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch, edge_attr=None):
        # For claim-level anomaly detection, we need node-level predictions, not graph-level
        # So we skip global pooling and produce per-node outputs
        
        # Store original input for residual connection
        original_x = x
        
        # Process through GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            # Apply GAT layer
            x = gat_layer(x, edge_index, edge_attr=edge_attr)
            
            # Apply batch normalization if enabled
            if self.batch_norm:
                x = self.batch_norms[i](x)
            
            # Apply activation and dropout
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Apply residual connection if enabled and dimensions match
            if self.residual and i > 0 and x.size(-1) == original_x.size(-1):
                x = x + original_x
                original_x = x
        
        # Final layers (node-level predictions, skip global pooling)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)

    def get_node_embeddings(self, x, edge_index, batch, edge_attr=None):
        """Return per-node embeddings (post-GAT, pre-classifier) for visualization."""
        h = x
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(h, edge_index, edge_attr=edge_attr)
            if self.batch_norm:
                h = self.batch_norms[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

def create_graph_data(features, edge_index, labels=None):
    """Create PyTorch Geometric Data object from features and edge indices"""
    x = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edge_index)
    
    if labels is not None:
        y = torch.LongTensor(labels)
        return Data(x=x, edge_index=edge_index, y=y)
    
    return Data(x=x, edge_index=edge_index)

def train_model(model, train_loader, optimizer, device, scheduler=None):
    """Train the GNN model"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.y.size(0)
        
        total_loss += loss.item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy

def evaluate_model(model, loader, device, detailed=False):
    """Evaluate the GNN model"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            
            # Collect predictions and labels for detailed metrics
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
            correct += int((pred == data.y).sum())
            total += data.y.size(0)
    
    accuracy = correct / total
    
    if detailed:
        # Calculate detailed metrics
        # Convert both predictions and labels to integers to ensure consistency
        all_preds = [int(p) for p in all_preds]
        all_labels = [int(l) for l in all_labels]
        
        try:
            f1 = f1_score(all_labels, all_preds, average='weighted')
            report = classification_report(all_labels, all_preds, output_dict=True)
            cm = confusion_matrix(all_labels, all_preds)
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            print(f"Label types: {type(all_labels[0])}, Prediction types: {type(all_preds[0])}")
            print(f"Unique labels: {set(all_labels)}, Unique predictions: {set(all_preds)}")
            # Return default values if metrics calculation fails
            f1 = 0.0
            report = {}
            cm = np.zeros((2, 2))
        
        return accuracy, f1, report, cm, all_preds, all_labels
    
    return accuracy

def predict_new_data(model, data_loader, device):
    """Make predictions on new data"""
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.exp(out)  # Convert log_softmax to probabilities
            pred = out.argmax(dim=1)
            
            predictions.extend(pred.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)

def save_model(model, model_path, hyperparams=None, scaler=None):
    """Save model, hyperparameters, and scaler"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), model_path)
    
    # Save hyperparameters if provided
    if hyperparams:
        with open(f"{os.path.splitext(model_path)[0]}_hyperparams.json", 'w') as f:
            json.dump(hyperparams, f, indent=4)
    
    # Save scaler if provided
    if scaler:
        joblib.dump(scaler, f"{os.path.splitext(model_path)[0]}_scaler.pkl")
    
    print(f"Model saved to {model_path}")

def load_model(model_path, num_features, num_classes):
    """Load model and hyperparameters"""
    # Load hyperparameters
    hyperparams_path = f"{os.path.splitext(model_path)[0]}_hyperparams.json"
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
    else:
        # Default hyperparameters if file doesn't exist
        hyperparams = {
            'hidden_channels': 64,
            'num_heads': 8,
            'dropout': 0.2,
            'num_layers': 2,
            'pooling': 'add',
            'residual': False,
            'batch_norm': False
        }
    
    # Create model with loaded hyperparameters
    model = InsuranceAnomalyGNNModel(
        num_features=num_features,
        num_classes=num_classes,
        hidden_channels=hyperparams.get('hidden_channels', 64),
        num_heads=hyperparams.get('num_heads', 8),
        dropout=hyperparams.get('dropout', 0.2),
        num_layers=hyperparams.get('num_layers', 2),
        pooling=hyperparams.get('pooling', 'add'),
        residual=hyperparams.get('residual', False),
        batch_norm=hyperparams.get('batch_norm', False)
    )
    
    # Load model state
    model.load_state_dict(torch.load(model_path))
    
    # Load scaler if it exists
    scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    return model, hyperparams, scaler

def objective(trial, train_loader, val_loader, num_features, num_classes, device, epochs=30):
    """Optuna objective function for hyperparameter optimization"""
    # Define hyperparameters to optimize
    hidden_channels = trial.suggest_int('hidden_channels', 32, 256, step=32)
    num_heads = trial.suggest_int('num_heads', 1, 8)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    pooling = trial.suggest_categorical('pooling', ['add', 'mean'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    residual = trial.suggest_categorical('residual', [True, False])
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    
    # Create model with trial hyperparameters
    model = InsuranceAnomalyGNNModel(
        num_features=num_features,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
        num_layers=num_layers,
        pooling=pooling,
        residual=residual,
        batch_norm=batch_norm
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    patience = 5  # Early stopping patience
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_model(model, train_loader, optimizer, device)
        
        # Validate
        val_acc_result = evaluate_model(model, val_loader, device)
        val_acc = val_acc_result[0] if isinstance(val_acc_result, tuple) else val_acc_result
        
        # Report intermediate metric
        trial.report(val_acc, epoch)
        
        # Handle pruning (early stopping for this trial)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_acc

def optimize_hyperparameters(train_loader, val_loader, num_features, num_classes, device, n_trials=50, study_name="ids_gnn_optimization"):
    """Run hyperparameter optimization using Optuna"""
    # Create study directory
    os.makedirs("studies", exist_ok=True)
    
    # Create or load study
    study_path = f"studies/{study_name}.pkl"
    if os.path.exists(study_path):
        print(f"Loading existing study from {study_path}")
        study = joblib.load(study_path)
    else:
        print(f"Creating new study: {study_name}")
        study = optuna.create_study(direction="maximize", study_name=study_name, 
                                   pruner=optuna.pruners.MedianPruner())
    
    # Run optimization
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, num_features, num_classes, device), 
                  n_trials=n_trials, timeout=3600)  # 1 hour timeout
    
    # Save study
    joblib.dump(study, study_path)
    
    # Get best parameters
    best_params = study.best_params
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    
    # Create and return model with best parameters
    best_model = InsuranceAnomalyGNNModel(
        num_features=num_features,
        num_classes=num_classes,
        hidden_channels=best_params.get('hidden_channels', 64),
        num_heads=best_params.get('num_heads', 8),
        dropout=best_params.get('dropout', 0.2),
        num_layers=best_params.get('num_layers', 2),
        pooling=best_params.get('pooling', 'add'),
        residual=best_params.get('residual', False),
        batch_norm=best_params.get('batch_norm', False)
    ).to(device)
    
    return best_model, best_params

def train_with_best_params(model, train_loader, val_loader, test_loader, device, hyperparams, 
                          epochs=100, patience=10, model_save_path="models/best_ids_gnn_model.pt"):
    """Train model with best hyperparameters"""
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyperparams.get('learning_rate', 0.001),
        weight_decay=hyperparams.get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    except TypeError as e:
        logger.warning(f"ReduceLROnPlateau initialization failed: {e}. Using scheduler=None")
        scheduler = None
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training with best hyperparameters...")
    for epoch in tqdm(range(epochs)):
        # Train
        train_loss, train_acc = train_model(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_acc_result = evaluate_model(model, val_loader, device)
        val_acc = val_acc_result[0] if isinstance(val_acc_result, tuple) else val_acc_result
        val_accs.append(val_acc)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save model
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved at epoch {epoch+1} with validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set with error handling
    try:
        eval_result = evaluate_model(model, test_loader, device, detailed=True)
        if isinstance(eval_result, tuple) and len(eval_result) == 6:
            test_acc, test_f1, test_report, test_cm, _, _ = eval_result
        else:
            test_acc, test_f1, test_report, test_cm = eval_result, 0.0, {}, np.zeros((2, 2))  # type: ignore
        
        print(f"\nBest model from epoch {best_epoch+1}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print("\nClassification Report:")
        for class_id, metrics in (test_report.items() if isinstance(test_report, dict) else []):
            if isinstance(metrics, dict):
                print(f"Class {class_id}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("plots/confusion_matrix.png")
        
    except Exception as e:
        print(f"Error during final evaluation: {str(e)}")
        test_acc = best_val_acc  # Use validation accuracy as fallback
        test_f1 = 0.0
        test_report = {}
        test_cm = np.zeros((2, 2))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Test Accuracy: {test_acc:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Create directory for plots
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/training_curves.png")
    
    # Save hyperparameters
    save_model(model, model_save_path, hyperparams)
    
    return model, test_acc, test_f1, test_report, test_cm

def main(train_loader, val_loader, test_loader, num_features, num_classes, device=None):
    """Main function to run the entire model training pipeline"""
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("studies", exist_ok=True)
    
    # Hyperparameter optimization
    print("Starting hyperparameter optimization...")
    best_model, best_params = optimize_hyperparameters(
        train_loader, val_loader, num_features, num_classes, device, n_trials=20
    )
    
    # Train with best hyperparameters
    print("\nTraining with best hyperparameters...")
    final_model, test_acc, test_f1, test_report, test_cm = train_with_best_params(
        best_model, train_loader, val_loader, test_loader, device, best_params
    )
    
    print("\nTraining complete!")
    print(f"Best hyperparameters: {best_params}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    return final_model, best_params, test_acc

class APILogAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super(APILogAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        
        # First encoder layer
        self.encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Additional encoder layers
        for i in range(1, len(hidden_dims)):
            self.encoder_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        # Decoder layers (in reverse)
        self.decoder_layers = nn.ModuleList()
        
        # Build decoder layers in reverse order
        for i in range(len(hidden_dims)-1, 0, -1):
            self.decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
        
        # Final decoder layer to original dimensions
        self.decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        
        self.dropout = dropout
    
    def encode(self, x):
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def decode(self, x):
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def forward(self, x):
        # Encode
        encoded = self.encode(x)
        # Decode
        decoded = self.decode(encoded)
        return decoded

def train_autoencoder(model, train_loader, optimizer, device):
    """Train the autoencoder model"""
    model.train()
    total_loss = 0
    
    for data in train_loader:
        # For autoencoders, input is also the target
        inputs = data.x.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute reconstruction loss
        loss = F.mse_loss(outputs, inputs)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
    
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def evaluate_autoencoder(model, loader, device, threshold=None):
    """Evaluate the autoencoder model and detect anomalies"""
    model.eval()
    total_loss = 0
    reconstruction_errors = []
    
    with torch.no_grad():
        for data in loader:
            inputs = data.x.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute reconstruction error for each sample
            errors = F.mse_loss(outputs, inputs, reduction='none').mean(dim=1)
            reconstruction_errors.extend(errors.cpu().numpy())
            
            # Compute average loss
            loss = F.mse_loss(outputs, inputs)
            total_loss += loss.item() * inputs.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    reconstruction_errors = np.array(reconstruction_errors)
    
    # Detect anomalies if threshold is provided
    anomalies = None
    if threshold is not None:
        anomalies = reconstruction_errors > threshold
    
    return avg_loss, reconstruction_errors, anomalies

def calculate_anomaly_threshold(errors, method='std', contamination=0.01):
    """Calculate anomaly threshold based on reconstruction errors"""
    if method == 'std':
        # Use mean + n*std as threshold
        threshold = np.mean(errors) + 3 * np.std(errors)
    elif method == 'percentile':
        # Use percentile as threshold
        threshold = np.percentile(errors, 100 * (1 - contamination))
    else:
        raise ValueError(f"Unsupported threshold method: {method}")
    
    return threshold

class ImbalanceHandler:
    """Robust imbalance handling for anomaly detection"""
    
    def __init__(self):
        self.sampling_strategy = 'auto'
        self.random_state = 42
        
    def apply_smote(self, X, y, sampling_strategy='auto'):
        """Apply SMOTE oversampling"""
        if not IMBLEARN_AVAILABLE:
            print("Warning: imbalanced-learn not available. Skipping SMOTE.")
            return X, y
            
        try:
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
            resampled = smote.fit_resample(X, y)
            X_resampled, y_resampled = resampled[0], resampled[1]
            print(f"SMOTE applied: {X.shape} -> {X_resampled.shape}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def apply_undersampling(self, X, y, sampling_strategy='auto'):
        """Apply random undersampling"""
        if not IMBLEARN_AVAILABLE:
            return X, y
            
        try:
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
            resampled = rus.fit_resample(X, y)
            X_resampled, y_resampled = resampled[0], resampled[1]
            print(f"Undersampling applied: {X.shape} -> {X_resampled.shape}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Undersampling failed: {e}. Using original data.")
            return X, y
    
    def apply_smotetomek(self, X, y, sampling_strategy='auto'):
        """Apply SMOTE + Tomek links cleaning"""
        if not IMBLEARN_AVAILABLE:
            return X, y
            
        try:
            smt = SMOTETomek(sampling_strategy=sampling_strategy, random_state=self.random_state)
            resampled = smt.fit_resample(X, y)
            X_resampled, y_resampled = resampled[0], resampled[1]
            print(f"SMOTETomek applied: {X.shape} -> {X_resampled.shape}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"SMOTETomek failed: {e}. Using original data.")
            return X, y
    
    def calculate_class_weights(self, y):
        """Calculate class weights for imbalanced dataset"""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        # For XGBoost format
        if len(class_weights) == 2:
            # Calculate scale_pos_weight for XGBoost (ratio of negative to positive)
            neg_count = np.sum(y == 0)
            pos_count = np.sum(y == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            class_weights['scale_pos_weight'] = scale_pos_weight
            
        return class_weights
    
    def optimize_threshold(self, y_true, y_scores, metric='f1'):
        """Optimize threshold based on specified metric"""
        if len(np.unique(y_true)) < 2:
            return 0.5, 0.0
            
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        if metric == 'f1':
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_score = f1_scores[best_idx]
        elif metric == 'precision':
            best_idx = np.argmax(precision[:-1])  # Exclude last point
            best_threshold = thresholds[best_idx]
            best_score = precision[best_idx]
        elif metric == 'recall':
            best_idx = np.argmax(recall[:-1])  # Exclude last point
            best_threshold = thresholds[best_idx]
            best_score = recall[best_idx]
        else:
            best_threshold = 0.5
            best_score = 0.0
            
        return best_threshold, best_score
    
    def evaluate_imbalance_metrics(self, y_true, y_scores, threshold=None):
        """Comprehensive evaluation for imbalanced classification"""
        if threshold is None:
            threshold, _ = self.optimize_threshold(y_true, y_scores, metric='f1')
            
        y_pred = (y_scores >= threshold).astype(int)
        
        # Basic metrics
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except:
            roc_auc = 0.0
            
        # F1 score
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics for imbalanced data
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        balanced_accuracy = (specificity + sensitivity) / 2
        
        # Anomaly detection specific metrics
        fraud_detection_rate = sensitivity  # True positive rate for anomaly
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        
        metrics = {
            'threshold': threshold,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy,
            'anomaly_detection_rate': fraud_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'confusion_matrix': cm,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        return metrics

if TORCH_AVAILABLE:
    class ClaimAnomalyAutoencoder(nn.Module):
        """Standard autoencoder for anomaly detection.

        QW4: optionally behaves as a Variational Autoencoder (VAE) when
        ``vae=True`` and as a denoising autoencoder when ``noise_factor>0``.
        For VAE we add a KL term to the reconstruction loss; for denoising
        we corrupt the input with Gaussian noise and reconstruct the *clean*
        target. Both can be combined.
        """
        def __init__(self, input_dim, encoding_dim=32, hidden_dims=None,
                     dropout=0.2, vae=False, noise_factor=0.0):
            super(ClaimAnomalyAutoencoder, self).__init__()

            if hidden_dims is None:
                hidden_dims = [64, 48]

            # QW4: VAE mode requires a Gaussian sampling head at the
            # bottleneck. The ``mu`` and ``logvar`` projections share the
            # encoder trunk; the decoder reads from a sampled ``z``.
            self.vae = bool(vae)
            self.noise_factor = float(noise_factor)

            # Encoder layers
            encoder_layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            self.encoder = nn.Sequential(*encoder_layers)

            if self.vae:
                self.fc_mu = nn.Linear(prev_dim, encoding_dim)
                self.fc_logvar = nn.Linear(prev_dim, encoding_dim)
                prev_dim = encoding_dim

            # Decoder layers
            decoder_layers = []
            for hidden_dim in reversed(hidden_dims):
                decoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            decoder_layers.append(nn.Linear(prev_dim, input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def _reparameterize(self, mu, logvar):
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            return mu  # deterministic at inference

        def forward(self, x):
            mu = logvar = None
            h = self.encoder(x)
            if self.vae:
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                z = self._reparameterize(mu, logvar)
            else:
                z = h
            decoded = self.decoder(z)
            if self.vae:
                return decoded, mu, logvar
            return decoded

        def encode(self, x):
            h = self.encoder(x)
            if self.vae:
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                return mu, logvar
            return h

        @staticmethod
        def vae_loss(recon, target, mu, logvar):
            """Combined reconstruction (MSE) + KL divergence."""
            recon_loss = F.mse_loss(recon, target, reduction='mean')
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + 1e-3 * kl, recon_loss
else:
    class ClaimAnomalyAutoencoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for the autoencoder model. Install the project requirements for the full ML stack.")

class ClaimAnomalyXGBoostModel:
    """XGBoost model for anomaly detection"""
    
    def __init__(self, model_type='xgboost', **params):
        self.model_type = model_type
        self.model = None
        self.params = params
        
        # Default parameters
        default_params: dict = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        if model_type == 'xgboost' and XGB_AVAILABLE:
            default_params.update({
                'objective': 'binary:logistic',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'tree_method': 'hist',  # Optimized for big data
                'enable_categorical': True  # Handle categorical features
            })
        elif model_type == 'lightgbm' and LGB_AVAILABLE:
            default_params.update({
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'force_row_wise': True  # Memory efficiency
            })
        elif model_type == 'catboost' and CATBOOST_AVAILABLE:
            default_params.update({
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'bagging_temperature': 1.0,
                'random_seed': 42,
                'verbose': False,
                'allow_writing_files': False  # Prevent writing to disk
            })
        elif model_type == 'random_forest':
            default_params.update({
                'n_estimators': 200,
                'max_depth': None,
                'random_state': 42,
                'n_jobs': -1
            })
        elif model_type == 'svm':
            default_params.update({
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            })
        
        # Merge user params with defaults
        self.params = {**default_params, **params}
        
    def fit(self, X, y):
        """Train the model with early stopping for tree-based models"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.params.get('n_estimators', 200),
                max_depth=self.params.get('max_depth', None),
                random_state=self.params.get('random_state', 42),
                n_jobs=self.params.get('n_jobs', -1)
            )
            self.model.fit(X, y)
            return

        if self.model_type == 'svm':
            self.model = SVC(
                C=self.params.get('C', 1.0),
                kernel=self.params.get('kernel', 'rbf'),
                gamma=self.params.get('gamma', 'scale'),
                probability=True,
                random_state=self.params.get('random_state', 42)
            )
            self.model.fit(X, y)
            return

        # Split data for early stopping validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if self.model_type == 'xgboost' and XGB_AVAILABLE:
            self.model = xgb.XGBClassifier(**self.params)
            fit_kwargs = {
                'X': X_train,
                'y': y_train,
                'eval_set': [(X_val, y_val)],
                'verbose': False,
            }
            if 'early_stopping_rounds' in inspect.signature(self.model.fit).parameters:
                fit_kwargs['early_stopping_rounds'] = 10
                fit_kwargs['eval_metric'] = 'logloss'
            self.model.fit(**fit_kwargs)
        elif self.model_type == 'lightgbm' and LGB_AVAILABLE:
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
        elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
            cat_params = self.params.copy()
            cat_params.pop('allow_writing_files', None)
            self.model = cb.CatBoostClassifier(**cat_params)
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False,
                early_stopping_rounds=10,
            )
        else:
            # Fallback to RandomForest if XGBoost/LightGBM/CatBoost not available
            self.model = RandomForestClassifier(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', 6),
                random_state=self.params.get('random_state', 42),
                n_jobs=self.params.get('n_jobs', -1)
            )
            self.model_type = 'random_forest'
            self.model.fit(X, y)
        
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Get predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return np.zeros(0)
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = joblib.load(filepath)

    def optimize_hyperparameters(self, X, y, n_trials=50, timeout=3600, cv_folds=5, 
                                 random_state=42, imbalance_handler=None):
        """Optimize hyperparameters using Optuna for XGBoost/LightGBM/CatBoost
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training labels
        n_trials : int
            Number of Optuna trials
        timeout : int
            Timeout in seconds
        cv_folds : int
            Number of cross-validation folds
        random_state : int
            Random state for reproducibility
        imbalance_handler : ImbalanceHandler
            Optional imbalance handler for class weights
            
        Returns
        -------
        dict
            Best hyperparameters found
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Using default parameters.")
            return self.params
            
        if self.model_type not in ['xgboost', 'lightgbm', 'catboost']:
            logger.warning(f"Hyperparameter optimization not supported for {self.model_type}")
            return self.params
            
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, roc_auc_score
        
        # Set seeds for reproducibility
        _set_global_seeds(random_state)
        
        # Handle class imbalance
        if imbalance_handler is not None:
            class_weights = imbalance_handler.calculate_class_weights(y)
            scale_pos_weight = class_weights.get('scale_pos_weight', 1.0)
        else:
            # Calculate scale_pos_weight manually
            neg_count = np.sum(y == 0)
            pos_count = np.sum(y == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        def objective(trial):
            # Define hyperparameter search space (narrowed for faster tuning)
            if self.model_type == 'xgboost' and XGB_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),  # Narrowed from 50-500
                    'max_depth': trial.suggest_int('max_depth', 4, 8),  # Narrowed from 3-12
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),  # Narrowed from 0.01-0.3
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # Narrowed from 0.6-1.0
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),  # Narrowed from 0.6-1.0
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5, log=True),  # Narrowed from 0.0-1.0
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0, log=True),  # Narrowed from 0.0-2.0
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),  # Narrowed from 1-10
                    'gamma': trial.suggest_float('gamma', 0.0, 0.3),  # Narrowed from 0.0-1.0
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': random_state,
                    'n_jobs': -1,
                    'eval_metric': 'logloss',
                    'objective': 'binary:logistic',
                    'tree_method': 'hist',
                    'enable_categorical': True
                }
            elif self.model_type == 'lightgbm' and LGB_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),  # Narrowed from 50-500
                    'max_depth': trial.suggest_int('max_depth', 4, 8),  # Narrowed from 3-12
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),  # Narrowed from 0.01-0.3
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # Narrowed from 0.6-1.0
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),  # Narrowed from 0.6-1.0
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5, log=True),  # Narrowed from 0.0-1.0
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0, log=True),  # Narrowed from 0.0-2.0
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),  # Narrowed from 5-50
                    'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),  # Narrowed from 0.0-1.0
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': random_state,
                    'n_jobs': -1,
                    'objective': 'binary',
                    'boosting_type': 'gbdt',
                    'force_row_wise': True,
                    'verbose': -1
                }
            elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 300, step=50),  # Narrowed from 50-500
                    'depth': trial.suggest_int('depth', 4, 8),  # Narrowed from 3-12
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),  # Narrowed from 0.01-0.3
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.0, 8.0),  # Narrowed from 1.0-10.0
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 0.5),  # Narrowed from 0.0-1.0
                    'random_seed': random_state,
                    'verbose': False,
                    'allow_writing_files': False,
                    'loss_function': 'Logloss',
                    'eval_metric': 'AUC'
                }
            
            # Cross-validation with optional parallel execution
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            # Helper function for single CV fold
            def evaluate_fold(train_idx, val_idx):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create and train model
                if self.model_type == 'xgboost' and XGB_AVAILABLE:
                    model = xgb.XGBClassifier(**params)
                elif self.model_type == 'lightgbm' and LGB_AVAILABLE:
                    model = lgb.LGBMClassifier(**params)
                elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
                    cat_params = params.copy()
                    cat_params.pop('allow_writing_files', None)
                    model = cb.CatBoostClassifier(**cat_params)
                else:
                    return 0.0
                    
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Use F1 score as primary metric for imbalanced data
                try:
                    # Find optimal threshold
                    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                    best_f1 = np.max(f1_scores)
                    return best_f1
                except:
                    # Fallback to ROC-AUC
                    try:
                        roc_auc = roc_auc_score(y_val, y_pred_proba)
                        return roc_auc
                    except:
                        return 0.0
            
            # Try parallel CV, fallback to sequential if it fails
            cv_n_jobs = self.params.get('cv_n_jobs', 1)  # Default to 1 to avoid resource contention
            try:
                if cv_n_jobs > 1:
                    cv_scores = joblib.Parallel(n_jobs=cv_n_jobs)(
                        joblib.delayed(evaluate_fold)(train_idx, val_idx)
                        for train_idx, val_idx in cv.split(X, y)
                    )
                else:
                    cv_scores = [evaluate_fold(train_idx, val_idx) for train_idx, val_idx in cv.split(X, y)]
            except Exception as e:
                logger.warning(f"Parallel CV failed, falling back to sequential: {e}")
                cv_scores = [evaluate_fold(train_idx, val_idx) for train_idx, val_idx in cv.split(X, y)]
            
            # Return mean CV score
            return np.mean(cv_scores)
        
        # Create Optuna study
        study_name = f"{self.model_type}_anomaly_optimization"
        try:
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Run optimization with parallel trials for faster tuning
            n_jobs = self.params.get('optuna_n_jobs', -1)  # -1 = use all CPUs
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False, n_jobs=n_jobs)
            
            # Get best parameters
            best_params = study.best_params
            logger.info(f"Best {self.model_type} trial: {study.best_trial.number}")
            logger.info(f"Best CV score: {study.best_value:.4f}")
            logger.info(f"Best hyperparameters: {best_params}")
            
            # Update model parameters
            self.params.update(best_params)
            
            return best_params
            
        except Exception as e:
            logger.warning(f"Optuna optimization failed for {self.model_type}: {e}. Using default parameters.")
            return self.params

class StackingEnsemble:
    """Stacking ensemble with meta-learner for better model combination
    
    Uses cross-validation to generate meta-features from base models,
    then trains a meta-learner to optimally combine predictions.
    """
    
    def __init__(self, base_models=None, meta_learner_type='logistic', cv_folds=5, 
                 use_proba=True, random_state=42):
        """
        Parameters
        ----------
        base_models : dict
            Dictionary of base models {name: model_instance}
        meta_learner_type : str
            Type of meta-learner: 'logistic', 'random_forest', 'xgboost', 'lightgbm'
        cv_folds : int
            Number of cross-validation folds for generating meta-features
        use_proba : bool
            Whether to use probability predictions (True) or class predictions (False)
        random_state : int
            Random state for reproducibility
        """
        self.base_models = base_models or {}
        self.meta_learner_type = meta_learner_type
        self.cv_folds = cv_folds
        self.use_proba = use_proba
        self.random_state = random_state
        self.meta_learner = None
        self.meta_features_train = None
        self.fitted_base_models = {}
        
    def add_base_model(self, name, model):
        """Add a base model to the ensemble"""
        self.base_models[name] = model
        
    def _generate_meta_features(self, X, y=None, fit=True):
        """Generate meta-features using cross-validation
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like, optional
            Training labels (required for fitting)
        fit : bool
            Whether to fit base models (True) or use existing fitted models (False)
            
        Returns
        -------
        array-like
            Meta-features (predictions from base models)
        """
        from sklearn.model_selection import StratifiedKFold
        
        if not self.base_models:
            raise ValueError("No base models available")
            
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Initialize meta-features array
        if self.use_proba:
            meta_features = np.zeros((n_samples, n_models))
        else:
            meta_features = np.zeros((n_samples, n_models), dtype=int)
        
        # Use stratified K-Fold for generating meta-features
        if y is not None:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                               random_state=self.random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.cv_folds, shuffle=True, 
                      random_state=self.random_state)
        
        # Generate meta-features for each base model
        for model_name, model in self.base_models.items():
            model_idx = list(self.base_models.keys()).index(model_name)
            
            if fit:
                # Fit model on full training data
                try:
                    if hasattr(model, 'fit'):
                        model.fit(X, y)
                        self.fitted_base_models[model_name] = model
                except Exception as e:
                    logger.warning(f"Failed to fit base model {model_name}: {e}")
                    continue
            
            # Generate predictions using CV to avoid overfitting
            cv_predictions = np.zeros(n_samples)
            
            for train_idx, val_idx in cv.split(X, y if y is not None else X):
                X_train, X_val = X[train_idx], X[val_idx]
                
                if fit:
                    # Fit on training fold
                    try:
                        model_fold = self._clone_model(model)
                        if y is not None:
                            model_fold.fit(X_train, y[train_idx])
                        else:
                            # For unsupervised models
                            if hasattr(model_fold, 'fit'):
                                model_fold.fit(X_train)
                    except Exception as e:
                        logger.warning(f"Failed to fit fold for {model_name}: {e}")
                        cv_predictions[val_idx] = 0
                        continue
                else:
                    model_fold = self.fitted_base_models.get(model_name, model)
                
                # Predict on validation fold
                try:
                    if self.use_proba:
                        if hasattr(model_fold, 'predict_proba'):
                            preds = model_fold.predict_proba(X_val)
                            if preds.shape[1] > 1:
                                cv_predictions[val_idx] = preds[:, 1]
                            else:
                                cv_predictions[val_idx] = preds[:, 0]
                        elif hasattr(model_fold, 'decision_function'):
                            cv_predictions[val_idx] = model_fold.decision_function(X_val)
                            # Normalize to [0, 1]
                            cv_predictions[val_idx] = 1 / (1 + np.exp(-cv_predictions[val_idx]))
                        else:
                            # Fallback to class predictions
                            cv_predictions[val_idx] = model_fold.predict(X_val)
                    else:
                        cv_predictions[val_idx] = model_fold.predict(X_val)
                except Exception as e:
                    logger.warning(f"Failed to predict for {model_name}: {e}")
                    cv_predictions[val_idx] = 0
            
            meta_features[:, model_idx] = cv_predictions
        
        return meta_features
    
    def _clone_model(self, model):
        """Clone a model for CV training"""
        try:
            import sklearn.base
            return sklearn.base.clone(model)
        except:
            # If sklearn clone fails, try to create a new instance
            try:
                return type(model)(**model.get_params())
            except:
                return model
    
    def fit(self, X, y):
        """Fit the stacking ensemble
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training labels
        """
        # Set seeds for reproducibility
        _set_global_seeds(self.random_state)
        
        # Generate meta-features using CV
        logger.info("Generating meta-features using cross-validation...")
        self.meta_features_train = self._generate_meta_features(X, y, fit=True)
        
        # Train meta-learner
        logger.info(f"Training meta-learner ({self.meta_learner_type})...")
        self.meta_learner = self._create_meta_learner()
        
        try:
            self.meta_learner.fit(self.meta_features_train, y)
            logger.info("Meta-learner training completed successfully")
        except Exception as e:
            logger.warning(f"Meta-learner training failed: {e}. Using fallback simple averaging.")
            self.meta_learner = None
    
    def _create_meta_learner(self):
        """Create meta-learner based on specified type"""
        if self.meta_learner_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.meta_learner_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.meta_learner_type == 'xgboost' and XGB_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss',
                objective='binary:logistic'
            )
        elif self.meta_learner_type == 'lightgbm' and LGB_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_state,
                objective='binary',
                verbose=-1
            )
        else:
            # Fallback to logistic regression
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
    
    def predict_proba(self, X):
        """Predict probability using stacking ensemble
        
        Parameters
        ----------
        X : array-like
            Features
            
        Returns
        -------
        array-like
            Predicted probabilities
        """
        if not self.fitted_base_models:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Generate meta-features for new data
        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models)
        
        if self.use_proba:
            meta_features = np.zeros((n_samples, n_models))
        else:
            meta_features = np.zeros((n_samples, n_models), dtype=int)
        
        # Get predictions from each fitted base model
        for model_name, model in self.fitted_base_models.items():
            model_idx = list(self.fitted_base_models.keys()).index(model_name)
            
            try:
                if self.use_proba:
                    if hasattr(model, 'predict_proba'):
                        preds = model.predict_proba(X)
                        if preds.shape[1] > 1:
                            meta_features[:, model_idx] = preds[:, 1]
                        else:
                            meta_features[:, model_idx] = preds[:, 0]
                    elif hasattr(model, 'decision_function'):
                        preds = model.decision_function(X)
                        meta_features[:, model_idx] = 1 / (1 + np.exp(-preds))
                    else:
                        meta_features[:, model_idx] = model.predict(X)
                else:
                    meta_features[:, model_idx] = model.predict(X)
            except Exception as e:
                logger.warning(f"Failed to predict with {model_name}: {e}")
                meta_features[:, model_idx] = 0
        
        # Use meta-learner if available, otherwise use simple averaging
        if self.meta_learner is not None:
            try:
                return self.meta_learner.predict_proba(meta_features)[:, 1]
            except Exception as e:
                logger.warning(f"Meta-learner prediction failed: {e}. Using simple averaging.")
        
        # Fallback to simple averaging
        return np.mean(meta_features, axis=1)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels
        
        Parameters
        ----------
        X : array-like
            Features
        threshold : float
            Classification threshold
            
        Returns
        -------
        array-like
            Predicted class labels
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def get_feature_importance(self):
        """Get meta-learner feature importance (importance of each base model)"""
        if self.meta_learner is None:
            return None
        
        try:
            if hasattr(self.meta_learner, 'feature_importances_'):
                return dict(zip(self.base_models.keys(), self.meta_learner.feature_importances_))
            elif hasattr(self.meta_learner, 'coef_'):
                # For logistic regression, use absolute coefficients
                importance = np.abs(self.meta_learner.coef_[0])
                return dict(zip(self.base_models.keys(), importance))
            else:
                return None
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return None

class DynamicWeightOptimizer:
    """Optimize algorithm weights based on data characteristics and performance"""
    
    def __init__(self):
        self.weight_history = []
        self.performance_history = {}
        
    def analyze_data_characteristics(self, features, edge_index=None):
        """Analyze data to determine optimal algorithm weights"""
        n_samples, n_features = features.shape
        characteristics = {}
        
        # Data size analysis
        characteristics['data_size'] = n_samples
        characteristics['feature_count'] = n_features
        characteristics['data_density'] = n_samples / (n_features * 100)  # Normalized density
        
        # Feature variance analysis
        feature_variances = np.var(features, axis=0)
        characteristics['avg_variance'] = np.mean(feature_variances)
        characteristics['variance_std'] = np.std(feature_variances)
        
        # Correlation analysis
        if n_samples > 1:
            corr_matrix = np.corrcoef(features.T)
            # Remove diagonal and take absolute values
            corr_values = np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
            characteristics['avg_correlation'] = np.mean(corr_values) if len(corr_values) > 0 else 0
            characteristics['max_correlation'] = np.max(corr_values) if len(corr_values) > 0 else 0
        
        # Graph analysis (if available)
        if edge_index is not None:
            try:
                # Handle edge_index that might be a tensor or numpy array
                if hasattr(edge_index, 'shape'):
                    if len(edge_index.shape) >= 2:
                        n_edges = edge_index.shape[1] // 2  # Undirected graph
                    else:
                        n_edges = len(edge_index) // 2
                else:
                    # Fallback for other types
                    n_edges = len(edge_index) // 2 if hasattr(edge_index, '__len__') else 0
                
                characteristics['edge_density'] = n_edges / (n_samples * (n_samples - 1) / 2) if n_samples > 1 else 0
                characteristics['has_graph'] = True
            except Exception as e:
                # If there's any error analyzing the graph, treat as no graph
                characteristics['edge_density'] = 0
                characteristics['has_graph'] = False
        else:
            characteristics['edge_density'] = 0
            characteristics['has_graph'] = False
        
        return characteristics
    
    def calculate_optimal_weights(self, characteristics):
        """Calculate optimal weights based on data characteristics.
        
        Includes 4 algorithms: isolation_forest, autoencoder, xgboost, gnn.
        GNN is only weighted positively when a graph is actually present.
        """
        weights = {'isolation': 0.25, 'autoencoder': 0.25, 'xgboost': 0.30, 'gnn': 0.20}  # Default
        
        # If no graph data is available, redistribute GNN weight to other models
        if not characteristics.get('has_graph', False):
            weights['gnn'] = 0.0
            # Redistribute evenly to the remaining models
            for k in ('isolation', 'autoencoder', 'xgboost'):
                weights[k] = weights[k] / 0.80  # Renormalize (0.25+0.25+0.30 = 0.80)
        
        # Data size adjustments
        if characteristics['data_size'] < 500:
            # Small dataset: favor Isolation Forest (less overfitting)
            weights['isolation'] = 0.40
            weights['autoencoder'] = 0.25
            weights['xgboost'] = 0.20
            weights['gnn'] = 0.15 if characteristics.get('has_graph', False) else 0.0
        elif characteristics['data_size'] > 5000:
            # Large dataset: favor XGBoost (can handle complex patterns)
            weights['isolation'] = 0.15
            weights['autoencoder'] = 0.25
            weights['xgboost'] = 0.40
            weights['gnn'] = 0.20 if characteristics.get('has_graph', False) else 0.0
        
        # Graph-aware adjustments: when graph has rich structure, give GNN more weight
        if characteristics.get('has_graph', False):
            edge_density = characteristics.get('edge_density', 0)
            if edge_density > 0.01:
                # Rich graph structure: GNN can leverage neighborhood information
                weights['gnn'] = min(weights['gnn'] + 0.10, 0.40)
                weights['xgboost'] = max(weights['xgboost'] - 0.05, 0.15)
                weights['isolation'] = max(weights['isolation'] - 0.025, 0.10)
                weights['autoencoder'] = max(weights['autoencoder'] - 0.025, 0.10)
        
        # Feature variance adjustments
        if characteristics['avg_variance'] < 0.1:
            # Low variance: favor Isolation Forest (better for subtle anomalies)
            weights['isolation'] = min(weights['isolation'] + 0.1, 0.5)
            weights['autoencoder'] = max(weights['autoencoder'] - 0.05, 0.1)
            weights['xgboost'] = max(weights['xgboost'] - 0.05, 0.1)
            if characteristics.get('has_graph', False):
                weights['gnn'] = max(weights['gnn'] - 0.025, 0.05)
        elif characteristics['avg_variance'] > 1.0:
            # High variance: favor XGBoost (better for complex patterns)
            weights['xgboost'] = min(weights['xgboost'] + 0.1, 0.5)
            weights['isolation'] = max(weights['isolation'] - 0.05, 0.1)
            weights['autoencoder'] = max(weights['autoencoder'] - 0.05, 0.1)
            if characteristics.get('has_graph', False):
                weights['gnn'] = min(weights['gnn'] + 0.025, 0.30)
        
        # Correlation adjustments
        if characteristics['avg_correlation'] > 0.7:
            # High correlation: favor XGBoost (can capture complex relationships)
            weights['xgboost'] = min(weights['xgboost'] + 0.1, 0.5)
            weights['isolation'] = max(weights['isolation'] - 0.05, 0.1)
            weights['autoencoder'] = max(weights['autoencoder'] - 0.05, 0.1)
            if characteristics.get('has_graph', False):
                weights['gnn'] = min(weights['gnn'] + 0.025, 0.30)
        
        # Feature density adjustments
        if characteristics['data_density'] < 0.1:
            # Low density: reduce XGBoost weight
            xgb_weight = weights['xgboost']
            weights['xgboost'] = max(0.1, weights['xgboost'] - 0.2)
            # Redistribute to other algorithms
            redistribution = (xgb_weight - weights['xgboost']) / 3
            weights['isolation'] = min(weights['isolation'] + redistribution, 0.6)
            weights['autoencoder'] = min(weights['autoencoder'] + redistribution, 0.6)
            if characteristics.get('has_graph', False):
                weights['gnn'] = min(weights['gnn'] + redistribution, 0.4)
        elif characteristics['data_density'] > 0.5:
            # High density: favor XGBoost
            weights['xgboost'] = min(weights['xgboost'] + 0.1, 0.5)
            weights['isolation'] = max(weights['isolation'] - 0.05, 0.1)
            weights['autoencoder'] = max(weights['autoencoder'] - 0.05, 0.1)
            if characteristics.get('has_graph', False):
                weights['gnn'] = min(weights['gnn'] + 0.025, 0.30)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight <= 0:
            # Safety net
            return {'isolation': 0.25, 'autoencoder': 0.25, 'xgboost': 0.30, 'gnn': 0.20}
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def update_weights_based_on_performance(self, current_weights, performance_metrics):
        """Update weights based on actual performance feedback"""
        # Store performance history
        self.performance_history.update(performance_metrics)
        
        # Simple performance-based adjustment
        if 'isolation_f1' in performance_metrics and 'autoencoder_f1' in performance_metrics:
            iso_perf = performance_metrics['isolation_f1']
            ae_perf = performance_metrics['autoencoder_f1']
            gnn_perf = performance_metrics.get('gnn_f1', performance_metrics.get('xgboost_f1', 0.5))
            xgb_perf = performance_metrics.get('xgboost_f1', gnn_perf)
            
            # Calculate performance-based adjustments (4-way)
            total_perf = iso_perf + ae_perf + xgb_perf + gnn_perf
            if total_perf > 0:
                adjusted_weights = {
                    'isolation': 0.7 * current_weights['isolation'] + 0.3 * (iso_perf / total_perf),
                    'autoencoder': 0.7 * current_weights['autoencoder'] + 0.3 * (ae_perf / total_perf),
                    'xgboost': 0.7 * current_weights['xgboost'] + 0.3 * (xgb_perf / total_perf),
                    'gnn': 0.7 * current_weights.get('gnn', 0.0) + 0.3 * (gnn_perf / total_perf)
                }
                
                # Normalize
                total_weight = sum(adjusted_weights.values())
                if total_weight > 0:
                    adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
                return adjusted_weights
        
        return current_weights

    def optimize_weights_with_optuna(self, individual_scores, y_true=None, active_algorithms=None,
                                     n_trials=30, timeout=120, lambda_fpr=0.5, cv_folds=5):
        """Optimize ensemble weights dynamically using Optuna to minimize False Positive Rate.
        
        Args:
            individual_scores: Dictionary of algorithm_name -> np.ndarray of continuous scores/probabilities.
            y_true: True binary labels (0/1) or consensus pseudo-labels.
            active_algorithms: List of algorithms to include in ensemble.
            n_trials: Number of Optuna optimization trials.
            timeout: Optimization timeout in seconds.
            lambda_fpr: Weight penalty for False Positive Rate in objective function.
            cv_folds: Number of Stratified K-Fold CV folds.
            
        Returns:
            Dictionary with optimal weights and comparative performance metrics.
        """
        optimizer = OptunaEnsembleOptimizer(
            n_trials=n_trials,
            timeout=timeout,
            lambda_fpr=lambda_fpr,
            cv_folds=cv_folds
        )
        return optimizer.optimize(
            individual_scores=individual_scores,
            y_true=y_true,
            active_algorithms=active_algorithms
        )


class OptunaEnsembleOptimizer:
    """Dynamic Ensemble Weight Optimizer powered by Optuna.
    
    Optimizes ensemble weights across multiple anomaly detection models (XGBoost, 
    Isolation Forest, Autoencoder, GNN) specifically designed to minimize False Positive 
    Rate (FPR) while preserving high recall using Stratified K-Fold Cross-Validation.
    """
    def __init__(self, n_trials=30, timeout=120, lambda_fpr=0.5, cv_folds=5, beta=0.5, random_state=42):
        self.n_trials = n_trials
        self.timeout = timeout
        self.lambda_fpr = lambda_fpr
        self.cv_folds = cv_folds
        self.beta = beta  # Beta < 1.0 (e.g. 0.5) puts higher emphasis on Precision (fewer false positives)
        self.random_state = random_state
        self.best_weights = None
        self.study_summary = {}

    def _normalize_scores(self, scores_dict):
        """Scale all model score outputs safely to [0, 1] range."""
        norm_scores = {}
        for algo, scores in scores_dict.items():
            if scores is None or len(scores) == 0:
                continue
            arr = np.asarray(scores, dtype=float)
            s_min, s_max = np.min(arr), np.max(arr)
            denom = s_max - s_min
            if denom > 1e-8:
                norm_scores[algo] = (arr - s_min) / denom
            else:
                norm_scores[algo] = np.zeros_like(arr)
        return norm_scores

    def _compute_metrics_at_threshold(self, y_true, combined_scores, threshold=0.5):
        """Compute comprehensive QA metrics: FPR, Precision, Recall, F1, F_beta."""
        y_pred = (combined_scores >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        beta_sq = self.beta ** 2
        f_beta_denom = (beta_sq * precision + recall)
        f_beta = (1 + beta_sq) * (precision * recall) / f_beta_denom if f_beta_denom > 0 else 0.0
        
        f1_denom = precision + recall
        f1 = 2 * (precision * recall) / f1_denom if f1_denom > 0 else 0.0
        
        return {
            'fpr': float(fpr),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'f_beta': float(f_beta),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }

    def optimize(self, individual_scores, y_true=None, active_algorithms=None):
        """Run Optuna study to find optimal ensemble weights."""
        norm_scores = self._normalize_scores(individual_scores)
        
        if not norm_scores:
            logger.warning("No valid individual scores provided for Optuna ensemble tuning.")
            return {'weights': {}, 'status': 'no_data'}
        
        algos = active_algorithms if active_algorithms else list(norm_scores.keys())
        algos = [a for a in algos if a in norm_scores]
        
        if len(algos) == 0:
            return {'weights': {}, 'status': 'no_matching_algos'}
        
        if len(algos) == 1:
            return {
                'weights': {algos[0]: 1.0},
                'status': 'single_algo',
                'best_score': 1.0,
                'metric_comparison': {}
            }
            
        n_samples = len(next(iter(norm_scores.values())))
        
        # If no ground truth labels, generate consensus pseudo labels (top 5% anomalies)
        if y_true is None:
            mean_score = np.mean([norm_scores[a] for a in algos], axis=0)
            threshold = np.percentile(mean_score, 95)
            y_eval = (mean_score >= threshold).astype(int)
        else:
            y_eval = np.asarray(y_true, dtype=int)
            
        # Baseline / Default equal weights evaluation
        equal_weight = 1.0 / len(algos)
        default_combined = sum(equal_weight * norm_scores[a] for a in algos)
        default_metrics = self._compute_metrics_at_threshold(y_eval, default_combined)

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna is not installed. Using default normalized equal weights.")
            return {
                'weights': {a: equal_weight for a in algos},
                'status': 'optuna_not_available',
                'metric_comparison': {
                    'default': default_metrics,
                    'optimized': default_metrics
                }
            }

        # Setup Stratified K-Fold Cross Validation
        from sklearn.model_selection import StratifiedKFold
        
        unique_classes, counts = np.unique(y_eval, return_counts=True)
        use_cv = len(unique_classes) > 1 and np.min(counts) >= self.cv_folds
        
        if use_cv:
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            splits = list(skf.split(np.zeros(n_samples), y_eval))
        else:
            splits = [(np.arange(n_samples), np.arange(n_samples))]

        # Optuna Objective Function
        def objective(trial):
            # Sample simplex weights
            raw_w = {}
            for a in algos:
                raw_w[a] = trial.suggest_float(f'weight_{a}', 0.01, 1.0)
            
            total_w = sum(raw_w.values())
            weights = {a: raw_w[a] / total_w for a in algos}
            
            cv_scores = []
            for train_idx, val_idx in splits:
                val_combined = sum(weights[a] * norm_scores[a][val_idx] for a in algos)
                metrics = self._compute_metrics_at_threshold(y_eval[val_idx], val_combined)
                
                # Custom objective: Maximize F_beta while penalizing False Positive Rate (FPR)
                # Score = F_beta - (lambda * FPR)
                fold_obj = metrics['f_beta'] - (self.lambda_fpr * metrics['fpr'])
                cv_scores.append(fold_obj)
                
            return float(np.mean(cv_scores))

        # Run Optuna Study with suppress logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False
        )
        
        # Extract best weights
        best_raw = {a: study.best_params.get(f'weight_{a}', 1.0) for a in algos}
        best_sum = sum(best_raw.values())
        optimal_weights = {a: float(best_raw[a] / best_sum) for a in algos}
        
        # Evaluate optimized ensemble
        opt_combined = sum(optimal_weights[a] * norm_scores[a] for a in algos)
        optimized_metrics = self._compute_metrics_at_threshold(y_eval, opt_combined)
        
        logger.info(
            "Optuna Ensemble Optimization Completed: Best Score=%.4f, FPR Reduction: %.2f%% -> %.2f%%",
            study.best_value, default_metrics['fpr'] * 100, optimized_metrics['fpr'] * 100
        )
        
        self.best_weights = optimal_weights
        return {
            'weights': optimal_weights,
            'status': 'success',
            'best_objective_value': float(study.best_value),
            'n_trials_completed': len(study.trials),
            'metric_comparison': {
                'default': default_metrics,
                'optimized': optimized_metrics,
                'fpr_reduction_pct': float(
                    ((default_metrics['fpr'] - optimized_metrics['fpr']) / default_metrics['fpr'] * 100)
                    if default_metrics['fpr'] > 0 else 0.0
                )
            }
        }


class CombinedAnomalyDetector:
    """Enhanced anomaly detection with dynamic weight optimization and imbalance handling"""
    def __init__(self, isolation_forest_params=None, autoencoder_params=None, dbscan_params=None, xgboost_params=None,
                 gnn_params=None, algorithms=None, use_dynamic_weights=True, imbalance_config=None,
                 random_state=42, verbose=False):
        self.random_state = random_state
        self.verbose = verbose

        # Isolation Forest for initial anomaly detection
        # Optimized: Reduced n_estimators from 100 to 50 for faster training
        iso_n_estimators = 50 if isolation_forest_params is None else isolation_forest_params.get('n_estimators', 50)
        # Ensure n_estimators is at least 1 to avoid errors
        iso_n_estimators = max(1, iso_n_estimators) if iso_n_estimators > 0 else 50
        self.isolation_forest = IsolationForest(
            contamination=0.05 if isolation_forest_params is None else isolation_forest_params.get('contamination', 0.05),
            random_state=42,
            n_estimators=iso_n_estimators
        )

        # Autoencoder parameters
        # Optimized: Disabled VAE by default, reduced epochs
        self.autoencoder_params = autoencoder_params or {}
        self.autoencoder = None
        self.autoencoder_threshold = None

        # XGBoost parameters
        self.xgboost_params = xgboost_params or {}
        self.xgboost_model = None

        # DBSCAN parameters
        self.dbscan_params = dbscan_params or {}
        self.dbscan = None

        # Dynamic weight optimization
        self.use_dynamic_weights = use_dynamic_weights
        self.weight_optimizer = DynamicWeightOptimizer() if use_dynamic_weights else None
        self.weight_optimization_results = None

        # Imbalance handling
        self.imbalance_config = imbalance_config or {}
        self.imbalance_handler = ImbalanceHandler()

        # Default weights (optimized for faster training)
        self.isolation_weight = 0.30  # Increased from 0.25 (faster, no GPU needed)
        self.autoencoder_weight = 0.20  # Reduced from 0.25 (slower)
        self.xgboost_weight = 0.40  # Increased from 0.30 (fastest supervised)
        self.gnn_weight = 0.10  # Reduced from 0.20 (slowest)
        self.dbscan_weight = 0.0

        # GNN model
        self.gnn_model = None
        self.gnn_params = gnn_params or {}

        # Thresholds for each algorithm
        self.isolation_threshold = 0.5
        self.autoencoder_threshold = None
        self.xgboost_threshold = 0.5

        # Active algorithms (optimized order: fastest first)
        # GNN included by default for visualization, but with lower weight for performance
        self.algorithms = algorithms or ['isolation_forest', 'xgboost', 'autoencoder', 'gnn']
        
        # Preprocessing: Imputer and Scaler for data preprocessing
        self.imputer = SimpleImputer(strategy='median')  # Impute missing values with median
        self.scaler = StandardScaler()
        self.training_metadata = {}

        # QW3: rank-normalize per-algorithm scores before weighted sum
        # (default True; can be disabled by setting ``rank_ensemble=False``).
        self.rank_ensemble = True
    
        # Stacking ensemble configuration
        self.use_stacking = False  # Disabled by default
        self.stacking_ensemble = None
        self.stacking_params = {
            'meta_learner_type': 'logistic',
            'cv_folds': 5,
            'use_proba': True,
            'random_state': 42
        }
    
    def fit(self, features, edge_index=None, edge_type=None, labels=None, device='cpu', 
            optimize_hyperparams=False, optuna_n_trials=15, optuna_timeout=600, 
            optimize_ensemble_weights=False, lambda_fpr=0.5):
        """Train the combined anomaly detection model with dynamic weight optimization and imbalance handling"""
        # Set seeds for reproducibility
        _set_global_seeds(42)

        # GPU detection and optimization
        if TORCH_AVAILABLE:
            if device == 'cpu':
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    logger.info(f"Using GPU for training")
                    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
                else:
                    device = torch.device('cpu')
                    logger.info("No GPU detected, using CPU for training")
            else:
                device = torch.device(device)
                logger.info(f"Using specified device: {device}")
        else:
            device = 'cpu'
            logger.info("PyTorch not installed; proceeding with non-deep-learning algorithms only.")

        # Impute missing values first, then scale features
        features_imputed = self.imputer.fit_transform(features)
        features_scaled = self.scaler.fit_transform(features_imputed)
        
        # Dynamic weight optimization based on data characteristics
        if self.use_dynamic_weights and self.weight_optimizer:
            characteristics = self.weight_optimizer.analyze_data_characteristics(features, edge_index)
            optimal_weights = self.weight_optimizer.calculate_optimal_weights(characteristics)
            
            # Update weights (only for active algorithms)
            self.isolation_weight = optimal_weights['isolation']
            self.autoencoder_weight = optimal_weights['autoencoder']
            self.xgboost_weight = optimal_weights['xgboost']
            self.gnn_weight = optimal_weights.get('gnn', 0.0) if 'gnn' in self.algorithms else 0.0

            logger.info(
                "Dynamic weights calculated: Isolation=%.3f, Autoencoder=%.3f, "
                "XGBoost=%.3f, GNN=%.3f",
                self.isolation_weight, self.autoencoder_weight,
                self.xgboost_weight, self.gnn_weight,
            )
        
        # Train Isolation Forest
        if 'isolation_forest' in self.algorithms:
            self.isolation_forest.fit(features_scaled)
        
        # Train Autoencoder
        if 'autoencoder' in self.algorithms and TORCH_AVAILABLE:
            input_dim = features_scaled.shape[1]
            encoding_dim = self.autoencoder_params.get('encoding_dim', 32)
            # QW4: VAE and denoising toggles
            ae_vae = bool(self.autoencoder_params.get('vae', False))
            ae_noise = float(self.autoencoder_params.get('noise_factor', 0.0))

            self.autoencoder = ClaimAnomalyAutoencoder(
                input_dim=input_dim,
                encoding_dim=encoding_dim,
                hidden_dims=self.autoencoder_params.get('hidden_dims', [64, 48]),
                dropout=self.autoencoder_params.get('dropout', 0.2),
                vae=ae_vae,
                noise_factor=ae_noise,
            ).to(device)

            # Train autoencoder using DataLoader for memory efficiency
            from torch.utils.data import DataLoader, TensorDataset

            self.autoencoder.train()

            # Optimized training parameters for CPU/GPU with dynamic batch size
            device_type = device.type if hasattr(device, 'type') else 'cpu'
            
            # Use dynamic batch size calculation
            default_batch = 4096 if device_type == 'cpu' else 1024
            batch_size = get_optimal_batch_size(
                device, 
                len(features_scaled), 
                features_scaled.shape[1], 
                default_batch=default_batch
            )
            
            # Allow user override if specified
            user_batch_size = self.autoencoder_params.get('batch_size')
            if user_batch_size and user_batch_size > 0:
                batch_size = min(user_batch_size, batch_size)
            
            if device_type == 'cpu':
                epochs = self.autoencoder_params.get('epochs', 50)  # Reduced epochs for CPU
                # Ensure epochs is at least 1, if 0 use default
                epochs = max(1, epochs) if epochs > 0 else 50
                patience = self.autoencoder_params.get('early_stopping_patience', 8)  # More aggressive early stopping
                patience = max(1, patience) if patience > 0 else 8
            else:
                epochs = self.autoencoder_params.get('epochs', 100)
                # Ensure epochs is at least 1, if 0 use default
                epochs = max(1, epochs) if epochs > 0 else 100
                patience = self.autoencoder_params.get('early_stopping_patience', 10)
                patience = max(1, patience) if patience > 0 else 10

            min_delta = self.autoencoder_params.get('early_stopping_min_delta', 1e-4)
            progress_callback = self.autoencoder_params.get('progress_callback', None)

            # Disable VAE mode by default for faster training (can be enabled via params)
            ae_vae = bool(self.autoencoder_params.get('vae', False))
            if ae_vae:
                logger.warning("VAE mode enabled - this adds ~30% training overhead")

            dataset = TensorDataset(torch.FloatTensor(features_scaled))
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
            
            # Add learning rate scheduler for faster convergence
            try:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    autoencoder_optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=3
                )
            except TypeError as e:
                logger.warning(f"ReduceLROnPlateau initialization failed: {e}. Using scheduler=None")
                scheduler = None
            
            # Initialize mixed precision training for GPU
            use_amp = device.type == 'cuda'
            scaler = None
            if use_amp:
                try:
                    from torch.cuda.amp import GradScaler, autocast
                    scaler = GradScaler()
                    logger.info("Mixed precision training enabled for GPU")
                except ImportError:
                    logger.warning("AMP not available, falling back to FP32")
                    use_amp = False

            best_loss = float('inf')
            epochs_no_improve = 0
            best_model_state = None

            for epoch in range(epochs):
                total_loss = 0
                for batch in train_loader:
                    clean = batch[0].to(device)
                    # QW4 denoising: corrupt input; target is always the
                    # clean signal so the AE learns to denoise.
                    if ae_noise > 0:
                        noisy = clean + ae_noise * torch.randn_like(clean)
                        inp = noisy
                    else:
                        inp = clean
                    
                    if use_amp and scaler is not None:
                        # Mixed precision training
                        with autocast():
                            out = self.autoencoder(inp)
                            if self.autoencoder.vae:
                                # QW4 VAE: combined recon + KL loss
                                recon, mu, logvar = out
                                loss, recon_part = ClaimAnomalyAutoencoder.vae_loss(
                                    recon, clean, mu, logvar,
                                )
                            else:
                                recon = out
                                loss = F.mse_loss(recon, clean)
                        
                        autoencoder_optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(autoencoder_optimizer)
                        scaler.update()
                    else:
                        # Standard FP32 training
                        autoencoder_optimizer.zero_grad()
                        out = self.autoencoder(inp)
                        if self.autoencoder.vae:
                            # QW4 VAE: combined recon + KL loss
                            recon, mu, logvar = out
                            loss, recon_part = ClaimAnomalyAutoencoder.vae_loss(
                                recon, clean, mu, logvar,
                            )
                        else:
                            recon = out
                            loss = F.mse_loss(recon, clean)
                        loss.backward()
                        autoencoder_optimizer.step()
                    
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)

                # Update learning rate scheduler
                if scheduler is not None:
                    scheduler.step(avg_loss)

                # Early stopping check
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                    # Save best model state
                    best_model_state = self.autoencoder.state_dict().copy()
                else:
                    epochs_no_improve += 1

                # Update progress callback if provided
                if progress_callback:
                    progress_callback(epoch, epochs, avg_loss)

                if epoch % 20 == 0:
                    logger.info("Autoencoder epoch %d, avg loss: %.4f", epoch, avg_loss)

                # Auto-save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    try:
                        import os
                        os.makedirs("models/checkpoints", exist_ok=True)
                        torch.save(self.autoencoder.state_dict(), f"models/checkpoints/autoencoder_checkpoint_latest.pt")
                    except Exception as e:
                        logger.warning(f"Failed to save autoencoder checkpoint: {e}")

                # Stop early if no improvement
                if epochs_no_improve >= patience:
                    logger.info("Early stopping at epoch %d! Best loss: %.4f",
                                epoch, best_loss)
                    if best_model_state is not None:
                        self.autoencoder.load_state_dict(best_model_state)
                    break

            if best_model_state is not None:
                self.autoencoder.load_state_dict(best_model_state)
                # Save best model to disk for checkpointing
                try:
                    import os
                    model_dir = os.path.dirname(self.autoencoder_params.get('model_prefix', 'models/fraud_detector'))
                    os.makedirs(model_dir, exist_ok=True)
                    checkpoint_path = os.path.join(model_dir, 'autoencoder_best_checkpoint.pt')
                    torch.save(best_model_state, checkpoint_path)
                    logger.info(f"Best autoencoder model checkpointed to {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to save autoencoder checkpoint: {e}")

            # Calculate autoencoder threshold in batches for memory efficiency
            # Cache reconstruction errors for reuse in pseudo-label generation
            self.autoencoder.eval()
            all_errors = []
            with torch.no_grad():
                for batch in train_loader:
                    clean = batch[0].to(device)
                    out = self.autoencoder(clean)
                    if self.autoencoder.vae:
                        recon = out[0]
                    else:
                        recon = out
                    batch_errors = F.mse_loss(recon, clean, reduction='none').mean(dim=1)
                    all_errors.extend(batch_errors.cpu().numpy())

            self.autoencoder_threshold = np.percentile(all_errors, 95)
            
            # Cache reconstruction errors for pseudo-label generation (avoid redundant computation)
            self._cached_reconstruction_errors = np.array(all_errors)

        # Train HDBSCAN (or DBSCAN if HDBSCAN not available)
        if 'dbscan' in self.algorithms:
            if HDBSCAN_AVAILABLE:
                logger.info("Using HDBSCAN for faster big data clustering")
                self.dbscan = hdbscan.HDBSCAN(
                    min_cluster_size=int(self.dbscan_params.get('min_samples', 5)),
                    min_samples=int(self.dbscan_params.get('min_samples', 5))
                )
            else:
                logger.warning("HDBSCAN not available, falling back to DBSCAN")
                self.dbscan = DBSCAN(
                    eps=float(self.dbscan_params.get('eps', 0.5)),
                    min_samples=int(self.dbscan_params.get('min_samples', 5))
                )
            self.dbscan.fit(features_scaled)

        # Progressive model training: Skip GNN for small datasets
        if len(features) < 1000 and 'gnn' in self.algorithms:
            logger.info(f"Dataset too small for GNN ({len(features)} samples). Skipping GNN training.")
            self.algorithms = [a for a in self.algorithms if a != 'gnn']
            self.gnn_weight = 0.0

        # Disable Optuna by default for faster training (can be enabled via parameter)
        if optimize_hyperparams:
            logger.warning("Optuna hyperparameter optimization enabled - this adds significant training time")
        else:
            logger.info("Using default hyperparameters for faster training")

        # Train supervised model (XGBoost/LightGBM/RandomForest/SVM)
        pseudo_labels = None
        if 'xgboost' in self.algorithms:
            logger.info("Starting supervised training")
            self.xgboost_model = ClaimAnomalyXGBoostModel(
                model_type=self.xgboost_params.get('model_type', 'xgboost'),
                **{k: v for k, v in self.xgboost_params.items() if k != 'model_type'}
            )


            if labels is None:
                # Use a consensus of IF and Autoencoder to create better pseudo-labels
                consensus_scores = np.zeros(len(features_scaled))
                if 'isolation_forest' in self.algorithms and self.isolation_forest is not None:
                    iso_scores = self.isolation_forest.decision_function(features_scaled)
                    denom = (iso_scores.max() - iso_scores.min())
                    iso_probs = 1 - (iso_scores - iso_scores.min()) / (denom if denom != 0 else 1.0)
                    consensus_scores += iso_probs
                
                if 'autoencoder' in self.algorithms and self.autoencoder is not None and self.autoencoder_threshold is not None:
                    # Use cached reconstruction errors if available (avoid redundant forward pass)
                    if hasattr(self, '_cached_reconstruction_errors') and self._cached_reconstruction_errors is not None:
                        all_errors = self._cached_reconstruction_errors
                        logger.info("Using cached reconstruction errors for pseudo-label generation")
                    else:
                        self.autoencoder.eval()
                        batch_size = self.autoencoder_params.get('batch_size', 1024)
                        all_errors = []
                        with torch.no_grad():
                            for i in range(0, len(features_scaled), batch_size):
                                batch_features = torch.FloatTensor(features_scaled[i:i+batch_size]).to(device)
                                out = self.autoencoder(batch_features)
                                reconstructed = out[0] if self.autoencoder.vae else out
                                batch_errors = F.mse_loss(reconstructed, batch_features, reduction='none').mean(dim=1)
                                all_errors.extend(batch_errors.cpu().numpy())
                    ae_probs = np.array(all_errors) / self.autoencoder_threshold
                    ae_probs = np.clip(ae_probs, 0, 1)
                    consensus_scores += ae_probs
                
                # Take top 5% as anomalies (consensus)
                threshold = np.percentile(consensus_scores, 95)
                pseudo_labels = (consensus_scores >= threshold).astype(int)
                
                logger.info("Generated %d consensus pseudo-labels for supervised training",
                            pseudo_labels.sum())
            else:
                pseudo_labels = labels
                logger.info("Using %d provided labels for supervised training",
                            len(pseudo_labels))

            if len(features_scaled) < 10:
                logger.warning("Very small dataset for supervised training")

            logger.info("Supervised training data shape: %s",
                        features_scaled.shape)
            logger.info("Supervised training with model type: %s",
                        self.xgboost_model.model_type)

            # Optimize hyperparameters if requested
            if optimize_hyperparams and self.xgboost_model.model_type in ['xgboost', 'lightgbm', 'catboost']:
                try:
                    logger.info(f"Starting hyperparameter optimization for {self.xgboost_model.model_type}")
                    optuna_n_trials_supervised = self.xgboost_params.get('optuna_n_trials', optuna_n_trials)
                    optuna_timeout_supervised = self.xgboost_params.get('optuna_timeout', optuna_timeout)
                    
                    # Adaptive CV folds based on dataset size for faster training
                    n_samples = len(features_scaled)
                    if n_samples > 100000:
                        cv_folds_adaptive = 3  # Reduced for large datasets
                        logger.info(f"Using {cv_folds_adaptive} CV folds for large dataset ({n_samples:,} samples)")
                    elif n_samples > 50000:
                        cv_folds_adaptive = 4
                        logger.info(f"Using {cv_folds_adaptive} CV folds for medium dataset ({n_samples:,} samples)")
                    else:
                        cv_folds_adaptive = 5  # Default for smaller datasets
                        logger.info(f"Using {cv_folds_adaptive} CV folds for small dataset ({n_samples:,} samples)")
                    
                    best_params = self.xgboost_model.optimize_hyperparameters(
                        features_scaled, 
                        pseudo_labels,
                        n_trials=optuna_n_trials_supervised,
                        timeout=optuna_timeout_supervised,
                        cv_folds=cv_folds_adaptive,
                        random_state=42,
                        imbalance_handler=self.imbalance_handler
                    )
                    logger.info(f"Supervised hyperparameter optimization completed")
                except Exception as e:
                    logger.warning(f"Supervised hyperparameter optimization failed: {e}. Using default parameters.")

            self.xgboost_model.fit(features_scaled, pseudo_labels)
            logger.info("Supervised training completed successfully")

        # GNN training (if graph is available and GNN is enabled)
        # Need pseudo-labels for the graph nodes; reuse supervised labels if available
        if TORCH_AVAILABLE and edge_index is not None and (
            'gnn' in self.algorithms or self.gnn_weight > 0 or optimize_hyperparams
        ):
            # Determine labels for GNN training
            gnn_labels = None
            if labels is not None:
                gnn_labels = labels
            elif 'xgboost' in self.algorithms and hasattr(self, 'xgboost_model') and self.xgboost_model is not None:
                # Use XGBoost-predicted pseudo labels probabilities
                xgb_probs = self.xgboost_model.predict_proba(features_scaled)[:, 1]
                threshold = np.percentile(xgb_probs, 95)
                gnn_labels = (xgb_probs >= threshold).astype(int)
            else:
                # Fallback to ensemble/IF pseudo labels (using earlier consensus logic if available)
                if pseudo_labels is not None:
                    gnn_labels = pseudo_labels
                else:
                    gnn_labels = (self.isolation_forest.predict(features_scaled) == -1).astype(int)

            if 'gnn' not in self.algorithms and self.gnn_weight > 0:
                # Auto-enable GNN if weight > 0 but not in algorithms list
                self.algorithms.append('gnn')

            # Optionally run Optuna to find better hyperparams first
            if optimize_hyperparams:
                try:
                    best_params = self._optimize_gnn_hyperparams(
                        features_scaled, edge_index, gnn_labels,
                        device=device, n_trials=optuna_n_trials, timeout=optuna_timeout
                    )
                    # Merge with existing params
                    self.gnn_params.update(best_params)
                except Exception as e:
                    logger.warning("Optuna step skipped: %s", e)

            # Run actual training with best/default params
            try:
                self._train_gnn(features_scaled, edge_index, gnn_labels, device=device, edge_type=edge_type)
            except Exception as e:
                logger.warning("GNN training failed: %s. Continuing without GNN.", e)
                self.gnn_model = None

        # Add 'gnn' to algorithms list if model is trained and weight > 0
        if self.gnn_model is not None and self.gnn_weight > 0 and 'gnn' not in self.algorithms:
            self.algorithms.append('gnn')
        
        # Optimize ensemble weights dynamically with Optuna (FPR minimization)
        if optimize_ensemble_weights or (optimize_hyperparams and OPTUNA_AVAILABLE):
            try:
                self.optimize_ensemble_weights(
                    features=features,
                    labels=pseudo_labels,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    device=device,
                    n_trials=optuna_n_trials,
                    timeout=optuna_timeout,
                    lambda_fpr=lambda_fpr
                )
            except Exception as opt_err:
                logger.warning("Dynamic Optuna ensemble weight optimization failed: %s", opt_err)

        # Train stacking ensemble if enabled and labels are available
        if self.use_stacking and labels is not None:
            self._train_stacking_ensemble(features_scaled, labels, device)

    def optimize_ensemble_weights(self, features, labels=None, edge_index=None, edge_type=None,
                                  device='cpu', n_trials=30, timeout=120, lambda_fpr=0.5, cv_folds=5):
        """Dynamically optimize ensemble weights via Optuna specifically to minimize False Positive Rate.
        
        Args:
            features: Input feature matrix (raw / unscaled).
            labels: Ground truth binary labels or pseudo-labels.
            edge_index: Graph edge index (for GNN scoring).
            edge_type: Graph edge types (optional).
            device: Computing device ('cpu' or 'cuda').
            n_trials: Number of Optuna optimization trials.
            timeout: Optimization timeout in seconds.
            lambda_fpr: Weight penalty for False Positive Rate in objective function.
            cv_folds: Number of Stratified K-Fold CV splits.
            
        Returns:
            Dictionary containing optimization results, best weights, and metrics comparison.
        """
        logger.info("Starting Dynamic Ensemble Weight Optimization (Optuna FPR Minimization)...")
        
        features_imputed = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features_imputed)
        
        individual_scores = {}
        
        # 1. Isolation Forest scores
        if self.isolation_forest is not None and 'isolation_forest' in self.algorithms:
            iso_scores = self.isolation_forest.decision_function(features_scaled)
            denom = (iso_scores.max() - iso_scores.min())
            individual_scores['isolation'] = 1 - (iso_scores - iso_scores.min()) / (denom if denom != 0 else 1.0)
            
        # 2. Autoencoder reconstruction error scores
        if self.autoencoder is not None and self.autoencoder_threshold is not None and 'autoencoder' in self.algorithms:
            self.autoencoder.eval()
            batch_size = get_optimal_batch_size(device, len(features_scaled), features_scaled.shape[1], default_batch=2048)
            all_errors = []
            with torch.no_grad():
                for i in range(0, len(features_scaled), batch_size):
                    batch_features = torch.FloatTensor(features_scaled[i:i+batch_size]).to(device)
                    out = self.autoencoder(batch_features)
                    recon = out[0] if self.autoencoder.vae else out
                    batch_errors = F.mse_loss(recon, batch_features, reduction='none').mean(dim=1)
                    all_errors.extend(batch_errors.cpu().numpy())
            ae_scores = np.array(all_errors) / self.autoencoder_threshold
            individual_scores['autoencoder'] = np.clip(ae_scores, 0, 1)

        # 3. XGBoost / Supervised scores
        if self.xgboost_model is not None and 'xgboost' in self.algorithms:
            try:
                individual_scores['xgboost'] = self.xgboost_model.predict_proba(features_scaled)[:, 1]
            except Exception as e:
                logger.warning("Error getting XGBoost scores for weight optimization: %s", e)

        # 4. GNN scores
        if self.gnn_model is not None and 'gnn' in self.algorithms and edge_index is not None:
            try:
                self.gnn_model.eval()
                with torch.no_grad():
                    num_nodes = features_scaled.shape[0]
                    ei = torch.LongTensor(edge_index).to(device)
                    if ei.dim() == 2 and ei.size(0) != 2 and ei.size(1) == 2:
                        ei = ei.t().contiguous()
                    feat_t = torch.FloatTensor(features_scaled).to(device)
                    batch_t = torch.zeros(num_nodes, dtype=torch.long, device=device)
                    out = self.gnn_model(feat_t, ei, batch_t, edge_type)
                    individual_scores['gnn'] = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            except Exception as e:
                logger.warning("Error getting GNN scores for weight optimization: %s", e)

        if not individual_scores:
            logger.warning("No individual model scores available for weight optimization.")
            return None

        # Run Optuna weight optimization
        optimizer = OptunaEnsembleOptimizer(
            n_trials=n_trials,
            timeout=timeout,
            lambda_fpr=lambda_fpr,
            cv_folds=cv_folds
        )
        
        result = optimizer.optimize(
            individual_scores=individual_scores,
            y_true=labels,
            active_algorithms=list(individual_scores.keys())
        )
        
        if result and result.get('status') == 'success':
            weights = result.get('weights', {})
            if 'isolation' in weights:
                self.isolation_weight = weights['isolation']
            if 'autoencoder' in weights:
                self.autoencoder_weight = weights['autoencoder']
            if 'xgboost' in weights:
                self.xgboost_weight = weights['xgboost']
            if 'gnn' in weights:
                self.gnn_weight = weights['gnn']
                
            self.weight_optimization_results = result
            logger.info("Updated CombinedAnomalyDetector weights with Optuna optimal weights: %s", weights)
            
        return result

    def _train_stacking_ensemble(self, features_scaled, labels, device='cpu'):
        """Train stacking ensemble with meta-learner"""
        try:
            logger.info("Training stacking ensemble...")
            
            # Collect base models that can provide predictions
            base_models = {}
            
            # Add XGBoost/LightGBM if available
            if self.xgboost_model is not None and hasattr(self.xgboost_model, 'model'):
                base_models['xgboost'] = self.xgboost_model.model
            
            # Create sklearn-style wrappers for other models
            if 'isolation_forest' in self.algorithms and self.isolation_forest is not None:
                base_models['isolation_forest'] = self._create_model_wrapper(
                    self.isolation_forest, 'isolation_forest', features_scaled
                )
            
            if 'autoencoder' in self.algorithms and self.autoencoder is not None:
                base_models['autoencoder'] = self._create_autoencoder_wrapper(
                    self.autoencoder, self.autoencoder_threshold, device
                )
            
            if len(base_models) < 2:
                logger.warning("Not enough base models for stacking (need at least 2). Skipping stacking.")
                self.use_stacking = False
                return
            
            # Create stacking ensemble
            self.stacking_ensemble = StackingEnsemble(
                base_models=base_models,
                meta_learner_type=self.stacking_params.get('meta_learner_type', 'logistic'),
                cv_folds=self.stacking_params.get('cv_folds', 5),
                use_proba=self.stacking_params.get('use_proba', True),
                random_state=self.stacking_params.get('random_state', 42)
            )
            
            # Fit stacking ensemble
            self.stacking_ensemble.fit(features_scaled, labels)
            
            # Get feature importance (importance of each base model)
            importance = self.stacking_ensemble.get_feature_importance()
            if importance:
                logger.info(f"Stacking ensemble base model importance: {importance}")
            
            logger.info("Stacking ensemble training completed successfully")
            
        except Exception as e:
            logger.warning(f"Stacking ensemble training failed: {e}. Falling back to weighted ensemble.")
            self.use_stacking = False
    
    def _create_model_wrapper(self, model, model_type, features_scaled):
        """Create a sklearn-compatible wrapper for non-sklearn models"""
        class ModelWrapper:
            def __init__(self, model, model_type):
                self.model = model
                self.model_type = model_type
                
            def fit(self, X, y=None):
                # Model already fitted
                return self
                
            def predict_proba(self, X):
                if self.model_type == 'isolation_forest':
                    scores = self.model.decision_function(X)
                    # Normalize to [0, 1] probability
                    denom = (scores.max() - scores.min())
                    if denom == 0:
                        return np.column_stack([1 - scores, scores])
                    probs = 1 - (scores - scores.min()) / denom
                    return np.column_stack([1 - probs, probs])
                return self.model.predict_proba(X)
                
            def predict(self, X):
                if self.model_type == 'isolation_forest':
                    return (self.model.predict(X) == -1).astype(int)
                return self.model.predict(X)
                
            def get_params(self, deep=True):
                return {}
                
            def set_params(self, **params):
                return self
        
        return ModelWrapper(model, model_type)
    
    def _create_autoencoder_wrapper(self, autoencoder, threshold, device):
        """Create a wrapper for autoencoder to make it sklearn-compatible"""
        class AutoencoderWrapper:
            def __init__(self, autoencoder, threshold, device):
                self.autoencoder = autoencoder
                self.threshold = threshold
                self.device = device
                
            def fit(self, X, y=None):
                # Autoencoder already fitted
                return self
                
            def predict_proba(self, X):
                self.autoencoder.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    reconstructed = self.autoencoder(X_tensor)
                    if self.autoencoder.vae:
                        reconstructed = reconstructed[0]
                    errors = F.mse_loss(reconstructed, X_tensor, reduction='none').mean(dim=1)
                    probs = errors.cpu().numpy() / self.threshold
                    probs = np.clip(probs, 0, 1)
                    return np.column_stack([1 - probs, probs])
                    
            def predict(self, X):
                probas = self.predict_proba(X)[:, 1]
                return (probas >= 0.5).astype(int)
                
            def get_params(self, deep=True):
                return {}
                
            def set_params(self, **params):
                return self
        
        return AutoencoderWrapper(autoencoder, threshold, device)
    
    def enable_stacking(self, meta_learner_type='logistic', cv_folds=5, use_proba=True, random_state=42):
        """Enable stacking ensemble
        
        Parameters
        ----------
        meta_learner_type : str
            Type of meta-learner: 'logistic', 'random_forest', 'xgboost', 'lightgbm'
        cv_folds : int
            Number of cross-validation folds
        use_proba : bool
            Whether to use probability predictions
        random_state : int
            Random state for reproducibility
        """
        self.use_stacking = True
        self.stacking_params = {
            'meta_learner_type': meta_learner_type,
            'cv_folds': cv_folds,
            'use_proba': use_proba,
            'random_state': random_state
        }
        logger.info(f"Stacking ensemble enabled with {meta_learner_type} meta-learner")

    def _optimize_gnn_hyperparams(self, node_features, edge_index, labels, device='cpu', n_trials=15, timeout=600):
        """Optuna-based hyperparameter search for the GNN.

        Splits the data into train/val and uses validation F1 as the objective.
        Returns the best hyperparameters dict.
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default GNN hyperparameters")
            return self.gnn_params

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score as _f1_score

        # Set seeds for reproducibility
        _set_global_seeds(42)

        labels_np = np.asarray(labels).astype(int)
        num_features = node_features.shape[1]
        x = torch.FloatTensor(node_features).to(device)
        # Ensure edge_index is in (2, n_edges) format for torch_geometric
        ei_tensor = torch.LongTensor(edge_index)
        if ei_tensor.dim() == 2 and ei_tensor.size(0) != 2:
            if ei_tensor.size(1) == 2:
                ei_tensor = ei_tensor.t().contiguous()
        ei = ei_tensor.to(device)
        y = torch.LongTensor(labels).to(device)
        num_nodes = node_features.shape[0]
        batch_tensor = torch.zeros(num_nodes, dtype=torch.long, device=device)

        # Stratified split
        idx = np.arange(num_nodes)
        try:
            train_idx, val_idx = train_test_split(
                idx, test_size=0.2, stratify=labels_np, random_state=42
            )
        except ValueError:
            train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        class_counts = np.bincount(labels_np, minlength=2)
        class_counts = np.where(class_counts == 0, 1, class_counts)
        class_weights = torch.FloatTensor(
            (len(labels_np) / (2 * class_counts))
        ).to(device)

        def objective(trial):
            hidden = trial.suggest_categorical('hidden_channels', [32, 64, 96, 128])
            heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            n_layers = trial.suggest_int('num_layers', 1, 3)
            lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            wd = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
            epochs = trial.suggest_int('epochs', 50, 200, step=25)
            # Skip incompatible head/hidden combos (hidden*heads must remain consistent)
            if hidden * heads > 1024:
                raise optuna.exceptions.TrialPruned()

            model = InsuranceAnomalyGNNModel(
                num_features=num_features,
                num_classes=2,
                hidden_channels=hidden,
                num_heads=heads,
                dropout=dropout,
                num_layers=n_layers,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            crit = torch.nn.CrossEntropyLoss(weight=class_weights)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

            best_val = -1.0
            patience_ctr = 0
            for ep in range(epochs):
                model.train()
                opt.zero_grad()
                out = model(x, ei, batch_tensor)
                loss = crit(out[train_mask], y[train_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    val_out = model(x, ei, batch_tensor)
                    val_pred = val_out[val_mask].argmax(dim=1).cpu().numpy()
                    val_true = y[val_mask].cpu().numpy()
                    val_f1 = _f1_score(val_true, val_pred, average='binary', zero_division=0)
                trial.report(val_f1, ep)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if val_f1 > best_val:
                    best_val = val_f1
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= 8:
                        break
            return best_val

        try:
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(),
                study_name='gnn_anomaly_optimization',
            )
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
            best = dict(study.best_params)
            self.gnn_history_best = {
                'best_value': study.best_value,
                'n_trials': len(study.trials),
            }
            logger.info("Optuna GNN tuning done. Best val F1=%.4f in %d trials.",
                        study.best_value, len(study.trials))
            return best
        except Exception as e:
            logger.warning("Optuna tuning failed: %s. Falling back to default params.", e)
            return self.gnn_params

    def _train_gnn_sampled(self, node_features, edge_index, labels,
                           train_idx, val_idx, device, criterion, epochs,
                           patience, num_layers, cancel_event, soft_labels,
                           devnet_margin, devnet_weight, use_soft_labels,
                           edge_attr=None):
        """Train node predictions on bounded neighbor-sampled subgraphs."""
        from torch_geometric.loader import NeighborLoader

        data = Data(
            x=torch.as_tensor(node_features, dtype=torch.float32),
            edge_index=edge_index.cpu(),
            y=torch.as_tensor(labels, dtype=torch.long),
        )
        if edge_attr is not None:
            data.edge_attr = edge_attr.cpu()
        batch_size = int(self.gnn_params.get(
            'batch_size', 1024 if getattr(device, 'type', 'cpu') == 'cuda' else 512
        ))
        neighbors = self.gnn_params.get('num_neighbors', [15, 10] if num_layers == 2 else [15] * num_layers)
        if isinstance(neighbors, int):
            neighbors = [neighbors] * num_layers
        neighbors = list(neighbors)
        if len(neighbors) != num_layers:
            neighbors = (neighbors + [neighbors[-1] if neighbors else 15] * num_layers)[:num_layers]

        train_loader = NeighborLoader(
            data, input_nodes=torch.as_tensor(train_idx),
            num_neighbors=neighbors, batch_size=batch_size, shuffle=True,
            num_workers=0,
        )
        val_loader = NeighborLoader(
            data, input_nodes=torch.as_tensor(val_idx),
            num_neighbors=neighbors, batch_size=batch_size, shuffle=False,
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(
            self.gnn_model.parameters(),
            lr=self.gnn_params.get('learning_rate', 0.005),
            weight_decay=self.gnn_params.get('weight_decay', 5e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        history = {
            'train_loss': [], 'val_loss': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [], 'lr': [],
            'dev_loss': [], 'ce_loss': [],
        }
        best_state = None
        best_val_f1 = -1.0
        patience_counter = 0

        for epoch in range(epochs):
            if cancel_event.is_set():
                break
            self.gnn_model.train()
            train_losses = []
            for batch in train_loader:
                batch = batch.to(device)
                seed_count = int(batch.batch_size)
                optimizer.zero_grad()
                out = self.gnn_model(batch.x, batch.edge_index, None, getattr(batch, 'edge_attr', None))[:seed_count]
                ce_loss = criterion(out, batch.y[:seed_count])
                dev_loss = torch.tensor(0.0, device=device)
                if use_soft_labels and soft_labels is not None:
                    seed_ids = batch.n_id[:seed_count].cpu().numpy()
                    target = torch.as_tensor(soft_labels[seed_ids], dtype=torch.float32, device=device)
                    pos_prob = torch.softmax(out, dim=-1)[:, 1]
                    dev_loss = F.smooth_l1_loss(pos_prob, target)
                    margin_loss = torch.relu(devnet_margin * (1.0 - target) - pos_prob * devnet_margin).mean()
                    dev_loss = dev_loss + margin_loss
                    loss = (1.0 - devnet_weight) * ce_loss + devnet_weight * dev_loss
                else:
                    loss = ce_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gnn_model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))
            scheduler.step()

            self.gnn_model.eval()
            val_losses, predictions, truths = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    seed_count = int(batch.batch_size)
                    out = self.gnn_model(batch.x, batch.edge_index, None, getattr(batch, 'edge_attr', None))[:seed_count]
                    val_losses.append(float(criterion(out, batch.y[:seed_count]).item()))
                    predictions.extend(out.argmax(dim=1).cpu().numpy())
                    truths.extend(batch.y[:seed_count].cpu().numpy())
            val_f1 = f1_score(truths, predictions, average='binary', zero_division=0)
            val_prec = precision_score(truths, predictions, average='binary', zero_division=0)
            val_rec = recall_score(truths, predictions, average='binary', zero_division=0)
            history['train_loss'].append(float(np.mean(train_losses)))
            history['val_loss'].append(float(np.mean(val_losses)))
            history['val_f1'].append(float(val_f1))
            history['val_precision'].append(float(val_prec))
            history['val_recall'].append(float(val_rec))
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['dev_loss'].append(float(dev_loss.item()))
            history['ce_loss'].append(float(ce_loss.item()))
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.gnn_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.gnn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        self.gnn_history = history
        self.gnn_best_val_f1 = float(best_val_f1)
        self.gnn_use_soft_labels = use_soft_labels
        self.gnn_dev_prior = None
        self.gnn_soft_labels = soft_labels
        logger.info("Sampled GNN training completed. Best val F1 = %.4f", best_val_f1)

    def _train_gnn(self, node_features, edge_index, labels, device='cpu',
                   cancel_event=None, edge_type=None):
        """Train Graph Neural Network for anomaly detection

        Improvements over naive implementation:
        - Uses train/val split with stratification for proper early stopping
        - Class-weighted CrossEntropyLoss to handle imbalance
        - AdamW + CosineAnnealingLR scheduler
        - Best-checkpoint based on validation F1
        - Always builds a valid `batch` tensor for global pooling
        - Records training history for downstream UI visualization
        - Optional DevNet-style soft labels + deviation loss (P3-13)
        - QW5: cooperative cancellation via ``cancel_event`` — checked
          at every epoch boundary; on cancel, saves the best checkpoint
          seen so far and returns instead of running remaining epochs.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score as _f1_score
        import threading as _threading
        if cancel_event is None:
            cancel_event = _threading.Event()

        # Set seeds for reproducibility
        _set_global_seeds(42)

        num_nodes = node_features.shape[0]
        num_features = node_features.shape[1]
        num_classes = 2  # Anomaly or not

        # Optimized GNN parameters for CPU/GPU
        device_type = device.type if hasattr(device, 'type') else 'cpu'

        # Simplified architecture for faster training
        if device_type == 'cpu':
            # Reduced complexity for CPU training
            hidden_channels = self.gnn_params.get('hidden_channels', 32)  # Reduced from 64
            # Ensure minimum values, if 0 use defaults
            hidden_channels = max(1, hidden_channels) if hidden_channels > 0 else 32
            num_heads = self.gnn_params.get('num_heads', 2)  # Reduced from 4
            num_heads = max(1, num_heads) if num_heads > 0 else 2
            num_layers = self.gnn_params.get('num_layers', 1)  # Reduced from 2
            num_layers = max(1, num_layers) if num_layers > 0 else 1
            epochs = self.gnn_params.get('epochs', 50)  # Reduced from 200
            epochs = max(1, epochs) if epochs > 0 else 50
            patience = self.gnn_params.get('early_stopping_patience', 8)  # More aggressive
            patience = max(1, patience) if patience > 0 else 8
        else:
            # Full complexity for GPU training
            hidden_channels = self.gnn_params.get('hidden_channels', 64)
            hidden_channels = max(1, hidden_channels) if hidden_channels > 0 else 64
            num_heads = self.gnn_params.get('num_heads', 4)
            num_heads = max(1, num_heads) if num_heads > 0 else 4
            num_layers = self.gnn_params.get('num_layers', 2)
            num_layers = max(1, num_layers) if num_layers > 0 else 2
            epochs = self.gnn_params.get('epochs', 200)
            epochs = max(1, epochs) if epochs > 0 else 200
            patience = self.gnn_params.get('early_stopping_patience', 15)
            patience = max(1, patience) if patience > 0 else 15

        # DevNet-style soft labels configuration (disabled by default for speed)
        use_soft_labels = bool(self.gnn_params.get('use_soft_labels', False))
        if use_soft_labels:
            logger.warning("Soft labels enabled - this adds significant training overhead")

        devnet_margin = float(self.gnn_params.get('devnet_margin', 5.0))
        devnet_weight = float(self.gnn_params.get('devnet_weight', 0.5))

        edge_attr = None
        edge_dim = None
        edge_index_for_attr = torch.as_tensor(edge_index, dtype=torch.long)
        if edge_index_for_attr.dim() == 2 and edge_index_for_attr.size(0) != 2 and edge_index_for_attr.size(1) == 2:
            edge_index_for_attr = edge_index_for_attr.t().contiguous()
        if edge_type is not None:
            edge_type_tensor = torch.as_tensor(edge_type, dtype=torch.long).view(-1)
            if edge_type_tensor.numel() == edge_index_for_attr.size(1):
                edge_attr = F.one_hot(edge_type_tensor.clamp(0, 2), num_classes=3).float()
                edge_dim = 3

        # Initialize GNN model with optimized parameters
        self.gnn_model = InsuranceAnomalyGNNModel(
            num_features=num_features,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            dropout=self.gnn_params.get('dropout', 0.2),
            num_layers=num_layers,
            edge_dim=edge_dim,
        ).to(device)

        # Class-weighted loss for imbalance
        labels_np = np.asarray(labels).astype(int)
        class_counts = np.bincount(labels_np, minlength=num_classes)
        # Avoid division by zero for missing classes
        class_counts = np.where(class_counts == 0, 1, class_counts)
        class_weights = torch.FloatTensor(
            (len(labels_np) / (num_classes * class_counts))
        ).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Soft labels: derive continuous anomaly scores in [0, 1] from autoencoder
        # reconstruction error (z-score on normal-data distribution).
        # This implements the "DevNet" idea: use a deviation prior for the
        # anomaly score instead of relying on hard 0/1 targets.
        soft_labels_np = None
        dev_prior = None  # (mu, sigma) of the normal-deviation reference
        if use_soft_labels:
            try:
                if self.autoencoder is not None and self.autoencoder_threshold is not None:
                    self.autoencoder.eval()
                    with torch.no_grad():
                        x_full = torch.FloatTensor(node_features).to(device)
                        # Use a larger batch and split to avoid OOM
                        bs = 1024
                        recon_errors = []
                        for i in range(0, x_full.size(0), bs):
                            recon = self.autoencoder(x_full[i:i + bs])
                            err = F.mse_loss(recon, x_full[i:i + bs], reduction='none').mean(dim=1)
                            recon_errors.append(err.cpu().numpy())
                    recon_errors = np.concatenate(recon_errors, axis=0)
                    normal_mask = labels_np == 0
                    if normal_mask.sum() >= 2:
                        ref = recon_errors[normal_mask]
                        mu = float(np.mean(ref))
                        sigma = float(np.std(ref) + 1e-6)
                    else:
                        mu = 0.0
                        sigma = float(np.std(recon_errors) + 1e-6)
                    z = (recon_errors - mu) / sigma
                    # Map deviation to a smooth probability in [0, 1]
                    soft_labels_np = 1.0 / (1.0 + np.exp(-z))
                    dev_prior = (mu, sigma)
                else:
                    # No autoencoder available -> fall back to a small noise
                    # around the hard label so that the model still trains
                    # with continuous targets (better than nothing).
                    rng = np.random.default_rng(42)
                    soft_labels_np = labels_np.astype(float) * 0.9 + 0.05
                    soft_labels_np = np.clip(
                        soft_labels_np + rng.normal(0, 0.05, size=soft_labels_np.shape),
                        0.0, 1.0,
                    )
            except Exception as e:
                logger.warning("Soft label generation failed: %s. Falling back to hard labels.", e)
                use_soft_labels = False

        soft_labels_t = (
            torch.FloatTensor(soft_labels_np).to(device)
            if soft_labels_np is not None else None
        )

        # AdamW + cosine scheduler
        optimizer = torch.optim.AdamW(
            self.gnn_model.parameters(),
            lr=self.gnn_params.get('learning_rate', 0.005),
            weight_decay=self.gnn_params.get('weight_decay', 5e-4),
        )
        # Use optimized epochs (already set based on device type above)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Prepare tensors
        x = torch.FloatTensor(node_features).to(device)
        # Ensure edge_index is in (2, n_edges) format for torch_geometric
        edge_index_tensor = torch.LongTensor(edge_index)
        if edge_index_tensor.dim() == 2 and edge_index_tensor.size(0) != 2:
            # If shape is (n_edges, 2), transpose to (2, n_edges)
            if edge_index_tensor.size(1) == 2:
                edge_index_tensor = edge_index_tensor.t().contiguous()
        edge_index_tensor = edge_index_tensor.to(device)
        edge_attr_device = edge_attr.to(device) if edge_attr is not None else None
        y = torch.LongTensor(labels).to(device)

        # Stratified train/val split (use indices to keep masks)
        idx = np.arange(num_nodes)
        try:
            train_idx, val_idx = train_test_split(
                idx, test_size=0.2, stratify=labels_np, random_state=42
            )
        except ValueError:
            # Stratify fails if a class has <2 samples
            train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        # Use sampled subgraphs for large graphs to bound per-step memory.
        adaptive_thresh = get_adaptive_gnn_threshold(device, num_features, num_nodes)
        sampling_threshold = int(self.gnn_params.get('sampling_threshold_nodes', adaptive_thresh))
        use_neighbor_sampling = bool(self.gnn_params.get(
            'use_neighbor_sampling', num_nodes > sampling_threshold
        ))
        if use_neighbor_sampling:
            try:
                return self._train_gnn_sampled(
                    node_features=node_features,
                    edge_index=edge_index_tensor.cpu(),
                    labels=labels_np,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    device=device,
                    criterion=criterion,
                    epochs=epochs,
                    patience=patience,
                    num_layers=num_layers,
                    cancel_event=cancel_event,
                    soft_labels=soft_labels_np,
                    devnet_margin=devnet_margin,
                    devnet_weight=devnet_weight,
                    use_soft_labels=use_soft_labels,
                    edge_attr=edge_attr,
                )
            except Exception as exc:
                logger.warning(
                    "Neighbor sampling unavailable (%s); falling back to full-batch GNN.",
                    exc,
                )

        # Build a single graph-level batch tensor (all nodes -> graph 0) for pooling
        batch_tensor = torch.zeros(num_nodes, dtype=torch.long, device=device)

        # Training history (for UI curves)
        history = {
            'train_loss': [], 'val_loss': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [], 'lr': [],
            'dev_loss': [], 'ce_loss': [],
        }

        best_val_f1 = -1.0
        best_state = None
        patience = int(self.gnn_params.get('early_stopping_patience', 15))
        patience_counter = 0

        for epoch in range(epochs):
            # QW5: cooperative cancellation — exit the epoch loop if
            # the user has pressed the cancel button. Best checkpoint
            # (so far) is preserved and ``history`` is truncated.
            if cancel_event.is_set():
                logger.info("GNN training cancelled at epoch %d by user.", epoch)
                break
            self.gnn_model.train()
            optimizer.zero_grad()
            out = self.gnn_model(x, edge_index_tensor, batch_tensor, edge_attr_device)
            ce_loss = criterion(out[train_mask], y[train_mask])

            # DevNet-style deviation loss (P3-13): use the model's
            # positive-class probability as a "deviation score" and
            # regress it toward the soft label / push anomalies beyond
            # the normal reference.
            if use_soft_labels and soft_labels_t is not None:
                pos_prob = torch.softmax(out, dim=-1)[:, 1]
                # BCE-with-logits-like soft target: smooth L1 between
                # predicted positive prob and the soft label.
                dev_loss = F.smooth_l1_loss(pos_prob[train_mask], soft_labels_t[train_mask])
                # Margin term: push anomalies past a deviation margin
                # relative to the normal reference. This is the classic
                # DevNet hinge on top of a z-score prior.
                if dev_prior is not None:
                    mu_n, sigma_n = dev_prior
                    # Use the soft anomaly prob as proxy of deviation.
                    # For normals we want prob near 0, for anomalies
                    # we want prob > margin in standardized space.
                    margin_term = torch.relu(
                        devnet_margin * (1.0 - soft_labels_t[train_mask]) - pos_prob[train_mask] * devnet_margin
                    )
                    dev_loss = dev_loss + margin_term.mean()
                loss = (1.0 - devnet_weight) * ce_loss + devnet_weight * dev_loss
            else:
                dev_loss = torch.tensor(0.0, device=device)
                loss = ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gnn_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Validation
            self.gnn_model.eval()
            with torch.no_grad():
                val_out = self.gnn_model(x, edge_index_tensor, batch_tensor, edge_attr_device)
                val_loss = criterion(val_out[val_mask], y[val_mask]).item()
                val_pred = val_out[val_mask].argmax(dim=1).cpu().numpy()
                val_true = y[val_mask].cpu().numpy()
                val_f1 = _f1_score(val_true, val_pred, average='binary', zero_division=0)
                try:
                    val_prec = precision_score(val_true, val_pred, average='binary', zero_division=0)
                    val_rec = recall_score(val_true, val_pred, average='binary', zero_division=0)
                except Exception:
                    val_prec, val_rec = 0.0, 0.0

            history['train_loss'].append(float(loss.item()))
            history['val_loss'].append(float(val_loss))
            history['val_f1'].append(float(val_f1))
            history['val_precision'].append(float(val_prec))
            history['val_recall'].append(float(val_rec))
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['dev_loss'].append(float(dev_loss.item()) if torch.is_tensor(dev_loss) else 0.0)
            history['ce_loss'].append(float(ce_loss.item()))

            if epoch % 20 == 0:
                logger.info(
                    "GNN Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                    "val_f1=%.4f | val_prec=%.4f | val_rec=%.4f",
                    epoch, epochs, loss.item(), val_loss,
                    val_f1, val_prec, val_rec,
                )

            # Auto-save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                try:
                    import os
                    os.makedirs("models/checkpoints", exist_ok=True)
                    torch.save(self.gnn_model.state_dict(), f"models/checkpoints/gnn_checkpoint_latest.pt")
                except Exception as e:
                    logger.warning(f"Failed to save GNN checkpoint: {e}")

            # Best-checkpoint by val F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.gnn_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("GNN early stopping at epoch %d (best val F1=%.4f)",
                                epoch, best_val_f1)
                    break
            
            # Periodic checkpointing every 10 epochs
            if epoch % 10 == 0 and best_state is not None:
                try:
                    import os
                    model_dir = os.path.dirname(self.gnn_params.get('model_prefix', 'models/fraud_detector'))
                    os.makedirs(model_dir, exist_ok=True)
                    checkpoint_path = os.path.join(model_dir, f'gnn_checkpoint_epoch{epoch}.pt')
                    torch.save(best_state, checkpoint_path)
                    logger.info(f"GNN checkpointed at epoch {epoch} to {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to save GNN checkpoint at epoch {epoch}: {e}")

        if best_state is not None:
            self.gnn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            # Save best GNN model to disk for checkpointing
            try:
                import os
                model_dir = os.path.dirname(self.gnn_params.get('model_prefix', 'models/fraud_detector'))
                os.makedirs(model_dir, exist_ok=True)
                checkpoint_path = os.path.join(model_dir, 'gnn_best_checkpoint.pt')
                torch.save(best_state, checkpoint_path)
                logger.info(f"Best GNN model checkpointed to {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to save GNN best checkpoint: {e}")
        # Persist history for UI rendering
        self.gnn_history = history
        self.gnn_best_val_f1 = float(best_val_f1)
        # Persist DevNet prior + soft labels for downstream analysis
        self.gnn_use_soft_labels = use_soft_labels
        self.gnn_dev_prior = dev_prior
        self.gnn_soft_labels = soft_labels_np
        logger.info("GNN training completed. Best val F1 = %.4f%s",
                    best_val_f1,
                    " (soft labels / DevNet)" if use_soft_labels else "")

    def update_weights_with_performance(self, performance_metrics):
        """Update weights based on performance feedback"""
        if self.use_dynamic_weights and self.weight_optimizer:
            current_weights = {
                'isolation': self.isolation_weight,
                'autoencoder': self.autoencoder_weight,
                'xgboost': self.xgboost_weight,
                'gnn': self.gnn_weight,
            }

            updated_weights = self.weight_optimizer.update_weights_based_on_performance(
                current_weights, performance_metrics
            )

            # Update weights
            self.isolation_weight = updated_weights['isolation']
            self.autoencoder_weight = updated_weights['autoencoder']
            self.xgboost_weight = updated_weights['xgboost']
            self.gnn_weight = updated_weights.get('gnn', 0.0)

            logger.info(
                "Weights updated based on performance: Isolation=%.3f, "
                "Autoencoder=%.3f, XGBoost=%.3f, GNN=%.3f",
                self.isolation_weight, self.autoencoder_weight,
                self.xgboost_weight, self.gnn_weight,
            )

    def predict_anomaly_probability(self, features, edge_index=None, edge_type=None, device='cpu'):
        """Predict anomaly probability using all three methods"""
        # Impute missing values, then scale features
        features_imputed = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features_imputed)
        
        # Isolation Forest predictions
        if self.isolation_forest is not None and 'isolation_forest' in self.algorithms:
            iso_scores = self.isolation_forest.decision_function(features_scaled)
            denom = (iso_scores.max() - iso_scores.min())
            iso_probabilities = 1 - (iso_scores - iso_scores.min()) / (denom if denom != 0 else 1.0)
        else:
            iso_probabilities = np.zeros(len(features))
        
        # Autoencoder predictions
        if self.autoencoder is not None and self.autoencoder_threshold is not None and 'autoencoder' in self.algorithms:
            self.autoencoder.eval()
            # Use dynamic batch size for inference
            default_batch = 4096 if str(device) == 'cpu' else 1024
            batch_size = get_optimal_batch_size(
                device, 
                len(features_scaled), 
                features_scaled.shape[1], 
                default_batch=default_batch
            )
            # Allow user override if specified
            user_batch_size = self.autoencoder_params.get('batch_size')
            if user_batch_size and user_batch_size > 0:
                batch_size = min(user_batch_size, batch_size)
            
            all_errors = []

            with torch.no_grad():
                # Process in batches to avoid OOM
                for i in range(0, len(features_scaled), batch_size):
                    batch_features = torch.FloatTensor(features_scaled[i:i+batch_size]).to(device)
                    out = self.autoencoder(batch_features)
                    # QW4: VAE forward returns a (recon, mu, logvar) tuple
                    if self.autoencoder.vae:
                        reconstructed = out[0]
                    else:
                        reconstructed = out
                    batch_errors = F.mse_loss(reconstructed, batch_features, reduction='none').mean(dim=1)
                    all_errors.extend(batch_errors.cpu().numpy())

            ae_probabilities = np.array(all_errors) / self.autoencoder_threshold
            ae_probabilities = np.clip(ae_probabilities, 0, 1)
        else:
            ae_probabilities = np.zeros(len(features))

        # DBSCAN/HDBSCAN predictions
        if self.dbscan is not None and 'dbscan' in self.algorithms:
            try:
                # For HDBSCAN: use stored outlier_scores_ from training (no re-fit)
                if HDBSCAN_AVAILABLE and hasattr(self.dbscan, 'outlier_scores_') and self.dbscan.outlier_scores_ is not None:
                    dbscan_probabilities = np.clip(self.dbscan.outlier_scores_, 0, 1)
                elif hasattr(self.dbscan, 'labels_') and self.dbscan.labels_ is not None:
                    # Use distance-to-nearest-core-sample as continuous score (DBSCAN)
                    from sklearn.neighbors import NearestNeighbors
                    labels_arr = np.asarray(self.dbscan.labels_)
                    core_mask = labels_arr != -1
                    if core_mask.sum() > 1:
                        # Distance to nearest core sample (in scaled space)
                        nbrs = NearestNeighbors(n_neighbors=1).fit(features_scaled[core_mask])
                        dists, _ = nbrs.kneighbors(features_scaled)
                        d = dists[:, 0]
                        d_min, d_max = d.min(), d.max()
                        dbscan_probabilities = (d - d_min) / (d_max - d_min + 1e-8)
                    else:
                        dbscan_probabilities = (labels_arr == -1).astype(float)
                else:
                    dbscan_probabilities = np.zeros(len(features))
            except Exception as e:
                logger.warning("DBSCAN/HDBSCAN prediction error: %s", e)
                dbscan_probabilities = np.zeros(len(features))
        else:
            dbscan_probabilities = np.zeros(len(features))
        
        # Supervised predictions (if available)
        if self.xgboost_model is not None and 'xgboost' in self.algorithms:
            xgb_probabilities = self.xgboost_model.predict_proba(features_scaled)[:, 1]
        else:
            xgb_probabilities = np.zeros(len(features))

        # GNN predictions
        if self.gnn_model is not None and ('gnn' in self.algorithms or self.gnn_weight > 0):
            self.gnn_model.eval()
            with torch.no_grad():
                try:
                    features_tensor = torch.FloatTensor(features_scaled).to(device)
                    # Always build a valid batch tensor (all nodes -> graph 0)
                    num_nodes = features_scaled.shape[0]
                    batch_tensor = torch.zeros(num_nodes, dtype=torch.long, device=device)
                    
                    if edge_index is not None:
                        # Ensure edge_index is in (2, n_edges) format
                        ei_tensor = torch.LongTensor(edge_index)
                        if ei_tensor.dim() == 2 and ei_tensor.size(0) != 2:
                            if ei_tensor.size(1) == 2:
                                ei_tensor = ei_tensor.t().contiguous()
                        ei = ei_tensor.to(device)
                        edge_attr = None
                        if edge_type is not None:
                            type_tensor = torch.as_tensor(edge_type, dtype=torch.long).view(-1)
                            if type_tensor.numel() == ei.size(1):
                                edge_attr = F.one_hot(type_tensor.clamp(0, 2), num_classes=3).float().to(device)
                        # Use adaptive threshold for NeighborLoader based on device memory
                        adaptive_threshold = get_adaptive_gnn_threshold(device, features_scaled.shape[1], num_nodes)
                        
                        if num_nodes > adaptive_threshold:
                            try:
                                from torch_geometric.loader import NeighborLoader
                                data = Data(x=features_tensor, edge_index=ei)
                                if edge_attr is not None:
                                    data.edge_attr = edge_attr
                                # Use dynamic batch size for NeighborLoader
                                loader_batch_size = get_optimal_batch_size(
                                    device, 
                                    num_nodes, 
                                    features_scaled.shape[1], 
                                    default_batch=1024
                                )
                                # Use bounded fanout [15, 10] instead of [-1] to prevent neighbor explosion OOM
                                loader = NeighborLoader(
                                    data,
                                    num_neighbors=[15, 10],
                                    batch_size=loader_batch_size,
                                    shuffle=False,
                                    num_workers=0
                                )
                                gnn_probs_list = []
                                for batch in loader:
                                    out = self.gnn_model(batch.x, batch.edge_index, None, getattr(batch, 'edge_attr', None))
                                    # Only take the predictions for the seed nodes
                                    probs = torch.softmax(out[:batch.batch_size], dim=1)[:, 1]
                                    gnn_probs_list.append(probs.cpu().numpy())
                                gnn_probabilities = np.concatenate(gnn_probs_list)
                            except Exception as sampler_err:
                                logger.warning("NeighborLoader inference failed (%s). Using chunked direct forward.", sampler_err)
                                # Fallback chunked forward without full-batch OOM
                                chunk_size = min(2048, max(256, num_nodes // 10))
                                gnn_probs_list = []
                                for i in range(0, num_nodes, chunk_size):
                                    chunk_end = min(i + chunk_size, num_nodes)
                                    out = self.gnn_model(features_tensor[i:chunk_end], None, None, None)
                                    probs = torch.softmax(out, dim=1)[:, 1]
                                    gnn_probs_list.append(probs.cpu().numpy())
                                gnn_probabilities = np.concatenate(gnn_probs_list)
                        else:
                            # Direct forward for smaller graphs (use real batch_tensor)
                            out = self.gnn_model(features_tensor, ei, batch_tensor, edge_attr)
                            gnn_probabilities = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    else:
                        gnn_probabilities = np.zeros(len(features))
                except Exception as e:
                    logger.warning("GNN inference error: %s. Falling back to zero.", e)
                    gnn_probabilities = np.zeros(len(features))
        else:
            gnn_probabilities = np.zeros(len(features))

        weights = {
            'isolation_forest': self.isolation_weight if 'isolation_forest' in self.algorithms else 0.0,
            'autoencoder': self.autoencoder_weight if 'autoencoder' in self.algorithms else 0.0,
            'dbscan': self.dbscan_weight if 'dbscan' in self.algorithms else 0.0,
            'xgboost': self.xgboost_weight if 'xgboost' in self.algorithms else 0.0,
            'gnn': self.gnn_weight if 'gnn' in self.algorithms else 0.0
        }
        total_w = sum(weights.values())
        if total_w <= 0:
            total_w = 1.0
            weights = {k: 1.0 / 4 for k in weights}
        else:
            weights = {k: v / total_w for k, v in weights.items()}

        # QW3: Rank-based ensemble — every algorithm contributes on a
        # comparable [0, 1] scale regardless of the raw range of its
        # scores. Falls back to the raw score when the array is empty
        # or all-equal (rank would be undefined / constant).
        # Use fast rank normalization for large arrays
        def _rank_norm(arr):
            return fast_rank_normalize(arr, threshold=10000)

        individual = {
            'isolation_forest': iso_probabilities,
            'autoencoder': ae_probabilities,
            'dbscan': dbscan_probabilities,
            'xgboost': xgb_probabilities,
            'gnn': gnn_probabilities,
        }
        if bool(getattr(self, 'rank_ensemble', True)):
            individual = {k: _rank_norm(v) for k, v in individual.items()}

        combined_probabilities = sum(
            weights[k] * individual[k] for k in individual
        )

        # Clip for safety — though rank-norm is already in (0, 1).
        combined_probabilities = np.clip(combined_probabilities, 0.0, 1.0)

        # Use stacking ensemble if enabled and available
        if self.use_stacking and self.stacking_ensemble is not None:
            try:
                stacking_probabilities = self.stacking_ensemble.predict_proba(features_scaled)
                # Return stacking predictions as combined, but keep individual for reference
                return stacking_probabilities, {
                    'isolation_forest': iso_probabilities,
                    'autoencoder': ae_probabilities,
                    'dbscan': dbscan_probabilities,
                    'xgboost': xgb_probabilities,
                    'gnn': gnn_probabilities,
                    'stacking': stacking_probabilities
                }
            except Exception as e:
                logger.warning(f"Stacking ensemble prediction failed: {e}. Falling back to weighted ensemble.")
                self.use_stacking = False

        return combined_probabilities, {
            'isolation_forest': iso_probabilities,
            'autoencoder': ae_probabilities,
            'dbscan': dbscan_probabilities,
            'xgboost': xgb_probabilities,
            'gnn': gnn_probabilities
        }
    
    def predict_anomaly_labels(self, features, edge_index=None, threshold=0.5, device='cpu'):
        """Predict anomaly labels"""
        probabilities, _ = self.predict_anomaly_probability(features, edge_index=edge_index, device=device)
        return (probabilities > threshold).astype(int)

    def predict(self, features, edge_index=None, threshold=0.5, device='cpu'):
        """Predict binary anomaly labels (0 for normal, 1 for anomaly)"""
        return self.predict_anomaly_labels(features, edge_index=edge_index, threshold=threshold, device=device)

    # ------------------------------------------------------------------ #
    # QW1: Stratified K-Fold evaluation
    # ------------------------------------------------------------------ #
    def cross_validate(self, features, labels, edge_index=None, n_splits=5,
                       device='cpu', random_state=42, refit=False):
        """Stratified K-Fold cross-validation (supervised, hard labels).

        For each fold:
          - split train/val with stratification
          - fit a *fresh* detector on the train fold
          - compute PR-AUC, ROC-AUC, Brier, F1 (at threshold 0.5) on the val fold

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (will be re-imputed/scaled inside ``fit``).
        labels : array-like
            Hard binary labels (0 / 1).
        edge_index : torch.LongTensor, optional
            Graph edges. If provided, *all* train-fold nodes are used to
            build the graph (the graph is rebuilt per fold to keep the
            splits honest).
        n_splits : int
            Number of folds (default 5).
        refit : bool
            If True, refit the detector on the full data after CV so the
            caller's instance behaves as if ``fit`` had been called.
            Default False (the caller keeps whatever state it had).

        Returns
        -------
        dict
            Per-fold metric dict + mean/std summary.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import (precision_recall_curve, auc, roc_auc_score,
                                     brier_score_loss, f1_score)
        import copy as _copy

        labels_np = np.asarray(labels).astype(int)
        skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True,
                              random_state=int(random_state))
        per_fold = []
        pr_aucs, roc_aucs, briers, f1s = [], [], [], []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(features, labels_np)):
            try:
                fold_detector = _copy.deepcopy(self)
                # Reset fitted components so fit() rebuilds them cleanly
                setattr(fold_detector, 'isolation_forest', None)
                fold_detector.autoencoder = None
                fold_detector.autoencoder_threshold = None
                fold_detector.dbscan = None
                fold_detector.xgboost_model = None
                fold_detector.gnn_model = None
                fold_detector.imputer = SimpleImputer(strategy='median')
                fold_detector.scaler = StandardScaler()
                from sklearn.ensemble import IsolationForest as _IF
                fold_detector.isolation_forest = _IF(
                    contamination=0.05,
                    random_state=42,
                    n_estimators=getattr(self.isolation_forest, 'n_estimators', 100)
                        if self.isolation_forest is not None else 100,
                )

                X_tr = features[tr_idx]
                y_tr = labels_np[tr_idx]
                X_va = features[va_idx]
                y_va = labels_np[va_idx]

                # Build the graph on training nodes only (when provided).
                if edge_index is not None and 'gnn' in fold_detector.algorithms:
                    tr_set = set(tr_idx.tolist())
                    edge_np = edge_index.cpu().numpy() if hasattr(edge_index, 'cpu') else np.asarray(edge_index)
                    keep = np.array([s in tr_set and d in tr_set
                                      for s, d in zip(edge_np[0], edge_np[1])])
                    fold_edge = torch.LongTensor(edge_np[:, keep])
                else:
                    fold_edge = edge_index

                fold_detector.fit(X_tr, edge_index=fold_edge, labels=y_tr, device=device)

                probs, _ = fold_detector.predict_anomaly_probability(X_va, device=device)
                preds = (probs >= 0.5).astype(int)
                prec_c, rec_c, _ = precision_recall_curve(y_va, probs)
                pr_auc = float(auc(rec_c, prec_c)) if len(prec_c) > 1 else 0.0
                try:
                    roc_auc = float(roc_auc_score(y_va, probs))
                except Exception:
                    roc_auc = float('nan')
                brier = float(brier_score_loss(y_va, probs))
                f1 = float(f1_score(y_va, preds, zero_division=0))

                per_fold.append({
                    'fold': fold_idx,
                    'n_train': len(tr_idx),
                    'n_val': len(va_idx),
                    'pr_auc': pr_auc,
                    'roc_auc': roc_auc,
                    'brier': brier,
                    'f1_at_0.5': f1,
                })
                pr_aucs.append(pr_auc); roc_aucs.append(roc_auc)
                briers.append(brier); f1s.append(f1)
            except Exception as e:
                logger.warning("Fold %d failed: %s", fold_idx, e)
                per_fold.append({'fold': fold_idx, 'error': str(e)})

        def _summary(arr):
            if not arr:
                return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
            a = np.asarray(arr, dtype=float)
            # Ignore NaN for summary stats
            a = a[~np.isnan(a)]
            if a.size == 0:
                return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
            return {'mean': float(a.mean()), 'std': float(a.std()), 'n': int(a.size)}

        summary = {
            'pr_auc': _summary(pr_aucs),
            'roc_auc': _summary(roc_aucs),
            'brier': _summary(briers),
            'f1_at_0.5': _summary(f1s),
        }
        result = {'per_fold': per_fold, 'summary': summary,
                  'n_splits': int(n_splits)}

        if refit:
            try:
                self.fit(features, edge_index=edge_index, labels=labels_np, device=device)
            except Exception as e:
                logger.warning("Refit on full data after CV failed: %s", e)
        return result
    
    def save_models(self, path_prefix, training_metadata=None):
        """Save all models.

        Files are written to local disk first (so an in-flight load on
        the same instance keeps working), then mirrored to Google Cloud
        Storage when the GCS adapter is enabled (R4 — controlled by
        ``GOOGLE_CLOUD_BUCKET`` env var).
        """
        import os as _os
        # Local import keeps the top of the file clean and avoids
        # import-time errors when google-cloud-storage is absent.
        import cloud_storage
        _os.makedirs(_os.path.dirname(path_prefix), exist_ok=True)

        if training_metadata is not None:
            self.training_metadata = training_metadata

        artefacts: list[str] = []

        # Save Isolation Forest
        p_if = f"{path_prefix}_isolation_forest.pkl"
        joblib.dump(self.isolation_forest, p_if); artefacts.append(p_if)

        # Save Autoencoder
        if self.autoencoder is not None:
            p_ae = f"{path_prefix}_autoencoder.pt"
            torch.save(self.autoencoder.state_dict(), p_ae); artefacts.append(p_ae)

        # Save XGBoost
        if self.xgboost_model is not None:
            p_xgb = f"{path_prefix}_xgboost.pkl"
            self.xgboost_model.save_model(p_xgb); artefacts.append(p_xgb)

        # Save GNN
        if self.gnn_model is not None:
            p_gnn = f"{path_prefix}_gnn.pt"
            torch.save(self.gnn_model.state_dict(), p_gnn); artefacts.append(p_gnn)

        # Save DBSCAN
        if self.dbscan is not None:
            p_db = f"{path_prefix}_dbscan.pkl"
            joblib.dump(self.dbscan, p_db); artefacts.append(p_db)

        # Save imputer and scaler
        p_imp = f"{path_prefix}_imputer.pkl"
        p_scl = f"{path_prefix}_scaler.pkl"
        joblib.dump(self.imputer, p_imp); artefacts.append(p_imp)
        joblib.dump(self.scaler, p_scl); artefacts.append(p_scl)

        params = {
            'autoencoder_threshold': self.autoencoder_threshold,
            'isolation_weight': self.isolation_weight,
            'autoencoder_weight': self.autoencoder_weight,
            'dbscan_weight': self.dbscan_weight,
            'xgboost_weight': self.xgboost_weight,
            'gnn_weight': self.gnn_weight,
            'autoencoder_params': self.autoencoder_params,
            'xgboost_params': self.xgboost_params,
            'gnn_params': self.gnn_params,
            'dbscan_params': self.dbscan_params,
            'algorithms': self.algorithms,
            'training_metadata': self.training_metadata,
            'gnn_architecture': {
                'num_layers': self.gnn_params.get('num_layers', 1),
                'dropout': self.gnn_params.get('dropout', 0.2),
                'hidden_channels': self.gnn_params.get('hidden_channels', 64),
                'num_heads': self.gnn_params.get('num_heads', 4),
                'edge_dim': getattr(self.gnn_model, 'edge_dim', None),
            },
            # QW7: feature schema — list of column names used at training
            # time, so the detection page can validate the inference
            # data against the trained schema.
            'feature_columns': list(self.training_metadata.get('feature_columns', []))
                              if isinstance(self.training_metadata, dict) else [],
            'feature_dtypes': dict(self.training_metadata.get('feature_dtypes', {}))
                              if isinstance(self.training_metadata, dict) else {},
        }
        p_params = f"{path_prefix}_params.json"
        
        def default_serializer(obj):
            import numpy as np
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if callable(obj):
                return None
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(p_params, 'w') as f:
            json.dump(params, f, indent=4, default=default_serializer)
        artefacts.append(p_params)

        # R4: mirror everything to GCS (no-op when env var is not set).
        cloud_storage.sync_artefacts_after_save(artefacts)
        return artefacts

    def load_models(self, path_prefix, num_features=None, num_classes=None, device='cpu'):
        """Load all models.

        On Cloud Run (or anywhere GCS mirroring is enabled) the adapter
        is asked to populate the local cache directory first so that the
        rest of the load logic can stay filesystem-based.
        """
        import cloud_storage
        local_dir = os.path.dirname(path_prefix) or '.'
        prefix_name = os.path.basename(path_prefix)
        basenames = [
            f'{prefix_name}_params.json', f'{prefix_name}_isolation_forest.pkl',
            f'{prefix_name}_autoencoder.pt', f'{prefix_name}_xgboost.pkl',
            f'{prefix_name}_gnn.pt', f'{prefix_name}_dbscan.pkl',
            f'{prefix_name}_imputer.pkl', f'{prefix_name}_scaler.pkl',
        ]
        cloud_storage.ensure_artefacts_loaded(local_dir, basenames)

        with open(f"{path_prefix}_params.json", 'r') as f:
            params = json.load(f)

        self.autoencoder_threshold = params.get('autoencoder_threshold')
        self.isolation_weight = params.get('isolation_weight', 0.3)
        self.autoencoder_weight = params.get('autoencoder_weight', 0.3)
        self.dbscan_weight = params.get('dbscan_weight', 0.0)
        self.xgboost_weight = params.get('xgboost_weight', 0.4)
        self.gnn_weight = params.get('gnn_weight', 0.0)
        self.autoencoder_params = params.get('autoencoder_params', {})
        self.xgboost_params = params.get('xgboost_params', {})
        self.gnn_params = params.get('gnn_params', {})
        self.dbscan_params = params.get('dbscan_params', {})
        self.algorithms = params.get('algorithms', ['isolation_forest', 'autoencoder', 'xgboost'])
        self.training_metadata = params.get('training_metadata', {})

        # Load Isolation Forest
        self.isolation_forest = joblib.load(f"{path_prefix}_isolation_forest.pkl")
        
        # Load Autoencoder
        if os.path.exists(f"{path_prefix}_autoencoder.pt"):
            input_dim = num_features or 64  # Default, should be set properly
            encoding_dim = self.autoencoder_params.get('encoding_dim', 32)
            self.autoencoder = ClaimAnomalyAutoencoder(
                input_dim=input_dim,
                encoding_dim=encoding_dim,
                hidden_dims=self.autoencoder_params.get('hidden_dims', [64, 48])
            ).to(device)
            self.autoencoder.load_state_dict(torch.load(f"{path_prefix}_autoencoder.pt", map_location=device))
        
        # Load XGBoost
        if os.path.exists(f"{path_prefix}_xgboost.pkl"):
            self.xgboost_model = ClaimAnomalyXGBoostModel(
                model_type=self.xgboost_params.get('model_type', 'xgboost'),
                **{k: v for k, v in self.xgboost_params.items() if k != 'model_type'}
            )
            self.xgboost_model.load_model(f"{path_prefix}_xgboost.pkl")

        # Load GNN
        if os.path.exists(f"{path_prefix}_gnn.pt"):
            input_dim = num_features or 64
            num_classes = num_classes or 2
            self.gnn_model = InsuranceAnomalyGNNModel(
                num_features=input_dim,
                num_classes=num_classes,
                hidden_channels=self.gnn_params.get('hidden_channels', 64),
                num_heads=self.gnn_params.get('num_heads', 4),
                dropout=self.gnn_params.get('dropout', 0.2),
                num_layers=self.gnn_params.get(
                    'num_layers',
                    params.get('gnn_architecture', {}).get('num_layers', 1),
                ),
                edge_dim=params.get('gnn_architecture', {}).get('edge_dim'),
            ).to(device)
            self.gnn_model.load_state_dict(torch.load(f"{path_prefix}_gnn.pt", map_location=device))

        # Load DBSCAN
        if os.path.exists(f"{path_prefix}_dbscan.pkl"):
            self.dbscan = joblib.load(f"{path_prefix}_dbscan.pkl")
        
        # Load imputer and scaler
        # Check if imputer file exists (for backward compatibility with old models)
        if os.path.exists(f"{path_prefix}_imputer.pkl"):
            self.imputer = joblib.load(f"{path_prefix}_imputer.pkl")
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")

@st.cache_data(ttl=1800, max_entries=5)
def create_claim_graph(df, feature_columns, method='star', max_nodes=20000, max_edges=200000, **kwargs):
    """
    Dispatcher for graph construction. Supports:
    - 'star'  : original star-topology based on shared categorical IDs (default)
    - 'knn'   : k-nearest-neighbor graph over node features
    - 'heterogeneous' : multi-relational graph (provider / patient / diagnosis)
                        with edge-type one-hot appended to node features

    For 'star' / 'heterogeneous' the DataFrame must contain at least one
    of: provider_id, patient_id, diagnosis_code. k-NN only needs the
    feature matrix.
    """
    method = (method or 'star').lower()
    if method == 'knn':
        if isinstance(df, pd.DataFrame):
            feats = df[feature_columns].values
        else:
            feats = df
        return create_knn_graph(feats, max_nodes=max_nodes, **kwargs)
    if method == 'heterogeneous':
        if not isinstance(df, pd.DataFrame):
            raise ValueError("heterogeneous graph requires a pandas DataFrame")
        return create_heterogeneous_graph(df, feature_columns, max_nodes=max_nodes, max_edges=max_edges, **kwargs)

    # Default: original star-topology
    try:
        # Create nodes (claims)
        if isinstance(df, pd.DataFrame):
            node_features = df[feature_columns].values[:max_nodes]
            df = df.iloc[:max_nodes].reset_index(drop=True)

            # Identify columns for relationships
            original_cols = ['provider_id', 'patient_id', 'diagnosis_code']
            available_original_cols = [col for col in original_cols if col in df.columns]

            if len(available_original_cols) < 1:
                return create_similarity_graph(node_features)
        else:
            node_features = df[:max_nodes]
            return create_similarity_graph(node_features)

        edge_indices = []

        # Optimized edge creation using Star Topology
        # Instead of connecting every pair (N^2), we connect every node to the first node in the group (N)
        for col in available_original_cols:
            groups = df.groupby(col).groups
            for val, indices in groups.items():
                if len(indices) > 1:
                    # Star topology: connect all members to the first member
                    center_node = indices[0]
                    for other_node in indices[1:]:
                        if len(edge_indices) >= max_edges:
                            break
                        edge_indices.append([center_node, other_node])
                        edge_indices.append([other_node, center_node])
                    if len(edge_indices) >= max_edges:
                        break
            if len(edge_indices) >= max_edges:
                break

        # If no edges created, create similarity graph
        if not edge_indices:
            return create_similarity_graph(node_features)

        edge_index = torch.LongTensor(edge_indices).t().contiguous()
        return node_features, edge_index

    except Exception as e:
        logger.error("Error creating optimized graph: %s", e)
        return create_similarity_graph(df[feature_columns].values if isinstance(df, pd.DataFrame) else df)


def create_knn_graph(node_features, k=5, max_nodes=20000, metric='euclidean'):
    """Build a k-nearest-neighbor graph using FAISS (if available) or sklearn NearestNeighbors.

    For very large datasets we use FAISS to keep memory bounded and speed up construction.
    Returns (node_features, edge_index) with edges in both directions.
    """
    try:
        n_nodes = min(node_features.shape[0], int(max_nodes))
        node_features = node_features[:n_nodes]
        if n_nodes < 2:
            edge_index = torch.LongTensor([[0], [0]]).t().contiguous()
            return node_features, edge_index
            
        k_eff = max(1, min(int(k), n_nodes - 1))
        
        try:
            import faiss
            # FAISS expects float32 arrays
            feats_f32 = np.ascontiguousarray(node_features, dtype=np.float32)
            d = feats_f32.shape[1]
            
            # IndexFlatL2 is exact search for L2 (Euclidean) distance
            index = faiss.IndexFlatL2(d)
            index.add(feats_f32)
            distances, indices = index.search(feats_f32, k_eff + 1)
        except ImportError:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, algorithm='auto')
            nn.fit(node_features)
            distances, indices = nn.kneighbors(node_features)
            
        # Skip self and build undirected edges
        src_list, dst_list = [], []
        for i in range(n_nodes):
            for j in indices[i, 1:]:
                if j != -1 and j != i:  # FAISS returns -1 if not enough neighbors
                    src_list.append(i)
                    dst_list.append(int(j))
                    src_list.append(int(j))
                    dst_list.append(i)
                    
        edge_index = torch.LongTensor([src_list, dst_list]).contiguous()
        return node_features, edge_index
    except Exception as e:
        logger.error("Error creating k-NN graph: %s. Falling back to similarity graph.", e)
        return create_similarity_graph(node_features)


def create_heterogeneous_graph(df, feature_columns,
                               provider_col='provider_id',
                               patient_col='patient_id',
                               diagnosis_col='diagnosis_code',
                               edge_type_dim=3,
                               max_nodes=20000,
                               max_edges=200000):
    """Build a heterogeneous graph with explicit edge types.

    Each shared attribute (provider, patient, diagnosis) generates a
    star-topology edge set. Edge-type one-hot is appended to the node
    feature matrix so the GNN can distinguish the relation.

    Returns (node_features, edge_index, edge_type) where edge_type is
    an [num_edges] LongTensor with values in {0,1,2}.
    """
    try:
        df = df.iloc[:max_nodes].reset_index(drop=True)
        node_features = df[feature_columns].values.astype(np.float32)
        n_nodes = node_features.shape[0]

        cols = []
        if provider_col in df.columns:
            cols.append((0, provider_col))
        if patient_col in df.columns:
            cols.append((1, patient_col))
        if diagnosis_col in df.columns:
            cols.append((2, diagnosis_col))

        if not cols:
            # Fallback to similarity
            return create_similarity_graph(node_features)

        # Edge-type one-hot
        edge_type_oh = np.zeros((n_nodes, edge_type_dim), dtype=np.float32)
        src, dst, etype = [], [], []
        for etype_id, col in cols:
            groups = df.groupby(col).groups
            for _, indices in groups.items():
                if len(indices) > 1:
                    center = indices[0]
                    for other in indices[1:]:
                        if len(src) >= max_edges:
                            break
                        src.append(int(center)); dst.append(int(other)); etype.append(etype_id)
                        src.append(int(other)); dst.append(int(center)); etype.append(etype_id)
                    if len(src) >= max_edges:
                        break
            if len(src) >= max_edges:
                break
            # Mark participation: any node appearing in this relation
            members = np.unique(np.concatenate([np.asarray(list(g)) for g in groups.values()]))
            edge_type_oh[members, etype_id] = 1.0

        if not src:
            return create_similarity_graph(node_features)

        node_features = np.concatenate([node_features, edge_type_oh], axis=1)
        edge_index = torch.LongTensor([src, dst]).contiguous()
        edge_type = torch.LongTensor(etype)
        return node_features, edge_index, edge_type
    except Exception as e:
        logger.error("Error creating heterogeneous graph: %s. Falling back to star graph.", e)
        return create_claim_graph(df, feature_columns, method='star')


def create_similarity_graph(node_features, similarity_threshold=0.8, max_edges=1000, sample_size=5000):
    """
    Optimized similarity graph creation.
    For large datasets, it uses sampling to avoid OOM and O(N^2) complexity.
    """
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        n_nodes = node_features.shape[0]
        if n_nodes > sample_size:
            logger.info("Dataset too large for full similarity matrix (%d nodes). Using simple chain graph.", n_nodes)
            edge_indices = []
            for i in range(min(n_nodes - 1, max_edges)):
                edge_indices.append([i, i + 1])
                edge_indices.append([i + 1, i])
            edge_index = torch.LongTensor(edge_indices).t().contiguous()
            return node_features, edge_index

        # Calculate similarity matrix for small enough datasets
        similarities = cosine_similarity(node_features)

        # Vectorized edge creation for better performance
        edge_indices = []

        # Create upper triangular mask to avoid duplicate edges
        mask = np.triu(np.ones_like(similarities), k=1).astype(bool)

        # Find all pairs above threshold (vectorized)
        i_indices, j_indices = np.where((similarities > similarity_threshold) & mask)

        # Limit to max_edges (convert to pairs and add reverse edges)
        max_pairs = max_edges // 2
        if len(i_indices) > max_pairs:
            # Take top similarities if we have too many
            similarity_values = similarities[i_indices, j_indices]
            top_indices = np.argsort(similarity_values)[-max_pairs:]
            i_indices = i_indices[top_indices]
            j_indices = j_indices[top_indices]

        # Create bidirectional edges
        for i, j in zip(i_indices, j_indices):
            edge_indices.append([i, j])
            edge_indices.append([j, i])

        # If still no edges, create a simple chain
        if not edge_indices:
            for i in range(min(n_nodes-1, 100)):
                edge_indices.append([i, i+1])
                edge_indices.append([i+1, i])

        edge_index = torch.LongTensor(edge_indices).t().contiguous() if edge_indices else torch.LongTensor([[0], [0]])
        return node_features, edge_index
        
    except Exception as e:
        logger.error("Error creating similarity graph: %s", e)
        # Ultimate fallback: create minimal graph
        n_nodes = node_features.shape[0]
        if n_nodes > 1:
            edge_index = torch.LongTensor([[0, 1], [1, 0]]).t().contiguous()
        else:
            edge_index = torch.LongTensor([[0], [0]]).t().contiguous()
        return node_features, edge_index

def analyze_anomaly_networks(df, fraud_predictions):
    """Analyze networks of anomaly transactions"""
    try:
        fraud_df = df[fraud_predictions == 1]
        
        # Provider anomaly analysis
        provider_fraud_rates = fraud_df.groupby('provider_id').size().sort_values(ascending=False)
        
        # Diagnosis anomaly patterns
        diagnosis_fraud_patterns = fraud_df['diagnosis_code'].value_counts().head(10)
        
        # Service type anomaly analysis
        service_fraud_patterns = fraud_df['service_type'].value_counts().head(10)
        
        # Amount analysis for anomaly vs legitimate
        fraud_amounts = fraud_df['billed_amount']
        legitimate_amounts = df[fraud_predictions == 0]['billed_amount']
        
        analysis_results = {
            'total_anomaly_claims': len(fraud_df),
            'anomaly_rate': len(fraud_df) / len(df),
            'top_anomaly_providers': provider_fraud_rates.head(10).to_dict(),
            'top_anomaly_diagnoses': diagnosis_fraud_patterns.to_dict(),
            'top_anomaly_services': service_fraud_patterns.to_dict(),
            'anomaly_amount_stats': {
                'mean': fraud_amounts.mean(),
                'median': fraud_amounts.median(),
                'std': fraud_amounts.std()
            },
            'legitimate_amount_stats': {
                'mean': legitimate_amounts.mean(),
                'median': legitimate_amounts.median(),
                'std': legitimate_amounts.std()
            }
        }
        
        return analysis_results
    except Exception as e:
        logger.error("Error analyzing anomaly networks: %s", e)
        return {
            'total_anomaly_claims': 0,
            'anomaly_rate': 0,
            'top_anomaly_providers': {},
            'top_anomaly_diagnoses': {},
            'top_anomaly_services': {},
            'anomaly_amount_stats': {'mean': 0, 'median': 0, 'std': 0},
            'legitimate_amount_stats': {'mean': 0, 'median': 0, 'std': 0}
        }
