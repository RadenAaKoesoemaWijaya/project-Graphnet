"""
Fraud Detection Evaluation Metrics

Provides specialized metrics for fraud detection and anomaly detection tasks,
particularly useful for imbalanced datasets.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef
)

logger = logging.getLogger(__name__)


def calculate_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Precision-Recall AUC (Area Under Curve).

    PR-AUC is more informative than ROC-AUC for imbalanced datasets,
    as it focuses on the positive (fraud) class performance.

    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted probabilities or scores

    Returns:
        PR-AUC score (0.0 to 1.0), or 0.0 if calculation fails
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        return float(pr_auc)
    except Exception as e:
        logger.warning(f"Failed to calculate PR-AUC: {e}")
        return 0.0


def calculate_f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F2-Score.

    F2-Score gives more weight to recall than precision (2:1 ratio).
    This is critical for fraud detection where missing a fraud case
    (false negative) is more costly than false positive.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)

    Returns:
        F2-Score (0.0 to 1.0), or 0.0 if calculation fails
    """
    try:
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        return float(f2)
    except Exception as e:
        logger.warning(f"Failed to calculate F2-Score: {e}")
        return 0.0


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Specificity (True Negative Rate).

    Specificity measures the proportion of actual negatives (non-fraud)
    that are correctly identified.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)

    Returns:
        Specificity (0.0 to 1.0), or 0.0 if calculation fails
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(specificity)
    except Exception as e:
        logger.warning(f"Failed to calculate Specificity: {e}")
        return 0.0


def calculate_false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate False Negative Rate (FNR).

    FNR measures the proportion of actual fraud cases that are
    incorrectly classified as non-fraud. This is critical for
    fraud detection as it represents fraud that slips through.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)

    Returns:
        FNR (0.0 to 1.0), or 0.0 if calculation fails
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return float(fnr)
    except Exception as e:
        logger.warning(f"Failed to calculate FNR: {e}")
        return 0.0


def calculate_false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate False Positive Rate (FPR).

    FPR measures the proportion of actual non-fraud cases that are
    incorrectly classified as fraud.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)

    Returns:
        FPR (0.0 to 1.0), or 0.0 if calculation fails
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(fpr)
    except Exception as e:
        logger.warning(f"Failed to calculate FPR: {e}")
        return 0.0


def calculate_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC).

    MCC is a balanced measure that takes all four confusion matrix
    values into account. It's particularly useful for imbalanced
    datasets and ranges from -1 to +1, where +1 is perfect prediction.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)

    Returns:
        MCC (-1.0 to 1.0), or 0.0 if calculation fails
    """
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
        return float(mcc)
    except Exception as e:
        logger.warning(f"Failed to calculate MCC: {e}")
        return 0.0


def calculate_all_fraud_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all fraud detection metrics.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_scores: Predicted probabilities or scores

    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}

    try:
        metrics['pr_auc'] = calculate_pr_auc(y_true, y_scores)
        metrics['f2_score'] = calculate_f2_score(y_true, y_pred)
        metrics['specificity'] = calculate_specificity(y_true, y_pred)
        metrics['false_negative_rate'] = calculate_false_negative_rate(y_true, y_pred)
        metrics['false_positive_rate'] = calculate_false_positive_rate(y_true, y_pred)
        metrics['mcc'] = calculate_mcc(y_true, y_pred)

        # Calculate additional derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

    except Exception as e:
        logger.error(f"Failed to calculate fraud metrics: {e}", exc_info=True)

    return metrics


def get_threshold_tradeoff_analysis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate precision and recall at various thresholds.

    This analysis helps in selecting the optimal threshold based on
    business requirements (e.g., high recall vs high precision).

    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted probabilities or scores
        thresholds: Array of threshold values to evaluate. If None,
                    uses 50 thresholds from 0.0 to 1.0

    Returns:
        Dictionary with keys: 'thresholds', 'precision', 'recall', 'f1', 'f2'
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 50)

    precision_values = []
    recall_values = []
    f1_values = []
    f2_values = []

    try:
        from sklearn.metrics import precision_score, recall_score, f1_score

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)

            try:
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

                precision_values.append(prec)
                recall_values.append(rec)
                f1_values.append(f1)
                f2_values.append(f2)
            except Exception:
                # If calculation fails for a threshold, append 0
                precision_values.append(0.0)
                recall_values.append(0.0)
                f1_values.append(0.0)
                f2_values.append(0.0)

    except Exception as e:
        logger.error(f"Failed to calculate threshold tradeoff analysis: {e}", exc_info=True)
        # Return empty arrays on failure
        return {
            'thresholds': thresholds,
            'precision': np.zeros_like(thresholds),
            'recall': np.zeros_like(thresholds),
            'f1': np.zeros_like(thresholds),
            'f2': np.zeros_like(thresholds)
        }

    return {
        'thresholds': thresholds,
        'precision': np.array(precision_values),
        'recall': np.array(recall_values),
        'f1': np.array(f1_values),
        'f2': np.array(f2_values)
    }


def get_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = 'f2'
) -> Tuple[float, float]:
    """
    Find the optimal threshold that maximizes a given metric.

    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted probabilities or scores
        metric: Metric to optimize ('f1', 'f2', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, optimal_score)
    """
    try:
        tradeoff = get_threshold_tradeoff_analysis(y_true, y_scores)

        if metric == 'f1':
            idx = np.argmax(tradeoff['f1'])
            return float(tradeoff['thresholds'][idx]), float(tradeoff['f1'][idx])
        elif metric == 'f2':
            idx = np.argmax(tradeoff['f2'])
            return float(tradeoff['thresholds'][idx]), float(tradeoff['f2'][idx])
        elif metric == 'precision':
            idx = np.argmax(tradeoff['precision'])
            return float(tradeoff['thresholds'][idx]), float(tradeoff['precision'][idx])
        elif metric == 'recall':
            idx = np.argmax(tradeoff['recall'])
            return float(tradeoff['thresholds'][idx]), float(tradeoff['recall'][idx])
        else:
            logger.warning(f"Unknown metric: {metric}, defaulting to F2")
            idx = np.argmax(tradeoff['f2'])
            return float(tradeoff['thresholds'][idx]), float(tradeoff['f2'][idx])

    except Exception as e:
        logger.error(f"Failed to find optimal threshold: {e}", exc_info=True)
        return 0.5, 0.0
