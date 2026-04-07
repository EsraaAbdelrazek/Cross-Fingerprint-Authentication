"""
Core evaluation metrics for biometric authentication.
"""

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Compute Equal Error Rate (EER) — the threshold where FAR == FRR.
    Lower EER = better authentication system.

    Args:
        y_true:   Ground truth labels (0=similar, 1=not similar).
        y_scores: Model's predicted scores (sigmoid output).

    Returns:
        (eer, threshold)  — both as floats in [0, 1].
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    threshold = float(thresholds[eer_idx])
    return eer, threshold
