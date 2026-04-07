"""
Stage 2 Siamese network evaluation — accuracy, EER, and single-pair inference.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model

from src.evaluation.metrics import compute_eer


def evaluate_siamese(siamese: Model, X_test: list,
                     y_test: np.ndarray) -> tuple:
    """
    Full evaluation: accuracy at EER threshold + classification report.

    Args:
        siamese: Trained Siamese model.
        X_test:  [img1_test, img2_test]
        y_test:  Ground truth labels (0=similar, 1=not similar).

    Returns:
        (accuracy, eer) as floats.
    """
    print("\n" + "═" * 55)
    print("  STAGE 2 EVALUATION")
    print("═" * 55)

    scores = siamese.predict(X_test, batch_size=32, verbose=1).flatten()
    eer, eer_thresh = compute_eer(y_test, scores)
    preds = (scores >= eer_thresh).astype(int)
    acc = accuracy_score(y_test, preds)

    print(f"\n  Accuracy (at EER threshold) : {acc * 100:.2f}%")
    print(f"  EER                         : {eer:.4f}  ({eer * 100:.2f}%)")
    print(f"  EER threshold               : {eer_thresh:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, preds,
        target_names=["Similar (0)", "Not Similar (1)"],
    ))
    return acc, eer


def predict_pair(siamese: Model, img_path1: str, img_path2: str,
                 img_size: tuple = (224, 224),
                 threshold: float = 0.5) -> dict:
    """
    Predict whether two fingerprint images belong to the same person.

    Args:
        siamese:    Trained Siamese model.
        img_path1:  Path to first fingerprint image.
        img_path2:  Path to second fingerprint image.
        img_size:   Resize target (must match training size).
        threshold:  Decision boundary (default 0.5; use EER threshold for best results).

    Returns:
        {
            "score":      float — raw sigmoid output (0=similar, 1=different),
            "prediction": "Similar" or "Not Similar",
            "confidence": float — confidence in the prediction,
        }
    """
    def _load(path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
        return tf.keras.preprocessing.image.img_to_array(img)[np.newaxis] / 255.0

    score = float(siamese.predict(
        [_load(img_path1), _load(img_path2)], verbose=0
    )[0, 0])

    return {
        "score":      round(score, 4),
        "prediction": "Not Similar" if score >= threshold else "Similar",
        "confidence": round(score if score >= threshold else 1 - score, 4),
    }
