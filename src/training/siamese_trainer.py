"""
Stage 2 Siamese network trainer.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.training.callbacks import make_callbacks


def train_siamese(siamese: Model, X_train: list, y_train: np.ndarray,
                  X_val: list, y_val: np.ndarray,
                  cfg: dict, paths: dict) -> dict:
    """
    Compile and train the Siamese network.

    Args:
        siamese:  Siamese model (two inputs → sigmoid output).
        X_train:  [img1_train, img2_train]
        y_train:  Labels array (0=similar, 1=not similar)
        X_val:    [img1_val, img2_val]
        y_val:    Validation labels
        cfg:      Full experiment config dict.
        paths:    Experiment paths dict from setup_experiment_dir().

    Returns:
        History dict.
    """
    stage2_cfg = cfg["training"]["stage2"]
    ckpt_path  = f"{paths['checkpoints']}/stage2_best.keras"

    siamese.compile(
        optimizer=Adam(float(stage2_cfg["lr"])),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    siamese.summary()

    print(f"\n  Training samples  : {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Similar (0)       : {int(np.sum(y_train == 0))} train / {int(np.sum(y_val == 0))} val")
    print(f"  Not Similar (1)   : {int(np.sum(y_train == 1))} train / {int(np.sum(y_val == 1))} val")

    history = siamese.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=stage2_cfg["batch_size"],
        epochs=stage2_cfg["epochs"],
        callbacks=make_callbacks(ckpt_path),
        verbose=1,
    )
    return history.history
