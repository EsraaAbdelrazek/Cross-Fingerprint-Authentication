"""
Unified callback factory for all training stages.
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def make_callbacks(checkpoint_path: str,
                   monitor: str = "val_accuracy",
                   patience_early_stop: int = 10,
                   patience_lr: int = 5,
                   lr_factor: float = 0.5,
                   min_lr: float = 1e-7) -> list:
    """
    Build standard callbacks for a training run.

    Args:
        checkpoint_path:     Where to save the best model weights.
        monitor:             Metric to watch (default: val_accuracy).
        patience_early_stop: Epochs without improvement before stopping.
        patience_lr:         Epochs without improvement before reducing LR.
        lr_factor:           Factor to multiply LR by on plateau.
        min_lr:              Minimum allowed learning rate.

    Returns:
        [EarlyStopping, ReduceLROnPlateau, ModelCheckpoint]
    """
    return [
        EarlyStopping(
            monitor=monitor,
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=lr_factor,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
    ]
