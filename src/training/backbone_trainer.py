"""
Stage 1 backbone trainer — two-phase fine-tuning strategy.

Phase A: All backbone layers frozen, train classification head only.
Phase B: Unfreeze one backbone layer (default: block5_conv3), fine-tune.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.training.callbacks import make_callbacks


def train(model: Model, train_gen, val_gen, cfg: dict,
          paths: dict) -> tuple:
    """
    Two-phase training.

    Args:
        model:     Full classification model (backbone + head).
        train_gen: Keras training generator.
        val_gen:   Keras validation generator.
        cfg:       Full experiment config dict.
        paths:     Experiment paths dict from setup_experiment_dir().

    Returns:
        (hist_a, hist_b) — history dicts for Phase A and Phase B.
    """
    stage1_cfg    = cfg["training"]["stage1"]
    phase_a_cfg   = stage1_cfg["phase_a"]
    phase_b_cfg   = stage1_cfg["phase_b"]
    unfreeze_layer = phase_b_cfg.get("unfreeze_layer", "block5_conv3")
    ckpt_path     = f"{paths['checkpoints']}/stage1_best.keras"

    # ── Phase A: freeze all backbone layers ──────────────────────────────────
    print("\n" + "═" * 55)
    print("  PHASE A — Freeze backbone, train head only")
    print("═" * 55)

    backbone_layers = _get_backbone_layer_names(model)
    for layer in model.layers:
        layer.trainable = layer.name not in backbone_layers

    model.compile(
        optimizer=Adam(float(phase_a_cfg["lr"])),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist_a = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase_a_cfg["epochs"],
        callbacks=make_callbacks(ckpt_path, patience_early_stop=8, patience_lr=4),
        verbose=1,
    ).history

    # ── Phase B: unfreeze one backbone layer ─────────────────────────────────
    print("\n" + "═" * 55)
    print(f"  PHASE B — Unfreeze {unfreeze_layer}, fine-tune")
    print("═" * 55)

    for layer in model.layers:
        if layer.name in backbone_layers:
            layer.trainable = (layer.name == unfreeze_layer)

    model.compile(
        optimizer=Adam(float(phase_b_cfg["lr"])),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist_b = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase_b_cfg["epochs"],
        callbacks=make_callbacks(ckpt_path, patience_early_stop=8, patience_lr=4),
        verbose=1,
    ).history

    return hist_a, hist_b


def _get_backbone_layer_names(model: Model) -> set:
    """
    Infer backbone layer names as all layers that are NOT part of the
    custom head (layers after GlobalAveragePooling2D).
    Works for any backbone since the head always starts at 'gap'.
    """
    backbone_layers = set()
    in_backbone = True
    for layer in model.layers:
        if layer.name == "gap":
            in_backbone = False
        if in_backbone:
            backbone_layers.add(layer.name)
    return backbone_layers
