"""
Stage 1 backbone trainer — two-phase fine-tuning strategy.

Phase A: All backbone layers frozen, train classification head only.
Phase B: Unfreeze one backbone layer (default: block5_conv3), fine-tune.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.training.callbacks import make_callbacks


def train(model: Model, train_gen, val_gen, cfg: dict,
          paths: dict, fold_suffix: str = "") -> tuple:
    """
    Two-phase training.

    Args:
        model:       Full classification model (backbone + head).
        train_gen:   Keras training generator.
        val_gen:     Keras validation generator.
        cfg:         Full experiment config dict.
        paths:       Experiment paths dict from setup_experiment_dir().
        fold_suffix: Optional suffix for checkpoint filename (e.g. '_fold1').

    Returns:
        (hist_a, hist_b) — history dicts for Phase A and Phase B.
    """
    stage1_cfg    = cfg["training"]["stage1"]
    phase_a_cfg   = stage1_cfg["phase_a"]
    phase_b_cfg   = stage1_cfg["phase_b"]
    unfreeze_layer = phase_b_cfg.get("unfreeze_layer", "block5_conv3")
    ckpt_path     = f"{paths['checkpoints']}/stage1_best{fold_suffix}.keras"

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

    # ── Phase B: unfreeze backbone from a given layer onwards ───────────────
    print("\n" + "═" * 55)
    print(f"  PHASE B — Unfreeze from {unfreeze_layer} onwards, fine-tune")
    print("═" * 55)

    backbone_layer_list = [l.name for l in model.layers if l.name in backbone_layers]
    try:
        unfreeze_idx = backbone_layer_list.index(unfreeze_layer)
    except ValueError:
        raise ValueError(
            f"unfreeze_layer '{unfreeze_layer}' not found in backbone. "
            f"Available backbone layers: {backbone_layer_list}"
        )
    layers_to_unfreeze = set(backbone_layer_list[unfreeze_idx:])

    for layer in model.layers:
        if layer.name in backbone_layers:
            layer.trainable = layer.name in layers_to_unfreeze

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


def train_kfold(model_builder, folds: list, cfg: dict, paths: dict) -> tuple:
    """
    Run k-fold cross-validation training.

    Args:
        model_builder: Zero-argument callable that returns a fresh compiled model.
        folds:         List of (train_gen, val_gen) from create_kfold_generators().
        cfg:           Full experiment config dict.
        paths:         Experiment paths dict.

    Returns:
        (fold_results, best_model) where fold_results is a list of dicts
        with keys: fold, hist_a, hist_b, best_val_acc.
    """
    fold_results = []
    best_val_acc = -1.0
    best_model   = None

    for fold_idx, (train_gen, val_gen) in enumerate(folds):
        print(f"\n{'═' * 55}")
        print(f"  FOLD {fold_idx + 1}/{len(folds)}")
        print(f"  Train: {train_gen.n} samples  |  Val: {val_gen.n} samples")
        print(f"{'═' * 55}")

        model    = model_builder()
        suffix   = f"_fold{fold_idx + 1}"
        hist_a, hist_b = train(model, train_gen, val_gen, cfg, paths,
                               fold_suffix=suffix)
        fold_best = max(hist_b["val_accuracy"])
        fold_results.append({
            "fold":         fold_idx + 1,
            "hist_a":       hist_a,
            "hist_b":       hist_b,
            "best_val_acc": fold_best,
        })
        print(f"  Fold {fold_idx + 1} best val accuracy: {fold_best * 100:.2f}%")

        if fold_best > best_val_acc:
            best_val_acc = fold_best
            best_model   = model

    accs = [r["best_val_acc"] for r in fold_results]
    print(f"\n{'═' * 55}")
    print(f"  K-FOLD SUMMARY  ({len(folds)} folds)")
    for r in fold_results:
        print(f"    Fold {r['fold']}: {r['best_val_acc'] * 100:.2f}%")
    print(f"  Mean ± Std : {sum(accs)/len(accs)*100:.2f}% ± "
          f"{(sum((a - sum(accs)/len(accs))**2 for a in accs)/len(accs))**0.5 * 100:.2f}%")
    print(f"  Best fold  : {max(fold_results, key=lambda x: x['best_val_acc'])['fold']} "
          f"({best_val_acc * 100:.2f}%)")
    print(f"{'═' * 55}")

    return fold_results, best_model


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
