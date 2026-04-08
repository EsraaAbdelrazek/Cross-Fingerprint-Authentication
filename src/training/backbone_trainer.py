"""
Stage 1 backbone trainer — two-phase fine-tuning strategy.

Phase A: All backbone layers frozen, train classification head only.
Phase B: Unfreeze layers from a given point onwards, fine-tune at lower LR.

Supports resume: if a checkpoint + progress.json exist, training continues
from the saved epoch rather than starting from scratch.
"""

import json
from pathlib import Path

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from src.training.callbacks import make_callbacks


def train(model: Model, train_gen, val_gen, cfg: dict,
          paths: dict, fold_suffix: str = "") -> tuple:
    """
    Two-phase training with resume support.

    If a checkpoint and progress.json exist in paths['checkpoints'], training
    resumes from the saved epoch. Otherwise starts from scratch.

    Returns:
        (hist_a, hist_b) — history dicts for Phase A and Phase B.
    """
    stage1_cfg     = cfg["training"]["stage1"]
    phase_a_cfg    = stage1_cfg["phase_a"]
    phase_b_cfg    = stage1_cfg["phase_b"]
    unfreeze_layer = phase_b_cfg.get("unfreeze_layer", "block5_conv3")
    ckpt_path      = f"{paths['checkpoints']}/stage1_best{fold_suffix}.keras"
    progress_path  = Path(f"{paths['checkpoints']}/progress{fold_suffix}.json")

    backbone_layers = _get_backbone_layer_names(model)

    # ── Resume detection ─────────────────────────────────────────────────────
    progress = _load_progress(progress_path)
    if progress:
        print(f"\n  [Resume] Found checkpoint — continuing from Phase {progress['phase']}, "
              f"epoch {progress['epoch_done'] + 1}")
        model = load_model(ckpt_path)

    # ── Phase A ──────────────────────────────────────────────────────────────
    hist_a = progress.get("hist_a", {}) if progress else {}

    if not progress or progress["phase"] == "A":
        print("\n" + "═" * 55)
        print("  PHASE A — Freeze backbone, train head only")
        print("═" * 55)

        for layer in model.layers:
            layer.trainable = layer.name not in backbone_layers

        model.compile(
            optimizer=Adam(float(phase_a_cfg["lr"])),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        initial_epoch = progress["epoch_done"] if progress and progress["phase"] == "A" else 0
        total_epochs  = phase_a_cfg["epochs"]

        result_a = model.fit(
            train_gen,
            validation_data=val_gen,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            callbacks=make_callbacks(ckpt_path, patience_early_stop=8, patience_lr=4),
            verbose=1,
        )
        # Merge resumed history with new history
        hist_a = _merge_histories(hist_a, result_a.history)

        # Mark Phase A done
        _save_progress(progress_path, phase="B", epoch_done=0, hist_a=hist_a)

        # Reload best checkpoint before Phase B
        model = load_model(ckpt_path)

    # ── Phase B ──────────────────────────────────────────────────────────────
    hist_b = progress.get("hist_b", {}) if progress else {}

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

    initial_epoch_b = progress["epoch_done"] if progress and progress["phase"] == "B" else 0
    total_epochs_b  = phase_b_cfg["epochs"]

    result_b = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=initial_epoch_b,
        epochs=total_epochs_b,
        callbacks=make_callbacks(ckpt_path, patience_early_stop=8, patience_lr=4),
        verbose=1,
    )
    hist_b = _merge_histories(hist_b, result_b.history)

    # Clear progress on successful completion
    if progress_path.exists():
        progress_path.unlink()

    return hist_a, hist_b


def train_kfold(model_builder, folds: list, cfg: dict, paths: dict) -> tuple:
    """Run k-fold cross-validation training."""
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_backbone_layer_names(model: Model) -> set:
    """Return names of all backbone layers (everything before 'gap')."""
    backbone_layers = set()
    in_backbone = True
    for layer in model.layers:
        if layer.name == "gap":
            in_backbone = False
        if in_backbone:
            backbone_layers.add(layer.name)
    return backbone_layers


def _save_progress(path: Path, phase: str, epoch_done: int,
                   hist_a: dict = None, hist_b: dict = None):
    """Save training progress to JSON for resume support."""
    data = {"phase": phase, "epoch_done": epoch_done}
    if hist_a:
        data["hist_a"] = {k: [float(v) for v in vals] for k, vals in hist_a.items()}
    if hist_b:
        data["hist_b"] = {k: [float(v) for v in vals] for k, vals in hist_b.items()}
    with open(path, "w") as f:
        json.dump(data, f)


def _load_progress(path: Path) -> dict:
    """Load progress JSON if it exists and a checkpoint is present."""
    if not path.exists():
        return {}
    ckpt = path.parent / (path.stem.replace("progress", "stage1_best") + ".keras")
    if not ckpt.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _merge_histories(existing: dict, new: dict) -> dict:
    """Concatenate existing history with new history from resumed training."""
    if not existing:
        return new
    merged = {}
    for key in new:
        merged[key] = existing.get(key, []) + list(new[key])
    return merged
