"""
Main training entry point.

Usage:
    python scripts/train.py --config configs/polyu_vgg16_ead.yaml
    python scripts/train.py --config configs/polyu_vgg16_ead.yaml --stage 1
    python scripts/train.py --config configs/polyu_vgg16_ead.yaml --stage 2
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.utils import load_config, setup_experiment_dir, save_history
from src.data import DATASET_REGISTRY
from src.data.pair_generator import generate_pairs, split_pairs
from src.models import get_backbone, get_encoder_layer, get_siamese_head
from src.models.encoder_utils import extract_encoder, load_encoder
from src.training import train, train_kfold, train_siamese
from src.evaluation import evaluate, evaluate_siamese
from src.visualization import (
    plot_phase_a, plot_phase_b, plot_stage1_combined, plot_kfold_summary,
    plot_siamese, plot_eer, plot_full_pipeline,
)


def run_stage1(cfg: dict, paths: dict):
    """Run Stage 1: dataset prep + backbone training + encoder extraction.

    Supports two modes via config:
      - Single split : dataset.sources is null  (original behaviour)
      - Multi-source + k-fold CV : dataset.sources is a list of dicts
    """
    dataset_cls  = DATASET_REGISTRY[cfg["dataset"]["name"]]
    dataset      = dataset_cls(cfg)
    img_size     = tuple(cfg["dataset"]["img_size"])
    num_subjects = cfg["dataset"]["num_subjects"]
    stage1_cfg   = cfg["training"]["stage1"]
    use_kfold    = stage1_cfg.get("cross_validation", False)
    sources      = cfg["dataset"].get("sources")          # None → single-source
    multi_source = sources is not None

    # ── 1. Organise & preprocess ────────────────────────────────────────────
    if multi_source:
        organized = "dataset/fingerprints_all_sources"
        processed = "dataset/fingerprints_all_sources_224"
        dataset.reorganize_multi(sources, organized, num_subjects)
    else:
        organized = "dataset/fingerprints_organized"
        processed = "dataset/fingerprints_224"
        dataset.reorganize(cfg["dataset"]["source_dir"], organized, num_subjects)

    dataset.preprocess(organized, processed, img_size)

    # ── 2. Build backbone factory (called once per fold if using k-fold) ────
    backbone_name = cfg["backbone"]["name"]
    backbone_fn   = get_backbone(backbone_name)

    def model_builder():
        n = len([d for d in Path(processed).iterdir() if d.is_dir()])
        m = backbone_fn(
            num_classes=n,
            dense_units=cfg["backbone"]["dense_units"],
            dropout=cfg["backbone"]["dropout"],
            img_size=img_size,
        )
        return m

    print("\n" + "=" * 55)
    print(f"  STAGE 1: {backbone_name.upper()} Training")
    if multi_source:
        print(f"  Sources  : {len(sources)} (contact + contactless, both sessions)")
    if use_kfold:
        print(f"  Mode     : {stage1_cfg['n_folds']}-fold cross-validation")
    print("=" * 55)

    # ── 3a. K-fold CV path ──────────────────────────────────────────────────
    rescale = None if backbone_name == "efficientnet" else 1.0 / 255
    if use_kfold:
        n_folds = stage1_cfg.get("n_folds", 5)
        folds   = dataset.create_kfold_generators(
            processed, img_size, stage1_cfg["batch_size"], n_folds,
            rescale=rescale,
        )
        fold_results, best_model = train_kfold(model_builder, folds, cfg, paths)

        # Save per-fold histories
        for r in fold_results:
            fold_tag = f"fold{r['fold']}"
            save_history(r["hist_a"], f"{paths['histories']}/hist_a_{fold_tag}.json")
            save_history(r["hist_b"], f"{paths['histories']}/hist_b_{fold_tag}.json")

        # Save best model & extract encoder
        final_path = f"{paths['checkpoints']}/stage1_final.keras"
        best_model.save(final_path)
        print(f"  Best fold model saved → {final_path}")

        encoder_layer = get_encoder_layer(backbone_name)
        encoder_path  = f"{paths['checkpoints']}/encoder.keras"
        extract_encoder(best_model, encoder_path, encoder_layer)

        # Summary stats
        accs = [r["best_val_acc"] for r in fold_results]
        mean_acc = np.mean(accs)
        print(f"  CV mean val accuracy: {mean_acc * 100:.2f}%")

        # Plots
        plot_kfold_summary(fold_results, f"{paths['plots']}/kfold_summary.png")

        # Return best fold histories for pipeline plot
        best_r = max(fold_results, key=lambda x: x["best_val_acc"])
        return best_r["hist_a"], best_r["hist_b"]

    # ── 3b. Single-split path (original behaviour) ──────────────────────────
    # EfficientNet has a built-in rescaling layer — do not apply rescale=1/255
    rescale = None if backbone_name == "efficientnet" else 1.0 / 255
    train_gen, val_gen = dataset.create_generators(
        processed, img_size,
        stage1_cfg["batch_size"],
        cfg["dataset"]["val_split"],
        rescale=rescale,
    )

    num_classes = train_gen.num_classes
    print(f"\n  Subjects (classes)  : {num_classes}")
    print(f"  Training samples    : {train_gen.samples}")
    print(f"  Validation samples  : {val_gen.samples}")

    model = backbone_fn(
        num_classes=num_classes,
        dense_units=cfg["backbone"]["dense_units"],
        dropout=cfg["backbone"]["dropout"],
        img_size=img_size,
    )
    model.summary()

    hist_a, hist_b = train(model, train_gen, val_gen, cfg, paths)

    save_history(hist_a, f"{paths['histories']}/hist_a.json")
    save_history(hist_b, f"{paths['histories']}/hist_b.json")

    final_path = f"{paths['checkpoints']}/stage1_final.keras"
    model.save(final_path)
    print(f"  Final model saved → {final_path}")

    encoder_layer = get_encoder_layer(backbone_name)
    encoder_path  = f"{paths['checkpoints']}/encoder.keras"
    extract_encoder(model, encoder_path, encoder_layer)

    evaluate(model, val_gen)

    plot_phase_a(hist_a, f"{paths['plots']}/phase_a_history.png")
    plot_phase_b(hist_b, f"{paths['plots']}/phase_b_history.png",
                 phase_a_best=max(hist_a["val_accuracy"]))
    plot_stage1_combined(hist_a, hist_b, f"{paths['plots']}/combined_stage1_history.png")

    return hist_a, hist_b


def run_stage2(cfg: dict, paths: dict):
    """Run Stage 2: load encoder + Siamese training + evaluation."""
    dataset_cls = DATASET_REGISTRY[cfg["dataset"]["name"]]
    dataset = dataset_cls(cfg)

    img_size  = tuple(cfg["dataset"]["img_size"])
    processed = "dataset/fingerprints_224"

    # 1. Load encoder from Stage 1
    encoder_path = f"{paths['checkpoints']}/encoder.keras"
    encoder = load_encoder(encoder_path)

    # 2. Load images into memory
    images = dataset.load_images(processed, img_size)

    # 3. Generate pairs
    stage2_cfg = cfg["training"]["stage2"]
    print("\n  Generating training pairs ...")
    img1, img2, labels = generate_pairs(images, stage2_cfg["num_pairs"], seed=42)
    X_train, y_train, X_val, y_val = split_pairs(
        img1, img2, labels, cfg["dataset"]["val_split"]
    )

    print("\n  Generating test pairs ...")
    t_img1, t_img2, t_labels = generate_pairs(images, stage2_cfg["test_pairs"], seed=99)
    X_test = [t_img1, t_img2]
    y_test = t_labels

    # 4. Build Siamese network
    head_name = cfg["siamese_head"]["name"]
    head_fn   = get_siamese_head(head_name)
    siamese   = head_fn(encoder, cfg)

    # 5. Train
    print("\n" + "=" * 55)
    print(f"  STAGE 2: Siamese ({head_name.upper()}) Training")
    print("=" * 55)
    history = train_siamese(siamese, X_train, y_train, X_val, y_val, cfg, paths)

    # 6. Save history
    save_history(history, f"{paths['histories']}/hist_siamese.json")

    # 7. Save final model
    final_path = f"{paths['checkpoints']}/stage2_final.keras"
    siamese.save(final_path)
    print(f"  Siamese model saved → {final_path}")

    # 8. Evaluate
    acc, eer = evaluate_siamese(siamese, X_test, y_test)

    # 9. EER plot
    scores = siamese.predict(X_test, batch_size=32, verbose=0).flatten()
    plot_eer(y_test, scores, f"{paths['plots']}/siamese_eer_curve.png")
    plot_siamese(history, f"{paths['plots']}/siamese_history.png")

    return history, acc, eer


def main():
    parser = argparse.ArgumentParser(description="Train fingerprint authentication pipeline")
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both",
                        help="Which stage to run (default: both)")
    args = parser.parse_args()

    # Load and validate config
    cfg   = load_config(args.config)
    paths = setup_experiment_dir(cfg)

    exp_name = cfg["experiment"]["name"]
    print("\n" + "=" * 55)
    print(f"  Experiment : {exp_name}")
    print(f"  Backbone   : {cfg['backbone']['name']}")
    print(f"  Dataset    : {cfg['dataset']['name']}")
    print(f"  Head       : {cfg['siamese_head']['name']}")
    print(f"  Stage      : {args.stage}")
    print("=" * 55)

    hist_a = hist_b = hist_s = None
    acc = eer = None

    if args.stage in ("1", "both"):
        hist_a, hist_b = run_stage1(cfg, paths)

    if args.stage in ("2", "both"):
        hist_s, acc, eer = run_stage2(cfg, paths)

    # Full pipeline plot (only if both stages ran)
    if hist_a and hist_b and hist_s:
        plot_full_pipeline(
            hist_a, hist_b, hist_s,
            save_path=f"{paths['plots']}/full_pipeline_history.png",
            experiment_name=exp_name,
            eer=eer,
            test_acc=acc,
        )

    # Summary
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    if hist_a:
        print(f"  Stage 1 best val accuracy : {max(hist_b['val_accuracy']) * 100:.2f}%")
    if hist_s:
        print(f"  Stage 2 test accuracy     : {acc * 100:.2f}%")
        print(f"  Stage 2 EER               : {eer * 100:.2f}%")
    print(f"  Results saved to          : {paths['root']}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
