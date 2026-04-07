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
from src.training import train, train_siamese
from src.evaluation import evaluate, evaluate_siamese
from src.evaluation.metrics import compute_eer
from src.visualization import (
    plot_phase_a, plot_phase_b, plot_stage1_combined,
    plot_siamese, plot_eer, plot_full_pipeline,
)


def run_stage1(cfg: dict, paths: dict):
    """Run Stage 1: dataset prep + backbone training + encoder extraction."""
    dataset_cls = DATASET_REGISTRY[cfg["dataset"]["name"]]
    dataset = dataset_cls(cfg)

    img_size    = tuple(cfg["dataset"]["img_size"])
    source_dir  = cfg["dataset"]["source_dir"]
    num_subjects = cfg["dataset"]["num_subjects"]
    organized   = f"dataset/fingerprints_organized"
    processed   = f"dataset/fingerprints_224"

    # 1. Prepare data
    dataset.reorganize(source_dir, organized, num_subjects)
    dataset.preprocess(organized, processed, img_size)
    train_gen, val_gen = dataset.create_generators(
        processed, img_size,
        cfg["training"]["stage1"]["batch_size"],
        cfg["dataset"]["val_split"],
    )

    num_classes = train_gen.num_classes
    print(f"\n  Subjects (classes)  : {num_classes}")
    print(f"  Training samples    : {train_gen.samples}")
    print(f"  Validation samples  : {val_gen.samples}")

    # 2. Build backbone
    backbone_name = cfg["backbone"]["name"]
    backbone_fn   = get_backbone(backbone_name)
    model = backbone_fn(
        num_classes=num_classes,
        dense_units=cfg["backbone"]["dense_units"],
        dropout=cfg["backbone"]["dropout"],
        img_size=img_size,
    )
    model.summary()

    # 3. Train (Phase A + Phase B)
    print("\n" + "=" * 55)
    print(f"  STAGE 1: {backbone_name.upper()} Training")
    print("=" * 55)
    hist_a, hist_b = train(model, train_gen, val_gen, cfg, paths)

    # 4. Save histories
    save_history(hist_a, f"{paths['histories']}/hist_a.json")
    save_history(hist_b, f"{paths['histories']}/hist_b.json")

    # 5. Save final model
    final_path = f"{paths['checkpoints']}/stage1_final.keras"
    model.save(final_path)
    print(f"  Final model saved → {final_path}")

    # 6. Extract encoder
    encoder_layer = get_encoder_layer(backbone_name)
    encoder_path  = f"{paths['checkpoints']}/encoder.keras"
    extract_encoder(model, encoder_path, encoder_layer)

    # 7. Evaluate
    evaluate(model, val_gen)

    # 8. Save plots
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
