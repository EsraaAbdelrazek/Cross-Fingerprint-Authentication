"""
Standalone evaluation script — no training, just metrics and plots.

Usage:
    python scripts/evaluate.py --config configs/polyu_vgg16_ead.yaml --stage 1
    python scripts/evaluate.py --config configs/polyu_vgg16_ead.yaml --stage 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tensorflow.keras.models import load_model

from src.utils import load_config, setup_experiment_dir
from src.data import DATASET_REGISTRY
from src.data.pair_generator import generate_pairs, split_pairs
from src.evaluation import evaluate, evaluate_siamese
from src.evaluation.metrics import compute_eer
from src.visualization import (
    plot_phase_a, plot_phase_b, plot_stage1_combined,
    plot_siamese, plot_eer,
)
from src.utils.history_io import load_history


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained fingerprint model without retraining."
    )
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--stage", choices=["1", "2"], required=True,
                        help="Which stage to evaluate")
    args = parser.parse_args()

    cfg   = load_config(args.config)
    paths = setup_experiment_dir(cfg)

    if args.stage == "1":
        # Load best Stage 1 checkpoint
        ckpt = f"{paths['checkpoints']}/stage1_best.keras"
        print(f"\n  Loading Stage 1 model: {ckpt}")
        model = load_model(ckpt)

        # Recreate validation generator
        dataset_cls = DATASET_REGISTRY[cfg["dataset"]["name"]]
        dataset = dataset_cls(cfg)
        img_size = tuple(cfg["dataset"]["img_size"])
        backbone_name = cfg["backbone"]["name"]
        rescale = None if backbone_name == "efficientnet" else 1.0 / 255
        _, val_gen = dataset.create_generators(
            "dataset/fingerprints_224", img_size,
            cfg["training"]["stage1"]["batch_size"],
            cfg["dataset"]["val_split"],
            rescale=rescale,
        )
        evaluate(model, val_gen)

        # Regenerate plots from saved JSON
        hist_a = load_history(f"{paths['histories']}/hist_a.json")
        hist_b = load_history(f"{paths['histories']}/hist_b.json")
        plot_phase_a(hist_a, f"{paths['plots']}/phase_a_history.png")
        plot_phase_b(hist_b, f"{paths['plots']}/phase_b_history.png",
                     phase_a_best=max(hist_a["val_accuracy"]))
        plot_stage1_combined(hist_a, hist_b, f"{paths['plots']}/combined_stage1_history.png")

    elif args.stage == "2":
        # Load best Stage 2 checkpoint
        ckpt = f"{paths['checkpoints']}/stage2_best.keras"
        print(f"\n  Loading Stage 2 model: {ckpt}")
        siamese = load_model(ckpt)

        # Generate test pairs
        dataset_cls = DATASET_REGISTRY[cfg["dataset"]["name"]]
        dataset = dataset_cls(cfg)
        img_size   = tuple(cfg["dataset"]["img_size"])
        images     = dataset.load_images("dataset/fingerprints_224", img_size)
        stage2_cfg = cfg["training"]["stage2"]
        t_img1, t_img2, t_labels = generate_pairs(images, stage2_cfg["test_pairs"], seed=99)
        X_test = [t_img1, t_img2]
        y_test = t_labels

        acc, eer_val = evaluate_siamese(siamese, X_test, y_test)
        scores = siamese.predict(X_test, batch_size=32, verbose=0).flatten()
        plot_eer(y_test, scores, f"{paths['plots']}/siamese_eer_curve.png")

        hist_s = load_history(f"{paths['histories']}/hist_siamese.json")
        plot_siamese(hist_s, f"{paths['plots']}/siamese_history.png")


if __name__ == "__main__":
    main()
