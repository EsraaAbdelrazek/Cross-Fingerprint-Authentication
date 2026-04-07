"""
Stage 2 Siamese training history and EER curve plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.visualization.plot_styles import STYLE, plot_acc_loss, annotate_best


def plot_siamese(history: dict, save_path: str) -> None:
    """Plot Siamese training accuracy and loss curves."""
    eps = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stage 2 — Siamese Network Training History",
                 fontsize=14, fontweight="bold")

    plot_acc_loss(ax1, ax2, eps,
                  history["accuracy"], history["val_accuracy"],
                  history["loss"],     history["val_loss"],
                  t_acc="Stage 2 — Accuracy", t_loss="Stage 2 — Loss")
    annotate_best(ax1, history["val_accuracy"], offset_x=0.3, offset_y=-0.03)

    best = max(history["val_accuracy"])
    ax1.text(0.02, 0.05,
             f"Epochs: {len(eps)}  |  Best val: {best * 100:.1f}%",
             transform=ax1.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", facecolor="white",
                       edgecolor="#ccc", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_eer(y_test: np.ndarray, scores: np.ndarray, save_path: str) -> None:
    """Plot FAR vs FRR curve with the EER operating point marked."""
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, fpr[:len(thresholds)], label="FAR (False Accept Rate)")
    plt.plot(thresholds, fnr[:len(thresholds)], label="FRR (False Reject Rate)")
    plt.axvline(thresholds[eer_idx], color="red", linestyle="--",
                label=f"EER = {eer * 100:.2f}%")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("FAR vs FRR — EER Curve")
    plt.legend()
    plt.grid(True, alpha=STYLE["grid_alpha"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")
