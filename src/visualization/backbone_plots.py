"""
Stage 1 training history plots (Phase A, Phase B, combined, k-fold summary).
"""

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plot_styles import STYLE, plot_acc_loss, annotate_best


def plot_kfold_summary(fold_results: list, save_path: str) -> None:
    """
    Plot per-fold val accuracy bars + combined Phase A/B curves for each fold.
    """
    n_folds = len(fold_results)
    accs    = [r["best_val_acc"] for r in fold_results]
    mean    = np.mean(accs)
    std     = np.std(accs)

    fig, axes = plt.subplots(n_folds, 2, figsize=(14, 4 * n_folds))
    fig.suptitle(f"{n_folds}-Fold CV — Stage 1  |  Mean {mean*100:.1f}% ± {std*100:.1f}%",
                 fontsize=14, fontweight="bold")

    for i, r in enumerate(fold_results):
        hist_a = r["hist_a"]
        hist_b = r["hist_b"]
        acc    = hist_a["accuracy"]     + hist_b["accuracy"]
        val    = hist_a["val_accuracy"] + hist_b["val_accuracy"]
        loss   = hist_a["loss"]         + hist_b["loss"]
        vloss  = hist_a["val_loss"]     + hist_b["val_loss"]
        split  = len(hist_a["accuracy"])
        eps    = range(1, len(acc) + 1)

        ax_acc, ax_loss = axes[i] if n_folds > 1 else axes

        plot_acc_loss(ax_acc, ax_loss, eps, acc, val, loss, vloss,
                      t_acc=f"Fold {r['fold']} — Accuracy",
                      t_loss=f"Fold {r['fold']} — Loss",
                      vline=split, vlabel=f"Phase B (ep {split})")

        ax_acc.axhline(mean, color="#888", linestyle="--", lw=1.2,
                       label=f"Mean: {mean*100:.1f}%")
        ax_acc.text(0.98, 0.03,
                    f"Best: {r['best_val_acc']*100:.1f}%",
                    transform=ax_acc.transAxes, fontsize=8,
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="#ccc", alpha=0.8))
        ax_acc.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_phase_a(hist_a: dict, save_path: str) -> None:
    """Plot Phase A accuracy and loss curves."""
    eps = range(1, len(hist_a["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase A — Frozen Backbone, Train Head Only",
                 fontsize=14, fontweight="bold")

    plot_acc_loss(ax1, ax2, eps,
                  hist_a["accuracy"], hist_a["val_accuracy"],
                  hist_a["loss"],     hist_a["val_loss"],
                  t_acc="Phase A — Accuracy", t_loss="Phase A — Loss")
    annotate_best(ax1, hist_a["val_accuracy"])

    best = max(hist_a["val_accuracy"])
    ax1.text(0.02, 0.97,
             f"Epochs: {len(eps)}  |  Best val: {best * 100:.1f}%\n"
             f"All backbone layers frozen",
             transform=ax1.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", facecolor="white",
                       edgecolor="#ccc", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_phase_b(hist_b: dict, save_path: str,
                 phase_a_best: float = None) -> None:
    """Plot Phase B accuracy and loss curves."""
    eps = range(1, len(hist_b["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase B — Unfreeze Last Conv Layer, Fine-tune",
                 fontsize=14, fontweight="bold")

    plot_acc_loss(ax1, ax2, eps,
                  hist_b["accuracy"], hist_b["val_accuracy"],
                  hist_b["loss"],     hist_b["val_loss"],
                  t_acc="Phase B — Accuracy", t_loss="Phase B — Loss")

    if phase_a_best:
        ax1.axhline(phase_a_best, color=STYLE["train_color"],
                    linestyle=":", lw=1.5, alpha=0.6,
                    label=f"Phase A best: {phase_a_best * 100:.1f}%")
        ax1.legend(fontsize=9)

    ax1.annotate("Unfreeze dip",
                 xy=(1, hist_b["val_accuracy"][0]),
                 xytext=(3, hist_b["val_accuracy"][0] - 0.06),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=9, color="#444")
    annotate_best(ax1, hist_b["val_accuracy"])

    best = max(hist_b["val_accuracy"])
    ax1.text(0.02, 0.97,
             f"Epochs: {len(eps)}  |  Best val: {best * 100:.1f}%",
             transform=ax1.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", facecolor="white",
                       edgecolor="#ccc", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_stage1_combined(hist_a: dict, hist_b: dict, save_path: str) -> None:
    """Plot Phase A + Phase B concatenated in one figure."""
    acc   = hist_a["accuracy"]     + hist_b["accuracy"]
    val   = hist_a["val_accuracy"] + hist_b["val_accuracy"]
    loss  = hist_a["loss"]         + hist_b["loss"]
    vloss = hist_a["val_loss"]     + hist_b["val_loss"]
    split = len(hist_a["accuracy"])
    eps   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage 1 Combined — Phase A + Phase B",
                 fontsize=14, fontweight="bold")

    plot_acc_loss(ax1, ax2, eps, acc, val, loss, vloss,
                  t_acc="Stage 1 — Accuracy", t_loss="Stage 1 — Loss",
                  vline=split, vlabel=f"Phase B starts (ep {split})")

    for ax in [ax1, ax2]:
        ax.axvspan(1,         split,        alpha=0.05, color="#2E6FD8")
        ax.axvspan(split + 1, len(acc) + 1, alpha=0.05, color="#2EA860")

    ymin = ax1.get_ylim()[0]
    ax1.text(split / 2,                    ymin + 0.01, "Phase A",
             ha="center", fontsize=9, color="#2E6FD8", alpha=0.8)
    ax1.text(split + (len(acc) - split)/2, ymin + 0.01, "Phase B",
             ha="center", fontsize=9, color="#2EA860", alpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")
