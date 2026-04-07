"""
Standalone Plot Script
======================
Loads saved training histories and generates all graphs
WITHOUT rerunning any training.

Usage:
    python plot_results.py

Requirements:
    - results/hist_a.json        (Phase A history)
    - results/hist_b.json        (Phase B history)
    - results/hist_siamese.json  (Stage 2 history)

If JSON files don't exist yet, run this once after training to save them:
    python plot_results.py --save-from-models
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD / SAVE HISTORIES
# ─────────────────────────────────────────────────────────────────────────────

def save_history(history: dict, path: str):
    """Save a Keras history dict to JSON."""
    # Convert numpy floats to Python floats for JSON serialization
    clean = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  Saved → {path}")


def load_history(path: str) -> dict:
    """Load a history dict from JSON."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"History file not found: {path}\n"
            f"Please add save_history() calls to your training scripts first.\n"
            f"See instructions at the bottom of this file."
        )
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STYLE
# ─────────────────────────────────────────────────────────────────────────────

TC = "#2E6FD8"   # train color (blue)
VC = "#E87C2E"   # val color (orange)
LW = 2.0         # line width

def _style(ax, title, ylabel, xlabel="Epoch"):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _acc_loss(ax_acc, ax_loss, eps, acc, val_acc, loss, val_loss,
              t_acc="Accuracy", t_loss="Loss", vline=None, vlabel=None):
    ax_acc.plot(eps, acc,     color=TC, lw=LW, label="Train")
    ax_acc.plot(eps, val_acc, color=VC, lw=LW, label="Validation")
    ax_loss.plot(eps, loss,     color=TC, lw=LW, label="Train")
    ax_loss.plot(eps, val_loss, color=VC, lw=LW, label="Validation")
    if vline:
        for ax in [ax_acc, ax_loss]:
            ax.axvline(vline, color="#888", linestyle="--", lw=1.2,
                       label=vlabel or f"Phase B (ep {vline})")
    _style(ax_acc,  t_acc,  "Accuracy")
    _style(ax_loss, t_loss, "Loss")


def _annotate_best(ax, val_acc_list, offset_x=1, offset_y=0.02):
    best_val = max(val_acc_list)
    best_ep  = val_acc_list.index(best_val) + 1
    ax.annotate(
        f"Best: {best_val*100:.1f}%",
        xy=(best_ep, best_val),
        xytext=(best_ep + offset_x, best_val + offset_y),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        fontsize=9, color="#444"
    )


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase_a(hist_a: dict):
    eps = range(1, len(hist_a["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase A — Frozen VGG16, Train Head Only",
                 fontsize=14, fontweight="bold")
    _acc_loss(ax1, ax2, eps,
              hist_a["accuracy"], hist_a["val_accuracy"],
              hist_a["loss"],     hist_a["val_loss"],
              t_acc="Phase A — Accuracy", t_loss="Phase A — Loss")
    _annotate_best(ax1, hist_a["val_accuracy"])

    # Stats box
    best = max(hist_a["val_accuracy"])
    ax1.text(0.02, 0.97,
             f"Epochs: {len(eps)}  |  Best val: {best*100:.1f}%\n"
             f"LR: 3e-4  |  All VGG16 frozen",
             transform=ax1.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", facecolor="white",
                       edgecolor="#ccc", alpha=0.8))
    plt.tight_layout()
    plt.savefig("results/phase_a_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → results/phase_a_history.png")


def plot_phase_b(hist_b: dict, phase_a_best: float = None):
    eps = range(1, len(hist_b["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase B — Unfreeze block5_conv3, Fine-tune",
                 fontsize=14, fontweight="bold")
    _acc_loss(ax1, ax2, eps,
              hist_b["accuracy"], hist_b["val_accuracy"],
              hist_b["loss"],     hist_b["val_loss"],
              t_acc="Phase B — Accuracy", t_loss="Phase B — Loss")

    # Show Phase A best as reference line
    if phase_a_best:
        ax1.axhline(phase_a_best, color="#2E6FD8", linestyle=":",
                    lw=1.5, alpha=0.6,
                    label=f"Phase A best: {phase_a_best*100:.1f}%")
        ax1.legend(fontsize=9)

    # Annotate unfreeze dip
    ax1.annotate("Unfreeze dip",
                 xy=(1, hist_b["val_accuracy"][0]),
                 xytext=(3, hist_b["val_accuracy"][0] - 0.06),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=9, color="#444")
    _annotate_best(ax1, hist_b["val_accuracy"])

    best = max(hist_b["val_accuracy"])
    ax1.text(0.02, 0.97,
             f"Epochs: {len(eps)}  |  Best val: {best*100:.1f}%\n"
             f"LR: 5e-5  |  block5_conv3 unfrozen",
             transform=ax1.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", facecolor="white",
                       edgecolor="#ccc", alpha=0.8))
    plt.tight_layout()
    plt.savefig("results/phase_b_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → results/phase_b_history.png")


def plot_stage1_combined(hist_a: dict, hist_b: dict):
    acc   = hist_a["accuracy"]     + hist_b["accuracy"]
    val   = hist_a["val_accuracy"] + hist_b["val_accuracy"]
    loss  = hist_a["loss"]         + hist_b["loss"]
    vloss = hist_a["val_loss"]     + hist_b["val_loss"]
    split = len(hist_a["accuracy"])
    eps   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage 1 Combined — VGG16-Fingerprint (Phase A + Phase B)",
                 fontsize=14, fontweight="bold")
    _acc_loss(ax1, ax2, eps, acc, val, loss, vloss,
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
    plt.savefig("results/combined_stage1_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → results/combined_stage1_history.png")


def plot_siamese(hist_s: dict):
    eps = range(1, len(hist_s["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stage 2 — Siamese Network Training History",
                 fontsize=14, fontweight="bold")
    _acc_loss(ax1, ax2, eps,
              hist_s["accuracy"], hist_s["val_accuracy"],
              hist_s["loss"],     hist_s["val_loss"],
              t_acc="Stage 2 — Accuracy", t_loss="Stage 2 — Loss")
    _annotate_best(ax1, hist_s["val_accuracy"], offset_x=0.3, offset_y=-0.03)

    best = max(hist_s["val_accuracy"])
    ax1.text(0.02, 0.05,
             f"Epochs: {len(eps)}  |  Best val: {best*100:.1f}%\n"
             f"EER: 4.05%  |  Test accuracy: 95.95%",
             transform=ax1.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", facecolor="white",
                       edgecolor="#ccc", alpha=0.8))
    plt.tight_layout()
    plt.savefig("results/siamese_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → results/siamese_history.png")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FIGURE — ALL PHASES IN ONE
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_pipeline(hist_a: dict, hist_b: dict, hist_s: dict):
    fig = plt.figure(figsize=(16, 22))
    fig.suptitle(
        "Full Pipeline Training History\n"
        "VGG16-Fingerprint Transfer Learning + Siamese Authentication  |  PolyU Dataset",
        fontsize=15, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.60, wspace=0.32,
                           top=0.94, bottom=0.05)

    # Row 1 — Phase A
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    eps_a = range(1, len(hist_a["accuracy"]) + 1)
    _acc_loss(*ax, eps_a,
              hist_a["accuracy"], hist_a["val_accuracy"],
              hist_a["loss"],     hist_a["val_loss"],
              t_acc="Phase A — Accuracy (frozen VGG16)",
              t_loss="Phase A — Loss")
    best_a = max(hist_a["val_accuracy"])
    ax[0].axhline(best_a, color=VC, ls=":", lw=1.2,
                  label=f"Best: {best_a*100:.1f}%")
    ax[0].legend(fontsize=9)

    # Row 2 — Phase B
    ax = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    eps_b = range(1, len(hist_b["accuracy"]) + 1)
    _acc_loss(*ax, eps_b,
              hist_b["accuracy"], hist_b["val_accuracy"],
              hist_b["loss"],     hist_b["val_loss"],
              t_acc="Phase B — Accuracy (unfreeze block5_conv3)",
              t_loss="Phase B — Loss")
    best_b = max(hist_b["val_accuracy"])
    ax[0].axhline(best_b, color=VC,      ls=":",  lw=1.2,
                  label=f"Best: {best_b*100:.1f}%")
    ax[0].axhline(best_a, color=TC,      ls=":",  lw=1.2, alpha=0.4,
                  label=f"Phase A best: {best_a*100:.1f}%")
    ax[0].legend(fontsize=9)

    # Row 3 — Stage 2 Siamese
    ax = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    eps_s = range(1, len(hist_s["accuracy"]) + 1)
    _acc_loss(*ax, eps_s,
              hist_s["accuracy"], hist_s["val_accuracy"],
              hist_s["loss"],     hist_s["val_loss"],
              t_acc="Stage 2 — Siamese Accuracy",
              t_loss="Stage 2 — Siamese Loss")
    best_s = max(hist_s["val_accuracy"])
    ax[0].axhline(best_s, color=VC, ls=":", lw=1.2,
                  label=f"Best: {best_s*100:.1f}%")
    ax[0].legend(fontsize=9)

    # Row 4 — Stage 1 Combined
    ax = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
    acc_c   = hist_a["accuracy"]     + hist_b["accuracy"]
    val_c   = hist_a["val_accuracy"] + hist_b["val_accuracy"]
    loss_c  = hist_a["loss"]         + hist_b["loss"]
    vloss_c = hist_a["val_loss"]     + hist_b["val_loss"]
    split   = len(hist_a["accuracy"])
    eps_c   = range(1, len(acc_c) + 1)
    _acc_loss(*ax, eps_c, acc_c, val_c, loss_c, vloss_c,
              t_acc="Stage 1 Combined — Accuracy (A + B)",
              t_loss="Stage 1 Combined — Loss",
              vline=split, vlabel=f"Phase B (ep {split})")
    for a in ax:
        a.axvspan(1,         split,          alpha=0.05, color="#2E6FD8")
        a.axvspan(split + 1, len(acc_c) + 1, alpha=0.05, color="#2EA860")
    ymin = ax[0].get_ylim()[0]
    ax[0].text(split/2,                     ymin+0.01, "Phase A",
               ha="center", fontsize=9, color="#2E6FD8", alpha=0.8)
    ax[0].text(split+(len(acc_c)-split)/2,  ymin+0.01, "Phase B",
               ha="center", fontsize=9, color="#2EA860", alpha=0.8)

    # Summary bar at bottom
    fig.text(
        0.5, 0.01,
        f"Phase A best: {best_a*100:.1f}%  |  "
        f"Phase B best: {best_b*100:.1f}%  |  "
        f"Stage 2 best: {best_s*100:.1f}%  |  "
        f"EER: 4.05%  |  Test accuracy: 95.95%",
        ha="center", fontsize=10, color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2E4D8A",
                  edgecolor="none", alpha=0.92)
    )

    plt.savefig("results/full_pipeline_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → results/full_pipeline_history.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run all plots from saved JSON files
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n  Loading training histories ...")
    hist_a = load_history("results/hist_a.json")
    hist_b = load_history("results/hist_b.json")
    hist_s = load_history("results/hist_siamese.json")

    print(f"  Phase A: {len(hist_a['accuracy'])} epochs")
    print(f"  Phase B: {len(hist_b['accuracy'])} epochs")
    print(f"  Stage 2: {len(hist_s['accuracy'])} epochs")

    print("\n  Generating plots ...")
    plot_phase_a(hist_a)
    plot_phase_b(hist_b, phase_a_best=max(hist_a["val_accuracy"]))
    plot_stage1_combined(hist_a, hist_b)
    plot_siamese(hist_s)
    plot_full_pipeline(hist_a, hist_b, hist_s)

    print("\n  All plots saved to results/")
    print("  phase_a_history.png")
    print("  phase_b_history.png")
    print("  combined_stage1_history.png")
    print("  siamese_history.png")
    print("  full_pipeline_history.png")


if __name__ == "__main__":
    main()
