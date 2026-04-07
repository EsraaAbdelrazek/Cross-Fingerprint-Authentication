"""
Plotting functions for VGG16-Fingerprint + Siamese training history.
Produces:
  - results/phase_a_history.png        (Phase A only)
  - results/phase_b_history.png        (Phase B only)
  - results/combined_stage1_history.png (Phase A + Phase B combined)
  - results/siamese_history.png         (Stage 2 only)
  - results/full_pipeline_history.png   (All 3 phases in one figure)

Drop this file in your project folder and import from it, or
copy the functions directly into your existing scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STYLE
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "train_color"  : "#2E6FD8",   # blue
    "val_color"    : "#E87C2E",   # orange
    "phase_color"  : "#888888",   # gray dashed divider
    "grid_alpha"   : 0.25,
    "line_width"   : 2.0,
    "font_family"  : "DejaVu Sans",
}

def _apply_style(ax, title, ylabel, xlabel="Epoch"):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=STYLE["grid_alpha"], linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _plot_acc_loss(ax_acc, ax_loss, epochs, acc, val_acc, loss, val_loss,
                   title_acc="Accuracy", title_loss="Loss",
                   vline=None, vline_label=None):
    """Plot accuracy and loss on two axes."""
    lw = STYLE["line_width"]
    tc = STYLE["train_color"]
    vc = STYLE["val_color"]

    ax_acc.plot(epochs, acc,     color=tc, lw=lw, label="Train")
    ax_acc.plot(epochs, val_acc, color=vc, lw=lw, label="Validation")
    ax_loss.plot(epochs, loss,     color=tc, lw=lw, label="Train")
    ax_loss.plot(epochs, val_loss, color=vc, lw=lw, label="Validation")

    if vline is not None:
        for ax in [ax_acc, ax_loss]:
            ax.axvline(vline, color=STYLE["phase_color"],
                       linestyle="--", lw=1.2,
                       label=vline_label or f"Phase B (ep {vline})")

    _apply_style(ax_acc,  title_acc,  "Accuracy")
    _apply_style(ax_loss, title_loss, "Loss")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PHASE A ONLY
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase_a(hist_a: dict, save_path="results/phase_a_history.png"):
    """Individual graph for Phase A (frozen VGG16, train head only)."""
    eps = range(1, len(hist_a["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Phase A — Frozen VGG16, Train Head Only",
        fontsize=14, fontweight="bold", y=1.01
    )

    _plot_acc_loss(
        ax1, ax2, eps,
        hist_a["accuracy"], hist_a["val_accuracy"],
        hist_a["loss"],     hist_a["val_loss"],
        title_acc="Phase A — Accuracy",
        title_loss="Phase A — Loss",
    )

    # Annotate best val accuracy
    best_val = max(hist_a["val_accuracy"])
    best_ep  = hist_a["val_accuracy"].index(best_val) + 1
    ax1.annotate(
        f"Best: {best_val*100:.1f}%",
        xy=(best_ep, best_val),
        xytext=(best_ep + 2, best_val - 0.05),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="gray"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Phase A plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PHASE B ONLY
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase_b(hist_b: dict, save_path="results/phase_b_history.png"):
    """Individual graph for Phase B (unfreeze block5_conv3)."""
    eps = range(1, len(hist_b["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Phase B — Unfreeze block5_conv3 + Fine-tune",
        fontsize=14, fontweight="bold", y=1.01
    )

    _plot_acc_loss(
        ax1, ax2, eps,
        hist_b["accuracy"], hist_b["val_accuracy"],
        hist_b["loss"],     hist_b["val_loss"],
        title_acc="Phase B — Accuracy",
        title_loss="Phase B — Loss",
    )

    # Annotate the initial dip (epoch 1 drop after unfreezing)
    ax1.annotate(
        "Unfreeze dip",
        xy=(1, hist_b["val_accuracy"][0]),
        xytext=(3, hist_b["val_accuracy"][0] - 0.05),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="gray"
    )

    # Annotate best val accuracy
    best_val = max(hist_b["val_accuracy"])
    best_ep  = hist_b["val_accuracy"].index(best_val) + 1
    ax1.annotate(
        f"Best: {best_val*100:.1f}%",
        xy=(best_ep, best_val),
        xytext=(best_ep + 1, best_val + 0.02),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="gray"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Phase B plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. STAGE 1 COMBINED (Phase A + Phase B)
# ─────────────────────────────────────────────────────────────────────────────

def plot_stage1_combined(hist_a: dict, hist_b: dict,
                         save_path="results/combined_stage1_history.png"):
    """Combined Stage 1 graph: Phase A + Phase B joined with divider line."""
    acc   = hist_a["accuracy"]     + hist_b["accuracy"]
    val   = hist_a["val_accuracy"] + hist_b["val_accuracy"]
    loss  = hist_a["loss"]         + hist_b["loss"]
    vloss = hist_a["val_loss"]     + hist_b["val_loss"]
    eps   = range(1, len(acc) + 1)
    split = len(hist_a["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Stage 1 — VGG16-Fingerprint Training History (Phase A + Phase B)",
        fontsize=14, fontweight="bold", y=1.01
    )

    _plot_acc_loss(
        ax1, ax2, eps,
        acc, val, loss, vloss,
        title_acc="Stage 1 — Accuracy",
        title_loss="Stage 1 — Loss",
        vline=split,
        vline_label=f"Phase B starts (ep {split})"
    )

    # Shade Phase A and Phase B regions
    for ax in [ax1, ax2]:
        ax.axvspan(1,         split,      alpha=0.04, color="#2E6FD8", label="_Phase A region")
        ax.axvspan(split + 1, len(acc),   alpha=0.04, color="#2EA860", label="_Phase B region")

    # Phase labels on accuracy plot
    ax1.text(split / 2, ax1.get_ylim()[0] + 0.02, "Phase A",
             ha="center", fontsize=9, color="#2E6FD8", alpha=0.7)
    ax1.text(split + (len(acc) - split) / 2, ax1.get_ylim()[0] + 0.02, "Phase B",
             ha="center", fontsize=9, color="#2EA860", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Stage 1 combined plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. STAGE 2 SIAMESE ONLY
# ─────────────────────────────────────────────────────────────────────────────

def plot_siamese(history: dict, save_path="results/siamese_history.png"):
    """Individual graph for Stage 2 Siamese network."""
    eps = range(1, len(history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Stage 2 — Siamese Network Training History",
        fontsize=14, fontweight="bold", y=1.01
    )

    _plot_acc_loss(
        ax1, ax2, eps,
        history["accuracy"], history["val_accuracy"],
        history["loss"],     history["val_loss"],
        title_acc="Stage 2 — Accuracy",
        title_loss="Stage 2 — Loss",
    )

    # Annotate final accuracy
    final_val = max(history["val_accuracy"])
    final_ep  = history["val_accuracy"].index(final_val) + 1
    ax1.annotate(
        f"Best: {final_val*100:.1f}%",
        xy=(final_ep, final_val),
        xytext=(final_ep + 0.5, final_val - 0.03),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="gray"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Siamese plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE — ALL 3 PHASES IN ONE FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_pipeline(hist_a: dict, hist_b: dict, hist_siamese: dict,
                       save_path="results/full_pipeline_history.png"):
    """
    Master figure showing all three phases:
      Row 1: Phase A accuracy | Phase A loss
      Row 2: Phase B accuracy | Phase B loss
      Row 3: Stage 2 accuracy | Stage 2 loss
      Row 4 (wide): Combined Stage 1 (A+B) accuracy | Combined Stage 1 loss
    """
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(
        "Full Pipeline Training History\nVGG16-Fingerprint → Siamese Authentication (PolyU)",
        fontsize=16, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           top=0.93, bottom=0.04)

    # ── Row 1: Phase A ────────────────────────────────────────────────────────
    ax_a_acc  = fig.add_subplot(gs[0, 0])
    ax_a_loss = fig.add_subplot(gs[0, 1])
    eps_a = range(1, len(hist_a["accuracy"]) + 1)
    _plot_acc_loss(
        ax_a_acc, ax_a_loss, eps_a,
        hist_a["accuracy"], hist_a["val_accuracy"],
        hist_a["loss"],     hist_a["val_loss"],
        title_acc="Phase A — Accuracy (frozen VGG16)",
        title_loss="Phase A — Loss",
    )
    best_a = max(hist_a["val_accuracy"])
    ax_a_acc.axhline(best_a, color="orange", linestyle=":", lw=1,
                     label=f"Best val: {best_a*100:.1f}%")
    ax_a_acc.legend(fontsize=9)

    # ── Row 2: Phase B ────────────────────────────────────────────────────────
    ax_b_acc  = fig.add_subplot(gs[1, 0])
    ax_b_loss = fig.add_subplot(gs[1, 1])
    eps_b = range(1, len(hist_b["accuracy"]) + 1)
    _plot_acc_loss(
        ax_b_acc, ax_b_loss, eps_b,
        hist_b["accuracy"], hist_b["val_accuracy"],
        hist_b["loss"],     hist_b["val_loss"],
        title_acc="Phase B — Accuracy (unfreeze block5_conv3)",
        title_loss="Phase B — Loss",
    )
    best_b = max(hist_b["val_accuracy"])
    ax_b_acc.axhline(best_b, color="orange", linestyle=":", lw=1,
                     label=f"Best val: {best_b*100:.1f}%")
    ax_b_acc.axhline(best_a, color="blue", linestyle=":", lw=1, alpha=0.4,
                     label=f"Phase A best: {best_a*100:.1f}%")
    ax_b_acc.legend(fontsize=9)

    # ── Row 3: Stage 2 Siamese ────────────────────────────────────────────────
    ax_s_acc  = fig.add_subplot(gs[2, 0])
    ax_s_loss = fig.add_subplot(gs[2, 1])
    eps_s = range(1, len(hist_siamese["accuracy"]) + 1)
    _plot_acc_loss(
        ax_s_acc, ax_s_loss, eps_s,
        hist_siamese["accuracy"], hist_siamese["val_accuracy"],
        hist_siamese["loss"],     hist_siamese["val_loss"],
        title_acc="Stage 2 — Siamese Accuracy",
        title_loss="Stage 2 — Siamese Loss",
    )
    best_s = max(hist_siamese["val_accuracy"])
    ax_s_acc.axhline(best_s, color="orange", linestyle=":", lw=1,
                     label=f"Best val: {best_s*100:.1f}%")
    ax_s_acc.legend(fontsize=9)

    # ── Row 4: Combined Stage 1 (A + B) ──────────────────────────────────────
    ax_c_acc  = fig.add_subplot(gs[3, 0])
    ax_c_loss = fig.add_subplot(gs[3, 1])
    acc_full   = hist_a["accuracy"]     + hist_b["accuracy"]
    val_full   = hist_a["val_accuracy"] + hist_b["val_accuracy"]
    loss_full  = hist_a["loss"]         + hist_b["loss"]
    vloss_full = hist_a["val_loss"]     + hist_b["val_loss"]
    split      = len(hist_a["accuracy"])
    eps_full   = range(1, len(acc_full) + 1)

    _plot_acc_loss(
        ax_c_acc, ax_c_loss, eps_full,
        acc_full, val_full, loss_full, vloss_full,
        title_acc="Stage 1 Combined — Accuracy (Phase A + B)",
        title_loss="Stage 1 Combined — Loss",
        vline=split,
        vline_label=f"Phase B starts (ep {split})"
    )

    # Shade phases
    for ax in [ax_c_acc, ax_c_loss]:
        ax.axvspan(1,         split,           alpha=0.05, color="#2E6FD8")
        ax.axvspan(split + 1, len(acc_full),   alpha=0.05, color="#2EA860")

    ax_c_acc.text(split / 2, min(val_full) + 0.01, "Phase A",
                  ha="center", fontsize=9, color="#2E6FD8", alpha=0.8)
    ax_c_acc.text(split + (len(acc_full) - split) / 2,
                  min(val_full) + 0.01, "Phase B",
                  ha="center", fontsize=9, color="#2EA860", alpha=0.8)

    # ── Summary box ───────────────────────────────────────────────────────────
    summary = (
        f"Phase A best val: {best_a*100:.1f}%  |  "
        f"Phase B best val: {best_b*100:.1f}%  |  "
        f"Stage 2 best val: {best_s*100:.1f}%  |  "
        f"Stage 2 EER: 4.05%"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10,
             color="white",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#2E4D8A",
                       edgecolor="none", alpha=0.9))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Full pipeline plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO USE
# ─────────────────────────────────────────────────────────────────────────────
#
# In polyu_vgg16_fingerprint.py, replace the plot_history() call with:
#
#   from plot_utils import (plot_phase_a, plot_phase_b,
#                           plot_stage1_combined, plot_full_pipeline)
#
#   plot_phase_a(hist_a)
#   plot_phase_b(hist_b)
#   plot_stage1_combined(hist_a, hist_b)
#
# In polyu_siamese_fingerprint.py, replace the plot_history() call with:
#
#   from plot_utils import plot_siamese, plot_full_pipeline
#
#   plot_siamese(history)
#
#   # After both stages are done, generate the master figure:
#   plot_full_pipeline(hist_a, hist_b, history)
#
# ─────────────────────────────────────────────────────────────────────────────
