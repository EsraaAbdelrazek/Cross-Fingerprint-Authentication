"""
Master pipeline figure — all training phases in one plot.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.visualization.plot_styles import STYLE, plot_acc_loss


def plot_full_pipeline(hist_a: dict, hist_b: dict, hist_s: dict,
                       save_path: str,
                       experiment_name: str = "",
                       eer: float = None,
                       test_acc: float = None) -> None:
    """
    4-row master figure: Phase A / Phase B / Stage 2 / Stage 1 Combined.

    Args:
        hist_a:          Phase A history dict.
        hist_b:          Phase B history dict.
        hist_s:          Stage 2 Siamese history dict.
        save_path:       Output PNG path.
        experiment_name: Shown in the figure title.
        eer:             Final EER (shown in summary bar).
        test_acc:        Final test accuracy (shown in summary bar).
    """
    TC = STYLE["train_color"]
    VC = STYLE["val_color"]

    fig = plt.figure(figsize=(16, 22))
    title = "Full Pipeline Training History"
    if experiment_name:
        title += f" — {experiment_name}"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.60, wspace=0.32,
                           top=0.94, bottom=0.05)

    best_a = max(hist_a["val_accuracy"])
    best_b = max(hist_b["val_accuracy"])
    best_s = max(hist_s["val_accuracy"])

    # Row 1 — Phase A
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    eps_a = range(1, len(hist_a["accuracy"]) + 1)
    plot_acc_loss(*ax, eps_a,
                  hist_a["accuracy"], hist_a["val_accuracy"],
                  hist_a["loss"],     hist_a["val_loss"],
                  t_acc="Phase A — Accuracy (frozen backbone)",
                  t_loss="Phase A — Loss")
    ax[0].axhline(best_a, color=VC, ls=":", lw=1.2,
                  label=f"Best: {best_a * 100:.1f}%")
    ax[0].legend(fontsize=9)

    # Row 2 — Phase B
    ax = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    eps_b = range(1, len(hist_b["accuracy"]) + 1)
    plot_acc_loss(*ax, eps_b,
                  hist_b["accuracy"], hist_b["val_accuracy"],
                  hist_b["loss"],     hist_b["val_loss"],
                  t_acc="Phase B — Accuracy (unfreeze last conv)",
                  t_loss="Phase B — Loss")
    ax[0].axhline(best_b, color=VC, ls=":", lw=1.2,
                  label=f"Best: {best_b * 100:.1f}%")
    ax[0].axhline(best_a, color=TC, ls=":", lw=1.2, alpha=0.4,
                  label=f"Phase A best: {best_a * 100:.1f}%")
    ax[0].legend(fontsize=9)

    # Row 3 — Stage 2 Siamese
    ax = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    eps_s = range(1, len(hist_s["accuracy"]) + 1)
    plot_acc_loss(*ax, eps_s,
                  hist_s["accuracy"], hist_s["val_accuracy"],
                  hist_s["loss"],     hist_s["val_loss"],
                  t_acc="Stage 2 — Siamese Accuracy",
                  t_loss="Stage 2 — Siamese Loss")
    ax[0].axhline(best_s, color=VC, ls=":", lw=1.2,
                  label=f"Best: {best_s * 100:.1f}%")
    ax[0].legend(fontsize=9)

    # Row 4 — Stage 1 Combined
    ax = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
    acc_c   = hist_a["accuracy"]     + hist_b["accuracy"]
    val_c   = hist_a["val_accuracy"] + hist_b["val_accuracy"]
    loss_c  = hist_a["loss"]         + hist_b["loss"]
    vloss_c = hist_a["val_loss"]     + hist_b["val_loss"]
    split   = len(hist_a["accuracy"])
    eps_c   = range(1, len(acc_c) + 1)
    plot_acc_loss(*ax, eps_c, acc_c, val_c, loss_c, vloss_c,
                  t_acc="Stage 1 Combined (A + B)",
                  t_loss="Stage 1 Combined — Loss",
                  vline=split, vlabel=f"Phase B (ep {split})")
    for a in ax:
        a.axvspan(1,         split,          alpha=0.05, color="#2E6FD8")
        a.axvspan(split + 1, len(acc_c) + 1, alpha=0.05, color="#2EA860")
    ymin = ax[0].get_ylim()[0]
    ax[0].text(split / 2,                     ymin + 0.01, "Phase A",
               ha="center", fontsize=9, color="#2E6FD8", alpha=0.8)
    ax[0].text(split + (len(acc_c) - split)/2, ymin + 0.01, "Phase B",
               ha="center", fontsize=9, color="#2EA860", alpha=0.8)

    # Summary bar
    eer_str  = f"EER: {eer * 100:.2f}%  |  " if eer is not None else ""
    acc_str  = f"Test accuracy: {test_acc * 100:.2f}%" if test_acc is not None else ""
    summary  = (f"Phase A best: {best_a * 100:.1f}%  |  "
                f"Phase B best: {best_b * 100:.1f}%  |  "
                f"Stage 2 best: {best_s * 100:.1f}%  |  "
                f"{eer_str}{acc_str}")
    fig.text(0.5, 0.01, summary,
             ha="center", fontsize=10, color="white",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#2E4D8A",
                       edgecolor="none", alpha=0.92))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")
