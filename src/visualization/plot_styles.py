"""
Shared plot style primitives used by all visualization modules.
"""

import matplotlib.pyplot as plt

STYLE = {
    "train_color": "#2E6FD8",   # blue
    "val_color":   "#E87C2E",   # orange
    "phase_color": "#888888",   # gray dashed divider
    "grid_alpha":  0.25,
    "line_width":  2.0,
}


def apply_style(ax, title: str, ylabel: str, xlabel: str = "Epoch") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=STYLE["grid_alpha"], linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def plot_acc_loss(ax_acc, ax_loss, eps,
                  acc, val_acc, loss, val_loss,
                  t_acc: str = "Accuracy", t_loss: str = "Loss",
                  vline: int = None, vlabel: str = None) -> None:
    """Plot accuracy and loss curves on two axes."""
    TC = STYLE["train_color"]
    VC = STYLE["val_color"]
    LW = STYLE["line_width"]

    ax_acc.plot(eps, acc,     color=TC, lw=LW, label="Train")
    ax_acc.plot(eps, val_acc, color=VC, lw=LW, label="Validation")
    ax_loss.plot(eps, loss,     color=TC, lw=LW, label="Train")
    ax_loss.plot(eps, val_loss, color=VC, lw=LW, label="Validation")

    if vline:
        for ax in [ax_acc, ax_loss]:
            ax.axvline(vline, color="#888", linestyle="--", lw=1.2,
                       label=vlabel or f"Phase B (ep {vline})")

    apply_style(ax_acc,  t_acc,  "Accuracy")
    apply_style(ax_loss, t_loss, "Loss")


def annotate_best(ax, val_acc_list: list,
                  offset_x: float = 1, offset_y: float = 0.02) -> None:
    """Add an arrow annotation pointing to the best validation accuracy."""
    best_val = max(val_acc_list)
    best_ep  = val_acc_list.index(best_val) + 1
    ax.annotate(
        f"Best: {best_val * 100:.1f}%",
        xy=(best_ep, best_val),
        xytext=(best_ep + offset_x, best_val + offset_y),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        fontsize=9, color="#444",
    )
