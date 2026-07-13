"""
Figures for the replication report.

Design rules: Okabe-Ito colour-blind-safe palette with a FIXED model->colour
mapping reused in every figure; one shared legend per figure; recessive grid;
2px data lines; single y-axis per panel.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")

# fixed entity -> colour (Okabe-Ito); never re-assigned between figures
COLORS = {
    "Observed":    "#000000",
    "LC":          "#E69F00",
    "RNN":         "#56B4E9",
    "LSTM":        "#009E73",
    "CNN":         "#0072B2",
    "Transformer": "#D55E00",
}
MODEL_ORDER = ["LC", "RNN", "LSTM", "CNN", "Transformer"]
DL_ORDER    = ["RNN", "LSTM", "CNN", "Transformer"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


def _grid(n_panels, ncols=3, panel=(4.0, 3.0)):
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel[0]*ncols, panel[1]*nrows),
                             constrained_layout=True)
    return fig, np.atleast_1d(axes).ravel()


def _save(fig, fname):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_loss_curves(curves_by_country: dict, fname: str):
    """curves_by_country: {country: {model: mean_curve (epochs,)}}."""
    countries = list(curves_by_country)
    fig, axes = _grid(len(countries))
    for ax, c in zip(axes, countries):
        for m in DL_ORDER:
            ax.plot(curves_by_country[c][m], color=COLORS[m], lw=1.8, label=m)
        ax.set_yscale("log")
        ax.set_title(c)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training MSE")
    for ax in axes[len(countries):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    _save(fig, fname)


def plot_final_year(preds_by_country: dict, true_by_country: dict,
                    ages: np.ndarray, year: int, fname: str):
    """Observed vs predicted log-mortality age profile in `year`."""
    countries = list(preds_by_country)
    fig, axes = _grid(len(countries))
    for ax, c in zip(axes, countries):
        ax.plot(ages, true_by_country[c], color=COLORS["Observed"],
                lw=2.2, label="Observed", zorder=5)
        for m in MODEL_ORDER:
            ax.plot(ages, preds_by_country[c][m], color=COLORS[m],
                    lw=1.6, alpha=0.9, label=m)
        ax.set_title(c)
        ax.set_xlabel("Age")
        ax.set_ylabel(f"log m(x, {year})")
    for ax in axes[len(countries):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    _save(fig, fname)


def plot_rmse_profile(profiles_by_country: dict, x: np.ndarray,
                      xlabel: str, ylabel: str, fname: str):
    """Generic per-age / per-year RMSE small-multiples."""
    from matplotlib.ticker import MaxNLocator
    countries = list(profiles_by_country)
    fig, axes = _grid(len(countries))
    for ax, c in zip(axes, countries):
        for m in MODEL_ORDER:
            ax.plot(x, profiles_by_country[c][m], color=COLORS[m],
                    lw=1.8, label=m)
        ax.set_title(c)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    for ax in axes[len(countries):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    _save(fig, fname)
