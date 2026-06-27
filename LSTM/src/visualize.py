"""
Publication-quality plots for the LSTM mortality project.

All functions save figures to outputs/figures/ and return the Figure object.

Colour palette: Okabe-Ito (colour-blind safe, 8 colours).
Theme: seaborn-v0_8-whitegrid, consistent font sizes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── colour-blind-safe Okabe-Ito palette ──────────────────────────────────────
COLORS = {
    "Lee-Carter":       "#E69F00",   # orange
    "Vanilla LSTM":     "#56B4E9",   # sky blue
    "Stacked LSTM":     "#009E73",   # bluish green
    "BiLSTM":           "#F0E442",   # yellow
    "CNN-LSTM":         "#0072B2",   # blue
    "Attention LSTM":   "#D55E00",   # vermillion
    "true":             "#000000",   # black
}
MODEL_ORDER = ["Lee-Carter", "Vanilla LSTM", "Stacked LSTM", "BiLSTM", "CNN-LSTM", "Attention LSTM"]

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
})


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loss curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(histories: dict, title: str = "", fname: str = "loss_curves.png") -> None:
    """
    Plot training and validation loss curves for multiple models side-by-side.

    Parameters
    ----------
    histories : {"ModelName": {"train_loss": [...], "val_loss": [...]}}
    title     : figure suptitle
    fname     : output filename
    """
    n = len(histories)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, (name, h) in zip(axes, histories.items()):
        ax.plot(h["train_loss"], color="#0072B2", label="train", linewidth=1.2)
        ax.plot(h["val_loss"],   color="#D55E00", label="val",   linewidth=1.2, linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (std. log-mx)")
        ax.legend(frameon=False)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    for ax in axes[n:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    _save(fig, fname)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Forecast vs true log(mx) for selected ages
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast_ages(
    years_all: np.ndarray,
    log_mx_true: np.ndarray,
    forecasts: dict,
    test_start_idx: int,
    ages_to_plot: list = [0, 20, 40, 60, 80],
    ages: np.ndarray = None,
    country: str = "",
    sex: str = "",
    fname: str = "forecast_ages.png",
) -> None:
    """
    For each selected age, plot true log(mx) (full period) and model forecasts
    over the test period.

    Parameters
    ----------
    years_all     : (T,) array of calendar years
    log_mx_true   : (n_ages, T) full observed log(mx)
    forecasts     : {"ModelName": (n_ages, T_test) predicted log(mx)}
    test_start_idx: column index where the test period begins in log_mx_true
    ages_to_plot  : list of age indices to display
    ages          : (n_ages,) actual age labels (0, 1, ..., 99)
    """
    n  = len(ages_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    test_years = years_all[test_start_idx:]

    for ax, ai in zip(axes, ages_to_plot):
        age_label = ages[ai] if ages is not None else ai
        ax.plot(years_all, log_mx_true[ai, :], color=COLORS["true"],
                linewidth=1.5, label="Observed", zorder=5)
        ax.axvline(years_all[test_start_idx], color="grey", linewidth=0.8,
                   linestyle=":", alpha=0.8, label="Test start")
        for name in MODEL_ORDER:
            if name in forecasts:
                ax.plot(test_years, forecasts[name][ai, :],
                        color=COLORS[name], linewidth=1.2,
                        linestyle="--", label=name, alpha=0.85)
        ax.set_title(f"Age {age_label}")
        ax.set_xlabel("Year")
        ax.set_ylabel("log m(x,t)")
        ax.legend(frameon=False, fontsize=7)

    title = f"Forecast log(mx) — {country} {sex}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _save(fig, fname)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Life expectancy over test period
# ─────────────────────────────────────────────────────────────────────────────

def plot_life_expectancy(
    years_test: np.ndarray,
    e0_true: np.ndarray,
    e0_preds: dict,
    country: str = "",
    sex: str = "",
    fname: str = "life_expectancy.png",
) -> None:
    """
    Plot observed vs predicted e0 over the test period for all models.

    Parameters
    ----------
    years_test : (T_test,) years in the test period
    e0_true    : (T_test,) observed life expectancy
    e0_preds   : {"ModelName": (T_test,) predicted e0}
    """
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    ax.plot(years_test, e0_true, color=COLORS["true"],
            linewidth=2, label="Observed", zorder=5)
    for name in MODEL_ORDER:
        if name in e0_preds:
            ax.plot(years_test, e0_preds[name], color=COLORS[name],
                    linewidth=1.3, linestyle="--", label=name, alpha=0.9)

    ax.set_xlabel("Year")
    ax.set_ylabel("Life expectancy at birth (e₀)")
    ax.set_title(f"Period Life Expectancy — {country} {sex}")
    ax.legend(frameon=False, ncol=2)
    _save(fig, fname)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metrics bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_bar(results_df, metric: str = "RMSE",
                     title: str = "", fname: str = "metrics_bar.png") -> None:
    """
    Horizontal bar chart of a single metric across models.

    Parameters
    ----------
    results_df : DataFrame (Model × metric columns)
    metric     : column name to plot
    """
    df = results_df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(6, 0.55 * len(df) + 1.5), constrained_layout=True)

    bars = ax.barh(df.index, df[metric],
                   color=[COLORS.get(m, "#999999") for m in df.index])
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} by Model (lower is better)")
    ax.invert_yaxis()
    _save(fig, fname)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Heatmap of residuals
# ─────────────────────────────────────────────────────────────────────────────

def plot_residual_heatmap(
    log_mx_true: np.ndarray,
    log_mx_pred: np.ndarray,
    years: np.ndarray,
    ages: np.ndarray,
    model_name: str = "",
    country: str = "",
    sex: str = "",
    fname: str = "residuals.png",
) -> None:
    """
    Heatmap of (true − predicted) log(mx) over the test period.

    Diverging palette centred at 0; red = underestimate, blue = overestimate.
    """
    residuals = log_mx_true - log_mx_pred   # (n_ages, T_test)
    vmax = np.percentile(np.abs(residuals), 95)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(residuals, aspect="auto", origin="lower",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[years[0] - 0.5, years[-1] + 0.5,
                            ages[0]  - 0.5, ages[-1]  + 0.5])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("true − predicted log(mx)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Age")
    ax.set_title(f"Residuals: {model_name} — {country} {sex}")
    _save(fig, fname)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary table as figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_table(results_df, fname: str = "results_table.png") -> None:
    """Render the results DataFrame as a styled table figure."""
    df = results_df.round(4).reset_index()
    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(figsize=(n_cols * 1.6, n_rows * 0.5 + 0.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0072B2")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
    fig.suptitle("Model Comparison — Test Set Metrics", fontsize=11, fontweight="bold")
    _save(fig, fname)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Attention weights (for AttentionLSTM only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_weights(model, sample_x: np.ndarray,
                           seq_len: int, fname: str = "attention_weights.png") -> None:
    """
    Visualise mean attention weight over time steps for a batch of samples.

    Parameters
    ----------
    model    : trained AttentionLSTM
    sample_x : (N, seq_len, n_ages) numpy array
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        x     = torch.tensor(sample_x, dtype=torch.float32)
        hidden, _ = model.lstm(x)
        q     = model.attn_q(hidden[:, -1:, :])
        scale = hidden.shape[-1] ** 0.5
        scores  = torch.bmm(q, hidden.transpose(1, 2)) / scale  # (N,1,T)
        weights = F.softmax(scores, dim=-1).squeeze(1).numpy()  # (N, T)

    mean_w = weights.mean(axis=0)   # (T,)
    steps  = np.arange(1, seq_len + 1)

    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.bar(steps, mean_w, color="#56B4E9")
    ax.set_xlabel("Time step (1 = oldest, T = most recent)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Temporal Attention Weights — Attention LSTM")
    _save(fig, fname)
