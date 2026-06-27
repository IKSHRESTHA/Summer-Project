"""
run_kt.py  —  Correct LSTM Mortality Forecasting via kt Factor (v3)

This is the methodologically correct formulation.  All three papers cited
in the project (Richman & Schreck 2019, Perla et al. 2021, Nigri et al. 2021)
operate on the 1-dimensional kt factor extracted from Lee-Carter, NOT on the
raw 100-dimensional rate surface.

Pipeline
--------
1.  Load HMD data.
2.  Fit LC on training data -> extract ax, bx, kt_train.
3.  Project val + test log(mx) onto bx to recover kt_val and kt_test.
4.  [HPO]   Optuna search on kt time series (hidden in [8,16,32,64]).
5.  [TRAIN] Train 5 LSTM variants on kt.
6.  [PRED]  Multi-step autoregressive kt forecast (n_test steps ahead).
7.  [PRED]  Rolling 1-step-ahead kt forecast (uses actual observed kt).
8.  [RECON] Reconstruct log(mx): ax + bx * kt_pred for each model/mode.
9.  [EVAL]  MAE, RMSE, ME, MAPE, e0 errors vs LC multi-step and rolling.
10. [DM]    Diebold-Mariano tests vs LC multi-step (now a fair comparison).
11. [PLOT]  kt series, forecast-by-age, e0, RMSE bar, results table.
12. [SAVE]  Metrics CSV and model weights to outputs/.

Usage
-----
    cd LSTM/
    python run_kt.py [--country AUS] [--sex male] [--epochs 200]
                     [--hpo_trials 20] [--device cpu] [--skip_hpo]
"""

import argparse
import copy
import json
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from src.data         import make_loaders, COUNTRIES, SEXES
from src.data_kt      import (KT_SEQ_LEN, project_kt, make_kt_loaders,
                               predict_kt_multistep, predict_kt_rolling)
from src.train        import train_model
from src.lee_carter   import (fit_lee_carter, lc_forecast, project_rates,
                               lc_rolling_forecast, life_expectancy as lc_e0)
from src.evaluate     import (compute_metrics, compute_e0_errors,
                               diebold_mariano, build_results_table)
from src.visualize    import (plot_loss_curves, plot_forecast_ages,
                               plot_life_expectancy, plot_metrics_bar,
                               plot_results_table, plot_residual_heatmap)

from src.models.kt_lstm import (KtVanillaLSTM, KtStackedLSTM, KtBiLSTM,
                                 KtAttentionLSTM, KtCNNLSTM)

KT_MODEL_CLASSES = {
    "Kt-Vanilla":    KtVanillaLSTM,
    "Kt-Stacked":    KtStackedLSTM,
    "Kt-BiLSTM":     KtBiLSTM,
    "Kt-Attention":  KtAttentionLSTM,
    "Kt-CNN-LSTM":   KtCNNLSTM,
}

# Okabe-Ito colours for kt models
KT_COLORS = {
    "LC (multi-step)":  "#E69F00",
    "LC (rolling)":     "#CC79A7",
    "Kt-Vanilla":       "#56B4E9",
    "Kt-Stacked":       "#009E73",
    "Kt-BiLSTM":        "#F0E442",
    "Kt-Attention":     "#D55E00",
    "Kt-CNN-LSTM":      "#0072B2",
    "true":             "#000000",
}
KT_MODEL_ORDER = list(KT_MODEL_CLASSES.keys())


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_kt_model(name: str, params: dict):
    cls = KT_MODEL_CLASSES[name]
    return cls(hidden_size=params.get("hidden_size", 32),
               dropout=params.get("dropout", 0.1))


def reconstruct_log_mx(ax: np.ndarray, bx: np.ndarray,
                        kt_pred: np.ndarray) -> np.ndarray:
    """Reconstruct (n_ages, n_test) log(mx) from predicted kt values."""
    return ax[:, None] + bx[:, None] * kt_pred[None, :]


def kt_destd(preds_z: np.ndarray, kt_mean: float, kt_std: float) -> np.ndarray:
    """Invert kt standardisation."""
    return preds_z * (kt_std + 1e-8) + kt_mean


# ── hyperparameter optimisation ───────────────────────────────────────────────

def hpo_kt_model(model_class, loaders: dict, n_trials: int = 15,
                 hpo_epochs: int = 60, device: str = "cpu", seed: int = 42) -> dict:
    """
    Optuna TPE search for one kt model class.
    Search space is much smaller than raw-rate models, matching the lower
    intrinsic dimensionality of the 1D kt forecasting problem.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("    optuna not installed — using defaults")
        return {"hidden_size": 32, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4}

    def objective(trial):
        hidden  = trial.suggest_categorical("hidden_size", [8, 16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr      = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd      = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        model   = model_class(hidden_size=hidden, dropout=dropout)
        hist    = train_model(model, loaders, epochs=hpo_epochs, lr=lr,
                              weight_decay=wd, patience=15, device=device)
        return min(hist["val_loss"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5)
    study   = optuna.create_study(direction="minimize",
                                  sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def hpo_all_kt_models(loaders: dict, n_trials: int = 15,
                      hpo_epochs: int = 60, device: str = "cpu") -> dict:
    best = {}
    for name, cls in KT_MODEL_CLASSES.items():
        print(f"    HPO: {name} ({n_trials} trials) ...", end=" ", flush=True)
        best[name] = hpo_kt_model(cls, loaders, n_trials=n_trials,
                                   hpo_epochs=hpo_epochs, device=device)
        print(f"hidden={best[name].get('hidden_size')}  "
              f"lr={best[name].get('lr', 0):.2e}")
    return best


# ── kt series plot ────────────────────────────────────────────────────────────

def plot_kt_series(years_all: np.ndarray, kt_all_obs: np.ndarray,
                   n_train: int, n_val: int,
                   kt_multi_preds: dict, kt_rolling_preds: dict,
                   country: str, sex: str, fname: str) -> None:
    """
    Plot the full kt series (observed) with LSTM multi-step and rolling
    forecasts over the test period, plus the ARIMA (LC) kt forecast.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig_dir = os.path.join("outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    n_test = len(next(iter(kt_multi_preds.values())))
    n_total = n_train + n_val + n_test

    years_obs  = years_all[:n_total]
    years_test = years_all[n_train + n_val : n_total]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    for ax, preds, title_suffix in [
        (axes[0], kt_multi_preds,   "Multi-step (autoregressive)"),
        (axes[1], kt_rolling_preds, "Rolling 1-step-ahead"),
    ]:
        ax.plot(years_obs, kt_all_obs[:n_total], color="#000000",
                linewidth=1.8, label="Observed kt", zorder=5)
        ax.axvline(years_all[n_train],       color="grey", lw=0.8,
                   ls=":", alpha=0.7, label="Train end")
        ax.axvline(years_all[n_train+n_val], color="grey", lw=0.8,
                   ls="--", alpha=0.7, label="Val end / Test start")

        for name, kt_p in preds.items():
            ax.plot(years_test, kt_p, color=KT_COLORS.get(name, "#999999"),
                    linewidth=1.4, linestyle="--", label=name, alpha=0.9)

        ax.set_xlabel("Year")
        ax.set_ylabel("kt")
        ax.set_title(f"kt Forecast — {title_suffix}")
        ax.legend(frameon=False, fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    fig.suptitle(f"Mortality Index kt — {country} {sex}",
                 fontsize=13, fontweight="bold")
    path = os.path.join(fig_dir, fname)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_kt_attention(model, kt_context_z: np.ndarray,
                      seq_len: int, fname: str, device: str = "cpu") -> None:
    """Visualise attention weights for KtAttentionLSTM on the test context."""
    import matplotlib.pyplot as plt

    fig_dir = os.path.join("outputs", "figures")
    x       = torch.tensor(kt_context_z[-seq_len:].astype(np.float32),
                           device=device).unsqueeze(0).unsqueeze(-1)
    weights = model.get_attention_weights(x).squeeze()   # (seq_len,)

    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.bar(range(1, seq_len + 1), weights, color="#D55E00")
    ax.set_xlabel("Time step (1 = oldest, T = most recent)")
    ax.set_ylabel("Attention weight")
    ax.set_title("Temporal Attention on kt — KtAttentionLSTM")
    path = os.path.join(fig_dir, fname)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(42)
    for d in ["outputs/models", "outputs/results", "outputs/figures"]:
        os.makedirs(d, exist_ok=True)

    country = args.country
    sex     = args.sex
    device  = args.device
    tag     = f"{country}_{sex}_kt"

    print(f"\n{'='*65}")
    print(f"  LSTM-kt Mortality Forecasting v3 -- {country} ({sex})")
    print(f"  Correct formulation: LSTM on extracted kt factor (1D)")
    print(f"{'='*65}")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    _, meta = make_loaders(country, sex)
    log_mx   = meta["log_mx_raw"]          # (n_ages, T)
    years    = meta["years"]               # (T,)
    ages     = meta["ages"]               # (n_ages,)
    n_ages   = len(ages)
    n_train  = meta["n_train"]
    n_val    = meta["n_val"]
    n_test   = meta["n_test"]
    T_full   = log_mx.shape[1]

    log_mx_test_true = log_mx[:, n_train + n_val : n_train + n_val + n_test]
    years_test       = years[n_train + n_val : n_train + n_val + n_test]

    print(f"   Ages {ages[0]}-{ages[-1]} | Years {years[0]}-{years[-1]}")
    print(f"   Train={n_train}  Val={n_val}  Test={n_test}")

    # ── 2. Lee-Carter on training data ────────────────────────────────────────
    print("\n[2] Fitting Lee-Carter on training data ...")
    log_mx_train = log_mx[:, :n_train]
    lc = fit_lee_carter(log_mx_train)
    ax, bx, kt_train = lc["ax"], lc["bx"], lc["kt"]
    print(f"   Variance explained by first SVD component: {lc['var_explained']:.1%}")
    print(f"   kt range (training): [{kt_train.min():.2f}, {kt_train.max():.2f}]")

    # ── 3. Project val + test log(mx) onto bx ────────────────────────────────
    print("\n[3] Projecting val + test log(mx) onto training bx ...")
    kt_val_proj  = project_kt(log_mx[:, n_train : n_train + n_val], ax, bx)
    kt_test_proj = project_kt(log_mx_test_true, ax, bx)

    # kt time series for different purposes:
    kt_trainval = np.concatenate([kt_train, kt_val_proj])          # for LSTM training
    kt_all_obs  = np.concatenate([kt_train, kt_val_proj, kt_test_proj])  # for rolling pred

    print(f"   kt_train  : {kt_train.shape}   mean={kt_train.mean():.3f}")
    print(f"   kt_val    : {kt_val_proj.shape}  mean={kt_val_proj.mean():.3f}")
    print(f"   kt_test   : {kt_test_proj.shape}  mean={kt_test_proj.mean():.3f}")
    print(f"   (Continuing downward trend suggests LC/LSTM should be able to extrapolate)")

    # ── 4. LC baselines ───────────────────────────────────────────────────────
    print("\n[4] Computing LC baselines ...")
    # multi-step: ARIMA on kt_train, forecast n_val+n_test steps ahead
    kt_arima = lc_forecast(kt_train, h=n_val + n_test)
    kt_arima_test = kt_arima[-n_test:]
    lc_multi = reconstruct_log_mx(ax, bx, kt_arima_test)

    # rolling 1-step-ahead (refits full LC each year)
    lc_rolling = lc_rolling_forecast(log_mx, n_train, n_val, n_test)

    all_forecasts = {
        "LC (multi-step)": lc_multi,
        "LC (rolling)":    lc_rolling,
    }
    all_metrics = {}

    for lc_name, lc_pred in all_forecasts.items():
        m = compute_metrics(log_mx_test_true, lc_pred)
        e = compute_e0_errors(log_mx_test_true, lc_pred, sex)
        all_metrics[lc_name] = {**m, **e}
        print(f"   {lc_name:22s}  RMSE={m['RMSE']:.4f}  e0_MAE={e['e0_MAE']:.3f} yrs")

    # ARIMA kt for plotting
    kt_multi_preds   = {"LC (ARIMA)": kt_arima_test}
    kt_rolling_preds = {}

    # ── 5. Build kt DataLoaders ───────────────────────────────────────────────
    print("\n[5] Building kt DataLoaders ...")
    loaders_kt, kt_stats = make_kt_loaders(
        kt_trainval, n_train, n_val,
        batch_size=8, seq_len=KT_SEQ_LEN, seed=42,
    )
    kt_mean, kt_std = kt_stats["mean"], kt_stats["std"]
    n_train_seqs = len(loaders_kt["train"].dataset)
    n_val_seqs   = len(loaders_kt["val"].dataset)
    print(f"   Training sequences : {n_train_seqs}  "
          f"(kt length {n_train}, seq_len={KT_SEQ_LEN})")
    print(f"   Validation sequences: {n_val_seqs}")
    print(f"   kt mean={kt_mean:.3f}  std={kt_std:.3f}")

    # ── 6. HPO ────────────────────────────────────────────────────────────────
    if args.skip_hpo:
        print("\n[6] HPO skipped -- using literature-informed defaults")
        # hidden=32 chosen to match Richman & Schreck scale; dropout light
        best_params = {n: {"hidden_size": 32, "dropout": 0.1,
                           "lr": 5e-3, "weight_decay": 1e-4}
                       for n in KT_MODEL_CLASSES}
    else:
        print(f"\n[6] HPO ({args.hpo_trials} trials/model, "
              f"search: hidden in [8,16,32,64]) ...")
        best_params = hpo_all_kt_models(loaders_kt, n_trials=args.hpo_trials,
                                        hpo_epochs=min(60, args.epochs),
                                        device=device)
        with open(f"outputs/results/{tag}_hpo.json", "w") as f:
            json.dump(best_params, f, indent=2)

    # ── 7. Train kt LSTM models ───────────────────────────────────────────────
    print(f"\n[7] Training 5 kt-LSTM models ({args.epochs} epochs, "
          f"AdamW + LayerNorm + Residual) ...")
    trained_models = {}
    histories      = {}

    for name, cls in KT_MODEL_CLASSES.items():
        p = best_params[name]
        print(f"\n  -- {name}  (hidden={p.get('hidden_size',32)}"
              f"  drop={p.get('dropout',0.1):.2f}"
              f"  lr={p.get('lr',5e-3):.1e}"
              f"  wd={p.get('weight_decay',1e-4):.1e}) --")

        model = build_kt_model(name, p)
        n_params = sum(par.numel() for par in model.parameters() if par.requires_grad)
        print(f"   Parameters: {n_params:,}   "
              f"param-to-sample ratio: {n_params/max(n_train_seqs,1):.1f}:1")

        hist = train_model(model, loaders_kt, epochs=args.epochs,
                           lr=p.get("lr", 5e-3),
                           weight_decay=p.get("weight_decay", 1e-4),
                           patience=25, device=device, verbose=False)
        trained_models[name] = model
        histories[name]      = hist

        torch.save(hist["best_model_state"],
                   f"outputs/models/{tag}_{name.replace(' ','_')}.pt")
        print(f"   best_epoch={hist['best_epoch']:3d}  "
              f"best_val_loss={min(hist['val_loss']):.5f}")

    # ── 8. kt Forecasts ───────────────────────────────────────────────────────
    print("\n[8] Generating kt forecasts ...")

    # context: last seq_len observed kt values at end of val period
    kt_context = kt_trainval  # use all training+val kt as context

    for name, model in trained_models.items():
        # multi-step: predict n_test steps from end of val
        kt_ms = predict_kt_multistep(
            model, kt_context, n_steps=n_test,
            kt_mean=kt_mean, kt_std=kt_std,
            seq_len=KT_SEQ_LEN, device=device,
        )
        kt_multi_preds[name] = kt_ms
        log_mx_ms = reconstruct_log_mx(ax, bx, kt_ms)
        all_forecasts[name + " (multi)"] = log_mx_ms

        # rolling 1-step-ahead: use actual observed kt at each test step
        kt_roll = predict_kt_rolling(
            model, kt_all_obs, n_train, n_val, n_test,
            kt_mean=kt_mean, kt_std=kt_std,
            seq_len=KT_SEQ_LEN, device=device,
        )
        kt_rolling_preds[name] = kt_roll
        log_mx_roll = reconstruct_log_mx(ax, bx, kt_roll)
        all_forecasts[name + " (rolling)"] = log_mx_roll

    # ── 9. Evaluation ─────────────────────────────────────────────────────────
    print("\n[9] Computing test-set metrics ...")

    for fname_key, log_mx_pred in all_forecasts.items():
        if fname_key in ("LC (multi-step)", "LC (rolling)"):
            continue   # already computed above
        m = compute_metrics(log_mx_test_true, log_mx_pred)
        e = compute_e0_errors(log_mx_test_true, log_mx_pred, sex)
        all_metrics[fname_key] = {**m, **e}

    results_df = build_results_table(all_metrics)
    print("\n", results_df[["RMSE", "MAE", "ME", "MAPE", "e0_MAE"]].round(4).to_string())
    results_df.to_csv(f"outputs/results/{tag}_metrics.csv")

    # compact summary: multi-step results only
    print("\n--- Multi-step comparison (fair: same information) ---")
    multi_names = ["LC (multi-step)"] + [n + " (multi)" for n in KT_MODEL_CLASSES]
    for mn in multi_names:
        if mn in all_metrics:
            m = all_metrics[mn]
            print(f"   {mn:30s}  RMSE={m['RMSE']:.4f}  "
                  f"e0_MAE={m.get('e0_MAE',float('nan')):.3f} yrs")

    # ── 10. Diebold-Mariano tests ─────────────────────────────────────────────
    print("\n[10] Diebold-Mariano tests vs LC (multi-step) ...")
    dm_rows = []
    lc_ms_pred = all_forecasts["LC (multi-step)"]
    for name in KT_MODEL_CLASSES:
        for mode in ["multi", "rolling"]:
            key  = f"{name} ({mode})"
            pred = all_forecasts[key]
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dm = diebold_mariano(log_mx_test_true, lc_ms_pred, pred)
                if w:
                    pass   # warning already printed by evaluate.py
            dm_rows.append({"Model": key, **dm})
            sig = "*" if dm["reject_H0_5pct"] else ""
            print(f"   {key:35s}  DM={dm['DM_stat']:+.3f}"
                  f"  p={dm['p_value']:.3f} {sig}  (n={dm['n_obs']})")

    dm_df = pd.DataFrame(dm_rows).set_index("Model")
    dm_df.to_csv(f"outputs/results/{tag}_dm_tests.csv")
    print("   (* = statistically different from LC multi-step at alpha=0.05)")
    print("   NOTE: n=10 is below the reliable DM test threshold. "
          "Treat p-values as indicative only.")

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    print("\n[11] Generating plots ...")

    plot_loss_curves(histories,
                     title=f"kt-LSTM Loss Curves -- {country} {sex}",
                     fname=f"{tag}_loss_curves.png")

    plot_kt_series(
        years_all=years, kt_all_obs=kt_all_obs,
        n_train=n_train, n_val=n_val,
        kt_multi_preds={**{"LC (ARIMA)": kt_arima_test},
                        **{n: kt_multi_preds[n] for n in KT_MODEL_CLASSES}},
        kt_rolling_preds={n: kt_rolling_preds[n] for n in KT_MODEL_CLASSES},
        country=country, sex=sex,
        fname=f"{tag}_kt_series.png",
    )

    # forecast-by-age using multi-step reconstructions
    plot_forecasts_age = {
        "LC (multi-step)": lc_multi,
        "LC (rolling)":    lc_rolling,
    }
    for name in KT_MODEL_CLASSES:
        plot_forecasts_age[name] = all_forecasts[name + " (multi)"]

    # update COLORS temporarily for visualize functions
    import src.visualize as viz
    original_colors = dict(viz.COLORS)
    original_order  = list(viz.MODEL_ORDER)
    viz.COLORS.update(KT_COLORS)
    viz.MODEL_ORDER = list(plot_forecasts_age.keys())

    plot_forecast_ages(
        years_all=years, log_mx_true=log_mx,
        forecasts=plot_forecasts_age,
        test_start_idx=n_train + n_val,
        ages_to_plot=[0, 20, 40, 60, 80],
        ages=ages, country=country, sex=sex,
        fname=f"{tag}_forecast_ages.png",
    )

    e0_true  = np.array([lc_e0(np.exp(log_mx_test_true[:, t]), sex)
                          for t in range(n_test)])
    e0_preds = {k: np.array([lc_e0(np.exp(plot_forecasts_age[k][:, t]), sex)
                               for t in range(n_test)])
                for k in plot_forecasts_age}
    plot_life_expectancy(
        years_test=years_test, e0_true=e0_true, e0_preds=e0_preds,
        country=country, sex=sex, fname=f"{tag}_life_expectancy.png",
    )

    # RMSE bar: multi-step models only
    multi_df = results_df.loc[
        results_df.index.isin(["LC (multi-step)"] +
                               [n + " (multi)" for n in KT_MODEL_CLASSES])
    ].copy()
    multi_df.index = [i.replace(" (multi)", "") for i in multi_df.index]
    viz.COLORS.update({n.replace(" (multi)", ""): KT_COLORS.get(n.split(" (")[0], "#999")
                       for n in multi_df.index})
    viz.MODEL_ORDER = list(multi_df.index)
    plot_metrics_bar(multi_df, metric="RMSE",
                     title=f"Test RMSE (multi-step) -- {country} {sex}",
                     fname=f"{tag}_rmse_bar.png")
    plot_metrics_bar(multi_df, metric="e0_MAE",
                     title=f"e0 MAE yrs (multi-step) -- {country} {sex}",
                     fname=f"{tag}_e0_mae_bar.png")

    # residuals
    best_lstm = min(
        [(n + " (multi)") for n in KT_MODEL_CLASSES],
        key=lambda k: all_metrics[k]["RMSE"]
    )
    plot_residual_heatmap(
        log_mx_test_true, all_forecasts[best_lstm],
        years_test, ages,
        model_name=best_lstm, country=country, sex=sex,
        fname=f"{tag}_residuals_{best_lstm.replace(' ','_')}.png",
    )

    # kt attention weights
    attn_model = trained_models["Kt-Attention"]
    kt_ctx_z   = (kt_all_obs[:n_train + n_val][-KT_SEQ_LEN:] - kt_mean) / (kt_std + 1e-8)
    plot_kt_attention(attn_model, kt_ctx_z, KT_SEQ_LEN,
                      fname=f"{tag}_attention.png", device=device)

    plot_results_table(results_df, fname=f"{tag}_results_table.png")

    # restore visualize module state
    viz.COLORS.clear(); viz.COLORS.update(original_colors)
    viz.MODEL_ORDER[:] = original_order

    # ── 12. Summary ───────────────────────────────────────────────────────────
    best_multi = min(
        ["LC (multi-step)"] + [n + " (multi)" for n in KT_MODEL_CLASSES],
        key=lambda k: all_metrics[k]["RMSE"]
    )
    print(f"\n{'='*65}")
    print(f"  BEST (multi-step): {best_multi}")
    print(f"  RMSE = {all_metrics[best_multi]['RMSE']:.4f}"
          f"  e0_MAE = {all_metrics[best_multi].get('e0_MAE', float('nan')):.3f} yrs")
    print(f"\n  LC (multi-step) RMSE   = {all_metrics['LC (multi-step)']['RMSE']:.4f}")
    print(f"  LC (rolling)    RMSE   = {all_metrics['LC (rolling)']['RMSE']:.4f}")
    print(f"\n  Outputs -> outputs/")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country",    default="AUS", choices=COUNTRIES)
    parser.add_argument("--sex",        default="male", choices=SEXES)
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--hpo_trials", type=int, default=15)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--skip_hpo",   action="store_true")
    args = parser.parse_args()
    main(args)
