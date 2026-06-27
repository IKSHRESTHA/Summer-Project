"""
run_all.py  —  LSTM Mortality Forecasting (v2, industry-best-practice edition)

Usage
-----
    cd LSTM/
    python run_all.py [--country AUS] [--sex male] [--epochs 100]
                      [--hpo_trials 20] [--cv_folds 3] [--mc_samples 200]
                      [--device cpu] [--skip_hpo] [--skip_cv]

Pipeline
--------
1.  Load HMD data for the chosen country/sex.
2.  [HPO]  Optuna search (20 trials per model) → best hyperparameters.
3.  [TRAIN] Re-train all 5 LSTMs with best params (full 100 epochs).
4.  [LC]   Fit Lee-Carter + both multi-step and rolling 1-step-ahead forecast.
5.  [EVAL] Test-set metrics: MAE, RMSE, ME, MAPE, SMAPE, e0 MAE/RMSE.
6.  [DM]   Diebold-Mariano significance tests vs Lee-Carter baseline.
7.  [UQ]   Monte Carlo Dropout: 95% prediction intervals for each LSTM.
8.  [CV]   Walk-forward cross-validation (3 folds, expanding window).
9.  [PLOT] All figures saved to outputs/figures/.
10. [SAVE] Results CSV to outputs/results/; model weights to outputs/models/.
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from src.data       import make_loaders, COUNTRIES, SEXES, SEQ_LEN
from src.train      import train_model, predict_all, destandardise
from src.lee_carter import (fit_lee_carter, lc_forecast, project_rates,
                             lc_rolling_forecast, life_expectancy as lc_e0)
from src.evaluate   import (compute_metrics, compute_e0_errors,
                             compute_age_stratified_rmse,
                             diebold_mariano, build_results_table)
from src.tuning     import hpo_all_models
from src.crossval   import make_folds, crossval_model
from src.uncertainty import mc_predict_destd
from src.visualize  import (plot_loss_curves, plot_forecast_ages,
                             plot_life_expectancy, plot_metrics_bar,
                             plot_residual_heatmap, plot_results_table,
                             plot_attention_weights, MODEL_ORDER)

from src.models.vanilla_lstm       import VanillaLSTM
from src.models.stacked_lstm       import StackedLSTM
from src.models.bidirectional_lstm import BidirectionalLSTM
from src.models.cnn_lstm           import CNNLSTM
from src.models.attention_lstm     import AttentionLSTM


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


MODEL_CLASSES = {
    "Vanilla LSTM":   VanillaLSTM,
    "Stacked LSTM":   StackedLSTM,
    "BiLSTM":         BidirectionalLSTM,
    "CNN-LSTM":       CNNLSTM,
    "Attention LSTM": AttentionLSTM,
}


def build_model(name: str, n_ages: int, params: dict):
    import inspect
    cls = MODEL_CLASSES[name]
    sig = inspect.signature(cls.__init__)
    kwargs = {"n_ages": n_ages,
              "hidden_size": params.get("hidden_size", 128),
              "dropout":     params.get("dropout", 0.2)}
    return cls(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def get_test_preds(model, loaders, meta, device):
    """
    De-standardise test DataLoader predictions to log(mx).
    Test loader produces exactly n_test samples (bug-fixed data.py).
    """
    preds_z  = predict_all(model, loaders["test"], device)   # (n_test, n_ages)
    preds_lm = destandardise(preds_z, meta["mean"], meta["std"]).T  # (n_ages, n_test)
    return preds_lm


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(42)
    for d in ["outputs/models", "outputs/results", "outputs/figures"]:
        os.makedirs(d, exist_ok=True)

    country = args.country
    sex     = args.sex
    device  = args.device
    tag     = f"{country}_{sex}"

    print(f"\n{'='*65}")
    print(f"  LSTM Mortality Forecasting v2 -- {country} ({sex})")
    print(f"  Industry best-practice: HPO + CV + UQ + DM tests")
    print(f"{'='*65}")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    loaders, meta = make_loaders(country, sex, batch_size=32, seq_len=SEQ_LEN)
    years   = meta["years"]
    ages    = meta["ages"]
    log_mx  = meta["log_mx_raw"]
    n_ages  = len(ages)
    n_test  = meta["n_test"]
    T_full  = len(years)
    log_mx_test_true = log_mx[:, -n_test:]
    years_test       = years[-n_test:]
    print(f"   Ages {ages[0]}-{ages[-1]} | Years {years[0]}-{years[-1]}")
    print(f"   Train={meta['n_train']}  Val={meta['n_val']}  Test={n_test}")

    # ── 2. HPO ────────────────────────────────────────────────────────────────
    if args.skip_hpo:
        print("\n[2] HPO skipped -- using defaults")
        best_params = {n: {"hidden_size": 128, "dropout": 0.2,
                           "lr": 1e-3, "weight_decay": 1e-4}
                       for n in MODEL_CLASSES}
    else:
        print(f"\n[2] Hyperparameter optimisation ({args.hpo_trials} trials/model) ...")
        best_params = hpo_all_models(MODEL_CLASSES, loaders, n_ages=n_ages,
                                     n_trials=args.hpo_trials,
                                     hpo_epochs=min(40, args.epochs),
                                     device=device)
        with open(f"outputs/results/{tag}_hpo.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"   HPO complete. Results saved to outputs/results/{tag}_hpo.json")

    # ── 3. Lee-Carter ─────────────────────────────────────────────────────────
    print("\n[3] Fitting Lee-Carter baselines ...")
    log_mx_train = log_mx[:, :meta["n_train"]]
    lc = fit_lee_carter(log_mx_train)
    print(f"   Variance explained (1st SVD): {lc['var_explained']:.1%}")

    # Multi-step LC (original approach — LC doesn't see test data)
    kt_proj     = lc_forecast(lc["kt"], h=meta["n_val"] + n_test)
    lc_multi    = project_rates(lc, kt_proj[-n_test:])

    # Rolling 1-step-ahead LC (fair comparison with LSTM's oracle context)
    lc_rolling  = lc_rolling_forecast(log_mx, meta["n_train"], meta["n_val"], n_test)

    all_forecasts = {"Lee-Carter (multi-step)":  lc_multi,
                     "Lee-Carter (1-step)":       lc_rolling}
    all_metrics   = {}

    for lc_name, lc_pred in [("Lee-Carter (multi-step)", lc_multi),
                               ("Lee-Carter (1-step)",    lc_rolling)]:
        m = compute_metrics(log_mx_test_true, lc_pred)
        e = compute_e0_errors(log_mx_test_true, lc_pred, sex)
        all_metrics[lc_name] = {**m, **e}
        print(f"   {lc_name:28s} RMSE={m['RMSE']:.4f}  e0_MAE={e['e0_MAE']:.3f} yrs")

    # ── 4. Train LSTM models ──────────────────────────────────────────────────
    print(f"\n[4] Training 5 LSTM models ({args.epochs} epochs, AdamW+LayerNorm+Residual) ...")
    histories  = {}
    trained_models = {}

    for name, cls in MODEL_CLASSES.items():
        params = best_params[name]
        print(f"\n  -- {name}  (hidden={params.get('hidden_size',128)}"
              f"  drop={params.get('dropout',0.2):.2f}"
              f"  lr={params.get('lr',1e-3):.1e}"
              f"  wd={params.get('weight_decay',1e-4):.1e}) --")

        model = build_model(name, n_ages, params)
        hist  = train_model(model, loaders,
                            epochs=args.epochs,
                            lr=params.get("lr", 1e-3),
                            weight_decay=params.get("weight_decay", 1e-4),
                            patience=15, device=device, verbose=False)
        # model already has best weights loaded (train_model restores before return)
        trained_models[name] = model
        histories[name]      = hist

        torch.save(hist["best_model_state"],
                   f"outputs/models/{tag}_{name.replace(' ','_')}.pt")

        pred_lm = get_test_preds(model, loaders, meta, device)
        all_forecasts[name] = pred_lm

        m = compute_metrics(log_mx_test_true, pred_lm)
        e = compute_e0_errors(log_mx_test_true, pred_lm, sex)
        all_metrics[name] = {**m, **e}
        print(f"   best_epoch={hist['best_epoch']:3d}"
              f"  RMSE={m['RMSE']:.4f}  e0_MAE={e['e0_MAE']:.3f} yrs"
              f"  MAPE={m['MAPE']:.2f}%")

    # ── 5. Results table ──────────────────────────────────────────────────────
    print("\n[5] Test-set results ...")
    results_df = build_results_table(all_metrics)
    print("\n", results_df.round(4).to_string())
    results_df.to_csv(f"outputs/results/{tag}_metrics.csv")

    # ── 6. Diebold-Mariano tests ──────────────────────────────────────────────
    print("\n[6] Diebold-Mariano significance tests vs Lee-Carter (1-step) ...")
    dm_rows = []
    for name in list(MODEL_CLASSES.keys()):
        dm = diebold_mariano(log_mx_test_true, lc_rolling,
                             all_forecasts[name])
        dm_rows.append({"Model": name, **dm})
        sig = "*" if dm["reject_H0_5pct"] else ""
        print(f"   {name:20s}  DM={dm['DM_stat']:+.3f}  p={dm['p_value']:.3f} {sig}")
    dm_df = pd.DataFrame(dm_rows).set_index("Model")
    dm_df.to_csv(f"outputs/results/{tag}_dm_tests.csv")
    print("   (* = significantly different from LC at alpha=0.05)")

    # ── 7. MC Dropout uncertainty ─────────────────────────────────────────────
    print(f"\n[7] Monte Carlo Dropout uncertainty ({args.mc_samples} samples) ...")
    mc_intervals = {}
    for name, model in trained_models.items():
        mc = mc_predict_destd(model, loaders["test"],
                              meta["mean"], meta["std"],
                              n_samples=args.mc_samples, device=device)
        mc_intervals[name] = mc
    print("   MC Dropout complete.")

    # ── 8. Walk-forward cross-validation ──────────────────────────────────────
    if args.skip_cv:
        print("\n[8] CV skipped.")
        cv_results = {}
    else:
        print(f"\n[8] Walk-forward cross-validation ({args.cv_folds} folds) ...")
        folds = make_folds(T_full, n_folds=args.cv_folds)
        cv_results = {}
        for name, cls in MODEL_CLASSES.items():
            params = best_params[name]
            cv = crossval_model(cls, params, log_mx, meta["mean"], meta["std"],
                                folds, n_ages=n_ages,
                                epochs=min(60, args.epochs),
                                device=device)
            cv_results[name] = cv
            print(f"   {name:20s}  CV RMSE={cv['rmse_mean']:.4f} ± {cv['rmse_std']:.4f}")
        cv_df = pd.DataFrame({n: {"cv_rmse_mean": v["rmse_mean"],
                                   "cv_rmse_std":  v["rmse_std"],
                                   "cv_mae_mean":  v["mae_mean"],
                                   "cv_mae_std":   v["mae_std"]}
                              for n, v in cv_results.items()}).T
        cv_df.to_csv(f"outputs/results/{tag}_cv.csv")

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    print("\n[9] Generating plots ...")

    plot_loss_curves(histories, title=f"Loss Curves -- {country} {sex}",
                     fname=f"{tag}_loss_curves.png")

    # for forecast plots use LSTM forecasts + both LC variants
    plot_forecasts = {k: v for k, v in all_forecasts.items()}
    plot_forecast_ages(
        years_all=years, log_mx_true=log_mx,
        forecasts=plot_forecasts,
        test_start_idx=T_full - n_test,
        ages_to_plot=[0, 20, 40, 60, 80],
        ages=ages, country=country, sex=sex,
        fname=f"{tag}_forecast_ages.png",
    )

    e0_true  = np.array([lc_e0(np.exp(log_mx_test_true[:, t]), sex) for t in range(n_test)])
    e0_preds = {name: np.array([lc_e0(np.exp(all_forecasts[name][:, t]), sex)
                                for t in range(n_test)])
                for name in all_forecasts}
    plot_life_expectancy(years_test=years_test, e0_true=e0_true, e0_preds=e0_preds,
                         country=country, sex=sex, fname=f"{tag}_life_expectancy.png")

    plot_metrics_bar(results_df, metric="RMSE",
                     title=f"Test RMSE -- {country} {sex}",
                     fname=f"{tag}_rmse_bar.png")
    plot_metrics_bar(results_df, metric="e0_MAE",
                     title=f"e0 MAE (yrs) -- {country} {sex}",
                     fname=f"{tag}_e0_mae_bar.png")

    # Residual heatmaps for LC (1-step) and best LSTM
    best_lstm = [m for m in results_df.index if m in MODEL_CLASSES][0]
    for mname, mpred in [("Lee-Carter_1step", lc_rolling),
                          (best_lstm,           all_forecasts[best_lstm])]:
        plot_residual_heatmap(log_mx_test_true, mpred, years_test, ages,
                              model_name=mname, country=country, sex=sex,
                              fname=f"{tag}_residuals_{mname.replace(' ','_')}.png")

    # Attention weights
    attn_model = trained_models["Attention LSTM"]
    sample_x   = next(iter(loaders["test"]))[0].numpy()
    plot_attention_weights(attn_model, sample_x, seq_len=SEQ_LEN,
                           fname=f"{tag}_attention_weights.png")

    plot_results_table(results_df, fname=f"{tag}_results_table.png")

    # ── 10. Summary ───────────────────────────────────────────────────────────
    best_model = results_df.iloc[0].name
    print(f"\n{'='*65}")
    print(f"  BEST MODEL: {best_model}  "
          f"(test RMSE={results_df['RMSE'].iloc[0]:.4f})")
    print(f"  All outputs -> outputs/")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country",    default="AUS", choices=COUNTRIES)
    parser.add_argument("--sex",        default="male", choices=SEXES)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--hpo_trials", type=int, default=20)
    parser.add_argument("--cv_folds",   type=int, default=3)
    parser.add_argument("--mc_samples", type=int, default=200)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--skip_hpo",   action="store_true")
    parser.add_argument("--skip_cv",    action="store_true")
    args = parser.parse_args()
    main(args)
