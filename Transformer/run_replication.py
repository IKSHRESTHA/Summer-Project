"""
run_replication.py --- replication of Wang, Wen, Xiao & Wang (2024),
"Time-series forecasting of mortality rates using transformer",
Scandinavian Actuarial Journal 2024(2), 109-123.

Protocol
--------
- 6 countries (UK*, France, Italy, Denmark, Canada, Finland; *GBRTENW proxy)
- ages 0-100, train 1950-2000, test 2001-2019, male mortality
- models: LC + {RNN, LSTM, CNN, Transformer}
- 10 independent runs per DL model (seeds 0-9); mean and std reported
- training-set metrics: in-sample 1-step predictions
- test-set metrics: recursive multi-step forecast (no test data seen)

Usage
-----
    cd Transformer/
    python run_replication.py [--epochs 100] [--runs 10] [--device cpu]
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from src.data import (COUNTRY_CODES, SEQ_LEN, TRAIN_START, TRAIN_END,
                       TEST_START, TEST_END, prepare_country,
                       predict_train_insample, predict_test_recursive,
                       predict_test_rolling)
from src.models import MODEL_CLASSES
from src.train import train_model
from src.lee_carter import lc_forecast, lc_rolling_forecast
from src.evaluate import metrics, rmse_by_age, rmse_by_year
from src.visualize import (plot_loss_curves, plot_final_year,
                            plot_rmse_profile)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    for d in ["outputs/results", "outputs/figures", "outputs/models"]:
        os.makedirs(d, exist_ok=True)

    countries = list(COUNTRY_CODES)
    ages = np.arange(0, 101)
    t0 = time.time()

    train_rows, test_rows = [], []
    curves_by_country   = {}
    rmse_a_by_country   = {}
    rmse_t_by_country   = {}
    pred2019_by_country = {}
    true2019_by_country = {}

    for country in countries:
        print(f"\n{'='*60}\n  {country}\n{'='*60}")
        loader, bundle = prepare_country(country, seed=0)
        log_mx  = bundle["log_mx"]
        n_train, n_test = bundle["n_train"], bundle["n_test"]
        true_test  = log_mx[:, n_train:]
        true_train = log_mx[:, SEQ_LEN:n_train]   # targets of in-sample windows

        # ---- Lee-Carter (deterministic, single fit) -----------------------
        lc_pred = lc_forecast(log_mx[:, :n_train], h=n_test)
        m = metrics(true_test, lc_pred)
        test_rows.append({"Country": country, "Model": "LC", "run": 0,
                          "Protocol": "recursive", **m})
        print(f"  LC           test RMSE={m['RMSE']:.4f}")

        lc_roll = lc_rolling_forecast(log_mx, n_train, n_test)
        m_roll = metrics(true_test, lc_roll)
        test_rows.append({"Country": country, "Model": "LC", "run": 0,
                          "Protocol": "rolling", **m_roll})
        print(f"  LC (rolling) test RMSE={m_roll['RMSE']:.4f}")

        rmse_a_by_country[country] = {"LC": rmse_by_age(true_test, lc_pred)}
        rmse_t_by_country[country] = {"LC": rmse_by_year(true_test, lc_pred)}
        pred2019_by_country[country] = {"LC": lc_pred[:, -1]}
        true2019_by_country[country] = true_test[:, -1]
        curves_by_country[country] = {}

        # ---- deep learning models, 10 runs each ---------------------------
        for name, cls in MODEL_CLASSES.items():
            curves, ra, rt, p19 = [], [], [], []
            for run in range(args.runs):
                set_seed(run)
                loader, bundle = prepare_country(country, seed=run)
                model = cls(n_ages=101)
                curve = train_model(model, loader, epochs=args.epochs,
                                    lr=1e-3, device=args.device)
                curves.append(curve)

                pred_tr = predict_train_insample(model, bundle, args.device)
                train_rows.append({"Country": country, "Model": name,
                                   "run": run, **metrics(true_train, pred_tr)})

                pred_te = predict_test_recursive(model, bundle, args.device)
                test_rows.append({"Country": country, "Model": name,
                                  "run": run, "Protocol": "recursive",
                                  **metrics(true_test, pred_te)})
                ra.append(rmse_by_age(true_test, pred_te))
                rt.append(rmse_by_year(true_test, pred_te))
                p19.append(pred_te[:, -1])

                pred_ro = predict_test_rolling(model, bundle, args.device)
                test_rows.append({"Country": country, "Model": name,
                                  "run": run, "Protocol": "rolling",
                                  **metrics(true_test, pred_ro)})

                if run == 0:
                    torch.save(model.state_dict(),
                               f"outputs/models/{country}_{name}_run0.pt")

            curves_by_country[country][name]  = np.mean(curves, axis=0)
            rmse_a_by_country[country][name]  = np.mean(ra, axis=0)
            rmse_t_by_country[country][name]  = np.mean(rt, axis=0)
            pred2019_by_country[country][name] = np.mean(p19, axis=0)

            sub = [r for r in test_rows
                   if r["Country"] == country and r["Model"] == name
                   and r["Protocol"] == "recursive"]
            rm = np.mean([r["RMSE"] for r in sub])
            sub_ro = [r for r in test_rows
                      if r["Country"] == country and r["Model"] == name
                      and r["Protocol"] == "rolling"]
            rm_ro = np.mean([r["RMSE"] for r in sub_ro])
            print(f"  {name:12s} test RMSE={rm:.4f} (recursive) "
                  f"{rm_ro:.4f} (rolling)  ({args.runs} runs, "
                  f"{time.time()-t0:.0f}s elapsed)")

    # ---- save per-run and aggregated tables -------------------------------
    train_df = pd.DataFrame(train_rows)
    test_df  = pd.DataFrame(test_rows)
    train_df.to_csv("outputs/results/train_metrics_runs.csv", index=False)
    test_df.to_csv("outputs/results/test_metrics_runs.csv", index=False)

    def agg(df, keys):
        g = df.groupby(keys)[["MAE", "RMSE", "MAPE"]]
        out = g.mean().round(4)
        out_std = g.std().round(4).add_suffix("_std")
        return out.join(out_std)

    agg(train_df, ["Country", "Model"]).to_csv(
        "outputs/results/train_metrics_summary.csv")
    agg(test_df, ["Country", "Model", "Protocol"]).to_csv(
        "outputs/results/test_metrics_summary.csv")

    # Table-4 analogue: average RMSE(a) and RMSE(t) per model
    t4 = []
    for c in countries:
        for mname in ["LC"] + list(MODEL_CLASSES):
            t4.append({"Country": c, "Model": mname,
                       "avg_RMSE_a": float(np.mean(rmse_a_by_country[c][mname])),
                       "avg_RMSE_t": float(np.mean(rmse_t_by_country[c][mname]))})
    pd.DataFrame(t4).to_csv("outputs/results/rmse_age_year_summary.csv",
                            index=False)

    # profiles for figures
    np.savez("outputs/results/profiles.npz",
             **{f"ra_{c}_{m}": rmse_a_by_country[c][m]
                for c in countries for m in rmse_a_by_country[c]},
             **{f"rt_{c}_{m}": rmse_t_by_country[c][m]
                for c in countries for m in rmse_t_by_country[c]})

    # ---- figures -----------------------------------------------------------
    print("\nGenerating figures ...")
    plot_loss_curves(curves_by_country, "loss_curves.png")
    plot_final_year(pred2019_by_country, true2019_by_country, ages, 2019,
                    "prediction_2019.png")
    plot_rmse_profile(rmse_a_by_country, ages, "Age", "RMSE(a)",
                      "rmse_by_age.png")
    years_test = np.arange(TEST_START, TEST_END + 1)
    plot_rmse_profile(rmse_t_by_country, years_test, "Year", "RMSE(t)",
                      "rmse_by_year.png")

    with open("outputs/results/config.json", "w") as f:
        json.dump({"countries": countries, "sex": "male",
                   "ages": "0-100", "seq_len": SEQ_LEN,
                   "train": f"{TRAIN_START}-{TRAIN_END}",
                   "test": f"{TEST_START}-{TEST_END}",
                   "epochs": args.epochs, "runs": args.runs,
                   "lr": 1e-3, "batch_size": 32,
                   "transformer": {"d_model": 32, "heads": 2,
                                    "enc_dec_layers": 1, "ffn": 16,
                                    "dropout": 0.1}}, f, indent=2)

    print(f"\nDone in {time.time()-t0:.0f}s. Outputs -> outputs/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--runs",   type=int, default=10)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
