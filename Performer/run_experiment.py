"""
run_experiment.py --- Performer mortality-forecasting study.

Compares four models under identical conditions:
  Lee-Carter | LSTM | Transformer (exact attention) | Performer (FAVOR+)

Protocol
--------
- 6 countries (UK*, France, Italy, Denmark, Canada, Finland; *GBRTENW proxy)
- ages 0-100, male mortality, HMD, years 1950-2019
- TWO splits: "extended" (train 1950-2009, test 2010-2019, primary) and
  "paper" (train 1950-2000, test 2001-2019, comparability with Wang et al. 2024)
- 10 independent runs per DL model (seeds 0-9); mean and std reported
- test forecasts under BOTH protocols: recursive (information-fair) and
  rolling 1-step-ahead (observed inputs; LC refit annually)
- metrics: MAE, RMSE, MAPE on log-mortality + RMSE(age), RMSE(year)

Usage
-----
    cd Performer/
    python run_experiment.py [--epochs 100] [--runs 10] [--device cpu]
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

from src.data import (COUNTRY_CODES, SPLITS, SEQ_LEN, YEAR_START, YEAR_END,
                       prepare_country, predict_train_insample,
                       predict_test_recursive, predict_test_rolling)
from src.models import MODEL_CLASSES
from src.performer import PerformerModel
MODEL_CLASSES["Performer"] = PerformerModel
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
    fig_store = {}          # {split: {kind: {country: {model: array}}}}

    for split_name, train_end in SPLITS.items():
        print(f"\n{'#'*60}\n#  SPLIT: {split_name}  "
              f"(train {YEAR_START}-{train_end}, test {train_end+1}-{YEAR_END})"
              f"\n{'#'*60}")
        store = fig_store[split_name] = {
            "curves": {}, "ra": {}, "rt": {}, "p_final": {}, "t_final": {}}

        for country in countries:
            print(f"\n{'='*60}\n  {country} [{split_name}]\n{'='*60}")
            _, bundle = prepare_country(country, train_end=train_end, seed=0)
            log_mx  = bundle["log_mx"]
            n_train, n_test = bundle["n_train"], bundle["n_test"]
            true_test  = log_mx[:, n_train:]
            true_train = log_mx[:, SEQ_LEN:n_train]

            # ---- Lee-Carter (deterministic) --------------------------------
            lc_pred = lc_forecast(log_mx[:, :n_train], h=n_test)
            m = metrics(true_test, lc_pred)
            test_rows.append({"Split": split_name, "Country": country,
                              "Model": "LC", "run": 0,
                              "Protocol": "recursive", **m})
            print(f"  LC           test RMSE={m['RMSE']:.4f}")

            lc_roll = lc_rolling_forecast(log_mx, n_train, n_test)
            m_roll = metrics(true_test, lc_roll)
            test_rows.append({"Split": split_name, "Country": country,
                              "Model": "LC", "run": 0,
                              "Protocol": "rolling", **m_roll})
            print(f"  LC (rolling) test RMSE={m_roll['RMSE']:.4f}")

            store["ra"][country] = {"LC": rmse_by_age(true_test, lc_pred)}
            store["rt"][country] = {"LC": rmse_by_year(true_test, lc_pred)}
            store["p_final"][country] = {"LC": lc_pred[:, -1]}
            store["t_final"][country] = true_test[:, -1]
            store["curves"][country] = {}

            # ---- deep learning models --------------------------------------
            for name, cls in MODEL_CLASSES.items():
                curves, ra, rt, pf = [], [], [], []
                for run in range(args.runs):
                    set_seed(run)
                    loader, bundle = prepare_country(country,
                                                     train_end=train_end,
                                                     seed=run)
                    model = cls(n_ages=101)
                    curve = train_model(model, loader, epochs=args.epochs,
                                        lr=1e-3, device=args.device)
                    curves.append(curve)

                    pred_tr = predict_train_insample(model, bundle, args.device)
                    train_rows.append({"Split": split_name, "Country": country,
                                       "Model": name, "run": run,
                                       **metrics(true_train, pred_tr)})

                    pred_te = predict_test_recursive(model, bundle, args.device)
                    test_rows.append({"Split": split_name, "Country": country,
                                      "Model": name, "run": run,
                                      "Protocol": "recursive",
                                      **metrics(true_test, pred_te)})
                    ra.append(rmse_by_age(true_test, pred_te))
                    rt.append(rmse_by_year(true_test, pred_te))
                    pf.append(pred_te[:, -1])

                    pred_ro = predict_test_rolling(model, bundle, args.device)
                    test_rows.append({"Split": split_name, "Country": country,
                                      "Model": name, "run": run,
                                      "Protocol": "rolling",
                                      **metrics(true_test, pred_ro)})

                    if run == 0:
                        torch.save(model.state_dict(),
                                   f"outputs/models/{split_name}_{country}_{name}_run0.pt")

                store["curves"][country][name]  = np.mean(curves, axis=0)
                store["ra"][country][name]      = np.mean(ra, axis=0)
                store["rt"][country][name]      = np.mean(rt, axis=0)
                store["p_final"][country][name] = np.mean(pf, axis=0)

                sub = [r for r in test_rows
                       if r["Split"] == split_name and r["Country"] == country
                       and r["Model"] == name]
                rm    = np.mean([r["RMSE"] for r in sub
                                 if r["Protocol"] == "recursive"])
                rm_ro = np.mean([r["RMSE"] for r in sub
                                 if r["Protocol"] == "rolling"])
                print(f"  {name:12s} test RMSE={rm:.4f} (recursive) "
                      f"{rm_ro:.4f} (rolling)  ({args.runs} runs, "
                      f"{time.time()-t0:.0f}s elapsed)")

    # ---- save tables --------------------------------------------------------
    train_df = pd.DataFrame(train_rows)
    test_df  = pd.DataFrame(test_rows)
    train_df.to_csv("outputs/results/train_metrics_runs.csv", index=False)
    test_df.to_csv("outputs/results/test_metrics_runs.csv", index=False)

    def agg(df, keys):
        g = df.groupby(keys)[["MAE", "RMSE", "MAPE"]]
        return g.mean().round(4).join(g.std().round(4).add_suffix("_std"))

    agg(train_df, ["Split", "Country", "Model"]).to_csv(
        "outputs/results/train_metrics_summary.csv")
    agg(test_df, ["Split", "Country", "Model", "Protocol"]).to_csv(
        "outputs/results/test_metrics_summary.csv")

    # average RMSE(a) / RMSE(t) per split, country, model
    t4 = [{"Split": s, "Country": c, "Model": m,
           "avg_RMSE_a": float(np.mean(fig_store[s]["ra"][c][m])),
           "avg_RMSE_t": float(np.mean(fig_store[s]["rt"][c][m]))}
          for s in SPLITS for c in countries for m in fig_store[s]["ra"][c]]
    pd.DataFrame(t4).to_csv("outputs/results/rmse_age_year_summary.csv",
                            index=False)

    # persist all figure inputs so plots can be re-styled without retraining
    flat = {}
    for s in SPLITS:
        st = fig_store[s]
        for c in countries:
            flat[f"tfinal_{s}_{c}"] = st["t_final"][c]
            for m in st["ra"][c]:
                flat[f"ra_{s}_{c}_{m}"] = st["ra"][c][m]
                flat[f"rt_{s}_{c}_{m}"] = st["rt"][c][m]
                flat[f"pfinal_{s}_{c}_{m}"] = st["p_final"][c][m]
            for m in st["curves"][c]:
                flat[f"curve_{s}_{c}_{m}"] = st["curves"][c][m]
    np.savez("outputs/results/figure_data.npz", **flat)

    # ---- figures (per split) ------------------------------------------------
    print("\nGenerating figures ...")
    for s, train_end in SPLITS.items():
        st = fig_store[s]
        final_year = YEAR_END
        years_test = np.arange(train_end + 1, YEAR_END + 1)
        plot_loss_curves(st["curves"], f"loss_curves_{s}.png")
        plot_final_year(st["p_final"], st["t_final"], ages, final_year,
                        f"prediction_{final_year}_{s}.png")
        plot_rmse_profile(st["ra"], ages, "Age", "RMSE(a)",
                          f"rmse_by_age_{s}.png")
        plot_rmse_profile(st["rt"], years_test, "Year", "RMSE(t)",
                          f"rmse_by_year_{s}.png")

    with open("outputs/results/config.json", "w") as f:
        json.dump({"countries": countries, "sex": "male", "ages": "0-100",
                   "seq_len": SEQ_LEN,
                   "splits": {s: f"train {YEAR_START}-{e}, test {e+1}-{YEAR_END}"
                              for s, e in SPLITS.items()},
                   "epochs": args.epochs, "runs": args.runs,
                   "lr": 1e-3, "batch_size": 32,
                   "models": list(MODEL_CLASSES) + ["LC"],
                   "transformer": {"d_model": 32, "heads": 2,
                                    "enc_dec_layers": 1, "ffn": 16,
                                    "dropout": 0.1},
                   "performer": {"m_features": 48,
                                  "features": "positive orthogonal, fixed per seed"}},
                  f, indent=2)

    print(f"\nDone in {time.time()-t0:.0f}s. Outputs -> outputs/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--runs",   type=int, default=10)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
