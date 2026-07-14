"""
run_confirmation.py -- pre-registered confirmatory run on the 14 populations
never used for any design decision (see HYPOTHESES.md, committed first).

Per population (resumable; one JSON + one NPZ per population, skipped if done):
  - both splits (extended: 1950-2009/2010-2019; paper: 1950-2000/2001-2019)
  - LC recursive + rolling-refit
  - {LSTM, Transformer, Performer} x 10 seeds x {recursive, rolling} protocols
  - annually-retrained LSTM arm (extended split, 10 seeds x 10 test years)
  - per-year RMSE saved per seed for Diebold-Mariano testing

Usage:  python run_confirmation.py [--smoke]
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from src.data import (CONFIRMATION_POPS, SPLITS, SEQ_LEN, YEAR_START, YEAR_END,
                       prepare_country, predict_train_insample,
                       predict_test_recursive, predict_test_rolling)
from src.models import MODEL_CLASSES
from src.performer import PerformerModel
from src.train import train_model
from src.lee_carter import lc_forecast, lc_rolling_forecast
from src.evaluate import metrics, rmse_by_year

DL = dict(MODEL_CLASSES)
DL["Performer"] = PerformerModel

OUT = "outputs/confirmation"
os.makedirs(OUT, exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_population(country, sex, epochs, runs, device="cpu"):
    tag = f"{country}_{sex}"
    jpath, npath = f"{OUT}/{tag}.json", f"{OUT}/{tag}_rt.npz"
    if os.path.exists(jpath) and os.path.exists(npath):
        print(f"  {tag}: already done, skipping")
        return

    rows, rt_store = [], {}
    t0 = time.time()

    for split, train_end in SPLITS.items():
        _, bundle = prepare_country(country, train_end=train_end, sex=sex, seed=0)
        log_mx = bundle["log_mx"]
        n_train, n_test = bundle["n_train"], bundle["n_test"]
        true_test  = log_mx[:, n_train:]
        true_train = log_mx[:, SEQ_LEN:n_train]

        lc_rec = lc_forecast(log_mx[:, :n_train], h=n_test)
        lc_rol = lc_rolling_forecast(log_mx, n_train, n_test)
        for proto, pred in [("recursive", lc_rec), ("rolling", lc_rol)]:
            rows.append({"Country": country, "Sex": sex, "Split": split,
                         "Model": "LC", "Protocol": proto, "run": 0,
                         "stage": "test", **metrics(true_test, pred)})
            rt_store[f"rt_{split}_LC_{proto}"] = rmse_by_year(true_test, pred)

        for name, cls in DL.items():
            for run in range(runs):
                set_seed(run)
                loader, bundle = prepare_country(country, train_end=train_end,
                                                 sex=sex, seed=run)
                model = cls(n_ages=101)
                train_model(model, loader, epochs=epochs, lr=1e-3, device=device)

                pred_tr = predict_train_insample(model, bundle, device)
                rows.append({"Country": country, "Sex": sex, "Split": split,
                             "Model": name, "Protocol": "insample", "run": run,
                             "stage": "train", **metrics(true_train, pred_tr)})

                for proto, fn in [("recursive", predict_test_recursive),
                                  ("rolling",   predict_test_rolling)]:
                    pred = fn(model, bundle, device)
                    rows.append({"Country": country, "Sex": sex, "Split": split,
                                 "Model": name, "Protocol": proto, "run": run,
                                 "stage": "test", **metrics(true_test, pred)})
                    rt_store[f"rt_{split}_{name}_{proto}_seed{run}"] = \
                        rmse_by_year(true_test, pred)

    # annually-retrained LSTM arm, extended split
    train_end0 = SPLITS["extended"]
    _, b0 = prepare_country(country, train_end=train_end0, sex=sex, seed=0)
    true_test = b0["log_mx"][:, b0["n_train"]:]
    n_test = b0["n_test"]
    for run in range(runs):
        yearly = []
        for i in range(n_test):
            set_seed(run * 1000 + i)
            loader, bundle = prepare_country(country, train_end=train_end0 + i,
                                             sex=sex, seed=run * 1000 + i)
            model = DL["LSTM"](n_ages=101)
            train_model(model, loader, epochs=epochs, lr=1e-3, device=device)
            z, s = bundle["z_full"], bundle["seq_len"]
            nt = bundle["n_train"]
            with torch.no_grad():
                x = torch.tensor(z[:, nt - s:nt].T[None], dtype=torch.float32)
                yhat_z = model(x).numpy()[0]
            yearly.append(yhat_z * (bundle["std"] + 1e-8) + bundle["mean"])
        pred = np.stack(yearly, axis=1)                    # (A, n_test)
        rows.append({"Country": country, "Sex": sex, "Split": "extended",
                     "Model": "LSTM-retrained", "Protocol": "rolling",
                     "run": run, "stage": "test", **metrics(true_test, pred)})
        rt_store[f"rt_extended_LSTM-retrained_rolling_seed{run}"] = \
            rmse_by_year(true_test, pred)

    with open(jpath, "w") as f:
        json.dump(rows, f)
    np.savez(npath, **rt_store)
    print(f"  {tag}: done in {time.time()-t0:.0f}s "
          f"({len(rows)} metric rows)")


def main(args):
    runs   = 2 if args.smoke else 10
    epochs = 5 if args.smoke else 100
    pops   = CONFIRMATION_POPS[:1] if args.smoke else CONFIRMATION_POPS
    print(f"Confirmation run: {len(pops)} populations, "
          f"{runs} seeds, {epochs} epochs")
    for country, sex in pops:
        run_population(country, sex, epochs, runs)
    print("\nAll populations complete -> outputs/confirmation/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
