"""
run_pandemic.py -- pre-registered COVID-19 shock/recovery/revision study.
See PANDEMIC_HYPOTHESES.md (committed before this script was first run).

Per population (resumable; JSON + NPZ per population):
  A. SHOCK: LC + {LSTM, Transformer, Performer} x 10 seeds trained through
     2019; recursive forecasts 2020..year_end; per-year, per-age errors and
     raw mean forecast matrices saved.
  B. RECOVERY: maintained models at origins 2015..min(2022, last-1):
     rolling-refit LC and annually-retrained LSTM (10 seeds); one-step
     forecasts per origin.
  C. REVISION: at each origin both maintained models emit a 5-year forecast
     path (LC: drift extrapolation; LSTM: recursive rollout); saved for the
     revision-stability analysis.

Usage:  python run_pandemic.py [--smoke]
"""

import argparse, json, os, random, sys, time
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from src.data import (COUNTRY_CODES, SEQ_LEN, YEAR_START, prepare_country,
                       load_log_mx, predict_test_recursive)
from src.models import MODEL_CLASSES
from src.performer import PerformerModel
from src.train import train_model
from src.lee_carter import fit_lee_carter, lc_forecast
from src.evaluate import metrics, rmse_by_year, rmse_by_age

DL = dict(MODEL_CLASSES); DL["Performer"] = PerformerModel
ALL_POPS = [(c, s) for c in COUNTRY_CODES for s in ["male", "female"]]
LAST_YEAR = {"Australia": 2021, "UK": 2022}          # else capped at 2023
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
PATH_H = 5                                            # revision path length


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def last_year(country):
    return LAST_YEAR.get(country, 2023)


@torch.no_grad()
def recursive_path(model, z_full, mean, std, n_ctx_end, h, s=SEQ_LEN):
    """Recursive h-step path in log-mortality from context ending at n_ctx_end."""
    buf = [z_full[:, t].astype(np.float32) for t in range(n_ctx_end - s, n_ctx_end)]
    out = []
    for _ in range(h):
        x = torch.tensor(np.stack(buf[-s:])[None])
        y = model(x).numpy()[0]
        out.append(y); buf.append(y.astype(np.float32))
    return np.stack(out).T * (std[:, None] + 1e-8) + mean[:, None]   # (A, h)


def run_population(country, sex, epochs, runs):
    tag = f"{country}_{sex}"
    jpath, npath = f"{OUT}/{tag}.json", f"{OUT}/{tag}_arrays.npz"
    if os.path.exists(jpath) and os.path.exists(npath):
        print(f"  {tag}: done, skipping"); return
    t0 = time.time()
    ye = last_year(country)
    rows, arrays = [], {}

    # ---- A. SHOCK: frozen models trained through 2019 ----------------------
    _, b = prepare_country(country, train_end=2019, sex=sex, year_end=ye, seed=0)
    log_mx = b["log_mx"]; n_train = b["n_train"]; n_shock = b["n_test"]
    true_shock = log_mx[:, n_train:]                       # (A, n_shock)
    arrays["true_shock"] = true_shock

    lc_fit = fit_lee_carter(log_mx[:, :n_train])
    lc_pred = lc_forecast(log_mx[:, :n_train], h=n_shock)
    arrays["shock_LC"] = lc_pred
    rows.append({"arm": "shock", "Model": "LC", "run": 0,
                 **metrics(true_shock, lc_pred)})
    arrays["shock_rt_LC"] = rmse_by_year(true_shock, lc_pred)
    arrays["shock_ra_LC"] = rmse_by_age(true_shock, lc_pred)

    for name, cls in DL.items():
        preds = []
        for run in range(runs):
            set_seed(run)
            loader, b = prepare_country(country, train_end=2019, sex=sex,
                                        year_end=ye, seed=run)
            model = cls(n_ages=101)
            train_model(model, loader, epochs=epochs, lr=1e-3)
            p = predict_test_recursive(model, b)
            preds.append(p)
            rows.append({"arm": "shock", "Model": name, "run": run,
                         **metrics(true_shock, p)})
        pm = np.mean(preds, axis=0)
        arrays[f"shock_{name}"] = pm
        arrays[f"shock_rt_{name}"] = rmse_by_year(true_shock, pm)
        arrays[f"shock_ra_{name}"] = rmse_by_age(true_shock, pm)

    # ---- B + C. MAINTAINED models: origins 2015 .. ye-1 --------------------
    origins = list(range(2015, ye))
    for o in origins:
        _, bo = prepare_country(country, train_end=o, sex=sex,
                                year_end=ye, seed=0)
        lmx = bo["log_mx"]; nt = bo["n_train"]
        true_next = lmx[:, nt]                             # year o+1
        h = min(PATH_H, lmx.shape[1] - nt)                 # path may be capped

        lc_path = lc_forecast(lmx[:, :nt], h=PATH_H)       # path beyond data ok
        arrays[f"path_LC_{o}"] = lc_path
        rows.append({"arm": "maintain", "Model": "LC", "origin": o, "run": 0,
                     **metrics(true_next[:, None], lc_path[:, :1])})

        lstm_paths = []
        for run in range(runs):
            set_seed(run * 1000 + o)
            loader, bo = prepare_country(country, train_end=o, sex=sex,
                                         year_end=ye, seed=run * 1000 + o)
            model = DL["LSTM"](n_ages=101)
            train_model(model, loader, epochs=epochs, lr=1e-3)
            path = recursive_path(model, bo["z_full"], bo["mean"], bo["std"],
                                  bo["n_train"], PATH_H)
            lstm_paths.append(path)
            rows.append({"arm": "maintain", "Model": "LSTM-retrained",
                         "origin": o, "run": run,
                         **metrics(true_next[:, None], path[:, :1])})
        arrays[f"path_LSTM_{o}"] = np.mean(lstm_paths, axis=0)
        arrays[f"path_LSTM_std_{o}"] = np.std(lstm_paths, axis=0)

    with open(jpath, "w") as f:
        json.dump({"country": country, "sex": sex, "last_year": ye,
                   "origins": origins, "rows": rows}, f)
    np.savez(npath, **arrays)
    print(f"  {tag}: done in {time.time()-t0:.0f}s ({len(rows)} rows)")


def main(a):
    runs, epochs = (2, 5) if a.smoke else (10, 100)
    pops = ALL_POPS[:1] if a.smoke else ALL_POPS
    print(f"Pandemic study: {len(pops)} populations, {runs} seeds, {epochs} epochs")
    for c, s in pops:
        run_population(c, s, epochs, runs)
    print("\nComplete -> outputs/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
