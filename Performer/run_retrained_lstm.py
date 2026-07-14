"""
run_retrained_lstm.py --- the missing fair comparison.

The paper's strongest claim is that annually REFITTED Lee-Carter beats every
network. But the rolling networks in the main experiment keep FROZEN weights
and only receive updated inputs. This script levels the field: the LSTM is
RETRAINED each test year on the expanding window (exactly mirroring the
annually refit LC), then makes a 1-step-ahead forecast.

Extended split: for test year 2010+i the model trains on 1950..2009+i
(scaler refit on the same expanded window) and predicts year 2010+i.

6 countries x 10 seeds x 10 test years = 600 trainings.
"""

import os, sys, json, random, time
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from src.data import COUNTRY_CODES, SEQ_LEN, YEAR_START, YEAR_END, prepare_country
from src.models import LSTMModel
from src.train import train_model
from src.evaluate import metrics


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


BASE_TRAIN_END = 2009          # extended split
N_TEST = YEAR_END - BASE_TRAIN_END   # 10
RUNS = 10
EPOCHS = 100

t0 = time.time()
results = {}

for country in COUNTRY_CODES:
    run_rmses = []
    per_run_preds = []
    for run in range(RUNS):
        preds = []
        for i in range(N_TEST):
            set_seed(run * 1000 + i)   # independent init per (run, year)
            loader, b = prepare_country(country,
                                        train_end=BASE_TRAIN_END + i,
                                        seed=run * 1000 + i)
            model = LSTMModel(n_ages=101)
            train_model(model, loader, epochs=EPOCHS, lr=1e-3, device="cpu")
            model.eval()
            z, s, n_tr = b["z_full"], b["seq_len"], b["n_train"]
            with torch.no_grad():
                x = torch.tensor(z[:, n_tr - s:n_tr].T[None],
                                 dtype=torch.float32)
                y = model(x).numpy()[0]
            preds.append(y * (b["std"] + 1e-8) + b["mean"])
        preds = np.stack(preds, axis=1)                    # (101, 10)
        per_run_preds.append(preds)

        _, b0 = prepare_country(country, train_end=BASE_TRAIN_END, seed=0)
        true_test = b0["log_mx"][:, b0["n_train"]:]
        run_rmses.append(metrics(true_test, preds)["RMSE"])

    results[country] = {"rmse_mean": float(np.mean(run_rmses)),
                        "rmse_std":  float(np.std(run_rmses)),
                        "rmse_runs": [float(r) for r in run_rmses]}
    print(f"  {country:8s} retrained-rolling LSTM RMSE = "
          f"{np.mean(run_rmses):.4f} ({np.std(run_rmses):.4f})   "
          f"[{time.time()-t0:.0f}s]", flush=True)

with open("outputs/results/retrained_lstm_extended.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nDone in {time.time()-t0:.0f}s -> outputs/results/retrained_lstm_extended.json")
