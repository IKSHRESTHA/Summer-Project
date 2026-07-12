"""
retrain_bilstm.py  —  Retrain only BidirectionalLSTM and KtBiLSTM after
the backward-direction bug fix, using the original HPO params.

Usage
-----
    cd LSTM/
    python retrain_bilstm.py
"""

import os
import sys
import json
import random
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from src.data         import make_loaders, SEQ_LEN
from src.data_kt      import (KT_SEQ_LEN, project_kt, make_kt_loaders,
                               predict_kt_multistep, predict_kt_rolling)
from src.train        import train_model, predict_all, destandardise
from src.lee_carter   import fit_lee_carter, lc_forecast, lc_rolling_forecast
from src.evaluate     import compute_metrics, compute_e0_errors

from src.models.bidirectional_lstm import BidirectionalLSTM
from src.models.kt_lstm            import KtBiLSTM


COUNTRY, SEX = "AUS", "male"
TAG          = f"{COUNTRY}_{SEX}"
EPOCHS       = 100
KT_EPOCHS    = 200
DEVICE       = "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    set_seed()

    # ── load HPO params saved from original run ────────────────────────────
    with open(f"outputs/results/{TAG}_hpo.json") as f:
        hpo = json.load(f)
    bilstm_params = hpo["BiLSTM"]
    print(f"BiLSTM HPO params: {bilstm_params}")

    # KtBiLSTM: no HPO json was saved; use same defaults as original run_kt.py
    kt_params = {"hidden_size": 32, "dropout": 0.1, "lr": 5e-3, "weight_decay": 1e-4}
    print(f"KtBiLSTM params  : {kt_params}\n")

    # ── 1. Data ────────────────────────────────────────────────────────────
    loaders, meta = make_loaders(COUNTRY, SEX, batch_size=32, seq_len=SEQ_LEN)
    log_mx    = meta["log_mx_raw"]
    n_ages    = len(meta["ages"])
    n_train   = meta["n_train"]
    n_val     = meta["n_val"]
    n_test    = meta["n_test"]
    log_mx_test_true = log_mx[:, -n_test:]

    # ── 2. Retrain BidirectionalLSTM ───────────────────────────────────────
    print("=" * 55)
    print("  Retraining BidirectionalLSTM (direct, 100-dim)")
    print("=" * 55)
    model = BidirectionalLSTM(
        n_ages=n_ages,
        hidden_size=bilstm_params["hidden_size"],
        dropout=bilstm_params["dropout"],
    )
    hist = train_model(
        model, loaders,
        epochs=EPOCHS,
        lr=bilstm_params["lr"],
        weight_decay=bilstm_params["weight_decay"],
        patience=15, device=DEVICE, verbose=False,
    )
    print(f"  best_epoch={hist['best_epoch']}  best_val={min(hist['val_loss']):.5f}")

    preds_z  = predict_all(model, loaders["test"], DEVICE)
    preds_lm = destandardise(preds_z, meta["mean"], meta["std"]).T
    m = compute_metrics(log_mx_test_true, preds_lm)
    e = compute_e0_errors(log_mx_test_true, preds_lm, SEX)
    print(f"  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  ME={m['ME']:.4f}  e0_MAE={e['e0_MAE']:.3f} yrs")

    torch.save(hist["best_model_state"],
               f"outputs/models/{TAG}_BiLSTM_fixed.pt")

    new_bilstm = {"RMSE": m["RMSE"], "MAE": m["MAE"], "ME": m["ME"],
                  "MAPE": m["MAPE"], "e0_MAE": e["e0_MAE"], "e0_RMSE": e["e0_RMSE"]}

    # ── 3. Retrain KtBiLSTM ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Retraining KtBiLSTM (kt-factor, 1-dim)")
    print("=" * 55)

    log_mx_train = log_mx[:, :n_train]
    lc = fit_lee_carter(log_mx_train)
    ax, bx, kt_train = lc["ax"], lc["bx"], lc["kt"]

    kt_val_proj  = project_kt(log_mx[:, n_train : n_train + n_val], ax, bx)
    kt_test_proj = project_kt(log_mx_test_true, ax, bx)
    kt_trainval  = np.concatenate([kt_train, kt_val_proj])
    kt_all_obs   = np.concatenate([kt_train, kt_val_proj, kt_test_proj])

    loaders_kt, kt_stats = make_kt_loaders(
        kt_trainval, n_train, n_val, batch_size=8, seq_len=KT_SEQ_LEN, seed=42,
    )
    kt_mean, kt_std = kt_stats["mean"], kt_stats["std"]

    kt_model = KtBiLSTM(hidden_size=kt_params["hidden_size"],
                         dropout=kt_params["dropout"])
    kt_hist = train_model(
        kt_model, loaders_kt,
        epochs=KT_EPOCHS,
        lr=kt_params["lr"],
        weight_decay=kt_params["weight_decay"],
        patience=25, device=DEVICE, verbose=False,
    )
    print(f"  best_epoch={kt_hist['best_epoch']}  best_val={min(kt_hist['val_loss']):.5f}")

    torch.save(kt_hist["best_model_state"],
               f"outputs/models/{TAG}_kt_KtBiLSTM_fixed.pt")

    def reconstruct(kt_pred):
        return ax[:, None] + bx[:, None] * kt_pred[None, :]

    # multi-step
    kt_ms   = predict_kt_multistep(kt_model, kt_trainval, n_steps=n_test,
                                    kt_mean=kt_mean, kt_std=kt_std,
                                    seq_len=KT_SEQ_LEN, device=DEVICE)
    lm_ms   = reconstruct(kt_ms)
    m_ms    = compute_metrics(log_mx_test_true, lm_ms)
    e_ms    = compute_e0_errors(log_mx_test_true, lm_ms, SEX)
    print(f"  [multi]   RMSE={m_ms['RMSE']:.4f}  ME={m_ms['ME']:.4f}  e0_MAE={e_ms['e0_MAE']:.3f}")

    # rolling 1-step
    kt_roll = predict_kt_rolling(kt_model, kt_all_obs, n_train, n_val, n_test,
                                  kt_mean=kt_mean, kt_std=kt_std,
                                  seq_len=KT_SEQ_LEN, device=DEVICE)
    lm_roll = reconstruct(kt_roll)
    m_roll  = compute_metrics(log_mx_test_true, lm_roll)
    e_roll  = compute_e0_errors(log_mx_test_true, lm_roll, SEX)
    print(f"  [rolling] RMSE={m_roll['RMSE']:.4f}  ME={m_roll['ME']:.4f}  e0_MAE={e_roll['e0_MAE']:.3f}")

    new_kt_bilstm = {
        "multi":   {"RMSE": m_ms["RMSE"],   "MAE": m_ms["MAE"],   "ME": m_ms["ME"],
                    "MAPE": m_ms["MAPE"],   "e0_MAE": e_ms["e0_MAE"]},
        "rolling": {"RMSE": m_roll["RMSE"], "MAE": m_roll["MAE"], "ME": m_roll["ME"],
                    "MAPE": m_roll["MAPE"], "e0_MAE": e_roll["e0_MAE"]},
    }

    # ── 4. Summary of changes ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  UPDATED METRICS (fixed vs old)")
    print("=" * 55)
    print(f"\n  BiLSTM (direct):")
    print(f"    OLD  RMSE=0.2693  MAE=0.2072  ME=-0.1845  e0_MAE=1.904")
    print(f"    NEW  RMSE={new_bilstm['RMSE']:.4f}  MAE={new_bilstm['MAE']:.4f}  "
          f"ME={new_bilstm['ME']:.4f}  e0_MAE={new_bilstm['e0_MAE']:.3f}")

    print(f"\n  KtBiLSTM (multi-step):")
    print(f"    OLD  RMSE=0.2616  MAE=0.2081  ME=-0.1422  e0_MAE=1.753")
    print(f"    NEW  RMSE={new_kt_bilstm['multi']['RMSE']:.4f}  "
          f"MAE={new_kt_bilstm['multi']['MAE']:.4f}  "
          f"ME={new_kt_bilstm['multi']['ME']:.4f}  "
          f"e0_MAE={new_kt_bilstm['multi']['e0_MAE']:.3f}")

    print(f"\n  KtBiLSTM (rolling):")
    print(f"    OLD  RMSE=0.2526  MAE=0.2023  ME=-0.1217  e0_MAE=1.557")
    print(f"    NEW  RMSE={new_kt_bilstm['rolling']['RMSE']:.4f}  "
          f"MAE={new_kt_bilstm['rolling']['MAE']:.4f}  "
          f"ME={new_kt_bilstm['rolling']['ME']:.4f}  "
          f"e0_MAE={new_kt_bilstm['rolling']['e0_MAE']:.3f}")

    # save results for manual report update
    results = {
        "BiLSTM_direct": new_bilstm,
        "KtBiLSTM_multi": new_kt_bilstm["multi"],
        "KtBiLSTM_rolling": new_kt_bilstm["rolling"],
    }
    with open(f"outputs/results/{TAG}_bilstm_fixed_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> outputs/results/{TAG}_bilstm_fixed_metrics.json")


if __name__ == "__main__":
    main()
