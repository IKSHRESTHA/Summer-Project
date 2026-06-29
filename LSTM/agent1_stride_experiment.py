"""
Agent 1 — Stride Experiment
============================
Tests whether stride=1 (overlapping windows) in the training data constitutes
data leakage or merely correlated samples, by comparing test RMSE across three
stride configurations using the Stacked LSTM model.

Configurations
--------------
  Config A: stride=1   (~35 training windows) — paper's baseline
  Config B: stride=5   (~7 training windows)  — partial de-correlation
  Config C: stride=10  (~3-4 training windows) — strong de-correlation
"""

import os
import sys
import random
import copy
import warnings

# ── must be first: set working directory so src/ imports resolve ──────────────
os.chdir(r"c:\Users\krish\OneDrive\Desktop\Summer Project\LSTM")
sys.path.insert(0, r"c:\Users\krish\OneDrive\Desktop\Summer Project\LSTM")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import load_country_sex, MortalityDataset, SEQ_LEN, LOG_CLIP_LO, LOG_CLIP_HI
from src.models.stacked_lstm import StackedLSTM
from src.train import train_model, predict_all, destandardise
from src.evaluate import compute_metrics

# ── reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── StrideDataset: subclass of MortalityDataset that supports stride > 1 ─────

class StrideDataset(MortalityDataset):
    """
    MortalityDataset subclass that supports configurable stride for the
    sliding window.  stride=1 reproduces the original behaviour.

    Parameters
    ----------
    log_mx  : (n_ages, T)
    mean    : (n_ages,)
    std     : (n_ages,)
    seq_len : int
    stride  : int  — step between consecutive windows
    """

    def __init__(self, log_mx, mean, std, seq_len=SEQ_LEN, stride=1):
        super().__init__(log_mx, mean, std, seq_len)
        self.stride = stride

    def __len__(self):
        # number of complete (input, target) pairs given the stride
        return max(0, (self.z.shape[0] - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        real_idx = idx * self.stride
        x = self.z[real_idx : real_idx + self.seq_len]     # (seq_len, n_ages)
        y = self.z[real_idx + self.seq_len]                 # (n_ages,)
        return torch.tensor(x), torch.tensor(y)


# ── data preparation ──────────────────────────────────────────────────────────

def build_loaders(stride: int, country: str = "AUS", sex: str = "male",
                  seq_len: int = SEQ_LEN, batch_size: int = 32,
                  seed: int = 42):
    """
    Build train/val/test DataLoaders for the given stride.

    The val/test loaders always use stride=1 (standard evaluation).
    Only the training loader uses the configurable stride.

    Returns
    -------
    loaders : dict["train"|"val"|"test"] → DataLoader
    meta    : dict
    """
    df = load_country_sex(country, sex)
    ages  = np.array(df.index,   dtype=int)
    years = np.array(df.columns, dtype=int)
    log_mx = np.clip(np.log(df.values), LOG_CLIP_LO, LOG_CLIP_HI)  # (n_ages, T)

    T       = log_mx.shape[1]
    n_test  = 10
    n_val   = 5
    n_train = T - n_test - n_val

    # scaler from training data only
    train_lm = log_mx[:, :n_train]
    mean     = train_lm.mean(axis=1)
    std      = train_lm.std(axis=1)

    # splits: val/test receive a leading context window of seq_len years
    val_lm  = log_mx[:, n_train - seq_len : n_train + n_val]
    test_lm = log_mx[:, n_train + n_val - seq_len : n_train + n_val + n_test]

    # training uses StrideDataset; val/test always use stride=1
    train_ds = StrideDataset(train_lm, mean, std, seq_len, stride=stride)
    val_ds   = MortalityDataset(val_lm, mean, std, seq_len)
    test_ds  = MortalityDataset(test_lm, mean, std, seq_len)

    g = torch.Generator()
    g.manual_seed(seed)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    }
    meta = {
        "mean":       mean,
        "std":        std,
        "years":      years,
        "ages":       ages,
        "log_mx_raw": log_mx,
        "n_train":    n_train,
        "n_val":      n_val,
        "n_test":     n_test,
        "seq_len":    seq_len,
    }
    n_windows = len(train_ds)
    return loaders, meta, n_windows


# ── run one configuration ─────────────────────────────────────────────────────

def run_config(label: str, stride: int, country: str = "AUS", sex: str = "male",
               max_epochs: int = 80, patience: int = 15,
               hidden_size: int = 128, dropout: float = 0.2,
               lr: float = 1e-3, weight_decay: float = 1e-4,
               device: str = "cpu", seed: int = 42):
    """
    Train Stacked LSTM with the given stride and return test RMSE.
    """
    set_seed(seed)
    print(f"\n{'-'*60}")
    print(f"  {label}: stride={stride}")

    loaders, meta, n_windows = build_loaders(stride, country, sex, seed=seed)
    n_ages = len(meta["ages"])

    print(f"  Training windows: {n_windows}")
    print(f"  (n_train={meta['n_train']}, seq_len={meta['seq_len']}, "
          f"stride={stride} -> windows = ({meta['n_train']} - {meta['seq_len']} - 1) // {stride} + 1 "
          f"= {n_windows})")

    if n_windows == 0:
        print(f"  ERROR: no training windows for stride={stride}. Skipping.")
        return {"label": label, "stride": stride, "n_windows": 0, "test_rmse": float("nan"),
                "best_epoch": -1}

    model = StackedLSTM(n_ages=n_ages, hidden_size=hidden_size,
                        num_layers=3, dropout=dropout)

    history = train_model(
        model, loaders,
        epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
        verbose=False,
    )
    best_epoch = history["best_epoch"]
    print(f"  Best epoch: {best_epoch}")

    # evaluate on test set
    preds_z  = predict_all(model, loaders["test"], device)           # (n_test, n_ages)
    preds_lm = destandardise(preds_z, meta["mean"], meta["std"]).T   # (n_ages, n_test)

    log_mx       = meta["log_mx_raw"]
    n_test       = meta["n_test"]
    log_mx_test  = log_mx[:, -n_test:]

    metrics = compute_metrics(log_mx_test, preds_lm)
    rmse    = metrics["RMSE"]
    mae     = metrics["MAE"]

    print(f"  Test RMSE={rmse:.4f}  MAE={mae:.4f}")

    return {
        "label":       label,
        "stride":      stride,
        "n_windows":   n_windows,
        "test_rmse":   rmse,
        "test_mae":    mae,
        "best_epoch":  best_epoch,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    os.makedirs("outputs/results", exist_ok=True)

    print("=" * 60)
    print("  AGENT 1 — STRIDE EXPERIMENT")
    print("  Stacked LSTM · AUS male · seed=42")
    print("  Paper reported RMSE: Stacked LSTM=0.2578, BiLSTM=0.2693")
    print("=" * 60)

    # Theoretical window counts for documentation
    n_train_approx = 55   # typical value: T=70, val=5, test=10 → train=55
    seq_len = SEQ_LEN     # =20
    for s in [1, 5, 10]:
        w = max(0, (n_train_approx - seq_len - 1) // s + 1)
        print(f"  stride={s:2d}  -> approx {w} training windows")

    configs = [
        ("Config A (Baseline, stride=1)",   1),
        ("Config B (stride=5)",             5),
        ("Config C (stride=10)",           10),
    ]

    results = []
    for label, stride in configs:
        try:
            r = run_config(label, stride,
                           max_epochs=80, patience=15,
                           hidden_size=128, dropout=0.2,
                           lr=1e-3, weight_decay=1e-4,
                           device="cpu", seed=42)
            results.append(r)
        except Exception as e:
            print(f"  ERROR in {label}: {e}")
            results.append({"label": label, "stride": stride,
                            "n_windows": -1, "test_rmse": float("nan"),
                            "test_mae": float("nan"), "best_epoch": -1})

    # ── save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df["paper_rmse"] = 0.2578   # paper's Stacked LSTM baseline
    df["delta_rmse"] = df["test_rmse"] - df["paper_rmse"]

    out_path = "outputs/results/agent1_stride_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"  Results saved to {out_path}")
    print(f"\n  SUMMARY TABLE")
    print(f"  {'Config':<35} {'Stride':>6} {'N Windows':>10} {'RMSE':>8} {'vs Paper':>10}")
    print(f"  {'-'*75}")
    for _, row in df.iterrows():
        delta = f"{row['delta_rmse']:+.4f}" if not pd.isna(row['delta_rmse']) else "   n/a"
        rmse  = f"{row['test_rmse']:.4f}"   if not pd.isna(row['test_rmse'])  else "   n/a"
        print(f"  {row['label']:<35} {int(row['stride']):>6} {int(row['n_windows']):>10} "
              f"{rmse:>8} {delta:>10}")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    main()
