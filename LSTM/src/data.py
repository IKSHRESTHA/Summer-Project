"""
Data loading and preprocessing for HMD mortality data.

DESIGN NOTES (Agent 3 bug-fix applied):
  The original val/test DataLoaders passed the full history (including
  training years) to MortalityDataset, which then generated samples whose
  TARGETS lay inside the training window.  This meant early-stopping val
  loss was partly measured on training-period targets — leakage.

  Fix: each split receives only log_mx[:, start:end] where `start` is
  chosen so that every sequence window starts exactly at the boundary,
  and no target can fall in the training window.

    train : log_mx[:, :n_train]               → n_train - SEQ_LEN targets
    val   : log_mx[:, n_train-SEQ_LEN:n_train+n_val] → exactly n_val targets
    test  : log_mx[:, n_train+n_val-SEQ_LEN:]        → exactly n_test targets
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data")
COUNTRIES = ["AUS", "CAN", "CHE", "DNK", "FIN", "FRATNP", "GBRTENW", "ITA", "NOR", "SWE"]
SEXES     = ["male", "female"]
AGE_MIN, AGE_MAX   = 0, 99
YEAR_MIN, YEAR_MAX = 1950, 2019
SEQ_LEN            = 20
VAL_YEARS          = 5
TEST_YEARS         = 10
LOG_CLIP_LO        = -10.0
LOG_CLIP_HI        = 1.5    # exp(1.5) ≈ 4.5 — wide enough to never bind for
                             # single-year-of-age HMD data (max realistic mx ≈ 0.8)


def load_country_sex(code: str, sex: str) -> pd.DataFrame:
    """
    Load parquet file → Age × Year DataFrame of m(x,t) values.
    Ages 0–99, years clipped to [YEAR_MIN, YEAR_MAX].
    Zeros are replaced with 1e-5 (they indicate missing data in HMD).
    """
    path = os.path.join(DATA_DIR, f"{code}_{sex}.parquet")
    df = pd.read_parquet(path)
    ages  = [a for a in df.index  if AGE_MIN <= a <= AGE_MAX]
    years = [y for y in df.columns if YEAR_MIN <= y <= YEAR_MAX]
    df = df.loc[ages, years].astype(float)
    df = df.replace(0.0, np.nan).ffill(axis=1).fillna(1e-5)
    return df


class MortalityDataset(Dataset):
    """
    Sliding-window Dataset over standardised log(mx) time series.

    Parameters
    ----------
    log_mx  : (n_ages, T)  — raw log(mx) slice for this split
    mean    : (n_ages,)    — per-age mean from training data ONLY
    std     : (n_ages,)    — per-age std  from training data ONLY
    seq_len : int          — number of input time steps

    Each sample:
      x : (seq_len, n_ages)  standardised log(mx)
      y : (n_ages,)          standardised log(mx) for the NEXT year
    """

    def __init__(self, log_mx: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 seq_len: int = SEQ_LEN):
        z = (log_mx - mean[:, None]) / (std[:, None] + 1e-8)
        self.z       = z.T.astype(np.float32)   # (T, n_ages)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, self.z.shape[0] - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.z[idx : idx + self.seq_len]       # (seq_len, n_ages)
        y = self.z[idx + self.seq_len]              # (n_ages,)
        return torch.tensor(x), torch.tensor(y)


def make_loaders(code: str, sex: str, batch_size: int = 32,
                 seq_len: int = SEQ_LEN, seed: int = 42):
    """
    Build train/val/test DataLoaders for one country-sex combination.

    The split ensures NO target in val or test falls in the training window
    (fixes the data-leakage bug identified in the QA review).

    Returns
    -------
    loaders : dict["train" | "val" | "test"] → DataLoader
    meta    : dict with keys:
              mean, std, years, ages, log_mx_raw, n_train, n_val, n_test
    """
    df     = load_country_sex(code, sex)
    ages   = np.array(df.index,   dtype=int)
    years  = np.array(df.columns, dtype=int)
    log_mx = np.clip(np.log(df.values), LOG_CLIP_LO, LOG_CLIP_HI)  # (n_ages, T)

    T       = log_mx.shape[1]
    n_test  = TEST_YEARS
    n_val   = VAL_YEARS
    n_train = T - n_test - n_val

    # scaler from training data only
    train_lm = log_mx[:, :n_train]
    mean     = train_lm.mean(axis=1)
    std      = train_lm.std(axis=1)

    # BUG-FIX: each split receives only the columns it owns (+ context window)
    # so the DataLoader never generates targets inside the training period.
    val_lm  = log_mx[:, n_train - seq_len : n_train + n_val]      # seq_len+n_val cols
    test_lm = log_mx[:, n_train + n_val - seq_len : n_train + n_val + n_test]  # seq_len+n_test cols

    train_ds = MortalityDataset(train_lm, mean, std, seq_len)
    val_ds   = MortalityDataset(val_lm,   mean, std, seq_len)
    test_ds  = MortalityDataset(test_lm,  mean, std, seq_len)

    g = torch.Generator()
    g.manual_seed(seed)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  generator=g),
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
    return loaders, meta


def make_loaders_from_slice(log_mx: np.ndarray, mean: np.ndarray, std: np.ndarray,
                            n_train: int, n_val: int, n_test: int,
                            batch_size: int = 32, seq_len: int = SEQ_LEN,
                            seed: int = 42):
    """
    Build DataLoaders from a pre-split log_mx array.
    Used by walk-forward cross-validation to avoid re-loading from disk.
    """
    train_lm = log_mx[:, :n_train]
    val_lm   = log_mx[:, n_train - seq_len : n_train + n_val]
    test_lm  = log_mx[:, n_train + n_val - seq_len : n_train + n_val + n_test]

    train_ds = MortalityDataset(train_lm, mean, std, seq_len)
    val_ds   = MortalityDataset(val_lm,   mean, std, seq_len)
    test_ds  = MortalityDataset(test_lm,  mean, std, seq_len)

    g = torch.Generator(); g.manual_seed(seed)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    }
