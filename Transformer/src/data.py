"""
Data pipeline for the Wang et al. (2024) Transformer mortality replication.

Replication protocol (Scandinavian Actuarial Journal 2024(2), 109-123):
  - HMD log-mortality, ages 0-100 (A = 101 age groups)
  - Years 1950-2019; train = 1950-2000 (51 years), test = 2001-2019 (19 years)
  - Moving-window inputs: (seq_len, 101) -> next-year target (101,)

Documented deviations from the paper (data availability):
  - 6 of their 8 countries (no ESP / JPN parquet files locally);
    UK is proxied by GBRTENW (England & Wales civilian population)
  - Male mortality (the paper does not state which population is used)

Documented assumptions (the paper omits these):
  - Window length seq_len = 10
  - Per-age z-score standardisation, statistics fitted on training years only
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")
AGE_MIN, AGE_MAX = 0, 100          # A = 101 ages, as in the paper
TRAIN_START, TRAIN_END = 1950, 2000
TEST_START,  TEST_END  = 2001, 2019
SEQ_LEN    = 10                    # assumption: paper does not state this

# paper country -> our HMD parquet code
COUNTRY_CODES = {
    "UK":      "GBRTENW",
    "France":  "FRATNP",
    "Italy":   "ITA",
    "Denmark": "DNK",
    "Canada":  "CAN",
    "Finland": "FIN",
}
SEX = "male"


def load_log_mx(country: str) -> np.ndarray:
    """
    Load one country -> (101, 70) log-mortality matrix, ages 0-100, 1950-2019.
    Zeros (missing-data markers in small populations) are forward-filled
    along the year axis before the log transform.
    """
    code = COUNTRY_CODES[country]
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{code}_{SEX}.parquet"))
    ages  = [a for a in df.index   if AGE_MIN <= a <= AGE_MAX]
    years = [y for y in df.columns if TRAIN_START <= y <= TEST_END]
    df = df.loc[ages, years].astype(float)
    df = df.replace(0.0, np.nan).ffill(axis=1).bfill(axis=1).fillna(1e-5)
    return np.log(df.values)                     # (101, 70)


class WindowDataset(Dataset):
    """Moving windows over standardised log-mortality: x (seq_len, A) -> y (A,)."""

    def __init__(self, z: np.ndarray, seq_len: int = SEQ_LEN):
        # z: (A, T) standardised log-mortality
        self.z = z.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, self.z.shape[1] - self.seq_len)

    def __getitem__(self, i):
        x = self.z[:, i : i + self.seq_len].T        # (seq_len, A)
        y = self.z[:, i + self.seq_len]              # (A,)
        return torch.tensor(x), torch.tensor(y)


def prepare_country(country: str, batch_size: int = 32, seq_len: int = SEQ_LEN,
                    seed: int = 0):
    """
    Returns
    -------
    loader   : training DataLoader (windows drawn from 1950-2000 only)
    bundle   : dict with raw matrices and the train-only scaler
    """
    log_mx = load_log_mx(country)                # (101, 70)
    n_train = TRAIN_END - TRAIN_START + 1        # 51
    n_test  = TEST_END  - TEST_START  + 1        # 19

    train_lm = log_mx[:, :n_train]
    mean = train_lm.mean(axis=1)                 # per-age, train only
    std  = train_lm.std(axis=1)

    z_full  = (log_mx  - mean[:, None]) / (std[:, None] + 1e-8)
    z_train = z_full[:, :n_train]

    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(WindowDataset(z_train, seq_len),
                        batch_size=batch_size, shuffle=True, generator=g)

    return loader, {
        "log_mx": log_mx, "z_full": z_full,
        "mean": mean, "std": std,
        "n_train": n_train, "n_test": n_test, "seq_len": seq_len,
    }


def destandardise(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(N, A) standardised -> (N, A) log-mortality."""
    return z * (std[None, :] + 1e-8) + mean[None, :]


@torch.no_grad()
def predict_train_insample(model, bundle, device="cpu") -> np.ndarray:
    """
    In-sample 1-step predictions over the training period (mirrors the
    paper's training-set metrics). Returns (A, n_train - seq_len) log-mortality.
    """
    model.eval()
    z, s = bundle["z_full"], bundle["seq_len"]
    n_train = bundle["n_train"]
    preds = []
    for i in range(n_train - s):
        x = torch.tensor(z[:, i:i+s].T[None], dtype=torch.float32, device=device)
        preds.append(model(x).cpu().numpy()[0])
    preds = np.stack(preds)                                    # (n, A)
    return destandardise(preds, bundle["mean"], bundle["std"]).T


@torch.no_grad()
def predict_test_rolling(model, bundle, device="cpu") -> np.ndarray:
    """
    Rolling 1-step-ahead forecast 2001-2019: each test year is predicted
    from the OBSERVED previous seq_len years (test observations enter the
    input windows as they are realised). This is the alternative reading of
    the paper's protocol; it gives the network an information set that the
    19-year Lee-Carter extrapolation does not have.
    Returns (A, n_test) log-mortality.
    """
    model.eval()
    z, s = bundle["z_full"], bundle["seq_len"]
    n_train, n_test = bundle["n_train"], bundle["n_test"]
    preds = []
    for i in range(n_test):
        x = torch.tensor(z[:, n_train + i - s : n_train + i].T[None],
                         dtype=torch.float32, device=device)
        preds.append(model(x).cpu().numpy()[0])
    preds = np.stack(preds)                                    # (n_test, A)
    return destandardise(preds, bundle["mean"], bundle["std"]).T


@torch.no_grad()
def predict_test_recursive(model, bundle, device="cpu") -> np.ndarray:
    """
    Recursive multi-step forecast 2001-2019: the window is seeded with the
    last seq_len training years and each prediction is fed back as input.
    Neither the model nor the window ever sees observed test data --- the
    same information set as the Lee-Carter 19-year extrapolation.
    Returns (A, n_test) log-mortality.
    """
    model.eval()
    z, s = bundle["z_full"], bundle["seq_len"]
    n_train, n_test = bundle["n_train"], bundle["n_test"]
    buf = [z[:, t].astype(np.float32) for t in range(n_train - s, n_train)]
    preds = []
    for _ in range(n_test):
        x = torch.tensor(np.stack(buf[-s:])[None], device=device)  # (1, s, A)
        y = model(x).cpu().numpy()[0]
        preds.append(y)
        buf.append(y.astype(np.float32))
    preds = np.stack(preds)                                    # (n_test, A)
    return destandardise(preds, bundle["mean"], bundle["std"]).T
