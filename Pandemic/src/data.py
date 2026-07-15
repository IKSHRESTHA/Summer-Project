"""
Data pipeline for the Performer mortality-forecasting study
(LC vs LSTM vs Transformer vs Performer).

  - HMD male log-mortality, ages 0-100 (A = 101 age groups), 1950-2019
  - 6 countries (UK proxied by GBRTENW = England & Wales)
  - Moving-window inputs: (seq_len=10, 101) -> next-year target (101,)
  - Per-age z-score standardisation, statistics fitted on training years only
  - Two train/test splits (see SPLITS below); every model sees identical data
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")
AGE_MIN, AGE_MAX = 0, 100          # A = 101 age groups
YEAR_START, YEAR_END = 1950, 2019  # 1950 start avoids war-shock regimes and
                                   # pre-war data-quality issues (lit. standard)
SEQ_LEN    = 10                    # moving-window length

# Two train/test splits studied:
#   "extended" : train 1950-2009 (60 yrs, 50 windows), test 2010-2019 (10 yrs)
#                -> primary design: more training data, actuarially typical horizon
#   "paper"    : train 1950-2000 (51 yrs, 41 windows), test 2001-2019 (19 yrs)
#                -> comparability with Wang et al. (2024) and our replication
SPLITS = {"extended": 2009, "paper": 2000}

# paper country -> our HMD parquet code
COUNTRY_CODES = {
    "UK":      "GBRTENW",
    "France":  "FRATNP",
    "Italy":   "ITA",
    "Denmark": "DNK",
    "Canada":  "CAN",
    "Finland": "FIN",
    # confirmation-set countries (publication run)
    "Australia":   "AUS",
    "Switzerland": "CHE",
    "Norway":      "NOR",
    "Sweden":      "SWE",
}
SEX = "male"   # default; pass sex= explicitly for the confirmation set

# Discovery set: the 6 male populations used in the replication and all
# exploratory analysis. Confirmation set: 14 populations never used for any
# design decision (see HYPOTHESES.md, written before the confirmatory run).
DISCOVERY_POPS    = [("UK", "male"), ("France", "male"), ("Italy", "male"),
                     ("Denmark", "male"), ("Canada", "male"), ("Finland", "male")]
CONFIRMATION_POPS = ([(c, "female") for c, _ in DISCOVERY_POPS] +
                     [(c, s) for c in ["Australia", "Switzerland", "Norway", "Sweden"]
                      for s in ["male", "female"]])


def load_log_mx(country: str, sex: str = SEX,
                year_end: int = YEAR_END) -> np.ndarray:
    """
    Load one population -> (101, T) log-mortality matrix, ages 0-100,
    1950..year_end (pandemic study passes year_end up to 2023; years beyond
    the population's last available year are silently truncated).
    Zeros (missing-data markers in small populations) are forward-filled
    along the year axis before the log transform.
    """
    code = COUNTRY_CODES[country]
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{code}_{sex}.parquet"))
    ages  = [a for a in df.index   if AGE_MIN <= a <= AGE_MAX]
    years = [y for y in df.columns if YEAR_START <= y <= year_end]
    df = df.loc[ages, years].astype(float)
    df = df.replace(0.0, np.nan).ffill(axis=1).bfill(axis=1).fillna(1e-5)
    return np.log(df.values)                     # (101, T)


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


def prepare_country(country: str, train_end: int = 2009,
                    batch_size: int = 32, seq_len: int = SEQ_LEN,
                    seed: int = 0, sex: str = SEX,
                    year_end: int = YEAR_END):
    """
    Build the training DataLoader and data bundle for one population and split.

    train_end : last calendar year included in training; test years are
                train_end+1 .. year_end (year_end may extend past 2019 for the
                pandemic study; silently truncated to data availability).
    Training windows are drawn strictly from years <= train_end, and the
    per-age standardisation statistics come from training years only.

    Returns
    -------
    loader   : training DataLoader
    bundle   : dict with raw matrices, the train-only scaler, and split sizes
    """
    log_mx = load_log_mx(country, sex, year_end)  # (101, T)
    n_train = train_end - YEAR_START + 1
    n_test  = log_mx.shape[1] - n_train

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
        "train_end": train_end,
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
    Rolling 1-step-ahead forecast over the test years: each year is predicted
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
    Recursive multi-step forecast over the test years: the window is seeded with the
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
