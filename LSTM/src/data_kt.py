"""
Data utilities for kt-based LSTM mortality forecasting.

The correct formulation (per Richman & Schreck 2019, Perla et al. 2021,
Nigri et al. 2021) operates on the 1-dimensional kt factor extracted from
Lee-Carter, not on the raw 100-dimensional rate surface.

This reduces the problem to a scalar time series forecasting task where:
  - Training sequences: n_train - seq_len  (same count, now meaningful)
  - Model parameters:   ~4k-13k  (vs 400k-1M in the raw-rate approach)
  - The long-run trend is preserved (not destroyed by per-age z-scoring)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

KT_SEQ_LEN = 10   # sequence length for kt (shorter than raw-rate SEQ_LEN=20)


# ── kt extraction ─────────────────────────────────────────────────────────────

def project_kt(log_mx: np.ndarray, ax: np.ndarray, bx: np.ndarray) -> np.ndarray:
    """
    Project actual log(mx) onto the training-fitted bx direction to recover kt.

    Derivation: Lee-Carter model gives  log m(x,t) = ax + bx * kt + eps
    Projecting:  kt[t] = bx^T (log_mx[:,t] - ax) / (bx^T bx)

    This gives kt values on the same scale as the training kt series,
    making them directly concatenable for LSTM context.

    Parameters
    ----------
    log_mx : (n_ages, T) — actual log-mortality for any time period
    ax     : (n_ages,)   — training LC age intercepts (after kt centering fix)
    bx     : (n_ages,)   — training LC age factors (normalised, sum=1)

    Returns
    -------
    kt : (T,) projected mortality index
    """
    Z = log_mx - ax[:, None]        # (n_ages, T) — de-mean by ax
    return (bx @ Z) / (bx @ bx)    # least-squares projection  (T,)


# ── Dataset ───────────────────────────────────────────────────────────────────

class KtDataset(Dataset):
    """
    Sliding-window Dataset on the standardised kt time series.

    Each sample:
      x : (seq_len, 1)  standardised kt history
      y : (1,)          next standardised kt value (target)
    """

    def __init__(self, kt_z: np.ndarray, seq_len: int = KT_SEQ_LEN):
        self.kt_z    = kt_z.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.kt_z) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.kt_z[idx : idx + self.seq_len, None]   # (seq_len, 1)
        y = self.kt_z[idx + self.seq_len, None]          # (1,)
        return torch.tensor(x), torch.tensor(y)


# ── DataLoaders ───────────────────────────────────────────────────────────────

def make_kt_loaders(
    kt_trainval: np.ndarray,
    n_train: int,
    n_val: int,
    batch_size: int = 8,
    seq_len: int = KT_SEQ_LEN,
    seed: int = 42,
):
    """
    Build train and val DataLoaders from the kt time series.

    Parameters
    ----------
    kt_trainval : (n_train + n_val,)
        kt for training period (from LC fit) and validation period
        (from project_kt using actual val log_mx and training ax, bx).
    n_train : int
        Length of the training portion.
    n_val : int
        Length of the validation portion.

    Returns
    -------
    loaders  : {"train": DataLoader, "val": DataLoader}
    kt_stats : {"mean": float, "std": float}  — computed on training kt only
    """
    kt_mean = float(kt_trainval[:n_train].mean())
    kt_std  = float(kt_trainval[:n_train].std())

    # standardise full series with training stats only
    kt_z = (kt_trainval - kt_mean) / (kt_std + 1e-8)

    train_z = kt_z[:n_train]
    # val receives seq_len context window + n_val targets
    val_z   = kt_z[n_train - seq_len : n_train + n_val]

    g = torch.Generator()
    g.manual_seed(seed)

    loaders = {
        "train": DataLoader(KtDataset(train_z, seq_len), batch_size=batch_size,
                            shuffle=True, generator=g),
        "val":   DataLoader(KtDataset(val_z,   seq_len), batch_size=batch_size,
                            shuffle=False),
    }
    return loaders, {"mean": kt_mean, "std": kt_std}


# ── Test-time prediction ──────────────────────────────────────────────────────

def predict_kt_multistep(
    model,
    kt_context: np.ndarray,
    n_steps: int,
    kt_mean: float,
    kt_std: float,
    seq_len: int = KT_SEQ_LEN,
    device: str = "cpu",
) -> np.ndarray:
    """
    Autoregressive multi-step kt forecast.

    Feeds the last `seq_len` observed kt values as seed, then predicts
    n_steps ahead by feeding each prediction back as the next input.
    Comparable to LC multi-step: neither model sees test-period data.

    Parameters
    ----------
    kt_context : (>= seq_len,) actual observed kt; uses the LAST seq_len entries

    Returns
    -------
    kt_pred : (n_steps,) predicted kt values in original (un-standardised) scale
    """
    model = model.to(device)
    model.eval()

    kt_z_seed  = (kt_context[-seq_len:] - kt_mean) / (kt_std + 1e-8)
    buf        = list(kt_z_seed.astype(np.float32))
    preds_z    = []

    with torch.no_grad():
        for _ in range(n_steps):
            x   = torch.tensor(buf[-seq_len:], dtype=torch.float32,
                               device=device).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
            y_z = model(x).item()
            preds_z.append(y_z)
            buf.append(y_z)

    return np.array(preds_z) * (kt_std + 1e-8) + kt_mean


def predict_kt_rolling(
    model,
    kt_all_observed: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    kt_mean: float,
    kt_std: float,
    seq_len: int = KT_SEQ_LEN,
    device: str = "cpu",
) -> np.ndarray:
    """
    Rolling 1-step-ahead kt forecast using actual observed kt at each step.

    At test year i the model receives actual kt from the preceding seq_len
    years (drawn from the full observed + projected kt series), then predicts
    the next kt.  Comparable to rolling LC 1-step-ahead.

    Parameters
    ----------
    kt_all_observed : (n_train + n_val + n_test,)
        Projected kt for ALL years, including test years (obtained by
        projecting actual test log_mx onto training ax, bx).

    Returns
    -------
    kt_pred : (n_test,) rolling 1-step-ahead predictions
    """
    model = model.to(device)
    model.eval()
    preds_kt = []

    with torch.no_grad():
        for i in range(n_test):
            ctx_end = n_train + n_val + i
            ctx     = kt_all_observed[ctx_end - seq_len : ctx_end]
            ctx_z   = (ctx - kt_mean) / (kt_std + 1e-8)
            x       = torch.tensor(ctx_z.astype(np.float32), device=device
                                   ).unsqueeze(0).unsqueeze(-1)    # (1, seq_len, 1)
            y_z     = model(x).item()
            preds_kt.append(y_z * (kt_std + 1e-8) + kt_mean)

    return np.array(preds_kt)
