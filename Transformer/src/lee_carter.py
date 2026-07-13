"""
Lee-Carter baseline for the replication: SVD estimation on the training
period, kappa_t extrapolated as a random walk with drift (the canonical
Lee & Carter 1992 specification the paper benchmarks against).

Adapted from the validated implementation in LSTM/src/lee_carter.py.
"""

import numpy as np


def fit_lee_carter(log_mx_train: np.ndarray) -> dict:
    """log_mx_train: (A, T) -> {ax, bx, kt}."""
    ax = log_mx_train.mean(axis=1)
    Z  = log_mx_train - ax[:, None]
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    bx = U[:, 0]
    kt = S[0] * Vt[0, :]
    if bx.sum() < 0:                       # sign convention before scaling
        bx, kt = -bx, -kt
    sc  = bx.sum()
    bx  = bx / sc
    kt  = kt * sc
    ktm = kt.mean()                        # centre kt, absorb into ax
    kt  = kt - ktm
    ax  = ax + bx * ktm
    return {"ax": ax, "bx": bx, "kt": kt}


def lc_forecast(log_mx_train: np.ndarray, h: int) -> np.ndarray:
    """
    Fit on the training matrix and extrapolate h years ahead via random
    walk with drift on kappa_t. Returns (A, h) log-mortality forecasts.
    """
    lc = fit_lee_carter(log_mx_train)
    kt = lc["kt"]
    drift  = (kt[-1] - kt[0]) / (len(kt) - 1)
    kt_fut = kt[-1] + drift * np.arange(1, h + 1)
    return lc["ax"][:, None] + lc["bx"][:, None] * kt_fut[None, :]


def lc_rolling_forecast(log_mx: np.ndarray, n_train: int,
                        n_test: int) -> np.ndarray:
    """
    Rolling 1-step-ahead Lee-Carter: refit on all observed data up to each
    test year and forecast one year ahead. The fair classical comparator
    when neural models are evaluated with observed-input rolling windows.
    Returns (A, n_test) log-mortality forecasts.
    """
    preds = []
    for i in range(n_test):
        preds.append(lc_forecast(log_mx[:, : n_train + i], h=1)[:, 0])
    return np.stack(preds, axis=1)
