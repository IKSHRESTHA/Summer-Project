"""
Evaluation metrics for the replication, all on the log-mortality scale
exactly as in the paper (Section 4.3):

  MAE   = mean |y_hat - y|
  RMSE  = sqrt(mean (y_hat - y)^2)
  MAPE  = mean |(y_hat - y) / y|            (y = log-mortality, negative)
  RMSE(a) = per-age RMSE across test years   (Li & Lu 2017)
  RMSE(t) = per-year RMSE across ages
"""

import numpy as np


def metrics(true_lm: np.ndarray, pred_lm: np.ndarray) -> dict:
    """true_lm, pred_lm: (A, T) log-mortality."""
    err = pred_lm - true_lm
    return {
        "MAE":  float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err / true_lm))) * 100.0,
    }


def rmse_by_age(true_lm: np.ndarray, pred_lm: np.ndarray) -> np.ndarray:
    """RMSE(a): (A,) --- RMSE across years at each age."""
    return np.sqrt(np.mean((pred_lm - true_lm) ** 2, axis=1))


def rmse_by_year(true_lm: np.ndarray, pred_lm: np.ndarray) -> np.ndarray:
    """RMSE(t): (T,) --- RMSE across ages in each year."""
    return np.sqrt(np.mean((pred_lm - true_lm) ** 2, axis=0))
