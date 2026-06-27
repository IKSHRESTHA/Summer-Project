"""
Evaluation utilities for mortality model comparison.

Metrics (Agent 1/5 additions over v1):
  - MAE, RMSE on log(mx)          — standard in mortality forecasting literature
  - MAPE, SMAPE on raw mx          — required for actuarial benchmark comparability
  - e0 MAE, e0 RMSE                — period life expectancy error (years)
  - Age-stratified RMSE            — breaks down error by age group
  - Diebold-Mariano test           — statistical significance of pairwise differences
  - Mean Error (bias)              — directional accuracy
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.lee_carter import life_expectancy


# ── basic metrics ─────────────────────────────────────────────────────────────

def compute_metrics(log_mx_true: np.ndarray, log_mx_pred: np.ndarray) -> dict:
    """
    MAE and RMSE on log(mx), MAPE and SMAPE on raw mx, and Mean Error (bias).

    Parameters
    ----------
    log_mx_true : (n_ages, T_test)
    log_mx_pred : (n_ages, T_test)

    Returns
    -------
    dict with MAE, RMSE, ME (bias), MAPE, SMAPE  (all floats)
    """
    err        = log_mx_true - log_mx_pred
    mx_true    = np.exp(log_mx_true)
    mx_pred    = np.exp(log_mx_pred)
    # SMAPE guards against near-zero denominators (young-age rates)
    smape_denom = (np.abs(mx_true) + np.abs(mx_pred)) / 2.0 + 1e-12
    return {
        "MAE":   float(np.mean(np.abs(err))),
        "RMSE":  float(np.sqrt(np.mean(err ** 2))),
        "ME":    float(np.mean(err)),                  # positive = model underestimates mx
        "MAPE":  float(np.mean(np.abs(mx_true - mx_pred) / (mx_true + 1e-12)) * 100),
        "SMAPE": float(np.mean(np.abs(mx_true - mx_pred) / smape_denom) * 100),
    }


def compute_e0_errors(log_mx_true: np.ndarray, log_mx_pred: np.ndarray,
                      sex: str = "male") -> dict:
    """
    Life expectancy MAE and RMSE (in years) over the test period.
    """
    T    = log_mx_true.shape[1]
    errs = [life_expectancy(np.exp(log_mx_true[:, t]), sex) -
            life_expectancy(np.exp(log_mx_pred[:, t]), sex)
            for t in range(T)]
    errs = np.array(errs)
    return {
        "e0_MAE":  float(np.mean(np.abs(errs))),
        "e0_RMSE": float(np.sqrt(np.mean(errs ** 2))),
        "e0_ME":   float(np.mean(errs)),
    }


def compute_age_stratified_rmse(log_mx_true: np.ndarray,
                                log_mx_pred: np.ndarray) -> np.ndarray:
    """
    Per-age RMSE array.

    Returns
    -------
    rmse_by_age : (n_ages,)
    """
    err = log_mx_true - log_mx_pred   # (n_ages, T)
    return np.sqrt(np.mean(err ** 2, axis=1))


# ── Diebold-Mariano test ──────────────────────────────────────────────────────

def diebold_mariano(
    log_mx_true: np.ndarray,
    log_mx_pred_a: np.ndarray,
    log_mx_pred_b: np.ndarray,
) -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: E[loss_A] = E[loss_B]  (squared error loss)
    H1: E[loss_A] != E[loss_B]

    The test statistic is computed on the series of ANNUAL squared-error
    differences (averaged over all ages per year), which gives T_test
    paired observations.

    Parameters
    ----------
    log_mx_true   : (n_ages, T_test)
    log_mx_pred_a : (n_ages, T_test) — model A predictions
    log_mx_pred_b : (n_ages, T_test) — model B predictions

    Returns
    -------
    dict with "DM_stat", "p_value", "reject_H0_5pct"
    """
    loss_a = ((log_mx_true - log_mx_pred_a) ** 2).mean(axis=0)  # (T_test,)
    loss_b = ((log_mx_true - log_mx_pred_b) ** 2).mean(axis=0)  # (T_test,)
    d      = loss_a - loss_b                                      # differential loss

    if len(d) < 3:
        return {"DM_stat": np.nan, "p_value": np.nan, "reject_H0_5pct": False}

    t_stat, p_val = stats.ttest_1samp(d, popmean=0.0)

    # NOTE: with T_test < 30 this test is unreliable.
    # The t-test assumes iid loss differentials; adjacent 1-step forecasts
    # share (seq_len - 1) input years, inducing serial correlation that
    # inflates the test statistic. Harvey-Leybourne-Newbold (1997) small-
    # sample correction and a HAC variance estimator are required for
    # valid inference when T_test is small.
    if len(d) < 30:
        import warnings
        warnings.warn(
            f"DM test has only {len(d)} paired observations (< 30). "
            "Results are unreliable: serial correlation is not corrected "
            "and small-sample size properties are poor. Interpret p-values "
            "as indicative only.", stacklevel=2
        )

    return {
        "DM_stat":        float(t_stat),
        "p_value":        float(p_val),
        "reject_H0_5pct": bool(p_val < 0.05),
        "n_obs":          int(len(d)),
    }


# ── summary table ─────────────────────────────────────────────────────────────

def build_results_table(all_results: dict) -> pd.DataFrame:
    """
    Assemble a summary DataFrame from per-model metric dicts.

    Parameters
    ----------
    all_results : {"ModelName": {"MAE": ..., "RMSE": ..., ...}}

    Returns
    -------
    DataFrame sorted by RMSE ascending.
    """
    rows = [{"Model": name, **m} for name, m in all_results.items()]
    df   = pd.DataFrame(rows).set_index("Model")
    col_order = [c for c in ["RMSE", "MAE", "ME", "MAPE", "SMAPE", "e0_MAE", "e0_RMSE", "e0_ME"]
                 if c in df.columns]
    return df[col_order].sort_values("RMSE")
