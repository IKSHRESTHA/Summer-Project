"""
Lee-Carter (1992) model — classical mortality forecasting baseline.

Functions
---------
fit_lee_carter()          -- SVD estimation with correct sign/normalisation
lc_forecast()             -- ARIMA(0,1,0) + drift on kt
project_rates()           -- reconstruct log(mx) from LC params + kt path
lc_rolling_forecast()     -- 1-step-ahead LC for FAIR comparison with LSTMs
                             (Agent 5 finding: multi-step LC vs 1-step LSTM
                              gave LSTM an unfair data advantage)
life_expectancy()         -- Chiang (1984) period life table e0
a0_andreev()              -- Andreev-Kingkade (2015) sex/rate-specific a0
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def fit_lee_carter(log_mx: np.ndarray) -> dict:
    """
    Fit Lee-Carter on an (n_ages, T) array of log(mx) values.

    Constraints: sum(bx) = 1, sum(kt) = 0.
    Sign fix is applied BEFORE normalisation (fixes the original dead-code bug
    where the check occurred after bx/=scale, at which point bx.sum()==1 always).

    Returns
    -------
    dict: ax (n_ages,), bx (n_ages,), kt (T,), var_explained (float)
    """
    ax         = log_mx.mean(axis=1)
    Z          = log_mx - ax[:, None]
    U, S, Vt   = np.linalg.svd(Z, full_matrices=False)

    bx = U[:, 0].copy()
    kt = S[0] * Vt[0, :].copy()

    # sign fix BEFORE normalisation
    if bx.sum() < 0:
        bx, kt = -bx, -kt

    sc   = bx.sum()
    bx  /= sc
    kt  *= sc
    # kt centering: ax already absorbs the time mean of log_mx,
    # so kt.mean() should be near zero from SVD; the explicit centering
    # is a safeguard — we compensate ax to keep the reconstruction identity
    kt_mean  = kt.mean()
    kt      -= kt_mean
    ax      += bx * kt_mean   # keep ax + bx*kt ≈ log_mx

    return {
        "ax":           ax,
        "bx":           bx,
        "kt":           kt,
        "var_explained": float(S[0] ** 2 / (S ** 2).sum()),
    }


def lc_forecast(kt: np.ndarray, h: int) -> np.ndarray:
    """Multi-step kt forecast with ARIMA(0,1,0) + drift."""
    result = ARIMA(kt, order=(0, 1, 0), trend="t").fit()
    return result.forecast(steps=h)


def project_rates(lc: dict, kt_future: np.ndarray) -> np.ndarray:
    """
    Reconstruct projected log(mx).

    Returns
    -------
    log_mx_proj : (n_ages, h)
    """
    return lc["ax"][:, None] + lc["bx"][:, None] * kt_future[None, :]


def lc_rolling_forecast(log_mx: np.ndarray, n_train: int,
                        n_val: int, n_test: int) -> np.ndarray:
    """
    1-step-ahead rolling LC forecast for the test period.

    At each test year t, LC is re-fitted on all data up to t-1 and used to
    forecast 1 step ahead.  This mirrors exactly what LSTMs do (use actual
    observed data as context for each prediction), making the comparison fair.

    Parameters
    ----------
    log_mx  : (n_ages, T) full log-mortality array
    n_train : number of training years
    n_val   : number of validation years
    n_test  : number of test years

    Returns
    -------
    log_mx_1step : (n_ages, n_test)  1-step-ahead LC forecasts for test period
    """
    T        = log_mx.shape[1]
    n_ages   = log_mx.shape[0]
    forecasts = np.empty((n_ages, n_test))

    for i in range(n_test):
        # fit on all data observed up to (but not including) this test year
        fit_end  = n_train + n_val + i
        lc       = fit_lee_carter(log_mx[:, :fit_end])
        kt_next  = lc_forecast(lc["kt"], h=1)
        forecasts[:, i] = (lc["ax"] + lc["bx"] * kt_next[0])

    return forecasts


def a0_andreev(m0: float, sex: str) -> float:
    """Andreev-Kingkade (2015) age-0 fraction a0 of the last age interval."""
    if sex == "male":
        return 0.330 if m0 >= 0.107 else 0.045 + 2.684 * m0
    return 0.350 if m0 >= 0.107 else 0.053 + 2.800 * m0


def life_expectancy(mx_vec: np.ndarray, sex: str = "male") -> float:
    """
    Period life-table e0 using Chiang (1984) formula.
    Open interval: Lx[-1] = lx[-1] / mx[-1]  (constant force of mortality).
    Age-0 fraction: Andreev-Kingkade (2015) sex- and rate-specific formula.
    """
    mx    = np.asarray(mx_vec, dtype=float)
    A     = len(mx)
    a     = np.full(A, 0.5)
    a[0]  = a0_andreev(mx[0], sex)

    qx       = mx / (1.0 + (1.0 - a) * mx)
    qx       = np.minimum(qx, 1.0)
    qx[-1]   = 1.0

    lx  = np.cumprod(np.concatenate([[1.0], 1.0 - qx[:-1]]))
    dx  = lx * qx
    Lx  = np.empty(A)
    Lx[:-1] = lx[1:] + a[:-1] * dx[:-1]
    Lx[-1]  = lx[-1] / mx[-1]

    Tx = np.cumsum(Lx[::-1])[::-1]
    return float(Tx[0] / lx[0])
