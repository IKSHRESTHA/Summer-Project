

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.stats import t as t_dist
import warnings, os, io, zipfile, urllib.request, getpass, ssl
warnings.filterwarnings('ignore')

# Windows Python 3.14 ships without the system CA bundle; bypass SSL verification.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: HMD DATA FETCHER
# ─────────────────────────────────────────────────────────────────────────────

import http.cookiejar, urllib.parse, re as _re

def _hmd_login(username, password):
    """
    Log in to mortality.org using the cookie-based form and return an
    authenticated opener that can be reused for all subsequent requests.
    """
    jar    = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_SSL_CTX),
        urllib.request.HTTPCookieProcessor(jar),
    )
    login_url = "https://mortality.org/Account/Login"

    # 1. GET the login page to collect the CSRF token
    r    = opener.open(login_url, timeout=20)
    html = r.read().decode("utf-8", errors="replace")
    m    = _re.search(r'__RequestVerificationToken[^>]*value=["\']([^"\']+)', html)
    csrf = m.group(1) if m else ""

    # 2. POST credentials
    payload = urllib.parse.urlencode({
        "__RequestVerificationToken": csrf,
        "Email":    username,
        "Password": password,
    }).encode()
    req = urllib.request.Request(
        login_url, data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": login_url,
        },
    )
    opener.open(req, timeout=20)
    return opener


def _fetch_lt_file(opener, url):
    """
    Download one HMD life-table text file and return a DataFrame(Age x Year)
    of the mx column.  Format: whitespace-delimited, 2-line header, mx in col 3.
    """
    with opener.open(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8")

    if "<title>Login</title>" in raw or "<!DOCTYPE html>" in raw[:100]:
        raise RuntimeError("Authentication failed — check your HMD email/password.")

    lines = [l for l in raw.splitlines() if l.strip() and not l.startswith("#")]
    header_idx = next(i for i, l in enumerate(lines) if "Year" in l and "Age" in l)
    col_map    = {c.lower(): i for i, c in enumerate(lines[header_idx].split())}
    yi = col_map.get("year", 0)
    ai = col_map.get("age",  1)
    mi = col_map.get("mx",   2)   # mltper / fltper files have a single "mx" column

    rows = []
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if len(parts) <= max(yi, ai, mi):
            continue
        try:
            year = int(parts[yi])
            age  = int(parts[ai].replace("+", ""))
            mx   = float(parts[mi]) if parts[mi] != "." else np.nan
            rows.append((year, age, mx))
        except ValueError:
            continue

    df = pd.DataFrame(rows, columns=["Year", "Age", "mx"])
    df["mx"].replace(0, 1e-6, inplace=True)
    df["mx"].fillna(1e-6, inplace=True)
    return df.pivot(index="Age", columns="Year", values="mx")


def fetch_hmd_mx(country_code, opener):
    """
    Fetch central death rates from HMD for a given country using an
    already-authenticated opener (returned by _hmd_login).
    Returns a dict with keys 'male', 'female', each a DataFrame
    with index=Age, columns=Year.
    Uses mltper_1x1.txt (male) and fltper_1x1.txt (female).
    """
    base = f"https://mortality.org/File/GetDocument/hmd.v6/{country_code}/STATS"
    return {
        "male":   _fetch_lt_file(opener, f"{base}/mltper_1x1.txt"),
        "female": _fetch_lt_file(opener, f"{base}/fltper_1x1.txt"),
    }


def load_synthetic_demo():
    """
    Generate synthetic but realistic HMD-like mortality data for Sweden (male)
    using a Makeham-Gompertz model. Used when HMD credentials are not provided.
    Returns DataFrame with index=Age (0-94), columns=Year (1900-2000).
    """
    np.random.seed(42)
    ages  = np.arange(0, 95)
    years = np.arange(1900, 2001)
    
    # Gompertz-Makeham: mu(x,t) = A(t) + B(t)*exp(c*x)
    # with gradual improvement over time
    A_base, B_base, c = 0.0007, 0.00005, 0.09
    
    Mx = np.zeros((len(ages), len(years)))
    for j, yr in enumerate(years):
        improvement = (yr - 1900) / 100   # 0 to 1
        A = A_base * (1 - 0.5 * improvement)
        B = B_base * (1 - 0.6 * improvement)
        mu = A + B * np.exp(c * ages)
        # Add infant mortality hump
        mu[0] *= (1 + 5 * np.exp(-improvement * 3))
        # Add noise
        noise = np.random.lognormal(0, 0.03, len(ages))
        Mx[:, j] = np.clip(mu * noise, 1e-6, 1.0)
    
    return pd.DataFrame(Mx, index=ages, columns=years)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CORE MORTALITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def life_expectancy(mx_vec, ages=None):
    """Compute period life expectancy at birth from age-specific death rates."""
    if ages is None:
        ages = np.arange(len(mx_vec))
    n = len(mx_vec)
    ax = np.where(ages == 0, 0.1, 0.5)
    qx = mx_vec / (1 + (1 - ax) * mx_vec)
    qx = np.minimum(qx, 1.0)
    qx[-1] = 1.0   # terminal age
    lx = np.ones(n + 1)
    for i in range(n):
        lx[i + 1] = lx[i] * (1 - qx[i])
    Lx = (lx[:-1] + lx[1:]) / 2
    Tx = np.cumsum(Lx[::-1])[::-1]
    return Tx[0] / lx[0]


def smooth_bspline_mx(log_mx_mat, df=10):
    """
    Simple B-spline-like age smoothing row-by-row per year.
    Uses polynomial regression as a proxy for the spline smoothing in HU/DJT.
    """
    from numpy.polynomial import polynomial as P
    ages = np.arange(log_mx_mat.shape[0])
    smoothed = np.zeros_like(log_mx_mat)
    for j in range(log_mx_mat.shape[1]):
        col = log_mx_mat[:, j]
        valid = ~np.isnan(col) & ~np.isinf(col)
        if valid.sum() > df:
            coef = np.polyfit(ages[valid], col[valid], deg=min(df - 1, 8))
            smoothed[:, j] = np.polyval(coef, ages)
        else:
            smoothed[:, j] = col
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: LEE-CARTER SVD AND kt ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

def fit_lc_svd(log_mx):
    """
    Fit the Lee-Carter model via SVD.
    log_mx: array (n_ages x n_years)
    Returns ax, bx, kt (raw, not adjusted).
    """
    ax  = log_mx.mean(axis=1)
    Zxt = log_mx - ax[:, np.newaxis]
    U, s, Vt = svd(Zxt, full_matrices=False)
    # First singular component
    bx_raw = U[:, 0]
    kt_raw = s[0] * Vt[0, :]
    # Normalise: bx sums to 1, kt sums to 0
    bx_sum = bx_raw.sum()
    bx = bx_raw / bx_sum
    kt = kt_raw * bx_sum
    kt -= kt.mean()
    return ax, bx, kt


def adjust_kt_dt(ax, bx, kt, log_mx):
    """
    LC adjustment: refit kt so that sum of fitted deaths matches observed deaths.
    Since we only have Mx (not exposure), we minimise sum(fitted_logMx - actual_logMx)^2
    by a scalar shift at each t — equivalent to matching total log mortality level.
    (Full Dt adjustment requires population counts; this is the closest approximation.)
    """
    kt_adj = kt.copy()
    n_years = log_mx.shape[1]
    for t in range(n_years):
        fitted_no_adj = ax + bx * kt[t]
        residual = log_mx[:, t] - fitted_no_adj
        # Shift kt to minimise total residual weighted by bx
        kt_adj[t] = kt[t] + residual.sum() / (bx**2).sum() * bx.mean()
    return kt_adj


def adjust_kt_e0(ax, bx, kt, mx_actual, ages):
    """
    LM adjustment: refit kt so that fitted e0 matches actual e0 each year.
    Bisection over kt to find the value that matches e0.
    """
    kt_adj = kt.copy()
    for t in range(mx_actual.shape[1]):
        target_e0 = life_expectancy(mx_actual[:, t], ages)
        
        def e0_from_kt(k):
            fitted_log_mx = ax + bx * k
            fitted_mx = np.exp(np.clip(fitted_log_mx, -20, 5))
            return life_expectancy(fitted_mx, ages)
        
        # Bisect over kt
        lo, hi = kt[t] - 50, kt[t] + 50
        for _ in range(50):
            mid = (lo + hi) / 2
            if e0_from_kt(mid) < target_e0:
                lo = mid
            else:
                hi = mid
        kt_adj[t] = (lo + hi) / 2
    return kt_adj


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: RANDOM WALK WITH DRIFT FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def rw_drift_forecast(kt, h):
    """
    Forecast kt using a random walk with drift for h steps.
    d = mean of first differences of kt.
    """
    d = np.diff(kt).mean()
    kt_last = kt[-1]
    return np.array([kt_last + (i + 1) * d for i in range(h)])


def damped_holts_forecast(kt, h, alpha=0.2, beta=0.1, phi=0.9):
    """
    Damped Holt's exponential smoothing (used in HU method for kt,j).
    phi < 1 gives a damped trend.
    """
    n = len(kt)
    level  = np.zeros(n)
    trend  = np.zeros(n)
    level[0] = kt[0]
    trend[0] = kt[1] - kt[0] if n > 1 else 0
    for t in range(1, n):
        level[t] = alpha * kt[t] + (1 - alpha) * (level[t-1] + phi * trend[t-1])
        trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * phi * trend[t-1]
    # Forecast
    forecasts = np.zeros(h)
    phi_powers = np.array([sum(phi**(j+1) for j in range(i+1)) for i in range(h)])
    for i in range(h):
        forecasts[i] = level[-1] + phi_powers[i] * trend[-1]
    return forecasts


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: THE FIVE METHODS
# ─────────────────────────────────────────────────────────────────────────────

def method_LC(mx_full, ages, fit_end=1985, h=15, start_long=1900):
    """
    Original Lee-Carter:
    - Long fitting period (start_long to fit_end)
    - kt adjusted to Dt (approximated here as total log mortality match)
    - Jump-off: fitted rates (kt passes through 0 at fit_end)
    """
    years = np.array(mx_full.columns)
    fit_mask = (years >= start_long) & (years <= fit_end)
    log_mx   = np.log(np.clip(mx_full.values[:, fit_mask], 1e-9, None))
    
    ax, bx, kt = fit_lc_svd(log_mx)
    kt = adjust_kt_dt(ax, bx, kt, log_mx)
    kt_fc = rw_drift_forecast(kt, h)
    
    log_mx_fc = ax[:, np.newaxis] + bx[:, np.newaxis] * kt_fc[np.newaxis, :]
    return np.exp(log_mx_fc)   # shape (n_ages, h)


def method_LM(mx_full, ages, fit_end=1985, h=15):
    """
    Lee-Miller variant:
    - Fitting period starts 1950
    - kt adjusted to e(0)
    - Jump-off: actual rates in fit_end year
    """
    years = np.array(mx_full.columns)
    fit_mask = (years >= 1950) & (years <= fit_end)
    mx_fit   = mx_full.values[:, fit_mask]
    log_mx   = np.log(np.clip(mx_fit, 1e-9, None))
    
    ax, bx, kt = fit_lc_svd(log_mx)
    kt = adjust_kt_e0(ax, bx, kt, mx_fit, ages)
    kt_fc = rw_drift_forecast(kt, h)
    
    # Jump-off: actual rates in fit_end year, constrain kt to pass through 0
    # at jump-off by shifting all kt so kt[-1]=0
    kt_fc = kt_fc - kt[-1]   # shift: kt at jump-off = 0
    
    # Actual jump-off log rates
    actual_jumpoff = np.log(np.clip(
        mx_full.values[:, years == fit_end].flatten(), 1e-9, None))
    
    log_mx_fc = actual_jumpoff[:, np.newaxis] + bx[:, np.newaxis] * kt_fc[np.newaxis, :]
    return np.exp(log_mx_fc)


def method_BMS(mx_full, ages, fit_end=1985, h=15, start_long=1900):
    """
    Booth-Maindonald-Smith variant:
    - Fitting period chosen to maximise linearity of kt (algorithmic)
    - kt adjusted to age distribution of deaths (Dx,t) — approximated
    - Jump-off: fitted rates under adjustment
    """
    years = np.array(mx_full.columns)
    all_start_years = years[(years >= start_long) & (years <= fit_end - 15)]
    
    best_start = start_long
    best_ratio = np.inf
    
    for sy in all_start_years:
        fit_mask = (years >= sy) & (years <= fit_end)
        log_mx   = np.log(np.clip(mx_full.values[:, fit_mask], 1e-9, None))
        if log_mx.shape[1] < 10:
            continue
        _, _, kt = fit_lc_svd(log_mx)
        
        # Linearity of kt: ratio of deviance of linear fit to overall model deviance
        t_idx  = np.arange(len(kt))
        lin_coef = np.polyfit(t_idx, kt, 1)
        residuals_lin  = kt - np.polyval(lin_coef, t_idx)
        residuals_lca  = kt - kt.mean()
        ratio = np.var(residuals_lin) / (np.var(residuals_lca) + 1e-12)
        if ratio < best_ratio:
            best_ratio = ratio
            best_start = sy
    
    fit_mask = (years >= best_start) & (years <= fit_end)
    log_mx   = np.log(np.clip(mx_full.values[:, fit_mask], 1e-9, None))
    ax, bx, kt = fit_lc_svd(log_mx)
    # Adjustment: fit to age distribution (simplified: no adjustment)
    kt_fc = rw_drift_forecast(kt, h)
    log_mx_fc = ax[:, np.newaxis] + bx[:, np.newaxis] * kt_fc[np.newaxis, :]
    return np.exp(log_mx_fc), best_start


def method_HU(mx_full, ages, fit_end=1985, h=15, J=6):
    """
    Hyndman-Ullah functional data method:
    - Fitting period: 1950 onwards
    - Smooth mortality curves (polynomial proxy for spline)
    - J=6 principal components
    - Damped Holt's method for kt,j
    """
    years = np.array(mx_full.columns)
    fit_mask = (years >= 1950) & (years <= fit_end)
    log_mx   = np.log(np.clip(mx_full.values[:, fit_mask], 1e-9, None))
    
    # Step 1: smooth each year's log(mx) curve
    log_mx_smooth = smooth_bspline_mx(log_mx)
    
    # Step 2: mean curve
    ax = log_mx_smooth.mean(axis=1)
    Z  = log_mx_smooth - ax[:, np.newaxis]
    
    # Step 3: J-component SVD
    U, s, Vt = svd(Z, full_matrices=False)
    J_use = min(J, U.shape[1])
    
    # Step 4: forecast each kt,j with damped Holt's
    log_mx_fc = np.tile(ax[:, np.newaxis], (1, h))
    for j in range(J_use):
        bj  = U[:, j]
        ktj = s[j] * Vt[j, :]
        ktj_fc = damped_holts_forecast(ktj, h)
        log_mx_fc += bj[:, np.newaxis] * ktj_fc[np.newaxis, :]
    
    return np.exp(log_mx_fc)


def method_DJT(mx_full, ages, fit_end=1985, h=15):
    """
    De Jong-Tickle LC(smooth) state-space method:
    - Fitting period: 1950 onwards
    - B-spline smoothed mortality (polynomial proxy)
    - Random walk with drift for kt
    - Jump-off: smoothed average of last 2 years
    """
    years = np.array(mx_full.columns)
    fit_mask = (years >= 1950) & (years <= fit_end)
    log_mx   = np.log(np.clip(mx_full.values[:, fit_mask], 1e-9, None))
    
    # Smooth each year
    log_mx_smooth = smooth_bspline_mx(log_mx)
    
    ax, bx, kt = fit_lc_svd(log_mx_smooth)
    kt_fc = rw_drift_forecast(kt, h)
    
    # Jump-off: smoothed average of last 2 actual years
    jumpoff_log = log_mx_smooth[:, -2:].mean(axis=1)
    kt_fc_shifted = kt_fc - kt[-1]
    
    log_mx_fc = jumpoff_log[:, np.newaxis] + bx[:, np.newaxis] * kt_fc_shifted[np.newaxis, :]
    return np.exp(log_mx_fc)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ERROR METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_mae_log(mx_forecast, mx_actual):
    """Mean absolute error in log death rates."""
    log_fc  = np.log(np.clip(mx_forecast, 1e-9, None))
    log_act = np.log(np.clip(mx_actual,   1e-9, None))
    return np.abs(log_fc - log_act).mean()

def compute_me_log(mx_forecast, mx_actual):
    """Mean error (bias) in log death rates."""
    log_fc  = np.log(np.clip(mx_forecast, 1e-9, None))
    log_act = np.log(np.clip(mx_actual,   1e-9, None))
    return (log_fc - log_act).mean()

def compute_mae_e0(mx_forecast, mx_actual, ages):
    """Mean absolute error in life expectancy over forecast horizon."""
    h = mx_forecast.shape[1]
    errors = []
    for t in range(h):
        e0_fc  = life_expectancy(mx_forecast[:, t], ages)
        e0_act = life_expectancy(mx_actual[:, t], ages)
        errors.append(e0_fc - e0_act)
    return np.mean(np.abs(errors)), np.mean(errors)

def compute_me_e0(mx_forecast, mx_actual, ages):
    h = mx_forecast.shape[1]
    errors = []
    for t in range(h):
        e0_fc  = life_expectancy(mx_forecast[:, t], ages)
        e0_act = life_expectancy(mx_actual[:, t], ages)
        errors.append(e0_fc - e0_act)
    return np.mean(errors)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

# Paper's 10 countries with HMD codes and long fitting period start years
COUNTRIES = [
    {"name": "Australia",    "code": "AUS",     "start_long": 1921},
    {"name": "Canada",       "code": "CAN",     "start_long": 1921},
    {"name": "Denmark",      "code": "DNK",     "start_long": 1900},
    {"name": "England",      "code": "GBRTENW", "start_long": 1922},
    {"name": "Finland",      "code": "FIN",     "start_long": 1900},
    {"name": "France",       "code": "FRATNP",  "start_long": 1900},
    {"name": "Italy",        "code": "ITA",     "start_long": 1900},
    {"name": "Norway",       "code": "NOR",     "start_long": 1900},
    {"name": "Sweden",       "code": "SWE",     "start_long": 1900},
    {"name": "Switzerland",  "code": "CHE",     "start_long": 1900},
]

SEXES       = ["male", "female"]
FIT_END     = 1985
EVAL_START  = 1986
EVAL_END    = 2000
H           = EVAL_END - EVAL_START + 1    # 15 years
MAX_AGE     = 94

def run_evaluation(mx_data_all, demo_mode=False):
    """
    mx_data_all: dict keyed by (country_name, sex) -> DataFrame(Age x Year)
    Returns summary DataFrames matching Tables 2-5 in the paper.
    """
    records_log = []
    records_e0  = []

    METHODS = ["LC", "LM", "BMS", "HU", "DJT"]
    
    for c in COUNTRIES:
        cname = c["name"]
        for sex in SEXES:
            key = (cname, sex)
            if key not in mx_data_all:
                print(f"  Skipping {cname} {sex}: no data")
                continue

            mx_df = mx_data_all[key]
            # Restrict ages to 0-MAX_AGE
            mx_df  = mx_df[mx_df.index <= MAX_AGE]
            ages   = mx_df.index.values.astype(float)
            years  = mx_df.columns.values.astype(int)

            # Actual mortality in evaluation period
            eval_mask = (years >= EVAL_START) & (years <= EVAL_END)
            if eval_mask.sum() == 0:
                print(f"  Skipping {cname} {sex}: no data for {EVAL_START}-{EVAL_END}")
                continue
            mx_actual = mx_df.values[:, eval_mask]

            print(f"  {cname} {sex}: ages {ages[0]:.0f}-{ages[-1]:.0f}, "
                  f"years {years[0]}-{years[-1]}, eval years: {eval_mask.sum()}")

            for mname in METHODS:
                try:
                    if mname == "LC":
                        mx_fc = method_LC(mx_df, ages, FIT_END, H, c["start_long"])
                    elif mname == "LM":
                        mx_fc = method_LM(mx_df, ages, FIT_END, H)
                    elif mname == "BMS":
                        mx_fc, bstart = method_BMS(mx_df, ages, FIT_END, H, c["start_long"])
                    elif mname == "HU":
                        mx_fc = method_HU(mx_df, ages, FIT_END, H, J=6)
                    elif mname == "DJT":
                        mx_fc = method_DJT(mx_df, ages, FIT_END, H)
                    
                    mae_log = compute_mae_log(mx_fc, mx_actual)
                    me_log  = compute_me_log(mx_fc, mx_actual)
                    mae_e0, _ = compute_mae_e0(mx_fc, mx_actual, ages)
                    me_e0     = compute_me_e0(mx_fc, mx_actual, ages)
                    
                    records_log.append({
                        "Country": cname, "Sex": sex, "Method": mname,
                        "ME":  round(me_log,  3),
                        "MAE": round(mae_log, 3)
                    })
                    records_e0.append({
                        "Country": cname, "Sex": sex, "Method": mname,
                        "ME_e0":  round(me_e0,  3),
                        "MAE_e0": round(mae_e0, 3)
                    })
                except Exception as ex:
                    print(f"    ERROR {mname}: {ex}")

    df_log = pd.DataFrame(records_log)
    df_e0  = pd.DataFrame(records_e0)
    return df_log, df_e0


def print_comparison_table(df_log, df_e0):
    """Print replicated results vs paper benchmarks."""
    if df_log.empty or "Sex" not in df_log.columns:
        print("\nNo results to display — all data fetches failed.")
        return

    print("\n" + "="*70)
    print("TABLE 3 EQUIVALENT: Mean Absolute Error in log death rates")
    print("="*70)
    
    for sex in SEXES:
        sub = df_log[df_log["Sex"] == sex].groupby("Method")["MAE"].mean()
        print(f"\n  {sex.upper()}:")
        print(f"  {'Method':<8} {'Replicated':>12} {'Paper':>10} {'Diff':>8}")
        paper_vals = {"LC": 0.31 if sex=="male" else 0.45,
                      "LM": 0.17, "BMS": 0.15 if sex=="male" else 0.16,
                      "HU": 0.15, "DJT": 0.15}
        for m in ["LC", "LM", "BMS", "HU", "DJT"]:
            if m in sub.index:
                diff = sub[m] - paper_vals[m]
                print(f"  {m:<8} {sub[m]:>12.3f} {paper_vals[m]:>10.2f} {diff:>+8.3f}")

    print("\n" + "="*70)
    print("TABLE 5 EQUIVALENT: Mean Absolute Error in life expectancy (years)")
    print("="*70)
    
    paper_e0 = {
        "male":   {"LC":0.89,"LM":1.05,"BMS":0.69,"HU":0.78,"DJT":0.98},
        "female": {"LC":0.66,"LM":0.43,"BMS":0.38,"HU":0.54,"DJT":0.44}
    }
    for sex in SEXES:
        sub = df_e0[df_e0["Sex"] == sex].groupby("Method")["MAE_e0"].mean()
        print(f"\n  {sex.upper()}:")
        print(f"  {'Method':<8} {'Replicated':>12} {'Paper':>10} {'Diff':>8}")
        for m in ["LC", "LM", "BMS", "HU", "DJT"]:
            if m in sub.index:
                pval = paper_e0[sex][m]
                diff = sub[m] - pval
                print(f"  {m:<8} {sub[m]:>12.3f} {pval:>10.2f} {diff:>+8.3f}")

    print("\n" + "="*70)
    print("TABLE 2 EQUIVALENT: Mean Error (bias) in log death rates")
    print("Paper finding: LC large negative (underestimates mortality)")
    print("="*70)
    for sex in SEXES:
        sub = df_log[df_log["Sex"] == sex].groupby("Method")["ME"].mean()
        paper_me = {"male":   {"LC":-0.11,"LM":0.06,"BMS":0.02,"HU":0.03,"DJT":0.05},
                    "female": {"LC":-0.38,"LM":-0.02,"BMS":-0.03,"HU":-0.02,"DJT":-0.03}}
        print(f"\n  {sex.upper()}:")
        for m in ["LC","LM","BMS","HU","DJT"]:
            if m in sub.index:
                print(f"  {m:<8} replicated={sub[m]:+.3f}  paper={paper_me[sex][m]:+.2f}")


def run_anova_check(df_log):
    """Simple F-test approximation for method differences (paper used 2-way ANOVA)."""
    if df_log.empty or "Method" not in df_log.columns:
        return
    from scipy.stats import f_oneway
    groups = [df_log[df_log["Method"]==m]["MAE"].values for m in ["LC","LM","BMS","HU","DJT"]]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        F, p = f_oneway(*groups)
        print(f"\n1-way ANOVA on MAE (log death rates): F={F:.3f}, p={p:.4f}")
        print(f"Paper finding: p < 0.001 (highly significant)")
        print(f"(Full 2-way with country effects would separate LC more clearly)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HMD_USER = input("HMD email: ")
    HMD_PASS = getpass.getpass("HMD password: ")

    
    DEMO_MODE = not (HMD_USER and HMD_PASS)

    if DEMO_MODE:
        print("="*70)
        print("DEMO MODE: No HMD credentials found.")
        print("Using synthetic Gompertz-Makeham mortality data for Sweden (male).")
        print("To use real data: set HMD_USER and HMD_PASS environment variables.")
        print("="*70)
        
        # Build synthetic dataset mimicking all 10 countries both sexes
        mx_all = {}
        np.random.seed(42)
        for c in COUNTRIES:
            for sex in SEXES:
                # Slightly vary parameters by country/sex for realism
                noise_seed = hash(c["name"] + sex) % 1000
                np.random.seed(noise_seed)
                base = load_synthetic_demo()
                # Female mortality ~20% lower
                factor = 0.80 if sex == "female" else 1.0
                mx_all[(c["name"], sex)] = base * factor * np.random.uniform(0.9, 1.1)
    else:
        print("Logging in to mortality.org...")
        opener = _hmd_login(HMD_USER, HMD_PASS)
        print("Fetching real HMD data...")
        mx_all = {}
        for c in COUNTRIES:
            print(f"  {c['name']} ({c['code']})...")
            try:
                data = fetch_hmd_mx(c["code"], opener)
                for sex in SEXES:
                    df_sex = data[sex]
                    # Filter ages: 0 to MAX_AGE
                    df_sex = df_sex[df_sex.index <= MAX_AGE]
                    # Filter years: country fitting start → EVAL_END (2006)
                    valid_cols = [yr for yr in df_sex.columns
                                  if c["start_long"] <= yr <= EVAL_END]
                    df_sex = df_sex[valid_cols]
                    print(f"    {sex}: ages {df_sex.index[0]}-{df_sex.index[-1]}, "
                          f"years {df_sex.columns[0]}-{df_sex.columns[-1]} "
                          f"({len(df_sex.columns)} years)")
                    mx_all[(c["name"], sex)] = df_sex
            except Exception as e:
                print(f"    FAILED ({type(e).__name__}): {e}")

    print(f"\nRunning evaluation on {len(mx_all)} population datasets...")
    df_log, df_e0 = run_evaluation(mx_all, demo_mode=DEMO_MODE)
    
    print_comparison_table(df_log, df_e0)
    run_anova_check(df_log)

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "mae_log_death_rates.csv")
    e0_path  = os.path.join(results_dir, "mae_life_expectancy.csv")
    try:
        df_log.to_csv(log_path, index=False)
        df_e0.to_csv(e0_path,  index=False)
        print(f"\nResults saved to {results_dir}")
    except PermissionError:
        print(f"\nCould not save CSVs — close any open Excel files in {results_dir} and re-run.")
    
    if DEMO_MODE:
        print("\n⚠  NOTE: Synthetic data results won't match paper exactly.")
        print("   Register at mortality.org and set HMD_USER/HMD_PASS for real results.")
    else:
        print("\nReal HMD results saved to results/")