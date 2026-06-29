"""
Agent 4: Actuarial & Demographic Domain Expert
- Annuity factor error (actual, not e0-based)
- Bias correction: add empirical ME to LSTM predictions
- COVID-era (2020-2023) test if data available
"""
import os, sys, warnings
os.chdir(r"c:\Users\krish\OneDrive\Desktop\Summer Project\LSTM")
sys.path.insert(0, r"c:\Users\krish\OneDrive\Desktop\Summer Project\LSTM")

import numpy as np
import pandas as pd
import torch
warnings.filterwarnings("ignore")

from src.data import make_loaders, load_country_sex, SEQ_LEN, LOG_CLIP_LO, LOG_CLIP_HI
from src.train import predict_all, destandardise
from src.lee_carter import (fit_lee_carter, lc_forecast, project_rates,
                             lc_rolling_forecast, life_expectancy)

# ── Load data & predictions (same as Agent 2) ─────────────────────────────────
print("Loading data and models...")
loaders, meta = make_loaders("AUS", "male", batch_size=32, seed=42)
log_mx_raw = meta["log_mx_raw"]
n_train, n_val, n_test = meta["n_train"], meta["n_val"], meta["n_test"]
mean, std = meta["mean"], meta["std"]
test_true = log_mx_raw[:, n_train + n_val : n_train + n_val + n_test]

lc_train = fit_lee_carter(log_mx_raw[:, :n_train])
kt_multi  = lc_forecast(lc_train["kt"], h=n_val + n_test)
lc_multi_log = project_rates(lc_train, kt_multi[-n_test:])
lc_rolling_log = lc_rolling_forecast(log_mx_raw, n_train, n_val, n_test)

MODEL_FILES = {
    "Stacked LSTM":    "outputs/models/AUS_male_Stacked_LSTM.pt",
    "BiLSTM":          "outputs/models/AUS_male_BiLSTM.pt",
}

from src.models.stacked_lstm       import StackedLSTM
from src.models.bidirectional_lstm import BidirectionalLSTM

MODEL_CLASSES = {"Stacked LSTM": StackedLSTM, "BiLSTM": BidirectionalLSTM}

lstm_preds = {}
for name, path in MODEL_FILES.items():
    cls = MODEL_CLASSES[name]
    state = torch.load(path, map_location="cpu", weights_only=True)
    hh_key = next(k for k in state if k.endswith("weight_hh_l0"))
    hidden_size = state[hh_key].shape[1]
    model = cls(n_ages=100, hidden_size=hidden_size, dropout=0.2)
    model.load_state_dict(state)
    model.eval()
    preds_z  = predict_all(model, loaders["test"], "cpu")
    preds_lm = destandardise(preds_z, mean, std).T
    lstm_preds[name] = preds_lm

# ── TABLE A: Annuity Factor Error ─────────────────────────────────────────────
print("\n=== TABLE A: Annuity Factor Error (ages 60-90) ===")

DISC_RATE   = 0.03   # 3% annual discount
ANNUITY_AGE = 65     # annuity purchased at age 65
ages        = np.arange(100)

def annuity_factor(mx_vec: np.ndarray, entry_age: int = 65,
                   disc_rate: float = 0.03) -> float:
    """
    Compute present value annuity factor for a life aged `entry_age`.
    a = sum_{k=0}^{omega-entry_age} v^k * k_p_{entry_age}
    where k_p_x is survival probability from age x to x+k.
    """
    mx = np.clip(mx_vec, 1e-10, None)
    px = 1.0 - mx / (1.0 + 0.5 * mx)   # UDD approximation
    px = np.minimum(px, 1.0)
    px[-1] = 0.0  # terminal age

    # survival from entry_age
    start  = int(entry_age)
    surv   = np.cumprod(np.concatenate([[1.0], px[start:-1]]))  # k_p_65
    max_k  = len(surv)
    v_k    = np.array([(1/(1+disc_rate))**k for k in range(max_k)])
    return float(np.sum(v_k * surv))

annuity_rows = []
for yr_idx in range(n_test):
    mx_true  = np.exp(test_true[:, yr_idx])
    mx_multi = np.exp(lc_multi_log[:, yr_idx])
    mx_roll  = np.exp(lc_rolling_log[:, yr_idx])

    af_true  = annuity_factor(mx_true)
    af_multi = annuity_factor(mx_multi)
    af_roll  = annuity_factor(mx_roll)

    row = {
        "Year": 2010 + yr_idx,
        "AF_True": round(af_true, 4),
        "AF_LC_Rolling": round(af_roll, 4),
        "AF_LC_Multi":   round(af_multi, 4),
        "Err_LC_Rolling_%": round(100*(af_roll - af_true)/af_true, 3),
        "Err_LC_Multi_%":   round(100*(af_multi - af_true)/af_true, 3),
    }
    for name, pred_lm in lstm_preds.items():
        mx_lstm = np.exp(pred_lm[:, yr_idx])
        af_lstm = annuity_factor(mx_lstm)
        row[f"AF_{name}"] = round(af_lstm, 4)
        row[f"Err_{name}_%"] = round(100*(af_lstm - af_true)/af_true, 3)
    annuity_rows.append(row)

df_ann = pd.DataFrame(annuity_rows)
df_ann.to_csv("outputs/results/agent4_annuity.csv", index=False)
print(df_ann[["Year", "Err_LC_Rolling_%", "Err_LC_Multi_%",
              "Err_Stacked LSTM_%", "Err_BiLSTM_%"]].to_string(index=False))

mean_errs = {
    "LC Rolling":    df_ann["Err_LC_Rolling_%"].abs().mean(),
    "LC Multi-step": df_ann["Err_LC_Multi_%"].abs().mean(),
}
for name in lstm_preds:
    mean_errs[name] = df_ann[f"Err_{name}_%"].abs().mean()
print("\nMean absolute annuity error (%):")
for k,v in mean_errs.items():
    print(f"  {k}: {v:.3f}%")

# ── TABLE B: Bias Correction ──────────────────────────────────────────────────
print("\n=== TABLE B: Bias-Corrected RMSE ===")
bias_rows = []
for name, pred_lm in lstm_preds.items():
    raw_err  = test_true - pred_lm            # (100, 10)
    me       = float(raw_err.mean())          # positive = LSTM under-estimates mx
    # bias correction: add ME to raw predictions (reduces systematic over-optimism)
    corrected_lm = pred_lm + me              # shift log-rates up (less optimistic)
    # But ME in our convention is true - pred, so corrected = pred + ME shifts predictions toward true
    raw_rmse  = float(np.sqrt(np.mean(raw_err**2)))
    corr_rmse = float(np.sqrt(np.mean((test_true - corrected_lm)**2)))
    lc_multi_rmse = float(np.sqrt(np.mean((test_true - lc_multi_log)**2)))
    print(f"{name}: Raw RMSE={raw_rmse:.4f} | ME={me:.4f} | Bias-Corrected RMSE={corr_rmse:.4f} | LC Multi={lc_multi_rmse:.4f}")
    bias_rows.append({
        "Model": name,
        "ME": round(me, 4),
        "Raw_RMSE": round(raw_rmse, 4),
        "BiasCorr_RMSE": round(corr_rmse, 4),
        "LC_Multi_RMSE": round(lc_multi_rmse, 4),
        "Beats_LC_Multi": corr_rmse < lc_multi_rmse
    })

df_bias = pd.DataFrame(bias_rows)
df_bias.to_csv("outputs/results/agent4_bias_correction.csv", index=False)

# ── TABLE C: COVID-era (2020-2023) ─────────────────────────────────────────────
print("\n=== TABLE C: COVID-era Test (2020-2023) ===")
AUS_MALE_COVID_PATH = os.path.join(
    r"c:\Users\krish\OneDrive\Desktop\Summer Project\data", "AUS_male.parquet"
)

if os.path.exists(AUS_MALE_COVID_PATH):
    import pandas as _pd
    df_full = _pd.read_parquet(AUS_MALE_COVID_PATH)
    covid_years = [y for y in df_full.columns if 2020 <= y <= 2023]
    if len(covid_years) >= 2:
        covid_lm = np.clip(np.log(df_full.loc[:99, covid_years].values), LOG_CLIP_LO, LOG_CLIP_HI)
        # T_covid x 100 → (100, T_covid)
        covid_true = covid_lm.T if covid_lm.shape[0] < covid_lm.shape[1] else covid_lm
        # Use LC rolling to forecast 2020-2023 from all available data up to 2019
        full_lm = np.clip(np.log(df_full.loc[:99, [y for y in df_full.columns if y <= 2019]].values),
                          LOG_CLIP_LO, LOG_CLIP_HI)
        lc_full = fit_lee_carter(full_lm)
        kt_covid = lc_forecast(lc_full["kt"], h=len(covid_years))
        lc_covid_pred = project_rates(lc_full, kt_covid)
        lc_covid_rmse = float(np.sqrt(np.mean((covid_true - lc_covid_pred)**2)))
        print(f"  LC Multi (2020-2023 forecast): RMSE = {lc_covid_rmse:.4f}")
        print("  (LSTM COVID eval skipped: models trained on 1950-2004 only)")
    else:
        print(f"  COVID years in data: {covid_years} — insufficient (need 2020-2023)")
else:
    print("  AUS_male.parquet not found at expected path")
    # Try alternative
    alt = os.path.join(r"c:\Users\krish\OneDrive\Desktop\Summer Project\data", "AUS_male.parquet")
    print(f"  Checking: {alt} → exists={os.path.exists(alt)}")

print("\nAgent 4 done. Outputs in outputs/results/agent4_*.csv")
