"""
Agent 2: Corrected Statistical Evaluation
- HAC (Newey-West) Diebold-Mariano test
- Block Bootstrap DM test
- Fair baseline: LC Multi-step vs LSTMs
"""
import os, sys, warnings
os.chdir(r"c:\Users\krish\OneDrive\Desktop\Summer Project\LSTM")
sys.path.insert(0, r"c:\Users\krish\OneDrive\Desktop\Summer Project\LSTM")

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.data import make_loaders, load_country_sex, SEQ_LEN
from src.train import predict_all, destandardise
from src.lee_carter import fit_lee_carter, lc_forecast, project_rates, lc_rolling_forecast

warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
loaders, meta = make_loaders("AUS", "male", batch_size=32, seed=42)
log_mx_raw = meta["log_mx_raw"]
n_train, n_val, n_test = meta["n_train"], meta["n_val"], meta["n_test"]
mean, std = meta["mean"], meta["std"]

# True test log-rates: shape (100, 10)
test_true = log_mx_raw[:, n_train + n_val : n_train + n_val + n_test]
# Training log-rates for LC fitting (1950-2004)
train_lm = log_mx_raw[:, :n_train]
ages = meta["ages"]
years = meta["years"]

# ── Lee-Carter predictions ─────────────────────────────────────────────────────
print("Computing Lee-Carter predictions...")

# LC Multi-step: fit once on 1950-2004, forecast 10+5=15 steps, take last 10
lc_train = fit_lee_carter(train_lm)
kt_multi  = lc_forecast(lc_train["kt"], h=n_val + n_test)  # forecast val+test years
lc_multi_log = project_rates(lc_train, kt_multi[-n_test:])  # (100, 10)

# LC Rolling: use the existing helper (refits at each test year)
lc_rolling_log = lc_rolling_forecast(log_mx_raw, n_train, n_val, n_test)  # (100, 10)

# ── Load LSTM models ───────────────────────────────────────────────────────────
print("Loading LSTM models and generating predictions...")

MODEL_FILES = {
    "Stacked LSTM":    "outputs/models/AUS_male_Stacked_LSTM.pt",
    "BiLSTM":          "outputs/models/AUS_male_BiLSTM.pt",
    "CNN-LSTM":        "outputs/models/AUS_male_CNN-LSTM.pt",
    "Attention LSTM":  "outputs/models/AUS_male_Attention_LSTM.pt",
    "Vanilla LSTM":    "outputs/models/AUS_male_Vanilla_LSTM.pt",
}

from src.models.stacked_lstm     import StackedLSTM
from src.models.bidirectional_lstm import BidirectionalLSTM as BiLSTM
from src.models.cnn_lstm         import CNNLSTM
from src.models.attention_lstm   import AttentionLSTM
from src.models.vanilla_lstm     import VanillaLSTM

MODEL_CLASSES = {
    "Stacked LSTM":   StackedLSTM,
    "BiLSTM":         BiLSTM,   # BidirectionalLSTM aliased above
    "CNN-LSTM":       CNNLSTM,
    "Attention LSTM": AttentionLSTM,
    "Vanilla LSTM":   VanillaLSTM,
}

lstm_preds = {}  # model_name -> (100, 10) log-rate predictions

for name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        continue
    cls = MODEL_CLASSES[name]
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Detect hidden_size from weight_hh key (shape [4H, H])
    hh_key = next(k for k in state if k.endswith("weight_hh_l0"))
    hidden_size = state[hh_key].shape[1]   # H dimension
    model = cls(n_ages=100, hidden_size=hidden_size, dropout=0.2)
    model.load_state_dict(state)
    model.eval()

    preds_z = predict_all(model, loaders["test"], "cpu")   # (10, 100) standardised
    preds_lm = destandardise(preds_z, mean, std).T          # (100, 10)
    lstm_preds[name] = preds_lm
    rmse = float(np.sqrt(np.mean((test_true - preds_lm)**2)))
    print(f"  {name}: RMSE={rmse:.4f}")

# ── HAC (Newey-West) DM test ───────────────────────────────────────────────────

def newey_west_variance(d: np.ndarray, lag: int) -> float:
    """Newey-West HAC variance estimator."""
    T = len(d)
    d_dm = d - d.mean()
    s = np.sum(d_dm**2) / T  # S_0
    for j in range(1, lag + 1):
        w = 1 - j / (lag + 1)  # Bartlett weight
        gamma_j = np.sum(d_dm[j:] * d_dm[:-j]) / T
        s += 2 * w * gamma_j
    return max(s, 1e-12)  # guard against negative (small T)


def dm_hac(true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, lag: int = 3):
    """HAC-corrected Diebold-Mariano test. Returns stat and approximate p-value."""
    loss_a = ((true - pred_a)**2).mean(axis=0)  # (T,)
    loss_b = ((true - pred_b)**2).mean(axis=0)
    d = loss_a - loss_b
    T = len(d)
    d_mean = d.mean()
    var_nw = newey_west_variance(d, lag)
    se_hac = np.sqrt(var_nw / T)
    stat = d_mean / se_hac
    # Use normal approximation (two-sided)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return {"stat": float(stat), "p_hac": float(p), "d_mean": float(d_mean),
            "d_series": d}


def dm_block_bootstrap(d: np.ndarray, block_size: int = 3, n_boot: int = 2000,
                        seed: int = 42) -> float:
    """Block bootstrap p-value for H0: mean(d) = 0."""
    rng = np.random.default_rng(seed)
    T = len(d)
    d_centred = d - d.mean()  # centre under H0
    boot_means = []
    for _ in range(n_boot):
        resample = []
        while len(resample) < T:
            start = rng.integers(0, T)
            block = d_centred[start : start + block_size]
            if len(block) == 0:
                continue
            resample.extend(block.tolist())
        boot_means.append(np.mean(resample[:T]))
    boot_means = np.array(boot_means)
    # two-sided p-value
    p = np.mean(np.abs(boot_means) >= abs(d.mean()))
    return float(p)


# ── Run all comparisons ────────────────────────────────────────────────────────
print("\n=== Agent 2: HAC and Bootstrap DM Tests ===\n")

results = []
for name, pred_lm in lstm_preds.items():
    # Comparison A: vs LC Rolling (paper's baseline)
    res_rolling_hac = dm_hac(test_true, pred_lm, lc_rolling_log, lag=3)
    res_rolling_boot_p = dm_block_bootstrap(res_rolling_hac["d_series"], block_size=3)

    # Comparison B: vs LC Multi-step (fair same-regime baseline)
    res_multi_hac = dm_hac(test_true, pred_lm, lc_multi_log, lag=3)
    res_multi_boot_p = dm_block_bootstrap(res_multi_hac["d_series"], block_size=3)

    # Plain t-test (paper's method) for reference
    loss_lstm   = ((test_true - pred_lm)**2).mean(axis=0)
    loss_rolling = ((test_true - lc_rolling_log)**2).mean(axis=0)
    loss_multi  = ((test_true - lc_multi_log)**2).mean(axis=0)

    d_r = loss_lstm - loss_rolling
    d_m = loss_lstm - loss_multi
    t_r, p_r = stats.ttest_1samp(d_r, popmean=0.0)
    t_m, p_m = stats.ttest_1samp(d_m, popmean=0.0)

    rmse = float(np.sqrt(np.mean((test_true - pred_lm)**2)))

    results.append({
        "Model": name,
        "RMSE": round(rmse, 4),
        # vs LC Rolling
        "Plain_t_vs_Rolling":   round(t_r, 3),
        "Plain_p_vs_Rolling":   f"{p_r:.4e}",
        "HAC_t_vs_Rolling":     round(res_rolling_hac["stat"], 3),
        "HAC_p_vs_Rolling":     f"{res_rolling_hac['p_hac']:.4e}",
        "Boot_p_vs_Rolling":    f"{res_rolling_boot_p:.4f}",
        "Sig_5pct_vs_Rolling":  res_rolling_hac["p_hac"] < 0.05,
        # vs LC Multi-step
        "Plain_t_vs_Multi":     round(t_m, 3),
        "Plain_p_vs_Multi":     f"{p_m:.4e}",
        "HAC_t_vs_Multi":       round(res_multi_hac["stat"], 3),
        "HAC_p_vs_Multi":       f"{res_multi_hac['p_hac']:.4e}",
        "Boot_p_vs_Multi":      f"{res_multi_boot_p:.4f}",
        "Sig_5pct_vs_Multi":    res_multi_hac["p_hac"] < 0.05,
    })

    print(f"{name}  (RMSE={rmse:.4f})")
    print(f"  vs LC Rolling | Plain t={t_r:.3f} p={p_r:.4e} | HAC t={res_rolling_hac['stat']:.3f} p={res_rolling_hac['p_hac']:.4e} | Boot p={res_rolling_boot_p:.4f}")
    print(f"  vs LC Multi   | Plain t={t_m:.3f} p={p_m:.4e} | HAC t={res_multi_hac['stat']:.3f} p={res_multi_hac['p_hac']:.4e} | Boot p={res_multi_boot_p:.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv("outputs/results/agent2_dm_corrected.csv", index=False)
print("\nSaved to outputs/results/agent2_dm_corrected.csv")

# ── Summary Table ─────────────────────────────────────────────────────────────
print("\n=== CORRECTED DM TEST SUMMARY ===")
print(f"{'Model':<20} {'Paper p':<12} {'HAC p (Roll)':<14} {'Boot p (Roll)':<14} {'HAC p (Multi)':<14} {'Boot p (Multi)':<15} {'Sig@5% vs Multi'}")
print("-" * 110)
for r in results:
    print(f"{r['Model']:<20} <0.001       {r['HAC_p_vs_Rolling']:<14} {r['Boot_p_vs_Rolling']:<14} {r['HAC_p_vs_Multi']:<14} {r['Boot_p_vs_Multi']:<15} {r['Sig_5pct_vs_Multi']}")

print("\nDone.")
