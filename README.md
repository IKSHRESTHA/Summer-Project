# Lee-Carter Mortality Forecasting

Stochastic mortality forecasting using the **Lee-Carter model** (Lee & Carter, 1992), with extensions covering multiple countries, LSTM-based deep learning, and multi-variant comparisons.

---

## Repository Contents

| File / Directory | Description |
|---|---|
| `lee_carter_final.ipynb` | Core Lee-Carter implementation for Australia (male & female) |
| `review_paper_1.ipynb` | Replication of Booth et al. (2006) across 10 countries and 5 LC variants |
| `LSTM/` | LSTM-based mortality forecasting pipeline (Richman & Schreck 2019) |
| `scripts/` | Utility scripts for data download and standalone replication |
| `data/` | HMD parquet files — gitignored, requires HMD registration |

### `lee_carter_final.ipynb`

Full Lee-Carter implementation for **Australia**, applied to HMD parquet data for both sexes.

| Section | Description |
|---|---|
| 1. Setup | Imports and plot configuration |
| 2. Data Loading | Load and reshape HMD parquet files to age x year matrices |
| 3. EDA | Log-mortality surface, age profiles, time trends |
| 4. Model Fitting | SVD estimation of ax, bx, kt |
| 5. Diagnostics | Residual heatmap, distribution, and marginal plots |
| 6. kt Forecasting | ARIMA model selection, Ljung-Box, ACF/PACF |
| 7. kt Projection | Multi-year forecast with prediction intervals |
| 8. Mortality Rates | Age-specific projected central death rates |
| 9. Life Expectancy | Period e0 history and projection |
| 10. Backtesting | Out-of-sample validation |
| 11. Bootstrap | Residual bootstrap for empirical uncertainty bands |

### `review_paper_1.ipynb`

Replication of **Booth et al. (2006)** across 10 countries (Australia, Canada, Denmark, England, Finland, France, Italy, Norway, Sweden, Switzerland) and 5 LC variants (LC, LM, BMS, HU, DJT), with 2-way ANOVA to assess method differences.

### `LSTM/`

LSTM-based mortality forecasting pipeline extending the Lee-Carter framework with deep learning. Run via:

```bash
python LSTM/run_all.py
```

### `scripts/`

- `download_hmd_data.py` — Downloads HMD data and saves as parquet files in `data/`
- `paper1.py` — Standalone command-line version of the Booth et al. replication

---

## The Model

$$\ln m(x,t) = a_x + b_x \, k_t + \varepsilon_{x,t}$$

Parameters are estimated by **SVD** of the centred log-rate matrix. The mortality index $k_t$ is forecast by **ARIMA(0,1,0) with drift** (random walk with drift).

---

## Data

Source: **Human Mortality Database (HMD)** — [mortality.org](https://mortality.org)

Free registration required. Download HMD data using:

```bash
python scripts/download_hmd_data.py
```

Or manually place Age x Year parquet files in `data/`. The `data/` directory is gitignored.

---

## Setup

```bash
python -m venv leecarter

# Windows
leecarter\Scripts\activate

# Mac/Linux
source leecarter/bin/activate

pip install -r requirements.txt

# PyTorch (CPU only)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Python**: 3.14  
**Key packages**: numpy, pandas, scipy, statsmodels, matplotlib, seaborn, pmdarima, pyarrow, torch, optuna, scikit-learn

---

## References

- Lee, R. D., & Carter, L. R. (1992). Modeling and forecasting U.S. mortality. *Journal of the American Statistical Association*, 87(419), 659-671.
- Booth, H., Hyndman, R. J., Tickle, L., & de Jong, P. (2006). Lee-Carter mortality forecasting: a multi-country comparison of variants and extensions. *Demographic Research*, 15, 289-310.
- Richman, R., & Schreck, M. (2019). Lee and Carter go machine learning: Recurrent neural networks. SSRN working paper.
