# Lee–Carter Mortality Forecasting

Stochastic mortality forecasting for the United States using the **Lee–Carter model** (Lee & Carter, 1992), applied to HMD period life tables covering 1950–2019.

---

## The Model

The Lee–Carter model decomposes the log central death rate as

$$\ln m(x,t) = a_x + b_x \, k_t + \varepsilon_{x,t}$$

| Parameter | Description |
|-----------|-------------|
| $a_x$ | Fixed age pattern — average log-mortality at age $x$ across all years |
| $b_x$ | Age sensitivity — how strongly age $x$ responds to a unit shift in $k_t$ |
| $k_t$ | Mortality index — the single time-varying driver of the entire surface |
| $\varepsilon_{x,t}$ | i.i.d. error |

The model is identified by the constraints $\sum_x b_x = 1$ and $\sum_t k_t = 0$.  
Parameters are estimated by **singular value decomposition (SVD)** of the centred log-rate matrix.

### Forecasting

$k_t$ follows a **random walk with drift** (ARIMA(0,1,0) + constant):

$$k_t = k_{t-1} + d + \sigma \varepsilon_t$$

where $d < 0$ is the average annual mortality improvement. Projecting $k_t$ forward yields the full age-specific forecast via $\hat{m}(x,t) = \exp(a_x + b_x \hat{k}_t)$.

---

## Data

Source: **Human Mortality Database**, USA period life tables (both sexes, 1×1).

```
https://mortality.org/File/GetDocument/hmd.v6/USA/STATS/bltper_1x1.txt
```

Free registration required at [mortality.org](https://mortality.org). After downloading, place the file at `data/bltper_1x1.txt`. See [`data/README.md`](data/README.md) for full instructions.

---

## Notebook Contents

[`lee_carter_final.ipynb`](lee_carter_final.ipynb)

| Section | Description |
|---------|-------------|
| 1. Setup | Imports and plot configuration |
| 2. Data | Load and reshape to age × year matrix |
| 3. EDA | Log-mortality surface, age profiles, time trends |
| 4. Model Fitting | SVD estimation of $a_x$, $b_x$, $k_t$ |
| 5. Diagnostics | Residual heatmap, distribution, and marginal plots |
| 6. Forecasting $k_t$ | ARIMA model selection, Ljung–Box, ACF/PACF |
| 7. $k_t$ Projection | 30-year forecast with 95% prediction intervals |
| 8. Mortality Rates | Age-specific projected rates |
| 9. Life Expectancy | Period $e_0$ history and 30-year projection |
| 10. Backtesting | Out-of-sample validation on 2001–2019 |
| 11. Bootstrap | Residual bootstrap for empirical uncertainty bands |

---

## Setup

```bash
# Clone
git clone https://github.com/<your-username>/lee-carter-mortality-forecasting.git
cd lee-carter-mortality-forecasting

# Install dependencies
pip install -r requirements.txt

# Download data (see data/README.md), then run the notebook
jupyter lab lee_carter_final.ipynb
```

---

## Results

| Metric | Value |
|--------|-------|
| Variance explained (1st component) | ~96% |
| Backtest RMSE (log mortality, 2001–2019) | ~0.025 |
| $e_0$ in 2019 (observed) | ~78.9 yr |
| $e_0$ in 2049 (projected) | ~84 yr |

---

## Reference

Lee, R. D., & Carter, L. R. (1992). Modeling and forecasting U.S. mortality. *Journal of the American Statistical Association*, 87(419), 659–671.
