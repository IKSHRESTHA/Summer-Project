# Pre-registered hypotheses: mortality forecasting through COVID-19
**Written 2026-07-14, BEFORE any model was fitted to, or any forecast compared
against, post-2019 mortality data.** The only post-2019 information examined to
date is an availability audit (which years exist per population; zero-cell
counts). No mortality level, trend, or model outcome has been inspected.

## Design (frozen)

Populations: all 20 (10 HMD countries x 2 sexes). Core shock window 2020–2021
(all 20); extended window to 2022 (18 populations) and 2023 (16), as available.
Models and every hyperparameter identical to the companion Performer study:
Lee-Carter (SVD + RWD drift), LSTM/Transformer/Performer (frozen configs,
10 seeds), training through 2019 on the full 1950–2019 window (s=10,
per-age z-scoring on training years only, Adam 1e-3, batch 32, 100 epochs).

Experiments:
- **A. Shock anatomy (frozen models):** train through 2019; recursive forecasts
  for 2020..2022/23; errors by year, by age, and signed bias (observed minus
  predicted log-mortality; positive = model under-predicted mortality).
  Secondary metric: implied period life expectancy at birth (Chiang).
- **B. Recovery under maintenance:** annually maintained models (rolling-refit
  LC; annually-retrained LSTM, 10 seeds) at origins 2015..2022 (as data allow);
  one-step forecasts per origin; recovery ratio = post-shock one-step RMSE
  divided by that population's mean 2016–2019 one-step RMSE.
- **C. Revision stability (new metric):** at each origin, each maintained model
  emits a 5-year forecast path; revision(o) = mean absolute change, over
  overlapping target years, between the paths issued at origins o-1 and o.
  Baseline revisions = mean over calm origins (2016–2019); shock revision =
  revision at origin 2020 (first ingestion of a shock year).

## Hypotheses

**P1 (shock is model-invariant).** In 2020, every frozen model under-predicts
aggregate mortality (positive mean signed bias) in at least 18 of 20
populations; and the spread across the four models' 2020 RMSE within a
population is less than half of the median 2020 RMSE itself in at least 15 of
20. Rationale: an exogenous shock cannot be anticipated from the training
window by any of these models; differences between them reflect baseline
trend, not shock capture.

**P2 (age anatomy).** The 2020 under-prediction is concentrated at older ages:
ages 60+ contribute a larger share of total squared error in 2020 than they
did in the same model's pre-COVID test errors (2010–2019, extended-split
recursive) in at least 14 of 20 populations.

**P3 (maintained models recover fast).** By origin 2021 (i.e., forecasting
2022 after ingesting two shock years), both maintained models' one-step RMSE
is within 50 per cent of their own calm-period (2016–2019) mean in at least
12 of the 18 populations with 2022 data.

**P4 (networks are noisier in calm times).** Over calm origins 2016–2019, the
retrained LSTM's mean 5-year-path revision exceeds rolling LC's in at least
14 of 20 populations. (An expected cost of the network's accuracy advantage;
we commit to reporting the multiple.)

**P5 (the classical model over-reacts to the shock).** The ratio
(shock revision at origin 2020) / (own calm-period mean revision) is larger
for rolling LC than for the retrained LSTM in at least 12 of 20 populations.
Mechanism: the RWD jump-off and endpoint-anchored drift transmit a single
extreme year directly into the entire projected path, whereas the network's
learned mapping partially smooths inputs unlike its training distribution.

**P6 (accuracy ranking survives the shock era).** Over post-shock one-step
forecasts (origins 2020 onward pooled), the retrained LSTM's RMSE is lower
than rolling LC's in at least 11 of 18 populations with such data --- i.e.,
the maintenance finding of the companion study persists through a structural
break.

## Falsification consequences
Failures will be reported as failures. P5 is the riskiest and most novel; its
failure would itself be a publishable finding about shock transmission in
maintained forecasting systems. No hypothesis will be re-scored, re-thresholded,
or dropped after data contact.
