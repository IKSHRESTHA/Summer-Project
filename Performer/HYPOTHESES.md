# Pre-registered hypotheses for the confirmation set
**Written: 2026-07-14, BEFORE any model was run on the confirmation populations.**

## Design

All findings to date were obtained on the **discovery set**: male mortality in
UK (GBRTENW), France, Italy, Denmark, Canada, Finland — the six populations used
in the Wang et al. (2024) replication. The split choice (extended: train
1950–2009 / test 2010–2019) and all protocol analyses were developed on that set,
making those results exploratory.

The **confirmation set** consists of the 14 populations never touched by any
experiment in this project:

- Female mortality: GBRTENW, FRATNP, ITA, DNK, CAN, FIN
- Both sexes: AUS, CHE, NOR, SWE

Pipeline, hyperparameters, seeds (0–9), splits, protocols, and metrics are
frozen as of this document; nothing will be tuned on the confirmation set.

## Hypotheses (stated before running)

**H1 (long-horizon classical advantage).** On the benchmark split (train
1950–2000, 19-year recursive test), Lee-Carter attains lower test RMSE than
every frozen network in at least 12 of 14 populations.

**H2 (horizon dependence).** On the extended split (train 1950–2009, 10-year
recursive test), the median relative RMSE gap of the best frozen network vs
Lee-Carter shrinks by at least half compared to the benchmark split, and the
best frozen network is within ±10% of Lee-Carter in at least 7 of 14
populations.

**H3 (maintenance decides).** The annually-retrained LSTM (rolling one-step,
extended split) attains lower test RMSE than annually-refitted rolling
Lee-Carter in at least 9 of 14 populations.

**H4 (attention equivalence).** Paired-by-seed t-tests between Performer and
Transformer (recursive RMSE) are non-significant at the 5% level in at least
10 of 14 populations on each split, and the Performer's seed standard deviation
is lower in at least 10 of 14.

**H5 (in-sample regularisation).** The Performer records the lowest in-sample
RMSE of the three networks in at least 9 of 14 populations on the extended split.

## Falsification consequences

If H1–H3 fail, the paper's central "extrapolation distance and maintenance
regime" narrative does not generalise and will be reported as
discovery-set-specific. If H4–H5 fail, the Performer regularisation claim will
be withdrawn. Confirmation results will be reported in full regardless of outcome.
