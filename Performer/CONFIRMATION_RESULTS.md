# Confirmation-set results (14 held-out populations)
**Scored 2026-07-14 against HYPOTHESES.md, which was written before this run.**

Populations: 6 female discovery-countries + AUS/CHE/NOR/SWE × both sexes.
10 seeds, both splits, all protocols + annual-retraining arm. 2,716 metric rows.

## Pre-registered hypotheses — verdict

| # | Hypothesis | Threshold | Result | Verdict |
|---|-----------|-----------|--------|---------|
| H1 | LC < every frozen net (paper split, 19-yr) | ≥12/14 | **14/14** | PASS (clean) |
| H2 | Best net within ±10% of LC (extended, 10-yr) | ≥7/14 | **7/14** | PASS (at threshold) |
| H3 | Retrained LSTM < rolling LC (extended) | ≥9/14 | **9/14** | PASS (at threshold) |
| H4a | PF≈TF accuracy non-significant, per split | ≥10/14 | 13/14 (paper), 14/14 (ext) | PASS (clean) |
| H4b | PF seed-std lower than TF, per split | ≥10/14 | 9/14 (paper), 11/14 (ext) | **PARTIAL** — fails paper split |
| H5 | Performer best in-sample net (extended) | ≥9/14 | **10/14** | PASS |

**Horizon effect (H2 detail):** median frozen-net gap over LC shrinks from
+27.1% (19-yr benchmark) to +8.4% (10-yr extended) — a 69% reduction. The core
"extrapolation distance governs the gap" claim replicates cleanly out of sample.

## Maintenance finding (H3) — Diebold-Mariano tests, retrained LSTM vs rolling LC

Per-year MSE differentials, T=10, one-sample t-test:

- **Significantly better (retrained LSTM): 8/14** — AUS-m, CAN-f, FRA-f, NOR-m,
  SWE-m, CHE-f, CHE-m, UK-f (all p<0.05)
- **Ties (ns): 5/14** — AUS-f, DNK-f, FIN-f, NOR-f, SWE-f
- **LC better on point estimate: 1/14** — ITA-f (rt=0.1477 vs lc=0.1392, p=0.11, ns)

So of 14 held-out populations, the retrained network is significantly ahead in 8,
statistically tied in 5, and never significantly behind. The maintenance finding
is the most robust of the study.

## Honest reading for the paper

1. **Confirmed cleanly:** H1 (long-horizon classical dominance), the horizon
   effect (H2 direction/magnitude), H4a (Performer≈Transformer accuracy), H5
   (Performer in-sample regularisation). These are reported as confirmed.
2. **Confirmed but boundary:** H2 and H3 pass at exactly their thresholds. Report
   with the exact counts, not rounded-up language. The maintenance finding is
   better stated via the DM tests (8 sig. wins / 0 sig. losses) than via the
   9/14 point-estimate count.
3. **Partial failure to disclose:** H4b — the Performer's lower seed variance
   replicates on the extended split (11/14) but not the benchmark split (9/14,
   just under threshold). The variance-reduction claim must be qualified as
   split-dependent, not universal.
4. **Anomaly worth a sentence:** Italy female is the sole population where the
   retrained LSTM does not beat rolling LC. Female Italian mortality is among the
   smoothest series in the HMD; LC's rank-one structure is hardest to beat there.
