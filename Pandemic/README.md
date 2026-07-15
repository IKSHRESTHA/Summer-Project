# Mortality Forecasting Through COVID-19 — Replication Package
Paper: paper.qmd (Quarto -> Typst PDF). Pre-specification: PANDEMIC_HYPOTHESES.md
(written before any post-2019 outcome was computed; see git history).
Run: `python run_pandemic.py` (resumable; outputs/<pop>.json + _arrays.npz).
Environment: Python 3.11+, torch, numpy, pandas, scipy, matplotlib (venv: ../leecarter).
Data: HMD parquet extracts in ../data (not redistributed; HMD user agreement).
Extraction vintage: June 2026. License: MIT (code); paper text (c) authors.
