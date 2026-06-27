"""
Walk-forward (expanding-window) cross-validation for mortality forecasting.

Protocol follows Agent 1 / R Journal 2025 recommendation:
  - Expanding window: use ALL data from year 0 to each fold origin
  - Fixed test horizon: TEST_YEARS years per fold
  - Val window: VAL_YEARS years immediately before each test period
  - Minimum training window: 30 years

Usage
-----
    folds  = make_folds(log_mx, years, n_folds=3)
    scores = crossval_model(model_class, best_params, log_mx, mean, std,
                            folds, n_ages=100, epochs=100, device="cpu")
"""

import numpy as np
import torch

from src.data  import MortalityDataset, make_loaders_from_slice, SEQ_LEN, VAL_YEARS, TEST_YEARS
from src.train import train_model, predict_all, destandardise


def make_folds(T: int, n_folds: int = 3,
               min_train: int = 30, val_years: int = VAL_YEARS,
               test_years: int = TEST_YEARS,
               reserved_test: int = TEST_YEARS) -> list:
    """
    Generate expanding-window fold boundaries.

    Each fold is a dict:
        {"n_train": int, "n_val": int, "n_test": int}

    IMPORTANT: the last `reserved_test` years of the full dataset are the
    held-out final test set and must never appear in any CV fold's test
    window.  Without this guard, the last fold's test period is identical
    to the final evaluation split, causing test-data leakage into HPO.

    Parameters
    ----------
    T             : total time steps available
    n_folds       : number of CV folds
    min_train     : minimum training years required
    reserved_test : years at the end reserved for final evaluation (excluded
                    from all CV folds). Defaults to TEST_YEARS.

    Returns
    -------
    list of fold dicts, chronologically ordered
    """
    # CV only operates on the first T - reserved_test years
    T_cv = T - reserved_test

    # within those T_cv years: latest training-end leaves val+test at the end
    max_train_end = T_cv - val_years - test_years
    min_train_end = min_train

    if max_train_end <= min_train_end:
        return [{"n_train": max_train_end, "n_val": val_years, "n_test": test_years}]

    train_ends = np.round(np.linspace(min_train_end, max_train_end, n_folds)).astype(int)
    folds = []
    for te in train_ends:
        folds.append({"n_train": int(te), "n_val": val_years, "n_test": test_years})
    return folds


def _make_model(model_class, n_ages: int, params: dict):
    """Instantiate a model from the HPO-best params dict."""
    import inspect
    sig    = inspect.signature(model_class.__init__)
    kwargs = {"n_ages": n_ages,
              "hidden_size": params.get("hidden_size", 128),
              "dropout":     params.get("dropout", 0.2)}
    valid  = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return model_class(**valid)


def crossval_model(
    model_class,
    best_params: dict,
    log_mx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    folds: list,
    n_ages: int = 100,
    epochs: int = 80,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Run walk-forward CV for one model class.

    Returns
    -------
    dict with:
        "mae_folds"  : (n_folds,) MAE per fold
        "rmse_folds" : (n_folds,) RMSE per fold
        "mae_mean"   : float
        "mae_std"    : float
        "rmse_mean"  : float
        "rmse_std"   : float
    """
    from src.evaluate import compute_metrics

    mae_list, rmse_list = [], []

    for i, fold in enumerate(folds):
        n_train = fold["n_train"]
        n_val   = fold["n_val"]
        n_test  = fold["n_test"]

        # scaler re-fitted on THIS fold's training data only
        fold_mean = log_mx[:, :n_train].mean(axis=1)
        fold_std  = log_mx[:, :n_train].std(axis=1)

        loaders = make_loaders_from_slice(
            log_mx, fold_mean, fold_std,
            n_train=n_train, n_val=n_val, n_test=n_test,
            batch_size=32, seq_len=SEQ_LEN, seed=seed
        )

        model = _make_model(model_class, n_ages, best_params)
        hist  = train_model(model, loaders, epochs=epochs,
                            lr=best_params.get("lr", 1e-3),
                            weight_decay=best_params.get("weight_decay", 1e-4),
                            patience=12, device=device, verbose=False)

        # test predictions — test loader has exactly n_test samples
        preds_z  = predict_all(model, loaders["test"], device)       # (n_test, n_ages)
        preds_lm = destandardise(preds_z, fold_mean, fold_std).T     # (n_ages, n_test)

        true_lm  = log_mx[:, n_train + n_val : n_train + n_val + n_test]
        metrics  = compute_metrics(true_lm, preds_lm)
        mae_list.append(metrics["MAE"])
        rmse_list.append(metrics["RMSE"])

    mae_arr  = np.array(mae_list)
    rmse_arr = np.array(rmse_list)
    return {
        "mae_folds":  mae_arr,
        "rmse_folds": rmse_arr,
        "mae_mean":   float(mae_arr.mean()),
        "mae_std":    float(mae_arr.std()),
        "rmse_mean":  float(rmse_arr.mean()),
        "rmse_std":   float(rmse_arr.std()),
    }
