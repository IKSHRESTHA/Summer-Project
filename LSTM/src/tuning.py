"""
Optuna-based hyperparameter optimisation for LSTM mortality models.

Uses Tree-structured Parzen Estimator (TPE) with MedianPruner.
Search space follows Agent 1 findings and actuarial literature (Hybrid-Lift 2025).

Usage
-----
    best = run_hpo(model_class, loaders, n_trials=20, device="cpu")
    # best is a dict of hyperparameters
"""

import numpy as np
import torch
import torch.nn as nn

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from src.train import train_model


# ── search space (Agent 1 literature-derived) ─────────────────────────────────
SEARCH_SPACE = {
    "hidden_size":   [64, 128, 256],
    "dropout":       (0.1, 0.4),
    "lr":            (1e-4, 5e-3),
    "weight_decay":  (1e-6, 1e-3),
}


def _suggest(trial, model_name: str) -> dict:
    """Sample one hyperparameter configuration from the search space."""
    hidden_size = trial.suggest_categorical("hidden_size", SEARCH_SPACE["hidden_size"])
    dropout     = trial.suggest_float("dropout",     *SEARCH_SPACE["dropout"],    step=0.05)
    lr          = trial.suggest_float("lr",          *SEARCH_SPACE["lr"],         log=True)
    weight_decay = trial.suggest_float("weight_decay", *SEARCH_SPACE["weight_decay"], log=True)
    return {"hidden_size": hidden_size, "dropout": dropout,
            "lr": lr, "weight_decay": weight_decay}


def _make_model(model_class, n_ages: int, hidden_size: int, dropout: float):
    """Instantiate a model with the given hidden_size and dropout."""
    import inspect
    sig = inspect.signature(model_class.__init__)
    kwargs = {"n_ages": n_ages, "hidden_size": hidden_size, "dropout": dropout}
    # StackedLSTM has no kernel_size; CNNLSTM has cnn_channels — pass only what it accepts
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return model_class(**valid)


def run_hpo(
    model_class,
    loaders: dict,
    n_ages: int = 100,
    n_trials: int = 20,
    hpo_epochs: int = 40,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Run Optuna HPO for `model_class` on the given DataLoaders.

    Parameters
    ----------
    model_class : one of VanillaLSTM, StackedLSTM, etc.
    loaders     : dict with "train" and "val" DataLoaders
    n_ages      : input / output dimension
    n_trials    : number of Optuna trials (20 recommended for a 3-value grid)
    hpo_epochs  : max epochs per trial (lower than final training for speed)
    device      : "cpu" or "cuda"
    seed        : for Optuna sampler reproducibility

    Returns
    -------
    best_params : dict with best hyperparameter values
    """
    if not HAS_OPTUNA:
        print("  [HPO] optuna not installed — using defaults")
        return {"hidden_size": 128, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4}

    def objective(trial):
        params = _suggest(trial, model_class.__name__)
        model  = _make_model(model_class, n_ages,
                             params["hidden_size"], params["dropout"])
        hist = train_model(
            model, loaders,
            epochs=hpo_epochs,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            patience=8,
            device=device,
            verbose=False,
        )
        return min(hist["val_loss"])   # objective: minimise best val loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"    best val_loss={study.best_value:.5f}  params={best}")
    return best


def hpo_all_models(model_classes: dict, loaders: dict, n_ages: int = 100,
                   n_trials: int = 20, hpo_epochs: int = 40,
                   device: str = "cpu") -> dict:
    """
    Run HPO for each model in `model_classes` dict.

    Returns
    -------
    best_params : {"ModelName": {hyperparameter dict}}
    """
    results = {}
    for name, cls in model_classes.items():
        print(f"  HPO: {name} ({n_trials} trials) ...")
        results[name] = run_hpo(cls, loaders, n_ages=n_ages,
                                n_trials=n_trials, hpo_epochs=hpo_epochs,
                                device=device)
    return results
