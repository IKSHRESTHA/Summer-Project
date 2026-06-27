"""
Monte Carlo Dropout uncertainty quantification for LSTM mortality models.

Implementation follows Gal & Ghahramani (2016): keep dropout ACTIVE at
inference time (model.train() mode) and run N stochastic forward passes.
The empirical distribution of N predictions gives a non-parametric
95% prediction interval at near-zero additional cost.

Usage
-----
    mean_pred, lo95, hi95 = mc_predict(model, x_tensor, n_samples=200)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train import destandardise


def mc_predict(
    model: nn.Module,
    loader: DataLoader,
    n_samples: int = 200,
    device: str = "cpu",
) -> tuple:
    """
    Monte Carlo Dropout inference: N stochastic forward passes.

    The model is kept in train() mode so dropout layers remain active,
    producing a distribution of predictions from which we derive
    empirical prediction intervals.

    Parameters
    ----------
    model     : trained LSTM (any variant with dropout layers)
    loader    : DataLoader for the target period (e.g., test)
    n_samples : number of stochastic forward passes
    device    : "cpu" or "cuda"

    Returns
    -------
    mean_z : (N, n_ages)  mean standardised prediction
    lo_z   : (N, n_ages)  2.5th percentile
    hi_z   : (N, n_ages)  97.5th percentile
    """
    model = model.to(device)
    was_training = model.training
    model.train()   # keep dropout active

    all_samples = []   # will be (n_samples, N_batches_total, n_ages)

    try:
        with torch.no_grad():
            for _ in range(n_samples):
                batch_preds = []
                for xb, _ in loader:
                    out = model(xb.to(device)).cpu().numpy()
                    batch_preds.append(out)
                all_samples.append(np.concatenate(batch_preds, axis=0))
    finally:
        # always restore original training/eval mode so subsequent
        # deterministic predict_all calls are not affected by active dropout
        if not was_training:
            model.eval()

    all_samples = np.stack(all_samples, axis=0)   # (n_samples, N, n_ages)
    mean_z = all_samples.mean(axis=0)             # (N, n_ages)
    lo_z   = np.percentile(all_samples,  2.5, axis=0)
    hi_z   = np.percentile(all_samples, 97.5, axis=0)
    return mean_z, lo_z, hi_z


def mc_predict_destd(
    model: nn.Module,
    loader: DataLoader,
    mean: np.ndarray,
    std: np.ndarray,
    n_samples: int = 200,
    device: str = "cpu",
) -> dict:
    """
    MC Dropout + de-standardisation → log(mx) prediction intervals.

    Returns
    -------
    dict with:
        "mean" : (n_ages, N)  mean log(mx) prediction
        "lo95" : (n_ages, N)  lower 95% interval
        "hi95" : (n_ages, N)  upper 95% interval
    """
    mean_z, lo_z, hi_z = mc_predict(model, loader, n_samples=n_samples, device=device)
    return {
        "mean": destandardise(mean_z, mean, std).T,   # (n_ages, N)
        "lo95": destandardise(lo_z,   mean, std).T,
        "hi95": destandardise(hi_z,   mean, std).T,
    }
