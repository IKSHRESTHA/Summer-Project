"""
Training utilities for LSTM mortality models.

Changes from v1 (Agent 3/5 fixes):
  - Adam → AdamW with weight_decay (Loshchilov & Hutter 2019)
  - ReduceLROnPlateau patience raised from 5 to 10 (fires after only 10
    gradient steps otherwise, which is too aggressive for small datasets)
  - best_model_state checkpoint saves and restores best weights before return
  - gradient clipping retained at max_norm=1.0
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    loaders: dict,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """
    Train model with AdamW + ReduceLROnPlateau + early stopping.

    Parameters
    ----------
    model        : un-trained module (will be modified in place)
    loaders      : dict with "train" and "val" DataLoaders
    epochs       : maximum training epochs
    lr           : initial learning rate
    weight_decay : L2 penalty in AdamW (default 1e-4)
    patience     : early-stopping patience (raised from original 10 to 15)
    device       : "cpu" or "cuda"
    verbose      : print loss per epoch

    Returns
    -------
    history : {
        "train_loss"       : list[float],
        "val_loss"         : list[float],
        "best_model_state" : state_dict of best checkpoint,
        "best_epoch"       : int,
    }
    """
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=10   # patience raised from 5→10
    )
    criterion = nn.MSELoss()

    best_val   = float("inf")
    wait       = 0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_loss += loss.item() * len(xb)
        t_loss /= max(len(loaders["train"].dataset), 1)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += criterion(model(xb), yb).item() * len(xb)
        v_loss /= max(len(loaders["val"].dataset), 1)

        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if verbose:
            print(f"Epoch {epoch:3d}  train={t_loss:.5f}  val={v_loss:.5f}")

        if v_loss < best_val:
            best_val   = v_loss
            wait       = 0
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best checkpoint before returning
    model.load_state_dict(best_state)

    return {
        "train_loss":       train_losses,
        "val_loss":         val_losses,
        "best_model_state": best_state,
        "best_epoch":       best_epoch,
    }


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    """
    Run model in eval mode over an entire DataLoader.

    Returns
    -------
    preds : (N, n_ages) standardised predictions (float32)
    """
    model.eval()
    model = model.to(device)
    preds = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def destandardise(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Invert per-age standardisation: log_mx = z * std + mean."""
    return z * (std[None, :] + 1e-8) + mean[None, :]
