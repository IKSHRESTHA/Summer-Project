"""
Training loop for the replication.

Follows the paper's stated protocol: MSE loss, learning rate 0.001,
batch size 32, 100 epochs, no early stopping (none is mentioned).
Optimizer is Adam (assumption --- the paper does not name one).
"""

import numpy as np
import torch
import torch.nn as nn


def train_model(model, loader, epochs: int = 100, lr: float = 1e-3,
                device: str = "cpu") -> list:
    """Train in place; returns the per-epoch training MSE curve."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    curve = []
    for _ in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n   += len(xb)
        curve.append(tot / max(n, 1))
    return curve
