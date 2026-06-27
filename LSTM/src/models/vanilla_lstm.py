"""Vanilla single-layer LSTM with LayerNorm for mortality forecasting."""

import torch.nn as nn


class VanillaLSTM(nn.Module):
    """
    Single-layer LSTM + LayerNorm + Dropout → linear head.

    LayerNorm is applied to LSTM outputs before dropout (Ba et al., 2016).
    Training is more stable than without normalisation on small mortality datasets.

    Input  : (batch, seq_len, n_ages)
    Output : (batch, n_ages)
    """

    def __init__(self, n_ages: int = 100, hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=n_ages, hidden_size=hidden_size,
                              num_layers=1, batch_first=True)
        self.norm   = nn.LayerNorm(hidden_size)
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, n_ages)

    def forward(self, x):
        out, _ = self.lstm(x)                   # (batch, seq_len, hidden)
        out    = self.norm(out[:, -1, :])        # last time step + LayerNorm
        return self.linear(self.drop(out))       # (batch, n_ages)
