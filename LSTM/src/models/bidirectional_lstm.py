"""Bidirectional LSTM with LayerNorm for mortality forecasting."""

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """
    Single BiLSTM layer.  Forward and backward final hidden states are
    concatenated (via hn), normalised, then projected to n_ages outputs.

    hn[0] = forward direction final hidden  (has seen steps 0..T-1)
    hn[1] = backward direction final hidden (has seen steps T-1..0)

    LayerNorm is applied over the 2*hidden concatenated vector.

    Input  : (batch, seq_len, n_ages)
    Output : (batch, n_ages)
    """

    def __init__(self, n_ages: int = 100, hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=n_ages, hidden_size=hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.norm   = nn.LayerNorm(hidden_size * 2)
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * 2, n_ages)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)                         # hn: (2, batch, hidden)
        out = torch.cat([hn[0], hn[1]], dim=-1)           # (batch, 2*hidden)
        out = self.norm(out)
        return self.linear(self.drop(out))                 # (batch, n_ages)
