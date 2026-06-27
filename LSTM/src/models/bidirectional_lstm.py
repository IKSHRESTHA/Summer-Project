"""Bidirectional LSTM with LayerNorm for mortality forecasting."""

import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """
    Single BiLSTM layer.  Forward and backward hidden states from the last
    time step are concatenated, normalised, then projected to n_ages outputs.

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
        out, _ = self.lstm(x)                    # (batch, seq_len, 2*hidden)
        out    = self.norm(out[:, -1, :])         # last step + LayerNorm
        return self.linear(self.drop(out))        # (batch, n_ages)
