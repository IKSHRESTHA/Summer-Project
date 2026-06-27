"""Stacked (deep) LSTM with residual connections and LayerNorm."""

import torch
import torch.nn as nn


class _ResidualLSTMBlock(nn.Module):
    """
    Single LSTM layer wrapped with a residual skip connection and LayerNorm.

    Residual connection: output = LayerNorm(LSTM(x) + proj(x))
    A linear projection aligns dimensions when hidden_size != input_size.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.proj = (nn.Linear(input_size, hidden_size, bias=False)
                     if input_size != hidden_size else nn.Identity())

    def forward(self, x):
        out, _ = self.lstm(x)                   # (batch, seq_len, hidden)
        out    = self.norm(out + self.proj(x))   # residual + LayerNorm
        return self.drop(out)


class StackedLSTM(nn.Module):
    """
    Three stacked LSTM blocks, each with a residual skip connection and LayerNorm.

    Residual connections (He et al., 2016) prevent gradient degradation and
    allow useful training beyond 2 LSTM layers.

    Input  : (batch, seq_len, n_ages)
    Output : (batch, n_ages)
    """

    def __init__(self, n_ages: int = 100, hidden_size: int = 128,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_size = n_ages if i == 0 else hidden_size
            layers.append(_ResidualLSTMBlock(in_size, hidden_size, dropout))
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(hidden_size, n_ages)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)          # (batch, seq_len, hidden)
        return self.linear(x[:, -1, :])   # last time step → (batch, n_ages)
