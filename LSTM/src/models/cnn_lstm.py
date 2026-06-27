"""CNN-LSTM hybrid for mortality forecasting with LayerNorm.

Architecture
------------
1. Two Conv1d layers applied across the AGE dimension at each time step to
   extract cross-age structural features.  Padding is symmetric (not causal)
   because ages are a spatial, not temporal, axis — causality applies only
   to the time dimension, which is handled by the LSTM.
2. Global average pooling collapses the age axis → (B*T, cnn_channels).
3. LSTM processes the resulting (seq_len, cnn_channels) sequence.
4. LayerNorm + Dropout + linear head produce (batch, n_ages) output.
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN extracts age-structure features per time step; LSTM models temporal dynamics.

    Input  : (batch, seq_len, n_ages)
    Output : (batch, n_ages)
    """

    def __init__(self, n_ages: int = 100, cnn_channels: int = 64,
                 hidden_size: int = 128, dropout: float = 0.2,
                 kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2   # same-length output; non-causal (age axis)
        self.conv = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.lstm   = nn.LSTM(input_size=cnn_channels, hidden_size=hidden_size,
                              num_layers=1, batch_first=True)
        self.norm   = nn.LayerNorm(hidden_size)
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, n_ages)

    def forward(self, x):
        B, T, A = x.shape
        x_flat = x.reshape(B * T, 1, A)             # (B*T, 1, n_ages)
        feat   = self.conv(x_flat)                   # (B*T, cnn_channels, n_ages)
        feat   = feat.mean(dim=-1)                   # global avg pool → (B*T, cnn_ch)
        feat   = feat.reshape(B, T, -1)              # (B, T, cnn_channels)
        out, _ = self.lstm(feat)                     # (B, T, hidden)
        out    = self.norm(out[:, -1, :])            # LayerNorm on last step
        return self.linear(self.drop(out))           # (B, n_ages)
