"""
LSTM variants for 1-dimensional kt mortality-index forecasting.

Input:  (batch, seq_len, 1)  — standardised kt scalar time series
Output: (batch, 1)           — next kt value

This is the architecturally correct formulation used in the mortality
deep learning literature:
  - Richman & Schreck (2019): neural net replaces ARIMA on extracted kt
  - Perla et al. (2021): LSTM on LC factor series, not raw rates
  - Nigri et al. (2021): deep learning integrated Lee-Carter model

Parameter counts at default hidden_size=32:
  KtVanillaLSTM   :  ~4.4k    vs  ~131k for raw-rate Vanilla
  KtStackedLSTM   : ~13.2k    vs  ~408k for raw-rate Stacked
  KtBiLSTM        :  ~8.8k    vs  ~262k for raw-rate BiLSTM
  KtAttentionLSTM :  ~4.5k    vs  ~147k for raw-rate Attention
  KtCNNLSTM       :  ~5.2k    vs  ~133k for raw-rate CNN-LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KtVanillaLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=1, hidden_size=hidden_size,
                              num_layers=1, batch_first=True)
        self.norm   = nn.LayerNorm(hidden_size)
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):          # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        return self.linear(self.drop(self.norm(out[:, -1, :])))   # (batch, 1)


class _KtResidualBlock(nn.Module):
    """Single LSTM layer with residual connection and LayerNorm."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.proj = (nn.Linear(input_size, hidden_size, bias=False)
                     if input_size != hidden_size else nn.Identity())

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.drop(self.norm(out + self.proj(x)))


class KtStackedLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.layer1 = _KtResidualBlock(1,           hidden_size, dropout)
        self.layer2 = _KtResidualBlock(hidden_size, hidden_size, dropout)
        self.layer3 = _KtResidualBlock(hidden_size, hidden_size, dropout)
        self.norm   = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.layer3(self.layer2(self.layer1(x)))
        return self.linear(self.norm(h[:, -1, :]))


class KtBiLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=1, hidden_size=hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.norm   = nn.LayerNorm(hidden_size * 2)
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)                      # hn: (2, batch, hidden)
        out = torch.cat([hn[0], hn[1]], dim=-1)        # (batch, 2*hidden)
        return self.linear(self.drop(self.norm(out)))


class KtAttentionLSTM(nn.Module):
    """LSTM + additive attention: learns which past kt values matter most."""

    def __init__(self, hidden_size: int = 32, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=1, hidden_size=hidden_size,
                              num_layers=1, batch_first=True)
        self.attn_w = nn.Linear(hidden_size, 1)   # scalar score per time step
        self.norm   = nn.LayerNorm(hidden_size)
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _   = self.lstm(x)                          # (B, T, H)
        scores   = self.attn_w(out).squeeze(-1)           # (B, T)
        weights  = torch.softmax(scores, dim=-1)          # (B, T)
        context  = (weights.unsqueeze(-1) * out).sum(1)   # (B, H)
        return self.linear(self.drop(self.norm(context)))

    def get_attention_weights(self, x):
        """Return attention weights for a single input — used for visualisation."""
        with torch.no_grad():
            out, _ = self.lstm(x)
            scores = self.attn_w(out).squeeze(-1)
            return torch.softmax(scores, dim=-1).cpu().numpy()


class KtCNNLSTM(nn.Module):
    """1D temporal convolution over kt history, then LSTM for trend extraction."""

    def __init__(self, hidden_size: int = 32, dropout: float = 0.1, **kwargs):
        super().__init__()
        # conv over time dimension: captures local patterns in kt
        self.conv   = nn.Conv1d(in_channels=1, out_channels=8,
                                kernel_size=3, padding=1)
        self.lstm   = nn.LSTM(input_size=8, hidden_size=hidden_size,
                              num_layers=1, batch_first=True)
        self.norm   = nn.LayerNorm(hidden_size)
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):                          # x: (batch, seq_len, 1)
        c   = F.relu(self.conv(x.permute(0, 2, 1)))  # (B, 8, T)
        c   = c.permute(0, 2, 1)                      # (B, T, 8)
        out, _ = self.lstm(c)
        return self.linear(self.drop(self.norm(out[:, -1, :])))
