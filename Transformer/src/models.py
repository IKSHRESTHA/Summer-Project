"""
Model zoo for the Wang et al. (2024) replication.

Transformer hyperparameters follow the paper's Table 1 exactly:
  encoder layers = 1, decoder layers = 1, attention heads = 2,
  embedding dimension d_model = 32, feed-forward hidden = 16, dropout = 0.1,
  sinusoidal positional encoding, NO decoder masking (the paper omits the
  causal mask, following Wang et al. 2022).

Interpretation choices the paper leaves open (documented in the report):
  - decoder input: the same embedded source sequence (mask-free encoder-
    decoder self/cross attention over identical inputs)
  - output head: last-position decoder state -> Linear(d_model, A)

Baselines (RNN / LSTM / CNN): the paper gives no baseline configurations;
we use hidden width 32 to match d_model so parameter budgets are comparable.
CNN follows the paper's Section 2.2: 1-D convolution along the time axis with
age-vector filters, pooling, then a fully connected output layer.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):                              # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, n_ages: int = 101, d_model: int = 32, n_heads: int = 2,
                 ffn_hidden: int = 16, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(n_ages, d_model)
        self.pos   = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=ffn_hidden, dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(d_model, n_ages)

    def forward(self, x):                              # x: (B, T, A)
        e = self.pos(self.embed(x))                    # (B, T, d_model)
        out = self.transformer(e, e)                   # mask-free, tgt = src
        return self.head(out[:, -1])                   # (B, A)


class RNNModel(nn.Module):
    def __init__(self, n_ages: int = 101, hidden: int = 32):
        super().__init__()
        self.rnn  = nn.RNN(n_ages, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, n_ages)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class LSTMModel(nn.Module):
    def __init__(self, n_ages: int = 101, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_ages, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, n_ages)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class CNNModel(nn.Module):
    """1-D convolution along time with age-vector filters (paper Section 2.2)."""

    def __init__(self, n_ages: int = 101, n_filters: int = 32, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(n_ages, n_filters, kernel_size=kernel)
        self.act  = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(n_filters, n_ages)

    def forward(self, x):                              # x: (B, T, A)
        c = self.act(self.conv(x.permute(0, 2, 1)))    # (B, K, T-k+1)
        return self.head(self.pool(c).squeeze(-1))     # (B, A)


MODEL_CLASSES = {
    "RNN":         RNNModel,
    "LSTM":        LSTMModel,
    "CNN":         CNNModel,
    "Transformer": TransformerModel,
}
