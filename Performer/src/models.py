"""
Model zoo for the Performer mortality-forecasting study.

Four models are compared, all on identical data, splits, and metrics:
  - Lee-Carter (src/lee_carter.py)          : classical benchmark
  - LSTM (here)                             : strongest neural baseline in our
                                              companion replication study
  - Transformer (here)                      : exact softmax attention
  - Performer (src/performer.py)            : FAVOR+ linear attention

The Transformer follows the configuration of Wang et al. (2024): 1 encoder +
1 decoder layer, 2 heads, d_model=32, feed-forward 16, dropout 0.1, sinusoidal
positional encoding, no decoder masking, decoder input = encoder input.
The LSTM uses a single layer of width 32 so parameter budgets stay comparable.
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


class LSTMModel(nn.Module):
    def __init__(self, n_ages: int = 101, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_ages, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, n_ages)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


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


# Performer is registered by run_experiment.py (avoids a circular import,
# since src/performer.py imports PositionalEncoding from this module).
MODEL_CLASSES = {
    "LSTM":        LSTMModel,
    "Transformer": TransformerModel,
}
