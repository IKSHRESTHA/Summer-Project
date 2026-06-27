"""LSTM with scaled dot-product temporal attention and LayerNorm.

Architecture
------------
1. LSTM encodes the (seq_len, n_ages) sequence → all hidden states.
2. Scaled dot-product attention (Vaswani et al., 2017) over the seq_len
   hidden states uses the last hidden state as query.
3. Context vector is layer-normalised, then projected to n_ages outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    """
    LSTM + temporal self-attention.

    Input  : (batch, seq_len, n_ages)
    Output : (batch, n_ages)
    """

    def __init__(self, n_ages: int = 100, hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=n_ages, hidden_size=hidden_size,
                              num_layers=1, batch_first=True)
        self.attn_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm   = nn.LayerNorm(hidden_size)
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, n_ages)
        self.scale  = hidden_size ** 0.5

    def forward(self, x):
        hidden, _ = self.lstm(x)                           # (B, T, hidden)
        q       = self.attn_q(hidden[:, -1:, :])           # (B, 1, hidden)
        scores  = torch.bmm(q, hidden.transpose(1, 2)) / self.scale  # (B, 1, T)
        weights = F.softmax(scores, dim=-1)                # (B, 1, T)
        context = torch.bmm(weights, hidden).squeeze(1)   # (B, hidden)
        context = self.norm(context)
        return self.linear(self.drop(context))             # (B, n_ages)
