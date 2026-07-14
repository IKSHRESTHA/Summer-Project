"""
Performer (Choromanski et al., ICLR 2021) for mortality forecasting.

Replaces the Transformer's softmax attention with FAVOR+ (Fast Attention
Via positive Orthogonal Random features). The softmax kernel

    SM(q, k) = exp(q . k / sqrt(d))

is approximated by the inner product of positive random features

    phi(u) = exp(w . u - ||u||^2 / 2) / sqrt(m),   w ~ N(0, I_d),

applied to q' = q / d^(1/4), k' = k / d^(1/4), so that
E[phi(q') . phi(k')] = SM(q, k).  Attention is then computed as

    Attn = D^{-1} ( phi(Q) ( phi(K)^T V ) ),   D = diag(phi(Q) phi(K)^T 1),

in O(T m d) time instead of O(T^2 d).

Best-practice details implemented here (following the reference
google-research/performer implementation):
  - ORTHOGONAL random feature blocks (QR of Gaussian), rows rescaled by
    chi-distributed norms so marginals match unstructured Gaussians;
  - numerical stabilisation inside the exponential: per-query max is
    subtracted for phi(Q) and a global max for phi(K) (the constants
    cancel in the normalised attention ratio);
  - random features drawn once at initialisation (registered buffer):
    deterministic per seed, stored in checkpoints, identical across the
    10 replication runs' seeding scheme.

Architecture parity: PerformerModel mirrors models.TransformerModel
EXACTLY (embedding, sinusoidal positional encoding, 1 encoder + 1 decoder
layer with post-LayerNorm residuals, 2 heads, d_model=32, FFN width 16,
dropout 0.1, decoder input = encoder input, no masks, last-position linear
head), with only the attention operator swapped for FAVOR+. Any accuracy
difference is therefore attributable to the attention mechanism.
"""

import math
import torch
import torch.nn as nn

from src.models import PositionalEncoding


# ── FAVOR+ machinery ──────────────────────────────────────────────────────────

def _orthogonal_random_features(m: int, d: int, generator=None) -> torch.Tensor:
    """
    (m, d) random feature matrix: stacked orthogonal blocks with
    chi-distributed row norms (Performer Sec. 3 / Lemma 1).
    """
    blocks = []
    n_full, rem = divmod(m, d)
    for _ in range(n_full + (rem > 0)):
        g = torch.randn(d, d, generator=generator)
        q, _ = torch.linalg.qr(g)
        blocks.append(q.T)
    W = torch.cat(blocks, dim=0)[:m]            # (m, d), orthonormal rows
    norms = torch.randn(m, d, generator=generator).norm(dim=1)   # chi_d norms
    return W * norms[:, None]


def _favor_features(x: torch.Tensor, W: torch.Tensor, is_query: bool,
                    eps: float = 1e-6) -> torch.Tensor:
    """
    Positive random features phi(x) for the softmax kernel.
    x : (B, h, T, d_head)   W : (m, d_head)
    returns (B, h, T, m)
    """
    d = x.shape[-1]
    m = W.shape[0]
    x = x / (d ** 0.25)                          # fold in the 1/sqrt(d) of SM
    u = x @ W.T                                  # (B, h, T, m)
    sq = 0.5 * (x ** 2).sum(dim=-1, keepdim=True)  # ||x'||^2 / 2
    logits = u - sq
    if is_query:
        stab = logits.max(dim=-1, keepdim=True).values      # per position
    else:
        stab = logits.amax(dim=(-2, -1), keepdim=True)      # global
    return torch.exp(logits - stab) / math.sqrt(m) + eps


class FAVORAttention(nn.Module):
    """Multi-head attention with the softmax kernel approximated by FAVOR+."""

    def __init__(self, d_model: int = 32, n_heads: int = 2,
                 m_features: int = 48, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        W = _orthogonal_random_features(m_features, self.dh)
        self.register_buffer("W", W)             # fixed per init; in state_dict

    def _split(self, x):                          # (B, T, d) -> (B, h, T, dh)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def forward(self, q_in, k_in, v_in):
        q = self._split(self.q_proj(q_in))
        k = self._split(self.k_proj(k_in))
        v = self._split(self.v_proj(v_in))

        qp = _favor_features(q, self.W, is_query=True)     # (B, h, T, m)
        kp = _favor_features(k, self.W, is_query=False)    # (B, h, T, m)

        kv  = kp.transpose(-2, -1) @ v                     # (B, h, m, dh)
        num = qp @ kv                                      # (B, h, T, dh)
        den = qp @ kp.sum(dim=-2, keepdim=True).transpose(-2, -1)  # (B,h,T,1)
        out = num / den

        B, _, T, _ = out.shape
        out = out.transpose(1, 2).reshape(B, T, self.h * self.dh)
        return self.drop(self.o_proj(out))


# ── Performer encoder / decoder layers (post-LN, as in nn.Transformer) ───────

class _FFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PerformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, m_features, dropout):
        super().__init__()
        self.attn = FAVORAttention(d_model, n_heads, m_features, dropout)
        self.ffn  = _FFN(d_model, ffn_hidden, dropout)
        self.n1   = nn.LayerNorm(d_model)
        self.n2   = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.n1(x + self.attn(x, x, x))
        return self.n2(x + self.ffn(x))


class PerformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, m_features, dropout):
        super().__init__()
        self.self_attn  = FAVORAttention(d_model, n_heads, m_features, dropout)
        self.cross_attn = FAVORAttention(d_model, n_heads, m_features, dropout)
        self.ffn = _FFN(d_model, ffn_hidden, dropout)
        self.n1  = nn.LayerNorm(d_model)
        self.n2  = nn.LayerNorm(d_model)
        self.n3  = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        x = self.n1(x + self.self_attn(x, x, x))
        x = self.n2(x + self.cross_attn(x, memory, memory))
        return self.n3(x + self.ffn(x))


class PerformerModel(nn.Module):
    """Drop-in Performer twin of models.TransformerModel."""

    def __init__(self, n_ages: int = 101, d_model: int = 32, n_heads: int = 2,
                 ffn_hidden: int = 16, m_features: int = 48,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(n_ages, d_model)
        self.pos   = PositionalEncoding(d_model)
        self.encoder = PerformerEncoderLayer(d_model, n_heads, ffn_hidden,
                                             m_features, dropout)
        self.decoder = PerformerDecoderLayer(d_model, n_heads, ffn_hidden,
                                             m_features, dropout)
        # nn.Transformer applies a final LayerNorm after the encoder and the
        # decoder stacks; mirrored here for strict architectural parity
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_ages)

    def forward(self, x):                          # x: (B, T, A)
        e = self.pos(self.embed(x))                # (B, T, d_model)
        mem = self.enc_norm(self.encoder(e))
        out = self.dec_norm(self.decoder(e, mem))  # decoder input = encoder input
        return self.head(out[:, -1])               # (B, A)
