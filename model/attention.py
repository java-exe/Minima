"""Multi-Head Attention + Causal Mask (Person 2, Woche 3).

Siehe plan.md, Phase 1, Woche 3.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape

        # Project and split into heads: (B, T, d_model) -> (B, n_heads, T, d_k)
        def split(w):
            return w.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = split(self.W_Q(x)), split(self.W_K(x)), split(self.W_V(x))

        out = scaled_dot_product_attention(Q, K, V, mask)          # (B, n_heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_k)
        return self.W_O(out)
