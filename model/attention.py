"""Multi-Head Attention + Causal Mask (Person 2, Woche 3).

Siehe plan.md, Phase 1, Woche 3.
"""

import torch
import torch.nn as nn


def scaled_dot_product_attention(Q, K, V, mask=None):
    # scores = QK^T / sqrt(d_k)
    # scores = scores.masked_fill(mask, -inf)   # Causal mask
    # weights = softmax(scores)
    # return weights @ V
    raise NotImplementedError


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # Linear W_Q, W_K, W_V, W_O
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # split in heads -> attention -> concat -> project
        raise NotImplementedError
