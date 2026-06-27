import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mit Flash Attention.
    Verwendet torch.nn.functional.scaled_dot_product_attention
    das automatisch FlashAttention-2 Kernel auf Ampere+ GPUs nutzt.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.dropout = dropout  # als float speichern für SDPA

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Flash Attention — automatisch optimiert auf RTX GPUs
        # is_causal=True ersetzt die manuelle Causal Mask
        out = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # ersetzt unsere triu Mask komplett
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)