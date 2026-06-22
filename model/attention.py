import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V

    Q, K, V: (B, n_heads, T, d_k)
    mask:    (1, 1, T, T) causal mask — True bedeutet "ignorieren"
    """
    d_k = Q.size(-1)

    # QK^T / sqrt(d_k) → (B, n_heads, T, T)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Causal Mask: zukünftige Tokens auf -inf setzen
    # → nach Softmax werden sie zu 0 (kein Attention auf die Zukunft)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Softmax über letzte Dimension (über alle Keys)
    weights = torch.softmax(scores, dim=-1)

    # Gewichtete Summe der Values → (B, n_heads, T, d_k)
    return torch.matmul(weights, V)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention aus "Attention is All You Need".

    Idee: statt einer großen Attention, n_heads parallele Attentions
    auf kleineren Subräumen. Jeder Head kann andere Beziehungen lernen.

    d_k = d_model / n_heads  (pro Head)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model muss durch n_heads teilbar sein"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads

        # Projektionsmatrizen für Q, K, V und Output
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # 1) Lineare Projektion → Q, K, V
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2) In n_heads aufteilen: (B, T, d_model) → (B, n_heads, T, d_k)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 3) Scaled Dot-Product Attention pro Head
        out = scaled_dot_product_attention(Q, K, V, mask)  # (B, n_heads, T, d_k)

        # 4) Heads zusammenführen: (B, n_heads, T, d_k) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 5) Output Projektion
        out = self.W_o(out)
        out = self.dropout(out)

        return out