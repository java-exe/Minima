"""Transformer-Block + GPT-Modell (Person 2, Woche 4).

Pre-LayerNorm-Variante (stabiler als Post-LN). Finale Linear-Schicht gibt
Logits aus (kein Softmax -> CrossEntropyLoss erwartet Logits).
Siehe plan.md, Phase 1, Woche 4.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model) -> Dropout
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        # x = x + Attention(LayerNorm(x))
        # x = x + FeedForward(LayerNorm(x))
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # token_embedding + positional_encoding
        # N x TransformerBlock
        # Final LayerNorm
        # Linear(d_model, vocab_size)
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        raise NotImplementedError
