"""Transformer-Block + GPT-Modell (Person 2, Woche 4).

Pre-LayerNorm-Variante (stabiler als Post-LN). Finale Linear-Schicht gibt
Logits aus (kein Softmax -> CrossEntropyLoss erwartet Logits).
Siehe plan.md, Phase 1, Woche 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedding import TokenEmbedding, PositionalEncoding
from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, config.context_length)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()  # (T, T)

        x = self.drop(self.pos_enc(self.tok_emb(idx)))
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.ln_f(x))    # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
