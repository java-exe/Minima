import torch
import torch.nn as nn

from model.config import ModelConfig, ModelConfigSmall
from model.embedding import Embedding
from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Zwei lineare Schichten mit GELU Aktivierung dazwischen.

    Linear(d_model, d_ff) → GELU → Dropout → Linear(d_ff, d_model)

    Warum GELU statt ReLU?
    - Smoothere Aktivierung, bessere Gradienten
    - Wird von GPT-2/3 verwendet
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Ein einzelner Transformer Block mit Pre-LayerNorm.

    Pre-LayerNorm (stabiler als Post-LayerNorm):
        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Warum Pre-LayerNorm?
    - Gradienten fließen stabiler durch den Residual Stream
    - Kein Warmup nötig, Training stabiler
    - GPT-2 verwendet auch Pre-LayerNorm
    """
    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1     = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2     = nn.LayerNorm(d_model)
        self.ff        = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Residual Connection + Attention
        x = x + self.attention(self.norm1(x), mask)
        # Residual Connection + FeedForward
        x = x + self.ff(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """
    Kompletter Decoder-only Transformer (GPT-Architektur).

    Pipeline:
        Token-IDs → Embedding + Positional Encoding
                  → N × TransformerBlock
                  → Final LayerNorm
                  → Linear(d_model, vocab_size)  ← Logits, kein Softmax
                                                    (CrossEntropyLoss macht Softmax intern)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = Embedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            context_length=config.context_length,
            dropout=config.dropout,
        )

        # Causal Mask vorberechnen: (1, 1, T, T)
        # triu mit diagonal=1 → alles oberhalb der Hauptdiagonale = True = maskiert
        mask = torch.triu(
            torch.ones(config.context_length, config.context_length), diagonal=1
        ).bool()
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

        # Logit-Projektion: d_model → vocab_size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying: Embedding-Matrix = LM-Head-Matrix
        # Spart ~8M Parameter und verbessert oft die Performance
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # Gewichte initialisieren
        self._init_weights()

    def _init_weights(self):
        """
        GPT-2 Style Initialisierung:
        - Linear + Embedding: N(0, 0.02)
        - Residual Projektionen skaliert mit 1/sqrt(n_layers)
          damit der Residual Stream nicht zu groß wird
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Residual Projektionen skalieren
        for name, param in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("net.3.weight"):
                nn.init.normal_(param, mean=0.0,
                                std=0.02 / (2 * self.config.n_layers) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) Token-IDs
        → (B, T, vocab_size) Logits
        """
        B, T = x.shape
        assert T <= self.config.context_length, \
            f"Sequenz zu lang: {T} > {self.config.context_length}"

        # Causal Mask auf aktuelle Sequenzlänge zuschneiden
        mask = self.mask[:, :, :T, :T]

        # Embedding + Positional Encoding
        x = self.embedding(x)          # (B, T, d_model)

        # N Transformer Blöcke
        for block in self.blocks:
            x = block(x, mask)

        # Final LayerNorm
        x = self.norm(x)

        # Logits
        return self.lm_head(x)         # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive Texterzeugung mit Top-K Sampling.

        input_ids: (1, T) Prompt Token-IDs
        → (1, T + max_new_tokens) generierte Token-IDs
        """
        for _ in range(max_new_tokens):
            # Sequenz auf context_length kürzen falls nötig
            ids = input_ids[:, -self.config.context_length:]

            # Forward Pass
            logits = self(ids)                    # (1, T, vocab_size)
            logits = logits[:, -1, :]             # nur letzter Token: (1, vocab_size)
            logits = logits / temperature

            # Top-K: alle Logits außer den Top-K auf -inf setzen
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_val, float("-inf"))

            # Sampling
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

            input_ids = torch.cat([input_ids, next_id], dim=1)

            # Bei EOS Token aufhören
            if next_id.item() == 1:  # eos_token_id
                break

        return input_ids