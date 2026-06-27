import torch
import torch.nn as nn
import torch.utils.checkpoint

from model.config import ModelConfig, ModelConfigSmall, ModelConfigMoE
from model.embedding import Embedding
from model.attention import MultiHeadAttention
from model.moe import MoELayer


class FeedForward(nn.Module):
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
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, use_moe: bool = False,
                 n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.norm1     = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2     = nn.LayerNorm(d_model)

        if use_moe:
            self.ff = MoELayer(d_model, d_ff, n_experts, top_k, dropout)
        else:
            self.ff = FeedForward(d_model, d_ff, dropout)

        self.use_moe = use_moe

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layer_drop_rate = 0.0

        use_moe   = hasattr(config, "n_experts")
        n_experts = getattr(config, "n_experts", 8)
        top_k     = getattr(config, "top_k", 2)

        self.embedding = Embedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            context_length=config.context_length,
            dropout=config.dropout,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                use_moe=use_moe,
                n_experts=n_experts,
                top_k=top_k,
            )
            for _ in range(config.n_layers)
        ])

        self.norm    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        for name, param in self.named_parameters():
            # net.3 = dense FF Output-Proj, net.2 = MoE-Expert Output-Proj
            if name.endswith("W_o.weight") or name.endswith("net.3.weight") \
               or name.endswith("net.2.weight"):
                nn.init.normal_(param, mean=0.0,
                                std=0.02 / (2 * self.config.n_layers) ** 0.5)

    def get_load_balancing_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            if isinstance(block.ff, MoELayer):
                loss = loss + block.ff.load_balancing_loss
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.config.context_length

        x = self.embedding(x)

        for block in self.blocks:
            # Layer Dropping — in früher Trainingsphase Layer zufällig überspringen
            if self.training and self.layer_drop_rate > 0:
                if torch.rand(1) < self.layer_drop_rate:
                    continue

            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, None, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200,
                 temperature=0.8, top_k=50, top_p=0.0,
                 repetition_penalty=1.3):
        for _ in range(max_new_tokens):
            ids    = input_ids[:, -self.config.context_length:]
            logits = self(ids)
            logits = logits[:, -1, :]

            # Repetition Penalty — bereits generierte Tokens abschwächen,
            # verhindert Degeneration ("int, int, int, ...").
            if repetition_penalty != 1.0:
                for tok_id in set(input_ids[0].tolist()):
                    if logits[0, tok_id] > 0:
                        logits[0, tok_id] /= repetition_penalty
                    else:
                        logits[0, tok_id] *= repetition_penalty

            logits = logits / temperature

            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(
                    logits < values[:, -1:], float("-inf"))

            # Nucleus (Top-p) Sampling — optional, oft besser als Top-k
            if top_p > 0.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0]  = False
                logits[0, sorted_idx[0, remove[0]]] = float("-inf")

            probs     = torch.softmax(logits, dim=-1)
            next_id   = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if next_id.item() == 1:
                break

        return input_ids