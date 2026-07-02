import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 n_experts: int = 8, top_k: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.router    = nn.Linear(d_model, n_experts, bias=False)
        self.experts   = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(n_experts)
        ])
        self.dropout = nn.Dropout(dropout)
        self.load_balancing_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        router_logits = self.router(x_flat)
        router_probs  = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

    # Dense Dispatch: alle Experten auf alle Token, dann mit den Routing-
    # Gewichten maskiert aufsummieren (nicht-Top-k Gewichte = 0). Auf kleinen
    # GPUs ohne Triton/torch.compile (z.B. RTX 3060, eager) MESSBAR schneller
    # als ein Sparse-Loop, dessen Boolean-Masking pro Experte einen GPU-Sync
    # auslöst (hier ~1.25x auf den ganzen Step). Ergebnis ist identisch.
        expert_outputs = torch.stack(
            [expert(x_flat) for expert in self.experts], dim=0)  # (E, N, D)

        routing_weights = torch.zeros(
            N, self.n_experts, device=x.device, dtype=x_flat.dtype)
        routing_weights.scatter_(1, top_k_indices, top_k_probs)
        output = (routing_weights.T.unsqueeze(-1) * expert_outputs).sum(dim=0)

    # Load balancing
        fraction_prob   = router_probs.mean(dim=0)
        fraction_tokens = routing_weights.bool().float().mean(dim=0).detach()
        self.load_balancing_loss = (
        self.n_experts * (fraction_tokens * fraction_prob).sum()
        )

        return self.dropout(output).view(B, T, D)