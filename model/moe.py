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

    # Sparse Dispatch: jeden Experten NUR auf seinen zugewiesenen Token rechnen
    # (statt alle Experten auf alle Token). Spart top_k/n_experts der FF-FLOPs.
        output = torch.zeros_like(x_flat)

    # (token, slot) → flache Zuweisungslisten
        flat_expert = top_k_indices.reshape(-1)               # (N*top_k,)
        flat_prob   = top_k_probs.reshape(-1).unsqueeze(-1)   # (N*top_k, 1)
        token_idx   = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        for e, expert in enumerate(self.experts):
            sel = token_idx[flat_expert == e]                 # Token für Experte e
            # leeres sel ist ok (kein .any()-Check → kein GPU-Sync, compile-safe)
            expert_out = expert(x_flat[sel]) * flat_prob[flat_expert == e]
            output.index_add_(0, sel, expert_out.to(output.dtype))

    # Load balancing — Routing-Gewichte wie zuvor rekonstruieren
        routing_weights = torch.zeros(
            N, self.n_experts, device=x.device, dtype=router_probs.dtype)
        routing_weights.scatter_(1, top_k_indices, top_k_probs)
        fraction_prob   = router_probs.mean(dim=0)
        fraction_tokens = routing_weights.bool().float().mean(dim=0).detach()
        self.load_balancing_loss = (
        self.n_experts * (fraction_tokens * fraction_prob).sum()
        )

        return self.dropout(output).view(B, T, D)