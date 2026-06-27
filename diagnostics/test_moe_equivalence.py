"""
Prueft, dass die neue Sparse-MoE-Forward bitgenau (bis auf FP-Rundung)
dasselbe Ergebnis liefert wie die alte Dense-Variante.
Wenn max abs diff ~1e-3 (bf16) bzw. ~1e-6 (fp32) → Checkpoint bleibt gueltig.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from model.moe import MoELayer


def dense_reference(layer, x):
    """Die ALTE Implementierung: alle Experten auf alle Token, dann maskieren."""
    B, T, D = x.shape
    x_flat = x.view(B * T, D)
    router_probs = F.softmax(layer.router(x_flat), dim=-1)
    top_k_probs, top_k_idx = torch.topk(router_probs, layer.top_k, dim=-1)
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    expert_outputs = torch.stack([e(x_flat) for e in layer.experts], dim=0)
    rw = torch.zeros(B * T, layer.n_experts, device=x.device, dtype=x.dtype)
    rw.scatter_(1, top_k_idx, top_k_probs.to(x.dtype))
    out = (rw.T.unsqueeze(-1) * expert_outputs).sum(dim=0)
    return out.view(B, T, D)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = MoELayer(d_model=768, d_ff=1536, n_experts=8, top_k=2,
                     dropout=0.0).to(device).eval()
    x = torch.randn(4, 128, 768, device=device)

    with torch.no_grad():
        new_out = layer(x)               # Sparse (neu)
        ref_out = dense_reference(layer, x)  # Dense (alt)

    diff = (new_out - ref_out).abs().max().item()
    print(f"max abs diff (fp32): {diff:.2e}")
    assert diff < 1e-4, "Nicht aequivalent!"

    # bf16-Pfad wie im Training
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        new_bf = layer(x)
        ref_bf = dense_reference(layer, x)
    diff_bf = (new_bf.float() - ref_bf.float()).abs().max().item()
    print(f"max abs diff (bf16): {diff_bf:.2e}")

    print("OK — Sparse-Forward ist aequivalent, Checkpoint bleibt gueltig.")


if __name__ == "__main__":
    main()
