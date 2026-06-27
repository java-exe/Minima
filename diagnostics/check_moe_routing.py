"""
Misst die Experten-Auslastung eines MoE-Checkpoints.
Wenn ein paar Experten ~alle Tokens bekommen und der Rest ~0 → Expert-Collapse.

Usage:
    python diagnostics/check_moe_routing.py \
        --ckpt training/checkpoints/run_moe_001/ckpt_step_0048000.pt \
        --data data/processed/val.bin \
        --batches 20
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from model.config import ModelConfigMoE
from model.transformer import GPTModel
from model.moe import MoELayer
from training.dataset import CodeDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    type=str, required=True)
    p.add_argument("--data",    type=str, required=True)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfigMoE()
    model = GPTModel(cfg).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint geladen: step {ckpt.get('step', '?')}")

    # Pro Layer die kumulierte Token-Zahl pro Experte sammeln
    counts = {}  # layer_idx -> tensor(n_experts)

    def make_hook(idx):
        def hook(module, inp, out):
            x_flat = inp[0].reshape(-1, inp[0].shape[-1])
            logits = module.router(x_flat)
            probs  = torch.softmax(logits, dim=-1)
            _, top_idx = torch.topk(probs, module.top_k, dim=-1)
            c = torch.bincount(top_idx.reshape(-1),
                               minlength=module.n_experts).float().cpu()
            counts[idx] = counts.get(idx, 0) + c
        return hook

    handles = []
    for i, block in enumerate(model.blocks):
        if isinstance(block.ff, MoELayer):
            handles.append(block.ff.register_forward_hook(make_hook(i)))

    ds = CodeDataset(args.data, context_length=cfg.context_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    with torch.no_grad():
        for b, (input_ids, _) in enumerate(loader):
            if b >= args.batches:
                break
            model(input_ids.to(device))

    for h in handles:
        h.remove()

    print(f"\n{cfg.n_experts} Experten, top_k={cfg.top_k}, "
          f"ideal = {100/cfg.n_experts:.1f}% pro Experte\n")
    n_dead_total = 0
    for idx in sorted(counts):
        c = counts[idx]
        share = (c / c.sum() * 100)
        dead = int((share < 1.0).sum())
        n_dead_total += dead
        bars = " ".join(f"{s:4.1f}" for s in share.tolist())
        print(f"Layer {idx:2d} | {bars}  | tote Experten: {dead}")

    total_experts = cfg.n_layers * cfg.n_experts
    frac_dead = n_dead_total / total_experts
    print(f"\nTote Experten gesamt (Anteil <1%): {n_dead_total} / {total_experts} "
          f"({frac_dead*100:.1f}%)")
    if n_dead_total == 0:
        print("→ Kein Collapse. Gesund.")
    elif frac_dead < 0.10:
        print("→ Nur milde Imbalance (oft nur Layer 0). Resume mit "
              "Load-Balancing reaktiviert die toten Experten in wenigen tausend "
              "Steps. KEIN Scratch nötig.")
    else:
        print("→ Breiter Collapse über viele Layer. Resume hilft, aber "
              "Scratch-Restart erwägen.")


if __name__ == "__main__":
    main()
