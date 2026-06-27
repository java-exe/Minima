"""
Kill-or-Confirm-Experiment für MoErge (Expert Migration).

Forkt einen Checkpoint in N simulierte Worker, trainiert jeden auf einem
EIGENEN Datenshard für ein paar hundert Steps, und vergleicht dann drei
Merge-Strategien per Val-Loss:

  1. no-sync   : eine Insel allein (Baseline, kein Merge)
  2. naive-avg : ALLE Parameter gemittelt (das, was bei MoE kaputtgeht)
  3. migration : unser Verfahren (Experten+Router-Zeile wandern, Backbone gemittelt)

Zusätzlich: tote Experten vor/nach Migration zählen (Revival-Nachweis).
Speicherschonend: immer nur EIN Modell gleichzeitig auf der GPU.

Usage:
    python merging/run_probe.py \
        --ckpt training/checkpoints/run_moe_001/ckpt_step_0048000.pt \
        --train data/processed/train.bin --val data/processed/val.bin \
        --n_workers 2 --local_steps 300
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

from model.config import ModelConfigMoE
from model.transformer import GPTModel
from training.dataset import CodeDataset
from merging import (
    measure_expert_utilization, merge_round, moe_layers,
)
from merging.device import enable_utf8_stdout

enable_utf8_stdout()


def train_worker(model, loader, steps, lr, device, lb_weight):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=(device.type == "cuda"))
    it = iter(loader)
    for s in range(steps):
        try:
            input_ids, target_ids = next(it)
        except StopIteration:
            it = iter(loader)
            input_ids, target_ids = next(it)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B * T, V),
                                               target_ids.view(B * T))
            if lb_weight > 0:
                loss = loss + model.get_load_balancing_loss() * lb_weight
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if (s + 1) % 50 == 0:
            print(f"    step {s+1}/{steps} | loss {loss.item():.3f}")


@torch.no_grad()
def evaluate(model, loader, n_batches, device):
    model.eval()
    total, n = 0.0, 0
    for b, (input_ids, target_ids) in enumerate(loader):
        if b >= n_batches:
            break
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B * T, V),
                                               target_ids.view(B * T))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def naive_average(states):
    avg = copy.deepcopy(states[0])
    for k in avg:
        avg[k] = (sum(s[k].float() for s in states) / len(states)).to(states[0][k].dtype)
    return avg


def count_dead(util, threshold=0.02):
    return {L: int((u < threshold).sum()) for L, u in util.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        type=str, required=True)
    p.add_argument("--train",       type=str, required=True)
    p.add_argument("--val",         type=str, required=True)
    p.add_argument("--n_workers",   type=int, default=2)
    p.add_argument("--local_steps", type=int, default=300)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--probe_batches", type=int, default=20)
    p.add_argument("--eval_batches",  type=int, default=50)
    p.add_argument("--dead_threshold", type=float, default=0.02)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfigMoE()
    lb_weight = getattr(cfg, "load_balance_weight", 0.0)

    base_sd = torch.load(args.ckpt, map_location="cpu")["model_state_dict"]
    train_ds = CodeDataset(args.train, context_length=cfg.context_length)
    val_ds   = CodeDataset(args.val,   context_length=cfg.context_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    N = args.n_workers
    shard = len(train_ds) // N
    print(f"\n{N} Worker | je {shard:,} Samples | {args.local_steps} lokale Steps\n")

    worker_states, worker_utils = [], []
    for i in range(N):
        print(f"[Worker {i}] trainiert auf Shard {i} ...")
        model = GPTModel(cfg).to(device)
        model.load_state_dict(base_sd)
        idx = list(range(i * shard, (i + 1) * shard))
        loader = DataLoader(Subset(train_ds, idx),
                            batch_size=args.batch_size, shuffle=True)
        train_worker(model, loader, args.local_steps, args.lr, device, lb_weight)
        util = measure_expert_utilization(model, val_loader,
                                          args.probe_batches, device)
        worker_states.append({k: v.detach().cpu()
                              for k, v in model.state_dict().items()})
        worker_utils.append(util)
        model.to("cpu"); del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def eval_sd(sd, tag):
        m = GPTModel(cfg).to(device)
        m.load_state_dict(sd)
        loss = evaluate(m, val_loader, args.eval_batches, device)
        m.to("cpu"); del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {tag:<24} val_loss = {loss:.4f}")
        return loss

    print("\n── Vergleich der Merge-Strategien ──")
    l_nosync = eval_sd(worker_states[0], "1. no-sync (Insel 0)")
    l_naive  = eval_sd(naive_average(worker_states), "2. naive average")

    print("\n  Migration (Experten+Router wandern):")
    merged = merge_round(worker_states, worker_utils,
                         dead_threshold=args.dead_threshold, verbose=True)
    l_merge = eval_sd(merged[0], "3. migration (Insel 0)")

    # Dead-Expert-Revival auf Insel 0
    m = GPTModel(cfg).to(device); m.load_state_dict(merged[0])
    util_after = measure_expert_utilization(m, val_loader,
                                            args.probe_batches, device)
    m.to("cpu"); del m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dead_before = count_dead(worker_utils[0], args.dead_threshold)
    dead_after  = count_dead(util_after, args.dead_threshold)
    n_before = sum(dead_before.values())
    n_after  = sum(dead_after.values())

    print("\n── Ergebnis ──")
    print(f"  no-sync   : {l_nosync:.4f}")
    print(f"  naive-avg : {l_naive:.4f}   (Δ vs no-sync {l_naive - l_nosync:+.4f})")
    print(f"  migration : {l_merge:.4f}   (Δ vs no-sync {l_merge - l_nosync:+.4f})")
    print(f"\n  tote Experten Insel 0:  vor {n_before}  →  nach {n_after}")
    print("\n  Hypothese bestätigt, wenn migration <= no-sync < naive "
          "UND tote Experten sinken.")


if __name__ == "__main__":
    main()
