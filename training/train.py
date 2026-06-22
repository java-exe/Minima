import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

import wandb

from training.dataset import CodeDataset
from training.scheduler import get_lr
from model.config import ModelConfig, ModelConfigSmall
from model.transformer import GPTModel


# ─────────────────────────────────────────────
# Argparse
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SLM Training Loop")
    p.add_argument("--config",             type=str,   default="large",
                   choices=["large", "small"])
    p.add_argument("--data",               type=str,   required=True)
    p.add_argument("--val_data",           type=str,   default=None)
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--accumulation_steps", type=int,   default=16)
    p.add_argument("--steps",              type=int,   default=100_000)
    p.add_argument("--warmup_steps",       type=int,   default=500)
    p.add_argument("--max_lr",             type=float, default=3e-4)
    p.add_argument("--min_lr",             type=float, default=3e-5)
    p.add_argument("--checkpoint_dir",     type=str,   default="training/checkpoints/run_001")
    p.add_argument("--checkpoint_every",   type=int,   default=2000)
    p.add_argument("--val_every",          type=int,   default=500)
    p.add_argument("--val_steps",          type=int,   default=50)
    p.add_argument("--wandb_project",      type=str,   default="slm-project")
    p.add_argument("--wandb_run",          type=str,   default=None)
    p.add_argument("--no_wandb",           action="store_true")
    p.add_argument("--resume",             type=str,   default=None,
                   help="Checkpoint Pfad um Training fortzusetzen")
    return p.parse_args()


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

@torch.no_grad()
def estimate_val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    val_steps: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for i, (input_ids, target_ids) in enumerate(val_loader):
        if i >= val_steps:
            break
        input_ids  = input_ids.to(device)
        target_ids = target_ids.to(device)

        with autocast("cuda"):
            logits = model(input_ids)
            loss = compute_loss(logits, target_ids)

        total_loss += loss.item()

    model.train()
    return total_loss / min(val_steps, i + 1)


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
    )


# ─────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    step: int,
    loss: float,
    checkpoint_dir: str,
) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_step_{step:07d}.pt")
    torch.save({
        "step":                 step,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "loss":                 loss,
    }, path)
    print(f"  [ckpt] gespeichert: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Config ──
    cfg = ModelConfig() if args.config == "large" else ModelConfigSmall()
    print(f"Config: {args.config} | d_model={cfg.d_model} | layers={cfg.n_layers}")

    # ── Modell ──
    model = GPTModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameter: {n_params:.1f}M")

    # ── Dataset & DataLoader ──
    train_ds = CodeDataset(args.data, context_length=cfg.context_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = None
    if args.val_data:
        val_ds = CodeDataset(args.val_data, context_length=cfg.context_length)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # ── Mixed Precision ──
    scaler = GradScaler("cuda")

    # ── Resume von Checkpoint ──
    start_step = 0
    if args.resume:
        print(f"Lade Checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"]
        print(f"  Fortgesetzt ab Step {start_step}")

    # ── W&B ──
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )
        wandb.watch(model, log_freq=500)

    # ── Training Loop ──
    model.train()
    step = start_step
    optimizer.zero_grad()

    def infinite_loader(loader):
        while True:
            yield from loader

    data_iter = infinite_loader(train_loader)
    t0 = time.time()

    total_steps = start_step + args.steps
    print(f"\nTraining startet | {args.steps} Steps | "
          f"effektiver Batch={args.batch_size * args.accumulation_steps}")

    while step < total_steps:

        # ── LR Update ──
        lr = get_lr(
            step,
            warmup_steps=args.warmup_steps,
            max_steps=total_steps,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Gradient Accumulation ──
        total_loss = 0.0
        for micro_step in range(args.accumulation_steps):
            input_ids, target_ids = next(data_iter)
            input_ids  = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            with autocast("cuda"):
                logits = model(input_ids)
                loss = compute_loss(logits, target_ids) / args.accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item()

        # ── Gradient Clipping ──
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step += 1

        # ── Logging ──
        if step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"step {step:6d} | loss {total_loss:.4f} | "
                  f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | {dt*1000/10:.0f}ms/step")

            if use_wandb:
                wandb.log({
                    "train/loss":      total_loss,
                    "train/lr":        lr,
                    "train/grad_norm": grad_norm,
                    "train/step":      step,
                })

        # ── Validation ──
        if val_loader and step % args.val_every == 0:
            val_loss = estimate_val_loss(model, val_loader, args.val_steps, device)
            print(f"  [val] step {step} | val_loss {val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss, "train/step": step})

        # ── Checkpoint ──
        if step % args.checkpoint_every == 0:
            save_checkpoint(model, optimizer, scaler, step, total_loss, args.checkpoint_dir)

    # Finaler Checkpoint
    save_checkpoint(model, optimizer, scaler, step, total_loss, args.checkpoint_dir)
    print("\nTraining fertig.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()