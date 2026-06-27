import argparse
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
from model.config import ModelConfig, ModelConfigSmall, ModelConfigMoE
from model.transformer import GPTModel
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

def parse_args():
    p = argparse.ArgumentParser(description="SLM Training Loop")
    p.add_argument("--config",             type=str,   default="large",
                   choices=["large", "small", "moe"])
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
    p.add_argument("--resume",             type=str,   default=None)
    p.add_argument("--compile",            action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--selective_backprop", action="store_true")
    p.add_argument("--layer_drop",         action="store_true")
    return p.parse_args()


@torch.no_grad()
def estimate_val_loss(model, val_loader, val_steps, device, cfg):
    model.eval()
    total_loss = 0.0
    for i, (input_ids, target_ids) in enumerate(val_loader):
        if i >= val_steps:
            break
        input_ids  = input_ids.to(device)
        target_ids = target_ids.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            logits  = model(input_ids)
            B, T, V = logits.shape
            loss    = nn.functional.cross_entropy(
                logits.view(B * T, V), target_ids.view(B * T))
        total_loss += loss.item()
    model.train()
    return total_loss / min(val_steps, i + 1)


def save_checkpoint(model, optimizer, scaler, step, loss, checkpoint_dir):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    state_dict = (model._orig_mod.state_dict()
                  if hasattr(model, "_orig_mod")
                  else model.state_dict())
    path = os.path.join(checkpoint_dir, f"ckpt_step_{step:07d}.pt")
    torch.save({
        "step":                 step,
        "model_state_dict":     state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "loss":                 loss,
    }, path)
    print(f"  [ckpt] gespeichert: {path}")


def set_layer_drop_rate(model, rate: float):
    if hasattr(model, "_orig_mod"):
        model._orig_mod.layer_drop_rate = rate
    elif hasattr(model, "layer_drop_rate"):
        model.layer_drop_rate = rate


def get_lb_loss(model):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.get_load_balancing_loss()
    return model.get_load_balancing_loss()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Schnellere Matmuls (TF32) + cudnn-Autotuning bei fixer Seq-Länge
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.config == "large":
        cfg = ModelConfig()
    elif args.config == "small":
        cfg = ModelConfigSmall()
    else:
        cfg = ModelConfigMoE()
    print(f"Config: {args.config} | d_model={cfg.d_model} | layers={cfg.n_layers}")

    model = GPTModel(cfg,
                     use_gradient_checkpointing=args.gradient_checkpointing
                     ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameter: {n_params:.1f}M")

    train_ds = CodeDataset(args.data, context_length=cfg.context_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=True, num_workers=0,
                            pin_memory=False)

    val_loader = None
    if args.val_data:
        val_ds = CodeDataset(args.val_data, context_length=cfg.context_length)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0,
                                pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr,
                                  betas=(0.9, 0.95), weight_decay=0.1,
                                  fused=(device.type == "cuda"))
    scaler    = GradScaler("cuda")

    start_step = 0
    if args.resume:
        print(f"Lade Checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"]
        print(f"  Fortgesetzt ab Step {start_step}")

    if args.compile:
        print("torch.compile aktiviert...")
        model = torch.compile(model)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run,
                   config=vars(args))
        wandb.watch(model, log_freq=500)

    model.train()
    step      = start_step
    lb_weight = getattr(cfg, "load_balance_weight", 0.0)
    optimizer.zero_grad()

    loss_ema            = 9.0
    ema_alpha           = 0.99
    selective_threshold = 0.7

    def infinite_loader(loader):
        while True:
            yield from loader

    data_iter   = infinite_loader(train_loader)
    print("Lade ersten Batch...")
    input_ids, target_ids = next(data_iter)
    print(f"Erster Batch geladen: {input_ids.shape}")
    t0          = time.time()
    total_steps = start_step + args.steps

    print(f"\nTraining startet | {args.steps} Steps | "
          f"effektiver Batch={args.batch_size * args.accumulation_steps}")
    if args.gradient_checkpointing: print("Gradient Checkpointing: AN")
    if args.compile:                print("torch.compile: AN")
    if args.selective_backprop:     print("Selective Backprop: AN")
    if args.layer_drop:             print("Layer Dropping: AN (erste 20k Steps)")

    while step < total_steps:

        if args.layer_drop:
            set_layer_drop_rate(model, 0.3 if step < 20000 else 0.0)

        lr = get_lr(step, warmup_steps=args.warmup_steps,
                    max_steps=total_steps,
                    max_lr=args.max_lr, min_lr=args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        total_loss     = 0.0
        skipped        = 0
        loss_threshold = loss_ema * selective_threshold

        for i  in range(args.accumulation_steps):
            input_ids, target_ids = next(data_iter)
            input_ids  = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
           # print(f"  micro_step {i} - vor forward")
            with autocast("cuda", dtype=torch.bfloat16):
                logits  = model(input_ids)
                B, T, V = logits.shape
                ce_loss = nn.functional.cross_entropy(
                    logits.view(B * T, V),
                    target_ids.view(B * T),
                )

    # Selective Backprop (auf Basis des reinen CE-Loss)
            if args.selective_backprop and ce_loss.item() < loss_threshold:
                skipped += 1
                continue

            # Load-Balancing-Loss im SELBEN Graphen wie der Forward addieren,
            # sonst Expert-Collapse (nur aktiv für MoE, wo lb_weight > 0).
            loss = ce_loss
            if lb_weight > 0:
                loss = loss + get_lb_loss(model) * lb_weight
            loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()
            total_loss += ce_loss.item() / args.accumulation_steps

        if total_loss == 0.0:
            continue

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        step += 1

        loss_ema = ema_alpha * loss_ema + (1 - ema_alpha) * total_loss

        if step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"step {step:6d} | loss {total_loss:.4f} | "
                  f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | "
                  f"{dt*1000/10:.0f}ms/step"
                  + (f" | skipped {skipped}" if args.selective_backprop else ""))
            if use_wandb:
                wandb.log({"train/loss": total_loss, "train/lr": lr,
                           "train/grad_norm": grad_norm, "train/step": step})

        if val_loader and step % args.val_every == 0:
            val_loss = estimate_val_loss(model, val_loader,
                                        args.val_steps, device, cfg)
            print(f"  [val] step {step} | val_loss {val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss, "train/step": step})

        if step % args.checkpoint_every == 0:
            save_checkpoint(model, optimizer, scaler, step,
                            total_loss, args.checkpoint_dir)

    save_checkpoint(model, optimizer, scaler, step, total_loss, args.checkpoint_dir)
    print("\nTraining fertig.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()