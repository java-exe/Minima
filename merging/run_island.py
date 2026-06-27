"""
Ein Worker / eine GPU. Auf jeder Karte EINMAL starten.

Ein Rechner, zwei GPUs (gemeinsamer lokaler Ordner):
    python merging/run_island.py --worker_id gpu0 --gpu 0 \
        --shared_dir runs/moerge --shard_index 0 --num_shards 2 \
        --ckpt training/checkpoints/run_moe_001/ckpt_step_0048000.pt \
        --train data/processed/train.bin --val data/processed/val.bin \
        --local_steps 300 --rounds 50
    python merging/run_island.py --worker_id gpu1 --gpu 1 \
        --shared_dir runs/moerge --shard_index 1 --num_shards 2  ...(gleich)

Mehrere Rechner: identischer Aufruf, aber --shared_dir zeigt auf einen
GETEILTEN Pfad (NFS / SMB / Syncthing / rclone-Mount / S3-FUSE).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from model.config import ModelConfigMoE
from model.transformer import GPTModel
from training.dataset import CodeDataset
from merging.transport import LocalDirTransport
from merging.coordinator import IslandCoordinator
from merging.device import (
    setup_device, build_adamw, parse_duration, enable_utf8_stdout,
)

enable_utf8_stdout()


def _wait_for_file(path: str, timeout: float, worker_id: str) -> str:
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() > deadline:
            raise SystemExit(f"[{worker_id}] Datei nicht aufgetaucht (Sync?): {path}")
        print(f"[{worker_id}] warte auf {os.path.basename(path)} ...")
        time.sleep(5.0)
    return path


def resolve_data(args):
    """Liefert (train_path, val_path) — entweder lokal oder aus dem Netz-Ordner."""
    if args.data_dir:
        from merging.shard_data import shard_name
        tp = os.path.join(args.data_dir, shard_name(args.shard_index, args.num_shards))
        vp = os.path.join(args.data_dir, "val.bin")
        _wait_for_file(tp, args.seed_timeout, args.worker_id)
        _wait_for_file(vp, args.seed_timeout, args.worker_id)
        return tp, vp
    if not (args.train and args.val):
        raise SystemExit("Entweder --data_dir ODER --train und --val angeben.")
    return args.train, args.val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker_id",   type=str, required=True)
    p.add_argument("--shared_dir",  type=str, required=True)
    p.add_argument("--ckpt",        type=str, default=None,
                   help="Init-Checkpoint. Weglassen ⇒ aus dem Netz bootstrappen.")
    p.add_argument("--seed_timeout", type=float, default=900.0,
                   help="Wie lange auf einen Netz-Snapshot warten (Bootstrap)")
    p.add_argument("--train",       type=str, default=None,
                   help="Lokale train .bin (Alternative zu --data_dir)")
    p.add_argument("--val",         type=str, default=None)
    p.add_argument("--data_dir",    type=str, default=None,
                   help="Netz-Datenordner (von shard_data.py befüllt); zieht den "
                        "eigenen Shard, statt lokale --train/--val zu brauchen")
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards",  type=int, default=1)
    p.add_argument("--local_steps", type=int, default=300)
    p.add_argument("--rounds",      type=int, default=50)
    p.add_argument("--minutes",     type=str, default=None,
                   help="Session-Dauer, z.B. '90', '30m', '2h' (leer = unbegrenzt)")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--force_fp16",  action="store_true",
                   help="bf16 erzwungen aus (für AMD-Karten ohne bf16)")
    p.add_argument("--min_workers", type=int, default=1,
                   help="Auf so viele Inseln pro Runde warten (1 = nie blockieren)")
    p.add_argument("--collect_timeout", type=float, default=600.0)
    p.add_argument("--dead_threshold",  type=float, default=0.02)
    p.add_argument("--ckpt_dir",    type=str, default=None)
    args = p.parse_args()

    dev = setup_device(prefer_index=args.gpu, force_fp16=args.force_fp16)
    device = dev.device
    print(f"[{args.worker_id}] {dev.describe()}")

    cfg = ModelConfigMoE()
    lb_weight = getattr(cfg, "load_balance_weight", 0.0)

    transport = LocalDirTransport(args.shared_dir, args.worker_id)

    model = GPTModel(cfg).to(device)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=device)["model_state_dict"]
        model.load_state_dict(sd)
        print(f"[{args.worker_id}] Init aus lokalem Checkpoint: {args.ckpt}")
    else:
        # Bootstrap: Gewichte aus dem Netz ziehen (kein lokaler Checkpoint nötig)
        print(f"[{args.worker_id}] Kein --ckpt — warte auf Netz-Snapshot ...")
        deadline = time.time() + args.seed_timeout
        snap = transport.fetch_latest()
        while snap is None and time.time() < deadline:
            time.sleep(5.0)
            snap = transport.fetch_latest()
        if snap is None:
            raise SystemExit(f"[{args.worker_id}] Kein Snapshot in {args.shared_dir} "
                             f"innerhalb {args.seed_timeout:.0f}s. Erst seeden "
                             f"(merging/seed.py) oder einen anderen Worker starten.")
        model.load_state_dict({k: v.to(device) for k, v in snap.state_dict.items()})
        print(f"[{args.worker_id}] Init aus Netz-Snapshot von '{snap.worker_id}'.")
    optimizer = build_adamw(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1,
                            fused=dev.use_fused_adam)

    train_path, val_path = resolve_data(args)
    train_ds = CodeDataset(train_path, context_length=cfg.context_length)
    if args.data_dir is None and args.num_shards > 1:
        # lokaler Datensatz wird hier in den eigenen Shard geschnitten
        # (bei --data_dir ist die Datei bereits der fertige Shard)
        shard = len(train_ds) // args.num_shards
        idx = list(range(args.shard_index * shard,
                         (args.shard_index + 1) * shard))
        train_ds = Subset(train_ds, idx)
        print(f"[{args.worker_id}] Shard {args.shard_index}/{args.num_shards} "
              f"= {len(train_ds):,} Samples")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CodeDataset(val_path, context_length=cfg.context_length),
                            batch_size=args.batch_size, shuffle=False)

    coord = IslandCoordinator(
        model=model, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader,
        transport=transport, worker_id=args.worker_id, dev=dev,
        local_steps=args.local_steps, lb_weight=lb_weight,
        min_workers=args.min_workers, collect_timeout=args.collect_timeout,
        max_runtime_s=parse_duration(args.minutes),
        ckpt_dir=args.ckpt_dir,
        migrate_kw={"dead_threshold": args.dead_threshold, "verbose": False},
    )
    coord.run(args.rounds)


if __name__ == "__main__":
    main()
