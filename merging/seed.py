"""
Einen vorhandenen Checkpoint als Bootstrap-Snapshot ins gemeinsame Verzeichnis
legen — damit andere Worker das Modell ÜBER DEN CODE beziehen können, statt eine
große .pt-Datei manuell zu kopieren.

Beispiel (PC seedet einmalig den 48k-Checkpoint):
    python merging/seed.py \
        --ckpt training/checkpoints/run_moe_001/ckpt_step_0048000.pt \
        --shared_dir /synced/moerge

Danach startet der Laptop ohne --ckpt und zieht die Gewichte aus dem Netz:
    python merging/run_island.py --worker_id laptop --shared_dir /synced/moerge ...
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from merging.expert_migration import IslandSnapshot
from merging.transport import LocalDirTransport
from merging.device import enable_utf8_stdout

enable_utf8_stdout()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, required=True)
    p.add_argument("--shared_dir", type=str, required=True)
    p.add_argument("--round",      type=int, default=0,
                   help="Runde, unter der der Seed abgelegt wird")
    p.add_argument("--half", action="store_true",
                   help="Gewichte als fp16 ablegen (halbiert die Übertragung)")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if args.half:
        state = {k: (v.half() if v.is_floating_point() else v)
                 for k, v in state.items()}

    snap = IslandSnapshot(worker_id="seed", step=int(ckpt.get("step", 0)),
                          state_dict=state, util={})
    tx = LocalDirTransport(args.shared_dir, "seed")
    tx.publish(snap, args.round)

    n = sum(v.numel() for v in state.values()) / 1e6
    print(f"Seed veröffentlicht: {n:.1f}M Werte → {args.shared_dir} "
          f"(round {args.round}, {'fp16' if args.half else 'fp32'})")
    print("Andere Worker können jetzt ohne --ckpt beitreten.")


if __name__ == "__main__":
    main()
