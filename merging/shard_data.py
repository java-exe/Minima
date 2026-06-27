"""
Teilt eine flache Token-.bin in N Shards und legt sie (plus val.bin) in ein
gemeinsames Verzeichnis — damit jeder Worker NUR seinen Shard übers Netz zieht,
statt den ganzen Datensatz manuell zu kopieren.

Einmalig auf dem Daten-Halter (z.B. PC) ausführen:
    python merging/shard_data.py \
        --data data/processed/train.bin --val data/processed/val.bin \
        --out_dir /synced/moerge/data --num_shards 2

Erzeugt:  train_shard_0_of_2.bin, train_shard_1_of_2.bin, val.bin

Hinweis: Bei einem simpel synchronisierten Ordner landen alle Shards auf allen
Maschinen. Um wirklich nur den eigenen Shard zu übertragen, den jeweiligen Shard
per selektiver Sync-Regel (Syncthing ignore / rclone filter) ausschließen — das
ist Deployment-Konfiguration, kein Code.
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from merging.device import enable_utf8_stdout

enable_utf8_stdout()


def shard_name(index: int, total: int) -> str:
    return f"train_shard_{index}_of_{total}.bin"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",           type=str, required=True, help="train .bin")
    p.add_argument("--val",            type=str, default=None)
    p.add_argument("--out_dir",        type=str, required=True)
    p.add_argument("--num_shards",     type=int, required=True)
    p.add_argument("--context_length", type=int, default=512)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ctx = args.context_length

    data = np.memmap(args.data, dtype=np.int32, mode="r")
    n_samples = (len(data) - 1) // ctx
    per = n_samples // args.num_shards
    if per == 0:
        raise SystemExit("Zu wenige Samples für so viele Shards.")

    print(f"{len(data):,} Tokens = {n_samples:,} Samples → "
          f"{args.num_shards} Shards à ~{per:,} Samples")

    for i in range(args.num_shards):
        s = i * per
        e = (i + 1) * per if i < args.num_shards - 1 else n_samples
        # +1 Token am Ende, damit der letzte Sample sein Target-Offset hat
        lo, hi = s * ctx, e * ctx + 1
        chunk = np.asarray(data[lo:hi], dtype=np.int32)
        out = os.path.join(args.out_dir, shard_name(i, args.num_shards))
        chunk.tofile(out)
        print(f"  {shard_name(i, args.num_shards)}: "
              f"{len(chunk):,} Tokens ({len(chunk)*4/1024**2:.0f} MB)")

    if args.val:
        shutil.copyfile(args.val, os.path.join(args.out_dir, "val.bin"))
        print(f"  val.bin kopiert ({os.path.getsize(args.val)/1024**2:.0f} MB)")

    print("Fertig. Worker starten mit --data_dir " + args.out_dir)


if __name__ == "__main__":
    main()
