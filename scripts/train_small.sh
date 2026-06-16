#!/usr/bin/env bash
# Kleines Training: 12M Modell fuer Experimente (8GB VRAM).
# Siehe plan.md, Phase 2, Woche 8.
set -euo pipefail

python training/train.py \
    --config small \
    --data data/processed/train.bin \
    --batch_size 4 \
    --accumulation_steps 8 \
    --steps 10000 \
    --checkpoint_dir training/checkpoints/run_small
