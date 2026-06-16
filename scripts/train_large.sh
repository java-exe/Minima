#!/usr/bin/env bash
# Grosses Training: 50M Modell (Haupt-Trainingsmaschine, 12GB VRAM).
# Siehe plan.md, Phase 2, Woche 9.
set -euo pipefail

python training/train.py \
    --config large \
    --data data/processed/train.bin \
    --batch_size 8 \
    --accumulation_steps 16 \
    --steps 100000 \
    --checkpoint_dir training/checkpoints/run_001
# Effektiver Batch = 8 x 16 = 128
