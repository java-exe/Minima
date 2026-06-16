"""Training-Loop (Person 2, Woche 5).

  - AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
  - Gradient Clipping: clip_grad_norm_(model.parameters(), 1.0)
  - Mixed Precision: autocast() + GradScaler
  - Checkpoint alle 1000 Steps
  - W&B Logging: loss, lr, grad_norm

Aufruf (siehe scripts/): python training/train.py --config large ...
Siehe plan.md, Phase 1, Woche 5 + Phase 2.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["large", "small"], default="small")
    parser.add_argument("--data", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--checkpoint_dir", type=str, default="training/checkpoints/run_001")
    args = parser.parse_args()

    raise NotImplementedError


if __name__ == "__main__":
    main()
