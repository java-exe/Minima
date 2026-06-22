import math

def get_lr(
    step: int,
    warmup_steps: int = 500,
    max_steps: int = 100_000,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
) -> float:

    if step < warmup_steps:
        return max_lr * step / warmup_steps

    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))