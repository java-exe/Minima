"""Cosine LR Schedule mit Warmup (Person 2, Woche 5).

Siehe plan.md, Phase 1, Woche 5.
"""


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    # TODO: Cosine decay danach
    raise NotImplementedError
