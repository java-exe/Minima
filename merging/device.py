"""
Backend-agnostische Geräte- und Präzisions-Einrichtung.

Ziel: derselbe Client läuft auf NVIDIA (CUDA), AMD (ROCm), Apple (MPS) und CPU.
Da Worker GEWICHTE austauschen (keine Gradienten), ist gemischte Präzision über
verschiedene Worker hinweg unproblematisch — kleine numerische Unterschiede
mitteln sich heraus.

Fallstricke, die hier zentral behandelt werden:
  • TF32 / float32_matmul_precision sind NVIDIA-Ampere-only → nur dort setzen.
  • fused AdamW ist oft nur auf NVIDIA stabil → Feature-Detection + Fallback.
  • bf16 ist auf vielen Consumer-AMD-Karten wackelig → auf fp16 zurückfallen.
  • ROCm meldet sich in PyTorch als device_type "cuda" (torch.version.hip != None).
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass
class DeviceConfig:
    device: torch.device
    kind: str                 # "cuda" | "rocm" | "mps" | "cpu"
    amp_device_type: str      # für torch.autocast: "cuda" | "cpu" | "mps"
    amp_dtype: torch.dtype | None   # None ⇒ kein autocast (fp32)
    use_fused_adam: bool

    def autocast(self):
        """Frischer autocast-Context (oder nullcontext bei fp32)."""
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(self.amp_device_type, dtype=self.amp_dtype)

    def describe(self) -> str:
        dt = {torch.bfloat16: "bf16", torch.float16: "fp16",
              None: "fp32"}.get(self.amp_dtype, str(self.amp_dtype))
        return (f"{self.kind} ({self.device}) | amp={dt} | "
                f"fused_adam={self.use_fused_adam}")


def setup_device(prefer_index: int = 0, force_fp16: bool = False) -> DeviceConfig:
    # ── NVIDIA oder AMD/ROCm (beide melden cuda.is_available) ──────────
    if torch.cuda.is_available():
        is_rocm = getattr(torch.version, "hip", None) is not None
        kind = "rocm" if is_rocm else "cuda"

        idx = prefer_index if prefer_index < torch.cuda.device_count() else 0
        torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")

        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        amp_dtype = torch.float16 if (force_fp16 or not bf16_ok) else torch.bfloat16

        if kind == "cuda":  # NUR NVIDIA: TF32-Pfade
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True   # auch unter MIOpen sinnvoll

        return DeviceConfig(device, kind, "cuda", amp_dtype,
                            use_fused_adam=(kind == "cuda"))

    # ── Apple Silicon ─────────────────────────────────────────────────
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # MPS-autocast ist versionsabhängig wackelig → sicherheitshalber fp32
        return DeviceConfig(torch.device("mps"), "mps", "mps", None, False)

    # ── CPU ───────────────────────────────────────────────────────────
    return DeviceConfig(torch.device("cpu"), "cpu", "cpu", torch.bfloat16, False)


def build_adamw(params, lr: float, betas=(0.9, 0.95),
                weight_decay: float = 0.1, fused: bool = False):
    """AdamW mit fused-Kernel, fällt bei Nichtunterstützung sauber zurück."""
    if fused:
        try:
            return torch.optim.AdamW(params, lr=lr, betas=betas,
                                     weight_decay=weight_decay, fused=True)
        except (RuntimeError, ValueError):
            pass
    return torch.optim.AdamW(params, lr=lr, betas=betas,
                             weight_decay=weight_decay)


def enable_utf8_stdout() -> None:
    """Windows-Konsolen (cp1252) brechen bei Unicode-Ausgaben — auf UTF-8 stellen."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def parse_duration(text: str | None) -> float | None:
    """'90' → 90s, '30m' → 1800s, '2h' → 7200s, None → None (unbegrenzt)."""
    if not text:
        return None
    text = text.strip().lower()
    mult = 1.0
    if text.endswith("h"):
        mult, text = 3600.0, text[:-1]
    elif text.endswith("m"):
        mult, text = 60.0, text[:-1]
    elif text.endswith("s"):
        text = text[:-1]
    return float(text) * mult
