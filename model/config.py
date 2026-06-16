"""Modell-Konfigurationen (Person 2, Woche 2).

Zwei Configs: 50M fuer die Haupt-Trainingsmaschine (12GB VRAM) und 12M fuer
Experimente auf 8GB-Karten. Siehe plan.md, Hardware & Training Setup.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """~50M Parameter, passt in 12GB VRAM mit Batch Size 8-16."""

    vocab_size: int = 16000
    context_length: int = 512
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1


@dataclass
class ModelConfigSmall:
    """~12M Parameter, passt sicher in 8GB VRAM."""

    vocab_size: int = 16000
    context_length: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
