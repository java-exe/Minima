from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int     = 16000
    context_length: int = 512
    d_model: int        = 512
    n_heads: int        = 8
    n_layers: int       = 8
    d_ff: int           = 2048
    dropout: float      = 0.1
    # → ca. 50M Parameter, 12GB VRAM

@dataclass
class ModelConfigSmall:
    vocab_size: int     = 16000
    context_length: int = 256
    d_model: int        = 256
    n_heads: int        = 8
    n_layers: int       = 4
    d_ff: int           = 1024
    dropout: float      = 0.1
    # → ca. 12M Parameter, 8GB VRAM