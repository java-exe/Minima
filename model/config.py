from dataclasses import dataclass


@dataclass
class ModelConfig:
    # ~268M Parameter (d_model 1024 × 20 Layer × d_ff 4096, tied embeddings)
    vocab_size: int     = 16000
    context_length: int = 512
    d_model: int        = 1024
    n_heads: int        = 16
    n_layers: int       = 20
    d_ff: int           = 4096
    dropout: float      = 0.1


@dataclass
class ModelConfigSmall:
    vocab_size: int     = 16000
    context_length: int = 256
    d_model: int        = 256
    n_heads: int        = 8
    n_layers: int       = 4
    d_ff: int           = 1024
    dropout: float      = 0.1


@dataclass
class ModelConfigMoE:
    vocab_size: int     = 16000
    context_length: int = 512
    d_model: int        = 768
    n_heads: int        = 12
    n_layers: int       = 12
    d_ff: int           = 1536
    dropout: float      = 0.1
    n_experts: int      = 8
    top_k: int          = 2
    load_balance_weight: float = 0.01   # >0 nötig, sonst Expert-Collapse