import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Lernt einen Vektor der Größe d_model für jedes Token im Vokabular.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) → (B, T, d_model)
        # Skalierung wie im "Attention is All You Need" Paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidale Positional Encoding — fix, nicht gelernt.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Warum sinusoidal statt gelernt?
    - Funktioniert auch für Sequenzlängen die beim Training nicht vorkamen
    - Weniger Parameter
    - Für unser Projekt ausreichend (GPT-2 verwendet gelernte PE, wir nicht)
    """
    def __init__(self, d_model: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # PE Matrix vorberechnen: (context_length, d_model)
        pe = torch.zeros(context_length, d_model)
        position = torch.arange(0, context_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # gerade Indizes
        pe[:, 1::2] = torch.cos(position * div_term)  # ungerade Indizes

        # (1, context_length, d_model) → kann auf Batch gebroadcastet werden
        pe = pe.unsqueeze(0)

        # register_buffer: wird mit dem Modell gespeichert aber nicht trainiert
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Embedding(nn.Module):
    """
    Kombiniert Token Embedding + Positional Encoding.
    Das ist der Einstiegspunkt für den Transformer.
    """
    def __init__(self, vocab_size: int, d_model: int,
                 context_length: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding    = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, context_length, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) Token-IDs
        # → (B, T, d_model) Embeddings + Positional Encoding
        return self.positional_encoding(self.token_embedding(x))