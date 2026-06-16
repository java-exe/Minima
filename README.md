# SLM Schulprojekt – Small Language Model für Code-Generierung

> Decoder-only Transformer (GPT-Architektur) von Grund auf, trainiert auf einem
> Code-Datensatz. HTL-Schulprojekt, zwei Personen, Python + C#.
> Vollständiger Projektplan: [plan.md](plan.md).

## Projektstruktur

```
.
├── tokenizer/      # BPE-Tokenizer (Person 1)
│   ├── bpe.py
│   ├── vocab.py
│   ├── encode.py
│   ├── mock_tokenizer.py
│   └── tests/
├── model/          # Transformer Core (Person 2)
│   ├── embedding.py
│   ├── attention.py
│   ├── transformer.py
│   └── config.py
├── training/       # Training-Loop (gemeinsam)
│   ├── dataset.py
│   ├── train.py
│   ├── scheduler.py
│   └── checkpoints/   # gitignored
├── inference/      # Generierung + API (gemeinsam)
│   ├── generate.py
│   └── api.py
├── frontend/       # C# WPF Demo (Person 1)
│   └── SLMDemo/
├── data/           # Datensatz-Scripts
│   ├── download.py
│   ├── preprocess.py
│   └── README.md
├── scripts/        # Training-Startscripts
│   ├── train_large.sh
│   └── train_small.sh
├── docs/
│   └── interface.md   # Tokenizer↔Modell Vertrag
└── plan.md
```

## Setup

```bash
# PyTorch mit CUDA (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Testen:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Status

Phase 0 – Setup & Interface. Die Code-Dateien sind aktuell dokumentierte Stubs,
die gemäß [plan.md](plan.md) ausimplementiert werden.
