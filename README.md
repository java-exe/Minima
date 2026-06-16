# SLM Schulprojekt – Small Language Model für Code-Generierung

> Decoder-only Transformer (GPT-Architektur) von Grund auf, trainiert auf einem
> Code-Datensatz. HTL-Schulprojekt, zwei Personen, Python + C++ + C#.
> Vollständiger Projektplan: [plan.md](plan.md).

## Pipeline

```
Datensatz → C++ Tokenizer (pybind11) → Training (PyTorch) → ONNX Export
          → C++ Inference Engine (onnxruntime) → C# WPF Frontend
```

Kein Python zur Laufzeit: nach dem Training läuft alles über die C++-DLL.

## Projektstruktur

```
.
├── tokenizer/      # BPE-Tokenizer in C++ (Person 2)
│   ├── src/
│   │   ├── bpe.cpp
│   │   ├── bpe.hpp
│   │   └── bindings.cpp    # pybind11 → Modul `tokenizer_cpp`
│   ├── CMakeLists.txt
│   └── tests/
├── model/          # Transformer Core, PyTorch (Person 2)
│   ├── embedding.py
│   ├── attention.py
│   ├── transformer.py
│   └── config.py
├── training/       # Training-Loop + Dataset, Python (Person 1)
│   ├── dataset.py
│   ├── train.py
│   ├── scheduler.py
│   ├── mock_tokenizer.py   # Stand-in bis tokenizer_cpp fertig
│   └── checkpoints/        # gitignored
├── inference/
│   ├── export_onnx.py      # Modell → model.onnx (gemeinsam)
│   └── engine/             # C++ Inference Engine (Person 2)
│       ├── inference.cpp
│       ├── inference.hpp
│       └── CMakeLists.txt
├── frontend/       # C# WPF Demo (Person 1)
│   └── SLMDemo/
├── data/           # Datensatz-Scripts (Person 1)
│   ├── download.py
│   ├── preprocess.py
│   └── README.md
├── scripts/
│   ├── train_large.sh
│   └── train_small.sh
├── docs/
│   └── interface.md        # Tokenizer↔Modell↔Inference Verträge
└── plan.md
```

## Sprachaufteilung

| Sprache | Wer       | Wofür                                         |
|---------|-----------|-----------------------------------------------|
| Python  | Beide     | Training Loop, Datensatz, Transformer Core    |
| C++     | Person 2  | BPE-Tokenizer (pybind11), ONNX Inference      |
| C#      | Person 1  | Demo-Frontend (WPF)                           |

## Setup

```bash
# PyTorch mit CUDA (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# C++ Tokenizer bauen (pybind11)
pip install pybind11
cmake -B tokenizer/build -S tokenizer -DCMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir)
cmake --build tokenizer/build --config Release
```

## Status

Phase 0 – Setup & Interface. Die Code-Dateien sind aktuell dokumentierte Stubs,
die gemäß [plan.md](plan.md) ausimplementiert werden.
