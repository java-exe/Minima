# SLM Schulprojekt – Projektplan
> Small Language Model für Code-Generierung  
> Zwei Personen | HTL-Schulprojekt | Python + C++ + C#
---
## Scope
### Was dieses Projekt IST
- Ein Decoder-only Transformer (GPT-Architektur) von Grund auf implementiert
- Trainiert auf einem Code-Datensatz (Python/Java Snippets + vollständige Programme)
- Fähig, einfache vollständige Programme zu schreiben (Funktionen, Klassen, Algorithmen)
- BPE-Tokenizer in C++ implementiert, via pybind11 ins Python-Training eingebunden
- Nach dem Training: ONNX-Export + C++ Inference Engine (kein Python zur Laufzeit)
- C# Demo-Frontend das direkt die C++ Inference Engine aufruft
- Lokal trainiert auf eigener Hardware (kein Cloud-Dienst nötig)
### Was dieses Projekt NICHT IST
- Kein GPT-4 Klon – das Modell wird klein sein (~50M Parameter)
- Kein Fine-Tuning von bestehenden Modellen (kein HuggingFace Pretrained)
- Keine Produktionsanwendung
- Kein vollständiger Code-Assistent – aber mehr als nur Snippet-Vervollständigung
### Sprachaufteilung
| Sprache | Wer | Wofür |
|---|---|---|
| Python | Beide (neu gelernt, gleiches Level) | Training Loop, Datensatz-Pipeline, Transformer Core |
| C++ | Person 2 | BPE-Tokenizer, ONNX Inference Engine |
| C# | Person 1 | Demo-Frontend (WPF) |
### Erfolgskriterien
- [ ] Modell schreibt syntaktisch korrekte, einfache Programme (Fibonacci, Bubble Sort, Stack-Klasse, etc.)
- [ ] Komplette Pipeline: Datensatz → C++ Tokenizer → Training → ONNX Export → C++ Inference → C# Frontend
- [ ] Alles selbst implementiert (kein HuggingFace Transformers, kein Keras)
- [ ] Funktionierende Demo für die Präsentation
- [ ] Dokumentierter, sauberer Code auf GitHub
---
## Hardware & Training Setup
| Gerät | GPU | VRAM | Rolle |
|---|---|---|---|
| PC mit 3060 | RTX 3060 | 12 GB | **Haupt-Trainingsmaschine** – 50M Modell overnight |
| PC mit 4060 | RTX 4060 | 8 GB | Experimente, kleine Runs, Hyperparameter-Suche |
| PC mit 5060 | RTX 5060 | 8–12 GB | Zweite Experimentierkiste, Inference-Tests |
### CUDA Setup (alle Maschinen)
```bash
# PyTorch mit CUDA installieren (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Testen:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
### Modell-Konfiguration – Groß (3060, 12GB VRAM, Haupt-Training)
```python
@dataclass
class ModelConfig:
    vocab_size: int     = 16000
    context_length: int = 512        # Längere Sequenzen → ganze Programme
    d_model: int        = 512
    n_heads: int        = 8
    n_layers: int       = 8
    d_ff: int           = 2048
    dropout: float      = 0.1
    # → ca. 50M Parameter, passt in 12GB VRAM mit batch_size=8 + accumulation
```
### Modell-Konfiguration – Klein (4060/5060, 8GB VRAM, Experimente)
```python
@dataclass
class ModelConfigSmall:
    vocab_size: int     = 16000
    context_length: int = 256
    d_model: int        = 256
    n_heads: int        = 8
    n_layers: int       = 4
    d_ff: int           = 1024
    dropout: float      = 0.1
    # → ca. 12M Parameter, passt sicher in 8GB VRAM
```
### Erwartete Trainingszeit (3060, 12GB VRAM)
| Steps | Zeit (ca.) | Loss (ca.) |
|---|---|---|
| 1.000 | ~15 Min | ~6–7 |
| 10.000 | ~2.5 Std | ~3–4 |
| 50.000 | ~12 Std | ~2–2.5 |
| 100.000 | ~24 Std | ~1.8–2.2 |
---
## Tech Stack
| Komponente | Technologie | Wer |
|---|---|---|
| BPE-Tokenizer | C++ + pybind11 | Person 2 |
| Transformer Core | Python + PyTorch | Person 2 |
| Training Loop | Python + PyTorch + CUDA | Person 1 (Hauptanteil) + Person 2 |
| Datensatz-Pipeline | Python | Person 1 |
| ONNX Export | Python (2 Zeilen) | Gemeinsam |
| Inference Engine | C++ + onnxruntime | Person 2 |
| Demo Frontend | C# WPF | Person 1 |
| Versionskontrolle | Git + GitHub | Beide |
| Experiment-Tracking | Weights & Biases | Beide |
---
## Projektstruktur (GitHub Repo)
```
slm-project/
├── tokenizer/                  # Person 2 (C++)
│   ├── src/
│   │   ├── bpe.cpp
│   │   ├── bpe.hpp
│   │   └── bindings.cpp        # pybind11 Python-Binding
│   ├── CMakeLists.txt
│   └── tests/
├── model/                      # Person 2 (Python)
│   ├── embedding.py
│   ├── attention.py
│   ├── transformer.py
│   └── config.py
├── training/                   # Person 1 (Python, Hauptanteil)
│   ├── dataset.py
│   ├── train.py
│   ├── scheduler.py
│   └── checkpoints/            # .gitignore
├── inference/
│   ├── export_onnx.py          # Gemeinsam – Modell → ONNX
│   └── engine/                 # Person 2 (C++)
│       ├── inference.cpp
│       ├── inference.hpp
│       └── CMakeLists.txt
├── frontend/                   # Person 1 (C#)
│   └── SLMDemo/
├── data/
│   ├── download.py             # Person 1
│   ├── preprocess.py           # Person 1
│   └── README.md
├── scripts/
│   ├── train_large.sh
│   └── train_small.sh
├── docs/
│   └── interface.md            # Tokenizer↔Modell Vertrag
└── README.md
```
---
## Interface-Dokument (Woche 1 gemeinsam festlegen)
Dieser Vertrag muss VOR paralleler Arbeit stehen.
### Python-seitige Tokenizer-Schnittstelle (via pybind11)
```python
# Nach dem Build: import tokenizer_cpp
import tokenizer_cpp
tok = tokenizer_cpp.Tokenizer("path/to/vocab.bin")
tok.encode("def foo():")       # → [2, 145, 67, 233, 891, 1]
tok.decode([2, 145, 67, 233])  # → "def foo"
tok.vocab_size    # 16000
tok.pad_token_id  # 0
tok.eos_token_id  # 1
tok.bos_token_id  # 2
tok.unk_token_id  # 3
```
### Mock-Tokenizer (Person 1 verwendet bis C++ Tokenizer fertig)
```python
# mock_tokenizer.py
import random
class MockTokenizer:
    def __init__(self):
        self.vocab_size    = 16000
        self.pad_token_id  = 0
        self.eos_token_id  = 1
        self.bos_token_id  = 2
        self.unk_token_id  = 3
    def encode(self, text: str):
        return [self.bos_token_id] + \
               [random.randint(4, self.vocab_size - 1) for _ in text.split()] + \
               [self.eos_token_id]
    def decode(self, ids):
        return "<mock decoded text>"
```
### C++ Inference → C# Interface
```
C# Frontend
  → P/Invoke oder named pipe
  → inference.dll (C++ onnxruntime)
  → model.onnx + vocab.bin
  → generierter Code (string zurück)
```
---
## Phasenplan
---
### Phase 0 – Setup & Interface (Woche 1, BEIDE gemeinsam)
**Ziel:** Alle Tools installiert, Interface-Vertrag dokumentiert, Repo steht.
- [ ] GitHub Repo anlegen, Branch-Strategie: `main`, `dev`, Feature-Branches
- [ ] Python + CUDA PyTorch auf allen PCs installieren und testen
- [ ] VS Code + Pylance (Python), Visual Studio (C#), CLion oder VS (C++)
- [ ] pybind11 installieren und Hello-World-Binding testen (Person 2)
- [ ] `docs/interface.md` gemeinsam schreiben
- [ ] Mock-Tokenizer committen damit Person 1 sofort mit Training-Code starten kann
- [ ] Weights & Biases Account: https://wandb.ai
- [ ] `.gitignore` einrichten (Checkpoints, Rohdaten, Build-Artefakte)
**Deliverable:** `python -c "import torch; print(torch.cuda.is_available())"` → True auf allen PCs. pybind11 Hello-World läuft auf Person 2 PC.
---
### Phase 1 – Parallele Kernentwicklung (Woche 2–7)
---
#### Person 1 – Python lernen + Training Loop + Datensatz
**Ziel:** Kompletter Training Loop der mit Mock-Tokenizer läuft, Datensatz-Pipeline fertig.
##### Woche 2 – Python lernen + Theorie
- [ ] Python Crashkurs: Listen, Dicts, Klassen, File I/O, List Comprehensions
  - Mit Java/C# Background: 3–4 Tage reichen
  - Übung: Script das Textdatei einliest, Wortfrequenzen zählt, nach Häufigkeit sortiert
- [ ] PyTorch Grundlagen: Tensoren, `.to(device)`, `nn.Module`, Forward Pass
  - Karpathy "Let's build GPT" Video schauen (2h) – gemeinsam mit Person 2
- [ ] BPE-Algorithmus verstehen (für Verständnis, nicht Implementation):
  - Sennrich 2016 Paper lesen (4 Seiten)
##### Woche 3 – Dataset-Klasse
- [ ] `training/dataset.py`:
  ```python
  class CodeDataset(torch.utils.data.Dataset):
      # Lädt vorverarbeitete Token-IDs aus .bin Datei
      # __len__: Anzahl Sequenzen
      # __getitem__(i): gibt (input_ids[:-1], target_ids[1:]) zurück
      #                 → Next-Token-Prediction
      # Nutzt Mock-Tokenizer bis C++ Tokenizer fertig
  ```
- [ ] DataLoader testen: Batch Shape `(batch=8, seq=511)` korrekt?
- [ ] `data/download.py` – Datensatz herunterladen:
  ```python
  from datasets import load_dataset
  ds_py   = load_dataset("code_search_net", "python")
  ds_java = load_dataset("code_search_net", "java")
  ```
##### Woche 4 – Preprocessing Pipeline
- [ ] `data/preprocess.py`:
  - Duplikate entfernen (MD5-Hash pro Funktion)
  - Python-Code syntaktisch validieren: `ast.parse(code)`
  - Zu lange Dateien auf 512 Token kürzen oder aufteilen
  - Train/Val Split 95%/5%
  - Alles als `.bin` speichern (numpy memmap für schnelles Laden)
- [ ] GitHub Gist Scraper (einfache Version):
  ```python
  # requests gegen https://api.github.com/gists/public
  # Nur Python/Java Gists, < 100 Zeilen
  # Ziel: ~20k vollständige kleine Programme
  ```
##### Woche 5 – Learning Rate Scheduler
- [ ] `training/scheduler.py` – Cosine LR mit Warmup:
  ```python
  def get_lr(step, warmup_steps=500, max_steps=100000,
             max_lr=3e-4, min_lr=3e-5) -> float:
      if step < warmup_steps:
          return max_lr * step / warmup_steps
      progress = (step - warmup_steps) / (max_steps - warmup_steps)
      return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
  ```
##### Woche 6 – Training Loop (Hauptanteil Person 1)
- [ ] `training/train.py`:
  ```python
  # Argparse: --config, --data, --batch_size, --steps, --checkpoint_dir
  # AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
  # Gradient Clipping: clip_grad_norm_(model.parameters(), 1.0)
  # Mixed Precision: torch.cuda.amp.autocast() + GradScaler
  # Gradient Accumulation: effektiver Batch = batch_size × accumulation_steps
  # Checkpoint alle 2000 Steps speichern
  # W&B: loss, val_loss, lr, grad_norm loggen
  ```
- [ ] Overfitting-Test: auf 20 Beispielen trainieren bis Loss < 0.1 → Training Loop korrekt
##### Woche 7 – Integration + Datensatz fertigstellen
- [ ] C++ Tokenizer (von Person 2) einbinden, Mock ersetzen
- [ ] Datensatz komplett vorverarbeiten und als `.bin` speichern
- [ ] Ende-zu-Ende Test: Datensatz laden → Tokenizer → Training 100 Steps → kein Fehler
---
#### Person 2 – C++ Tokenizer + Transformer Core + Inference Engine
**Ziel:** C++ BPE-Tokenizer via pybind11 eingebunden, Transformer Core fertig, ONNX Inference Engine.
##### Woche 2 – Python lernen + Theorie + pybind11
- [ ] Python Crashkurs: gleiche Basis wie Person 1 (3–4 Tage)
- [ ] Karpathy "Let's build GPT" Video gemeinsam schauen
- [ ] "Attention is All You Need" (Vaswani 2017) lesen
- [ ] pybind11 Hello-World:
  ```cpp
  // hello.cpp
  #include <pybind11/pybind11.h>
  namespace py = pybind11;
  int add(int a, int b) { return a + b; }
  PYBIND11_MODULE(hello, m) {
      m.def("add", &add);
  }
  // python: import hello; hello.add(1, 2)  → 3
  ```
- [ ] CMakeLists.txt für pybind11 Projekt aufsetzen
##### Woche 3 – BPE Kernalgorithmus in C++
- [ ] `tokenizer/src/bpe.hpp` + `bpe.cpp`:
  ```cpp
  class BPE {
  public:
      // Trainieren: Corpus einlesen, num_merges Mal häufigstes Paar mergen
      void train(const std::vector<std::string>& corpus, int num_merges);
      // Serialisieren
      void save(const std::string& path) const;
      void load(const std::string& path);
      // Anwenden
      std::vector<int> encode(const std::string& text) const;
      std::string      decode(const std::vector<int>& ids) const;
      int vocab_size()    const { return vocab_.size(); }
      int pad_token_id()  const { return 0; }
      int eos_token_id()  const { return 1; }
      int bos_token_id()  const { return 2; }
      int unk_token_id()  const { return 3; }
  private:
      std::unordered_map<std::string, int> vocab_;
      std::vector<std::string>             id_to_token_;
      std::vector<std::pair<std::string, std::string>> merges_;
  };
  ```
- [ ] Unit Tests (Catch2 oder einfache assert-Tests):
  - encode → decode Roundtrip
  - Unbekannte Zeichen → unk_token_id
  - Spezial-Tokens korrekt
##### Woche 4 – pybind11 Binding + Transformer Embeddings
- [ ] `tokenizer/src/bindings.cpp`:
  ```cpp
  #include <pybind11/pybind11.h>
  #include <pybind11/stl.h>
  #include "bpe.hpp"
  PYBIND11_MODULE(tokenizer_cpp, m) {
      py::class_<BPE>(m, "Tokenizer")
          .def(py::init<>())
          .def("train",        &BPE::train)
          .def("save",         &BPE::save)
          .def("load",         &BPE::load)
          .def("encode",       &BPE::encode)
          .def("decode",       &BPE::decode)
          .def_property_readonly("vocab_size",   &BPE::vocab_size)
          .def_property_readonly("pad_token_id", &BPE::pad_token_id)
          .def_property_readonly("eos_token_id", &BPE::eos_token_id)
          .def_property_readonly("bos_token_id", &BPE::bos_token_id)
          .def_property_readonly("unk_token_id", &BPE::unk_token_id);
  }
  ```
- [ ] Build testen: `import tokenizer_cpp` in Python funktioniert
- [ ] `model/embedding.py` – Token Embedding + sinusoidale Positional Encoding:
  ```python
  # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```
##### Woche 5 – Multi-Head Attention
- [ ] `model/attention.py`:
  ```python
  def scaled_dot_product_attention(Q, K, V, mask=None):
      # scores = QK^T / sqrt(d_k)
      # scores.masked_fill_(mask, float('-inf'))
      # return softmax(scores) @ V
  class MultiHeadAttention(nn.Module):
      # Linear W_Q, W_K, W_V, W_O
      # Split in n_heads → attention → concat → project
  ```
- [ ] Causal Mask: `torch.triu(torch.ones(seq, seq), diagonal=1).bool()`
- [ ] Tests: Input `(2, 512, 512)` → Output `(2, 512, 512)`, keine NaN
##### Woche 6 – Kompletter Transformer
- [ ] `model/transformer.py`:
  ```python
  class FeedForward(nn.Module):
      # Linear(d_model, d_ff) → GELU → Dropout → Linear(d_ff, d_model)
  class TransformerBlock(nn.Module):
      # Pre-LayerNorm (stabiler):
      # x = x + Attention(LayerNorm(x))
      # x = x + FeedForward(LayerNorm(x))
  class GPTModel(nn.Module):
      # Embedding + Positional Encoding
      # N × TransformerBlock
      # Final LayerNorm
      # Linear(d_model, vocab_size)  ← Logits, kein Softmax
  ```
- [ ] Parameteranzahl prüfen: `sum(p.numel() for p in model.parameters()) / 1e6` → ~50M
- [ ] Mit Person 1 Integration testen: Training Loop + Transformer + C++ Tokenizer
##### Woche 7 – ONNX Export vorbereiten + Integration
- [ ] Gemeinsam mit Person 1: Ende-zu-Ende Test
- [ ] `inference/export_onnx.py`:
  ```python
  import torch.onnx
  dummy_input = torch.zeros(1, 512, dtype=torch.long)
  torch.onnx.export(model, dummy_input, "model.onnx",
                    input_names=["input_ids"],
                    output_names=["logits"],
                    dynamic_axes={"input_ids": {1: "seq_len"}})
  ```
- [ ] ONNX Modell mit `onnxruntime` in Python testen (Sanity Check)
---
### Phase 2 – Training (Woche 8–10, Gemeinsam)
**Ziel:** 50M Modell trained, generiert brauchbaren Code.
##### Datensatz finalisieren (Woche 8)
- [ ] Vollständiger Datensatz vorverarbeitet als `.bin` gespeichert
- [ ] C++ Tokenizer auf CodeSearchNet + GitHub Gists trainiert (vocab_size=16000)
- [ ] Dataset-Klasse lädt `.bin` korrekt, Batches haben richtige Shape
##### Erstes Training – klein (Woche 8)
- [ ] 12M Modell auf 4060 oder 5060, 10.000 Steps
- [ ] W&B Dashboard: Loss, LR, Grad-Norm live beobachten
- [ ] Loss soll von ~9.7 (log2(16000)) auf ~3–4 fallen
- [ ] Erste Inference nach 5000 Steps: `"def add(a, b):"` → irgendwas Sinnvolles?
##### Großes Training – 50M (Woche 9)
- [ ] `scripts/train_large.sh`:
  ```bash
  python training/train.py \
      --config large \
      --data data/processed/train.bin \
      --batch_size 8 \
      --accumulation_steps 16 \
      --steps 100000 \
      --checkpoint_dir training/checkpoints/run_001
  # Effektiver Batch = 8 × 16 = 128
  ```
- [ ] Auf 3060 overnight (~24 Stunden)
- [ ] Checkpoint alle 5000 Steps
##### Evaluation & Tuning (Woche 10)
- [ ] Checkpoint bei niedrigstem Val-Loss laden
- [ ] Manuelle Evaluation:
  ```python
  "def fibonacci(n):"
  "# Bubble sort\ndef bubble_sort(arr):"
  "class Stack:\n    def __init__(self):"
  "def is_prime(n):\n    "
  "# Count vowels in a string\ndef count_vowels(s):"
  ```
- [ ] Qualitätsbewertung (0–3 pro Prompt, Durchschnitt ≥ 1.5 = Ziel erreicht)
- [ ] Falls zu schlecht: LR anpassen, nochmal trainieren
---
### Phase 3 – C++ Inference Engine + C# Frontend (Woche 10–12)
**Ziel:** Vollständige Demo-Pipeline ohne Python zur Laufzeit.
#### Person 2 – C++ Inference Engine (Woche 10–11)
- [ ] `inference/engine/inference.cpp`:
  ```cpp
  #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
  #include "bpe.hpp"
  class InferenceEngine {
  public:
      InferenceEngine(const std::string& model_path,
                      const std::string& vocab_path);
      std::string generate(const std::string& prompt,
                           int   max_new_tokens = 200,
                           float temperature    = 0.8f,
                           int   top_k          = 50);
  private:
      Ort::Session       session_;
      BPE                tokenizer_;
      // Top-K Sampling intern
  };
  ```
- [ ] Als DLL kompilieren (`inference.dll` auf Windows)
- [ ] Einfaches CLI-Test-Tool: `./inference "def fibonacci(n):"` gibt Code aus
#### Person 1 – C# WPF Frontend (Woche 10–12)
- [ ] Neues WPF Projekt in Visual Studio
- [ ] NuGet: `AvalonEdit` (Syntax Highlighting)
- [ ] P/Invoke zur `inference.dll`:
  ```csharp
  [DllImport("inference.dll")]
  private static extern IntPtr generate(
      string prompt, int maxTokens, float temperature, int topK);
  ```
- [ ] UI-Elemente:
  - Prompt-Eingabe (AvalonEdit, Python-Syntax-Highlighting)
  - "Generate" Button
  - Output-Feld (AvalonEdit, read-only)
  - Temperature Slider (0.1 – 1.5)
  - Max Tokens Eingabe
  - Top-K Slider
  - Statuszeile: "Generating..." / "Done"
- [ ] Dark Theme
---
### Phase 4 – Präsentation (Woche 12, Gemeinsam)
##### Struktur (~20 Minuten)
1. **Motivation** (2 min) – Was ist ein LLM? ChatGPT vereinfacht erklärt
2. **Architektur** (5 min) – Transformer-Diagramm, Attention erklärt
3. **Tokenizer** (3 min) – BPE Schritt-für-Schritt, warum C++
4. **Training** (3 min) – Loss-Kurven, Datensatz, eigene Hardware
5. **Live Demo** (5 min) – C# Frontend, Prompts live
6. **Reflexion** (2 min) – Was funktioniert, was nicht
##### Visualisierungen
- [ ] Loss-Kurve (W&B Export)
- [ ] Attention-Heatmap für einen Demo-Prompt
- [ ] BPE-Merge Schritt-für-Schritt Diagramm
- [ ] Architektur-Diagramm der gesamten Pipeline
##### Demo-Prompts (vorher testen)
```
"def fibonacci(n):"
"# Bubble sort\ndef bubble_sort(arr):"
"class Stack:\n    def __init__(self):"
"def is_prime(n):\n    if n < 2:\n        return"
"# Count vowels\ndef count_vowels(s):"
```
---
## Meilensteine & Zeitplan Übersicht
| Woche | Person 1 | Person 2 | Gemeinsam |
|---|---|---|---|
| 1 | Python + CUDA Setup | Python + CUDA + pybind11 Setup | Interface-Dok, Repo |
| 2 | Python lernen, Dataset-Theorie | Python lernen, BPE Theorie, pybind11 Hello-World | Karpathy Video schauen |
| 3 | Dataset-Klasse | BPE C++ Kernalgorithmus | – |
| 4 | Preprocessing Pipeline | pybind11 Binding + Embeddings | – |
| 5 | LR Scheduler | Multi-Head Attention | – |
| 6 | Training Loop | Kompletter Transformer | – |
| 7 | Datensatz fertigstellen | ONNX Export vorbereiten | Ende-zu-Ende Test |
| 8 | – | – | Kleines Training (12M), Pipeline validieren |
| 9 | – | – | Großes Training (50M, overnight auf 3060) |
| 10 | C# Frontend starten | C++ Inference Engine | Evaluation + Checkpoint auswählen |
| 11 | C# Frontend fertig | Inference Engine fertig | Demo-Prompts testen |
| 12 | – | – | Präsentation |
---
## Wichtige Ressourcen
### Must-Read / Must-Watch
- **Andrej Karpathy – "Let's build GPT from scratch"** (YouTube, 2h) – wichtigste Ressource, beide schauen
- **"Attention is All You Need"** (Vaswani 2017) – Original Transformer Paper
- **"Neural Machine Translation of Rare Words with Subword Units"** (Sennrich 2016) – BPE Paper
### Dokumentation
- PyTorch Docs: https://pytorch.org/docs
- PyTorch Mixed Precision: https://pytorch.org/docs/stable/amp.html
- pybind11 Docs: https://pybind11.readthedocs.io
- ONNX Runtime C++ API: https://onnxruntime.ai/docs/api/c/
- AvalonEdit: https://avalonedit.net
- Weights & Biases: https://docs.wandb.ai
### Datensätze
- CodeSearchNet: `load_dataset("code_search_net", "python")` und `"java"`
- GitHub Gists API: https://api.github.com/gists/public
---
## Risiken & Fallbacks
| Risiko | Wahrscheinlichkeit | Fallback |
|---|---|---|
| Loss sinkt nicht | Mittel | LR halbieren, Batch Size via Accumulation verdoppeln |
| VRAM OOM auf 3060 | Mittel | Batch Size 4 + Accumulation Steps 32 |
| pybind11 Build schlägt fehl | Mittel | Tokenizer temporär in Python, C++ Version nachliefern |
| ONNX Export/Runtime Probleme | Mittel | Python FastAPI als Fallback für Demo |
| C# P/Invoke zur DLL buggy | Niedrig | C# ruft stattdessen Python FastAPI auf |
| GPU überhitzt overnight | Niedrig | MSI Afterburner Power Limit auf 80%, Temperatur vorher testen |
---
## Definition of Done
- [ ] `python training/train.py --config large` läuft stabil durch
- [ ] Trainiertes 50M Checkpoint + `model.onnx` existieren
- [ ] C++ CLI: `./inference "def fibonacci(n):"` gibt korrekten Code aus
- [ ] Mindestens 3 von 5 Test-Prompts → syntaktisch korrekter, sinnvoller Code
- [ ] C# Frontend startet, ruft C++ Inference auf, zeigt Output an
- [ ] Alle Komponenten haben README und Docstrings/Kommentare
- [ ] GitHub Repo sauber (keine Checkpoints, keine Rohdaten, kein Build-Output committed)
- [ ] Live-Demo mindestens zweimal komplett durchgespielt
