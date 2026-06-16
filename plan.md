# SLM Schulprojekt – Projektplan
> Small Language Model für Code-Generierung  
> Zwei Personen | HTL-Schulprojekt | Python + C#

---

## Scope

### Was dieses Projekt IST
- Ein Decoder-only Transformer (GPT-Architektur) von Grund auf implementiert
- Trainiert auf einem Code-Datensatz (Python/Java Snippets)
- Fähig, einfache vollständige Programme zu schreiben (Funktionen, Klassen, Algorithmen)
- Eigener BPE-Tokenizer, selbst implementiert
- C# Demo-Frontend für die Präsentation
- Lokal trainiert auf eigener Hardware (kein Cloud-Dienst nötig)

### Was dieses Projekt NICHT IST
- Kein GPT-4 Klon – das Modell wird klein sein (~50M Parameter)
- Kein Fine-Tuning von bestehenden Modellen (z.B. kein HuggingFace Pretrained)
- Keine Produktionsanwendung
- Kein vollständiger Code-Assistent – aber mehr als nur Snippet-Vervollständigung

### Erfolgskriterien
- [ ] Modell schreibt syntaktisch korrekte, einfache Programme (Fibonacci, Bubble Sort, Stack-Klasse, etc.)
- [ ] Komplette Pipeline: Datensatz → Tokenizer → Training → Inference
- [ ] Alles selbst implementiert (kein HuggingFace Transformers, kein Keras)
- [ ] Funktionierende Demo für die Präsentation
- [ ] Dokumentierter, sauberer Code auf GitHub

---

## Hardware & Training Setup

| Gerät | GPU | VRAM | Rolle |
|---|---|---|---|
| PC 1 | RTX 4060 | 8 GB | **Haupt-Trainingsmaschine** – 50M Modell mit Gradient Accumulation |
| PC 2 | RTX 3060 | 12 GB | Experimente, kleinere Runs, Hyperparameter-Suche |
| PC 3 | RTX 5060 | 8–12 GB | Zweite Experimentierkiste, Inference-Tests |

### CUDA Setup (alle Maschinen)
```bash
# PyTorch mit CUDA installieren (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Testen:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Modell-Konfiguration für 3060 (Haupt-Training, 12GB VRAM)
```python
@dataclass
class ModelConfig:
    vocab_size: int    = 16000
    context_length: int = 512
    d_model: int       = 512
    n_heads: int       = 8
    n_layers: int      = 8
    d_ff: int          = 2048
    dropout: float     = 0.1
    # → ca. 50M Parameter, passt in 12GB VRAM mit Batch Size 8–16
```

### Modell-Konfiguration für 4060/5060 (8GB VRAM, Experimente)
```python
@dataclass
class ModelConfigSmall:
    vocab_size: int    = 16000
    context_length: int = 256
    d_model: int       = 256
    n_heads: int       = 8
    n_layers: int      = 4
    d_ff: int          = 1024
    dropout: float     = 0.1
    # → ca. 12M Parameter, passt sicher in 8GB VRAM
```

### Erwartete Trainingszeit (3060, 12GB VRAM)
| Steps | Zeit (ca.) | Loss (ca.) |
|---|---|---|
| 1.000 | ~15 Min | ~6–7 |
| 10.000 | ~2.5 Std | ~3–4 |
| 50.000 | ~12 Std | ~2–2.5 |
| 100.000 | ~24 Std | ~1.8–2.2 |

Overnight-Training auf dem 3060. 4060/5060 für schnelle Experimente mit dem kleinen Modell (12M), da dort Batch Size kleiner sein muss.

---

## Tech Stack

| Komponente | Technologie | Warum |
|---|---|---|
| Tokenizer | Python | Muss mit Training-Code kompatibel sein |
| Transformer Core | Python + PyTorch | PyTorch nur für Autograd, keine fertigen Transformer-Klassen |
| Training | Python + PyTorch + CUDA | Lokale GPU-Training auf eigenem Rechner |
| Datensatz | Python | Preprocessing-Scripts |
| Demo Frontend | C# (WPF) | Person 1 Stärke, gut für Präsentation |
| Versionskontrolle | Git + GitHub | Kollaboration |
| Experiment-Tracking | Weights & Biases (kostenlos) | Loss-Kurven, Checkpoints, Vergleiche zwischen Runs |

---

## Projektstruktur (GitHub Repo)

```
slm-project/
├── tokenizer/           # Person 1
│   ├── bpe.py
│   ├── vocab.py
│   ├── encode.py
│   └── tests/
├── model/               # Person 2
│   ├── embedding.py
│   ├── attention.py
│   ├── transformer.py
│   └── config.py
├── training/            # Gemeinsam
│   ├── dataset.py
│   ├── train.py
│   ├── scheduler.py
│   └── checkpoints/    # .gitignore – zu groß für GitHub
├── inference/           # Gemeinsam
│   ├── generate.py
│   └── api.py
├── frontend/            # Person 1
│   └── SLMDemo/        # C# WPF Projekt
├── data/
│   ├── download.py     # Datensatz herunterladen
│   ├── preprocess.py   # Bereinigung & Tokenisierung
│   └── README.md
├── scripts/
│   ├── train_large.sh  # Training auf 3090
│   └── train_small.sh  # Training auf 3060/5060
├── docs/
│   └── interface.md    # Tokenizer↔Modell Vertrag
└── README.md
```

---

## Interface-Dokument (Woche 1 gemeinsam festlegen)

Dieser Vertrag muss VOR paralleler Arbeit stehen. Beide arbeiten danach unabhängig.

### Tokenizer Output-Format

```python
# encode(text: str) -> List[int]
tokenizer.encode("def foo():") 
# → [2, 145, 67, 233, 891, 1]

# decode(ids: List[int]) -> str
tokenizer.decode([2, 145, 67, 233, 891, 1])
# → "def foo():"

# Spezial-Token IDs
tokenizer.vocab_size    # 16000
tokenizer.pad_token_id  # 0
tokenizer.eos_token_id  # 1
tokenizer.bos_token_id  # 2
tokenizer.unk_token_id  # 3
```

### Mock-Tokenizer (Person 2 verwendet dies bis echter Tokenizer fertig)

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

---

## Phasenplan

---

### Phase 0 – Setup & Interface (Woche 1, BEIDE gemeinsam)

**Ziel:** Entwicklungsumgebung auf allen Maschinen läuft, Interface-Vertrag steht, Repo eingerichtet.

- [ ] GitHub Repo anlegen, Branch-Strategie: `main` (stabil), `dev`, Feature-Branches
- [ ] Python + CUDA PyTorch auf allen drei PCs installieren und testen
- [ ] VS Code + Pylance + Python Extension (Person 1 falls neu)
- [ ] `docs/interface.md` gemeinsam schreiben – Token-Format, encode/decode Signaturen
- [ ] `mock_tokenizer.py` committen damit Person 2 sofort starten kann
- [ ] Weights & Biases Account anlegen: https://wandb.ai (kostenlos)
- [ ] `.gitignore` für Checkpoints, `data/raw/`, `__pycache__` einrichten

**Deliverable:** Alle drei PCs können `python -c "import torch; print(torch.cuda.is_available())"` mit `True` ausgeben. Repo-Struktur steht.

---

### Phase 1 – Parallele Kernentwicklung (Woche 2–6)

---

#### Person 1 – Tokenizer

**Ziel:** Vollständiger, getesteter BPE-Tokenizer mit vocab_size=16000, trainiert auf Code-Datensatz.

##### Woche 2 – Python lernen + BPE Theorie

- [ ] Python Crashkurs: Listen, Dicts, Klassen, File I/O
  - Mit Java/C# Background reichen 3–4 Tage intensiv
  - Übung: kleines Script das eine Textdatei einliest und Wortfrequenzen zählt
- [ ] BPE-Algorithmus verstehen:
  - Paper lesen: Sennrich et al. 2016 "Neural Machine Translation of Rare Words with Subword Units" (4 Seiten)
  - Kernidee: Start mit Zeichenebene, häufigste benachbarte Paare iterativ mergen
- [ ] Mini-Beispiel manuell auf Papier:
  - Input: `["low", "low", "lower", "newer", "wider"]`
  - 5 BPE-Merge-Schritte von Hand, Vocab danach aufschreiben

##### Woche 3 – BPE Kernalgorithmus

- [ ] `tokenizer/bpe.py`:
  ```
  get_word_frequencies(corpus: List[str]) -> Dict[str, int]
    # "def foo" → {"d e f": 1, "f o o": 1}
  
  get_pair_frequencies(vocab: Dict) -> Dict[tuple, int]
    # Zählt alle benachbarten Zeichenpaare
  
  merge_pair(pair: tuple, vocab: Dict) -> Dict
    # Führt genau einen Merge-Schritt durch
  
  train_bpe(corpus: List[str], num_merges: int) -> List[tuple]
    # Gibt Liste aller Merge-Regeln zurück
  ```
- [ ] `tokenizer/vocab.py`:
  ```
  save_tokenizer(merges, vocab, path: str)
  load_tokenizer(path: str) -> (merges, vocab)
  ```
- [ ] Auf kleinem Testkorpus (~500 Zeilen Code) validieren

##### Woche 4 – Encode/Decode + Spezial-Tokens + Tests

- [ ] `tokenizer/encode.py`:
  ```
  apply_bpe(word: str, merges: List[tuple]) -> List[str]
    # Wendet Merge-Regeln auf ein Wort an
  
  encode(text: str, merges, vocab) -> List[int]
    # Vollständige Tokenisierung inkl. BOS/EOS
  
  decode(ids: List[int], vocab_inv: Dict) -> str
    # Token-IDs zurück zu Text
  ```
- [ ] Spezial-Tokens: `<pad>=0`, `<eos>=1`, `<bos>=2`, `<unk>=3`
- [ ] Code-spezifische Behandlung: Einrückungen (Spaces/Tabs) als eigene Tokens
- [ ] `tokenizer/tests/` – Unit Tests:
  - `encode(decode(ids)) == ids` (Roundtrip)
  - Unbekannte Zeichen → `<unk>`
  - Leerer String → `[2, 1]` (BOS, EOS)
  - Einrückung bleibt erhalten

##### Woche 5 – Training auf Code-Datensatz + Tokenizer-Klasse

- [ ] Tokenizer auf dem echten Code-Datensatz trainieren (16000 Merges)
  - Datensatz: CodeSearchNet Python + Java kombiniert (~800k Funktionen)
  - Training dauert ca. 30–60 Minuten auf CPU
- [ ] `Tokenizer`-Klasse als sauberes Interface:
  ```python
  class Tokenizer:
      def __init__(self, vocab_size=16000): ...
      def train(self, corpus: List[str]): ...
      def encode(self, text: str) -> List[int]: ...
      def decode(self, ids: List[int]) -> str: ...
      def save(self, path: str): ...
      @classmethod
      def load(cls, path: str) -> "Tokenizer": ...
  ```
- [ ] Mit Person 2 Integration testen: echter Tokenizer ersetzt Mock

##### Woche 6 – Performance & Dokumentation

- [ ] Encode-Performance: mindestens 5k Token/Sekunde (für Training-Durchsatz)
  - Falls zu langsam: häufigste Merges in einem Dict cachen
- [ ] Docstrings für alle Funktionen und Klassen
- [ ] `tokenizer/README.md` mit Beispielen

---

#### Person 2 – Transformer Core & Training Loop

**Ziel:** Vollständiger 50M Parameter Transformer, trainierbar auf 3090, mit Inference.

##### Woche 2 – Theorie & Embeddings

- [ ] "Attention is All You Need" (Vaswani 2017) lesen – Fokus auf Decoder-Only Variante
- [ ] Andrej Karpathy "Let's build GPT from scratch" schauen (YouTube, 2h) – wichtigste Ressource
- [ ] `model/config.py` – beide Config-Klassen (50M für 3090, 12M für 3060/5060)
- [ ] `model/embedding.py`:
  - Token Embedding: `nn.Embedding(vocab_size, d_model)`
  - Positional Encoding: **sinusoidale Variante selbst implementieren** (nicht `nn.Embedding`)
    ```python
    # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    ```
  - Shape-Test: Input `(batch=2, seq=512)` → Output `(batch=2, seq=512, d_model=512)`

##### Woche 3 – Multi-Head Attention

- [ ] `model/attention.py`:
  ```python
  def scaled_dot_product_attention(Q, K, V, mask=None):
      # scores = QK^T / sqrt(d_k)
      # scores = scores.masked_fill(mask, -inf)  # Causal mask
      # weights = softmax(scores)
      # return weights @ V
  
  class MultiHeadAttention(nn.Module):
      # __init__: Linear W_Q, W_K, W_V, W_O
      # forward: split in heads → attention → concat → project
  ```
- [ ] Causal Mask: oberes Dreieck der Attention-Matrix auf -inf setzen
  ```python
  mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
  ```
- [ ] Tests:
  - Input `(2, 512, 512)` → Output `(2, 512, 512)` ✓
  - Attention weights summieren zu 1 pro Position ✓
  - Kein Token sieht Zukunft (Causal Mask korrekt) ✓

##### Woche 4 – Kompletter Transformer Block + Modell

- [ ] `model/transformer.py`:
  ```python
  class FeedForward(nn.Module):
      # Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model) → Dropout
  
  class TransformerBlock(nn.Module):
      # Pre-LayerNorm Variante (stabiler als Post-LN):
      # x = x + Attention(LayerNorm(x))
      # x = x + FeedForward(LayerNorm(x))
  
  class GPTModel(nn.Module):
      # token_embedding + positional_encoding
      # N × TransformerBlock
      # Final LayerNorm
      # Linear(d_model, vocab_size)  ← kein Softmax hier, CrossEntropyLoss erwartet Logits
  ```
- [ ] Parameter zählen und ausgeben:
  ```python
  total = sum(p.numel() for p in model.parameters())
  print(f"Parameter: {total/1e6:.1f}M")  # Soll ~50M sein
  ```
- [ ] Forward Pass Test: zufälliger Input → Loss berechnen → kein NaN/Inf

##### Woche 5 – Training Loop

- [ ] `training/dataset.py`:
  ```python
  class CodeDataset(torch.utils.data.Dataset):
      # Lädt tokenisierte Sequenzen
      # __getitem__: gibt (input_ids[:-1], target_ids[1:]) zurück
      #              → Next-Token-Prediction
  ```
- [ ] `training/scheduler.py` – Cosine LR Schedule mit Warmup:
  ```python
  def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
      if step < warmup_steps:
          return max_lr * step / warmup_steps
      # Cosine decay danach
  ```
- [ ] `training/train.py`:
  ```python
  # AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
  # Gradient Clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  # Mixed Precision: torch.cuda.amp.autocast() + GradScaler  ← verdoppelt Trainingsgeschwindigkeit
  # Checkpoint alle 1000 Steps
  # W&B Logging: loss, lr, grad_norm
  ```
- [ ] Overfitting-Test: auf 20 Beispielen trainieren bis Loss < 0.1

##### Woche 6 – Inference + Integration vorbereiten

- [ ] `inference/generate.py`:
  ```python
  def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200,
               temperature: float = 0.8, top_k: int = 50) -> str:
      # Greedy: argmax(logits)
      # Temperature: logits / temperature vor Softmax
      # Top-K: nur die K wahrscheinlichsten Tokens samplen
  ```
- [ ] Top-K Sampling implementieren (wichtig für Code-Qualität, verhindert repetitive Loops)
- [ ] Mit Person 1: Mock-Tokenizer durch echten ersetzen, Ende-zu-Ende testen

---

### Phase 2 – Datensatz & Training (Woche 7–10, Gemeinsam)

**Ziel:** 50M Modell trainiert, generiert brauchbaren Code.

##### Datensatz aufbereiten (Woche 7)

- [ ] **Primär: CodeSearchNet** (Python + Java, ~800k Funktionen)
  ```python
  from datasets import load_dataset
  ds_py   = load_dataset("code_search_net", "python")
  ds_java = load_dataset("code_search_net", "java")
  ```
- [ ] **Ergänzend: GitHub Gists** – kurze vollständige Programme (Person 2 schreibt Scraper)
  - Ziel: ~50k vollständige kleine Programme (nicht nur Funktionen)
  - Wichtig für "schreibt einfache Programme" – Ziel
- [ ] Preprocessing (`data/preprocess.py`):
  - Duplikate entfernen
  - Dateien > 1024 Token abschneiden oder aufteilen
  - Syntaktisch ungültigen Code filtern (`ast.parse()` für Python)
  - Train/Val Split: 95% / 5%
- [ ] Tokenizer auf diesem Datensatz trainieren (Person 1)
- [ ] Alles zu `.bin` Datei vorverarbeiten für schnelles Laden während Training

##### Erstes Training – klein (Woche 8)

- [ ] 12M Modell auf 4060 oder 5060 starten, 10.000 Steps
  - Ziel: Pipeline validieren, Loss beobachten
  - Loss soll von ~9.7 (= log2(16000)) auf ~3–4 fallen
- [ ] W&B Dashboard aufsetzen: Loss-Kurve, LR-Kurve, Grad-Norm live sehen
- [ ] Erste manuelle Inference nach 5000 Steps:
  - `"def add(a, b):"` → gibt es irgendwas Sinnvolles?

##### Großes Training – 50M auf 3090 (Woche 9)

- [ ] `scripts/train_large.sh`:
  ```bash
  python training/train.py \
      --config large \
      --data data/processed/train.bin \
      --batch_size 8 \
      --accumulation_steps 16 \
      --steps 100000 \
      --checkpoint_dir training/checkpoints/run_001
  # Effektiver Batch = 8 × 16 = 128, passt in 12GB VRAM
  ```
- [ ] Overnight auf 3060 laufen lassen (~24 Stunden für 100k Steps)
- [ ] Checkpoint alle 5000 Steps auf lokaler Festplatte speichern
- [ ] Gradient Accumulation falls Batch Size zu groß für VRAM:
  ```python
  # Effektiver Batch = batch_size × accumulation_steps
  accumulation_steps = 4  # effektiv batch_size=128 mit batch_size=32
  ```

##### Evaluation & Tuning (Woche 10)

- [ ] Checkpoint bei niedrigstem Val-Loss auswählen
- [ ] Manuelle Evaluation mit diesen Prompts:
  ```python
  "def fibonacci(n):"
  "# Bubble sort implementation\ndef bubble_sort(arr):"
  "class Stack:\n    def __init__(self):\n        self."
  "def is_prime(n):\n    "
  "# Read a file and count words\n"
  ```
- [ ] Qualitätsbewertung (0–3 Punkte pro Prompt):
  - 0: Nonsense
  - 1: Syntaktisch korrekt aber logisch falsch
  - 2: Syntaktisch + logisch korrekt aber unvollständig
  - 3: Funktionierendes Programm
- [ ] Falls Durchschnitt < 1.5: Hyperparameter tunen, nochmal trainieren
- [ ] Falls gut: weiter zu Phase 3

---

### Phase 3 – Demo Frontend (Woche 10–11, Person 1)

**Ziel:** C# WPF Anwendung als polierte Demo für die Präsentation.

##### Architektur

```
C# WPF  →  HTTP POST localhost:8000/generate  →  Python FastAPI  →  Modell (GPU)
        ←  { "output": "...", "tokens_per_sec": 42 }          ←
```

##### Python API (`inference/api.py`, Person 2 hilft)

```python
# pip install fastapi uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50

@app.post("/generate")
def generate(req: GenerateRequest):
    output = model_generate(req.prompt, req.max_tokens, req.temperature, req.top_k)
    return {"output": output}

# Start: uvicorn inference.api:app --port 8000
```

##### C# WPF Frontend (Person 1)

- [ ] Neues WPF Projekt in Visual Studio
- [ ] NuGet Packages: `AvalonEdit` (Syntax Highlighting), `Newtonsoft.Json`
- [ ] UI-Elemente:
  - Prompt-Eingabe (AvalonEdit mit Python-Syntax-Highlighting)
  - "Generate" Button
  - Output-Feld (AvalonEdit, read-only)
  - Temperature Slider (0.1 – 1.5, Default 0.8)
  - Max Tokens Eingabe (Default 200)
  - Top-K Slider (1 – 100, Default 50)
  - Statuszeile: "Generating..." / "Done (42 tok/s)"
- [ ] HTTP-Aufruf:
  ```csharp
  var client = new HttpClient();
  var body = JsonConvert.SerializeObject(new { prompt, max_tokens, temperature, top_k });
  var response = await client.PostAsync("http://localhost:8000/generate",
                     new StringContent(body, Encoding.UTF8, "application/json"));
  var result = JsonConvert.DeserializeObject<GenerateResult>(await response.Content.ReadAsStringAsync());
  OutputEditor.Text = result.Output;
  ```
- [ ] Dark Theme (passt zu Code-Editor)
- [ ] Copy-Button für Output

---

### Phase 4 – Präsentation (Woche 11–12, Gemeinsam)

**Ziel:** Überzeugende Präsentation mit Live-Demo, die zeigt dass ihr das System wirklich versteht.

##### Struktur der Präsentation (~20 Minuten)

1. **Motivation** (2 min) – Was ist ein LLM? Wie funktioniert ChatGPT grundsätzlich?
2. **Architektur** (5 min) – Transformer-Diagramm, Attention erklärt
3. **Tokenizer** (3 min) – BPE Schritt-für-Schritt mit Beispiel aus unserem Datensatz
4. **Training** (3 min) – Loss-Kurven zeigen, Datensatz beschreiben, Hardware erklären
5. **Live Demo** (5 min) – C# Frontend, Prompts live eingeben, Output zeigen
6. **Reflexion** (2 min) – Was funktioniert, was nicht, was würden wir anders machen

##### Visualisierungen vorbereiten

- [ ] Loss-Kurve (W&B Screenshot oder matplotlib Export)
- [ ] Attention-Heatmap für einen Demo-Prompt:
  ```python
  # Attention weights aus model.blocks[0].attn extrahieren
  # Als Heatmap mit seaborn plotten
  ```
- [ ] BPE-Merge Prozess als Schritt-für-Schritt Diagramm (z.B. in PowerPoint)
- [ ] Modell-Architektur Diagramm

##### Demo-Prompts (vorher testen und die besten auswählen)

```python
"def fibonacci(n):"
"# Bubble sort\ndef bubble_sort(arr):"
"class Stack:\n    def __init__(self):"
"def is_prime(n):\n    if n < 2:\n        return"
"# Count vowels in a string\ndef count_vowels(s):"
```

---

## Meilensteine & Zeitplan Übersicht

| Woche | Person 1 | Person 2 | Gemeinsam |
|---|---|---|---|
| 1 | Python + CUDA Setup | CUDA Setup | Interface-Dok, Repo-Struktur |
| 2 | Python lernen, BPE Theorie | Embeddings, beide Configs | – |
| 3 | BPE Kernalgorithmus | Multi-Head Attention | – |
| 4 | Encode/Decode + Tests | Transformer Block + Modell | – |
| 5 | Tokenizer-Klasse | Training Loop | Integration testen |
| 6 | Performance + Docs | Inference/generate.py | Ende-zu-Ende testen |
| 7 | Datensatz Preprocessing | GitHub Gist Scraper | Datensatz finalisieren |
| 8 | Tokenizer auf Datensatz trainieren | 12M Modell auf 4060/5060 starten | Pipeline validieren |
| 9 | – | 50M Modell auf 3060 (overnight) | W&B Dashboard |
| 10 | C# Frontend starten | Evaluation + Tuning | Beste Checkpoints auswählen |
| 11 | C# Frontend fertig | FastAPI Server | Demo-Prompts testen |
| 12 | – | – | Präsentation |

---

## Wichtige Ressourcen

### Must-Read / Must-Watch
- **Andrej Karpathy – "Let's build GPT from scratch"** (YouTube, 2h) – wichtigste Ressource überhaupt
- **"Attention is All You Need"** (Vaswani et al. 2017) – Original Transformer Paper
- **"Neural Machine Translation of Rare Words with Subword Units"** (Sennrich 2016) – BPE Paper

### Dokumentation
- PyTorch Docs: https://pytorch.org/docs
- PyTorch Mixed Precision: https://pytorch.org/docs/stable/amp.html
- FastAPI Docs: https://fastapi.tiangolo.com
- AvalonEdit: https://avalonedit.net
- Weights & Biases: https://docs.wandb.ai

### Datensätze
- CodeSearchNet: `load_dataset("code_search_net", "python")` und `"java"`
- GitHub Gists API: https://api.github.com/gists/public

---

## Risiken & Fallbacks

| Risiko | Wahrscheinlichkeit | Fallback |
|---|---|---|
| Loss sinkt nicht | Mittel | LR halbieren (3e-4 → 1.5e-4), Batch Size verdoppeln via Gradient Accumulation |
| VRAM Out-of-Memory auf 3060 | Mittel | Batch Size auf 4 reduzieren, Accumulation Steps auf 32 erhöhen |
| Tokenizer zu langsam für Training | Mittel | Datensatz komplett vorverarbeiten und als .bin speichern |
| Modell generiert Nonsense | Mittel | Mehr Steps trainieren, Datensatz-Qualität prüfen |
| C# Frontend nicht rechtzeitig | Niedrig | Terminal-Demo reicht für Präsentation, Frontend als "in Arbeit" zeigen |
| GPU überhitzt bei Overnight-Training | Niedrig | GPU-Temperatur vorher testen, MSI Afterburner Power Limit auf 80% |

---

## Definition of Done

Das Projekt ist abgeschlossen wenn:

- [ ] `python training/train.py --config large` läuft stabil durch
- [ ] Trainiertes 50M Checkpoint existiert und lädt korrekt
- [ ] `python inference/generate.py --prompt "def fibonacci(n):"` gibt korrekten Python-Code aus
- [ ] Mindestens 3 von 5 Test-Prompts erzeugen syntaktisch korrekten, sinnvollen Code
- [ ] C# Frontend startet, verbindet sich mit Python API, zeigt Output an
- [ ] Alle Komponenten haben README und Docstrings
- [ ] GitHub Repo ist sauber (keine Checkpoints, keine Rohdaten, kein API Key committed)
- [ ] Präsentation ist fertig, Live-Demo wurde mindestens zweimal durchgespielt
