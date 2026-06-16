# Interface-Vertrag

> Diese Verträge müssen **vor** der parallelen Arbeit stehen. Danach arbeiten
> beide Personen unabhängig. Siehe plan.md, Phase 0.

## 1. Tokenizer ↔ Training (Python, via pybind11)

Der C++-Tokenizer wird mit pybind11 als Python-Modul `tokenizer_cpp` gebaut.

```python
import tokenizer_cpp

tok = tokenizer_cpp.Tokenizer("path/to/vocab.bin")
tok.encode("def foo():")       # -> [2, 145, 67, 233, 891, 1]
tok.decode([2, 145, 67, 233])  # -> "def foo"

tok.vocab_size    # 16000
tok.pad_token_id  # 0
tok.eos_token_id  # 1
tok.bos_token_id  # 2
tok.unk_token_id  # 3
```

### Spezial-Token IDs

| Token   | ID | Attribut          |
|---------|----|-------------------|
| `<pad>` | 0  | `tok.pad_token_id`|
| `<eos>` | 1  | `tok.eos_token_id`|
| `<bos>` | 2  | `tok.bos_token_id`|
| `<unk>` | 3  | `tok.unk_token_id`|

### Regeln

- Jede Sequenz beginnt mit `<bos>` (2) und endet mit `<eos>` (1).
- Unbekannte Zeichen -> `<unk>` (3).
- Einrückungen (Spaces/Tabs) sind eigene Tokens und bleiben erhalten.
- `encode(decode(ids)) == ids` (Roundtrip muss halten).

### Mock-Tokenizer

Bis das C++-Modul fertig ist, nutzt Person 1 den
[`MockTokenizer`](../training/mock_tokenizer.py), der exakt dieses Interface
erfüllt. In Woche 7 wird er durch `import tokenizer_cpp` ersetzt.

## 2. Modell ↔ Inference (ONNX)

Nach dem Training wird das Modell via [export_onnx.py](../inference/export_onnx.py)
nach `model.onnx` exportiert:

- **Input:** `input_ids`, Shape `(1, seq_len)`, dtype `int64`, dynamische Achse `seq_len`
- **Output:** `logits`, Shape `(1, seq_len, vocab_size)`

## 3. Inference Engine ↔ Frontend (C++ -> C#)

```
C# Frontend
  -> P/Invoke oder named pipe
  -> inference.dll (C++ onnxruntime)
  -> model.onnx + vocab.bin
  -> generierter Code (string zurück)
```

C#-seitige Signatur (P/Invoke):

```csharp
[DllImport("inference.dll")]
private static extern IntPtr generate(
    string prompt, int maxTokens, float temperature, int topK);
```
