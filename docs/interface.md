# Interface-Vertrag: Tokenizer ↔ Modell

> Dieser Vertrag muss **vor** der parallelen Arbeit stehen. Danach arbeiten
> beide Personen unabhaengig. Siehe plan.md, Phase 0.

## Tokenizer Output-Format

```python
# encode(text: str) -> List[int]
tokenizer.encode("def foo():")
# -> [2, 145, 67, 233, 891, 1]

# decode(ids: List[int]) -> str
tokenizer.decode([2, 145, 67, 233, 891, 1])
# -> "def foo():"
```

## Spezial-Token IDs

| Token   | ID | Attribut                 |
|---------|----|--------------------------|
| `<pad>` | 0  | `tokenizer.pad_token_id` |
| `<eos>` | 1  | `tokenizer.eos_token_id` |
| `<bos>` | 2  | `tokenizer.bos_token_id` |
| `<unk>` | 3  | `tokenizer.unk_token_id` |

`tokenizer.vocab_size == 16000`

## Tokenizer-Klasse (Ziel-Interface, Person 1)

```python
class Tokenizer:
    def __init__(self, vocab_size=16000): ...
    def train(self, corpus: list[str]): ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def save(self, path: str): ...
    @classmethod
    def load(cls, path: str) -> "Tokenizer": ...
```

## Mock-Tokenizer

Bis der echte Tokenizer fertig ist, nutzt Person 2 den
[`MockTokenizer`](../tokenizer/mock_tokenizer.py), der exakt dieses Interface
erfuellt.

## Regeln

- Jede Sequenz beginnt mit `<bos>` (2) und endet mit `<eos>` (1).
- Leerer String -> `[2, 1]`.
- Unbekannte Zeichen -> `<unk>` (3).
- Einrueckungen (Spaces/Tabs) sind eigene Tokens und bleiben erhalten.
- `encode(decode(ids)) == ids` (Roundtrip muss halten).
