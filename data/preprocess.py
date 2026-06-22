"""
Preprocessing Pipeline: CodeSearchNet → tokenisierte .bin Datei
- Duplikate entfernen (MD5)
- Python-Code syntaktisch validieren (ast.parse)
- Jede Funktion einzeln encodieren: [bos] code [eos]
- Sequenzen > context_length werden aufgeteilt, nicht verworfen
- Train/Val Split 95%/5%
- Output: numpy memmap .bin (int32)
"""

import ast
import hashlib
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
# Tokenizer einbinden - falls noch nicht gebaut: Mock verwenden
try:
    import tokenizer_cpp
    def load_tokenizer(vocab_path: str):
        return tokenizer_cpp.Tokenizer(vocab_path)
except ImportError:
    print("[WARN] tokenizer_cpp nicht gefunden, verwende MockTokenizer")
    import random
    class MockTokenizer:
        vocab_size    = 16000
        pad_token_id  = 0
        eos_token_id  = 1
        bos_token_id  = 2
        unk_token_id  = 3
        def encode(self, text):
            return [self.bos_token_id] + \
                   [random.randint(4, self.vocab_size - 1) for _ in text.split()] + \
                   [self.eos_token_id]
        def decode(self, ids):
            return "<mock>"
    def load_tokenizer(vocab_path: str):
        return MockTokenizer()


# ─────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────

def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def chunk_ids(ids: list[int], context_length: int,
              pad_id: int) -> list[list[int]]:
    """
    Teilt eine lange ID-Sequenz in Chunks der Länge context_length auf.
    Letzter Chunk wird mit pad_id aufgefüllt falls zu kurz.
    """
    chunks = []
    for i in range(0, len(ids), context_length):
        chunk = ids[i : i + context_length]
        if len(chunk) < context_length:
            chunk += [pad_id] * (context_length - len(chunk))
        chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────
# Datensatz laden & filtern
# ─────────────────────────────────────────────

def load_code_samples(languages: list[str]) -> list[str]:
    """
    Liest bereits heruntergeladene Daten aus data/raw/ statt nochmal HuggingFace zu laden.
    """
    samples = []
    for lang in languages:
        path = Path(f"data/raw/{lang}.txt")
        if not path.exists():
            print(f"  [WARN] {path} nicht gefunden, überspringe {lang}")
            continue
        print(f"  Lade {path}...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(line.replace("\\n", "\n"))
        print(f"  → {len(samples):,} Samples geladen")
    return samples


def deduplicate(samples: list[str]) -> list[str]:
    seen = set()
    unique = []
    for s in samples:
        h = md5(s)
        if h not in seen:
            seen.add(h)
            unique.append(s)
    return unique


def validate_python(samples: list[str], language: str) -> list[str]:
    """Syntaxcheck nur für Python-Code."""
    if language != "python":
        return samples
    valid = [s for s in samples if is_valid_python(s)]
    print(f"  Syntaxcheck: {len(valid)}/{len(samples)} gültig")
    return valid


# ─────────────────────────────────────────────
# Tokenisieren & schreiben
# ─────────────────────────────────────────────

def tokenize_samples(
    samples: list[str],
    tokenizer,
    context_length: int,
) -> list[int]:
    """
    Option A: jede Funktion einzeln encodieren.
    encode() fügt [bos] und [eos] automatisch hinzu.
    Lange Sequenzen werden in Chunks aufgeteilt.
    Gibt flache Liste aller Token-IDs zurück.
    """
    all_ids = []
    skipped = 0

    for i, code in enumerate(samples):
        if i % 10_000 == 0:
            print(f"  Tokenisiere {i}/{len(samples)}...")

        ids = tokenizer.encode(code)

        # Sequenz die kürzer als 8 Token ist (nur bos+eos + wenige Tokens)
        # ist zu kurz um sinnvolle Batches zu ergeben
        if len(ids) < 8:
            skipped += 1
            continue

        # Lange Sequenzen aufteilen statt verwerfen
        if len(ids) > context_length:
            chunks = chunk_ids(ids, context_length, tokenizer.pad_token_id)
            for chunk in chunks:
                all_ids.extend(chunk)
        else:
            all_ids.extend(ids)

    print(f"  Skipped (zu kurz): {skipped}")
    return all_ids


def write_bin(ids: list[int], path: str) -> None:
    arr = np.array(ids, dtype=np.int32)
    out = np.memmap(path, dtype=np.int32, mode="w+", shape=(len(arr),))
    out[:] = arr[:]
    out.flush()
    print(f"  Gespeichert: {path} ({len(arr):,} tokens, "
          f"{len(arr) * 4 / 1024**2:.1f} MB)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--vocab",           type=str, default="tokenizer/vocab.bin")
    p.add_argument("--output_dir",      type=str, default="data/processed")
    p.add_argument("--context_length",  type=int, default=512)
    p.add_argument("--languages",       type=str, nargs="+",
                   default=["python", "java"],
                   help="CodeSearchNet Sprachen")
    p.add_argument("--val_ratio",       type=float, default=0.05)
    p.add_argument("--limit",           type=int, default=None,
                   help="Nur N Samples verarbeiten (zum Testen)")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer laden
    tokenizer = load_tokenizer(args.vocab)
    print(f"Tokenizer geladen | vocab_size={tokenizer.vocab_size}")

    # Samples laden
    print("\n[1/4] Lade Datensatz...")
    samples = load_code_samples(args.languages)
    print(f"  Gesamt: {len(samples):,} Samples")

    if args.limit:
        samples = samples[: args.limit]
        print(f"  Limitiert auf {args.limit} Samples (--limit Flag)")

    # Deduplizieren
    print("\n[2/4] Dedupliziere...")
    samples = deduplicate(samples)
    print(f"  Nach Deduplikation: {len(samples):,} Samples")

    # Syntaxcheck (nur Python)
    if "python" in args.languages:
        print("\n[3/4] Syntaxcheck Python...")
        # Nur Python-Samples filtern
        py_samples  = [s for s in samples if is_valid_python(s)]
        other       = [s for s in samples if not is_valid_python(s)]
        # Andere Sprachen brauchen keinen ast.parse Check
        samples = py_samples + other
        print(f"  Nach Syntaxcheck: {len(samples):,} Samples")
    else:
        print("\n[3/4] Syntaxcheck übersprungen (kein Python)")

    # Train/Val Split (vor dem Tokenisieren, auf Sample-Ebene)
    print("\n[4/4] Tokenisiere & speichere...")
    n_val   = max(1, int(len(samples) * args.val_ratio))
    n_train = len(samples) - n_val

    train_samples = samples[:n_train]
    val_samples   = samples[n_train:]
    print(f"  Train: {len(train_samples):,} | Val: {len(val_samples):,}")

    train_ids = tokenize_samples(train_samples, tokenizer, args.context_length)
    val_ids   = tokenize_samples(val_samples,   tokenizer, args.context_length)

    write_bin(train_ids, f"{args.output_dir}/train.bin")
    write_bin(val_ids,   f"{args.output_dir}/val.bin")

    print(f"\nFertig.")
    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Val tokens:   {len(val_ids):,}")
    print(f"  Vocab size:   {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()