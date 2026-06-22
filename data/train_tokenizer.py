"""
Trainiert den BPE-Tokenizer auf den heruntergeladenen Rohdaten
und speichert vocab.bin.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tokenizer_cpp
import argparse
from pathlib import Path

import tokenizer_cpp


def load_corpus(txt_path: str, limit: int = None) -> list[str]:
    print(f"Lade Corpus: {txt_path}")
    samples = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # \\n zurück zu echten Newlines
                samples.append(line.replace("\\n", "\n"))
            if limit and len(samples) >= limit:
                break
    print(f"  → {len(samples):,} Funktionen geladen")
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus",      type=str, default="data/raw/combined.txt")
    p.add_argument("--vocab_out",   type=str, default="tokenizer/vocab.bin")
    p.add_argument("--num_merges",  type=int, default=15_744,
                   help="Anzahl BPE-Merges. 4 Spezial + 256 Basis + 15744 Merges = 16004 ≈ 16k vocab")
    p.add_argument("--limit",       type=int, default=None,
                   help="Nur N Samples verwenden (zum Testen, z.B. 10000)")
    args = p.parse_args()

    Path(args.vocab_out).parent.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.corpus, limit=args.limit)

    print(f"\nTrainiere BPE Tokenizer...")
    print(f"  Corpus: {len(corpus):,} Funktionen")
    print(f"  Merges: {args.num_merges:,}")
    print(f"  Ziel vocab_size: ~{4 + 256 + args.num_merges:,}")

    tok = tokenizer_cpp.Tokenizer()
    tok.train(corpus, args.num_merges)

    print(f"\nVocab size nach Training: {tok.vocab_size}")

    tok.save(args.vocab_out)
    print(f"Gespeichert: {args.vocab_out}")

    # Schneller Smoke-Test
    print(f"\nSmoke-Test:")
    test_code = "def fibonacci(n):\n    if n <= 1:\n        return n"
    ids = tok.encode(test_code)
    decoded = tok.decode(ids)
    print(f"  Input:   {repr(test_code)}")
    print(f"  IDs:     {ids[:10]}... ({len(ids)} tokens)")
    print(f"  Decoded: {repr(decoded)}")
    print(f"  Roundtrip OK: {test_code == decoded}")


if __name__ == "__main__":
    main()