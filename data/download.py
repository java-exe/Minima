
import argparse
from pathlib import Path
from datasets import load_dataset


def download_split(language: str, split: str) -> list[str]:
    print(f"  Lade {language}/{split}...")
    ds = load_dataset("code-search-net/code_search_net", split=split)
    samples = []
    for row in ds:
        if row.get("language", "") != language:
            continue
        code = row.get("whole_func_string") or row.get("func_code_string", "")
        if code and code.strip():
            samples.append(code)
    print(f"  → {len(samples):,} Samples")
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="data/raw")
    p.add_argument("--languages",  type=str, nargs="+", default=["python", "java"])
    p.add_argument("--splits",     type=str, nargs="+", default=["train", "validation", "test"])
    p.add_argument("--limit",      type=int, default=None,
                   help="Nur N Samples pro Sprache (zum Testen)")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_samples = []

    for lang in args.languages:
        print(f"\n[{lang}]")
        lang_samples = []
        for split in args.splits:
            lang_samples.extend(download_split(lang, split))

        if args.limit:
            lang_samples = lang_samples[: args.limit]

        out_path = Path(args.output_dir) / f"{lang}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in lang_samples:
                f.write(sample.replace("\n", "\\n") + "\n")

        print(f"  Gespeichert: {out_path} ({len(lang_samples):,} Funktionen)")
        all_samples.extend(lang_samples)

    combined_path = Path(args.output_dir) / "combined.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(sample.replace("\n", "\\n") + "\n")

    print(f"\nGesamt: {len(all_samples):,} Funktionen")
    print(f"Kombiniert: {combined_path}")


if __name__ == "__main__":
    main()