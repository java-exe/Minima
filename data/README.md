# Daten

Scripts und Hinweise zum Datensatz. Die eigentlichen Daten werden **nicht**
ins Repo committet (siehe `.gitignore`: `data/raw/`, `data/processed/`).

## Pipeline

1. `python data/download.py` – Datensatz herunterladen
   - Primaer: CodeSearchNet (Python + Java, ~800k Funktionen)
   - Ergaenzend: GitHub Gists (kurze vollstaendige Programme)
2. `python data/preprocess.py` – Bereinigung & Tokenisierung
   - Duplikate entfernen, zu lange Dateien splitten
   - Ungueltigen Code filtern (`ast.parse()`)
   - Train/Val Split 95% / 5%
   - Export als `.bin` fuer schnelles Laden im Training

## Ordner (lokal, nicht im Git)

```
data/
├── raw/         # rohe heruntergeladene Daten
└── processed/   # train.bin / val.bin
```

## Quellen

- CodeSearchNet: `load_dataset("code_search_net", "python")` / `"java"`
- GitHub Gists API: https://api.github.com/gists/public
