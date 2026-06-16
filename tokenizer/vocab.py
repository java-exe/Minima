"""Vokabular speichern/laden (Person 1, Woche 3).

Siehe plan.md, Phase 1, Woche 3.
"""

from typing import Dict, List, Tuple


def save_tokenizer(merges: List[Tuple[str, str]], vocab: Dict, path: str) -> None:
    raise NotImplementedError


def load_tokenizer(path: str) -> Tuple[List[Tuple[str, str]], Dict]:
    raise NotImplementedError
