"""BPE-Kernalgorithmus (Person 1, Woche 3).

Implementiert den Byte-Pair-Encoding-Algorithmus von Grund auf:
Start auf Zeichenebene, haeufigste benachbarte Paare iterativ mergen.
Siehe plan.md, Phase 1, Woche 3.
"""

from typing import Dict, List, Tuple


def get_word_frequencies(corpus: List[str]) -> Dict[str, int]:
    """ "def foo" -> {"d e f": 1, "f o o": 1} """
    raise NotImplementedError


def get_pair_frequencies(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Zaehlt alle benachbarten Zeichenpaare."""
    raise NotImplementedError


def merge_pair(pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
    """Fuehrt genau einen Merge-Schritt durch."""
    raise NotImplementedError


def train_bpe(corpus: List[str], num_merges: int) -> List[Tuple[str, str]]:
    """Gibt die Liste aller Merge-Regeln zurueck."""
    raise NotImplementedError
