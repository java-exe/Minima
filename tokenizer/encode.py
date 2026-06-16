"""Encode/Decode + Spezial-Tokens (Person 1, Woche 4).

Spezial-Tokens: <pad>=0, <eos>=1, <bos>=2, <unk>=3.
Code-spezifisch: Einrueckungen (Spaces/Tabs) als eigene Tokens.
Siehe plan.md, Phase 1, Woche 4.
"""

from typing import Dict, List, Tuple


def apply_bpe(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Wendet die Merge-Regeln auf ein Wort an."""
    raise NotImplementedError


def encode(text: str, merges: List[Tuple[str, str]], vocab: Dict) -> List[int]:
    """Vollstaendige Tokenisierung inkl. BOS/EOS."""
    raise NotImplementedError


def decode(ids: List[int], vocab_inv: Dict) -> str:
    """Token-IDs zurueck zu Text."""
    raise NotImplementedError
