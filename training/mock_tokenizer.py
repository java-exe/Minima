"""Mock-Tokenizer (Phase 0).

Stand-in fuer den echten C++-Tokenizer (tokenizer_cpp), damit Person 1 sofort
am Training-Loop und der Dataset-Pipeline arbeiten kann, bevor das pybind11-
Modul fertig ist. Implementiert exakt das Interface aus docs/interface.md.
Wird in Woche 7 durch `import tokenizer_cpp` ersetzt.
Siehe plan.md, Phase 0 + Interface-Dokument.
"""

import random
from typing import List


class MockTokenizer:
    def __init__(self):
        self.vocab_size = 16000
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3

    def encode(self, text: str) -> List[int]:
        return (
            [self.bos_token_id]
            + [random.randint(4, self.vocab_size - 1) for _ in text.split()]
            + [self.eos_token_id]
        )

    def decode(self, ids: List[int]) -> str:
        return "<mock decoded text>"
