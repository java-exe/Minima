"""Dataset fuer Next-Token-Prediction (Person 2, Woche 5).

Siehe plan.md, Phase 1, Woche 5.
"""

import torch


class CodeDataset(torch.utils.data.Dataset):
    """Laedt tokenisierte Sequenzen.

    __getitem__ gibt (input_ids[:-1], target_ids[1:]) zurueck
    -> Next-Token-Prediction.
    """

    def __init__(self, path: str, context_length: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError
