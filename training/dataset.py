import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class CodeDataset(Dataset):
  

    def __init__(self, bin_path: str, context_length: int = 512):
        self.context_length = context_length

        path = Path(bin_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset nicht gefunden: {bin_path}")


        self.data = np.memmap(bin_path, dtype=np.int32, mode="r")

     
        self.n_samples = (len(self.data) - 1) // context_length

        if self.n_samples == 0:
            raise ValueError(f"Datei zu klein für context_length={context_length}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.context_length

        chunk = self.data[start : start + self.context_length + 1]
        chunk = torch.from_numpy(chunk.astype(np.int64))

        input_ids  = chunk[:-1] 
        target_ids = chunk[1:]   

        return input_ids, target_ids