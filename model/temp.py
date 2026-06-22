
import sys
sys.path.insert(0, r"C:\Users\janst\OneDrive\Dokumente\Minima")
from model.config import ModelConfig
from model.transformer import GPTModel

cfg = ModelConfig()
model = GPTModel(cfg)
n = sum(p.numel() for p in model.parameters()) / 1e6
print(f"{n:.1f}M Parameter")
n = sum(p.numel() for p in model.parameters()) / 1e6
n_tied = model.embedding.token_embedding.embedding.weight.numel() / 1e6
print(f"{n:.1f}M Parameter (mit Weight Tying)")
print(f"{n + n_tied:.1f}M Parameter (ohne Weight Tying)")