import sys
sys.path.insert(0, r"C:\Users\janst\OneDrive\Dokumente\Minima")

import torch
import tokenizer_cpp
from model.config import ModelConfigSmall
from model.transformer import GPTModel

device = torch.device("cuda")
cfg = ModelConfigSmall()
model = GPTModel(cfg).to(device)

ckpt = torch.load(r"training/checkpoints/run_small_001/ckpt_step_0010000.pt")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tok = tokenizer_cpp.Tokenizer(r"tokenizer/vocab.bin")

prompt = "def fibonacci(n):"
ids = tok.encode(prompt)
input_ids = torch.tensor([ids]).to(device)

output = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
print(tok.decode(output[0].tolist()))