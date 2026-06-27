import sys
import os
sys.path.insert(0, r"H:\HTL\Minima")
os.chdir(r"H:\HTL\Minima")

import torch
import tokenizer_cpp
from model.config import ModelConfig        # Large nicht Small
from model.transformer import GPTModel

device = torch.device("cuda")
cfg = ModelConfig()                          # Large Config
model = GPTModel(cfg).to(device)

ckpt = torch.load(r"training/checkpoints/run_large_001/ckpt_step_0036000.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tok = tokenizer_cpp.Tokenizer(r"tokenizer/vocab.bin")

prompts = [
    "def fibonacci(n):",
    "def bubble_sort(arr):",
    "class Stack:\n    def __init__(self):",
    "def is_prime(n):",
]

for prompt in prompts:
    ids = tok.encode(prompt)
    # encode() hängt automatisch <eos> an — vor der Generierung entfernen,
    # sonst gilt der Prompt als "fertig" und das Modell generiert unkonditioniert.
    if ids and ids[-1] == tok.eos_token_id:
        ids = ids[:-1]
    input_ids = torch.tensor([ids]).to(device)
    output = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
    print(f"--- {prompt[:20]} ---")
    print(tok.decode(output[0].tolist()))
    print()