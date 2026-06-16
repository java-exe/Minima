"""Text-/Code-Generierung (Person 2, Woche 6).

Greedy / Temperature / Top-K Sampling. Top-K ist wichtig fuer Code-Qualitaet
(verhindert repetitive Loops). Siehe plan.md, Phase 1, Woche 6.
"""


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 50) -> str:
    # Greedy: argmax(logits)
    # Temperature: logits / temperature vor Softmax
    # Top-K: nur die K wahrscheinlichsten Tokens samplen
    raise NotImplementedError
