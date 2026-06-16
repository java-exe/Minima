"""FastAPI-Server fuer das C#-Frontend (Phase 3).

Start: uvicorn inference.api:app --port 8000
Siehe plan.md, Phase 3.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50


@app.post("/generate")
def generate(req: GenerateRequest):
    # TODO: echtes Modell laden und generate.generate(...) aufrufen
    raise NotImplementedError
