"""Modell -> ONNX exportieren (Gemeinsam, Woche 7).

Nach dem Training: trainiertes GPTModel laden und als model.onnx exportieren,
damit die C++ Inference Engine es ohne Python laden kann.
Siehe plan.md, Phase 1, Woche 7.
"""

import torch


def export(model: torch.nn.Module, path: str = "model.onnx", seq_len: int = 512) -> None:
    model.eval()
    dummy_input = torch.zeros(1, seq_len, dtype=torch.long)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "seq_len"}},
    )


def main():
    # TODO: Checkpoint laden, Modell rekonstruieren, export(...) aufrufen,
    #       danach mit onnxruntime in Python als Sanity-Check testen.
    raise NotImplementedError


if __name__ == "__main__":
    main()
