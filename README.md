# Minima: Mixture-of-Experts-Sprachmodell für Code

Kleines Code-Sprachmodell (267M Parameter, ca. 97M pro Token aktiv), von Grund
auf in PyTorch gebaut und auf einer einzelnen RTX 3060 trainiert.

## Architektur
Decoder-only Transformer: 12 Layer, d_model 768, 12 Attention-Köpfe, Kontext 512,
Vokabular 16k (eigener BPE-Tokenizer in C++ mit pybind11). Die Feed-Forward-Blöcke
sind MoE-Layer mit 8 Experten, Top-2-Routing und Load-Balancing-Loss.

## Training
BF16, torch.compile, Gradient Checkpointing, optional Selective Backprop und
Layer Dropping. `diagnostics/` enthält Routing-Check und Äquivalenztest für die
gebatchte Expert-Implementierung.

## Forschung: Archipelago
Prototyp für verteiltes MoE-Training über langsame Verbindungen (Experten wandern
zusammen mit ihrer Router-Zeile zwischen Workern). System läuft, die zentrale
Hypothese ist noch nicht getestet. Siehe docs/Archipelago.md.

## Stand
Läuft: Tokenizer, Modell, Training, Textgenerierung .
In Arbeit: ONNX-Export, C++-Inference, WPF-Frontend.

## Entstehung
Als HTL-Schulprojekt begonnen, inzwischen ein Einzelprojekt. Architektur und erste
Implementierung sind von mir. Für Refactoring, Tests und das Härten des verteilten
Trainings habe ich Claude Code eingesetzt (siehe Commit-Historie).
