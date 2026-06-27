"""
Kern-Operatoren für Island-Model Expert Migration.

Begriffe
--------
Insel (Island) : ein Worker / eine GPU; hält ein vollständiges MoE-Modell.
Gen (Gene)     : EIN Experte = {net.0.weight, net.2.weight} + zugehörige
                 Router-Zeile router.weight[e]. Experte und seine "Adresse"
                 wandern IMMER zusammen → kein Permutations-Alignment nötig.
Fitness        : Token-Auslastung eines Experten (fraction_tokens), gemessen
                 auf einem gemeinsamen Probe-Batch → über Inseln vergleichbar.

Alles operiert auf state_dicts (CPU) und auf Listen beliebiger Länge N.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch


# ──────────────────────────────────────────────────────────────────────────
# Snapshot — das serialisierbare Objekt, das eine Insel über das Netz schickt
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class IslandSnapshot:
    worker_id: str
    step: int
    state_dict: dict                      # CPU-Tensoren
    util: dict = field(default_factory=dict)   # {layer:int -> Tensor(n_experts)}

    def to_cpu(self) -> "IslandSnapshot":
        self.state_dict = {k: v.detach().cpu() for k, v in self.state_dict.items()}
        self.util = {l: u.detach().cpu() for l, u in self.util.items()}
        return self


# ──────────────────────────────────────────────────────────────────────────
# Schlüssel-Helfer (an die Modell-Namensgebung gebunden)
# ──────────────────────────────────────────────────────────────────────────
def _router_key(layer: int) -> str:
    return f"blocks.{layer}.ff.router.weight"


def moe_layers(state_dict: dict) -> list[int]:
    """Alle Layer-Indizes, die eine MoE-Schicht (Router) besitzen."""
    return sorted(
        int(k.split(".")[1]) for k in state_dict if k.endswith("ff.router.weight")
    )


def _n_experts(state_dict: dict, layer: int) -> int:
    return state_dict[_router_key(layer)].shape[0]


def _expert_param_keys(state_dict: dict, layer: int, e: int) -> list[str]:
    pre = f"blocks.{layer}.ff.experts.{e}."
    return [k for k in state_dict if k.startswith(pre)]


def _expert_vector(state_dict: dict, layer: int, e: int) -> torch.Tensor:
    """Flacher Vektor zur Ähnlichkeitsmessung (erste Experten-Matrix reicht)."""
    return state_dict[f"blocks.{layer}.ff.experts.{e}.net.0.weight"].reshape(-1).float()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


# ──────────────────────────────────────────────────────────────────────────
# Fitness messen — Hooks auf den Router, Token-Anteile pro Experte
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def measure_expert_utilization(model, loader, n_batches: int, device) -> dict:
    """
    Liefert {layer -> Tensor(n_experts)} mit dem Anteil der Token, die jeder
    Experte (über top_k) erhält. Gemeinsamer Probe-Batch ⇒ inselübergreifend
    vergleichbar. Modell wird NICHT verändert.
    """
    from model.moe import MoELayer

    counts: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx):
        def hook(module, inp, _out):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            probs = torch.softmax(module.router(x), dim=-1)
            _, top = torch.topk(probs, module.top_k, dim=-1)
            c = torch.bincount(top.reshape(-1),
                               minlength=module.n_experts).float().cpu()
            counts[idx] = counts.get(idx, 0) + c
        return hook

    for i, block in enumerate(model.blocks):
        if isinstance(block.ff, MoELayer):
            handles.append(block.ff.register_forward_hook(make_hook(i)))

    was_training = model.training
    model.eval()
    for b, batch in enumerate(loader):
        if b >= n_batches:
            break
        input_ids = batch[0].to(device)
        model(input_ids)
    if was_training:
        model.train()

    for h in handles:
        h.remove()

    # auf Anteile normalisieren
    return {l: c / c.sum().clamp(min=1) for l, c in counts.items()}


# ──────────────────────────────────────────────────────────────────────────
# Migration — Gene zwischen N Inseln verschieben
# ──────────────────────────────────────────────────────────────────────────
def _inject_gene(dst_sd: dict, src_sd: dict, layer: int,
                 dst_e: int, src_e: int) -> None:
    """Experten-Gewichte UND Router-Zeile von (src_e) nach (dst_e) kopieren."""
    for k_src in _expert_param_keys(src_sd, layer, src_e):
        k_dst = k_src.replace(f".experts.{src_e}.", f".experts.{dst_e}.")
        dst_sd[k_dst] = src_sd[k_src].clone()
    rk = _router_key(layer)
    row = dst_sd[rk].clone()
    row[dst_e] = src_sd[rk][src_e].clone()
    dst_sd[rk] = row


def migrate_experts(
    states: list[dict],
    utils: list[dict],
    *,
    dead_threshold: float = 0.02,
    bottom_k: int | None = None,
    sim_threshold: float = 0.95,
    prefer_cross_island: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Migriert fitte Experten in schwache Slots — für N Inseln (N = len(states)).

    states  : Liste von state_dicts (CPU).
    utils   : Liste von {layer -> Tensor(n_experts)} (gleiche Reihenfolge).
    dead_threshold : Slots mit Auslastung < diesem Anteil gelten als "tot".
    bottom_k       : alternativ: pro Layer die k schwächsten Slots ersetzen.
    sim_threshold  : Diversitätsschutz — Spender, die einem vorhandenen
                     Experten der Zielinsel zu ähnlich sind (cos > thr),
                     werden übersprungen (verhindert Homogenisierung).

    Gibt NEUE state_dicts zurück (Eingaben bleiben unberührt).
    """
    N = len(states)
    assert N == len(utils) and N >= 1
    out = [copy.deepcopy(s) for s in states]
    if N == 1:
        return out  # eine Insel: nichts zu migrieren

    for L in moe_layers(states[0]):
        E = _n_experts(states[0], L)

        # globaler Spender-Pool, nach Fitness absteigend
        pool = sorted(
            ((i, e, float(utils[i][L][e])) for i in range(N) for e in range(E)),
            key=lambda t: t[2], reverse=True,
        )

        for i in range(N):
            u = utils[i][L]
            if bottom_k is not None:
                weak = torch.topk(u, min(bottom_k, E), largest=False).indices.tolist()
            else:
                weak = [e for e in range(E) if float(u[e]) < dead_threshold]
            if not weak:
                continue

            survivors = [e for e in range(E) if e not in weak]
            surv_vecs = [_expert_vector(states[i], L, e) for e in survivors]

            for slot in weak:
                donor = _pick_donor(states, pool, L, i, surv_vecs,
                                    dead_threshold, sim_threshold,
                                    prefer_cross_island)
                if donor is None:
                    continue
                di, de, dfit = donor
                _inject_gene(out[i], states[di], L, slot, de)
                surv_vecs.append(_expert_vector(states[di], L, de))
                if verbose:
                    print(f"  L{L:2d}: island{i} slot{slot} <- "
                          f"island{di} expert{de} (fit {dfit:.3f})")
    return out


def _pick_donor(states, pool, L, dst_island, surv_vecs,
                dead_threshold, sim_threshold, prefer_cross_island):
    """Bester Spender aus dem Pool, der die Diversitätsbedingung erfüllt."""
    fallback = None
    for (di, de, fit) in pool:
        if fit <= dead_threshold:          # tote Experten nicht migrieren
            continue
        if prefer_cross_island and di == dst_island:
            if fallback is None:
                fallback = (di, de, fit)
            continue
        v = _expert_vector(states[di], L, de)
        if any(_cosine(v, sv) > sim_threshold for sv in surv_vecs):
            continue                        # zu ähnlich → Diversität wahren
        return (di, de, fit)
    return fallback


# ──────────────────────────────────────────────────────────────────────────
# Backbone-Mittelung (DiLoCo-Stil) — alles außer Experten & Router
# ──────────────────────────────────────────────────────────────────────────
def _backbone_keys(state_dict: dict) -> list[str]:
    return [
        k for k in state_dict
        if ".ff.experts." not in k and not k.endswith("ff.router.weight")
    ]


def average_backbone(states: list[dict]) -> list[dict]:
    """
    Mittelt die geteilten Backbone-Parameter (Attention, Norms, Embedding,
    lm_head) über alle Inseln. Experten & Router-Zeilen bleiben unangetastet,
    da diese über Migration wandern.
    """
    N = len(states)
    out = [copy.deepcopy(s) for s in states]
    for k in _backbone_keys(states[0]):
        avg = sum(s[k].float() for s in states) / N
        avg = avg.to(states[0][k].dtype)
        for o in out:
            o[k] = avg.clone()
    return out


def merge_round(states: list[dict], utils: list[dict], **kw) -> list[dict]:
    """Ein vollständiger Sync: erst Experten migrieren, dann Backbone mitteln."""
    migrated = migrate_experts(states, utils, **kw)
    return average_backbone(migrated)


# ──────────────────────────────────────────────────────────────────────────
# Champion-Auswahl — finale "Auslieferung" ist die beste Insel
# ──────────────────────────────────────────────────────────────────────────
def select_champion(states: list[dict], val_losses: list[float]) -> dict:
    """state_dict der Insel mit der niedrigsten Val-Loss."""
    return states[int(min(range(len(states)), key=lambda i: val_losses[i]))]
