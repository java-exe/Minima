# MoErge: Island-Model Expert Migration for Low-Communication Distributed MoE Training

**Project:** Minima — a small code language model and its distributed training research
**Status:** system built and verified end-to-end; core scientific hypothesis not yet tested
**Last updated:** 2026-06-29

---

## Abstract

We train a 267M-parameter Mixture-of-Experts (MoE) code language model across
multiple, geographically distributed, heterogeneous GPUs that communicate only
infrequently (every few hundred local steps) over ordinary internet links. The
central obstacle is that independently trained MoE replicas cannot be merged by
naive weight averaging: each replica's router specializes its experts
differently, so averaging expert *k* across replicas blends unrelated functions
("permutation/specialization symmetry"). We propose **MoErge**, which reframes
distributed MoE training as an **island-model evolutionary process whose unit of
selection is the individual expert**. Instead of averaging experts, islands
**migrate** them as discrete units. The key mechanism that avoids the alignment
problem entirely is that **an expert and its router row migrate together** as one
atomic gene, so the receiving model immediately knows how to route to it. Expert
fitness is read from the per-expert token-utilization statistics already computed
for load balancing. The model backbone (attention, embeddings, norms) is merged
by low-communication averaging in the DiLoCo style. This document records the
motivation, method, system design, security model, experimental protocol, and
honest current status.

---

## 1. Motivation

Training capable models on a single consumer GPU (here, an RTX 3060) is slow, and
renting large accelerators is the usual escape hatch. We instead ask whether many
volunteers, each contributing a consumer GPU intermittently from anywhere, can
jointly train one model. This is attractive for a community/research model but
forces three hard constraints:

1. **Low communication.** Per-step gradient synchronization (DDP/NCCL) is
   impossible over the internet; we can only afford to exchange weights every few
   hundred steps.
2. **Heterogeneity and churn.** Contributors have different GPUs (NVIDIA, AMD),
   different speeds, and join/leave unpredictably.
3. **Untrusted participants.** An open network invites data/model poisoning.

MoE is our target architecture because it gives high capacity at low per-token
cost (267M total / ~97M active). But MoE is precisely the architecture that
naive merging breaks, which is what makes a *good* MoE merging rule a real
research contribution.

---

## 2. Background and Related Work

- **Local SGD / DiLoCo** (Douillard et al., 2023). Each worker trains locally for
  *H* steps; an outer optimizer (Nesterov momentum) applies the aggregated
  weight-deltas. Communication-efficient, near-DDP quality — but defined for
  dense models and silent on the expert-merging problem.
- **DiPaCo** (DeepMind, 2024). The nearest neighbor: DiLoCo applied to a modular,
  sparsely-activated network where different *paths* train on different workers.
  Crucially, DiPaCo routes at the **coarse path level**, not token-level top-k.
  A low-communication scheme for standard **token-level top-k MoE** remains
  differentiated — this is our wedge.
- **Git Re-Basin** (Ainsworth et al., 2022). Aligns neurons via permutation
  before averaging dense networks. We borrow the *idea* of alignment but avoid
  paying for it by coupling each expert to its router row.
- **Branch-Train-Merge / Branch-Train-MiX** (Li et al., 2022; Sukhbaatar et al.,
  2024). Train separate dense models, then *assemble* them into an MoE. The
  reverse of our loop (we keep an MoE throughout), but shares the insight that
  experts can originate from independent training runs.
- **Evolutionary Model Merge** (Sakana AI, 2024). Evolves the *hyperparameters*
  of a merge between fixed, finished models. We instead run evolution **during**
  training, at **expert granularity**, with **router-coupled** migration as the
  merge operator itself.
- **Byzantine-robust distributed learning** (Krum, trimmed mean, geometric
  median, etc.). The defensive toolkit we draw on for the open-network setting.

> Citations are recorded from working knowledge and should be verified against
> the primary sources before any external publication.

**Novelty claim (scoped honestly).** To our knowledge, no prior method performs
*token-level top-k* MoE distributed training where the merge operator is
*evolutionary expert migration* with *the expert and its router logits moving as
one unit*. DiPaCo is path-level; Re-Basin aligns-then-averages; Sakana evolves
merge recipes of finished models. MoErge differs on all three axes.

---

## 3. Method

### 3.1 The obstacle, precisely

Weight averaging only recovers a good model when the replicas lie in the same
loss basin. For MoE this fails because expert index *k* is an arbitrary label:
router *A* and router *B* assign different token distributions to "expert 3," so
the two "expert 3" functions are unrelated. Averaging by index corrupts both, and
the router/expert co-adaptation makes it worse than the dense neuron-permutation
problem.

### 3.2 Core idea: migrate, don't average; move expert + router row together

We treat each worker as an **island** and the individual **expert** as the unit
of selection. At each sync, fit experts from one island **overwrite** unfit
experts on another (migration), rather than being averaged.

The mechanism that sidesteps alignment: the router is `Linear(d_model, n_experts)`
whose row *k* produces the gating logit for expert *k*. When expert *k* migrates
from island *A* into slot *j* on island *B*, **router row *k* is copied into row
*j* as well**. The expert and the address that summons it travel together, so the
receiving model can route to the migrant immediately — no Hungarian matching, no
optimal transport, no activation alignment.

### 3.3 Fitness — free from existing instrumentation

Selection needs a per-expert fitness. The load-balancing loss already computes
`fraction_tokens` (the share of tokens routed to each expert). We reuse it as the
utilization signal, combined with a **redundancy penalty** (cosine similarity to
experts already present on the destination island) so that migrating an expert
that duplicates the destination is discouraged. Net rule: *migrate
high-utilization, low-redundancy experts; overwrite dead/low-utilization ones.*

This has an immediate payoff for a problem we observed in our own model
(Section 6): dead experts are the lowest-fitness slots, so migration naturally
overwrites them — distributed training and expert-collapse repair reinforce each
other.

### 3.4 Backbone merging

Attention, embeddings, LayerNorms and the LM head are shared and merged by
low-communication **averaging** (DiLoCo-style local SGD). A planned upgrade
replaces the plain average with a DiLoCo **outer optimizer** (Nesterov on the
aggregated backbone delta).

### 3.5 Diversity preservation

If all islands converge to the same experts, the ensemble collapses to a single
effective model and the method's value disappears. The redundancy penalty is the
primary defense; the sync interval *H* and migration rate are secondary controls.
Whether diversity is maintained in practice is an open empirical question.

---

## 4. System Architecture

The implementation lives in `merging/` and is deliberately layered so that the
merge math knows nothing about networking.

```
                 ┌─────────────────────────────────────────────┐
   per GPU ─────▶│ run_island.py  (CLI: 1 process per worker)   │
                 └───────────────┬─────────────────────────────┘
                                 │
        ┌────────────────────────▼───────────────────────────┐
        │ IslandCoordinator  (coordinator.py)                 │
        │  train H steps → publish → collect → merge → load   │
        │  time-box, clean GPU give-back, backend-agnostic    │
        └───────┬───────────────────────────┬─────────────────┘
                │                            │
   ┌────────────▼─────────┐     ┌────────────▼─────────────────┐
   │ Transport            │     │ expert_migration.py          │
   │ (transport.py)       │     │  migrate_experts (+router    │
   │  publish / collect / │     │  row), average_backbone,     │
   │  latest_round /      │     │  merge_round,                │
   │  fetch_latest        │     │  measure_expert_utilization  │
   └──────────────────────┘     └──────────────────────────────┘
```

- **Transport** is a shared-storage rendezvous (`LocalDirTransport`: a local
  folder, NFS, Syncthing/rclone-synced directory, or bucket mount). There is **no
  NCCL**; workers exchange whole snapshots as atomically-written `.pt` files. A
  future `Transport` subclass (sockets/gRPC/S3) can do per-file fetch without
  changing the coordinator.
- **device.py** provides backend-agnostic setup: NVIDIA CUDA, AMD ROCm (detected
  via `torch.version.hip`, presents as `cuda`), Apple MPS, and CPU; automatic
  `bf16 → fp16` fallback; TF32 and fused-AdamW enabled only where safe (NVIDIA);
  UTF-8 stdout for Windows consoles. Because workers exchange **weights, not
  gradients**, mixed precision *across* workers is harmless.

### 4.1 Network bootstrap (no out-of-band files)

A new machine needs neither a local checkpoint nor a local dataset:

- **Model:** `merging/seed.py` publishes an existing checkpoint into the shared
  directory (optionally fp16 to halve transfer). `run_island.py` with no `--ckpt`
  polls `Transport.fetch_latest()` and initializes from the newest snapshot.
- **Data:** `merging/shard_data.py` splits the flat token `.bin` into *N*
  context-aligned shards (plus `val.bin`) into the shared directory;
  `run_island.py --data_dir` pulls only this worker's shard.

### 4.2 Robustness properties

- **Dropout/rejoin.** A disconnected worker's snapshot simply does not appear that
  round; others merge with whoever is present (graceful *N → N−1*). With
  `min_workers=1` the loop never blocks. On reconnection, `latest_round()` lets
  the worker rejoin at the current round, carrying its accumulated local training.
- **Heterogeneous GPUs.** Independent local training means a 3060 and a 4090 can
  both participate; a barrier timeout (`collect_timeout`) bounds waiting on
  stragglers.
- **Contributor controls.** Time-boxed sessions (`--minutes`), and a SIGINT/SIGTERM
  handler that finishes the current step, checkpoints, releases CUDA, and exits —
  the GPU is always returned cleanly.

---

## 5. Security and Threat Model

For an **open** opt-in network, adversarial contributors are the central risk:

- **Model poisoning** — submitting garbage or adversarial weights.
- **Backdoor/targeted poisoning** — experts that misbehave only on a trigger.
- **Sybil attacks** — many fake workers to outvote honest ones.
- **Free-riding** — junk submissions claiming credit.

**Architectural decision:** before opening beyond trusted machines, move from
symmetric peer-averaging to a **trusted aggregator** that *validates then merges*,
rather than letting any peer's weights flow directly into everyone's model.

Planned defenses, by return on effort:

1. **Validation gating (highest ROI).** Evaluate each contribution on a held-out
   set; reject if it does not hold/improve loss, or shows NaNs/exploded norms. The
   `measure_expert_utilization` + val-loss machinery already exists.
2. **MoErge is inherently more poison-resistant than gradient averaging.** An
   expert enters the model only if it passes a trusted-probe fitness check, and a
   migrant expert can be **quarantined and evaluated in isolation** before
   acceptance — something gradient-averaging cannot do per-contributor.
3. **Robust backbone aggregation** — coordinate-wise trimmed mean / median / Krum
   instead of plain average.
4. **Reputation** — weight/gate workers by whether their past contributions helped
   held-out loss; new workers start at low trust.
5. **Sybil resistance** — begin invite-only/allowlisted; add identity or
   proof-of-work later.
6. **Spot-check replication** — occasionally assign one shard to two workers and
   compare (BOINC-style).

**Honest limit:** subtle backdoors remain an open problem; held-out evaluation
catches gross damage but not every trigger.

---

## 6. A Motivating Finding: Expert Collapse

While preparing the base model we found that the 267M MoE had trained for 48k
steps with load balancing effectively disabled (`load_balance_weight = 0.0` and
the auxiliary-loss backward commented out). Diagnostics (`diagnostics/
check_moe_routing.py`) showed a **mild, localized collapse**: only Layer 0 had 3
dead experts (its input is just embeddings + positional encoding, giving the
router little signal), while Layers 1–11 were healthy. We re-enabled the auxiliary
loss inside the gradient-accumulation loop and resumed. This is the exact
condition MoErge's migration is designed to repair, and it gives the method a
concrete in-house motivating story.

---

## 7. Experimental Protocol

### 7.1 Kill-or-confirm probe (cheap, single machine)

Implemented in `merging/run_probe.py`. From the current checkpoint:

1. Fork two copies of the model.
2. Train each for *H* steps on **different data shards** (simulating two islands;
   may run sequentially on one GPU).
3. Compare post-sync validation loss across merge strategies:
   - **no-sync** baseline (each island alone),
   - **naive average** of experts,
   - **MoErge migration** (expert + router-row).
4. Re-run the routing diagnostic to count revived dead experts.

**Primary hypothesis:** `naive-average` loss is worse than `no-sync`, while
`migration` ≥ `no-sync` *and* reduces dead-expert count. Confirming this justifies
the full distributed build-out.

### 7.2 Metrics

- Held-out validation loss (and per-shard val loss).
- Per-expert token utilization; number of dead experts (<1% share).
- Expert diversity across islands (pairwise cosine similarity).
- Communication volume per round; wall-clock to target loss vs. single-GPU.

### 7.3 Ablations (research knobs)

Sync interval *H*; migration rate (experts swapped per sync); selection policy
(elitist / tournament / fitness-proportional); redundancy-penalty strength;
backbone merge (plain average vs. DiLoCo outer optimizer); number of islands
(2/3/4+).

---

## 8. Current Status (honest)

**Built and verified**
- Migration operators, validated against the real 48k checkpoint (expert weights
  *and* router rows migrate; non-selected experts untouched; backbone averages).
- Sparse MoE dispatch (top-k only), verified output-equivalent to the dense
  implementation (fp32 diff ~3e-7) so existing checkpoints stay valid.
- Transport: publish / collect / dropout / rejoin / `fetch_latest`.
- Coordinator: time-boxing and clean GPU give-back; backend-agnostic autocast.
- Network bootstrap of **model** (seed) and **data** (shards).
- **Live two-island merge** on one machine (two coordinators rendezvous through a
  shared folder, complete merge rounds, shut down cleanly).

**Not yet done**
- The **scientific result**: `run_probe.py` has not been run, so the core
  hypothesis (migration ≥ averaging; dead experts revived) is *untested*.
- Trusted aggregator + poisoning defenses (required before opening to outsiders).
- DiLoCo outer optimizer for the backbone (currently plain averaging).
- Contributor UX: VRAM auto-fit, idle/busy pause.
- True per-shard transfer (a naive synced folder copies all shards everywhere).

---

## 9. Open Questions and Risks

- **Does migration actually beat averaging for MoE?** Unproven until §7.1 runs.
- **Router consistency after a swap.** The softmax over the other (un-migrated)
  rows was not trained around the migrant; a short router-only finetune may be
  needed — quantify how short.
- **Staleness.** How far can a migrant's backbone drift before the expert lands
  outside a useful basin? Controlled by *H*.
- **Diversity collapse.** Whether the redundancy penalty actually keeps islands
  diverse is an empirical unknown.
- **Subtle poisoning** beyond what held-out evaluation detects.

---

## 10. Reproducibility

**Model.** `ModelConfigMoE`: vocab 16000, context 512, d_model 768, 12 layers,
12 heads, d_ff 1536, 8 experts, top_k 2, `load_balance_weight 0.01`
(~267M total / ~97M active).

**Key files**
- `merging/expert_migration.py` — migration + merge operators
- `merging/transport.py` — shared-storage rendezvous
- `merging/coordinator.py` — distributed driver
- `merging/device.py` — backend-agnostic setup
- `merging/run_island.py` — per-GPU entrypoint
- `merging/seed.py`, `merging/shard_data.py` — network bootstrap of model/data
- `merging/run_probe.py` — the kill-or-confirm experiment
- `diagnostics/check_moe_routing.py` — expert-utilization / collapse check

**Quick start (two islands, one machine).** See `merging/run_island.py` header and
the commands in the project README/notes.

---

## 11. References

1. Douillard et al. *DiLoCo: Distributed Low-Communication Training of Language
   Models.* 2023.
2. DeepMind. *DiPaCo: Distributed Path Composition.* 2024.
3. Ainsworth, Hayase, Srinivasa. *Git Re-Basin: Merging Models modulo Permutation
   Symmetries.* 2022.
4. Li et al. *Branch-Train-Merge.* 2022. Sukhbaatar et al. *Branch-Train-MiX.*
   2024.
5. Sakana AI. *Evolutionary Optimization of Model Merging Recipes.* 2024.
6. Blanchard et al. *Krum: Machine Learning with Adversaries.* 2017; and the
   broader Byzantine-robust aggregation literature (trimmed mean, geometric
   median).

> Reference details are from working memory; verify against primary sources
> before external publication.
