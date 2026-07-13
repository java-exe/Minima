"""
Island-Coordinator — der verteilte Treiber.

Jede GPU führt EINEN Coordinator aus. Pro Runde:
  1. lokal H Steps trainieren (unabhängig, kein Per-Step-Sync)
  2. Snapshot (Modellgewichte + Experten-Auslastung) via Transport publizieren
  3. alle aktuell verfügbaren Snapshots einsammeln (mit Timeout)
  4. merge_round(): Experten migrieren + Backbone mitteln
  5. den für DIESE Insel gemergten Stand laden, weitertrainieren

Robustheit:
  • Verbindung = gemeinsamer Speicher (kein NCCL) → latenz-/ausfalltolerant.
  • Heterogene GPUs (auch AMD/ROCm) erlaubt: Worker trainieren unabhängig.
  • Ausfall eines Workers: andere mergen mit den vorhandenen Inseln weiter.
  • min_workers=1 ⇒ Loop blockiert nie.

Beitrags-Steuerung:
  • max_runtime_s: zeitlich begrenzte Session — nach Ablauf sauber beenden.
  • SIGINT/SIGTERM (Strg+C): laufenden Step beenden, checkpointen, GPU freigeben.
"""
from __future__ import annotations

import math
import os
import signal
import time

import torch
import torch.nn as nn

from .device import DeviceConfig
from .expert_migration import (
    IslandSnapshot, measure_expert_utilization, merge_round,
)
from .transport import Transport


class IslandCoordinator:
    def __init__(self, *, model, optimizer, train_loader, val_loader,
                 transport: Transport, worker_id: str, dev: DeviceConfig,
                 local_steps: int = 300, lb_weight: float = 0.0,
                 min_workers: int = 1, collect_timeout: float = 600.0,
                 probe_batches: int = 20, grad_clip: float = 1.0,
                 eval_batches: int = 20, keep_rounds: int = 3,
                 max_runtime_s: float | None = None,
                 ckpt_dir: str | None = None,
                 migrate_kw: dict | None = None):
        self.model = model
        self.opt = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tx = transport
        self.worker_id = worker_id
        self.dev = dev
        self.device = dev.device
        self.local_steps = local_steps
        self.lb_weight = lb_weight
        self.min_workers = min_workers
        self.collect_timeout = collect_timeout
        self.probe_batches = probe_batches
        self.grad_clip = grad_clip
        self.eval_batches = eval_batches
        self.keep_rounds = keep_rounds
        self.max_runtime_s = max_runtime_s
        self.ckpt_dir = ckpt_dir
        self.migrate_kw = migrate_kw or {}
        self._data = self._infinite(train_loader)
        self._stop = False
        self._deadline = math.inf
        self._global_step = 0            # kumulierte lokale Steps (für Snapshot)

    # ── interne Helfer ────────────────────────────────────────────────
    @staticmethod
    def _infinite(loader):
        while True:
            yield from loader

    def _install_signal_handlers(self):
        def handler(signum, _frame):
            print(f"\n[{self.worker_id}] Stop-Signal — beende sauber, gebe GPU frei ...")
            self._stop = True
        for sig in ("SIGINT", "SIGTERM"):
            if hasattr(signal, sig):
                try:
                    signal.signal(getattr(signal, sig), handler)
                except (ValueError, OSError):
                    pass  # z.B. nicht im Hauptthread

    def _time_left(self) -> bool:
        return time.time() < self._deadline and not self._stop

    def _train_local(self, steps: int):
        self.model.train()
        last = float("nan")
        done = 0
        for _ in range(steps):
            if not self._time_left():
                break
            input_ids, target_ids = next(self._data)
            input_ids = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)
            with self.dev.autocast():
                logits = self.model(input_ids)
                B, T, V = logits.shape
                loss = nn.functional.cross_entropy(
                    logits.view(B * T, V), target_ids.view(B * T))
                if self.lb_weight > 0:
                    loss = loss + self.model.get_load_balancing_loss() * self.lb_weight
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.opt.zero_grad()
            last = loss.item()
            done += 1
            self._global_step += 1
        return last, done

    @torch.no_grad()
    def _eval_val(self) -> float:
        """Held-out-Loss auf dem (geteilten!) val_loader — Fitness fürs
        Champion-Ranking. Setzt voraus, dass alle Inseln dieselbe val.bin
        nutzen (sonst sind die Losses nicht vergleichbar; siehe run_island)."""
        was_training = self.model.training
        self.model.eval()
        total, n = 0.0, 0
        for b, (input_ids, target_ids) in enumerate(self.val_loader):
            if b >= self.eval_batches:
                break
            input_ids = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)
            with self.dev.autocast():
                logits = self.model(input_ids)
                B, T, V = logits.shape
                loss = nn.functional.cross_entropy(
                    logits.view(B * T, V), target_ids.view(B * T))
            total += loss.item()
            n += 1
        if was_training:
            self.model.train()
        return total / max(n, 1)

    def _snapshot(self, val_loss: float) -> IslandSnapshot:
        util = measure_expert_utilization(
            self.model, self.val_loader, self.probe_batches, self.device)
        sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        return IslandSnapshot(worker_id=self.worker_id, step=self._global_step,
                              state_dict=sd, util=util, val_loss=val_loss)

    def _reset_opt_state(self, changed_keys: list[str]):
        """Optimizer-Momente der durch Migration überschriebenen Parameter
        verwerfen. Die Adam-Statistik gehörte zum alten (toten) Experten; auf
        einen frisch eingewanderten Experten angewandt gäbe sie einen
        destabilisierenden Kick direkt nach dem Sync. AdamW initialisiert den
        Zustand beim nächsten step() faul neu."""
        if not changed_keys:
            return
        name_to_param = dict(self.model.named_parameters())
        for k in set(changed_keys):
            p = name_to_param.get(k)
            if p is not None:
                self.opt.state.pop(p, None)

    def _load_merged(self, merged_state: dict):
        # in-place copy ⇒ Optimizer-Momente bleiben gültig
        self.model.load_state_dict(merged_state)

    def _save(self, round_idx: int):
        if not self.ckpt_dir:
            return
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir,
                            f"{self.worker_id}_round_{round_idx:04d}.pt")
        torch.save({"round": round_idx,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict()}, path)

    def _save_champion(self, round_idx: int, val_loss: float):
        """Diese Insel war die beste (niedrigste Val-Loss) → als Champion
        sichern. Muss VOR dem Merge aufgerufen werden, solange self.model noch
        exakt der bewertete Zustand ist. Das beantwortet 'welches Modell liefern
        wir aus?', das im verteilten Loop sonst offen bliebe."""
        if not self.ckpt_dir:
            return
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, "champion.pt")
        torch.save({"round": round_idx, "worker_id": self.worker_id,
                    "val_loss": val_loss,
                    "model_state_dict": self.model.state_dict()}, path)
        print(f"[{self.worker_id}] Champion (val_loss {val_loss:.4f}) "
              f"gesichert → {path}")

    def _release_gpu(self):
        try:
            self.model.to("cpu")
        except Exception:
            pass
        if self.dev.kind in ("cuda", "rocm") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{self.worker_id}] GPU freigegeben.")

    # ── Hauptschleife ─────────────────────────────────────────────────
    def run(self, total_rounds: int):
        self._install_signal_handlers()
        start = time.time()
        if self.max_runtime_s:
            self._deadline = start + self.max_runtime_s

        r = max(0, self.tx.latest_round())   # verspäteter Einstieg
        end = r + total_rounds
        budget = f"{self.max_runtime_s:.0f}s" if self.max_runtime_s else "unbegrenzt"
        print(f"[{self.worker_id}] Start Runde {r}-{end} | Zeitbudget {budget} "
              f"| {self.dev.describe()}")

        try:
            while r < end and self._time_left():
                t0 = time.time()
                train_loss, did = self._train_local(self.local_steps)
                if did == 0:                 # Zeit/Стоп vor erstem Step
                    break

                val_loss = self._eval_val()
                snap = self._snapshot(val_loss)
                self.tx.publish(snap, r)

                try:
                    snaps = self.tx.collect(r, expected=self.min_workers,
                                            timeout_s=self.collect_timeout)
                except TimeoutError:
                    snaps = [snap]           # niemand sonst → solo

                ids = [s.worker_id for s in snaps]
                if self.worker_id not in ids:
                    snaps.append(snap); ids.append(self.worker_id)

                # Champion-Wahl VOR dem Merge: self.model ist hier noch exakt der
                # bewertete Zustand. Beste Insel = niedrigste Val-Loss (NaN ans Ende).
                best = min(range(len(snaps)),
                           key=lambda k: (math.isnan(snaps[k].val_loss),
                                          snaps[k].val_loss))
                if snaps[best].worker_id == self.worker_id:
                    self._save_champion(r, snaps[best].val_loss)

                states = [s.state_dict for s in snaps]
                utils = [s.util for s in snaps]
                merged, changed = merge_round(states, utils, return_changed=True,
                                              **self.migrate_kw)
                mine = ids.index(self.worker_id)
                self._load_merged(merged[mine])
                self._reset_opt_state(changed[mine])   # stale Adam-Momente verwerfen
                self._save(r)
                self.tx.prune(self.keep_rounds, r)     # alte Runden aufräumen

                dt = time.time() - t0
                print(f"[{self.worker_id}] Runde {r} | {len(snaps)} Inseln "
                      f"| {did} steps | train_loss {train_loss:.3f} "
                      f"| val_loss {val_loss:.3f} | champ {snaps[best].worker_id} "
                      f"| {dt:.0f}s")

                latest = self.tx.latest_round()
                r = max(r + 1, latest)
        finally:
            # saubere Rückgabe der GPU in JEDEM Fall
            self._save(r)
            self._release_gpu()
            reason = "Stop-Signal" if self._stop else (
                "Zeitbudget" if not self._time_left() and self.max_runtime_s
                else "Runden erreicht")
            print(f"[{self.worker_id}] Session beendet ({reason}) nach Runde {r}.")
