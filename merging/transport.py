"""
Transport-Schicht — die Naht für "GPU jederzeit hinzufügen".

Die Merge-Operatoren kennen kein Netzwerk; sie sehen nur eine Liste von
IslandSnapshots. Ein Transport sammelt diese Snapshots ein und verteilt das
Merge-Ergebnis. Der Koordinator fragt bei JEDEM Sync neu ab, wie viele Inseln
gerade da sind → 2, 3, 4+ Worker funktionieren ohne Codeänderung.

LocalDirTransport implementiert das über ein gemeinsames Verzeichnis (lokal,
NFS, oder ein synchronisierter Cloud-Ordner). Für echtes Internet später eine
Transport-Subklasse über Sockets / S3 / gRPC schreiben — gleiche Schnittstelle.
"""
from __future__ import annotations

import glob
import os
import time
from abc import ABC, abstractmethod

import torch

from .expert_migration import IslandSnapshot


class Transport(ABC):
    """Abstrakte Schnittstelle: publish (eigener Snapshot) / collect (alle)."""

    @abstractmethod
    def publish(self, snapshot: IslandSnapshot, round_idx: int) -> None: ...

    @abstractmethod
    def collect(self, round_idx: int, expected: int | None = None,
                timeout_s: float = 600.0) -> list[IslandSnapshot]: ...

    @abstractmethod
    def latest_round(self) -> int:
        """Höchste Runde, die IRGENDEIN Worker bereits veröffentlicht hat
        (-1 wenn noch keine). Erlaubt verspäteten Workern den Wiedereinstieg."""
        ...


class LocalDirTransport(Transport):
    """
    Snapshots als .pt-Dateien in einem gemeinsamen Verzeichnis.
    Dateischema:  round_<r>__<worker_id>.pt
    """

    def __init__(self, shared_dir: str, worker_id: str):
        self.dir = shared_dir
        self.worker_id = worker_id
        os.makedirs(shared_dir, exist_ok=True)

    def _path(self, round_idx: int, worker_id: str) -> str:
        return os.path.join(self.dir, f"round_{round_idx:04d}__{worker_id}.pt")

    def publish(self, snapshot: IslandSnapshot, round_idx: int) -> None:
        snapshot.to_cpu()
        tmp = self._path(round_idx, self.worker_id) + ".tmp"
        torch.save(snapshot, tmp)
        os.replace(tmp, self._path(round_idx, self.worker_id))  # atomar

    def collect(self, round_idx: int, expected: int | None = None,
                timeout_s: float = 600.0) -> list[IslandSnapshot]:
        """
        Wartet, bis 'expected' Snapshots dieser Runde da sind (oder Timeout),
        und lädt ALLE vorhandenen. expected=None ⇒ nimm, was nach kurzer
        Sammelphase da ist (dynamische Worker-Zahl).
        """
        pattern = os.path.join(self.dir, f"round_{round_idx:04d}__*.pt")
        deadline = time.time() + timeout_s
        while True:
            files = sorted(glob.glob(pattern))
            if expected is not None and len(files) >= expected:
                break
            if expected is None and files:
                time.sleep(2.0)          # kurze Sammelphase für Nachzügler
                files = sorted(glob.glob(pattern))
                break
            if time.time() > deadline:
                if not files:
                    raise TimeoutError(f"Keine Snapshots für Runde {round_idx}")
                break
            time.sleep(1.0)
        return [torch.load(f, map_location="cpu", weights_only=False)
                for f in files]

    def latest_round(self) -> int:
        rounds = []
        for f in glob.glob(os.path.join(self.dir, "round_*__*.pt")):
            try:
                rounds.append(int(os.path.basename(f).split("__")[0].split("_")[1]))
            except (IndexError, ValueError):
                pass
        return max(rounds) if rounds else -1
