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

    @abstractmethod
    def fetch_latest(self) -> IslandSnapshot | None:
        """Neuesten verfügbaren Snapshot holen (zum Bootstrappen eines neuen
        Workers ohne lokalen Checkpoint). None wenn noch nichts da ist."""
        ...

    def prune(self, keep_last: int, current_round: int) -> None:
        """Optional: alte Runden-Artefakte löschen, damit der geteilte Ordner
        nicht unbegrenzt wächst (jeder Snapshot ist ein volles Modell). Default:
        No-op — Transports ohne eigenen Speicher müssen nichts tun."""
        return None


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
        # NUR ein reines dict aus Tensoren/Skalaren serialisieren (kein
        # Klassen-Objekt) → die Gegenseite kann mit weights_only=True laden.
        payload = snapshot.to_payload()
        tmp = self._path(round_idx, self.worker_id) + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self._path(round_idx, self.worker_id))  # atomar

    def _load_snapshot(self, path: str) -> IslandSnapshot:
        # weights_only=True: fremde .pt-Dateien dürfen KEINEN Code ausführen.
        # In einem offenen Netz ist jeder Snapshot untrusted → sonst RCE.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return IslandSnapshot.from_payload(payload)

    def collect(self, round_idx: int, expected: int | None = None,
                timeout_s: float = 600.0,
                settle_s: float = 3.0) -> list[IslandSnapshot]:
        """
        Wartet, bis 'expected' Snapshots dieser Runde da sind (oder Timeout),
        und lädt ALLE vorhandenen. expected=None ⇒ nimm, was nach kurzer
        Sammelphase da ist (dynamische Worker-Zahl).

        settle_s: kurze Nachlauf-Sammelphase, NACHDEM die Mindestzahl erreicht
        ist — so werden Nachzügler derselben Runde noch eingesammelt, statt in
        dem Moment abzuschneiden, in dem 'expected' knapp erfüllt ist. Das war
        vorher toter Code (nur im expected=None-Zweig) und ließ die dritte+
        Insel bei 3+ Workern regelmäßig ausfallen.
        """
        pattern = os.path.join(self.dir, f"round_{round_idx:04d}__*.pt")
        deadline = time.time() + timeout_s
        while True:
            files = sorted(glob.glob(pattern))
            enough = files if expected is None else len(files) >= expected
            if enough:
                if settle_s > 0:         # Nachzügler derselben Runde einsammeln
                    time.sleep(settle_s)
                    files = sorted(glob.glob(pattern))
                break
            if time.time() > deadline:
                if not files:
                    raise TimeoutError(f"Keine Snapshots für Runde {round_idx}")
                break
            time.sleep(1.0)
        return [self._load_snapshot(f) for f in files]

    def _round_of(self, f: str) -> int:
        try:
            return int(os.path.basename(f).split("__")[0].split("_")[1])
        except (IndexError, ValueError):
            return -1

    def latest_round(self) -> int:
        rounds = [self._round_of(f)
                  for f in glob.glob(os.path.join(self.dir, "round_*__*.pt"))]
        rounds = [r for r in rounds if r >= 0]
        return max(rounds) if rounds else -1

    def fetch_latest(self) -> IslandSnapshot | None:
        files = glob.glob(os.path.join(self.dir, "round_*__*.pt"))
        if not files:
            return None
        # neueste Runde, bei Gleichstand die zuletzt geschriebene Datei
        best = max(files, key=lambda f: (self._round_of(f), os.path.getmtime(f)))
        return self._load_snapshot(best)

    def prune(self, keep_last: int, current_round: int) -> None:
        """Runden älter als (current_round - keep_last) entfernen. Behält die
        letzten `keep_last` Runden als Puffer für langsamere Inseln, die evtl.
        noch daran mergen. keep_last<=0 deaktiviert das Aufräumen."""
        if keep_last is None or keep_last <= 0:
            return
        cutoff = current_round - keep_last   # letzte keep_last Runden behalten
        for f in glob.glob(os.path.join(self.dir, "round_*__*.pt")):
            r = self._round_of(f)
            if 0 <= r <= cutoff:
                try:
                    os.remove(f)
                except OSError:
                    pass  # anderer Worker/Sync hat sie evtl. schon entfernt
