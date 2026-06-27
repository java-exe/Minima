"""
MoErge — Island-Model Expert Migration für verteiltes MoE-Training.

Kernidee: Experten werden NICHT gemittelt (Permutationsproblem), sondern als
diskrete "Gene" zwischen Inseln (Workern) migriert. Ein Gen = die Experten-
Gewichte UND die zugehörige Router-Zeile ("Adresse"), die zusammen wandern —
dadurch entfällt jede Alignment-/Hungarian-Logik.

Alle Operatoren arbeiten auf state_dicts (CPU-Tensoren) und auf einer LISTE von
Inseln beliebiger Länge N → 2, 3, 4+ GPUs unterscheiden sich nur durch len(islands).
"""

from .expert_migration import (
    IslandSnapshot,
    measure_expert_utilization,
    migrate_experts,
    average_backbone,
    merge_round,
    moe_layers,
    select_champion,
)
from .transport import Transport, LocalDirTransport
from .coordinator import IslandCoordinator

__all__ = [
    "IslandSnapshot",
    "measure_expert_utilization",
    "migrate_experts",
    "average_backbone",
    "merge_round",
    "moe_layers",
    "select_champion",
    "Transport",
    "LocalDirTransport",
    "IslandCoordinator",
]
