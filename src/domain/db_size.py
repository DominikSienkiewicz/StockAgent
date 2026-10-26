"""NIEZ-37 — budżet rozmiaru bazy: raport i próg ostrzeżenia, bez retencji.

Darmowy plan Supabase ma limit rozmiaru bazy; po jego przekroczeniu projekt przechodzi
w tryb read-only, a zapisy predykcji cicho padają. `prediction_logs` trzyma wektory
1536-wymiarowe z indeksem, a `price_snapshots` rośnie co cykl, więc Slow Loop mierzy
co tydzień rozmiar relacji `public` i ostrzega od `DB_SIZE_WARNING_FRACTION` budżetu.
Decyzja 2026-10-24: tylko raport i ostrzeżenie — retencja to osobna decyzja.

Czysta logika (stdlib): suma, udział w budżecie i największe relacje do logu.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Od jakiego udziału budżetu Slow Loop ostrzega (decyzja właściciela: 70%).
DB_SIZE_WARNING_FRACTION = 0.7
# Megabajt dziesiętny: limit planu podany w MB liczony tak jest mniejszy niż w MiB,
# więc ostrzeżenie przychodzi raczej za wcześnie niż za późno.
BYTES_PER_MB = 1_000_000
# Ile największych relacji trafia do linii logu.
LARGEST_RELATIONS = 5


@dataclass(frozen=True)
class RelationSize:
    """Rozmiar jednej relacji `public` z indeksami i TOAST (`pg_total_relation_size`)."""

    name: str
    total_bytes: int


@dataclass(frozen=True)
class DbSizeAssessment:
    """Suma rozmiarów, budżet (None = bez progu) i największe relacje."""

    total_bytes: int
    budget_bytes: int | None
    largest: tuple[RelationSize, ...]

    @property
    def budget_fraction(self) -> float | None:
        if not self.budget_bytes:
            return None
        return self.total_bytes / self.budget_bytes

    @property
    def needs_warning(self) -> bool:
        fraction = self.budget_fraction
        return fraction is not None and fraction >= DB_SIZE_WARNING_FRACTION


def assess_db_size(sizes: Sequence[RelationSize], budget_mb: int | None) -> DbSizeAssessment:
    """Raport tygodnia: suma relacji, budżet w bajtach i `LARGEST_RELATIONS` największych."""
    largest = sorted(sizes, key=lambda size: (-size.total_bytes, size.name))
    return DbSizeAssessment(
        total_bytes=sum(size.total_bytes for size in sizes),
        budget_bytes=None if budget_mb is None else budget_mb * BYTES_PER_MB,
        largest=tuple(largest[:LARGEST_RELATIONS]),
    )
