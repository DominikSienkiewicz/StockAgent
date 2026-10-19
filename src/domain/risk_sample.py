"""Minimalne próby w matematyce ryzyka (PORT-22).

VaR 95% z kilku obserwacji to najgorszy dzień próbki podany jako kwantyl, a
korelacja z dwóch zwrotów wynosi ±1 — takie liczby w mailu udają wiedzę, której
nie ma. Progi są decyzją o ryzyku, więc żyją w domenie, a nie w configu.
`historical_var`, `pearson` i `beta_to_portfolio` zostają bez zmian (liczą
kwantyl / korelację z tego, co dostaną); bramkę stawia wołający.

Czysty stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Decyzja właściciela 2026-10-19. 60 obserwacji to ~3 miesiące sesji przy
# jednym snapshocie na dzień — dopiero wtedy 5% ogona to więcej niż 3 dni.
MIN_VAR_OBSERVATIONS = 60
# Sparowane zwroty dla korelacji i bety: poniżej 20 współczynnik jest szumem.
MIN_PAIRED_RETURNS = 20

RiskStat = Literal["var", "correlation", "beta"]

_REQUIRED: dict[RiskStat, int] = {
    "var": MIN_VAR_OBSERVATIONS,
    "correlation": MIN_PAIRED_RETURNS,
    "beta": MIN_PAIRED_RETURNS,
}


@dataclass(frozen=True)
class Sufficiency:
    """Czy próba `n` wystarcza do statystyki, która wymaga `required` punktów."""

    ok: bool
    n: int
    required: int


def sample_sufficiency(n: int, stat: RiskStat) -> Sufficiency:
    """Ocena próby dla statystyki ryzyka; `n` to liczba obserwacji (zwrotów)."""
    required = _REQUIRED[stat]
    return Sufficiency(ok=n >= required, n=n, required=required)
