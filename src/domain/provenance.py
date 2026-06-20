"""Odznaki proweniencji danych (Q6) — czysta logika domeny.

Każda predykcja opiera się na sygnałach o różnej świeżości i jakości:
sentyment AV bywa zdegradowany (wyczerpane klucze), fundamenty pochodzą
z cache z TTL, a walidacja wejścia zostawia `data_quality_flags`. Ten moduł
zamienia te już-istniejące sygnały na zwięzłe odznaki (FRESH / STALE /
DEGRADED / FLAGGED), które raport renderuje jako chipy przy wierszu symbolu.

Zero I/O, zero importów zewnętrznych — wejście to skalary + listy stringów,
wyjście to niemutowalne value objecty. Cała ocena progu TTL żyje TU, w domenie;
graf i raport są tylko wykonawcami (analogicznie do `Asset.evaluate_volatility`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Domyślny TTL fundamentów — lustro `FUNDAMENTALS_CACHE_TTL_HOURS` z
# value_objects.py. Trzymane lokalnie, by build_provenance_badges był
# samowystarczalny (wywołujący może nadpisać przez `ttl_hours`).
DEFAULT_FUNDAMENTALS_TTL_HOURS = 24


class ProvenanceLevel(Enum):
    """Poziom zaufania do sygnałów stojących za predykcją.

    FRESH — wszystko świeże i niezdegradowane (brak szumu w raporcie).
    STALE — kluczowy sygnał (fundamenty) z cache starszego niż TTL.
    DEGRADED — źródło jawnie oznaczyło dane jako skażone (degraded_reason).
    FLAGGED — walidacja wejścia zostawiła flagi jakości (`data_quality_flags`).
    """

    FRESH = "FRESH"
    STALE = "STALE"
    DEGRADED = "DEGRADED"
    FLAGGED = "FLAGGED"


@dataclass(frozen=True)
class ProvenanceBadge:
    """Pojedyncza odznaka proweniencji.

    `label` — krótka etykieta PL do chipa w mailu. `detail` — opcjonalny
    kontekst (powód degradacji, wiek fundamentów, liczba flag), pokazywany
    np. w tytule chipa.
    """

    level: ProvenanceLevel
    label: str
    detail: str | None = None


def _hours_between(later: datetime, earlier: datetime) -> float:
    """Różnica w godzinach. Naiwne i aware daty porównujemy spójnie —
    jeśli jedna strona jest naiwna, traktujemy obie jako ten sam zegar."""
    if later.tzinfo is None and earlier.tzinfo is not None:
        earlier = earlier.replace(tzinfo=None)
    elif later.tzinfo is not None and earlier.tzinfo is None:
        later = later.replace(tzinfo=None)
    return (later - earlier).total_seconds() / 3600.0


def build_provenance_badges(
    *,
    degraded_reason: str | None,
    fundamentals_fetched_at: datetime | None,
    data_quality_flags: Sequence[str],
    now: datetime,
    ttl_hours: int = DEFAULT_FUNDAMENTALS_TTL_HOURS,
) -> list[ProvenanceBadge]:
    """Buduje listę odznak proweniencji z już-policzonych sygnałów.

    Reguły (FinOps: ZERO nowych wywołań — czytamy wyłącznie to, co cykl już ma):
      - `degraded_reason` (truthy) → DEGRADED.
      - fundamenty starsze niż `ttl_hours` → STALE.
      - flagi jakości (poza tymi, które są już `degraded_reason`) → FLAGGED.
      - nic z powyższych → pojedyncza odznaka FRESH (brak szumu).

    DEGRADED i FLAGGED nie liczą tej samej flagi dwa razy: `degraded_reason`
    trafia też do `data_quality_flags` (tak robi predict_node), więc przy
    liczeniu FLAGGED ją wykluczamy, żeby nie dublować sensu.
    """
    badges: list[ProvenanceBadge] = []

    degraded = bool(degraded_reason)
    if degraded:
        badges.append(
            ProvenanceBadge(
                level=ProvenanceLevel.DEGRADED,
                label="Dane zdegradowane",
                detail=f"powód: {degraded_reason}",
            )
        )

    if fundamentals_fetched_at is not None:
        age_h = _hours_between(now, fundamentals_fetched_at)
        if age_h >= ttl_hours:
            badges.append(
                ProvenanceBadge(
                    level=ProvenanceLevel.STALE,
                    label="Dane z cache",
                    detail=f"fundamenty: {age_h:.0f}h (TTL {ttl_hours}h)",
                )
            )

    # FLAGGED liczy tylko flagi inne niż sam degraded_reason — inaczej ta sama
    # przyczyna byłaby raportowana dwiema odznakami.
    other_flags = [
        f for f in data_quality_flags if not (degraded and f == degraded_reason)
    ]
    if other_flags:
        badges.append(
            ProvenanceBadge(
                level=ProvenanceLevel.FLAGGED,
                label="Flagi jakości",
                detail=f"{len(other_flags)} flag(i): {', '.join(other_flags)}",
            )
        )

    if not badges:
        badges.append(
            ProvenanceBadge(level=ProvenanceLevel.FRESH, label="Świeże", detail=None)
        )

    return badges
