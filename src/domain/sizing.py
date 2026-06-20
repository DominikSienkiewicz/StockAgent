# src/domain/sizing.py
from __future__ import annotations

from dataclasses import dataclass

# Progi reguły Kelly-lite. Trzymane w domenie — to decyzja biznesowa o apetycie
# na ryzyko, nie szczegół prezentacji. Mocny konsensus rady, niski rozłam i
# przyzwoity track record agenta razem windują pozycję; słabość któregokolwiek
# czynnika (albo brak historii) ściąga sugestię w dół.
STRONG_CONSENSUS_THRESHOLD = 0.7
LOW_DISSENT_THRESHOLD = 0.34
GOOD_HIT_RATE_THRESHOLD = 0.55

# Bandy alokacji (% portfela). Świadomie konserwatywne — to sugestia magnitudy,
# nie rekomendacja inwestycyjna.
_STARTER = (1.0, 2.0)
_STANDARD = (2.0, 3.0)
_FULL = (4.0, 5.0)


@dataclass(frozen=True)
class SizeBand:
    """Sugerowana wielkość pozycji (banda procentowa) dla sygnału BUY/SELL.

    Czysty value object: `tier` to kategoria ("starter"/"standard"/"full"),
    `min_pct`/`max_pct` to zakres alokacji portfela w procentach, a `label`
    to gotowa, czytelna etykieta po polsku do maila.
    """

    tier: str
    min_pct: float
    max_pct: float
    label: str


def _clamp(value: float, low: float, high: float) -> float:
    """Przycina wartość do [low, high] — wejście spoza zakresu nie psuje reguły."""
    return max(low, min(high, value))


def suggest_band(
    consensus_strength: float,
    dissent_ratio: float,
    hit_rate: float | None,
) -> SizeBand:
    """Mapuje konsensus rady + track record agenta na sugerowaną bandę pozycji.

    Reguła Kelly-lite, deterministyczna i czysta (stdlib only):

    - Wejścia są clampowane do sensownych zakresów (consensus/dissent ∈ [0,1],
      hit_rate ∈ [0,1] gdy podany) — żadne ekstremum nie wysadza klasyfikacji.
    - `full` (4-5%) wymaga JEDNOCZEŚNIE: mocnego konsensusu, niskiego dissentu
      ORAZ znanego, przyzwoitego hit-rate. Brak track recordu (cold-start) lub
      słaby hit-rate nigdy nie sięgają po pełną bandę.
    - `standard` (2-3%): mocny konsensus + niski dissent, ale hit-rate nieznany
      lub poniżej progu (rada przekonana, agent jeszcze nieudowodniony).
    - `starter` (1-2%): słaby konsensus lub wysoki dissent — konserwatywnie.
    """
    consensus = _clamp(consensus_strength, 0.0, 1.0)
    dissent = _clamp(dissent_ratio, 0.0, 1.0)
    clamped_hit_rate = _clamp(hit_rate, 0.0, 1.0) if hit_rate is not None else None

    strong_consensus = consensus >= STRONG_CONSENSUS_THRESHOLD
    low_dissent = dissent <= LOW_DISSENT_THRESHOLD
    good_hit_rate = (
        clamped_hit_rate is not None and clamped_hit_rate >= GOOD_HIT_RATE_THRESHOLD
    )

    if not (strong_consensus and low_dissent):
        # Rada niepewna albo rozłam — zostajemy przy bandzie startowej.
        return _band("starter", _STARTER)
    if good_hit_rate:
        # Wszystkie trzy czynniki sprzyjają → pełna banda.
        return _band("full", _FULL)
    # Rada przekonana, ale brak/słaby track record → banda standardowa.
    return _band("standard", _STANDARD)


_TIER_LABEL_PREFIX = {
    "starter": "Pozycja startowa",
    "standard": "Pozycja standardowa",
    "full": "Pozycja pełna",
}


def _band(tier: str, span: tuple[float, float]) -> SizeBand:
    """Składa SizeBand z gotową polską etykietą `Prefiks min-max% portfela`."""
    min_pct, max_pct = span
    label = (
        f"{_TIER_LABEL_PREFIX[tier]} "
        f"{_fmt_pct(min_pct)}-{_fmt_pct(max_pct)}% portfela"
    )
    return SizeBand(tier=tier, min_pct=min_pct, max_pct=max_pct, label=label)


def _fmt_pct(value: float) -> str:
    """Formatuje procent bez zbędnego `.0` (4.0 → '4', 1.5 → '1.5')."""
    if value == int(value):
        return str(int(value))
    return f"{value:g}"
