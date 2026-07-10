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

# #15 faza 2 — istniejąca ekspozycja. Banda może zostać wyłącznie ZWĘŻONA:
# sizing nie ma prawa podbić słabego sygnału tylko dlatego, że portfel jest pusty.
# `CROWDED_*` = już sporo mamy → schodzimy o jeden stopień.
# `SATURATED_*` = mamy tak dużo, że jedyną rozsądną sugestią jest banda startowa.
CROWDED_WEIGHT_THRESHOLD = 0.15
SATURATED_WEIGHT_THRESHOLD = 0.30
# Klaster korelacji: trzy pozycje w jednym klastrze to jedna pozycja przebrana
# za trzy. Progi są luźniejsze niż dla pojedynczego symbolu, bo klaster agreguje.
CROWDED_CLUSTER_THRESHOLD = 0.40
SATURATED_CLUSTER_THRESHOLD = 0.60

# Bandy alokacji (% portfela). Świadomie konserwatywne — to sugestia magnitudy,
# nie rekomendacja inwestycyjna.
_STARTER = (1.0, 2.0)
_STANDARD = (2.0, 3.0)
_FULL = (4.0, 5.0)
_SPANS: dict[str, tuple[float, float]] = {
    "starter": _STARTER,
    "standard": _STANDARD,
    "full": _FULL,
}


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
    current_weight: float | None = None,
    cluster_exposure: float | None = None,
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

    #15 faza 2: `current_weight` (udział symbolu w portfelu) i `cluster_exposure`
    (udział jego klastra korelacji) mogą bandę wyłącznie ZWĘZIĆ. Sugerowanie
    „pełnej bandy 4-5%" komuś, kto ma już 40% kapitału w tym symbolu, było
    poradą liczoną na pustym portfelu. Oba argumenty domyślnie `None` →
    zachowanie identyczne jak przed fazą 2.
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
    # Wszystkie trzy czynniki sprzyjają → pełna banda; rada przekonana, ale
    # brak/słaby track record → banda standardowa.
    tier = "full" if good_hit_rate else "standard"

    # #15 faza 2: dopiero TERAZ patrzymy na to, co już mamy. Ekspozycja może
    # bandę wyłącznie zwęzić — nigdy poszerzyć.
    tier = _apply_exposure(tier, current_weight, cluster_exposure)
    return _band(tier, _SPANS[tier])


_DOWNGRADE = {"full": "standard", "standard": "starter", "starter": "starter"}


def _apply_exposure(
    tier: str, current_weight: float | None, cluster_exposure: float | None
) -> str:
    """Zwęża bandę o tyle stopni, ile każe istniejąca ekspozycja.

    Monotoniczność jest kontraktem: funkcja NIGDY nie podnosi bandy. Pusty
    portfel nie jest argumentem za większą pozycją — brak ekspozycji to brak
    informacji, a nie sygnał kupna.
    """
    weight = _clamp(current_weight, 0.0, 1.0) if current_weight is not None else 0.0
    cluster = (
        _clamp(cluster_exposure, 0.0, 1.0) if cluster_exposure is not None else 0.0
    )
    if weight >= SATURATED_WEIGHT_THRESHOLD or cluster >= SATURATED_CLUSTER_THRESHOLD:
        return "starter"
    if weight >= CROWDED_WEIGHT_THRESHOLD or cluster >= CROWDED_CLUSTER_THRESHOLD:
        return _DOWNGRADE[tier]
    return tier


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
