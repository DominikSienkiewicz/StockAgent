# src/domain/consensus_shift.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.domain.council import CouncilVerdict

# Progi detekcji zmiany nastawienia rady — zamrożone w domenie, nie w raporcie,
# żeby logika decyzyjna miała jedno źródło prawdy (dzielone przez testy i warstwę
# prezentacji). Granice są celowo asymetryczne i przetestowane co do znaku:
#   - skok sentymentu liczy się od >= 0.2 (włącznie),
#   - spadek konsensusu liczy się dopiero powyżej 0.25 (ostro, wyłącznie).
SENTIMENT_JUMP_THRESHOLD = 0.2
CONSENSUS_DROP_THRESHOLD = 0.25

# Próg wieku porównania. Cron Fast Loopa bywa ZAKOMENTOWANY, więc "wczoraj"
# potrafi być sprzed tygodnia. Powyżej tej luki degradujemy do STALE_COMPARISON,
# żeby nie robić fałszywej dramaturgii ze stęchłych danych (największe ryzyko #8).
STALE_GAP_HOURS = 48.0

_HOURS_PER_DAY = 24.0

# Pary rekomendacji uznawane za pełne odwrócenie kierunku (BUY↔SELL). Przejścia
# do/z HOLD to złagodzenie, nie flip — traktowane jako SOFTENING.
_REVERSALS: frozenset[tuple[str, str]] = frozenset({("BUY", "SELL"), ("SELL", "BUY")})


class ShiftKind(Enum):
    """Rodzaj zmiany nastawienia rady między dwoma cyklami.

    - FLIP: pełne odwrócenie kierunku (BUY↔SELL) — najbardziej actionable sygnał.
    - SOFTENING: osłabienie przekonania bez odwrócenia (spadek konsensusu, skok
      sentymentu, albo zmiana rekomendacji do/z HOLD).
    - STABLE: nic nie przekroczyło progów, rada trzyma kurs.
    - STALE_COMPARISON: luka czasowa > STALE_GAP_HOURS — porównanie jest zbyt
      stare, żeby mu ufać; warstwa wyżej pokazuje "porównanie z cyklu sprzed N dni".
    """

    FLIP = "FLIP"
    SOFTENING = "SOFTENING"
    STABLE = "STABLE"
    STALE_COMPARISON = "STALE_COMPARISON"


@dataclass(frozen=True)
class ConsensusShift:
    """Wynik porównania dwóch werdyktów rady.

    Niesie zarówno klasyfikację (`kind`), jak i surowe dane, żeby warstwa
    prezentacji mogła napisać pełną narrację ("Rada odwróciła się BUY→SELL,
    3 głosy zmienione, sentyment -0.5, porównanie z cyklu sprzed N dni").
    `gap_hours` zachowane w wyniku pozwala renderer-owi policzyć `gap_days`
    nawet gdy porównanie zostało oznaczone jako STALE_COMPARISON.
    """

    kind: ShiftKind
    previous_recommendation: str
    current_recommendation: str
    votes_changed: int
    sentiment_delta: float
    consensus_delta: float
    gap_hours: float

    @property
    def gap_days(self) -> float:
        """Luka czasowa w dniach — do zdania 'porównanie z cyklu sprzed N dni'."""
        return self.gap_hours / _HOURS_PER_DAY


def detect_shift(
    previous: CouncilVerdict,
    current: CouncilVerdict,
    *,
    previous_sentiment: float,
    current_sentiment: float,
    gap_hours: float,
) -> ConsensusShift:
    """Porównuje poprzedni werdykt rady z bieżącym i klasyfikuje zmianę nastawienia.

    Bierze typy domenowe (`CouncilVerdict`) + prymitywy sentymentu i lukę czasową
    — świadomie NIE dotyka DTO warstwy application (`SymbolResult`/`ResolvedPrediction`),
    bo import application w domenie jest zakazany.

    Liczbę zmienionych głosów pobieramy z `CouncilVerdict.opinion_shift()` — jedno
    źródło prawdy dla "ilu inwestorów zmieniło zdanie" (dopasowanie po nazwisku).

    Kolejność decyzji (bramka wieku ma pierwszeństwo NAD wszystkim):
      1. gap_hours > STALE_GAP_HOURS → STALE_COMPARISON, bez względu na dane.
      2. odwrócenie kierunku (BUY↔SELL) → FLIP.
      3. osłabienie (zmiana rekomendacji, spadek konsensusu > próg,
         lub skok sentymentu >= próg) → SOFTENING.
      4. w przeciwnym razie → STABLE.

    Surowe delty i `gap_hours` są niesione w wyniku ZAWSZE — także przy
    STALE_COMPARISON — żeby warstwa wyżej mogła jawnie oznaczyć porównanie.
    """
    sentiment_delta = current_sentiment - previous_sentiment
    consensus_delta = current.consensus_strength - previous.consensus_strength
    votes_changed = current.opinion_shift(previous.investor_opinions)

    kind = _classify(
        previous_recommendation=previous.final_recommendation,
        current_recommendation=current.final_recommendation,
        sentiment_delta=sentiment_delta,
        consensus_delta=consensus_delta,
        gap_hours=gap_hours,
    )

    return ConsensusShift(
        kind=kind,
        previous_recommendation=previous.final_recommendation,
        current_recommendation=current.final_recommendation,
        votes_changed=votes_changed,
        sentiment_delta=sentiment_delta,
        consensus_delta=consensus_delta,
        gap_hours=gap_hours,
    )


def _classify(
    *,
    previous_recommendation: str,
    current_recommendation: str,
    sentiment_delta: float,
    consensus_delta: float,
    gap_hours: float,
) -> ShiftKind:
    """Czysta logika progowa — wyodrębniona, żeby była trywialnie testowalna."""
    # Bramka wieku ma bezwzględne pierwszeństwo: stęchłe dane nie robią dramaturgii.
    if gap_hours > STALE_GAP_HOURS:
        return ShiftKind.STALE_COMPARISON

    if (previous_recommendation, current_recommendation) in _REVERSALS:
        return ShiftKind.FLIP

    recommendation_changed = previous_recommendation != current_recommendation
    # consensus_delta < 0 to spadek; interesuje nas spadek WIĘKSZY niż próg (ostro).
    consensus_dropped = -consensus_delta > CONSENSUS_DROP_THRESHOLD
    sentiment_jumped = abs(sentiment_delta) >= SENTIMENT_JUMP_THRESHOLD

    if recommendation_changed or consensus_dropped or sentiment_jumped:
        return ShiftKind.SOFTENING

    return ShiftKind.STABLE
