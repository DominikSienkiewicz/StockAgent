import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.domain.value_objects import (
    AssetType,
    Fundamentals,
    Money,
    Percent,
    Ratio,
    Threshold,
    ValuationVerdict,
    to_percent,
)

# Progi heurystyki wyceny — jawnie nazwane, łatwe do strojenia w testach.
PEG_UNDERVALUED_THRESHOLD = 1.0
PEG_OVERVALUED_THRESHOLD = 2.0
FORWARD_PE_HIGH = 30.0
EPS_GROWTH_LOW = 0.10

# FIN-14 (decyzja 2026-10-19: re-baseline): referencja ceny starsza niż tyle
# godzin to stan sprzed pauzy, nie poprzedni cykl. 96 h mieści weekend (~60 h)
# z jednym opuszczonym cyklem, a odcina tygodnie wstrzymanego crona.
REFERENCE_MAX_AGE_HOURS = 96

# ALR-10: okno dryfu — predykcja starsza niż tyle dni nie jest punktem odniesienia
# (stan sprzed pauzy), a bez predykcji w oknie liczy się pierwsza cena okna.
DRIFT_WINDOW_DAYS = 10


@dataclass(frozen=True)
class PriceDelta:
    """Względna zmiana ceny między ceną odniesienia a bieżącą.

    `fraction` to UŁAMEK zmiany (jednostka `Ratio`: 0.02 = +2%), NIE procent — na
    nim działa bramka volatility i on trafia do stanu grafu (`AgentState.delta`).
    Procent do prezentacji daje `percent`. Dawna nazwa `percentage` była myląca
    i groziła 100x błędem (audyt z czerwca 2026, gap4-1); alias usunięty w ARCH-14.
    """

    fraction: Decimal

    @property
    def percent(self) -> Percent:
        """Zmiana wyrażona w procentach (fraction * 100). Wygodny przelicznik
        dla warstwy prezentacji — nie używać do bramki volatility, która
        operuje na ułamku."""
        return to_percent(Ratio(self.fraction))

    @staticmethod
    def calculate(previous: Money, current: Money) -> "PriceDelta":
        if previous.amount == 0:
            return PriceDelta(Decimal("0"))
        delta = (current.amount - previous.amount) / previous.amount
        return PriceDelta(delta)


def is_reference_stale(
    reference_at: datetime | None,
    now: datetime,
    *,
    max_age_hours: int = REFERENCE_MAX_AGE_HOURS,
) -> bool:
    """FIN-14: czy referencja ceny (snapshot poprzedniego cyklu) jest stęchła.

    Stęchła = starsza niż `max_age_hours` (próg rozłączny: dokładnie 96 h to
    jeszcze poprzedni cykl). `None` (brak snapshotu) → False: cold-start ma
    własną ścieżkę (previous_price 0 → delta 0). Naiwną datę czytamy jako UTC,
    tak jak zapisuje czas całe repo. `now` jest wstrzykiwane — domena nie woła
    zegara."""
    if reference_at is None:
        return False
    aware_reference = (
        reference_at if reference_at.tzinfo is not None else reference_at.replace(tzinfo=UTC)
    )
    aware_now = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return aware_now - aware_reference > timedelta(hours=max_age_hours)


def effective_threshold(
    base: Threshold,
    asset_type: AssetType | None,
    crypto_override: Threshold | None,
    *,
    regime_multiplier: float = 1.0,
    earnings_multiplier: float = 1.0,
) -> Threshold:
    """FIN-08: efektywny próg bramki volatility — jedna reguła dla każdej bramki.

    CRYPTO ma własną zmienność (3-5% dziennie), więc dostaje `crypto_override`,
    gdy jest podany. Potem mnożniki: reżim rynku (R-25) i bliskość earnings (R-6)
    mnożą się, a oba są >= 1.0 z kontraktu domeny — bramka tylko się zacieśnia.
    `max(..., 1.0)` to pas bezpieczeństwa, gdyby kiedyś przeciekła wartość < 1."""
    chosen = (
        crypto_override if asset_type is AssetType.CRYPTO and crypto_override is not None else base
    )
    multiplier = max(regime_multiplier * earnings_multiplier, 1.0)
    if math.isclose(multiplier, 1.0):
        return chosen
    return Threshold(chosen.value * Decimal(str(multiplier)))


# MAIL-18: ruch od tej części progu efektywnego (włącznie), ale wciąż pod progiem,
# mail pokazuje jako „blisko progu” — kandydat na analizę w następnym cyklu.
NEAR_THRESHOLD_SHARE = Decimal("0.75")


def is_near_threshold(delta: Ratio, threshold: Ratio) -> bool:
    """MAIL-18: |Δ| w przedziale [75% progu, próg) — ruch pod bramką, ale blisko
    niej. Zerowy próg nie ma „blisko” (każdy ruch przechodzi bramkę)."""
    if threshold <= 0:
        return False
    move = abs(delta)
    return NEAR_THRESHOLD_SHARE * threshold <= move < threshold


def drift_reference(
    last_prediction: tuple[datetime, Decimal] | None,
    history: Sequence[tuple[datetime, Decimal]],
    now: datetime,
    *,
    window_days: int = DRIFT_WINDOW_DAYS,
) -> Decimal | None:
    """ALR-10: cena, od której liczy się dryf — powolny zjazd złożony z kroków
    pod progiem (10 × -1,9% = -17,5% bez żadnej analizy).

    Najpierw cena ostatniej predykcji, o ile powstała w oknie `window_days`: po
    analizie dryf liczy się od nowa, więc epizod kosztuje najwyżej jeden pipeline.
    Bez predykcji w oknie — pierwsza cena `history` (snapshoty, rosnąco po czasie)
    w oknie. Predykcja sprzed okna nie liczy się: po pauzie otworzyłaby bramkę
    każdemu symbolowi, który ruszył się od tamtej pory. Nic w oknie albo cena ≤ 0
    → None (bez dryfu). Naiwne daty czytamy jako UTC."""
    window_start = _as_utc(now) - timedelta(days=window_days)
    if last_prediction is not None:
        stamp, price = last_prediction
        if _as_utc(stamp) >= window_start and price > 0:
            return price
    for stamp, price in history:
        if _as_utc(stamp) >= window_start:
            return price if price > 0 else None
    return None


def _as_utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


@dataclass(frozen=True)
class Asset:
    symbol: str
    asset_type: AssetType = field(default=AssetType.STOCK)

    def evaluate_volatility(
        self,
        delta: PriceDelta,
        threshold: Threshold,
        *,
        reference_stale: bool = False,
        drift: PriceDelta | None = None,
    ) -> bool:
        # Bramka operuje na UŁAMKU (fraction), nie procencie — próg też jest
        # ułamkiem (np. 0.02 = 2%).
        # FIN-14: ruch od stęchłej referencji to skumulowana zmiana wielu dni,
        # nie ruch cyklu — bramka zostaje zamknięta niezależnie od jego wielkości.
        # Dryf (ALR-10) też: po pauzie najpierw re-baseline.
        if reference_stale:
            return False
        if abs(delta.fraction) >= threshold.value:
            return True
        # ALR-10: ten sam próg (z tymi samymi mnożnikami) dla ruchu od referencji dryfu.
        return drift is not None and abs(drift.fraction) >= threshold.value

    def evaluate_valuation(self, fundamentals: Fundamentals | None) -> ValuationVerdict:
        if self.asset_type is not AssetType.STOCK:
            return ValuationVerdict.UNKNOWN
        if fundamentals is None or fundamentals.peg_ratio is None:
            return ValuationVerdict.UNKNOWN

        peg = fundamentals.peg_ratio
        forward = fundamentals.forward_pe
        trailing = fundamentals.trailing_pe
        growth = fundamentals.eps_growth_yoy or 0.0

        if (
            peg < PEG_UNDERVALUED_THRESHOLD
            and forward is not None
            and trailing is not None
            and forward < trailing
        ):
            return ValuationVerdict.UNDERVALUED

        if peg > PEG_OVERVALUED_THRESHOLD or (
            forward is not None and forward > FORWARD_PE_HIGH and growth < EPS_GROWTH_LOW
        ):
            return ValuationVerdict.OVERVALUED

        return ValuationVerdict.FAIR
