from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from src.domain.value_objects import AssetType


def calibration_error(samples: Sequence[tuple[float, bool]]) -> float:
    """#4 — luka kalibracji: |śr. deklarowana pewność − realna trafność| ∈ [0,1].

    0 = idealnie skalibrowane; wysokie = rada przepewna (lub niedopewna).
    Każdy sample to (confidence∈[0,1], czy_trafiona_kierunkowo). Pusta → 0.0."""
    if not samples:
        return 0.0
    mean_conf = sum(_unit_confidence(c) for c, _ in samples) / len(samples)
    accuracy = sum(1 for _, ok in samples if ok) / len(samples)
    return abs(mean_conf - accuracy)


# WIA-37: wersja semantyki kolumny `prediction_logs.confidence_calibration`
# (`calibration_score_version`, migracja 044). NULL w bazie = v1.
LINEAR_CALIBRATION_SCORE_VERSION = 1
BRIER_CALIBRATION_SCORE_VERSION = 2


def calibration_score(confidence: float, correct: bool) -> float:
    """#4 — kalibracja pojedynczej predykcji: `1 − |confidence − wynik(0/1)|`
    ∈ [0,1]. 1 = pewność idealnie dopasowana do wyniku.

    Wycofywana (v1, WIA-37): reguła niewłaściwa — przy prawdziwej trafności 0.6
    zgłoszenie 1.0 daje oczekiwane 0.6, a uczciwe 0.6 tylko 0.52, więc nagradza
    przepewność. Zostaje dla wierszy v1; nowe wyniki liczy
    `brier_calibration_score`."""
    outcome = 1.0 if correct else 0.0
    return 1.0 - abs(_unit_confidence(confidence) - outcome)


def _unit_confidence(confidence: float) -> float:
    """TEST-12: pewność spoza skali (wiersze sprzed `clamp_unit`, np. 85 z LLM-a
    na skali 0–100) przycięta do [0, 1], żeby kalibracja nie wyszła poza [0, 1]."""
    return min(max(confidence, 0.0), 1.0)


def brier_calibration_score(confidence: float, correct: bool) -> float:
    """WIA-37 — kalibracja pojedynczej predykcji jako `1 − (confidence − wynik)²`
    ∈ [0,1] (v2, `BRIER_CALIBRATION_SCORE_VERSION`).

    Reguła właściwa: oczekiwany wynik jest największy, gdy pewność równa się
    prawdziwej trafności, więc przepewność się nie opłaca."""
    outcome = 1.0 if correct else 0.0
    return 1.0 - (confidence - outcome) ** 2


# Placeholdery, które reflect_node zapisuje w `correction_insights`, gdy nie ma
# prawdziwej diagnozy LLM. To nie są lekcje: lessons, analogi RAG, „dlaczego”
# w raporcie i recap odsiewają je przez `is_substantive_insight`.
CORRECT_PREDICTION_INSIGHT = "Trafiona predykcja."
# Diagnoza pominięta przy niskiej zmienności NIE jest ponawiana — predykcja
# zostaje zamknięta w tym samym cyklu.
DIAGNOSIS_SKIPPED_INSIGHT = "Niska zmienność — diagnoza LLM pominięta (bez ponowienia)."
# Tekst sprzed WIA-10 obiecywał „odłożenie”, którego nigdy nie było. Wiersze
# z nim zostają w bazie, więc predykat musi go nadal rozpoznawać.
LEGACY_DIAGNOSIS_DEFERRED_INSIGHT = "Niska zmienność — diagnoza LLM odłożona."
# DAT-14: rozliczenie na notowaniu sprzed sesji po predykcji nie mówi nic o trafności,
# więc nie ma czego diagnozować (płatny LLM pominięty).
STALE_QUOTE_INSIGHT = "Nieaktualne notowanie — rozliczenie bez oceny i bez diagnozy LLM."

# WIA-53: błędne predykcje z tymi placeholderami nie mają diagnozy, za którą
# ktoś zapłacił — job Slow Loopa (`BackfillDiagnosesUseCase`) diagnozuje je
# w tygodniowym budżecie. „Trafiona predykcja.” nie wymaga diagnozy.
UNDIAGNOSED_MISTAKE_PLACEHOLDERS: tuple[str, ...] = (
    DIAGNOSIS_SKIPPED_INSIGHT,
    LEGACY_DIAGNOSIS_DEFERRED_INSIGHT,
)

_PLACEHOLDER_INSIGHTS = frozenset(
    text.casefold()
    for text in (
        CORRECT_PREDICTION_INSIGHT,
        DIAGNOSIS_SKIPPED_INSIGHT,
        LEGACY_DIAGNOSIS_DEFERRED_INSIGHT,
        STALE_QUOTE_INSIGHT,
    )
)


def is_substantive_insight(insight: str | None) -> bool:
    """Czy `correction_insights` niesie prawdziwą diagnozę.

    False dla braku, pustego tekstu i każdego placeholdera reflect_node
    (także historycznego „odłożona”)."""
    if insight is None:
        return False
    stripped = insight.strip()
    return bool(stripped) and stripped.casefold() not in _PLACEHOLDER_INSIGHTS


class TrendDirection(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


def normalize_trend(value: object) -> tuple[TrendDirection, bool]:
    """Kanoniczny trend z surowej wartości i flaga „wartość nieprawidłowa”.

    Strip + upper; brak (None, pusty tekst) → SIDEWAYS bez flagi, bo to brak
    pola, nie skażony input. Wartość obecna spoza zbioru (np. „UP”) → SIDEWAYS
    z flagą. Ta sama reguła obsługuje odpowiedź LLM (ML-07) i stare wiersze
    bazy sprzed niej — inaczej `TrendDirection(...)` blokował reflect."""
    if value is None:
        return TrendDirection.SIDEWAYS, False
    normalized = str(value).strip().upper()
    if not normalized:
        return TrendDirection.SIDEWAYS, False
    if normalized in TrendDirection.__members__:
        return TrendDirection(normalized), False
    return TrendDirection.SIDEWAYS, True


# Horyzont prognozy: cel ceny na 12 h (`target_price_12h`, `actual_price_after_12h`,
# etykieta modelu ML). Wchodzi do commitmentu v2 i do protokołu ewaluacji.
PREDICTION_HORIZON_HOURS = 12

SIDEWAYS_TOLERANCE = Decimal("0.005")  # ±0.5% — pasmo FLAT dla akcji/ETF
# Krypto natywnie rusza się 3–5% dziennie, więc pasmo ±0.5% uznawałoby niemal
# każdy dzień za "ruch" i systematycznie zaniżało trafność predykcji SIDEWAYS
# na BTC/ETH. Osobny próg jest korektą POMIARU, nie poluzowaniem oceny.
CRYPTO_SIDEWAYS_TOLERANCE = Decimal("0.025")  # ±2.5%


def sideways_tolerance_for(asset_type: AssetType | None) -> Decimal:
    """Pasmo „bez ruchu” klasy aktywa: krypto ±2.5%, reszta (i brak klasy) ±0.5%.

    Jedno źródło reguły dla reflect (`is_trend_correct`) i recapu tygodniowego
    (`ClosedPrediction.flat_band`) — wcześniej każdy wybierał pasmo sam."""
    if asset_type is AssetType.CRYPTO:
        return CRYPTO_SIDEWAYS_TOLERANCE
    return SIDEWAYS_TOLERANCE


@dataclass(frozen=True)
class Prediction:
    symbol: str
    predicted_trend: TrendDirection
    price_at_prediction: Decimal
    predicted_target_price: Decimal
    id: str | None = None  # ustawiane przy odczycie z repozytorium
    # Wejście predykcji (nagłówki złączone " | " i uzasadnienie LLM), czytane
    # z repozytorium dla diagnozy błędu. None = brak zapisu (stare wiersze).
    news_summary: str | None = None
    reasoning: str | None = None
    # DAT-14: chwila zapisu predykcji — rozliczenie wymaga notowania z sesji po niej.
    # None = repozytorium jej nie podało (wtedy stęchłości rozliczenia nie oceniamy).
    created_at: datetime | None = None

    def is_trend_correct(
        self, actual_price: Decimal, tolerance: Decimal | None = None
    ) -> bool:
        """Czy KIERUNEK predykcji się potwierdził.

        `tolerance` dotyczy wyłącznie SIDEWAYS — szerokość pasma "brak ruchu".
        None → `SIDEWAYS_TOLERANCE` (±0.5%, akcje/ETF). Dla krypto wołający
        podaje `CRYPTO_SIDEWAYS_TOLERANCE`: przy dziennej zmienności BTC 3–5%
        pasmo akcyjne uznawałoby niemal każdy dzień za ruch i zaniżało trafność.
        """
        actual_delta = actual_price - self.price_at_prediction
        if self.predicted_trend == TrendDirection.BULLISH:
            return actual_delta > 0
        if self.predicted_trend == TrendDirection.BEARISH:
            return actual_delta < 0
        # SIDEWAYS — w granicach ±SIDEWAYS_TOLERANCE. Guard: cena bazowa 0
        # (skażony rekord) → brak punktu odniesienia. Traktujemy brak ruchu
        # jako SIDEWAYS-correct zamiast dzielić przez zero.
        if self.price_at_prediction == 0:
            return actual_delta == 0
        band = SIDEWAYS_TOLERANCE if tolerance is None else tolerance
        return abs(actual_delta / self.price_at_prediction) <= band

    def accuracy_score(self, actual_price: Decimal) -> Decimal:
        # Guard: bez ceny bazowej (0) nie da się policzyć błędu względnego —
        # zwracamy 0.0 (brak wiarygodnego sygnału), nie ZeroDivisionError.
        if self.price_at_prediction == 0:
            return Decimal("0.0")
        error = abs(actual_price - self.predicted_target_price) / self.price_at_prediction
        return max(Decimal("0.0"), Decimal("1.0") - error)
