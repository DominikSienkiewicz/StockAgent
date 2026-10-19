"""Domena Digest Lead — „Lead 5 sekund": ranking najważniejszych sygnałów cyklu.

Cel: pierwszy ekran maila (i temat) zawsze niesie najmocniejszy sygnał dnia,
żeby czytelnik w 5 sekund wiedział, „czy coś się pali". To czysta re-agregacja
sygnałów już policzonych w cyklu — zero I/O, zero portów, zero LLM.

Kolejność priorytetów (malejąco) jest wiążąca i zamrożona testami:

    1. CRITICAL QuotaAlert         — coś się realnie zepsuło w cyklu
    2. sygnał przeciw Twojej pozycji — rada SPRZEDAJ na pozycji ≥ 10% portfela
    3. ruch Twojej pozycji          — pozycja ≥ 10% portfela ponad bramką volatility
    4. rada PODZIELONA + duże |Δ|  — fundamentalny brak zgody co do kierunku
    5. największy |Δ| z silnym konsensusem
    6. werdykt zamkniętej predykcji

PORT-10 (decyzja 2026-10-19): tiery 2–3 istnieją tylko, gdy wołający poda wagę
pozycji (`LeadSignal.position_weight`). Bez portfela lead działa jak wcześniej.

NAJWIĘKSZE RYZYKO, które ten moduł świadomie neutralizuje: ranking NIE MOŻE
promować sensacji (dużego ruchu ceny) kosztem faktycznie krytycznych sygnałów.
Dlatego scoring jest tierowy — baza priorytetu ZAWSZE dominuje nad magnitudą
ruchu (patrz `_MAGNITUDE_CAP`), a goły ruch bez kwalifikującego sygnału w ogóle
nie generuje leadu.

Kontrakt sygnatur: `build_lead` bierze WYŁĄCZNIE typy domenowe (`QuotaAlert`,
`CouncilVerdict`) i prymitywy (`Decimal`, `str`). Mapowanie DTO warstwy
application (SymbolResult / ResolvedPrediction) na `LeadSignal` robi warstwa
wyżej (report_builder) — domena nie zna DTO.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from src.domain.council import CouncilVerdict
from src.domain.quota import QuotaAlert, QuotaSeverity

# --- Bazy scoringu (ADDYTYWNE, nie multiplikatywne) -------------------------
# Każdy kandydat dostaje: baza_priorytetu + magnituda(|Δ|). Suma, nie iloczyn —
# `council_verdict` bywa None, a mnożenie wyzerowałoby symbole bez rady.
# Odstępy między bazami (1000) są większe niż `_MAGNITUDE_CAP`, więc priorytet
# ZAWSZE bije magnitudę — to gwarancja invariantu anti-sensacja.
_CRITICAL_QUOTA_SCORE = 6000.0
_AGAINST_POSITION_SCORE = 5000.0
_POSITION_MOVE_SCORE = 4000.0
_SPLIT_SCORE = 3000.0
_STRONG_CONSENSUS_SCORE = 2000.0
_CLOSED_VERDICT_SCORE = 1000.0

# Górny limit wkładu magnitudy do wyniku. Realny |Δ| w % bywa duży (memy!),
# ale nigdy nie może przeskoczyć progu tieru — inaczej wielki ruch bez sygnału
# wspiąłby się ponad krytyczny alert. Cap < odstęp między bazami (1000).
_MAGNITUDE_CAP = 999.0

# Domyślna maksymalna liczba pozycji leadu — pierwszy ekran ma być krótki.
_DEFAULT_MAX_ITEMS = 3

# PORT-10: od tej wagi (udział w kapitale) pozycja dostaje własne tiery leadu.
POSITION_LEAD_MIN_WEIGHT = Decimal("0.10")


@dataclass(frozen=True)
class LeadItem:
    """Pojedyncza pozycja leadu — gotowa do renderu w HTML i plain-text.

    `icon` + `headline` to jednolinijkowy punch (headline nadaje się też na
    temat maila), `detail` to jedno zdanie kontekstu (może być puste).
    """

    icon: str
    headline: str
    detail: str


@dataclass(frozen=True)
class LeadSignal:
    """Znormalizowany wkład jednego symbolu do rankingu leadu.

    Świadomie zbudowany z prymitywów i typów domenowych — report_builder mapuje
    tu swoje DTO (SymbolResult / ResolvedPrediction), których domena nie importuje.

    - `price_delta_pct` to PROCENT (4.2 = 4,2%), nie ułamek. Mapper w warstwie
      wyżej przelicza ułamek z `PriceDelta` przez `.percent`.
    - `council_verdict` bywa None (cykl bez rady — np. odcięty bramką volatility).
    - `resolved_label` to gotowy, sformatowany tekst werdyktu zamkniętej predykcji
      (np. „trafiona predykcja BUY") albo None, gdy nic się w tym cyklu nie zamknęło.
    - `position_weight` (PORT-10) to udział symbolu w realnym portfelu jako UŁAMEK
      (0.24 = 24%), None = symbol spoza portfela albo portfela brak.
    - `crossed_volatility_gate` — ruch przekroczył próg bramki volatility w tym
      cyklu (symbol przeszedł do analizy); pod bramką ruch pozycji nie jest leadem.
    """

    symbol: str
    price_delta_pct: Decimal
    council_verdict: CouncilVerdict | None = None
    resolved_label: str | None = None
    position_weight: Decimal | None = None
    crossed_volatility_gate: bool = False


@dataclass(frozen=True)
class _Candidate:
    """Wewnętrzny kandydat do rankingu — item + jego wynik scoringu."""

    score: float
    item: LeadItem


def _magnitude(delta_pct: Decimal) -> float:
    """Wkład magnitudy |Δ| do wyniku, przycięty do `_MAGNITUDE_CAP`.

    Przycięcie chroni invariant tierów: żaden ruch, choćby ekstremalny, nie
    przeskoczy bazy wyższego priorytetu.
    """
    return min(abs(float(delta_pct)), _MAGNITUDE_CAP)


def _format_delta(delta_pct: Decimal) -> str:
    """Delta jako podpisany procent z jednym miejscem po przecinku (np. „-4.2%")."""
    return f"{delta_pct:+.1f}%"


def _quota_candidate(alert: QuotaAlert) -> _Candidate:
    """Priorytet 1: krytyczny alert limitu. Bez magnitudy — zawsze na szczycie."""
    item = LeadItem(
        icon="🚨",
        headline=f"{alert.source}: {alert.message}",
        detail=alert.action,
    )
    return _Candidate(score=_CRITICAL_QUOTA_SCORE, item=item)


def _format_weight(weight: Decimal) -> str:
    """Waga pozycji jako procent bez miejsc po przecinku (0.24 → „24%")."""
    return f"{weight * 100:.0f}%"


def _position_candidate(signal: LeadSignal) -> _Candidate | None:
    """PORT-10: tiery portfela — sygnał przeciw pozycji albo ruch pozycji.

    Portfel jest long-only (`Position.quantity > 0`), więc „przeciw” znaczy
    werdykt rady SELL. W obrębie tieru decyduje wpływ na kapitał (waga × |Δ|),
    nie sam ruch ceny. Pozycja < `POSITION_LEAD_MIN_WEIGHT` → None.
    """
    weight = signal.position_weight
    if weight is None or weight < POSITION_LEAD_MIN_WEIGHT:
        return None
    delta_label = _format_delta(signal.price_delta_pct)
    impact = _magnitude(signal.price_delta_pct) * float(weight)
    verdict = signal.council_verdict

    if verdict is not None and verdict.final_recommendation == "SELL":
        item = LeadItem(
            icon="⚠️",
            headline=(
                f"Sygnał przeciw Twojej pozycji: rada SPRZEDAJ na {signal.symbol} "
                f"({_format_weight(weight)} portfela) {delta_label}"
            ),
            detail=verdict.summary,
        )
        return _Candidate(score=_AGAINST_POSITION_SCORE + impact, item=item)

    if signal.crossed_volatility_gate:
        item = LeadItem(
            icon="💼",
            headline=f"{signal.symbol} ({_format_weight(weight)} portfela) {delta_label}",
            detail=(
                verdict.summary
                if verdict is not None
                else signal.resolved_label or "Twoja pozycja ruszyła się ponad próg bramki."
            ),
        )
        return _Candidate(score=_POSITION_MOVE_SCORE + impact, item=item)
    return None


def _signal_candidate(signal: LeadSignal) -> _Candidate | None:
    """Zamienia sygnał symbolu na kandydata najwyższego pasującego priorytetu.

    Jeden symbol → co najwyżej jeden kandydat (jego najmocniejszy aspekt), żeby
    pojedynczy ticker nie zajął całego leadu. Goły ruch bez rady i bez zamkniętej
    predykcji → None (anti-sensacja: sam duży |Δ| nie kwalifikuje się do leadu).
    """
    # PORT-10: najmocniejszy aspekt pozycji z portfela wygrywa z resztą tierów.
    position = _position_candidate(signal)
    if position is not None:
        return position

    delta_label = _format_delta(signal.price_delta_pct)
    magnitude = _magnitude(signal.price_delta_pct)
    verdict = signal.council_verdict

    # Priorytet 4 — rada PODZIELONA (sprzeczne BUY i SELL). Duże |Δ| podbija
    # pozycję w obrębie tieru, ale sam tier bije silny konsensus niezależnie od Δ.
    if verdict is not None and verdict.is_split_decision():
        item = LeadItem(
            icon="⚖️",
            headline=f"{signal.symbol} {delta_label}, rada PODZIELONA",
            detail=verdict.summary,
        )
        return _Candidate(score=_SPLIT_SCORE + magnitude, item=item)

    # Priorytet 5 — silny konsensus rady. W obrębie tieru decyduje |Δ|.
    if verdict is not None and verdict.has_strong_consensus():
        item = LeadItem(
            icon="🔥",
            headline=(
                f"{signal.symbol} {delta_label}, "
                f"rada zgodna: {verdict.final_recommendation}"
            ),
            detail=verdict.summary,
        )
        return _Candidate(score=_STRONG_CONSENSUS_SCORE + magnitude, item=item)

    # Priorytet 6 — werdykt zamkniętej predykcji.
    if signal.resolved_label:
        item = LeadItem(
            icon="📊",
            headline=f"{signal.symbol} {delta_label}: {signal.resolved_label}",
            detail=signal.resolved_label,
        )
        return _Candidate(score=_CLOSED_VERDICT_SCORE + magnitude, item=item)

    return None


def build_lead(
    signals: Sequence[LeadSignal],
    quota_alerts: Sequence[QuotaAlert] = (),
    *,
    max_items: int = _DEFAULT_MAX_ITEMS,
) -> list[LeadItem]:
    """Buduje ranking leadu cyklu — max `max_items` pozycji, najważniejsze pierwsze.

    Ranking jest deterministyczny i tierowy: baza priorytetu dominuje nad
    magnitudą |Δ|, więc kolejność priorytetów jest twarda, a wewnątrz jednego
    priorytetu decyduje wielkość ruchu. Pusty zestaw sygnałów → pusta lista
    (sekcja samosupresująca — orkiestrator po prostu jej nie renderuje).

    Tylko CRITICAL QuotaAlert-y wchodzą do leadu; WARNING mają własny banner.
    """
    candidates: list[_Candidate] = [
        _quota_candidate(alert)
        for alert in quota_alerts
        if alert.severity is QuotaSeverity.CRITICAL
    ]
    candidates.extend(
        candidate
        for candidate in (_signal_candidate(signal) for signal in signals)
        if candidate is not None
    )

    # Sortowanie malejące po wyniku; stabilne — przy remisie zachowuje kolejność
    # wejścia (np. dwa krytyczne alerty w kolejności emisji).
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    return [candidate.item for candidate in ranked[:max_items]]


def lead_headline(items: Sequence[LeadItem]) -> str | None:
    """Headline najważniejszej pozycji leadu — do dynamicznego tematu maila.

    Zwraca None, gdy lead jest pusty (orkiestrator zostawia wtedy domyślny temat).
    """
    return items[0].headline if items else None
