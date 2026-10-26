"""Domena Digest Lead — „Lead 5 sekund": ranking najważniejszych sygnałów cyklu.

Cel: pierwszy ekran maila (i temat) zawsze niesie najmocniejszy sygnał dnia,
żeby czytelnik w 5 sekund wiedział, „czy coś się pali". To czysta re-agregacja
sygnałów już policzonych w cyklu — zero I/O, zero portów, zero LLM.

Kolejność priorytetów (malejąco) jest wiążąca i zamrożona testami:

    1. CRITICAL QuotaAlert         — coś się realnie zepsuło w cyklu
    2. zmiana reżimu rynku          — wejście w RISK_OFF albo wyjście (ALR-05)
    3. sygnał przeciw Twojej pozycji — rada SPRZEDAJ na pozycji ≥ 10% portfela
    4. ruch Twojej pozycji          — pozycja ≥ 10% portfela ponad bramką volatility
                                      albo jej szok z watcha od ostatniego maila (ALR-06)
    5. wyniki Twojej spółki         — pozycja ≥ 10% portfela raportuje dziś albo jutro
                                      (DAT-20, kalendarz Finnhub)
    6. rada PODZIELONA + duże |Δ|  — fundamentalny brak zgody co do kierunku
    7. znaczący zakup insidera      — kod P officera ponad próg z kapitalizacji (DAT-21)
    8. rada się ODWRÓCIŁA (FLIP)   — KUP↔SPRZEDAJ względem poprzedniego cyklu
    9. największy |Δ| z silnym konsensusem
   10. werdykt zamkniętej predykcji
   11. rewizja konsensusu analityków m/m (DAT-13) — sygnał kontekstowy, okresy
       miesięczne, więc najniższy tier

PORT-10 (decyzja 2026-10-19): tiery 3–5 istnieją tylko, gdy wołający poda wagę
pozycji (`LeadSignal.position_weight`). Bez portfela lead działa jak wcześniej.
DAT-20: tier 5 wymaga też `LeadSignal.earnings_event` — wołający podaje go tylko
z flagą kalendarza wyników (`finnhub_earnings_calendar_enabled`).
ALR-05: tier 2 istnieje tylko, gdy wołający poda zmianę reżimu — DI liczy ją
wyłącznie z histerezy (`regime_history`), bez niej reżim migałby codziennie.
MAIL-19: tier 7 wymaga `LeadSignal.consensus_shift` rodzaju FLIP; porównanie
stęchłe (STALE_COMPARISON) nigdy nie trafia do leadu.

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

from src.domain.consensus_shift import ConsensusShift, ShiftKind
from src.domain.council import CouncilVerdict
from src.domain.earnings import (
    HELD_EARNINGS_LEAD_MAX_DAYS,
    EarningsEvent,
    earnings_timing_label,
)
from src.domain.number_format import format_number_pl, format_pct_pl
from src.domain.polish_labels import council_recommendation_label
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.regime import RegimeChange
from src.domain.value_objects import Percent

# --- Bazy scoringu (ADDYTYWNE, nie multiplikatywne) -------------------------
# Każdy kandydat dostaje: baza_priorytetu + magnituda(|Δ|). Suma, nie iloczyn —
# `council_verdict` bywa None, a mnożenie wyzerowałoby symbole bez rady.
# Odstępy między bazami (1000) są większe niż `_MAGNITUDE_CAP`, więc priorytet
# ZAWSZE bije magnitudę — to gwarancja invariantu anti-sensacja.
_CRITICAL_QUOTA_SCORE = 10000.0
_REGIME_CHANGE_SCORE = 9000.0
_AGAINST_POSITION_SCORE = 8000.0
_POSITION_MOVE_SCORE = 7000.0
_HELD_EARNINGS_SCORE = 6000.0
_SPLIT_SCORE = 5000.0
_INSIDER_BUY_SCORE = 4000.0
_FLIP_SCORE = 3000.0
_STRONG_CONSENSUS_SCORE = 2000.0
_CLOSED_VERDICT_SCORE = 1000.0
_ANALYST_REVISION_SCORE = 0.0

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
    `symbol` (MAIL-06) wskazuje ticker pozycji — raport daje mu pełny blok
    uzasadnień; None = pozycja bez symbolu (alert limitu).
    """

    icon: str
    headline: str
    detail: str
    symbol: str | None = None


@dataclass(frozen=True)
class LeadSignal:
    """Znormalizowany wkład jednego symbolu do rankingu leadu.

    Świadomie zbudowany z prymitywów i typów domenowych — report_builder mapuje
    tu swoje DTO (SymbolResult / ResolvedPrediction), których domena nie importuje.

    - `price_delta_pct` to PROCENT (`Percent`: 4.2 = 4,2%), nie ułamek. Mapper
      w warstwie wyżej przelicza ułamek z `PriceDelta` przez `.percent`.
    - `council_verdict` bywa None (cykl bez rady — np. odcięty bramką volatility).
    - `resolved_label` to gotowy, sformatowany tekst werdyktu zamkniętej predykcji
      (np. „trafiona prognoza wzrostu") albo None, gdy nic się w tym cyklu nie zamknęło.
    - `position_weight` (PORT-10) to udział symbolu w realnym portfelu jako UŁAMEK
      (0.24 = 24%), None = symbol spoza portfela albo portfela brak.
    - `crossed_volatility_gate` — ruch przekroczył próg bramki volatility w tym
      cyklu (symbol przeszedł do analizy); pod bramką ruch pozycji nie jest leadem.
    - `consensus_shift` (MAIL-19) — zmiana nastawienia rady względem poprzedniego
      cyklu; do leadu wchodzi tylko FLIP. None = brak porównania.
    - `insider_buy_label` (DAT-21) — gotowy tekst znaczącego zakupu insidera
      („CEO kupił akcje za 1,2 mln USD”) albo None, gdy zakupu ponad próg nie było.
    - `analyst_revision_label` (DAT-13) — gotowy tekst rewizji konsensusu m/m
      („analitycy podnieśli rekomendacje m/m (+3 Strong Buy)”) albo None (STABLE).
    - `earnings_event` (DAT-20) — najbliższy raport wyników symbolu z kalendarza;
      None = brak raportu w oknie albo flaga kalendarza off.
    """

    symbol: str
    price_delta_pct: Percent
    council_verdict: CouncilVerdict | None = None
    resolved_label: str | None = None
    position_weight: Decimal | None = None
    crossed_volatility_gate: bool = False
    consensus_shift: ConsensusShift | None = None
    insider_buy_label: str | None = None
    # ALR-06: największy szok z watcha od ostatniego maila (PROCENT); None = brak.
    shock_pct: Percent | None = None
    analyst_revision_label: str | None = None
    earnings_event: EarningsEvent | None = None


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
    """Delta jako podpisany procent z jednym miejscem po przecinku (np. „-4,2%”, MAIL-58)."""
    return format_pct_pl(delta_pct, signed=True)


def _quota_candidate(alert: QuotaAlert) -> _Candidate:
    """Priorytet 1: krytyczny alert limitu. Bez magnitudy — zawsze na szczycie."""
    item = LeadItem(
        icon="🚨",
        headline=f"{alert.source}: {alert.message}",
        detail=alert.action,
    )
    return _Candidate(score=_CRITICAL_QUOTA_SCORE, item=item)


def _regime_candidate(change: RegimeChange) -> _Candidate:
    """Priorytet 2 (ALR-05): zmiana reżimu dotyczy całego cyklu — próg bramki
    rośnie albo wraca, więc predykcji jest mniej albo więcej. Bez magnitudy."""
    if change.enters_risk_off:
        item = LeadItem(
            icon="🌧️",
            headline="Rynek przeszedł w RISK_OFF, analiza rzadsza",
            detail="Próg bramki volatility rośnie — mniej symboli dostaje pełną analizę.",
        )
    else:
        item = LeadItem(
            icon="🌤️",
            headline=f"Rynek wyszedł z RISK_OFF ({change.current.value}), analiza częstsza",
            detail="Próg bramki volatility wraca do bazowego.",
        )
    return _Candidate(score=_REGIME_CHANGE_SCORE, item=item)


def _format_weight(weight: Decimal) -> str:
    """Waga pozycji jako procent bez miejsc po przecinku (0.24 → „24%”, MAIL-58)."""
    return format_pct_pl(weight * 100, signed=False, decimals=0)


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
            symbol=signal.symbol,
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
            symbol=signal.symbol,
        )
        return _Candidate(score=_POSITION_MOVE_SCORE + impact, item=item)

    # ALR-06: szok z watcha od ostatniego maila — ruch pozycji, który mógł
    # przejść bokiem (push bez Telegrama), choć cykl jest już pod bramką.
    if signal.shock_pct is not None:
        shock_impact = _magnitude(signal.shock_pct) * float(weight)
        item = LeadItem(
            icon="⚡",
            headline=(
                f"{signal.symbol} ({_format_weight(weight)} portfela) "
                f"{_format_delta(signal.shock_pct)} między mailami"
            ),
            detail=f"Szok z watcha od ostatniego maila; ruch tego cyklu {delta_label}.",
            symbol=signal.symbol,
        )
        return _Candidate(score=_POSITION_MOVE_SCORE + shock_impact, item=item)
    return None


_EARNINGS_DAY_PL = {0: "dziś", 1: "jutro"}
# Dziś przed jutrem w obrębie tieru, potem większa pozycja (waga × 100 ≤ 100).
_EARNINGS_TODAY_BONUS = 100.0


def _held_earnings_candidate(signal: LeadSignal) -> _Candidate | None:
    """DAT-20: raport wyników pozycji ≥ `POSITION_LEAD_MIN_WEIGHT` dziś albo jutro.

    Mail idzie przed otwarciem NYSE, więc „dziś” to jeszcze przyszłość. Pora względem
    sesji i szacunek EPS są opcjonalne — nieznane po prostu nie wchodzą do tekstu."""
    weight = signal.position_weight
    event = signal.earnings_event
    if weight is None or weight < POSITION_LEAD_MIN_WEIGHT or event is None:
        return None
    if event.days_until is None or not 0 <= event.days_until <= HELD_EARNINGS_LEAD_MAX_DAYS:
        return None
    when = _EARNINGS_DAY_PL[event.days_until]
    timing = f" {earnings_timing_label(event.timing)}" if event.timing is not None else ""
    detail = "Raport kwartalny spółki z Twojego portfela (kalendarz Finnhub)"
    if event.eps_estimate is not None:
        eps = format_number_pl(Decimal(str(event.eps_estimate)), signed=False, decimals=2)
        detail += f", szac. EPS {eps}"
    item = LeadItem(
        icon="📅",
        headline=f"{signal.symbol} ({_format_weight(weight)} portfela): wyniki {when}{timing}",
        detail=f"{detail}.",
        symbol=signal.symbol,
    )
    bonus = _EARNINGS_TODAY_BONUS if event.days_until == 0 else 0.0
    return _Candidate(score=_HELD_EARNINGS_SCORE + bonus + float(weight) * 100, item=item)


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
    # DAT-20: wyniki Twojej spółki dziś albo jutro — pod ruchem pozycji.
    earnings = _held_earnings_candidate(signal)
    if earnings is not None:
        return earnings

    delta_label = _format_delta(signal.price_delta_pct)
    magnitude = _magnitude(signal.price_delta_pct)
    verdict = signal.council_verdict

    # Priorytet 5 — rada PODZIELONA (sprzeczne BUY i SELL). Duże |Δ| podbija
    # pozycję w obrębie tieru, ale sam tier bije silny konsensus niezależnie od Δ.
    if verdict is not None and verdict.is_split_decision():
        item = LeadItem(
            icon="⚖️",
            headline=f"{signal.symbol} {delta_label}, rada PODZIELONA",
            detail=verdict.summary,
            symbol=signal.symbol,
        )
        return _Candidate(score=_SPLIT_SCORE + magnitude, item=item)

    # Priorytet 6 — znaczący zakup insidera (DAT-21): kod P officera ponad próg
    # z kapitalizacji. Rzadki, publiczny sygnał o skali — bije zgodną radę.
    if signal.insider_buy_label:
        item = LeadItem(
            icon="🟢",
            headline=f"{signal.symbol} {delta_label}: {signal.insider_buy_label}",
            detail=(
                verdict.summary
                if verdict is not None
                else "Zakup na rynku (kod P, poza planem 10b5-1) z Form 4."
            ),
            symbol=signal.symbol,
        )
        return _Candidate(score=_INSIDER_BUY_SCORE + magnitude, item=item)
    # Priorytet 7 — rada odwróciła się względem poprzedniego cyklu (MAIL-19).
    flip = _flip_candidate(signal, delta_label, magnitude)
    if flip is not None:
        return flip

    # Priorytet 8 — silny konsensus rady. W obrębie tieru decyduje |Δ|.
    if verdict is not None and verdict.has_strong_consensus():
        item = LeadItem(
            icon="🔥",
            headline=(
                f"{signal.symbol} {delta_label}, "
                f"rada zgodna: {council_recommendation_label(verdict.final_recommendation)}"
            ),
            detail=verdict.summary,
            symbol=signal.symbol,
        )
        return _Candidate(score=_STRONG_CONSENSUS_SCORE + magnitude, item=item)

    # Priorytet 9 — werdykt zamkniętej predykcji.
    if signal.resolved_label:
        item = LeadItem(
            icon="📊",
            headline=f"{signal.symbol} {delta_label}: {signal.resolved_label}",
            detail=signal.resolved_label,
            symbol=signal.symbol,
        )
        return _Candidate(score=_CLOSED_VERDICT_SCORE + magnitude, item=item)

    # Priorytet 10 — rewizja konsensusu analityków m/m (DAT-13).
    if signal.analyst_revision_label:
        item = LeadItem(
            icon="📋",
            headline=f"{signal.symbol} {delta_label}: {signal.analyst_revision_label}",
            detail="Zmiana rozkładu rekomendacji sell-side miesiąc do miesiąca (Finnhub).",
            symbol=signal.symbol,
        )
        return _Candidate(score=_ANALYST_REVISION_SCORE + magnitude, item=item)

    return None


def _flip_candidate(signal: LeadSignal, delta_label: str, magnitude: float) -> _Candidate | None:
    """MAIL-19: odwrócenie rady (KUP↔SPRZEDAJ). Tylko FLIP — stęchłe porównanie
    (`STALE_COMPARISON`) i złagodzenie nie są leadem."""
    shift = signal.consensus_shift
    if shift is None or shift.kind is not ShiftKind.FLIP:
        return None
    previous = council_recommendation_label(shift.previous_recommendation)
    current = council_recommendation_label(shift.current_recommendation)
    verdict = signal.council_verdict
    item = LeadItem(
        icon="🔄",
        headline=f"{signal.symbol} {delta_label}: rada odwróciła się {previous}→{current}",
        detail=verdict.summary if verdict is not None else "",
        symbol=signal.symbol,
    )
    return _Candidate(score=_FLIP_SCORE + magnitude, item=item)


def build_lead(
    signals: Sequence[LeadSignal],
    quota_alerts: Sequence[QuotaAlert] = (),
    *,
    max_items: int = _DEFAULT_MAX_ITEMS,
    regime_change: RegimeChange | None = None,
) -> list[LeadItem]:
    """Buduje ranking leadu cyklu — max `max_items` pozycji, najważniejsze pierwsze.

    Ranking jest deterministyczny i tierowy: baza priorytetu dominuje nad
    magnitudą |Δ|, więc kolejność priorytetów jest twarda, a wewnątrz jednego
    priorytetu decyduje wielkość ruchu. Pusty zestaw sygnałów → pusta lista
    (sekcja samosupresująca — orkiestrator po prostu jej nie renderuje).

    Tylko CRITICAL QuotaAlert-y wchodzą do leadu; WARNING mają własny banner.
    `regime_change` (ALR-05) — wejście w RISK_OFF albo wyjście z niego w tym cyklu.
    """
    candidates: list[_Candidate] = [
        _quota_candidate(alert)
        for alert in quota_alerts
        if alert.severity is QuotaSeverity.CRITICAL
    ]
    if regime_change is not None:
        candidates.append(_regime_candidate(regime_change))
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


def lead_preview(items: Sequence[LeadItem]) -> str | None:
    """MAIL-13: tekst podglądu w skrzynce (preheader) — `detail` najważniejszej
    pozycji leadu, bo jej headline stoi już w temacie. Pusty detail → headline.

    Zwraca None, gdy lead jest pusty (renderer podstawia wtedy tekst zastępczy).
    """
    if not items:
        return None
    first = items[0]
    return first.detail.strip() or first.headline
