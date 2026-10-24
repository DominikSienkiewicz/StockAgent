"""ALR-08 — kontekst alertu szoku: Twoja pozycja, ostatni werdykt rady i uzasadnienie
ostatniej predykcji.

Sam push „NVDA -6,2%” nie mówi, czy ruch dotyczy czytelnika ani czy model go
przewidywał. Ten moduł liczy pozycję z portfela i składa linie kontekstu; odczyty
(Supabase, darmowe /quote) robi watch, każdy best-effort. Pusty kontekst = push
jak przed ALR-08. Zero płatnych wywołań.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.application.ports import LatestPrediction, PreviousVerdict
from src.application.report_formatting import format_local_date
from src.domain.number_format import format_pct_pl
from src.domain.polish_labels import council_recommendation_label
from src.domain.portfolio import Portfolio, is_stale
from src.domain.value_objects import Ratio, to_percent

# Uzasadnienie LLM bywa akapitem, a push ma zostać krótki (Telegram tnie 4096 znaków).
REASONING_MAX_CHARS = 240


@dataclass(frozen=True)
class PositionContext:
    """Pozycja w alertowanym symbolu.

    `weight` — udział w wartości portfela (ułamek) z bieżących cen, `return_pct` —
    zwrot od zakupu (ułamek); None = brak ceny symbolu. `fully_priced` — False, gdy
    część pozycji nie ma ceny, więc waga liczy się z wycenionej części portfela.
    `as_of` — chwila stanu tabeli pozycji (stęchły portfel daje fałszywy P/L)."""

    weight: Decimal | None
    return_pct: Ratio | None
    fully_priced: bool
    as_of: datetime


@dataclass(frozen=True)
class AlertContext:
    """Kontekst jednego alertu; każde pole None = odczyt pusty albo nieudany."""

    position: PositionContext | None = None
    verdict: PreviousVerdict | None = None
    prediction: LatestPrediction | None = None


def position_context(
    portfolio: Portfolio, symbol: str, prices: Mapping[str, Decimal]
) -> PositionContext | None:
    """Waga i zwrot pozycji w `symbol`; None, gdy symbolu nie ma w portfelu.

    Loty tego samego symbolu sumujemy. Ceny są natywne (bez kursów NBP), tak jak
    wagi w raporcie, gdy kursów brak."""
    lots = [lot for lot in portfolio.positions if lot.symbol == symbol]
    if not lots:
        return None
    price = prices.get(symbol)
    cost = sum((lot.cost_basis() for lot in lots), Decimal("0"))
    weight = portfolio.weights(prices).get(symbol) if price is not None else None
    return_pct = (
        Ratio(sum((lot.market_value(price) for lot in lots), Decimal("0")) / cost - 1)
        if price is not None and cost > 0
        else None
    )
    return PositionContext(
        weight=weight,
        return_pct=return_pct,
        fully_priced=not portfolio.unpriced_symbols(prices),
        as_of=portfolio.as_of,
    )


def days_ago_label(moment: datetime, now: datetime, tz: str) -> str:
    """Wiek chwili w dniach kalendarzowych strefy `tz`: „dziś”, „wczoraj”,
    „8 dni temu”. Werdykt sprzed tygodnia nie może udawać wczorajszego."""
    zone = ZoneInfo(tz)
    days = (_aware(now).astimezone(zone).date() - _aware(moment).astimezone(zone).date()).days
    if days <= 0:
        return "dziś"
    if days == 1:
        return "wczoraj"
    return f"{days} dni temu"


def render_alert_context(
    context: AlertContext, *, now: datetime, display_timezone: str
) -> list[str]:
    """Linie kontekstu do treści pusha (po polsku, liczby z przecinkiem)."""
    lines: list[str] = []
    if context.position is not None:
        lines.append(_position_line(context.position, now, display_timezone))
    if context.verdict is not None:
        lines.append(_verdict_line(context.verdict, now, display_timezone))
    if context.prediction is not None and context.prediction.reasoning:
        lines.append(_reasoning_line(context.prediction, now, display_timezone))
    return lines


def _position_line(position: PositionContext, now: datetime, tz: str) -> str:
    parts: list[str] = []
    if position.weight is not None:
        share = "portfela" if position.fully_priced else "wycenionej części portfela"
        parts.append(f"{format_pct_pl(position.weight * 100, signed=False, decimals=0)} {share}")
    if position.return_pct is not None:
        parts.append(f"{format_pct_pl(to_percent(position.return_pct), signed=True)} od zakupu")
    line = f"💼 Twoja pozycja: {', '.join(parts)}" if parts else "💼 Masz tę pozycję w portfelu"
    if is_stale(position.as_of, now):
        line += (
            f" (stan portfela z {format_local_date(position.as_of, tz)} — może być nieaktualny)"
        )
    return f"{line}."


def _verdict_line(previous: PreviousVerdict, now: datetime, tz: str) -> str:
    verdict = previous.verdict
    agreement = format_pct_pl(
        Decimal(str(verdict.consensus_strength)) * 100, signed=False, decimals=0
    )
    return (
        f"🏛️ Rada {days_ago_label(previous.timestamp, now, tz)}: "
        f"{council_recommendation_label(verdict.final_recommendation)}, zgoda {agreement}."
    )


def _reasoning_line(prediction: LatestPrediction, now: datetime, tz: str) -> str:
    text = " ".join((prediction.reasoning or "").split())
    if len(text) > REASONING_MAX_CHARS:
        text = text[:REASONING_MAX_CHARS].rstrip() + "…"
    return f"🧠 Ostatnia predykcja ({days_ago_label(prediction.timestamp, now, tz)}): {text}"


def _aware(moment: datetime) -> datetime:
    """Naiwny `datetime` czytamy jako UTC, jak w całym repo."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
