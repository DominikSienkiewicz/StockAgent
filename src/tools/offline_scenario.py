"""DX-04: scenariusz trybu offline i podsumowanie przebiegu — czyste funkcje.

Scenariusz (`data/offline/scenario.json`) opisuje kilka kolejnych cykli Fast
Loopa: start, odstęp, symbole z klasami aktywów, progi bramki, ceny „poprzedniego
cyklu” (referencja dla pierwszej delty) i ceny każdego cyklu. Fake'i i graf składa
podkomenda `main_tools.py offline` (ARCH-39); tu nie ma I/O, `Settings` ani adapterów.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.application.report_models import ResolvedPrediction


@dataclass(frozen=True)
class OfflineScenario:
    """Kolejne cykle offline: `cycles[i]` to ceny cyklu startującego w `cycle_at(i)`.

    `seed_prices` zapisuje się jako snapshot cen o jeden odstęp przed pierwszym
    cyklem — bez nich pierwszy cykl byłby cold-startem (delta 0, wszystko pod bramką)."""

    started_at: datetime
    cycle_interval: timedelta
    symbols: tuple[str, ...]
    etf_symbols: tuple[str, ...]
    crypto_symbols: tuple[str, ...]
    volatility_threshold: Decimal
    crypto_volatility_threshold: Decimal
    # Drugie sito rady (jak `council_volatility_threshold`); None = rada przy każdej analizie.
    council_volatility_threshold: Decimal | None
    reflection_min_age_hours: int
    track_record_days: int
    display_timezone: str
    paper_cost_bps: float
    crypto_paper_cost_bps: float
    seed_prices: Mapping[str, Decimal]
    cycles: tuple[Mapping[str, Decimal], ...]

    @property
    def all_symbols(self) -> list[str]:
        """Symbole cyklu w kolejności maila: akcje i ETF-y, potem krypto."""
        return [*self.symbols, *self.crypto_symbols]

    @property
    def seeded_at(self) -> datetime:
        """Chwila snapshotu `seed_prices` — „poprzedni cykl” przed pierwszym."""
        return self.started_at - self.cycle_interval

    def cycle_at(self, index: int) -> datetime:
        return self.started_at + index * self.cycle_interval


def _invalid(reason: str) -> ValueError:
    return ValueError(f"offline scenario: {reason}")


def _symbols(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(s, str) and s for s in value):
        raise _invalid(f"`{key}` must be a list of symbols")
    return tuple(value)


def _positive_decimal(value: Any, what: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise _invalid(f"{what} must be a number, got {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise _invalid(f"{what} must be positive, got {value!r}")
    return number


def _prices(value: Any, what: str) -> dict[str, Decimal]:
    if not isinstance(value, Mapping):
        raise _invalid(f"{what} must map symbols to prices")
    return {
        str(symbol): _positive_decimal(price, f"{what}[{symbol}]")
        for symbol, price in value.items()
    }


def _cost_bps(raw: Mapping[str, Any], key: str) -> float:
    value = raw.get(key, 0)
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
        raise _invalid(f"`{key}` must be a non-negative number of basis points")
    return float(value)


def _whole(raw: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise _invalid(f"`{key}` must be an integer >= {minimum}")
    return value


def parse_offline_scenario(raw: Mapping[str, Any]) -> OfflineScenario:
    """Scenariusz z JSON-a; zły kształt albo wartość → `ValueError` z nazwą pola."""
    started = raw.get("started_at")
    try:
        started_at = datetime.fromisoformat(str(started))
    except ValueError as exc:
        raise _invalid(f"`started_at` must be ISO 8601, got {started!r}") from exc
    if started_at.tzinfo is None:
        raise _invalid("`started_at` needs a time zone offset")
    cycles = raw.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        raise _invalid("`cycles` must be a non-empty list of price maps")
    symbols = _symbols(raw, "symbols")
    crypto = _symbols(raw, "crypto_symbols")
    if not symbols and not crypto:
        raise _invalid("the scenario needs at least one symbol")
    timezone = raw.get("display_timezone", "Europe/Warsaw")
    if not isinstance(timezone, str) or not timezone:
        raise _invalid("`display_timezone` must be a time zone name")
    return OfflineScenario(
        started_at=started_at,
        cycle_interval=timedelta(
            hours=float(_positive_decimal(raw.get("cycle_interval_hours"), "cycle_interval_hours"))
        ),
        symbols=symbols,
        etf_symbols=_symbols(raw, "etf_symbols"),
        crypto_symbols=crypto,
        volatility_threshold=_positive_decimal(
            raw.get("volatility_threshold"), "volatility_threshold"
        ),
        crypto_volatility_threshold=_positive_decimal(
            raw.get("crypto_volatility_threshold"), "crypto_volatility_threshold"
        ),
        council_volatility_threshold=(
            None
            if raw.get("council_volatility_threshold") is None
            else _positive_decimal(
                raw["council_volatility_threshold"], "council_volatility_threshold"
            )
        ),
        reflection_min_age_hours=_whole(raw, "reflection_min_age_hours", minimum=0),
        track_record_days=_whole(raw, "track_record_days", minimum=1),
        display_timezone=timezone,
        paper_cost_bps=_cost_bps(raw, "paper_cost_bps"),
        crypto_paper_cost_bps=_cost_bps(raw, "crypto_paper_cost_bps"),
        seed_prices=_prices(raw.get("seed_prices"), "seed_prices"),
        cycles=tuple(_prices(cycle, f"cycles[{index}]") for index, cycle in enumerate(cycles)),
    )


@dataclass(frozen=True)
class OfflineCycleSummary:
    """Wynik jednego cyklu offline: statusy symboli, rozliczenia i pliki maila."""

    index: int
    started_at: datetime
    subject: str
    statuses: Mapping[str, str]
    resolved: tuple[ResolvedPrediction, ...]
    html_path: Path
    text_path: Path


def _listed(symbols: Sequence[str]) -> str:
    return ", ".join(symbols) if symbols else "—"


def format_offline_summary(cycles: Sequence[OfflineCycleSummary]) -> str:
    """Kilka linii na cykl: co zapisano, pominięto, rozliczono i gdzie jest mail."""
    lines: list[str] = []
    for cycle in cycles:
        by_status: dict[str, list[str]] = {}
        for symbol, status in cycle.statuses.items():
            by_status.setdefault(status, []).append(symbol)
        settled = [f"{item.symbol} {'✓' if item.is_correct else '✗'}" for item in cycle.resolved]
        lines.append(
            f"Cykl {cycle.index} ({cycle.started_at:%Y-%m-%d %H:%M %Z}): "
            f"zapisane: {_listed(by_status.get('saved', []))} · "
            f"pominięte: {_listed(by_status.get('ignored', []))} · "
            f"błędy: {_listed(by_status.get('error', []))} · "
            f"rozliczone: {_listed(settled)}"
        )
        lines.append(f"  temat: {cycle.subject}")
        lines.append(f"  mail: {cycle.html_path} i {cycle.text_path.name}")
    return "\n".join(lines)
