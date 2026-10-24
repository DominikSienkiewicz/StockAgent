"""MAIL-21: stan każdego źródła danych w cyklu — jedna linijka na dostawcę w mailu.

Degradacja była rozsiana: chip przy symbolu, banner tylko dla kwot, a źródła alfa
(EDGAR, Reddit) psuły się po cichu. Tu z sygnałów, które cykl JUŻ ma — stanu
obwodów HTTP (`CircuitStatePort`, NIEZ-31), flag jakości symboli i licznika
błędnych odpowiedzi HTTP (NIEZ-24) — powstaje werdykt per dostawca. Zero
dodatkowych wywołań; to, które źródła są w ogóle włączone, mówi composition root.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from src.application.ports import CircuitState
from src.application.report_models import SymbolResult

# Kolejność linijek w mailu: ceny, sentyment, alfa, makro, LLM.
SOURCE_ORDER: tuple[str, ...] = (
    "Finnhub",
    "CoinGecko",
    "Alpha Vantage",
    "EDGAR",
    "Reddit",
    "FRED",
    "NBP",
    "Anthropic",
    "OpenAI",
)

# Flagi jakości symbolu, które oznaczają awarię konkretnego dostawcy.
_FLAG_SOURCES: Mapping[str, str] = {
    "av_keys_exhausted": "Alpha Vantage",
    "av_fetch_error": "Alpha Vantage",
    "av_unexpected_payload": "Alpha Vantage",
}
# Analiza LLM zastąpiona atrapą albo zrobiona przez zapasowego dostawcę — wina
# GŁÓWNEGO dostawcy LLM (którego, wie composition root).
LLM_DEGRADATION_FLAGS: frozenset[str] = frozenset({"llm_analysis_failed", "llm_failover"})


class SourceStatus(Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


@dataclass(frozen=True)
class SourceHealth:
    """Stan jednego dostawcy w cyklu.

    `affected` — symbole dotknięte awarią (zdegradowane albo pominięte przez
    otwarty obwód); `failed_requests` — odpowiedzi 429/5xx (i odcięcia 403
    źródeł alfa) policzone przez licznik HTTP, gdy nie da się ich przypisać
    symbolom."""

    source: str
    status: SourceStatus
    affected: int = 0
    failed_requests: int = 0

    @property
    def is_problem(self) -> bool:
        return self.status in (SourceStatus.DEGRADED, SourceStatus.UNAVAILABLE)


def degraded_symbols(
    results: Iterable[SymbolResult], *, llm_source: str
) -> dict[str, frozenset[str]]:
    """Dostawca → symbole cyklu, których flagi jakości wskazują jego awarię."""
    by_source: dict[str, set[str]] = {}
    for result in results:
        for flag in result.data_quality_flags:
            source = llm_source if flag in LLM_DEGRADATION_FLAGS else _FLAG_SOURCES.get(flag)
            if source is not None:
                by_source.setdefault(source, set()).add(result.symbol)
    return {source: frozenset(symbols) for source, symbols in by_source.items()}


def assess_source_health(
    *,
    enabled: Mapping[str, bool],
    circuits: Iterable[CircuitState],
    degraded: Mapping[str, Collection[str]],
    failed_requests: Mapping[str, int],
) -> tuple[SourceHealth, ...]:
    """Werdykt dla każdego dostawcy z `SOURCE_ORDER`, w tej kolejności.

    Wyłączony flagą → DISABLED (nie wysyła żądań, więc nic innego się nie liczy).
    Otwarty obwód → UNAVAILABLE z liczbą pominiętych symboli. Zdegradowane symbole
    albo błędne odpowiedzi → DEGRADED. Reszta → OK."""
    open_circuits: dict[str, set[str]] = {}
    for state in circuits:
        if state.is_open:
            open_circuits.setdefault(state.source, set()).update(state.skipped_symbols)
    rows = []
    for source in SOURCE_ORDER:
        failures = failed_requests.get(source, 0)
        if not enabled.get(source, False):
            rows.append(SourceHealth(source, SourceStatus.DISABLED))
        elif source in open_circuits:
            rows.append(
                SourceHealth(
                    source,
                    SourceStatus.UNAVAILABLE,
                    affected=len(open_circuits[source]),
                    failed_requests=failures,
                )
            )
        elif degraded.get(source) or failures:
            rows.append(
                SourceHealth(
                    source,
                    SourceStatus.DEGRADED,
                    affected=len(degraded.get(source, ())),
                    failed_requests=failures,
                )
            )
        else:
            rows.append(SourceHealth(source, SourceStatus.OK))
    return tuple(rows)
