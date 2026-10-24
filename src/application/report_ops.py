"""NIEZ-22: podsumowanie przebiegu do `GITHUB_STEP_SUMMARY` (strona runu w Actions).

Repo jest publiczne, więc strona runu też: renderery biorą pełne wyniki cyklu, ale
wypisują WYŁĄCZNIE agregaty — liczby per status, powód pominięcia i typ wyjątku
(nazwa klasy), płatne wywołania per źródło, koszt, alerty kwot per poziom, czas
i stan wysyłki maila. Inwariant jak w publicznym scorecardzie
(`report_public_scorecard`): zero symboli watchlisty, zero adresów e-mail, zero
treści błędów, reasoning, cen, kwot i pozycji portfela.

Funkcje są czyste — bez portów i I/O. Plik z env `GITHUB_STEP_SUMMARY` dopisują
wyłącznie entry pointy (`main_agent.append_step_summary`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from src.application.report_models import SymbolResult
from src.domain.direction_variant import DirectionReport
from src.domain.finops import CycleCost
from src.domain.quota import QuotaAlert

UNKNOWN_ERROR_TYPE = "unknown"
_NONE_ROW = "(none)"
_NOT_AVAILABLE = "n/a"


def render_cycle_summary_markdown(
    results: Sequence[SymbolResult],
    *,
    paid_calls: Mapping[str, int],
    cycle_cost: CycleCost | None,
    failures: int,
    duration_s: float,
    quota_alerts: Sequence[QuotaAlert],
    delivered: bool | None,
    http_calls: Mapping[str, Mapping[str, int]] | None = None,
) -> str:
    """Summary cyklu Fast Loopa.

    `delivered`: True = mail wyszedł, False = wysyłka padła, None = brak kanału
    e-mail (np. `notifications_enabled = false`). `cycle_cost` None = przebieg
    na atrapach (FIN-35), więc kosztu nie ma czego pokazywać. `http_calls`
    (NIEZ-24) — żądania per host; bez miernika sekcji nie ma."""
    statuses = Counter(result.status for result in results)
    skip_reasons = Counter(
        result.skip_reason.value for result in results if result.skip_reason is not None
    )
    error_types = Counter(
        result.error_type or UNKNOWN_ERROR_TYPE for result in results if result.status == "error"
    )
    severities = Counter(alert.severity.value for alert in quota_alerts)
    overview = _table(
        ("Metric", "Value"),
        [
            ("Duration", f"{duration_s:.1f} s"),
            ("Symbols", str(len(results))),
            ("Failures", str(failures)),
            ("Report email", _delivery_label(delivered)),
            ("Cycle cost", _cost_label(cycle_cost)),
        ],
    )
    return "\n".join(
        [
            "## Fast Loop cycle summary",
            "",
            overview,
            "### Symbols by status",
            "",
            _count_table("Status", statuses),
            "### Skip reasons",
            "",
            _count_table("Skip reason", skip_reasons),
            "### Errors by exception type",
            "",
            _count_table("Exception type", error_types),
            "### Paid calls by source",
            "",
            _count_table("Source", Counter(dict(paid_calls))),
            "### Quota alerts by severity",
            "",
            _count_table("Severity", severities),
            *_http_calls_section(http_calls),
        ]
    )


_HTTP_COLUMNS = ("2xx", "4xx", "429", "5xx", "retries")


def _http_calls_section(http_calls: Mapping[str, Mapping[str, int]] | None) -> list[str]:
    """NIEZ-24: tabela żądań per host (sam host — ścieżka i query niosą klucze)."""
    if not http_calls:
        return []
    lines = [
        "| Host | Requests | 2xx | 4xx | 429 | 5xx | Retries |",
        "|---|---|---|---|---|---|---|",
    ]
    for host, counts in sorted(http_calls.items()):
        cells = [host, *(str(counts.get(key, 0)) for key in ("requests", *_HTTP_COLUMNS))]
        lines.append("| " + " | ".join(cells) + " |")
    return ["### HTTP requests by host", "", "\n".join(lines) + "\n"]


def render_watch_summary_markdown(
    *, checked: int, stale: int, sent: int, failures: int, total: int
) -> str:
    """Summary Shock Watcha: same liczby — sprawdzone referencje, stęchłe, wysłane."""
    return "\n".join(
        [
            "## Shock Watch summary",
            "",
            _table(
                ("Metric", "Value"),
                [
                    ("Checked", str(checked)),
                    ("Stale reference", str(stale)),
                    ("Alerts sent", str(sent)),
                    ("Failures", f"{failures} / {total}"),
                ],
            ),
        ]
    )


def render_trainer_summary_markdown(
    *,
    symbol_statuses: Mapping[str, int],
    pooled_status: str | None,
    pooled_samples: int | None,
    failures: int,
    total: int,
    feature_store_refreshed: bool = True,
    pooled_label_overlap: tuple[int, int] | None = None,
) -> str:
    """Summary Slow Loopa: statusy ewaluacji per symbol jako liczby i wynik modelu
    wspólnego. `pooled_status` None = trening wspólny rzucił albo nie ruszył.
    `pooled_label_overlap` (ML-25) — (nakładające się, zmierzone) etykiety próbki
    modelu wspólnego; None = próbka bez chwili rozliczenia, wiersza nie ma."""
    rows = [
        ("Feature store refresh", "ok" if feature_store_refreshed else "failed"),
        ("Pooled model", pooled_status or "failed"),
        (
            "Pooled samples",
            _NOT_AVAILABLE if pooled_samples is None else str(pooled_samples),
        ),
    ]
    if pooled_label_overlap is not None:
        overlapping, measured = pooled_label_overlap
        rows.append(("Label overlaps (settle after the next row)", f"{overlapping} / {measured}"))
    rows.append(("Failures", f"{failures} / {total}"))
    return "\n".join(
        [
            "## Slow Loop training summary",
            "",
            _table(("Metric", "Value"), rows),
            "### Per-symbol evaluation by status",
            "",
            _count_table("Status", Counter(dict(symbol_statuses))),
        ]
    )


def _pair(first: float | None, second: float | None) -> str:
    def number(value: float | None) -> str:
        return _NOT_AVAILABLE if value is None else f"{value:.4f}"

    return f"{number(first)} / {number(second)}"


def render_direction_variant_markdown(report: DirectionReport) -> str:
    """ML-30: wariant binarny P(wzrost poza martwą strefą) w summary Slow Loopa —
    Brier i log loss klasyfikatora obok stopy bazowej, liczności i werdykt. Sama
    ewaluacja: serwowany model regresji się nie zmienia."""
    summary = report.summary
    rows = [
        ("Status", report.outcome.value),
        (f"Rows outside the dead band (±{report.dead_band:g})", str(report.n_samples)),
        ("Rows inside the dead band", str(report.n_in_dead_band)),
        (
            "Out-of-sample points",
            _NOT_AVAILABLE if summary is None else str(summary.n_oos_points),
        ),
        (
            "Brier (classifier / base rate)",
            _pair(None, None) if summary is None else _pair(summary.brier, summary.base_rate_brier),
        ),
        (
            "Log loss (classifier / base rate)",
            _pair(None, None)
            if summary is None
            else _pair(summary.log_loss, summary.base_rate_log_loss),
        ),
        ("Verdict", report.reason or _NOT_AVAILABLE),
    ]
    return "\n".join(
        [
            "### Direction classifier (ML-30)",
            "",
            _table(("Metric", "Value"), rows),
            "",
            "P(12h return above the dead band); the variant never replaces the served "
            "regression model.",
        ]
    )


def _delivery_label(delivered: bool | None) -> str:
    if delivered is None:
        return "not configured"
    return "delivered" if delivered else "not delivered"


def _cost_label(cost: CycleCost | None) -> str:
    if cost is None:
        return _NOT_AVAILABLE
    if cost.from_tokens:
        basis = "from tokens" if all(line.from_tokens for line in cost.lines) else "mixed"
    else:
        basis = "estimate"
    return f"${cost.total_usd:.4f} ({basis})"


def _count_table(label: str, counts: Counter[str]) -> str:
    """Wiersze malejąco po liczbie, remisy alfabetycznie; pusta tabela = `(none) | 0`."""
    rows = [(key, str(count)) for key, count in counts.items() if count > 0]
    if not rows:
        rows = [(_NONE_ROW, "0")]
    return _table((label, "Count"), sorted(rows, key=lambda row: (-int(row[1]), row[0])))


def _table(header: tuple[str, str], rows: Iterable[tuple[str, str]]) -> str:
    lines = [f"| {header[0]} | {header[1]} |", "|---|---|"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    return "\n".join(lines) + "\n"
