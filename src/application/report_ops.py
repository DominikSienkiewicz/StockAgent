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
) -> str:
    """Summary cyklu Fast Loopa.

    `delivered`: True = mail wyszedł, False = wysyłka padła, None = brak kanału
    e-mail (np. `notifications_enabled = false`). `cycle_cost` None = przebieg
    na atrapach (FIN-35), więc kosztu nie ma czego pokazywać."""
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
        ]
    )


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
) -> str:
    """Summary Slow Loopa: statusy ewaluacji per symbol jako liczby i wynik modelu
    wspólnego. `pooled_status` None = trening wspólny rzucił albo nie ruszył."""
    return "\n".join(
        [
            "## Slow Loop training summary",
            "",
            _table(
                ("Metric", "Value"),
                [
                    ("Feature store refresh", "ok" if feature_store_refreshed else "failed"),
                    ("Pooled model", pooled_status or "failed"),
                    (
                        "Pooled samples",
                        _NOT_AVAILABLE if pooled_samples is None else str(pooled_samples),
                    ),
                    ("Failures", f"{failures} / {total}"),
                ],
            ),
            "### Per-symbol evaluation by status",
            "",
            _count_table("Status", Counter(dict(symbol_statuses))),
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
