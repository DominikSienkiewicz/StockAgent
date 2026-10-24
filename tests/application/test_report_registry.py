"""ARCH-12: rejestr sekcji maila zamiast slotów w `report_builder`.

Mail to krotka `REPORT_SECTIONS` w kolejności, w jakiej go drukuje — jedna dla
HTML i wersji tekstowej. Każda sekcja ma renderer HTML i tekstu (sekcja tylko
w HTML świadomie bierze `no_text`). Że złożenie z rejestru daje mail bajt w bajt
jak dawne sloty, pilnuje golden snapshot (`test_report_builder_snapshot.py`).

Każde pole `ReportContext` ma dokładnie jednego konsumenta: jedną sekcję
(`ReportSection.context_fields`) albo wspólne wejście sekcji
(`SHARED_CONTEXT_FIELDS` — pola, z których `build_html_report` liczy
`SectionInput`, i pola czytane przez kilka sekcji). Deklaracje sprawdza
pośrednik kontekstu, który notuje odczyty na pełnym cyklu z golden snapshotu:
sekcja nie czyta pola innej sekcji, a żadne pole nie zostaje bez czytelnika.
Nowa sekcja = moduł, wpis w rejestrze i jedno pole w jej `context_fields`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

from src.application import report_builder
from src.application.report_builder import ReportContext, SymbolResult, to_symbol_result
from src.application.reporting.registry import REPORT_SECTIONS
from src.application.reporting.section import (
    SHARED_CONTEXT_FIELDS,
    ReportSection,
    SectionInput,
    no_text,
)
from src.domain.value_objects import AssetType
from tests.application.test_report_builder_snapshot import (
    DURATION_SECONDS,
    STARTED,
    _crypto_saved_raw,
    _degraded_stock_raw,
    _full_context,
    _ignored_raw,
    _mapped,
    _saved_stock_raw,
    _unsupported,
)

CONTEXT_FIELDS = frozenset(field.name for field in dataclasses.fields(ReportContext))
SHARED_INPUT = "SectionInput"

MAIL_ORDER = (
    "preheader",
    "masthead",
    "quota_banner",
    "source_health",
    "lead",
    "onboarding",
    "session",
    "first_screen",
    "between_mails",
    "user_portfolio",
    "resolved_predictions",
    "consensus_shift",
    "rebaselined",
    "trade_signals",
    "risk_signals",
    "predictions",
    "crypto",
    "this_week",
    "backoffice",
    "risk_watch",
    "portfolio_radar",
    "alpha",
    "track_record",
    "accuracy_history",
    "model_scorecard",
    "persona_leaderboard",
    "council_history",
    "mood",
    "delta_chart",
    "reflections",
    "suggestions",
    "correlation_chart",
    "finops",
    "budget_exhausted",
    "ignored",
    "errors",
    "footer",
)


def test_registry_covers_every_slot_with_html_and_text() -> None:
    assert tuple(section.key for section in REPORT_SECTIONS) == MAIL_ORDER
    for section in REPORT_SECTIONS:
        assert callable(section.render_html), section.key
        assert callable(section.render_text), section.key


# ---------------------------------------------------------------------------
# Każde pole ReportContext ma dokładnie jednego konsumenta
# ---------------------------------------------------------------------------


class _RecordingContext:
    """Pośrednik `ReportContext`: oddaje pola bez zmian i notuje, które odczytano."""

    def __init__(self, context: ReportContext) -> None:
        self._context = context
        self._read: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        self._read.add(name)
        return getattr(self._context, name)

    def take(self) -> set[str]:
        read, self._read = self._read, set()
        return read


def _full_cycle_results() -> list[SymbolResult]:
    """Wyniki scenariusza `full_cycle` z golden snapshotu — każdy status naraz."""
    return [
        _mapped("NVDA", _saved_stock_raw()),
        _mapped("AMD", _degraded_stock_raw()),
        _mapped("BTC", _crypto_saved_raw()),
        _mapped("ETH", _ignored_raw("ETH", AssetType.CRYPTO, "3200", "0.012")),
        _mapped("MSFT", _ignored_raw("MSFT", AssetType.STOCK, "410.00", "0.006")),
        _unsupported("XDWD.DE"),
        to_symbol_result("ASML", None, error="HTTPError: 429 Too Many Requests", now=STARTED),
    ]


def _reads_per_consumer(
    sections: tuple[ReportSection, ...],
) -> dict[str, set[str]]:
    """Konsument → pola `ReportContext` odczytane przy renderze pełnego cyklu
    (HTML i tekst). Wspólne wejście liczy się jako osobny konsument."""
    recorder = _RecordingContext(_full_context())
    data: SectionInput = report_builder._section_input(
        _full_cycle_results(), STARTED, DURATION_SECONDS, cast(ReportContext, recorder)
    )
    reads = {SHARED_INPUT: recorder.take()}
    for section in sections:
        section.render_html(data)
        section.render_text(data)
        reads[section.key] = recorder.take()
    return reads


def _foreign_reads(sections: tuple[ReportSection, ...]) -> dict[str, set[str]]:
    """Konsument → odczytane pola spoza jego deklaracji (i spoza wspólnych)."""
    declared = {section.key: section.context_fields for section in sections}
    foreign = {}
    for consumer, read in _reads_per_consumer(sections).items():
        allowed = SHARED_CONTEXT_FIELDS | declared.get(consumer, frozenset())
        if read - allowed:
            foreign[consumer] = read - allowed
    return foreign


def test_every_report_context_field_has_exactly_one_consumer() -> None:
    consumers: dict[str, list[str]] = {name: [] for name in CONTEXT_FIELDS}
    for name in SHARED_CONTEXT_FIELDS:
        consumers.setdefault(name, []).append(SHARED_INPUT)
    for section in REPORT_SECTIONS:
        for name in section.context_fields:
            consumers.setdefault(name, []).append(section.key)

    assert {
        name: owners for name, owners in consumers.items() if len(owners) != 1
    } == {}
    assert set(consumers) == CONTEXT_FIELDS, "deklaracja pola, którego ReportContext nie ma"


def test_sections_read_only_their_own_or_shared_context_fields() -> None:
    assert _foreign_reads(REPORT_SECTIONS) == {}


def test_every_report_context_field_is_read_by_its_consumer() -> None:
    reads = _reads_per_consumer(REPORT_SECTIONS)
    unread_own = {
        section.key: section.context_fields - reads[section.key]
        for section in REPORT_SECTIONS
        if section.context_fields - reads[section.key]
    }
    read_anywhere = set().union(*reads.values())

    assert unread_own == {}
    assert SHARED_CONTEXT_FIELDS - read_anywhere == set()


def test_the_guard_flags_a_section_reading_another_sections_field() -> None:
    def peeking_html(data: SectionInput) -> str:
        return str(data.context.cycle_cost)

    peeking = ReportSection(key="peeking", render_html=peeking_html, render_text=no_text)

    assert _foreign_reads((peeking,)) == {"peeking": {"cycle_cost"}}
