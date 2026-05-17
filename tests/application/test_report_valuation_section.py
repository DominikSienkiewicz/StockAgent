from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import build_html_report
from src.application.report_models import SymbolResult, ValuationSection
from src.domain.value_objects import ValuationVerdict


def _make_base_symbol_result(
    valuation: ValuationSection | None = None,
) -> SymbolResult:
    return SymbolResult(
        symbol="AAPL",
        status="saved",
        delta=Decimal("0.025"),
        current_price=Decimal("150.00"),
        trend="BULLISH",
        target_price=Decimal("155.00"),
        confidence_score=0.8,
        reasoning="Strong fundamentals.",
        sentiment_score=0.3,
        valuation=valuation,
    )


def test_valuation_section_dataclass_fields() -> None:
    section = ValuationSection(
        trailing_pe=25.0,
        forward_pe=20.0,
        peg_ratio=1.2,
        eps_growth_yoy=0.15,
        verdict=ValuationVerdict.FAIR,
        fetched_at=datetime(2026, 5, 17, tzinfo=UTC),
    )
    assert section.verdict is ValuationVerdict.FAIR
    assert section.peg_ratio == 1.2


def test_symbol_result_has_valuation_slot() -> None:
    # Default should be None to keep backward compat
    rep = _make_base_symbol_result(valuation=None)
    assert rep.valuation is None


def test_html_includes_valuation_section_when_present() -> None:
    valuation = ValuationSection(
        trailing_pe=25.0,
        forward_pe=20.0,
        peg_ratio=1.2,
        eps_growth_yoy=0.15,
        verdict=ValuationVerdict.FAIR,
        fetched_at=datetime(2026, 5, 17, tzinfo=UTC),
    )
    rep = _make_base_symbol_result(valuation=valuation)
    html, _ = build_html_report([rep], datetime(2026, 5, 17, tzinfo=UTC), 1.0)
    assert "Wycena fundamentalna" in html
    assert "FAIR" in html


def test_html_omits_valuation_section_when_none() -> None:
    rep = _make_base_symbol_result(valuation=None)
    html, _ = build_html_report([rep], datetime(2026, 5, 17, tzinfo=UTC), 1.0)
    assert "Wycena fundamentalna" not in html


def test_html_shows_trailing_pe_value() -> None:
    valuation = ValuationSection(
        trailing_pe=25.0,
        forward_pe=None,
        peg_ratio=None,
        eps_growth_yoy=None,
        verdict=ValuationVerdict.UNDERVALUED,
        fetched_at=datetime(2026, 5, 17, tzinfo=UTC),
    )
    rep = _make_base_symbol_result(valuation=valuation)
    html, _ = build_html_report([rep], datetime(2026, 5, 17, tzinfo=UTC), 1.0)
    assert "25.00" in html
    assert "UNDERVALUED" in html


def test_html_shows_eps_growth_as_percent() -> None:
    valuation = ValuationSection(
        trailing_pe=None,
        forward_pe=None,
        peg_ratio=None,
        eps_growth_yoy=0.15,
        verdict=ValuationVerdict.FAIR,
        fetched_at=datetime(2026, 5, 17, tzinfo=UTC),
    )
    rep = _make_base_symbol_result(valuation=valuation)
    html, _ = build_html_report([rep], datetime(2026, 5, 17, tzinfo=UTC), 1.0)
    assert "15.0%" in html


def test_html_shows_dash_for_none_pe() -> None:
    valuation = ValuationSection(
        trailing_pe=None,
        forward_pe=None,
        peg_ratio=None,
        eps_growth_yoy=None,
        verdict=ValuationVerdict.OVERVALUED,
        fetched_at=datetime(2026, 5, 17, tzinfo=UTC),
    )
    rep = _make_base_symbol_result(valuation=valuation)
    html, _ = build_html_report([rep], datetime(2026, 5, 17, tzinfo=UTC), 1.0)
    assert "Wycena fundamentalna" in html
    assert "OVERVALUED" in html


def test_plain_text_includes_valuation_when_present() -> None:
    valuation = ValuationSection(
        trailing_pe=18.5,
        forward_pe=15.0,
        peg_ratio=0.9,
        eps_growth_yoy=0.20,
        verdict=ValuationVerdict.UNDERVALUED,
        fetched_at=datetime(2026, 5, 17, tzinfo=UTC),
    )
    rep = _make_base_symbol_result(valuation=valuation)
    _, text = build_html_report([rep], datetime(2026, 5, 17, tzinfo=UTC), 1.0)
    assert "Wycena fundamentalna" in text
    assert "UNDERVALUED" in text


def test_plain_text_omits_valuation_when_none() -> None:
    rep = _make_base_symbol_result(valuation=None)
    _, text = build_html_report([rep], datetime(2026, 5, 17, tzinfo=UTC), 1.0)
    assert "Wycena fundamentalna" not in text
