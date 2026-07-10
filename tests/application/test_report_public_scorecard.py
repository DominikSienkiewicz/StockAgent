"""Testy publicznego scorecardu kalibracji (roadmap #10).

Kluczowy test to TWARDA asercja anty-PII: renderer dostaje dane niosące symbol
watchlisty, adres e-mail i kwoty (ceny), a strona publiczna NIE MOŻE ich
ujawnić. Publikujemy wyłącznie agregaty: N, hit-rate, ECE, krzywą kalibracji.
"""

from __future__ import annotations

from decimal import Decimal

from src.application.report_models import ResolvedPrediction
from src.application.report_public_scorecard import render_public_scorecard_html
from src.domain.equity_curve import equity_curve


def _resolved(
    symbol: str,
    trend: str,
    correct: bool,
    *,
    reasoning: str | None = None,
    price_at: Decimal | None = None,
    actual: Decimal | None = None,
) -> ResolvedPrediction:
    """Skrót do budowy zamkniętej predykcji w testach."""
    return ResolvedPrediction(
        symbol=symbol,
        predicted_trend=trend,
        is_correct=correct,
        reasoning=reasoning,
        price_at_prediction=price_at,
        actual_price=actual,
    )


def test_empty_inputs_self_suppress() -> None:
    """Brak jakichkolwiek danych → pusty string (nie publikujemy pustej strony)."""
    assert render_public_scorecard_html(None, None, None) == ""
    assert render_public_scorecard_html([], None, []) == ""


def test_renders_standalone_html_page() -> None:
    """Kompletna, samodzielna strona HTML (doctype + head + body)."""
    resolved = [_resolved("AAA", "BULLISH", True)]
    curve = equity_curve([("BULLISH", 0.03)])
    html = render_public_scorecard_html(resolved, curve, [(0.8, True)])
    assert html != ""
    lower = html.lower()
    assert "<!doctype html>" in lower
    assert "<html" in lower
    assert "</html>" in lower
    assert "<head" in lower
    assert "<body" in lower


def test_shows_aggregate_metrics_n_and_hit_rate() -> None:
    """Publikujemy liczbę predykcji N i hit-rate jako agregaty."""
    resolved = [
        _resolved("AAA", "BULLISH", True),
        _resolved("BBB", "BEARISH", True),
        _resolved("CCC", "BULLISH", False),
        _resolved("DDD", "BEARISH", True),
    ]
    curve = equity_curve(
        [("BULLISH", 0.02), ("BEARISH", -0.01), ("BULLISH", -0.02), ("BEARISH", -0.03)]
    )
    html = render_public_scorecard_html(resolved, curve, None)
    # N = 4 predykcje.
    assert "4" in html
    # hit-rate = 3/4 = 75%.
    assert "75" in html


def test_calibration_curve_rendered_as_div_sparkline_with_ece() -> None:
    """Krzywa kalibracji jako div-sparkline + wartość ECE na stronie."""
    samples = [
        (0.55, True),
        (0.55, False),
        (0.95, True),
        (0.95, True),
        (0.15, False),
    ]
    resolved = [_resolved("AAA", "BULLISH", True)]
    html = render_public_scorecard_html(resolved, None, samples)
    # Technika div-sparkline: słupki jako <div> o zadanej wysokości w px.
    assert "<div" in html
    assert "height:" in html.replace(" ", "")
    # ECE jest nazwane wprost na publicznej stronie.
    assert "ECE" in html


def test_equity_metrics_read_from_dataclass_fields() -> None:
    """Metryki equity pochodzą z PÓL EquityCurve (total_return, max_drawdown)."""
    curve = equity_curve([("BULLISH", 0.10), ("BEARISH", 0.05)])
    resolved = [
        _resolved("AAA", "BULLISH", True),
        _resolved("BBB", "BEARISH", False),
    ]
    html = render_public_scorecard_html(resolved, curve, None)
    # total_return dla [(+10%),(sygnał -1 * +5% = -5%)]: 1.1*0.95-1 = +4.5%.
    assert "+4.5%" in html


def test_methodology_section_is_honest_polish_copy() -> None:
    """Sekcja „Metodologia" po polsku — uczciwie tłumaczy okresy słabości."""
    resolved = [_resolved("AAA", "BULLISH", True)]
    html = render_public_scorecard_html(resolved, None, [(0.8, True)])
    assert "Metodologia" in html
    # Uczciwość: pokazujemy też własne pomyłki / słabe okresy.
    assert "pomyłk" in html.lower() or "słab" in html.lower()


def test_hard_anti_pii_no_symbols_emails_or_amounts_leak() -> None:
    """TWARDY test anty-PII — najważniejszy w tym zadaniu.

    Renderer dostaje dane z symbolem „NVDA", adresem e-mail i kwotami (ceny).
    Publiczna strona nie może zawierać ŻADNEGO z tych wycieków: ani symbolu,
    ani znaku „@" (e-mail), ani znaczników kwot („$", „zł", „PLN").
    """
    leaky_email = "insider@example.com"
    resolved = [
        _resolved(
            "NVDA",
            "BULLISH",
            True,
            reasoning=f"Kup wg cynku od {leaky_email} — cena 1234 USD",
            price_at=Decimal("1234.56"),
            actual=Decimal("1300.00"),
        ),
        _resolved(
            "TSLA",
            "BEARISH",
            False,
            reasoning="Short 999 PLN",
            price_at=Decimal("250.00"),
            actual=Decimal("260.00"),
        ),
    ]
    curve = equity_curve([("BULLISH", 0.05), ("BEARISH", 0.04)])
    samples = [(0.8, True), (0.6, False)]
    html = render_public_scorecard_html(resolved, curve, samples)

    assert html != ""
    # Zero symboli watchlisty.
    assert "NVDA" not in html
    assert "TSLA" not in html
    # Zero e-maili (żaden znak „@" nie może wyciec).
    assert leaky_email not in html
    assert "@" not in html
    # Zero kwot / walut.
    assert "$" not in html
    assert "zł" not in html
    assert "PLN" not in html
    assert "1234" not in html
    assert "1300" not in html
    assert "250" not in html
