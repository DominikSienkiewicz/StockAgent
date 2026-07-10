"""Testy sekcji "💼 Twój portfel" w raporcie e-mail (roadmap #15).

Renderer jest CZYSTY: bierze agregat `Portfolio`, bieżące ceny, mapę klastrów,
`now` i opcjonalny kurs USD/PLN — i produkuje fragment HTML/tekst. Zero I/O,
zero portów. Kluczowe wymagania sprawdzane tu:

  - samosupresja (brak policzalnych pozycji → ""),
  - badge STALE (stęchły portfel = FAŁSZYWY P/L — badge to wymaganie),
  - ujemny P/L nie jest ukrywany,
  - sekcja PLN tylko gdy podano kurs,
  - ostrzeżenie o klastrze korelacji po przekroczeniu progu.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from src.application.report_user_portfolio import (
    _CLUSTER_CONCENTRATION_THRESHOLD,
    render_user_portfolio_html,
    render_user_portfolio_text,
)
from src.domain.portfolio import Portfolio, Position

_NOW = datetime(2026, 7, 10, 12, 0, 0)


def _pos(
    symbol: str,
    quantity: str,
    purchase_price: str,
    *,
    days_ago: int = 0,
) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(quantity),
        purchase_price=Decimal(purchase_price),
        purchase_date=(_NOW - timedelta(days=days_ago)).date(),
    )


def _portfolio(positions: tuple[Position, ...], *, as_of_days_ago: int) -> Portfolio:
    return Portfolio(
        positions=positions,
        as_of=_NOW - timedelta(days=as_of_days_ago),
    )


class TestSelfSuppression:
    def test_no_positions_html_is_empty(self) -> None:
        portfolio = _portfolio((), as_of_days_ago=0)

        assert render_user_portfolio_html(portfolio, {}, now=_NOW) == ""

    def test_no_positions_text_is_empty(self) -> None:
        portfolio = _portfolio((), as_of_days_ago=0)

        assert render_user_portfolio_text(portfolio, {}, now=_NOW) == ""

    def test_positions_without_prices_suppress_section(self) -> None:
        # Bez bieżącej ceny nie ma P/L ani wagi — sekcja się chowa, nie kłamie.
        portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=0)

        assert render_user_portfolio_html(portfolio, {}, now=_NOW) == ""
        assert render_user_portfolio_text(portfolio, {}, now=_NOW) == ""


class TestStaleBadge:
    def test_stale_portfolio_shows_badge_in_html(self) -> None:
        # as_of 10 dni temu > próg 7 dni → STALE → badge (P/L może być fałszywy).
        portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=10)

        html = render_user_portfolio_html(
            portfolio, {"AAPL": Decimal("110")}, now=_NOW
        )

        assert "STALE" in html
        assert "nieaktualn" in html.lower()

    def test_stale_portfolio_shows_badge_in_text(self) -> None:
        portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=10)

        text = render_user_portfolio_text(
            portfolio, {"AAPL": Decimal("110")}, now=_NOW
        )

        assert "STALE" in text

    def test_fresh_portfolio_has_no_badge_html(self) -> None:
        portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=0)

        html = render_user_portfolio_html(
            portfolio, {"AAPL": Decimal("110")}, now=_NOW
        )

        assert "STALE" not in html

    def test_fresh_portfolio_has_no_badge_text(self) -> None:
        portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=0)

        text = render_user_portfolio_text(
            portfolio, {"AAPL": Decimal("110")}, now=_NOW
        )

        assert "STALE" not in text


class TestNegativePnl:
    def test_loss_is_rendered_not_hidden_html(self) -> None:
        # Cena spadła 100 → 80: strata −200 USD na 10 szt. NIE ukrywamy jej.
        portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=0)

        html = render_user_portfolio_html(
            portfolio, {"AAPL": Decimal("80")}, now=_NOW
        )

        assert "AAPL" in html
        assert "#dc2626" in html  # kolor straty
        assert "-" in html

    def test_loss_is_rendered_not_hidden_text(self) -> None:
        portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=0)

        text = render_user_portfolio_text(
            portfolio, {"AAPL": Decimal("80")}, now=_NOW
        )

        assert "AAPL" in text
        assert "-$200.00" in text


class TestPlnConversion:
    def test_no_rate_omits_pln_section_html(self) -> None:
        portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=0)

        html = render_user_portfolio_html(
            portfolio, {"AAPL": Decimal("110")}, now=_NOW, usd_pln=None
        )

        assert "zł" not in html
        assert "PLN" not in html

    def test_no_rate_omits_pln_section_text(self) -> None:
        portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=0)

        text = render_user_portfolio_text(
            portfolio, {"AAPL": Decimal("110")}, now=_NOW, usd_pln=None
        )

        assert "zł" not in text
        assert "PLN" not in text

    def test_rate_adds_pln_amounts_html(self) -> None:
        portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=0)

        html = render_user_portfolio_html(
            portfolio,
            {"AAPL": Decimal("110")},
            now=_NOW,
            usd_pln=Decimal("4.00"),
        )

        assert "zł" in html
        assert "PLN" in html


class TestClusterWarning:
    def _three_symbol_portfolio(self, third_qty: str) -> Portfolio:
        return _portfolio(
            (
                _pos("NVDA", "10", "100"),
                _pos("AMD", "10", "100"),
                _pos("TSM", third_qty, "100"),
            ),
            as_of_days_ago=0,
        )

    def test_warns_when_one_cluster_exceeds_threshold(self) -> None:
        # NVDA+AMD+TSM w jednym klastrze "SEMI" — cały kapitał w klastrze.
        portfolio = self._three_symbol_portfolio("10")
        prices = {
            "NVDA": Decimal("100"),
            "AMD": Decimal("100"),
            "TSM": Decimal("100"),
        }
        clusters = {"NVDA": "SEMI", "AMD": "SEMI", "TSM": "SEMI"}

        html = render_user_portfolio_html(
            portfolio, prices, clusters=clusters, now=_NOW
        )
        text = render_user_portfolio_text(
            portfolio, prices, clusters=clusters, now=_NOW
        )

        assert "klaster korelacji" in html
        assert "klaster korelacji" in text

    def test_no_warning_when_cluster_below_threshold(self) -> None:
        # Każdy symbol we własnym klastrze → największy udział poniżej progu.
        portfolio = self._three_symbol_portfolio("10")
        prices = {
            "NVDA": Decimal("100"),
            "AMD": Decimal("100"),
            "TSM": Decimal("100"),
        }
        clusters = {"NVDA": "A", "AMD": "B", "TSM": "C"}

        html = render_user_portfolio_html(
            portfolio, prices, clusters=clusters, now=_NOW
        )

        assert "klaster korelacji" not in html

    def test_threshold_is_frozen_constant(self) -> None:
        # Próg zamrożony jako stała modułu — zmiana ma świadomie łamać ten test.
        assert Decimal("0.40") == _CLUSTER_CONCENTRATION_THRESHOLD


class TestEscaping:
    def test_symbol_is_escaped_in_html(self) -> None:
        portfolio = _portfolio(
            (_pos("<b>X</b>", "10", "100"),), as_of_days_ago=0
        )

        html = render_user_portfolio_html(
            portfolio, {"<b>X</b>": Decimal("110")}, now=_NOW
        )

        assert "<b>X</b>" not in html
        assert "&lt;b&gt;X&lt;/b&gt;" in html
