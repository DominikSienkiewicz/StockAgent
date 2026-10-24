"""Testy sekcji "💼 Twój portfel" w raporcie e-mail (R-15).

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

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from src.application.report_user_portfolio import (
    _CLUSTER_CONCENTRATION_THRESHOLD,
    portfolio_change_label,
    render_portfolio_change_box_html,
    render_portfolio_change_text_line,
    render_user_portfolio_html,
    render_user_portfolio_text,
)
from src.domain.portfolio import PlnRates, Portfolio, PortfolioChange, Position

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
        assert "-$200,00" in text


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

        assert "to jeden sektor «SEMI»" in html
        assert "to jeden sektor «SEMI»" in text

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

        assert "to jeden sektor" not in html

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


class TestUnpricedPositions:
    """PORT-03: pozycja bez ceny w tym cyklu (np. XDWD.DE — UNSUPPORTED_PRICE)
    znikała z tabeli bez śladu, a sumy podpisane „Twój portfel” zawyżały wagi
    pozostałych pozycji."""

    _PRICES = {"NVDA": Decimal("120")}

    @staticmethod
    def _portfolio() -> Portfolio:
        # NVDA koszt 1000 (z ceną), XDWD.DE koszt 500 (bez ceny) → 33.3% kosztu.
        return _portfolio(
            (_pos("NVDA", "10", "100"), _pos("XDWD.DE", "5", "100")),
            as_of_days_ago=0,
        )

    def test_unpriced_positions_are_listed_with_cost_share(self) -> None:
        html = render_user_portfolio_html(self._portfolio(), self._PRICES, now=_NOW)

        assert "Bez wyceny" in html
        assert "XDWD.DE" in html
        assert "33,3% kosztu" in html
        assert "bez pozycji niewycenionych" in html

    def test_unpriced_positions_are_listed_in_text(self) -> None:
        text = render_user_portfolio_text(self._portfolio(), self._PRICES, now=_NOW)

        assert "Bez wyceny w tym cyklu: XDWD.DE (≈ 33,3% kosztu)" in text
        assert "bez pozycji niewycenionych" in text

    def test_fully_priced_portfolio_has_no_unpriced_note(self) -> None:
        portfolio = _portfolio((_pos("NVDA", "10", "100"),), as_of_days_ago=0)

        html = render_user_portfolio_html(portfolio, self._PRICES, now=_NOW)
        text = render_user_portfolio_text(portfolio, self._PRICES, now=_NOW)

        assert "Bez wyceny" not in html
        assert "Bez wyceny" not in text
        assert "niewycenionych" not in html


def test_pln_rate_is_written_with_a_decimal_comma() -> None:
    # MAIL-58: kurs USD/PLN w sekcji portfela po polsku — „4,00”, nie „4.00”.
    portfolio = _portfolio((_pos("AAPL", "10", "100"),), as_of_days_ago=0)
    args = (portfolio, {"AAPL": Decimal("110")})

    html = render_user_portfolio_html(*args, now=_NOW, usd_pln=Decimal("4.00"))
    text = render_user_portfolio_text(*args, now=_NOW, usd_pln=Decimal("4.00"))

    assert "kurs USD/PLN 4,00" in html
    assert "(USD/PLN 4,00)" in text


class TestPortfolioChange:
    """PORT-06: zmiana wartości portfela od ostatniego maila (kwota i %) z top
    kontrybutorami — w sekcji 💼 i jako box na pierwszym ekranie."""

    _PORTFOLIO = _portfolio(
        (_pos("NVDA", "75", "8"), _pos("AAPL", "25", "9"), _pos("XDWD.DE", "5", "100")),
        as_of_days_ago=0,
    )
    _PREVIOUS = {"NVDA": Decimal("10"), "AAPL": Decimal("10")}
    _CURRENT = {"NVDA": Decimal("11"), "AAPL": Decimal("9.80")}

    def _change(self) -> PortfolioChange:
        change = self._PORTFOLIO.change_since(self._PREVIOUS, self._CURRENT)
        assert change is not None
        return change

    def test_section_shows_change_amount_pct_and_contributors_html(self) -> None:
        html = render_user_portfolio_html(
            self._PORTFOLIO, self._CURRENT, now=_NOW, change=self._change()
        )

        assert "Zmiana od ostatniego maila" in html
        assert "+$70,00 (+7,0%)" in html
        assert "NVDA +$75,00" in html
        assert "AAPL -$5,00" in html
        assert "bez punktu odniesienia: XDWD.DE" in html

    def test_section_shows_change_in_text(self) -> None:
        text = render_user_portfolio_text(
            self._PORTFOLIO, self._CURRENT, now=_NOW, change=self._change()
        )

        assert "Zmiana od ostatniego maila: +$70,00 (+7,0%)" in text
        assert "NVDA +$75,00, AAPL -$5,00" in text

    def test_change_in_pln_when_the_rate_is_known(self) -> None:
        html = render_user_portfolio_html(
            self._PORTFOLIO, self._CURRENT, now=_NOW, change=self._change(),
            usd_pln=Decimal("4.00"),
        )
        text = render_user_portfolio_text(
            self._PORTFOLIO, self._CURRENT, now=_NOW, change=self._change(),
            usd_pln=Decimal("4.00"),
        )

        assert "+280,00 zł" in html
        assert "+280,00 zł" in text

    def test_after_a_pause_the_section_says_since_when(self) -> None:
        html = render_user_portfolio_html(
            self._PORTFOLIO, self._CURRENT, now=_NOW, change=self._change(),
            change_label="od 19.09.2026",
        )

        assert "Zmiana od 19.09.2026" in html

    def test_no_change_means_no_change_line(self) -> None:
        html = render_user_portfolio_html(self._PORTFOLIO, self._CURRENT, now=_NOW)
        text = render_user_portfolio_text(self._PORTFOLIO, self._CURRENT, now=_NOW)

        assert "Zmiana od" not in html
        assert "Zmiana od" not in text

    def test_first_screen_box_carries_pct_and_amount(self) -> None:
        box = render_portfolio_change_box_html(self._change(), usd_pln=Decimal("4.00"))

        assert "Twój portfel od ostatniego maila" in box
        assert "+7,0%" in box
        assert "+$70,00" in box
        assert "+280,00 zł" in box

    def test_first_screen_text_line(self) -> None:
        line = render_portfolio_change_text_line(self._change(), label="od 19.09.2026")

        assert line == "Twój portfel od 19.09.2026: +7,0% (+$70,00)"

    def test_label_is_since_last_email_for_a_fresh_reference(self) -> None:
        now = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

        assert portfolio_change_label(now - timedelta(hours=24), now, "Europe/Warsaw") == (
            "od ostatniego maila"
        )
        assert portfolio_change_label(None, now, "Europe/Warsaw") == "od ostatniego maila"

    def test_label_names_the_date_after_a_pause(self) -> None:
        now = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)
        reference = datetime(2026, 9, 19, 17, 30, tzinfo=UTC)

        assert portfolio_change_label(reference, now, "Europe/Warsaw") == "od 19.09.2026"


# --- PORT-11: „P/L zł” i „w tym kurs” po kursie NBP z dnia zakupu -------------

_BOUGHT = date(2025, 3, 3)


def _lot(symbol: str, qty: str, price: str, currency: str = "USD",
         day: date | None = _BOUGHT) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(qty),
        purchase_price=Decimal(price),
        purchase_date=day,
        currency=currency,
    )


_RATES = PlnRates(
    current={"USD": Decimal("4.40"), "EUR": Decimal("4.30")},
    on_purchase={("USD", _BOUGHT): Decimal("4.00"), ("EUR", _BOUGHT): Decimal("4.20")},
)


class TestPlnPnlColumns:
    def test_columns_show_pnl_in_zloty_and_the_fx_part(self) -> None:
        # 10 × 100 USD przy 4,00 → dziś 100 USD przy 4,40: +400 zł, w całości z kursu.
        portfolio = _portfolio((_lot("AAPL", "10", "100"),), as_of_days_ago=0)

        html = render_user_portfolio_html(
            portfolio, {"AAPL": Decimal("100")}, now=_NOW, pln_rates=_RATES
        )
        text = render_user_portfolio_text(
            portfolio, {"AAPL": Decimal("100")}, now=_NOW, pln_rates=_RATES
        )

        assert ">P/L zł</th>" in html
        assert ">w tym kurs</th>" in html
        assert "P/L (PLN)" not in html
        assert "+400,00\u00a0zł" in html
        assert "| P/L zł +400,00\u00a0zł (w tym kurs +400,00\u00a0zł)" in text

    def test_unknown_purchase_date_is_said_out_loud(self) -> None:
        portfolio = _portfolio((_lot("AAPL", "10", "100", day=None),), as_of_days_ago=0)

        html = render_user_portfolio_html(
            portfolio, {"AAPL": Decimal("110")}, now=_NOW, pln_rates=_RATES
        )
        text = render_user_portfolio_text(
            portfolio, {"AAPL": Decimal("110")}, now=_NOW, pln_rates=_RATES
        )

        assert "kurs z dnia zakupu nieznany" in html
        assert "| P/L zł — (kurs z dnia zakupu nieznany)" in text
        # Brak cichego przybliżenia bieżącym kursem: (110 − 100) × 10 × 4,40 = 440 zł.
        assert "+440,00\u00a0zł" not in html
        assert "+440,00\u00a0zł" not in text

    def test_missing_nbp_rate_of_the_purchase_day_has_its_own_note(self) -> None:
        portfolio = _portfolio((_lot("AAPL", "10", "100", day=date(2024, 1, 2)),),
                               as_of_days_ago=0)

        text = render_user_portfolio_text(
            portfolio, {"AAPL": Decimal("110")}, now=_NOW, pln_rates=_RATES
        )

        assert "(brak kursu NBP z dnia zakupu)" in text

    def test_eur_position_is_valued_in_euro_and_converted_at_eur_rates(self) -> None:
        portfolio = _portfolio((_lot("SAP.DE", "4", "200", "EUR"),), as_of_days_ago=0)

        html = render_user_portfolio_html(
            portfolio, {"SAP.DE": Decimal("200")}, now=_NOW, pln_rates=_RATES
        )

        assert "800,00 EUR" in html
        assert "$800,00" not in html
        # 4 × 200 × (4,30 − 4,20) = +80 zł, w całości z kursu.
        assert "+80,00\u00a0zł" in html

    def test_totals_in_zloty_name_the_rates_and_the_fx_part(self) -> None:
        portfolio = _portfolio(
            (_lot("AAPL", "10", "100"), _lot("SAP.DE", "4", "200", "EUR")), as_of_days_ago=0
        )
        prices = {"AAPL": Decimal("100"), "SAP.DE": Decimal("200")}

        text = render_user_portfolio_text(portfolio, prices, now=_NOW, pln_rates=_RATES)

        assert (
            "  W PLN (kursy NBP: EUR 4,30 · USD 4,40): wartość 7\u00a0840,00\u00a0zł, "
            "P/L +480,00\u00a0zł (w tym kurs +480,00\u00a0zł)"
        ) in text.splitlines()

    def test_totals_list_the_positions_left_out_of_the_pnl_in_zloty(self) -> None:
        portfolio = _portfolio(
            (_lot("AAPL", "10", "100"), _lot("NVDA", "1", "100", day=None)), as_of_days_ago=0
        )
        prices = {"AAPL": Decimal("100"), "NVDA": Decimal("100")}

        html = render_user_portfolio_html(portfolio, prices, now=_NOW, pln_rates=_RATES)
        text = render_user_portfolio_text(portfolio, prices, now=_NOW, pln_rates=_RATES)

        assert "bez P/L w zł: NVDA" in html
        assert text.splitlines()[-1].endswith("; bez P/L w zł: NVDA")

    def test_mixed_currencies_have_no_single_native_total(self) -> None:
        portfolio = _portfolio(
            (_lot("AAPL", "10", "100"), _lot("SAP.DE", "4", "200", "EUR")), as_of_days_ago=0
        )
        prices = {"AAPL": Decimal("100"), "SAP.DE": Decimal("200")}

        text = render_user_portfolio_text(portfolio, prices, now=_NOW, pln_rates=_RATES)

        assert "  Łącznie: pozycje w różnych walutach (EUR, USD)" in text.splitlines()
        assert "$1 800,00" not in text

    def test_pln_rates_win_over_the_legacy_usd_rate(self) -> None:
        portfolio = _portfolio((_lot("AAPL", "10", "100"),), as_of_days_ago=0)

        html = render_user_portfolio_html(
            portfolio, {"AAPL": Decimal("100")}, now=_NOW,
            usd_pln=Decimal("3.00"), pln_rates=_RATES,
        )

        assert "USD/PLN 3,00" not in html
        assert "kursy NBP: USD 4,40" in html


class TestChangeInPln:
    """Zmiana portfela policzona w złotych: kwoty w zł, bez dopisku po kursie USD/PLN."""

    @staticmethod
    def _change() -> PortfolioChange:
        # Ceny już w zł (cena × kurs): 400 → 440 zł × 10 sztuk = +400 zł.
        portfolio = Portfolio(positions=(_lot("NVDA", "10", "100"),), as_of=_NOW)
        change = portfolio.change_since({"NVDA": Decimal("400")}, {"NVDA": Decimal("440")})
        assert change is not None
        return change

    def test_box_and_line_show_zloty_once(self) -> None:
        change = self._change()

        box = render_portfolio_change_box_html(change, usd_pln=Decimal("4.00"), currency="PLN")
        line = render_portfolio_change_text_line(
            change, usd_pln=Decimal("4.00"), currency="PLN"
        )

        assert "+400,00 zł" in box
        assert "1 600,00" not in box
        assert line == "Twój portfel od ostatniego maila: +10,0% (+400,00 zł)"

    def test_section_line_lists_contributors_in_zloty(self) -> None:
        portfolio = _portfolio((_lot("NVDA", "10", "100"),), as_of_days_ago=0)

        text = render_user_portfolio_text(
            portfolio, {"NVDA": Decimal("110")}, now=_NOW, pln_rates=_RATES,
            change=self._change(), change_currency="PLN",
        )

        assert (
            "  Zmiana od ostatniego maila: +400,00 zł (+10,0%) · najwięcej: NVDA +400,00 zł"
        ) in text.splitlines()

    def test_unrated_position_is_named_under_the_table(self) -> None:
        portfolio = _portfolio(
            (_lot("NVDA", "10", "100"), _lot("VUSA.L", "4", "80", "GBP")), as_of_days_ago=0
        )
        prices = {"NVDA": Decimal("110"), "VUSA.L": Decimal("90")}

        html = render_user_portfolio_html(portfolio, prices, now=_NOW, pln_rates=_RATES)

        assert "Bez kursu NBP: VUSA.L — poza wagami i sumami w zł" in html
        assert "100,0%" in html
