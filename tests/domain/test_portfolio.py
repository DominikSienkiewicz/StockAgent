"""Testy domeny portfela użytkownika — realne wagi, P/L i klastry.

Czysta logika, stdlib only. Weryfikujemy trzy naprawione luki:
  - `weights()` liczy REALNE wagi wg wartości rynkowej (nie równe),
    renormalizowane po symbolach, dla których jest bieżąca cena.
  - `unrealized_pnl()` zwraca P/L per pozycja + łącznie na `Decimal`.
  - `cluster_exposure()` sumuje wagi po mapie symbol→klaster korelacji.

Osobno zamrażamy regułę świeżości (STALE) — stęchły portfel daje fałszywy
P/L, więc badge STALE jest wymaganiem, a próg musi być stały.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from src.domain.portfolio import (
    DEFAULT_MAX_POSITION_AGE_DAYS,
    DEFAULT_POSITION_CURRENCY,
    TOP_CHANGE_CONTRIBUTORS,
    ClusterExposure,
    PlnPnlGap,
    PlnPrices,
    PlnRates,
    PlnValuation,
    Portfolio,
    PortfolioFreshness,
    PortfolioPnL,
    Position,
    PositionPlnPnL,
    SymbolChange,
    freshness,
    is_stale,
    pln_rate,
)


def _pos(symbol: str, qty: str, price: str, day: date | None = None) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(qty),
        purchase_price=Decimal(price),
        purchase_date=day or date(2026, 1, 1),
    )


def _portfolio(*positions: Position, as_of: datetime | None = None) -> Portfolio:
    return Portfolio(
        positions=tuple(positions),
        as_of=as_of or datetime(2026, 7, 10),
    )


# --- Position ----------------------------------------------------------------


def test_position_cost_basis_market_value_and_pnl() -> None:
    pos = _pos("NVDA", "10", "100")
    assert pos.cost_basis() == Decimal("1000")
    assert pos.market_value(Decimal("150")) == Decimal("1500")
    assert pos.unrealized_pnl(Decimal("150")) == Decimal("500")
    assert pos.unrealized_pnl(Decimal("80")) == Decimal("-200")


def test_position_rejects_non_positive_quantity() -> None:
    with pytest.raises(ValueError):
        _pos("NVDA", "0", "100")
    with pytest.raises(ValueError):
        _pos("NVDA", "-5", "100")


def test_position_rejects_negative_price() -> None:
    with pytest.raises(ValueError):
        _pos("NVDA", "10", "-1")


def test_position_rejects_nan_inf() -> None:
    nan = Decimal("NaN")
    infinity = Decimal("Infinity")
    one = Decimal("1")
    bought_at = date(2026, 1, 1)

    with pytest.raises(ValueError):
        Position("NVDA", nan, one, bought_at)
    with pytest.raises(ValueError):
        Position("NVDA", one, infinity, bought_at)


# --- weights: realne wagi + renormalizacja ----------------------------------


def test_weights_are_market_value_based_not_equal() -> None:
    # 40% w NVDA, reszta rozłożona — dowód, że nie liczymy równych wag.
    portfolio = _portfolio(
        _pos("NVDA", "8", "50"),  # 400
        _pos("AAPL", "3", "50"),  # 150
        _pos("MSFT", "9", "50"),  # 450
    )
    prices = {"NVDA": Decimal("50"), "AAPL": Decimal("50"), "MSFT": Decimal("50")}
    weights = portfolio.weights(prices)
    assert weights["NVDA"] == Decimal("0.4")
    assert weights["AAPL"] == Decimal("0.15")
    assert weights["MSFT"] == Decimal("0.45")
    assert sum(weights.values()) == Decimal("1")


def test_weights_renormalize_over_usable_symbols() -> None:
    # Brak ceny dla MSFT → wypada z mianownika, reszta sumuje się do 1.
    portfolio = _portfolio(
        _pos("NVDA", "4", "100"),  # 400
        _pos("AAPL", "6", "100"),  # 600
        _pos("MSFT", "10", "100"),  # brak ceny
    )
    prices = {"NVDA": Decimal("100"), "AAPL": Decimal("100")}
    weights = portfolio.weights(prices)
    assert "MSFT" not in weights
    assert weights["NVDA"] == Decimal("0.4")
    assert weights["AAPL"] == Decimal("0.6")
    assert sum(weights.values()) == Decimal("1")


def test_weights_aggregate_multiple_lots_of_same_symbol() -> None:
    portfolio = _portfolio(
        _pos("NVDA", "5", "100"),  # 500
        _pos("NVDA", "5", "100"),  # 500 -> razem 1000
        _pos("AAPL", "10", "100"),  # 1000
    )
    prices = {"NVDA": Decimal("100"), "AAPL": Decimal("100")}
    weights = portfolio.weights(prices)
    assert weights["NVDA"] == Decimal("0.5")
    assert weights["AAPL"] == Decimal("0.5")


def test_weights_empty_portfolio_is_empty_not_zero_division() -> None:
    assert _portfolio().weights({}) == {}


def test_weights_no_usable_prices_is_empty() -> None:
    portfolio = _portfolio(_pos("NVDA", "10", "100"))
    assert portfolio.weights({}) == {}


# --- unrealized_pnl ----------------------------------------------------------


def test_unrealized_pnl_per_position_and_total() -> None:
    portfolio = _portfolio(
        _pos("NVDA", "10", "100"),  # koszt 1000
        _pos("AAPL", "5", "200"),  # koszt 1000
    )
    prices = {"NVDA": Decimal("150"), "AAPL": Decimal("180")}
    result = portfolio.unrealized_pnl(prices)
    assert isinstance(result, PortfolioPnL)
    by_symbol = {p.symbol: p for p in result.positions}
    assert by_symbol["NVDA"].unrealized_pnl == Decimal("500")
    assert by_symbol["NVDA"].return_ratio == Decimal("0.5")
    assert by_symbol["AAPL"].unrealized_pnl == Decimal("-100")
    assert result.total_cost_basis == Decimal("2000")
    assert result.total_market_value == Decimal("2400")
    assert result.total_unrealized_pnl == Decimal("400")
    assert result.total_return_ratio == Decimal("0.2")


def test_unrealized_pnl_skips_positions_without_price() -> None:
    portfolio = _portfolio(
        _pos("NVDA", "10", "100"),
        _pos("MSFT", "10", "100"),  # brak ceny
    )
    prices = {"NVDA": Decimal("120")}
    result = portfolio.unrealized_pnl(prices)
    assert {p.symbol for p in result.positions} == {"NVDA"}
    assert result.total_unrealized_pnl == Decimal("200")


def test_unrealized_pnl_empty_portfolio_is_zero_not_none() -> None:
    result = _portfolio().unrealized_pnl({})
    assert result.positions == ()
    assert result.total_cost_basis == Decimal("0")
    assert result.total_market_value == Decimal("0")
    assert result.total_unrealized_pnl == Decimal("0")
    assert result.total_return_ratio is None


# --- cluster_exposure --------------------------------------------------------


def test_cluster_exposure_sums_weights_by_cluster() -> None:
    # 3 największe pozycje = jeden klaster korelacji "AI".
    portfolio = _portfolio(
        _pos("NVDA", "30", "100"),  # 3000
        _pos("AMD", "20", "100"),  # 2000
        _pos("MSFT", "11", "100"),  # 1100 -> AI razem 6100
        _pos("KO", "39", "100"),  # 3900 -> defensywne
    )
    prices = {s: Decimal("100") for s in ("NVDA", "AMD", "MSFT", "KO")}
    clusters = {"NVDA": "AI", "AMD": "AI", "MSFT": "AI", "KO": "DEFENSIVE"}
    exposures = portfolio.cluster_exposure(clusters, prices)
    assert isinstance(exposures[0], ClusterExposure)
    top = exposures[0]
    assert top.cluster == "AI"
    assert top.weight == Decimal("0.61")
    assert set(top.symbols) == {"NVDA", "AMD", "MSFT"}
    assert sum(e.weight for e in exposures) == Decimal("1")


def test_cluster_exposure_sorted_by_weight_descending() -> None:
    portfolio = _portfolio(
        _pos("NVDA", "70", "100"),  # 7000
        _pos("KO", "30", "100"),  # 3000
    )
    prices = {"NVDA": Decimal("100"), "KO": Decimal("100")}
    clusters = {"NVDA": "AI", "KO": "DEFENSIVE"}
    exposures = portfolio.cluster_exposure(clusters, prices)
    assert [e.cluster for e in exposures] == ["AI", "DEFENSIVE"]
    assert exposures[0].weight == Decimal("0.7")


def test_cluster_exposure_unmapped_symbol_is_own_cluster() -> None:
    portfolio = _portfolio(
        _pos("NVDA", "60", "100"),
        _pos("TSLA", "40", "100"),
    )
    prices = {"NVDA": Decimal("100"), "TSLA": Decimal("100")}
    clusters = {"NVDA": "AI"}  # TSLA bez mapowania
    exposures = portfolio.cluster_exposure(clusters, prices)
    by_cluster = {e.cluster: e for e in exposures}
    assert by_cluster["AI"].weight == Decimal("0.6")
    assert "TSLA" in by_cluster
    assert by_cluster["TSLA"].weight == Decimal("0.4")


def test_cluster_exposure_empty_portfolio_is_empty() -> None:
    assert _portfolio().cluster_exposure({}, {}) == []


# --- freshness (badge STALE) -------------------------------------------------


def test_default_max_age_is_frozen_constant() -> None:
    # Zamrożony próg — zmiana tej wartości ma świadomie łamać ten test.
    assert DEFAULT_MAX_POSITION_AGE_DAYS == 7


def test_is_stale_when_older_than_threshold() -> None:
    as_of = datetime(2026, 7, 1)
    now = datetime(2026, 7, 10)  # 9 dni > 7
    assert is_stale(as_of, now) is True
    assert freshness(as_of, now) is PortfolioFreshness.STALE


def test_is_fresh_within_threshold() -> None:
    as_of = datetime(2026, 7, 8)
    now = datetime(2026, 7, 10)  # 2 dni < 7
    assert is_stale(as_of, now) is False
    assert freshness(as_of, now) is PortfolioFreshness.FRESH


def test_is_stale_exactly_at_threshold_is_stale() -> None:
    as_of = datetime(2026, 7, 1)
    now = datetime(2026, 7, 8)  # dokładnie 7 dni
    assert is_stale(as_of, now) is True


def test_freshness_respects_custom_threshold() -> None:
    as_of = datetime(2026, 7, 1)
    now = datetime(2026, 7, 5)  # 4 dni
    assert is_stale(as_of, now, max_age_days=3) is True
    assert is_stale(as_of, now, max_age_days=10) is False


def test_portfolio_freshness_methods_delegate() -> None:
    stale = _portfolio(as_of=datetime(2026, 6, 1))
    now = datetime(2026, 7, 10)
    assert stale.is_stale(now) is True
    assert stale.freshness(now) is PortfolioFreshness.STALE

    fresh = _portfolio(as_of=datetime(2026, 7, 9))
    assert fresh.is_stale(now) is False
    assert fresh.freshness(now) is PortfolioFreshness.FRESH


def test_freshness_future_as_of_is_fresh() -> None:
    as_of = datetime(2026, 7, 20)
    now = datetime(2026, 7, 10)
    assert is_stale(as_of, now) is False


class TestUnpricedPositions:
    """PORT-03: `weights` i `unrealized_pnl` po cichu pomijają symbole bez ceny.
    Raport musi je wymienić i powiedzieć, jaką część kosztu stanowią."""

    def test_lists_symbols_without_a_current_price(self) -> None:
        portfolio = _portfolio(
            _pos("NVDA", "10", "100"),
            _pos("XDWD.DE", "5", "100"),
            _pos("IUSN.DE", "2", "50"),
            _pos("XDWD.DE", "1", "100"),
        )

        unpriced = portfolio.unpriced_symbols({"NVDA": Decimal("120")})

        assert unpriced == ("IUSN.DE", "XDWD.DE")

    def test_fully_priced_portfolio_has_no_unpriced_symbols(self) -> None:
        portfolio = _portfolio(_pos("NVDA", "10", "100"))

        assert portfolio.unpriced_symbols({"NVDA": Decimal("120")}) == ()
        assert portfolio.unpriced_cost_share({"NVDA": Decimal("120")}) == Decimal("0")

    def test_cost_share_of_unpriced_lots(self) -> None:
        # NVDA koszt 1000, XDWD.DE koszt 600 + 100 = 700 → 700 / 1700.
        portfolio = _portfolio(
            _pos("NVDA", "10", "100"),
            _pos("XDWD.DE", "6", "100"),
            _pos("XDWD.DE", "1", "100"),
        )

        share = portfolio.unpriced_cost_share({"NVDA": Decimal("120")})

        assert share == Decimal("700") / Decimal("1700")

    def test_cost_share_is_none_without_any_cost(self) -> None:
        portfolio = _portfolio(_pos("FREE", "10", "0"))

        assert portfolio.unpriced_cost_share({}) is None


class TestChangeSince:
    """PORT-06: zmiana wartości portfela od poprzedniego cyklu (zł/$ i %) z top 3
    kontrybutorami. Ruch symbolu waży tyle, ile jego wartość rynkowa w portfelu."""

    def test_change_since_weights_symbol_moves_by_market_value(self) -> None:
        # NVDA to 75% kapitału przed zmianą (75 × $10 z $1000), AAPL stoi.
        portfolio = _portfolio(_pos("NVDA", "75", "8"), _pos("AAPL", "25", "9"))
        previous = {"NVDA": Decimal("10"), "AAPL": Decimal("10")}
        current = {"NVDA": Decimal("11"), "AAPL": Decimal("10")}  # NVDA +10%

        change = portfolio.change_since(previous, current)

        assert change is not None
        assert change.previous_value == Decimal("1000")
        assert change.current_value == Decimal("1075")
        assert change.change == Decimal("75")
        assert change.change_ratio == Decimal("0.075")
        assert change.top_contributors[0] == SymbolChange(
            symbol="NVDA", change=Decimal("75"), contribution=Decimal("0.075")
        )

    def test_top_three_contributors_by_absolute_change(self) -> None:
        portfolio = _portfolio(
            _pos("A", "10", "1"),
            _pos("B", "10", "1"),
            _pos("C", "10", "1"),
            _pos("D", "10", "1"),
        )
        previous = {s: Decimal("10") for s in "ABCD"}
        current = {
            "A": Decimal("10.5"),  # +5
            "B": Decimal("7"),  # -30
            "C": Decimal("12"),  # +20
            "D": Decimal("9"),  # -10
        }

        change = portfolio.change_since(previous, current)

        assert change is not None
        assert [c.symbol for c in change.top_contributors] == ["B", "C", "D"]
        assert change.change == Decimal("-15")

    def test_flat_positions_are_not_contributors(self) -> None:
        portfolio = _portfolio(_pos("NVDA", "1", "1"), _pos("AAPL", "1", "1"))
        prices = {"NVDA": Decimal("10"), "AAPL": Decimal("10")}

        change = portfolio.change_since(prices, {**prices, "NVDA": Decimal("12")})

        assert change is not None
        assert [c.symbol for c in change.top_contributors] == ["NVDA"]

    def test_positions_without_both_prices_are_listed_and_left_out(self) -> None:
        # Pozycja bez punktu odniesienia (np. spoza watchlisty, cold-start) nie
        # może udawać „bez zmian” — wypada z obu stron i jest wymieniona.
        portfolio = _portfolio(
            _pos("NVDA", "10", "1"), _pos("XDWD.DE", "5", "1"), _pos("NEW", "3", "1")
        )
        previous = {"NVDA": Decimal("10")}
        current = {"NVDA": Decimal("11"), "NEW": Decimal("50")}

        change = portfolio.change_since(previous, current)

        assert change is not None
        assert change.previous_value == Decimal("100")
        assert change.change_ratio == Decimal("0.1")
        assert change.unmeasured_symbols == ("NEW", "XDWD.DE")

    def test_lots_of_one_symbol_are_aggregated(self) -> None:
        portfolio = _portfolio(_pos("NVDA", "2", "1"), _pos("NVDA", "3", "1"))

        change = portfolio.change_since({"NVDA": Decimal("10")}, {"NVDA": Decimal("12")})

        assert change is not None
        assert change.top_contributors == (
            SymbolChange(symbol="NVDA", change=Decimal("10"), contribution=Decimal("0.2")),
        )

    def test_no_measurable_position_is_none(self) -> None:
        portfolio = _portfolio(_pos("NVDA", "1", "1"))

        assert portfolio.change_since({}, {"NVDA": Decimal("10")}) is None
        assert _portfolio().change_since({}, {}) is None

    def test_top_contributor_limit_is_three(self) -> None:
        assert TOP_CHANGE_CONTRIBUTORS == 3


# --- PORT-18: waluta pozycji i wycena w PLN właściwym kursem -----------------


def _lot(symbol: str, qty: str, price: str, currency: str) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(qty),
        purchase_price=Decimal(price),
        currency=currency,
    )


class TestPositionCurrency:
    def test_position_defaults_to_usd_and_an_unknown_purchase_date(self) -> None:
        position = Position(symbol="AAPL", quantity=Decimal("1"), purchase_price=Decimal("10"))

        assert position.currency == DEFAULT_POSITION_CURRENCY == "USD"
        assert position.purchase_date is None

    @pytest.mark.parametrize("currency", ["usd", "US", "USDT", "", "U$D", "ŁÓŻ"])
    def test_position_rejects_a_currency_outside_iso_4217_format(self, currency: str) -> None:
        with pytest.raises(ValueError, match="currency"):
            _lot("AAPL", "1", "10", currency)

    def test_pnl_rows_carry_the_position_currency(self) -> None:
        portfolio = _portfolio(_lot("SAP.DE", "2", "90", "EUR"))

        pnl = portfolio.unrealized_pnl({"SAP.DE": Decimal("100")})

        assert pnl.positions[0].currency == "EUR"


class TestMarketValueInPln:
    def test_eur_position_is_valued_at_the_eur_pln_rate(self) -> None:
        portfolio = _portfolio(_lot("SAP.DE", "10", "90", "EUR"), _lot("AAPL", "2", "150", "USD"))

        valuation = portfolio.market_value_pln(
            {"SAP.DE": Decimal("100"), "AAPL": Decimal("200")},
            {"EUR": Decimal("4.30"), "USD": Decimal("4.00")},
        )

        # 10 × 100 EUR × 4,30 + 2 × 200 USD × 4,00 — nie 1000 × 4,00 za euro.
        assert valuation == PlnValuation(value=Decimal("5900"), unrated_symbols=())

    def test_pln_position_needs_no_rate(self) -> None:
        portfolio = _portfolio(_lot("PKN.WA", "3", "60", "PLN"))

        valuation = portfolio.market_value_pln({"PKN.WA": Decimal("70")}, {})

        assert valuation.value == Decimal("210")
        assert pln_rate("PLN", {}) == Decimal("1")

    def test_position_without_its_rate_is_listed_not_valued_in_another_currency(self) -> None:
        portfolio = _portfolio(_lot("VUSA.L", "4", "80", "GBP"), _lot("AAPL", "1", "100", "USD"))

        valuation = portfolio.market_value_pln(
            {"VUSA.L": Decimal("90"), "AAPL": Decimal("110")}, {"USD": Decimal("4.00")}
        )

        assert valuation == PlnValuation(value=Decimal("440"), unrated_symbols=("VUSA.L",))

    def test_unpriced_position_stays_out_of_the_valuation(self) -> None:
        portfolio = _portfolio(_lot("XDWD.DE", "8", "110", "EUR"))

        valuation = portfolio.market_value_pln({}, {"EUR": Decimal("4.30")})

        assert valuation == PlnValuation(value=Decimal("0"), unrated_symbols=())


# --- PORT-11: P/L w PLN z kursem z dnia zakupu i rozbiciem efektu walutowego --

_BOUGHT = date(2025, 3, 3)


def _dated(
    symbol: str, qty: str, price: str, currency: str = "USD", day: date | None = _BOUGHT
) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(qty),
        purchase_price=Decimal(price),
        purchase_date=day,
        currency=currency,
    )


class TestPlnPnl:
    def test_pln_pnl_splits_instrument_and_fx_effect(self) -> None:
        # Zakup 10 × 100 USD przy 4,00; dziś 100 USD przy 4,40 → +400 zł, w całości
        # efekt walutowy, 0 zł instrumentu.
        portfolio = _portfolio(_dated("AAPL", "10", "100"))
        rates = PlnRates(
            current={"USD": Decimal("4.40")},
            on_purchase={("USD", _BOUGHT): Decimal("4.00")},
        )

        pnl = portfolio.pln_pnl({"AAPL": Decimal("100")}, rates)

        assert pnl.positions == (
            PositionPlnPnL(
                symbol="AAPL",
                currency="USD",
                value_pln=Decimal("4400.00"),
                cost_pln=Decimal("4000.00"),
                pnl_pln=Decimal("400.00"),
                instrument_pln=Decimal("0.00"),
                fx_pln=Decimal("400.00"),
                gap=None,
            ),
        )
        assert pnl.total_pnl_pln == Decimal("400")
        assert pnl.total_fx_pln == Decimal("400")
        assert pnl.total_instrument_pln == Decimal("0")

    def test_instrument_and_fx_effects_add_up_to_the_pln_pnl(self) -> None:
        # Cena 100 → 120 USD, kurs 4,00 → 3,80: instrument +760 zł, kurs −200 zł.
        portfolio = _portfolio(_dated("NVDA", "10", "100"))
        rates = PlnRates(
            current={"USD": Decimal("3.80")},
            on_purchase={("USD", _BOUGHT): Decimal("4.00")},
        )

        row = portfolio.pln_pnl({"NVDA": Decimal("120")}, rates).positions[0]

        assert row.instrument_pln == Decimal("760")
        assert row.fx_pln == Decimal("-200")
        assert row.pnl_pln == Decimal("560")
        assert row.value_pln is not None and row.cost_pln is not None
        assert row.pnl_pln == row.value_pln - row.cost_pln

    def test_eur_position_uses_the_eur_rates(self) -> None:
        portfolio = _portfolio(_dated("SAP.DE", "4", "200", "EUR"))
        rates = PlnRates(
            current={"USD": Decimal("4.00"), "EUR": Decimal("4.30")},
            on_purchase={("EUR", _BOUGHT): Decimal("4.20"), ("USD", _BOUGHT): Decimal("3.90")},
        )

        row = portfolio.pln_pnl({"SAP.DE": Decimal("200")}, rates).positions[0]

        assert (row.value_pln, row.cost_pln, row.fx_pln) == (
            Decimal("3440.00"),
            Decimal("3360.00"),
            Decimal("80.00"),
        )

    def test_unknown_purchase_date_is_reported_not_approximated(self) -> None:
        portfolio = _portfolio(_dated("AAPL", "10", "100", day=None))
        rates = PlnRates(current={"USD": Decimal("4.40")})

        pnl = portfolio.pln_pnl({"AAPL": Decimal("110")}, rates)

        row = pnl.positions[0]
        assert row.value_pln == Decimal("4840.00")
        assert (row.cost_pln, row.pnl_pln, row.instrument_pln, row.fx_pln) == (
            None,
            None,
            None,
            None,
        )
        assert row.gap is PlnPnlGap.PURCHASE_DATE_UNKNOWN
        assert pnl.incomplete_symbols == ("AAPL",)
        assert pnl.total_pnl_pln == Decimal("0")

    def test_missing_purchase_rate_is_its_own_gap(self) -> None:
        portfolio = _portfolio(_dated("AAPL", "10", "100"))

        row = portfolio.pln_pnl(
            {"AAPL": Decimal("110")}, PlnRates(current={"USD": Decimal("4.40")})
        ).positions[0]

        assert row.gap is PlnPnlGap.PURCHASE_RATE_MISSING
        assert row.pnl_pln is None

    def test_missing_current_rate_leaves_the_position_unvalued_in_pln(self) -> None:
        portfolio = _portfolio(_dated("VUSA.L", "4", "80", "GBP"), _dated("AAPL", "1", "100"))
        rates = PlnRates(
            current={"USD": Decimal("4.00")},
            on_purchase={("GBP", _BOUGHT): Decimal("5.00"), ("USD", _BOUGHT): Decimal("4.00")},
        )

        pnl = portfolio.pln_pnl({"VUSA.L": Decimal("90"), "AAPL": Decimal("100")}, rates)

        gbp = next(row for row in pnl.positions if row.symbol == "VUSA.L")
        assert (gbp.value_pln, gbp.gap) == (None, PlnPnlGap.CURRENT_RATE_MISSING)
        assert pnl.total_value_pln == Decimal("400")
        assert pnl.incomplete_symbols == ("VUSA.L",)

    def test_pln_position_has_no_fx_effect(self) -> None:
        portfolio = _portfolio(_dated("PKN.WA", "3", "60", "PLN"))

        row = portfolio.pln_pnl({"PKN.WA": Decimal("70")}, PlnRates()).positions[0]

        assert (row.pnl_pln, row.instrument_pln, row.fx_pln) == (
            Decimal("30"),
            Decimal("30"),
            Decimal("0"),
        )

    def test_unpriced_positions_are_left_out(self) -> None:
        portfolio = _portfolio(_dated("XDWD.DE", "8", "110", "EUR"))

        pnl = portfolio.pln_pnl({}, PlnRates(current={"EUR": Decimal("4.30")}))

        assert pnl.positions == ()
        assert pnl.total_value_pln == Decimal("0")


# --- Wagi, ekspozycja i zmiana portfela w PLN (jedna waluta, kursy NBP) --------

_RATES_USD_EUR = PlnRates(current={"USD": Decimal("4.00"), "EUR": Decimal("4.30")})


class TestPlnPrices:
    def test_usd_and_eur_positions_of_equal_pln_value_weigh_half_each(self) -> None:
        # 43 × 100 USD × 4,00 = 17 200 zł = 40 × 100 EUR × 4,30. Natywnie 4300 vs 4000
        # — suma dolarów z euro dawała 51,8% / 48,2%.
        portfolio = _portfolio(_lot("AAPL", "43", "90", "USD"), _lot("SAP.DE", "40", "90", "EUR"))
        prices = {"AAPL": Decimal("100"), "SAP.DE": Decimal("100")}

        pln = portfolio.pln_prices(prices, _RATES_USD_EUR)

        assert portfolio.weights(pln.prices) == {
            "AAPL": Decimal("0.5"),
            "SAP.DE": Decimal("0.5"),
        }
        assert pln.unrated_symbols == ()

    def test_position_without_its_rate_drops_out_of_the_weights_and_is_listed(self) -> None:
        portfolio = _portfolio(_lot("AAPL", "43", "90", "USD"), _lot("SAP.DE", "40", "90", "EUR"))
        prices = {"AAPL": Decimal("100"), "SAP.DE": Decimal("100")}

        pln = portfolio.pln_prices(prices, PlnRates(current={"USD": Decimal("4.00")}))

        assert pln.prices == {"AAPL": Decimal("400.00")}
        assert pln.unrated_symbols == ("SAP.DE",)
        assert portfolio.weights(pln.prices) == {"AAPL": Decimal("1")}

    def test_pln_position_needs_no_rate_and_unpriced_stay_out(self) -> None:
        portfolio = _portfolio(_lot("PKN.WA", "3", "60", "PLN"), _lot("XDWD.DE", "8", "110", "EUR"))

        pln = portfolio.pln_prices({"PKN.WA": Decimal("70")}, PlnRates())

        assert pln == PlnPrices(prices={"PKN.WA": Decimal("70")}, unrated_symbols=())

    def test_symbol_held_in_two_currencies_is_unrated(self) -> None:
        portfolio = _portfolio(_lot("ASML", "1", "600", "USD"), _lot("ASML", "1", "550", "EUR"))

        pln = portfolio.pln_prices({"ASML": Decimal("700")}, _RATES_USD_EUR)

        assert pln.unrated_symbols == ("ASML",)
        assert pln.prices == {}

    def test_only_held_symbols_are_converted(self) -> None:
        portfolio = _portfolio(_lot("AAPL", "1", "100", "USD"))

        pln = portfolio.pln_prices({"AAPL": Decimal("100"), "MSFT": Decimal("400")}, _RATES_USD_EUR)

        assert set(pln.prices) == {"AAPL"}

    def test_sector_exposure_is_measured_in_pln(self) -> None:
        portfolio = _portfolio(
            _lot("NVDA", "43", "90", "USD"),
            _lot("ASML.AS", "40", "90", "EUR"),
            _lot("AAPL", "86", "90", "USD"),
        )
        prices = {"NVDA": Decimal("100"), "ASML.AS": Decimal("100"), "AAPL": Decimal("100")}

        exposure = portfolio.cluster_exposure(
            {"NVDA": "Chipy", "ASML.AS": "Chipy"},
            portfolio.pln_prices(prices, _RATES_USD_EUR).prices,
        )

        # AAPL 86 × 100 × 4,00 = 34 400 zł; NVDA + ASML.AS = 17 200 + 17 200 zł.
        assert sorted((e.cluster, e.weight) for e in exposure) == [
            ("AAPL", Decimal("0.5")),
            ("Chipy", Decimal("0.5")),
        ]

    def test_unpriced_cost_share_is_measured_in_pln(self) -> None:
        # Koszt XDWD.DE 1000 EUR × 4,30 = 4300 zł, AAPL 1000 USD × 4,00 = 4000 zł.
        portfolio = _portfolio(
            _lot("XDWD.DE", "10", "100", "EUR"), _lot("AAPL", "10", "100", "USD")
        )
        prices = {"AAPL": Decimal("110")}

        assert portfolio.unpriced_cost_share(prices, rates=_RATES_USD_EUR) == (
            Decimal("4300") / Decimal("8300")
        )
        assert portfolio.unpriced_cost_share(prices) == Decimal("0.5")

    def test_cost_share_is_unknown_when_an_unpriced_position_has_no_rate(self) -> None:
        # Bez kursu GBP nie wiemy, ile VUSA.L waży w koszcie — nie zgadujemy 0%.
        portfolio = _portfolio(_lot("VUSA.L", "10", "100", "GBP"), _lot("AAPL", "10", "100", "USD"))

        share = portfolio.unpriced_cost_share({"AAPL": Decimal("110")}, rates=_RATES_USD_EUR)

        assert share is None

    def test_cost_share_leaves_out_a_priced_position_without_its_rate(self) -> None:
        portfolio = _portfolio(
            _lot("XDWD.DE", "10", "100", "EUR"),
            _lot("AAPL", "10", "100", "USD"),
            _lot("VUSA.L", "10", "100", "GBP"),
        )
        prices = {"AAPL": Decimal("110"), "VUSA.L": Decimal("90")}

        share = portfolio.unpriced_cost_share(prices, rates=_RATES_USD_EUR)

        assert share == Decimal("4300") / Decimal("8300")


def test_excluding_drops_the_symbols_and_keeps_the_freshness() -> None:
    portfolio = _portfolio(_lot("AAPL", "1", "100", "USD"), _lot("SAP.DE", "1", "100", "EUR"))

    rest = portfolio.excluding(("SAP.DE",))

    assert [p.symbol for p in rest.positions] == ["AAPL"]
    assert rest.as_of == portfolio.as_of
