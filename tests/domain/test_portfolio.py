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
    ClusterExposure,
    Portfolio,
    PortfolioFreshness,
    PortfolioPnL,
    Position,
    freshness,
    is_stale,
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
    assert by_symbol["NVDA"].return_pct == Decimal("0.5")
    assert by_symbol["AAPL"].unrealized_pnl == Decimal("-100")
    assert result.total_cost_basis == Decimal("2000")
    assert result.total_market_value == Decimal("2400")
    assert result.total_unrealized_pnl == Decimal("400")
    assert result.total_return_pct == Decimal("0.2")


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
    assert result.total_return_pct is None


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
