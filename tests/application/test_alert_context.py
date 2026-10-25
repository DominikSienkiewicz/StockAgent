"""ALR-08: udział pozycji w alercie szoku liczony w złotych (kursy NBP), nie w walutach
notowania — kwoty w USD i w PLN nie sumują się do jednego portfela."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.alert_context import (
    ComparablePrices,
    comparable_prices,
    position_context,
)
from src.domain.portfolio import PlnRates, Portfolio, Position

_AS_OF = datetime(2026, 10, 26, tzinfo=UTC)


def _position(symbol: str, quantity: str, price: str, currency: str) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(quantity),
        purchase_price=Decimal(price),
        currency=currency,
    )


def _mixed() -> Portfolio:
    # AAPL 10 szt. w USD, PKN 100 szt. w PLN.
    return Portfolio(
        positions=(
            _position("AAPL", "10", "80", "USD"),
            _position("PKN", "100", "50", "PLN"),
        ),
        as_of=_AS_OF,
    )


_PRICES = {"AAPL": Decimal("90"), "PKN": Decimal("50")}
_USD_AT_4 = PlnRates(current={"USD": Decimal("4.00")})


class TestShareInZlotys:
    def test_share_of_a_usd_position_is_computed_after_the_nbp_conversion(self) -> None:
        context = position_context(_mixed(), "AAPL", _PRICES, _USD_AT_4)

        assert context is not None
        # 10 × 90 USD × 4,00 = 3 600 zł z 8 600 zł; z kwot natywnych wyszłoby 900 / 5 900.
        assert context.weight == Decimal("3600") / Decimal("8600")
        assert context.fully_priced is True

    def test_share_of_the_pln_position_uses_the_same_zloty_denominator(self) -> None:
        context = position_context(_mixed(), "PKN", _PRICES, _USD_AT_4)

        assert context is not None
        assert context.weight == Decimal("5000") / Decimal("8600")

    def test_without_a_rate_the_share_is_left_out_not_guessed(self) -> None:
        context = position_context(_mixed(), "AAPL", _PRICES, None)

        assert context is not None
        assert context.weight is None
        # Zwrot od zakupu jest w jednej walucie, więc zostaje: 90 / 80 − 1.
        assert context.return_ratio == Decimal("0.125")

    def test_a_missing_rate_of_another_currency_marks_the_portfolio_partly_priced(self) -> None:
        portfolio = Portfolio(
            positions=(
                _position("AAPL", "10", "80", "USD"),
                _position("SAP", "10", "100", "EUR"),
                _position("PKN", "100", "50", "PLN"),
            ),
            as_of=_AS_OF,
        )
        prices = {**_PRICES, "SAP": Decimal("120")}

        context = position_context(portfolio, "AAPL", prices, _USD_AT_4)

        assert context is not None
        assert context.weight == Decimal("3600") / Decimal("8600")
        assert context.fully_priced is False

    def test_a_missing_rate_of_the_position_itself_leaves_the_share_out(self) -> None:
        context = position_context(_mixed(), "AAPL", _PRICES, PlnRates(current={}))

        assert context is not None
        assert context.weight is None

    def test_single_currency_portfolio_needs_no_rates(self) -> None:
        portfolio = Portfolio(
            positions=(
                _position("AAPL", "10", "80", "USD"),
                _position("MSFT", "1", "300", "USD"),
            ),
            as_of=_AS_OF,
        )

        context = position_context(
            portfolio, "AAPL", {"AAPL": Decimal("90"), "MSFT": Decimal("300")}
        )

        assert context is not None
        assert context.weight == Decimal("900") / Decimal("1200")
        assert context.fully_priced is True


class TestComparablePrices:
    def test_single_currency_keeps_the_quote_prices(self) -> None:
        portfolio = Portfolio(positions=(_position("AAPL", "1", "80", "USD"),), as_of=_AS_OF)

        assert comparable_prices(portfolio, {"AAPL": Decimal("90")}, None) == ComparablePrices(
            prices={"AAPL": Decimal("90")}, unrated=()
        )

    def test_mixed_currencies_are_converted_to_zlotys(self) -> None:
        comparable = comparable_prices(_mixed(), _PRICES, _USD_AT_4)

        assert comparable == ComparablePrices(
            prices={"AAPL": Decimal("360.00"), "PKN": Decimal("50")}, unrated=()
        )

    def test_mixed_currencies_without_any_rates_cannot_be_compared(self) -> None:
        assert comparable_prices(_mixed(), _PRICES, None) is None

    def test_a_currency_without_a_rate_is_reported_as_unrated(self) -> None:
        comparable = comparable_prices(_mixed(), _PRICES, PlnRates(current={}))

        assert comparable is not None
        assert comparable.unrated == ("AAPL",)
        assert "AAPL" not in comparable.prices
