"""PORT-11: zbieranie kursów NBP do P/L portfela w złotych (`FxRatePort` z mocka)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import Mock, call

from src.application.portfolio_fx import (
    collect_current_pln_rates,
    collect_pln_rates,
    nbp_today,
)
from src.application.ports import FxRatePort
from src.domain.fx import FxRate
from src.domain.portfolio import PlnRates, Portfolio, Position
from src.domain.value_objects import Ratio

TODAY = date(2026, 10, 19)


def _lot(symbol: str, currency: str, day: date | None) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal("1"),
        purchase_price=Decimal("10"),
        purchase_date=day,
        currency=currency,
    )


def _port(rates: dict[tuple[str, date], str]) -> Mock:
    port = Mock(spec=FxRatePort)

    def get_rate(currency: str, on_date: date) -> FxRate | None:
        raw = rates.get((currency, on_date))
        if raw is None:
            return None
        return FxRate(currency=currency, on_date=on_date, effective_date=on_date,
                      rate=Decimal(raw))

    port.get_rate.side_effect = get_rate
    return port


def test_collects_current_and_purchase_day_rates_of_foreign_currencies() -> None:
    portfolio = Portfolio(
        positions=(
            _lot("NVDA", "USD", date(2025, 3, 3)),
            _lot("AAPL", "USD", date(2025, 3, 3)),
            _lot("SAP.DE", "EUR", date(2024, 6, 3)),
            _lot("PKN.WA", "PLN", date(2024, 1, 2)),
        ),
        as_of=datetime(2026, 10, 18, tzinfo=UTC),
    )
    port = _port({
        ("USD", TODAY): "3.70", ("EUR", TODAY): "4.25",
        ("USD", date(2025, 3, 3)): "3.97", ("EUR", date(2024, 6, 3)): "4.29",
    })

    rates = collect_pln_rates(portfolio, port, today=TODAY)

    assert rates == PlnRates(
        current={"EUR": Decimal("4.25"), "USD": Decimal("3.70")},
        on_purchase={
            ("EUR", date(2024, 6, 3)): Decimal("4.29"),
            ("USD", date(2025, 3, 3)): Decimal("3.97"),
        },
    )
    # Jedna para (waluta, dzień) = jedno zapytanie; PLN nie pyta wcale.
    assert sorted(port.get_rate.call_args_list) == sorted([
        call("EUR", TODAY), call("USD", TODAY),
        call("EUR", date(2024, 6, 3)), call("USD", date(2025, 3, 3)),
    ])


def test_position_without_purchase_date_gets_no_purchase_day_rate() -> None:
    portfolio = Portfolio(positions=(_lot("NVDA", "USD", None),), as_of=datetime.now(UTC))
    port = _port({("USD", TODAY): "3.70"})

    rates = collect_pln_rates(portfolio, port, today=TODAY)

    assert rates.on_purchase == {}
    port.get_rate.assert_called_once_with("USD", TODAY)


def test_failed_rate_is_left_out_not_raised() -> None:
    portfolio = Portfolio(
        positions=(_lot("NVDA", "USD", date(2025, 3, 3)), _lot("SAP.DE", "EUR", None)),
        as_of=datetime.now(UTC),
    )
    port = Mock(spec=FxRatePort)

    def get_rate(currency: str, day: date) -> FxRate:
        if currency != "EUR":
            raise RuntimeError("NBP down")
        return FxRate(currency="EUR", on_date=day, effective_date=day, rate=Decimal("4.25"))

    port.get_rate.side_effect = get_rate

    rates = collect_pln_rates(portfolio, port, today=TODAY)

    assert rates == PlnRates(current={"EUR": Decimal("4.25")}, on_purchase={})


def test_nbp_day_is_the_warsaw_calendar_day() -> None:
    # 23:30 UTC 18.10 to już 19.10 w Warszawie (CEST).
    assert nbp_today(datetime(2026, 10, 18, 23, 30, tzinfo=UTC)) == date(2026, 10, 19)


def test_rates_collected_for_the_portfolio_win_over_risk_watch() -> None:
    from src.application.portfolio_fx import effective_pln_rates
    from src.domain.polish_macro import PolishMacroSnapshot

    collected = PlnRates(current={"USD": Decimal("3.70")})
    macro = PolishMacroSnapshot(
        eur_pln=Decimal("4.27"), usd_pln=Decimal("3.68"),
        eur_pln_30d_change_ratio=Ratio(Decimal("0")), usd_pln_30d_change_ratio=Ratio(Decimal("0")),
        fetched_at=datetime(2026, 10, 19, tzinfo=UTC),
    )

    assert effective_pln_rates(collected, macro) is collected
    assert effective_pln_rates(None, macro) == PlnRates(
        current={"USD": Decimal("3.68"), "EUR": Decimal("4.27")}
    )
    assert effective_pln_rates(None, None) is None


def test_current_rates_ask_only_for_today_not_for_purchase_days() -> None:
    """ALR-08: wagi w złotych potrzebują samych kursów bieżących — kursy z dni zakupu
    to dodatkowe zapytania do NBP bez wpływu na udział pozycji."""
    portfolio = Portfolio(
        positions=(
            _lot("NVDA", "USD", date(2025, 3, 3)),
            _lot("SAP.DE", "EUR", date(2024, 6, 3)),
            _lot("PKN.WA", "PLN", date(2024, 1, 2)),
        ),
        as_of=datetime(2026, 10, 18, tzinfo=UTC),
    )
    port = _port({("USD", TODAY): "3.70", ("EUR", TODAY): "4.25"})

    rates = collect_current_pln_rates(portfolio, port, today=TODAY)

    assert rates == PlnRates(current={"EUR": Decimal("4.25"), "USD": Decimal("3.70")})
    assert port.get_rate.call_args_list == [call("EUR", TODAY), call("USD", TODAY)]


def test_current_rates_survive_a_failing_source() -> None:
    portfolio = Portfolio(
        positions=(_lot("NVDA", "USD", None),), as_of=datetime(2026, 10, 18, tzinfo=UTC)
    )
    port = Mock(spec=FxRatePort)
    port.get_rate.side_effect = RuntimeError("NBP down")

    assert collect_current_pln_rates(portfolio, port, today=TODAY) == PlnRates()
