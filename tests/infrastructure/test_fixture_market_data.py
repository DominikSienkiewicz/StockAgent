"""DX-04: `FixtureMarketData` — ceny cyklu offline z fixture'a scenariusza, bez sieci."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.application.errors import NotFound
from src.application.ports import MarketDataPort
from src.domain.value_objects import Money
from src.infrastructure.adapters.fixture_market_data import FixtureMarketData


def test_serves_the_fixture_price_of_each_symbol() -> None:
    market = FixtureMarketData({"NVDA": Decimal("103.00"), "BTC": Decimal("64000")})

    assert isinstance(market, MarketDataPort)
    assert market.get_current_price("NVDA") == Money(Decimal("103.00"))
    assert market.get_current_price("BTC") == Money(Decimal("64000"))


def test_symbol_missing_from_the_fixture_is_a_port_error() -> None:
    market = FixtureMarketData({"NVDA": Decimal("103.00")})

    with pytest.raises(NotFound, match="AAPL"):
        market.get_current_price("AAPL")
