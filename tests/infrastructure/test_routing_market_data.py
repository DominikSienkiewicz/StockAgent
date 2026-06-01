"""Testy RoutingMarketDataPort — kompozyt MarketDataPort wybierający adapter.

Krypto idzie do CoinGecko, akcje/ETF do Finnhuba. Jeden interfejs dla
use case'a — graf nie wie, że są dwa źródła ceny.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import MarketDataPort
from src.domain.value_objects import Money
from src.infrastructure.adapters.routing_market_data import (
    RoutingMarketDataPort,
)


@pytest.fixture
def equity_port() -> Mock:
    port = Mock(spec=MarketDataPort)
    port.get_current_price.return_value = Money(Decimal("100"))
    return port


@pytest.fixture
def crypto_port() -> Mock:
    port = Mock(spec=MarketDataPort)
    port.get_current_price.return_value = Money(Decimal("65000"))
    return port


class TestRouting:
    def test_routes_crypto_symbol_to_crypto_port(
        self, equity_port: Mock, crypto_port: Mock
    ) -> None:
        router = RoutingMarketDataPort(
            equity=equity_port,
            crypto=crypto_port,
            crypto_symbols={"BTC", "ETH"},
        )

        price = router.get_current_price("BTC")

        assert price.amount == Decimal("65000")
        crypto_port.get_current_price.assert_called_once_with("BTC")
        equity_port.get_current_price.assert_not_called()

    def test_routes_equity_symbol_to_equity_port(
        self, equity_port: Mock, crypto_port: Mock
    ) -> None:
        router = RoutingMarketDataPort(
            equity=equity_port,
            crypto=crypto_port,
            crypto_symbols={"BTC", "ETH"},
        )

        price = router.get_current_price("AAPL")

        assert price.amount == Decimal("100")
        equity_port.get_current_price.assert_called_once_with("AAPL")
        crypto_port.get_current_price.assert_not_called()

    def test_crypto_symbols_default_empty(
        self, equity_port: Mock, crypto_port: Mock
    ) -> None:
        """Bez listy crypto_symbols wszystko idzie do equity portu."""
        router = RoutingMarketDataPort(
            equity=equity_port, crypto=crypto_port, crypto_symbols=set()
        )

        router.get_current_price("BTC")

        equity_port.get_current_price.assert_called_once()
        crypto_port.get_current_price.assert_not_called()


class TestImplementsPort:
    def test_is_market_data_port(self) -> None:
        assert issubclass(RoutingMarketDataPort, MarketDataPort)
