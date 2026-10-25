"""DAT-18: port opcji grafu kieruje krypto do Deribit DVOL, a resztę do źródła akcji."""

from __future__ import annotations

from unittest.mock import Mock

from src.application.ports import OptionsFlowPort
from src.domain.options_flow import OptionsFlowSnapshot
from src.infrastructure.adapters.routing_options_flow import RoutingOptionsFlowPort


def _port(symbol: str, iv: float) -> Mock:
    port = Mock(spec=OptionsFlowPort)
    port.get_options_flow.return_value = OptionsFlowSnapshot(symbol, None, iv)
    return port


def test_crypto_symbol_goes_to_the_crypto_source() -> None:
    equity, crypto = _port("AAPL", 0.3), _port("BTC", 0.55)
    router = RoutingOptionsFlowPort(equity=equity, crypto=crypto, crypto_symbols=["BTC"])

    snapshot = router.get_options_flow("BTC")

    assert snapshot is not None and snapshot.implied_vol == 0.55
    equity.get_options_flow.assert_not_called()


def test_other_symbol_goes_to_the_equity_source() -> None:
    equity, crypto = _port("AAPL", 0.3), _port("BTC", 0.55)
    router = RoutingOptionsFlowPort(equity=equity, crypto=crypto, crypto_symbols=["BTC"])

    snapshot = router.get_options_flow("AAPL")

    assert snapshot is not None and snapshot.implied_vol == 0.3
    crypto.get_options_flow.assert_not_called()
