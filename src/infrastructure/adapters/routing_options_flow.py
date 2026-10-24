"""Kompozyt OptionsFlowPort — routing per symbol (krypto vs akcje), DAT-18.

Graf dostaje JEDEN port opcji (implied edge, #18): krypto idzie do Deribit DVOL,
reszta do źródła akcji (Finnhub option-chain albo Null). Klasyfikacja symbolu
to detal infrastruktury — jak w `RoutingMarketDataPort`.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.application.ports import OptionsFlowPort
from src.domain.options_flow import OptionsFlowSnapshot


class RoutingOptionsFlowPort(OptionsFlowPort):
    def __init__(
        self,
        *,
        equity: OptionsFlowPort,
        crypto: OptionsFlowPort,
        crypto_symbols: Iterable[str],
    ) -> None:
        self._equity = equity
        self._crypto = crypto
        self._crypto_symbols = set(crypto_symbols)

    def get_options_flow(self, symbol: str) -> OptionsFlowSnapshot | None:
        source = self._crypto if symbol in self._crypto_symbols else self._equity
        return source.get_options_flow(symbol)
