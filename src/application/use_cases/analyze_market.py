from __future__ import annotations

import dataclasses
from collections import Counter
from decimal import Decimal
from typing import Any

from src.application.agent_graph import AgentGraphDeps, AgentState, create_agent_graph
from src.domain.alpha_fusion import AlphaFusionScore
from src.domain.asset import Asset


class AnalyzeMarketUseCase:
    """Fast Loop — pełen cykl decyzyjny dla pojedynczego symbolu.

    Spina:
    1. odczyt poprzedniej ceny z repozytorium,
    2. uruchomienie grafu LangGraph (check_price → reflect/sentiment/news/predict/save).
    """

    def __init__(self, deps: AgentGraphDeps) -> None:
        self._repository = deps.repository_port
        # Licznik płatnych wywołań CAŁEGO cyklu — jeden na use case, nie na symbol.
        # Graf dostaje TEN egzemplarz, nie ten z `deps`: gdy wołający nie podał
        # własnego, licznik musi być współdzielony między use case'em a węzłami,
        # inaczej `paid_calls` zostałoby puste mimo płatnych wywołań.
        self.paid_calls: Counter[str] = deps.paid_call_meter or Counter()
        workflow = create_agent_graph(
            dataclasses.replace(deps, paid_call_meter=self.paid_calls)
        )
        # Kompilacja jest deterministyczna (zależy tylko od topologii + portów),
        # więc kompilujemy RAZ tutaj i reużywamy aplikację w każdym run().
        # Inaczej przy 43 symbolach na cykl rekompilowalibyśmy graf 43×.
        self._app = workflow.compile()

    def run(
        self,
        symbol: str,
        asset: Asset | None = None,
        *,
        regime_context: str = "",
        regime_multiplier: float = 1.0,
        earnings_multiplier: float = 1.0,
        alpha_fusion: AlphaFusionScore | None = None,
        peer_context: tuple[tuple[str, str], ...] = (),
    ) -> dict[str, Any]:
        previous = self._repository.get_last_price(symbol)

        # Cold start: brak historii → previous=0 → delta=0 (guard w domenie) → ignore.
        initial_state: AgentState = {
            "symbol": symbol,
            "previous_price": previous.amount if previous else Decimal("0"),
            # #7 reżim (label + mnożnik progu) i #5 contagion (peery z tego cyklu).
            "regime_context": regime_context,
            "regime_multiplier": regime_multiplier,
            # #6 bramka earnings — mnożnik progu z domeny (zawsze >= 1.0).
            "earnings_multiplier": earnings_multiplier,
            # #14 composite alfa — do promptu LLM i do ósmej cechy ML.
            "alpha_fusion": alpha_fusion,
            "peer_context": peer_context,
        }
        # Przekazujemy asset z klasyfikacją (STOCK/ETF), gdy dostępny z zewnątrz.
        if asset is not None:
            initial_state["asset"] = asset

        result: dict[str, Any] = self._app.invoke(initial_state)
        return result
