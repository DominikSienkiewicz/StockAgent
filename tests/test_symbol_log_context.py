"""NIEZ-21: worker symbolu Fast Loopa ustawia symbol w kontekście logów.

`_process_one_symbol` biegnie też w wątku puli (`symbol_concurrency > 1`), do
którego ContextVar wołającego nie przechodzi sam — symbol ustawia więc sam worker.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import main_agent
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
from src.application.use_cases.enrich_alpha import AlphaEnrichmentUseCase
from src.infrastructure.logging_setup import current_symbol


def _process(symbol: str, use_case: AnalyzeMarketUseCase) -> main_agent._ProcessedSymbol:
    return main_agent._process_one_symbol(
        symbol,
        peer_ctx=(),
        use_case=use_case,
        crypto_set=set(),
        etf_set=set(),
        regime_context="",
        regime_multiplier=1.0,
        collect_alpha=False,
        alpha_use_case=Mock(spec=AlphaEnrichmentUseCase),
    )


def _use_case_recording(seen: list[str | None]) -> Mock:
    use_case = Mock(spec=AnalyzeMarketUseCase)

    def run(*_args: object, **_kwargs: object) -> dict[str, object]:
        seen.append(current_symbol())
        raise RuntimeError("stop after recording")

    use_case.run.side_effect = run
    return use_case


def test_symbol_is_in_log_context_during_processing_and_cleared_after() -> None:
    seen: list[str | None] = []

    processed = _process("NVDA", _use_case_recording(seen))

    assert processed.is_failure
    assert seen == ["NVDA"]
    assert current_symbol() is None


def test_each_pool_worker_sees_its_own_symbol() -> None:
    seen: list[str | None] = []
    use_case = _use_case_recording(seen)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda sym: _process(sym, use_case), ["AAPL", "MSFT", "NVDA"]))

    assert sorted(s for s in seen if s is not None) == ["AAPL", "MSFT", "NVDA"]
    assert None not in seen
