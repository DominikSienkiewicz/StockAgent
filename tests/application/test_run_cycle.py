"""ARCH-07: przebieg symboli Fast Loopa jako use case z `CycleOptions`.

Wiring z `Settings` (aliasy `main_agent._analyze_symbols` i
`_process_one_symbol`) pilnują testy w `tests/test_main_agent.py`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext
from decimal import Decimal
from unittest.mock import Mock, call

import pytest

from src.application.errors import ErrorCategory, QuotaExhausted
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
from src.application.use_cases.enrich_alpha import AlphaEnrichmentUseCase
from src.application.use_cases.run_cycle import CycleOptions, RunFastLoopCycle, SymbolScope
from src.domain.corporate_events import CorporateEventSnapshot


def _use_case() -> Mock:
    use_case = Mock(spec=AnalyzeMarketUseCase)
    use_case.run.return_value = {"status": "ignored", "delta": Decimal("0")}
    return use_case


def _cycle(
    use_case: Mock,
    options: CycleOptions,
    *,
    sleep: Callable[[float], None] | None = None,
    redact: Callable[[str], str] = lambda text: text,
    symbol_scope: SymbolScope = lambda symbol: nullcontext(),
) -> RunFastLoopCycle:
    return RunFastLoopCycle(
        use_case=use_case,
        alpha_use_case=Mock(spec=AlphaEnrichmentUseCase),
        options=options,
        sleep=Mock() if sleep is None else sleep,
        redact=redact,
        symbol_scope=symbol_scope,
    )


def test_throttle_uses_injected_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    def real_sleep(_: float) -> None:
        raise AssertionError("time.sleep wołany wprost")

    monkeypatch.setattr(time, "sleep", real_sleep)
    sleep = Mock()
    symbols = ["AAPL", "MSFT", "NVDA"]

    _cycle(_use_case(), CycleOptions(throttle_seconds=2), sleep=sleep).run(symbols)

    assert sleep.call_args_list == [call(2)] * (len(symbols) - 1)


def test_failure_of_one_symbol_is_redacted_and_isolated() -> None:
    use_case = _use_case()
    use_case.run.side_effect = [
        RuntimeError("token=secret-value"),
        {"status": "ignored", "delta": Decimal("0")},
    ]

    outcome = _cycle(
        use_case, CycleOptions(), redact=lambda text: text.replace("secret-value", "***")
    ).run(["AAPL", "MSFT"])

    assert outcome.failures == 1
    assert [r.symbol for r in outcome.results] == ["AAPL", "MSFT"]
    assert outcome.results[0].error_message == "token=***"
    assert outcome.results[0].error_type == "RuntimeError"


def test_failure_carries_the_category_of_its_port_error() -> None:
    # MAIL-32: mail pokazuje kategorię z taksonomii portów, nie surowy wyjątek.
    use_case = _use_case()
    use_case.run.side_effect = [QuotaExhausted("429 Too Many Requests"), RuntimeError("bug")]

    outcome = _cycle(use_case, CycleOptions()).run(["AAPL", "MSFT"])

    assert [r.error_category for r in outcome.results] == [
        ErrorCategory.QUOTA_EXHAUSTED,
        ErrorCategory.UNKNOWN,
    ]


def test_failure_names_the_provider_from_the_injected_hook() -> None:
    # MAIL-32: dostawcę zna infrastruktura; use case tylko woła wstrzyknięty hook.
    use_case = _use_case()
    failure = QuotaExhausted("429")
    use_case.run.side_effect = failure
    seen: list[BaseException] = []

    def provider(exc: BaseException) -> str | None:
        seen.append(exc)
        return "Finnhub"

    cycle = RunFastLoopCycle(
        use_case=use_case,
        alpha_use_case=Mock(spec=AlphaEnrichmentUseCase),
        options=CycleOptions(),
        sleep=Mock(),
        redact=lambda text: text,
        symbol_scope=lambda symbol: nullcontext(),
        error_provider=provider,
    )

    outcome = cycle.run(["AAPL"])

    assert outcome.results[0].error_provider == "Finnhub"
    assert seen == [failure]


def test_without_a_provider_hook_the_provider_is_unknown() -> None:
    use_case = _use_case()
    use_case.run.side_effect = QuotaExhausted("429")

    outcome = _cycle(use_case, CycleOptions()).run(["AAPL"])

    assert outcome.results[0].error_provider is None


def test_every_symbol_runs_inside_its_log_scope() -> None:
    scoped: list[str] = []

    def scope(symbol: str) -> nullcontext[None]:
        scoped.append(symbol)
        return nullcontext()

    _cycle(_use_case(), CycleOptions(symbol_concurrency=3), symbol_scope=scope).run(
        ["AAPL", "MSFT", "NVDA"]
    )

    assert sorted(scoped) == ["AAPL", "MSFT", "NVDA"]


def test_contagion_hands_earlier_stances_to_later_peers() -> None:
    use_case = _use_case()
    use_case.run.side_effect = [
        {
            "status": "saved",
            "delta": Decimal("0.05"),
            "llm_analysis": {"trend_direction": "BULLISH"},
        },
        {"status": "ignored", "delta": Decimal("0")},
    ]

    _cycle(
        use_case,
        CycleOptions(contagion_enabled=True, peer_groups={"chips": ["NVDA", "AMD"]}),
    ).run(["NVDA", "AMD"])

    assert use_case.run.call_args_list[1].kwargs["peer_context"] == (("NVDA", "BULLISH"),)


class TestCorporateEventsInputs:
    """DAT-17: raporty 8-K pobiera cykl PRZED grafem (jak alfę), tylko z flagą
    `corporate_events_enabled`, i podaje je use case'owi."""

    @staticmethod
    def _cycle(use_case: Mock, alpha: Mock, *, enabled: bool) -> RunFastLoopCycle:
        return RunFastLoopCycle(
            use_case=use_case,
            alpha_use_case=alpha,
            options=CycleOptions(corporate_events_enabled=enabled),
            sleep=Mock(),
            redact=lambda text: text,
            symbol_scope=lambda symbol: nullcontext(),
        )

    def test_flag_on_hands_the_snapshot_to_the_use_case(self) -> None:
        use_case = _use_case()
        alpha = Mock(spec=AlphaEnrichmentUseCase)
        snapshot = CorporateEventSnapshot("AAPL", (), 14)
        alpha.fetch_corporate_events.return_value = snapshot

        self._cycle(use_case, alpha, enabled=True).run(["AAPL"])

        alpha.fetch_corporate_events.assert_called_once_with("AAPL")
        assert use_case.run.call_args.kwargs["corporate_events"] is snapshot

    def test_flag_off_asks_nothing(self) -> None:
        use_case = _use_case()
        alpha = Mock(spec=AlphaEnrichmentUseCase)

        self._cycle(use_case, alpha, enabled=False).run(["AAPL"])

        alpha.fetch_corporate_events.assert_not_called()
        assert use_case.run.call_args.kwargs.get("corporate_events") is None
