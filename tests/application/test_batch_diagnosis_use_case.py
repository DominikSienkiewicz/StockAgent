"""EXP-17 — tygodniowa diagnoza zbiorcza w Slow Loopie: JEDNO wywołanie LLM nad
wszystkimi pudłami tygodnia zamiast diagnozy per błąd w `reflect`."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

from src.application.errors import ProviderUnavailable
from src.application.metering import PaidBudgetExhausted
from src.application.ports import DeferredDiagnosis, LLMPort, RepositoryPort
from src.application.use_cases.batch_diagnosis import (
    PAID_LLM_CALLS_PER_RUN,
    BatchDiagnosisUseCase,
)
from src.domain.mistake_taxonomy import MistakeCategory
from src.domain.prediction import (
    DIAGNOSIS_BATCH_PENDING_INSIGHT,
    Prediction,
    TrendDirection,
)

NOW = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)
SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMD", "BTC", "ETH")


def _miss(symbol: str, *, age: timedelta = timedelta(days=2)) -> DeferredDiagnosis:
    return DeferredDiagnosis(
        prediction=Prediction(
            id=f"id-{symbol}",
            symbol=symbol,
            predicted_trend=TrendDirection.BULLISH,
            price_at_prediction=Decimal("100"),
            predicted_target_price=Decimal("105"),
            news_summary=f"{symbol} beats estimates",
            reasoning="Momentum.",
        ),
        actual_price=Decimal("95"),
        is_trend_correct=False,
        placeholder=DIAGNOSIS_BATCH_PENDING_INSIGHT,
        predicted_at=NOW - age,
    )


def _answer(count: int) -> str:
    return json.dumps(
        {
            "diagnoses": [
                {"n": n, "category": "MACRO", "insight": f"Lekcja {n}."}
                for n in range(1, count + 1)
            ]
        }
    )


def _use_case(repo: Mock, llm: Mock) -> BatchDiagnosisUseCase:
    return BatchDiagnosisUseCase(repository_port=repo, llm_port=llm, clock=lambda: NOW)


def _repo(misses: list[DeferredDiagnosis]) -> Mock:
    repo = Mock(spec=RepositoryPort)
    repo.get_deferred_diagnoses.return_value = misses
    return repo


def test_six_misses_trigger_exactly_one_llm_call() -> None:
    repo = _repo([_miss(symbol) for symbol in SYMBOLS])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.return_value = _answer(6)

    result = _use_case(repo, llm).run()

    assert llm.analyze_mistake.call_count == 1
    prompt = llm.analyze_mistake.call_args.args[0]
    assert all(symbol in prompt for symbol in SYMBOLS)
    saved = repo.save_backfilled_diagnosis.call_args_list
    assert [call.args[0] for call in saved] == [f"id-{symbol}" for symbol in SYMBOLS]
    assert [call.args[1] for call in saved] == [f"Lekcja {n}." for n in range(1, 7)]
    assert all(call.kwargs["replaces"] == DIAGNOSIS_BATCH_PENDING_INSIGHT for call in saved)
    assert all(call.kwargs["correction_category"] is MistakeCategory.MACRO for call in saved)
    assert result == {"status": "diagnosed", "misses": 6, "diagnosed": 6}


def test_no_misses_make_no_llm_call() -> None:
    repo = _repo([])
    llm = Mock(spec=LLMPort)

    result = _use_case(repo, llm).run()

    llm.analyze_mistake.assert_not_called()
    repo.save_backfilled_diagnosis.assert_not_called()
    assert result["status"] == "skipped"


def test_one_call_per_run_is_the_budget_request() -> None:
    assert PAID_LLM_CALLS_PER_RUN == 1


def test_misses_older_than_the_week_wait_for_the_backfill() -> None:
    repo = _repo([_miss("AAPL"), _miss("MSFT", age=timedelta(days=8))])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.return_value = _answer(1)

    _use_case(repo, llm).run()

    prompt = llm.analyze_mistake.call_args.args[0]
    assert "AAPL" in prompt and "MSFT" not in prompt
    assert [call.args[0] for call in repo.save_backfilled_diagnosis.call_args_list] == [
        "id-AAPL"
    ]


def test_correct_predictions_and_rows_without_a_time_are_not_diagnosed() -> None:
    hit = _miss("AAPL")
    hit = DeferredDiagnosis(
        prediction=hit.prediction,
        actual_price=hit.actual_price,
        is_trend_correct=True,
        placeholder=hit.placeholder,
        predicted_at=hit.predicted_at,
    )
    timeless = DeferredDiagnosis(
        prediction=_miss("MSFT").prediction,
        actual_price=Decimal("95"),
        is_trend_correct=False,
        placeholder=DIAGNOSIS_BATCH_PENDING_INSIGHT,
    )
    repo = _repo([hit, timeless])
    llm = Mock(spec=LLMPort)

    assert _use_case(repo, llm).run()["status"] == "skipped"
    llm.analyze_mistake.assert_not_called()


def test_plain_text_answer_is_the_shared_diagnosis_of_every_miss() -> None:
    # Odpowiedź bez JSON-a nie przepada — ten sam fallback co diagnoza per błąd (OTHER).
    repo = _repo([_miss("AAPL"), _miss("MSFT")])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.return_value = "Rynek zignorował makro."

    result = _use_case(repo, llm).run()

    saved = repo.save_backfilled_diagnosis.call_args_list
    assert [call.args[1] for call in saved] == ["Rynek zignorował makro."] * 2
    assert all(call.kwargs["correction_category"] is MistakeCategory.OTHER for call in saved)
    assert result["diagnosed"] == 2


def test_a_miss_without_its_entry_keeps_the_placeholder() -> None:
    repo = _repo([_miss("AAPL"), _miss("MSFT")])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.return_value = json.dumps(
        {"diagnoses": [{"n": 2, "category": "NOISE", "insight": "Szum."}]}
    )

    result = _use_case(repo, llm).run()

    saved = repo.save_backfilled_diagnosis.call_args_list
    assert [call.args[:2] for call in saved] == [("id-MSFT", "Szum.")]
    assert result == {"status": "diagnosed", "misses": 2, "diagnosed": 1}


def test_exhausted_budget_leaves_every_placeholder() -> None:
    repo = _repo([_miss("AAPL")])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.side_effect = PaidBudgetExhausted("cap")

    result = _use_case(repo, llm).run()

    repo.save_backfilled_diagnosis.assert_not_called()
    assert result["status"] == "budget_exhausted"


def test_llm_failure_leaves_every_placeholder() -> None:
    repo = _repo([_miss("AAPL")])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.side_effect = ProviderUnavailable("503")

    result = _use_case(repo, llm).run()

    repo.save_backfilled_diagnosis.assert_not_called()
    assert result["status"] == "llm_failed"


def test_failed_write_of_one_row_does_not_stop_the_rest() -> None:
    repo = _repo([_miss("AAPL"), _miss("MSFT")])
    repo.save_backfilled_diagnosis.side_effect = [ProviderUnavailable("503"), None]
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.return_value = _answer(2)

    result = _use_case(repo, llm).run()

    assert repo.save_backfilled_diagnosis.call_count == 2
    assert result["diagnosed"] == 1


def test_saved_news_and_reasoning_go_through_the_untrusted_fence() -> None:
    repo = _repo([_miss("AAPL")])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.return_value = _answer(1)

    _use_case(repo, llm).run()

    prompt = llm.analyze_mistake.call_args.args[0]
    assert "DATA-ONLY" in prompt or "UNTRUSTED" in prompt
    assert "AAPL beats estimates" in prompt
