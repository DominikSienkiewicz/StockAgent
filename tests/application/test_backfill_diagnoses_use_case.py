"""WIA-53 — spłata długu diagnoz: osobny job Slow Loopa z twardym capem.

Bramka volatility pomija płatną diagnozę na spokojnych cyklach i nic jej nie
ponawia, więc placeholder zostaje w lessons i RAG na zawsze. Job bierze takie
błędne predykcje i diagnozuje najwyżej `max_paid_calls` z nich na przebieg."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock

from src.application.metering import PaidBudgetExhausted
from src.application.ports import DeferredDiagnosis, LLMPort, RepositoryPort
from src.application.use_cases.backfill_diagnoses import BackfillDiagnosesUseCase
from src.domain.mistake_taxonomy import MistakeCategory
from src.domain.prediction import (
    DIAGNOSIS_SKIPPED_INSIGHT,
    LEGACY_DIAGNOSIS_DEFERRED_INSIGHT,
    Prediction,
    TrendDirection,
)


def _deferred(
    index: int,
    *,
    correct: bool | None = False,
    placeholder: str = DIAGNOSIS_SKIPPED_INSIGHT,
    receipt: dict[str, object] | None = None,
) -> DeferredDiagnosis:
    return DeferredDiagnosis(
        prediction=Prediction(
            id=f"p-{index}",
            symbol="AAPL",
            predicted_trend=TrendDirection.BULLISH,
            price_at_prediction=Decimal("100"),
            predicted_target_price=Decimal("104"),
            news_summary=f"Nagłówek {index} | Drugi nagłówek",
            reasoning="Popyt rośnie.",
        ),
        actual_price=Decimal("97"),
        is_trend_correct=correct,
        placeholder=placeholder,
        predicted_at=datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        receipt=receipt,
    )


def _ports(rows: list[DeferredDiagnosis], answer: str = "") -> tuple[Mock, Mock]:
    repo = Mock(spec=RepositoryPort)
    repo.get_deferred_diagnoses.return_value = rows
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.return_value = answer or (
        '{"category": "MACRO", "insight": "Zignorowałem decyzję Fed."}'
    )
    return repo, llm


def test_cap_bounds_paid_calls_and_updates() -> None:
    repo, llm = _ports([_deferred(i) for i in range(7)])

    result = BackfillDiagnosesUseCase(repo, llm, max_paid_calls=5).run()

    assert llm.analyze_mistake.call_count == 5
    assert repo.save_backfilled_diagnosis.call_count == 5
    assert result["diagnosed"] == 5
    repo.get_deferred_diagnoses.assert_called_once_with(limit=5)


def test_a_correct_prediction_is_never_diagnosed() -> None:
    repo, llm = _ports([_deferred(1, correct=True), _deferred(2, correct=None), _deferred(3)])

    BackfillDiagnosesUseCase(repo, llm, max_paid_calls=5).run()

    assert llm.analyze_mistake.call_count == 1
    (call,) = repo.save_backfilled_diagnosis.call_args_list
    assert call.args[0] == "p-3"


def test_update_replaces_only_the_placeholder_it_read() -> None:
    repo, llm = _ports([_deferred(1, placeholder=LEGACY_DIAGNOSIS_DEFERRED_INSIGHT)])

    BackfillDiagnosesUseCase(repo, llm, max_paid_calls=5).run()

    repo.save_backfilled_diagnosis.assert_called_once_with(
        "p-1",
        "Zignorowałem decyzję Fed.",
        replaces=LEGACY_DIAGNOSIS_DEFERRED_INSIGHT,
        correction_category=MistakeCategory.MACRO,
    )


def test_llm_error_leaves_the_marker_and_goes_on() -> None:
    repo, llm = _ports([_deferred(1), _deferred(2)])
    llm.analyze_mistake.side_effect = [
        RuntimeError("LLM down"),
        '{"category": "NOISE", "insight": "Szum."}',
    ]

    result = BackfillDiagnosesUseCase(repo, llm, max_paid_calls=5).run()

    (call,) = repo.save_backfilled_diagnosis.call_args_list
    assert call.args[0] == "p-2"
    assert result["failed"] == 1


def test_exhausted_budget_stops_the_job() -> None:
    repo, llm = _ports([_deferred(1), _deferred(2)])
    llm.analyze_mistake.side_effect = PaidBudgetExhausted("diagnosis")

    result = BackfillDiagnosesUseCase(repo, llm, max_paid_calls=5).run()

    assert llm.analyze_mistake.call_count == 1
    repo.save_backfilled_diagnosis.assert_not_called()
    assert result["status"] == "budget_exhausted"


def test_empty_answer_is_not_saved_over_the_marker() -> None:
    repo, llm = _ports([_deferred(1)], answer="   ")

    BackfillDiagnosesUseCase(repo, llm, max_paid_calls=5).run()

    repo.save_backfilled_diagnosis.assert_not_called()


def test_prompt_is_built_from_the_persisted_inputs_and_receipt() -> None:
    receipt = {
        "schema_version": 1,
        "effective_threshold": "0.03",
        "precedents": [
            {
                "id": "3f2a1b9c-aaaa",
                "similarity": 0.91,
                "predicted_trend": "BULLISH",
                "is_trend_correct": False,
                "summary": "Analog: Fed podnosi stopy",
            },
        ],
    }
    repo, llm = _ports([_deferred(4, receipt=receipt)])

    BackfillDiagnosesUseCase(repo, llm, max_paid_calls=5).run()

    prompt = llm.analyze_mistake.call_args.args[0]
    assert "Nagłówek 4" in prompt
    assert "Popyt rośnie." in prompt
    assert "100" in prompt and "97" in prompt
    assert "3,0%" in prompt
    assert "Analog: Fed podnosi stopy" in prompt


def test_zero_allowance_makes_no_call_and_no_read() -> None:
    repo, llm = _ports([_deferred(1)])

    result = BackfillDiagnosesUseCase(repo, llm, max_paid_calls=0).run()

    assert result["status"] == "skipped"
    llm.analyze_mistake.assert_not_called()
    repo.get_deferred_diagnoses.assert_not_called()
