"""EXP-10 — sędzia diagnoz z rubryką: tygodniowa próbka najwyżej `MAX_JUDGED_PER_RUN`
diagnoz, jedno wywołanie LLM na diagnozę, a diagnoza poniżej progu wypada z lessons
i RAG (zapis odrzucenia)."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock

from src.application.errors import ProviderUnavailable
from src.application.metering import PaidBudgetExhausted
from src.application.ports import JudgeableDiagnosis, LLMPort, RepositoryPort
from src.application.use_cases.judge_diagnoses import (
    MAX_JUDGED_PER_RUN,
    JudgeDiagnosesUseCase,
)
from src.domain.prediction import (
    DIAGNOSIS_BATCH_PENDING_INSIGHT,
    DIAGNOSIS_SKIPPED_INSIGHT,
    Prediction,
    TrendDirection,
)

NOW = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)


def _diagnosis(n: int, insight: str | None = None) -> JudgeableDiagnosis:
    return JudgeableDiagnosis(
        prediction=Prediction(
            id=f"id-{n}",
            symbol=f"S{n:02d}",
            predicted_trend=TrendDirection.BULLISH,
            price_at_prediction=Decimal("100"),
            predicted_target_price=Decimal("105"),
            news_summary=f"Headline {n}",
            reasoning="Momentum.",
        ),
        actual_price=Decimal("95"),
        insight=insight if insight is not None else f"Zignorowałem raport Fed nr {n}.",
        predicted_at=NOW,
    )


def _verdict(factor: bool, falsifiable: bool, specific: bool) -> str:
    return json.dumps(
        {"factor_in_evidence": factor, "falsifiable": falsifiable, "specific": specific}
    )


def _use_case(
    repo: Mock, llm: Mock, *, max_samples: int = MAX_JUDGED_PER_RUN
) -> JudgeDiagnosesUseCase:
    return JudgeDiagnosesUseCase(
        repository_port=repo,
        llm_port=llm,
        clock=lambda: NOW,
        rng=random.Random(7),
        max_samples=max_samples,
    )


def _repo(rows: list[JudgeableDiagnosis]) -> Mock:
    repo = Mock(spec=RepositoryPort)
    repo.get_diagnoses_to_judge.return_value = rows
    return repo


def test_samples_at_most_cap_and_skips_placeholders() -> None:
    placeholders = [
        _diagnosis(90, DIAGNOSIS_SKIPPED_INSIGHT),
        _diagnosis(91, DIAGNOSIS_BATCH_PENDING_INSIGHT),
        _diagnosis(92, "   "),
    ]
    repo = _repo([*placeholders, *(_diagnosis(n) for n in range(15))])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.return_value = _verdict(True, True, True)

    result = _use_case(repo, llm).run()

    assert MAX_JUDGED_PER_RUN == 10
    assert llm.analyze_mistake.call_count == 10
    prompts = [call.args[0] for call in llm.analyze_mistake.call_args_list]
    assert not any(DIAGNOSIS_SKIPPED_INSIGHT in prompt for prompt in prompts)
    assert not any(DIAGNOSIS_BATCH_PENDING_INSIGHT in prompt for prompt in prompts)
    judged = [call.args[0] for call in repo.save_diagnosis_judgement.call_args_list]
    assert len(judged) == len(set(judged)) == 10
    assert not set(judged) & {"id-90", "id-91", "id-92"}
    assert result == {
        "status": "judged",
        "candidates": 15,
        "judged": 10,
        "rejected": 0,
        "failed": 0,
    }


def test_a_low_score_rejects_the_diagnosis_and_a_pass_keeps_it() -> None:
    repo = _repo([_diagnosis(1), _diagnosis(2)])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.side_effect = [_verdict(True, False, False), _verdict(True, True, False)]

    result = _use_case(repo, llm).run()

    saved = {
        call.args[0]: call.kwargs for call in repo.save_diagnosis_judgement.call_args_list
    }
    rejected_id = next(pid for pid, kwargs in saved.items() if kwargs["score"] == 1)
    kept_id = next(pid for pid, kwargs in saved.items() if kwargs["score"] == 2)
    rejected_n = rejected_id.removeprefix("id-")
    assert saved[rejected_id]["rejected_insight"] == f"Zignorowałem raport Fed nr {rejected_n}."
    assert saved[kept_id]["rejected_insight"] is None
    assert result["rejected"] == 1


def test_prompt_carries_the_rubric_the_evidence_and_the_fenced_diagnosis() -> None:
    repo = _repo([_diagnosis(1)])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.return_value = _verdict(True, True, True)

    _use_case(repo, llm).run()

    prompt = llm.analyze_mistake.call_args.args[0]
    for criterion in ("factor_in_evidence", "falsifiable", "specific"):
        assert criterion in prompt
    assert "S01" in prompt and "Headline 1" in prompt and "95" in prompt
    assert "Zignorowałem raport Fed nr 1." in prompt
    assert "UNTRUSTED_DIAGNOSIS_DATA" in prompt


def test_no_candidates_make_no_llm_call() -> None:
    repo = _repo([_diagnosis(1, DIAGNOSIS_SKIPPED_INSIGHT)])
    llm = Mock(spec=LLMPort)

    result = _use_case(repo, llm).run()

    llm.analyze_mistake.assert_not_called()
    assert result["status"] == "skipped"
    repo.get_diagnoses_to_judge.assert_called_once_with(7)


def test_a_zero_cap_makes_no_llm_call() -> None:
    repo = _repo([_diagnosis(1)])
    llm = Mock(spec=LLMPort)

    result = _use_case(repo, llm, max_samples=0).run()

    llm.analyze_mistake.assert_not_called()
    repo.get_diagnoses_to_judge.assert_not_called()
    assert result["status"] == "skipped"


def test_an_uncertain_verdict_writes_nothing_so_the_row_returns_next_week() -> None:
    repo = _repo([_diagnosis(1)])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.return_value = "Trudno powiedzieć."

    result = _use_case(repo, llm).run()

    repo.save_diagnosis_judgement.assert_not_called()
    assert result["failed"] == 1


def test_exhausted_budget_ends_the_job() -> None:
    repo = _repo([_diagnosis(1), _diagnosis(2)])
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.side_effect = PaidBudgetExhausted("cap")

    result = _use_case(repo, llm).run()

    assert llm.analyze_mistake.call_count == 1
    assert result["status"] == "budget_exhausted"


def test_llm_and_write_failures_skip_only_their_row() -> None:
    repo = _repo([_diagnosis(1), _diagnosis(2), _diagnosis(3)])
    repo.save_diagnosis_judgement.side_effect = [ProviderUnavailable("503"), None]
    llm = Mock(spec=LLMPort)
    llm.analyze_mistake.side_effect = [
        ProviderUnavailable("timeout"),
        _verdict(True, True, True),
        _verdict(True, True, True),
    ]

    result = _use_case(repo, llm).run()

    assert repo.save_diagnosis_judgement.call_count == 2
    assert result["judged"] == 1
    assert result["failed"] == 2
