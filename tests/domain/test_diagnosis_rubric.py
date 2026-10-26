"""EXP-10 — rubryka sędziego diagnoz: trzy kryteria tak/nie, próg 2 z 3."""

from __future__ import annotations

import json

import pytest

from src.domain.diagnosis_rubric import (
    MIN_PASSING_RUBRIC_SCORE,
    RUBRIC_CRITERIA,
    RubricVerdict,
    parse_rubric_verdict,
)


def _answer(factor: object, falsifiable: object, specific: object) -> str:
    return json.dumps(
        {"factor_in_evidence": factor, "falsifiable": falsifiable, "specific": specific}
    )


@pytest.mark.parametrize(
    ("verdict", "score", "passes"),
    [
        (RubricVerdict(True, True, True), 3, True),
        (RubricVerdict(True, False, True), 2, True),
        (RubricVerdict(False, False, True), 1, False),
        (RubricVerdict(False, False, False), 0, False),
    ],
)
def test_score_counts_met_criteria_and_two_pass(
    verdict: RubricVerdict, score: int, passes: bool
) -> None:
    assert verdict.score == score
    assert verdict.passes is passes
    assert MIN_PASSING_RUBRIC_SCORE == 2


def test_parses_the_three_criteria_from_json_in_text() -> None:
    raw = "Ocena:\n```json\n" + _answer(True, False, True) + "\n```"

    assert parse_rubric_verdict(raw) == RubricVerdict(True, False, True)
    assert RUBRIC_CRITERIA == ("factor_in_evidence", "falsifiable", "specific")


@pytest.mark.parametrize(
    "raw",
    [
        "bez JSON-a",
        "{not json}",
        "[1, 2]",
        '["a", {"factor_in_evidence": true}]',
        _answer(True, "tak", True),
        json.dumps({"factor_in_evidence": True, "falsifiable": True}),
    ],
)
def test_uncertain_answers_give_no_verdict(raw: str) -> None:
    assert parse_rubric_verdict(raw) is None
