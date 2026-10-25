"""DX-04: `ScriptedLLM` — deterministyczne odpowiedzi JSON wybierane regułą po treści promptu."""

from __future__ import annotations

from typing import Any

import pytest

from src.application.errors import MalformedPayload
from src.application.ports import LLMPort
from src.infrastructure.llm.scripted_llm import ScriptedLLM

SCRIPT: dict[str, Any] = {
    "replies": [
        {"when": ['"final_recommendation"'], "reply": {"summary": "Rada za kupnem."}},
        {
            "when": ['"trend_direction"', "aktywa: NVDA"],
            "reply": {"trend_direction": "BULLISH", "confidence_score": 0.72},
        },
        {"when": ['"trend_direction"'], "reply": {"trend_direction": "SIDEWAYS"}},
    ],
    "mistake": '{"category": "OTHER", "insight": "Skrypt: ruch przeciw tezie."}',
}


def test_first_rule_whose_markers_all_appear_in_the_prompt_answers() -> None:
    llm = ScriptedLLM.from_mapping(SCRIPT)

    assert isinstance(llm, LLMPort)
    assert llm.analyze('ceny aktywa: NVDA. {"trend_direction": ...}') == {
        "trend_direction": "BULLISH",
        "confidence_score": 0.72,
    }
    assert llm.analyze('aktywa: AAPL. {"trend_direction": ...}') == {"trend_direction": "SIDEWAYS"}
    assert llm.analyze('{"final_recommendation": ...}') == {"summary": "Rada za kupnem."}


def test_reply_is_a_fresh_copy_each_call() -> None:
    llm = ScriptedLLM.from_mapping(SCRIPT)

    first = llm.analyze('{"final_recommendation"}')
    first["summary"] = "zmienione przez wołającego"

    assert llm.analyze('{"final_recommendation"}') == {"summary": "Rada za kupnem."}


def test_mistake_diagnosis_comes_from_the_script() -> None:
    llm = ScriptedLLM.from_mapping(SCRIPT)

    assert llm.analyze_mistake("dowolny prompt diagnozy") == SCRIPT["mistake"]


def test_prompt_outside_the_script_is_a_malformed_payload() -> None:
    llm = ScriptedLLM.from_mapping(SCRIPT)

    with pytest.raises(MalformedPayload, match="żadna reguła"):
        llm.analyze("prompt bez znaczników")


@pytest.mark.parametrize(
    "script",
    [
        {"replies": [{"when": [], "reply": {}}], "mistake": "x"},
        {"replies": [{"when": ["a"], "reply": "nie obiekt"}], "mistake": "x"},
        {"replies": [], "mistake": 3},
        {"mistake": "x"},
    ],
)
def test_malformed_script_is_rejected_at_load(script: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="ScriptedLLM"):
        ScriptedLLM.from_mapping(script)
