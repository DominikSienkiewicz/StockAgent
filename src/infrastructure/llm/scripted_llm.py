"""DX-04: deterministyczny `LLMPort` trybu offline — odpowiedzi JSON ze skryptu.

Skrypt (`data/offline/llm_script.json`) to lista reguł `{"when": [...], "reply": {...}}`
i tekst diagnozy pomyłki. Odpowiada pierwsza reguła, której WSZYSTKIE znaczniki
występują w prompcie — po znacznikach `output_schema` da się odróżnić predykcję,
personę rady i przewodniczącego, a po symbolu albo nazwisku persony dobrać
odpowiedź. Prompt bez pasującej reguły to `MalformedPayload`: luka w skrypcie
widać w mailu jako degradację (atrapa „SIDEWAYS 50%”, brak głosu persony), a nie
jako cicho wymyśloną odpowiedź. Bez I/O i bez stanu — wolno wołać z wątków rady."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.application.errors import MalformedPayload
from src.application.ports import LLMPort


@dataclass(frozen=True)
class ScriptedReply:
    """Reguła skryptu: odpowiedź dla promptu, który zawiera wszystkie `markers`."""

    markers: tuple[str, ...]
    reply: Mapping[str, Any]


def _invalid(reason: str) -> ValueError:
    return ValueError(f"ScriptedLLM: niepoprawny skrypt — {reason}")


def _parse_reply(raw: Any) -> ScriptedReply:
    if not isinstance(raw, Mapping):
        raise _invalid("reguła nie jest obiektem")
    markers, reply = raw.get("when"), raw.get("reply")
    if not isinstance(markers, list) or not markers:
        raise _invalid("`when` musi być niepustą listą znaczników")
    if not all(isinstance(marker, str) and marker for marker in markers):
        raise _invalid("znacznik w `when` musi być niepustym tekstem")
    if not isinstance(reply, Mapping):
        raise _invalid("`reply` musi być obiektem JSON")
    return ScriptedReply(markers=tuple(markers), reply=dict(reply))


class ScriptedLLM(LLMPort):
    """`LLMPort`, który odpowiada wyłącznie z reguł skryptu."""

    def __init__(self, replies: Sequence[ScriptedReply], *, mistake: str) -> None:
        self._replies = tuple(replies)
        self._mistake = mistake

    @classmethod
    def from_mapping(cls, script: Mapping[str, Any]) -> ScriptedLLM:
        """Skrypt z JSON-a (`replies`, `mistake`); błąd kształtu → `ValueError`."""
        replies, mistake = script.get("replies"), script.get("mistake")
        if not isinstance(replies, list):
            raise _invalid("`replies` musi być listą reguł")
        if not isinstance(mistake, str):
            raise _invalid("`mistake` musi być tekstem odpowiedzi diagnozy")
        return cls([_parse_reply(raw) for raw in replies], mistake=mistake)

    def analyze(self, prompt: str) -> dict[str, Any]:
        for rule in self._replies:
            if all(marker in prompt for marker in rule.markers):
                return copy.deepcopy(dict(rule.reply))
        raise MalformedPayload("ScriptedLLM: żadna reguła skryptu nie pasuje do promptu")

    def analyze_mistake(self, prompt: str) -> str:
        return self._mistake
