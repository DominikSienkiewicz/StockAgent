"""BEZ-14: test „taint” promptów — pole z zewnątrz trafia do promptu tylko w fence'u.

Fence (`fence_untrusted`) stoi w kilku builderach, a poprawki punktowe nie chronią
przed kolejnym nowym polem. Tu każde pole nieufne każdego buildera (predykcja,
inwestor, rewizja, przewodniczący, diagnoza, recap) dostaje ładunki z
`tests/fixtures/prompt_injection_corpus.txt` z unikalnym kanarkiem, a test żąda,
żeby kanarek występował WYŁĄCZNIE między znacznikami fence'a — także wtedy, gdy
ładunek podrabia znacznik końca (zero-width, pełna szerokość, wielkość liter).

Pola półzaufane (lekcja z reflect, uzasadnienia person) pisze nasz LLM z danych
zewnętrznych. Ich ogrodzenie zmienia tekst promptu, więc w czasie EXP-02/03 idzie
za flagą `prompt_taint_fence_enabled` (domyślnie off); `KNOWN_LEAKS_WITHOUT_FLAG`
to jawna lista pól, które bez flagi wciąż idą do promptu surowo.

Pola zaufane (spoza testu): nazwy i filozofie person (pliki repo), reżim, peerzy,
Alpha Fusion i liczby — liczy je domena; wybroniony dysydent w diagnozie to nazwa
persony i rekomendacja z enumu.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from src.application.council_prompts import (
    chairman_prompt,
    investor_prompt,
    investor_revision_prompt,
)
from src.application.decision_receipt import DecisionReceipt
from src.application.prompts import get_mistake_diagnosis_prompt, get_prediction_prompt
from src.application.use_cases.weekly_recap import _narrative_prompt
from src.domain.council import CouncilInput, InvestorOpinion, InvestorPersona
from src.domain.value_objects import Percent
from src.domain.weekly_recap import RecapHighlight, WeeklyRecap

CORPUS = Path(__file__).parents[1] / "fixtures" / "prompt_injection_corpus.txt"
CANARY = "TAINTCANARY7F3A"

# Pola, które bez `prompt_taint_fence_enabled` idą do promptu surowo (decyzja: fence
# zmienia tekst promptu, więc czeka za flagą do końca EXP-02/03).
KNOWN_LEAKS_WITHOUT_FLAG = frozenset(
    {
        "prediction-reflection_context",
        "chairman-reasoning",
    }
)

_FENCE_RE = re.compile(
    r"<<<UNTRUSTED_(?P<label>[A-Z0-9_]+)_DATA>>>.*?<<<END_UNTRUSTED_(?P=label)_DATA>>>",
    re.DOTALL,
)


def _payloads() -> list[str]:
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    return [
        line.replace("{canary}", CANARY).replace("{newline}", "\n")
        for line in lines
        if line.strip() and not line.startswith("#")
    ]


def _council_input(news: tuple[str, ...] = ("Acme raises guidance",)) -> CouncilInput:
    return CouncilInput(
        symbol="ACME",
        current_price=Decimal("142.50"),
        price_delta_pct=Percent(Decimal("-3.20")),
        sentiment_score=-0.21,
        news_articles=news,
        llm_trend="BEARISH",
        llm_confidence=0.68,
        ml_price_target=Decimal("139.10"),
    )


_PERSONA = InvestorPersona(name="Syntetyczny Inwestor", style="Kupuje solidne bilanse.")


def _opinion(reasoning: str = "Wycena uczciwa.") -> InvestorOpinion:
    return InvestorOpinion(
        investor_name="Syntetyczny Inwestor",
        recommendation="HOLD",
        confidence=0.6,
        reasoning=reasoning,
        key_factors=("wycena",),
    )


def _market_data(**fields: object) -> dict[str, object]:
    return {
        "price": Decimal("142.50"),
        "delta_percentage": Decimal("-3.20"),
        "av_sentiment_score": -0.21,
        "av_sentiment_label": "Somewhat-Bearish",
        "news_volume_24h": 7,
        "high_relevance_count": 3,
        **fields,
    }


def _recap(*, insight: str = "Trafiona predykcja.", dissent: str = "Wycena.") -> WeeklyRecap:
    return WeeklyRecap(
        sample_size=12,
        best=RecapHighlight(
            symbol="BOLT",
            accuracy=Decimal("0.93"),
            correction_insight="Trafiona predykcja.",
            direction_correct=True,
        ),
        worst=RecapHighlight(
            symbol="ACME",
            accuracy=Decimal("0.41"),
            correction_insight=insight,
            vindicated_dissenters=(_opinion(dissent),),
            direction_correct=False,
        ),
        current_ece=0.083,
        previous_ece=0.101,
        ece_delta=-0.018,
    )


@dataclass(frozen=True)
class TaintCase:
    """Pole nieufne buildera: `render(ładunek, hardened)` daje prompt z ładunkiem
    w tym polu; `hardened` = prompt z `prompt_taint_fence_enabled`."""

    id: str
    render: Callable[[str, bool], str]


CASES = (
    TaintCase(
        "prediction-news_titles",
        lambda payload, hardened: get_prediction_prompt(
            "ACME",
            _market_data(news_titles=["Acme raises guidance", payload]),
            reflection_context="Brak danych historycznych do oceny.",
            fence_lessons=hardened,
        ),
    ),
    TaintCase(
        "prediction-news_summary",
        lambda payload, hardened: get_prediction_prompt(
            "ACME",
            _market_data(news_summary=f"Acme raises guidance | {payload}"),
            reflection_context="Brak danych historycznych do oceny.",
            fence_lessons=hardened,
        ),
    ),
    TaintCase(
        "prediction-similar_context",
        lambda payload, hardened: get_prediction_prompt(
            "ACME",
            _market_data(news_titles=["Acme raises guidance"]),
            reflection_context="Brak danych historycznych do oceny.",
            similar_context=f"- [BOLT, BULLISH, prognoza trafiła] wniosek: {payload} — "
            "nagłówki: Bolt recalls sensors",
            fence_lessons=hardened,
        ),
    ),
    TaintCase(
        "prediction-reflection_context",
        lambda payload, hardened: get_prediction_prompt(
            "ACME",
            _market_data(news_titles=["Acme raises guidance"]),
            reflection_context=f"Ostatni błąd: {payload}",
            fence_lessons=hardened,
        ),
    ),
    TaintCase(
        "investor-news_articles",
        lambda payload, hardened: investor_prompt(
            _PERSONA, _council_input(("Acme raises guidance", payload))
        ),
    ),
    TaintCase(
        "investor_revision-news_articles",
        lambda payload, hardened: investor_revision_prompt(
            _PERSONA, _council_input(("Acme raises guidance", payload)), [_opinion()]
        ),
    ),
    TaintCase(
        "chairman-reasoning",
        lambda payload, hardened: chairman_prompt(
            [_opinion(), _opinion(payload)], _council_input(), fence_reasoning=hardened
        ),
    ),
    TaintCase(
        "diagnosis-last_news_summary",
        lambda payload, hardened: get_mistake_diagnosis_prompt(
            "BULLISH", f"Acme raises guidance | {payload}", "136.80"
        ),
    ),
    TaintCase(
        "diagnosis-last_reasoning",
        lambda payload, hardened: get_mistake_diagnosis_prompt(
            "BULLISH", "Acme raises guidance", "136.80", last_reasoning=payload
        ),
    ),
    TaintCase(
        "diagnosis-receipt_precedent",
        lambda payload, hardened: get_mistake_diagnosis_prompt(
            "BULLISH",
            "Acme raises guidance",
            "136.80",
            receipt=DecisionReceipt(
                schema_version=1,
                precedents=({"summary": payload, "predicted_trend": "BULLISH"},),
            ),
        ),
    ),
    TaintCase(
        "recap-correction_insight",
        lambda payload, hardened: _narrative_prompt(_recap(insight=payload)),
    ),
    TaintCase(
        "recap-dissenter_reasoning",
        lambda payload, hardened: _narrative_prompt(_recap(dissent=payload)),
    ),
)
_CASE_IDS = [case.id for case in CASES]


def _outside_fences(prompt: str) -> str:
    return _FENCE_RE.sub("", prompt)


def _leaks(case: TaintCase, *, hardened: bool) -> list[str]:
    """Ładunki, których kanarek wyszedł poza fence (albo nie trafił do promptu)."""
    leaked = []
    for payload in _payloads():
        prompt = case.render(payload, hardened)
        if CANARY not in prompt or CANARY in _outside_fences(prompt):
            leaked.append(payload)
    return leaked


def test_corpus_has_payloads_with_the_canary() -> None:
    payloads = _payloads()

    assert len(payloads) >= 10
    assert all(CANARY in payload for payload in payloads)


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_untrusted_field_never_escapes_fence(case: TaintCase) -> None:
    assert _leaks(case, hardened=True) == []


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.id not in KNOWN_LEAKS_WITHOUT_FLAG],
    ids=[case_id for case_id in _CASE_IDS if case_id not in KNOWN_LEAKS_WITHOUT_FLAG],
)
def test_untrusted_field_is_fenced_without_the_flag(case: TaintCase) -> None:
    assert _leaks(case, hardened=False) == []


def test_known_leaks_without_the_flag_are_exactly_the_listed_fields() -> None:
    # Gdy pole dostanie fence bez flagi, lista musi się skurczyć — nie może kłamać.
    leaking = {case.id for case in CASES if _leaks(case, hardened=False)}

    assert leaking == KNOWN_LEAKS_WITHOUT_FLAG
