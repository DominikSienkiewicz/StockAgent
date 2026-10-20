"""EXP-03: prompt predykcji bez pamięci (`include_memory=False`) — bez lekcji
z reflect, bez analogów RAG i bez odwołania do nich w instrukcjach. Domyślny
prompt (z pamięcią) zostaje bez zmian."""

from __future__ import annotations

from typing import Any

from src.application.prompts import get_prediction_prompt

_REFLECTION = "Ostatni błąd: zignorowałem jastrzębi Fed."
_ANALOG = "  - [BEARISH, prognoza trafiła] Fed hawkish surprise spooks tech"


def _data() -> dict[str, Any]:
    return {
        "price": "90.0",
        "delta_percentage": "-10.0",
        "av_sentiment_score": -0.4,
        "av_sentiment_label": "Bearish",
        "news_volume_24h": 4,
        "high_relevance_count": 2,
        "news_titles": ["Reuters: Fed keeps rates"],
    }


def _prompt(**kwargs: Any) -> str:
    return get_prediction_prompt(
        "AAPL", _data(), reflection_context=_REFLECTION, similar_context=_ANALOG, **kwargs
    )


def test_memory_prompt_carries_reflection_and_analogs_by_default() -> None:
    prompt = _prompt()

    assert "<reflection_context>" in prompt
    assert _REFLECTION in prompt
    assert "<similar_past_situations>" in prompt
    assert "Fed hawkish surprise spooks tech" in prompt
    assert prompt == _prompt(include_memory=True)


def test_prompt_without_memory_has_no_reflection_or_analogs() -> None:
    prompt = _prompt(include_memory=False)

    assert "reflection_context" not in prompt
    assert _REFLECTION not in prompt
    assert "similar_past_situations" not in prompt
    assert "Fed hawkish surprise spooks tech" not in prompt
    assert "przeszłego błędu" not in prompt


def test_prompt_without_memory_keeps_the_market_data_and_schema() -> None:
    prompt = _prompt(include_memory=False)

    assert "Cena aktualna: 90.0" in prompt
    assert "<alpha_vantage_signal>" in prompt
    assert "Reuters: Fed keeps rates" in prompt
    assert "Przeanalizuj korelację między <current_market_data> a <alpha_vantage_signal>." in prompt
    assert "output_schema" in prompt
