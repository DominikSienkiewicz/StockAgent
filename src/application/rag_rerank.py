"""Outcome-aware rerank analogów RAG (#9).

`find_similar_predictions` zwraca historyczne sytuacje wg podobieństwa wektora
newsów (`similarity`) wraz z ich rzeczywistym wynikiem (`is_trend_correct`).
Sam dystans wektorowy nie odróżnia analogu, którego lekcja się potwierdziła, od
takiego, który zawiódł. Ten reranker miesza podobieństwo z realnym wynikiem,
żeby prompt uprzywilejował analogi z ZWERYFIKOWANĄ lekcją.
"""

from __future__ import annotations

from typing import Any


def _outcome_sign(is_trend_correct: Any) -> float:
    if is_trend_correct is True:
        return 1.0
    if is_trend_correct is False:
        return -1.0
    return 0.0  # None / nieznany wynik = neutralny


def rerank_analogs(
    records: list[dict[str, Any]],
    *,
    outcome_weight: float,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Sortuje analogi malejąco po `similarity + outcome_weight * outcome_sign`.

    `outcome_weight=0` zachowuje kolejność wg samego similarity (stary RAG).
    Sortowanie stabilne — przy równym wyniku kolejność wejściowa zachowana.
    `top_k` opcjonalnie tnie do najlepszych K.
    """

    def score(record: dict[str, Any]) -> float:
        similarity = float(record.get("similarity") or 0.0)
        return similarity + outcome_weight * _outcome_sign(record.get("is_trend_correct"))

    ranked = sorted(records, key=score, reverse=True)
    return ranked[:top_k] if top_k is not None else ranked
