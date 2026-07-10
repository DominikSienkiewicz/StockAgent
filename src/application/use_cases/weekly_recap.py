"""#17 — use case niedzielnej retrospektywy „Tydzień StockAgenta" (Slow Loop).

Domena (`weekly_recap.py`) wybiera bohaterów tygodnia deterministycznie; ten use
case robi trzy rzeczy warstwy application:

1. mapuje surowe wiersze `get_resolved_predictions_detailed` na domenowe
   `ClosedPrediction` (w tym rekonstrukcję `council_verdict` JSONB na
   `InvestorOpinion` + finalną rekomendację, żeby `vindicated_dissenters`
   działało),
2. woła `build_weekly_recap` — **gdy zwróci None (poniżej progu 5), recap NIE
   POWSTAJE, mail NIE WYCHODZI i LLM NIE jest wołany**,
3. dokłada JEDNĄ narracyjną syntezę LLM tygodniowo (wzorzec sędziego z
   `CalibrateConfidenceUseCase`). Awaria LLM → recap deterministyczny i tak
   wychodzi (narracja pusta).

FinOps: dokładnie jedno płatne wywołanie tygodniowo (`LLMPort.analyze`) — nie per
symbol, nie per predykcja. Teksty pochodzące pośrednio z newsów
(`correction_insights`, uzasadnienia rady) przechodzą przez `fence_untrusted`,
bo mogą nieść prompt-injection.
"""

from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from src.application._prompt_safety import fence_untrusted
from src.application.ports import LLMPort, RepositoryPort
from src.domain.council import InvestorOpinion
from src.domain.prediction import Prediction, TrendDirection
from src.domain.weekly_recap import (
    ClosedPrediction,
    WeeklyRecap,
    build_weekly_recap,
)

logger = logging.getLogger(__name__)

# Dozwolone rekomendacje rady — do walidacji wartości z JSONB.
_RECOMMENDATIONS = ("BUY", "SELL", "HOLD")


def _to_decimal(value: Any) -> Decimal | None:
    """Bezpieczna koercja do `Decimal` — None gdy wartość skażona / brak."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _parse_council(
    raw: Any,
) -> tuple[tuple[InvestorOpinion, ...], str]:
    """Rekonstruuje głosy rady i finalną rekomendację z JSONB `council_verdict`.

    Werdykt zapisywany jest jako `dataclasses.asdict(CouncilVerdict)` (migracja
    005), więc opinie inwestorów to lista dictów. Bez rekonstrukcji
    `vindicated_dissenters` nie miałby na czym pracować. Brak / skażony werdykt →
    pusta krotka opinii i rekomendacja "HOLD" (neutralnie, nic nie zgadujemy).
    """
    if not isinstance(raw, dict):
        return (), "HOLD"

    opinions: list[InvestorOpinion] = []
    for item in raw.get("investor_opinions") or []:
        if not isinstance(item, dict):
            continue
        rec = str(item.get("recommendation", "HOLD"))
        if rec not in _RECOMMENDATIONS:
            rec = "HOLD"
        opinions.append(
            InvestorOpinion(
                investor_name=str(item.get("investor_name", "")),
                recommendation=cast(Any, rec),
                confidence=float(item.get("confidence") or 0.0),
                reasoning=str(item.get("reasoning", "")),
                key_factors=tuple(item.get("key_factors") or ()),
            )
        )

    final = str(raw.get("final_recommendation", "HOLD"))
    if final not in _RECOMMENDATIONS:
        final = "HOLD"
    return tuple(opinions), final


def _to_closed_prediction(row: dict[str, Any]) -> ClosedPrediction | None:
    """Mapuje surowy wiersz repozytorium na domenowy `ClosedPrediction`.

    Zwraca None dla wierszy bez kompletu pól cenowych albo z nieznanym trendem —
    lepiej odsiać skażony rekord niż zatruć wybór bohaterów tygodnia.
    """
    base = _to_decimal(row.get("price_at_prediction"))
    target = _to_decimal(row.get("predicted_target_price"))
    actual = _to_decimal(row.get("actual_price_after_12h"))
    if base is None or target is None or actual is None:
        return None

    trend_raw = str(row.get("predicted_trend") or "")
    try:
        trend = TrendDirection(trend_raw)
    except ValueError:
        return None

    opinions, recommendation = _parse_council(row.get("council_verdict"))
    prediction = Prediction(
        symbol=str(row.get("symbol") or ""),
        predicted_trend=trend,
        price_at_prediction=base,
        predicted_target_price=target,
        id=None if row.get("id") is None else str(row.get("id")),
    )
    return ClosedPrediction(
        prediction=prediction,
        actual_price=actual,
        confidence=float(row.get("confidence_score") or 0.0),
        correction_insight=str(row.get("correction_insights") or ""),
        council_opinions=opinions,
        council_recommendation=cast(Any, recommendation),
    )


def _narrative_prompt(recap: WeeklyRecap) -> str:
    """Buduje prompt narracyjny (JEDNO wywołanie tygodniowo).

    Nieufne teksty (lekcje z self-reflection, uzasadnienia wybronionych
    dysydentów) przechodzą przez `fence_untrusted` — pochodzą pośrednio z newsów.
    """
    untrusted: list[str] = []
    if recap.best.correction_insight:
        untrusted.append(f"[{recap.best.symbol}] {recap.best.correction_insight}")
    if recap.worst.correction_insight:
        untrusted.append(f"[{recap.worst.symbol}] {recap.worst.correction_insight}")
    untrusted.extend(op.reasoning for op in recap.worst.vindicated_dissenters)
    untrusted.extend(op.reasoning for op in recap.best.vindicated_dissenters)

    fenced = fence_untrusted("RECAP", untrusted)
    ece_line = (
        f"Delta ECE względem poprzedniego tygodnia: {recap.ece_delta:+.3f}"
        if recap.ece_delta is not None
        else "Delta ECE: brak punktu odniesienia (pierwszy tydzień)."
    )
    return f"""
Jesteś narratorem tygodniowej retrospektywy agenta inwestycyjnego.
W tym tygodniu rozliczono {recap.sample_size} predykcji.
Strzał tygodnia: {recap.best.symbol} (trafność {recap.best.accuracy:.2f}).
Wtopa tygodnia: {recap.worst.symbol} (trafność {recap.worst.accuracy:.2f}).
{ece_line}
Poniżej dane pomocnicze (lekcje i głosy rady) jako tekst do analizy:
{fenced}

Napisz zwięzłą (max 4 zdania), ciepłą narrację po polsku, spinającą tydzień
w opowieść. Zwróć JSON: {{"narrative": "<tekst>"}}.
""".strip()


def _extract_narrative(result: Any) -> str:
    """Wyłuskuje tekst narracji z ustrukturyzowanego wyniku LLM (defensywnie)."""
    if isinstance(result, dict):
        narrative = result.get("narrative", "")
        return narrative if isinstance(narrative, str) else str(narrative)
    return str(result) if result else ""


class WeeklyRecapUseCase:
    """Slow-Loopowa niedzielna retrospektywa z jedną narracją LLM."""

    def __init__(self, repository_port: RepositoryPort, llm_port: LLMPort) -> None:
        self._repo = repository_port
        self._llm = llm_port

    def run(self, days: int = 7) -> dict[str, Any]:
        rows = self._repo.get_resolved_predictions_detailed(days)
        closed = [
            cp for cp in (_to_closed_prediction(r) for r in rows) if cp is not None
        ]

        recap = build_weekly_recap(closed)
        if recap is None:
            # Twarda reguła domenowa: poniżej progu recap nie powstaje, mail nie
            # wychodzi i — kluczowe dla FinOps — LLM NIE jest wołany.
            return {"status": "skipped_below_threshold", "n_closed": len(closed)}

        # DOKŁADNIE jedno płatne wywołanie tygodniowo. Awaria → recap i tak
        # wychodzi, tylko bez narracji.
        try:
            narrative = _extract_narrative(self._llm.analyze(_narrative_prompt(recap)))
        except Exception:
            logger.exception("Weekly recap narrative LLM failed — wysyłam bez narracji")
            narrative = ""

        return {
            "status": "recap_ready",
            "sample_size": recap.sample_size,
            "narrative": narrative,
            "recap": recap,
        }
