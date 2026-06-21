"""#4 — LLM-as-Judge oceny kalibracji pewności (Slow Loop).

Po rozliczeniu predykcji sędzia-LLM ocenia, czy DEKLAROWANA pewność rady była
uzasadniona realną trafnością. Per-predykcja `calibration_score` liczymy
arytmetycznie (bez LLM); JEDNO wywołanie LLM daje zbiorczy, ludzki wniosek
(„przepewna na krypto"). Slow Loop only — poza budżetem Fast Loopa.
"""

from __future__ import annotations

import logging
from typing import Any

from src.application.ports import LLMPort, RepositoryPort
from src.domain.prediction import calibration_error, calibration_score

logger = logging.getLogger(__name__)


def _judge_prompt(n: int, mean_conf: float, accuracy: float, gap: float) -> str:
    return f"""
Jesteś audytorem kalibracji prognoz.
W ostatnim oknie było {n} rozliczonych predykcji.
Średnia DEKLAROWANA pewność: {mean_conf:.2f}. Realna trafność kierunkowa: {accuracy:.2f}.
Luka kalibracji |pewność − trafność|: {gap:.2f}.
Oceń zwięźle (max 2 zdania), czy agent jest przepewny, niedopewny czy dobrze
skalibrowany, i na co zwrócić uwagę.
""".strip()


class CalibrateConfidenceUseCase:
    """Slow-Loopowy sędzia kalibracji pewności predykcji."""

    def __init__(self, repository_port: RepositoryPort, llm_port: LLMPort) -> None:
        self._repo = repository_port
        self._llm = llm_port

    def run(self, days: int = 30) -> dict[str, Any]:
        rows = self._repo.get_resolved_for_calibration(days)
        usable = [
            r
            for r in rows
            if r.get("confidence_score") is not None
            and r.get("is_trend_correct") is not None
        ]
        if not usable:
            return {
                "status": "skipped",
                "reason": "Brak rozliczonych predykcji z zapisaną pewnością.",
            }

        samples = [
            (float(r["confidence_score"]), bool(r["is_trend_correct"])) for r in usable
        ]
        gap = calibration_error(samples)
        mean_conf = sum(c for c, _ in samples) / len(samples)
        accuracy = sum(1 for _, ok in samples if ok) / len(samples)

        # JEDNO płatne wywołanie (Slow Loop) — zbiorczy wniosek o kalibracji.
        try:
            insight = self._llm.analyze_mistake(
                _judge_prompt(len(samples), mean_conf, accuracy, gap)
            )
        except Exception:
            logger.exception("Calibration judge LLM failed — zapisuję bez wniosku")
            insight = ""

        for r in usable:
            score = calibration_score(
                float(r["confidence_score"]), bool(r["is_trend_correct"])
            )
            try:
                self._repo.save_calibration(str(r["id"]), score, insight)
            except Exception:
                logger.exception("save_calibration failed for %s", r.get("id"))

        return {
            "status": "calibrated",
            "n_samples": len(samples),
            "calibration_gap": gap,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
            "insight": insight,
        }
