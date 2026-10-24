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
from src.domain.prediction import (
    BRIER_CALIBRATION_SCORE_VERSION,
    brier_calibration_score,
    calibration_error,
    calibration_score,
)

logger = logging.getLogger(__name__)

# FIN-30: płatne wywołania LLM jednego przebiegu (zbiorczy wniosek) — wkład
# sędziego do tygodniowego budżetu Slow Loopa.
PAID_LLM_CALLS_PER_RUN = 1


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
    """Slow-Loopowy sędzia kalibracji pewności predykcji.

    `brier_score_enabled` (WIA-37, flaga `calibration_score_version_enabled`,
    migracja 044) zapisuje właściwą regułę Briera razem z jej wersją. Bez flagi
    zapis jest jak dotąd: reguła liniowa v1, bez kolumny wersji — przed migracją
    nowy klucz zwróciłby PGRST204, a wynik Briera bez wersji migracja
    oznaczyłaby jako v1.
    """

    def __init__(
        self,
        repository_port: RepositoryPort,
        llm_port: LLMPort,
        *,
        brier_score_enabled: bool = False,
    ) -> None:
        self._repo = repository_port
        self._llm = llm_port
        self._brier_score_enabled = brier_score_enabled

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
            try:
                self._save_score(r, insight)
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

    def _save_score(self, row: dict[str, Any], insight: str) -> None:
        confidence = float(row["confidence_score"])
        correct = bool(row["is_trend_correct"])
        if self._brier_score_enabled:
            self._repo.save_calibration(
                str(row["id"]),
                brier_calibration_score(confidence, correct),
                insight,
                score_version=BRIER_CALIBRATION_SCORE_VERSION,
            )
            return
        self._repo.save_calibration(
            str(row["id"]), calibration_score(confidence, correct), insight
        )
