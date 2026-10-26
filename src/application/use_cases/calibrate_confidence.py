"""R-22 — ocena kalibracji pewności rozliczonych predykcji (Slow Loop).

Per-predykcja `calibration_score` liczymy arytmetycznie: czy DEKLAROWANA pewność
była uzasadniona realnym wynikiem. WIA-65 (decyzja 2026-10-23, opcja b): zbiorczy
wniosek sędziego-LLM usunięty — nic w raporcie go nie czytało, a każde włączenie
flagi oznaczało płacenie za wynik, którego nikt nie widzi. Zero płatnych wywołań;
oceny idą do bazy JEDNYM zapisem zbiorczym zamiast UPDATE wiersz po wierszu.
"""

from __future__ import annotations

from typing import Any

from src.application.ports import RepositoryPort
from src.domain.prediction import (
    BRIER_CALIBRATION_SCORE_VERSION,
    LINEAR_CALIBRATION_SCORE_VERSION,
    brier_calibration_score,
    calibration_error,
    calibration_score,
)


class CalibrateConfidenceUseCase:
    """Slow-Loopowa, arytmetyczna ocena kalibracji pewności predykcji.

    `brier_score_enabled` (WIA-37, flaga `calibration_score_version_enabled`)
    zapisuje właściwą regułę Briera z wersją 2; bez flagi reguła liniowa z wersją 1.
    Wersję zapisujemy zawsze — RPC zbiorczego zapisu (migracja 077) przychodzi po
    kolumnie wersji (044), więc nie ma już wiersza z oceną i bez wersji.
    """

    def __init__(
        self,
        repository_port: RepositoryPort,
        *,
        brier_score_enabled: bool = False,
    ) -> None:
        self._repo = repository_port
        self._brier_score_enabled = brier_score_enabled

    def run(self, days: int = 30) -> dict[str, Any]:
        rows = self._repo.get_resolved_for_calibration(days)
        usable = [
            r
            for r in rows
            if r.get("confidence_score") is not None and r.get("is_trend_correct") is not None
        ]
        if not usable:
            return {
                "status": "skipped",
                "reason": "Brak rozliczonych predykcji z zapisaną pewnością.",
            }

        samples = [(float(r["confidence_score"]), bool(r["is_trend_correct"])) for r in usable]
        scores = {
            str(row["id"]): self._score(confidence, correct)
            for row, (confidence, correct) in zip(usable, samples, strict=True)
        }
        self._repo.save_calibration_scores(scores, score_version=self._score_version())

        return {
            "status": "calibrated",
            "n_samples": len(samples),
            "calibration_gap": calibration_error(samples),
            "mean_confidence": sum(c for c, _ in samples) / len(samples),
            "accuracy": sum(1 for _, ok in samples if ok) / len(samples),
        }

    def _score(self, confidence: float, correct: bool) -> float:
        if self._brier_score_enabled:
            return brier_calibration_score(confidence, correct)
        return calibration_score(confidence, correct)

    def _score_version(self) -> int:
        if self._brier_score_enabled:
            return BRIER_CALIBRATION_SCORE_VERSION
        return LINEAR_CALIBRATION_SCORE_VERSION
