"""ML-33: werdykt promocji challengerów nogi ML w Slow Loopie.

Czyta prognozy cienia rozliczonych predykcji z okna `LIVE_WINDOW_DAYS` i liczy dla
każdego challengera pre-rejestrowaną regułę domeny (`challenger_verdicts`). Sam
raport: nic nie zapisuje i niczego nie promuje — podmiana serwowanych wag zostaje
decyzją właściciela.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from src.application.ports import RepositoryPort
from src.domain.promotion import LIVE_WINDOW_DAYS, ChallengerVerdict, challenger_verdicts


class ReviewChallengersUseCase:
    def __init__(self, repository: RepositoryPort, *, clock: Callable[[], datetime]) -> None:
        self._repository = repository
        self._clock = clock

    def run(self, *, dead_band: float) -> tuple[ChallengerVerdict, ...]:
        """Werdykt każdego challengera wobec championa na tych samych predykcjach
        live. `dead_band` — martwa strefa kierunku z protokołu ewaluacji (EXP-06)."""
        since = self._clock() - timedelta(days=LIVE_WINDOW_DAYS)
        rows = self._repository.get_settled_shadow_predictions(since)
        return challenger_verdicts(rows, dead_band=dead_band)
