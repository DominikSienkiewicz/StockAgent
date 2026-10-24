"""#10 — publiczny scorecard kalibracji jako use case (ARCH-02).

Zna tylko porty: repozytorium (darmowe odczyty okna `track_record_days`)
i publisher strony. Konfigurację okna i protokół ewaluacji dostaje
w `PublicScorecardOptions` od composition root — application nie importuje
`src.config`. Fast Loop publikuje stronę best-effort po wysyłce raportu,
a `main_scorecard` renderuje ją sam i przy błędzie odczytu kończy exit 1.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from src.application.ports import RepositoryPort, WebPublisherPort
from src.application.report_evidence import PaperCosts, portfolio_curve
from src.application.report_public_scorecard import render_public_scorecard_html
from src.application.track_record_window import (
    TrackRecordReads,
    window_benchmark,
    window_coverage,
)
from src.domain.evaluation_protocol import EvaluationProtocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublicScorecardOptions:
    """Okno i źródła strony; mapowane z `Settings` w composition root.

    `settle_after_hours` to minimalny wiek rozliczenia (`reflection_min_age_hours`)
    w porównaniu z benchmarkiem; `load_protocol` wczytuje protokół ewaluacji
    dopiero przy renderze, więc jego błąd jest błędem publikacji, nie składania.
    """

    window_days: int
    settle_after_hours: int
    symbols: Sequence[str]
    crypto_symbols: Sequence[str]
    load_protocol: Callable[[], EvaluationProtocol]


def render_public_scorecard(
    repository: RepositoryPort,
    options: PublicScorecardOptions,
    costs: PaperCosts | None = None,
    *,
    reads: TrackRecordReads | None = None,
) -> str:
    """#10 — strona z agregatami (N, hit-rate, Brier, ECE, krzywe) z okna
    `window_days`; "" gdy okno nie ma danych.

    Metryki liczone z `resolved`/`cal_rows` NIEZALEŻNIE od flag
    `equity_curve_enabled` / `calibration_curve_enabled` — tamte sterują
    sekcjami prywatnego maila, nie publicznym dowodem. Krzywa kapitału to
    portfel o równej wadze per cykl (WIA-17), po kosztach `costs` (WIA-50;
    None = brutto); strona ujawnia okno i datę generacji (WIA-33).
    Błąd odczytu leci wyżej: Fast Loop go połyka, `main_scorecard` kończy exit 1.
    `reads` (ARCH-16) — odczyty okna z dispatchu Fast Loopa; None = własne.
    """
    window = TrackRecordReads(repository, options.window_days) if reads is None else reads
    resolved = window.resolved()
    # WIA-39/WIA-58: kierunkowe próbki; cykle dają n_eff i bootstrap ECE.
    calibration = window.calibration()
    benchmark = (
        None
        if costs is None
        else window_benchmark(
            repository,
            resolved,
            costs,
            window_days=options.window_days,
            settle_after_hours=options.settle_after_hours,
        )
    )
    return render_public_scorecard_html(
        resolved,
        portfolio_curve(resolved, costs),
        calibration.directional,
        generated_on=datetime.now(UTC).date(),
        window_days=options.window_days,
        costs=costs,
        benchmark=benchmark,
        protocol=options.load_protocol(),
        calibration_by_cycle=calibration.directional_by_cycle,
        coverage=window_coverage(
            repository,
            symbols=options.symbols,
            crypto_symbols=options.crypto_symbols,
            days=options.window_days,
            settled_rows=window.resolved_rows,
        ),
    )


class PublishPublicScorecardUseCase:
    """#10 — publikuje `render_public_scorecard` przez osobny publisher strony.

    Osobna instancja publishera (nie ta od prywatnego digestu) wybiera
    composition root — obie piszą do pliku, więc jedna nadpisałaby drugą.
    """

    def __init__(
        self,
        repository: RepositoryPort,
        publisher: WebPublisherPort,
        options: PublicScorecardOptions,
        *,
        reads: TrackRecordReads | None = None,
    ) -> None:
        self._repository = repository
        self._publisher = publisher
        self._options = options
        self._reads = reads

    def render(self, costs: PaperCosts | None = None) -> str:
        """Strona okna albo "" bez danych; błąd odczytu leci do wołającego."""
        return render_public_scorecard(self._repository, self._options, costs, reads=self._reads)

    def run(self, costs: PaperCosts | None = None) -> None:
        """Best-effort: błąd odczytu ani `OSError` przy zapisie nie wywalają cyklu."""
        try:
            page = self.render(costs)
            if page:
                self._publisher.publish(page)
        except Exception:
            logger.exception("Public scorecard publish failed — pomijam")
