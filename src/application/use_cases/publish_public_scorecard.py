"""#10 — publiczny scorecard kalibracji jako use case (ARCH-02).

Zna tylko porty: repozytorium (darmowe odczyty okna `track_record_days`)
i publisher strony. Konfigurację okna i protokół ewaluacji dostaje
w `PublicScorecardOptions` od composition root — application nie importuje
`src.config`. Fast Loop publikuje stronę best-effort po wysyłce raportu,
a `main_scorecard` renderuje ją sam i przy błędzie odczytu kończy exit 1.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from src.application.ports import RepositoryPort, WebPublisherPort
from src.application.report_evidence import PaperCosts, portfolio_curve
from src.application.report_models import ResolvedPrediction
from src.application.report_public_scorecard import (
    SCORECARD_JSON_NAME,
    render_public_scorecard_html,
    render_public_scorecard_json,
)
from src.application.track_record_window import (
    TrackRecordReads,
    window_benchmark,
    window_coverage,
)
from src.domain.benchmark import BenchmarkComparison
from src.domain.coverage import CoverageReport
from src.domain.equity_curve import EquityCurve
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


@dataclass(frozen=True)
class ScorecardInputs:
    """Wejście strony i jej maszynowego bliźniaka (WIA-81) — liczone RAZ z okna,
    więc HTML i `scorecard.json` nie mogą się rozjechać."""

    resolved: Sequence[ResolvedPrediction]
    equity: EquityCurve | None
    calibration_samples: Sequence[tuple[float, bool]]
    generated_on: date
    window_days: int
    costs: PaperCosts | None
    benchmark: BenchmarkComparison | None
    protocol: EvaluationProtocol
    calibration_by_cycle: Sequence[Sequence[tuple[float, bool]]]
    coverage: CoverageReport | None

    def _kwargs(self) -> dict[str, Any]:
        return {
            "generated_on": self.generated_on,
            "window_days": self.window_days,
            "costs": self.costs,
            "benchmark": self.benchmark,
            "protocol": self.protocol,
            "calibration_by_cycle": self.calibration_by_cycle,
            "coverage": self.coverage,
        }

    def html(self) -> str:
        return render_public_scorecard_html(
            self.resolved, self.equity, self.calibration_samples, **self._kwargs()
        )

    def document(self) -> dict[str, Any] | None:
        return render_public_scorecard_json(
            self.resolved, self.equity, self.calibration_samples, **self._kwargs()
        )


def scorecard_inputs(
    repository: RepositoryPort,
    options: PublicScorecardOptions,
    costs: PaperCosts | None = None,
    *,
    reads: TrackRecordReads | None = None,
) -> ScorecardInputs:
    """#10 — agregaty z okna `window_days` (N, hit-rate, Brier, ECE, krzywe).

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
    return ScorecardInputs(
        resolved=resolved,
        equity=portfolio_curve(resolved, costs),
        calibration_samples=calibration.directional,
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


def render_public_scorecard(
    repository: RepositoryPort,
    options: PublicScorecardOptions,
    costs: PaperCosts | None = None,
    *,
    reads: TrackRecordReads | None = None,
) -> str:
    """#10 — strona z agregatami okna (`scorecard_inputs`); "" gdy okno nie ma danych."""
    return scorecard_inputs(repository, options, costs, reads=reads).html()


class PublishPublicScorecardUseCase:
    """#10 — publikuje stronę i `scorecard.json` (WIA-81) przez osobny publisher strony.

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

    def publish(self, costs: PaperCosts | None = None) -> bool | None:
        """Strona i obok niej `scorecard.json` z tego samego wejścia.

        None — okno bez danych, nic nie zapisano (Pages zostaje przy poprzedniej
        wersji); True — oba pliki zapisane; False — któryś zapis padł. Błąd odczytu
        leci do wołającego (`main_scorecard` kończy exit 1)."""
        inputs = scorecard_inputs(self._repository, self._options, costs, reads=self._reads)
        page = inputs.html()
        document = inputs.document()
        if not page or document is None:
            return None
        page_written = bool(self._publisher.publish(page))
        document_written = bool(
            self._publisher.publish_asset(
                SCORECARD_JSON_NAME,
                json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                "application/json",
            )
        )
        return page_written and document_written

    def run(self, costs: PaperCosts | None = None) -> None:
        """Best-effort: błąd odczytu ani `OSError` przy zapisie nie wywalają cyklu."""
        try:
            self.publish(costs)
        except Exception:
            logger.exception("Public scorecard publish failed — pomijam")
