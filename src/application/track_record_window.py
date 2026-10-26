"""Odczyty okna track recordu wspólne dla maila i publicznego scorecardu.

Porównanie z koszykiem i IVV (WIA-21) oraz pokrycie danych (DAT-16) liczą te
same darmowe odczyty repozytorium dla sekcji Track Record w mailu i dla
publicznego scorecardu (`use_cases/publish_public_scorecard.py`). Oba odczyty
są best-effort: błąd zostawia jawny brak danych, nigdy nie wywraca cyklu.

ARCH-16: `TrackRecordReads` czyta rozliczenia i próbki kalibracji okna RAZ na
dispatch; każda sekcja dostaje ten sam wynik (albo ten sam błąd odczytu).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from src.application.data_coverage import build_data_coverage
from src.application.ports import RepositoryPort
from src.application.report_evidence import BENCHMARK_SYMBOL, PaperCosts, benchmark_comparison
from src.application.report_models import ResolvedPrediction
from src.application.report_signals import parse_resolved_predictions
from src.application.signal_calibration import CalibrationSamples, split_calibration_samples
from src.domain.benchmark import BenchmarkComparison
from src.domain.coverage import CoverageReport

logger = logging.getLogger(__name__)


class _Once[T]:
    """Wynik pierwszego udanego wywołania `load`, oddawany przy każdym `get`.

    Wyjątek nie jest zapamiętywany: leci do wołającego, a następna sekcja ponawia
    odczyt i łapie błąd w swoim `try` jak przy własnym odczycie (jak przed ARCH-16).
    Bez szerokiego `except` — zapadka ARCH-33 (`tests/test_exception_ratchet.py`)."""

    def __init__(self, load: Callable[[], T]) -> None:
        self._load = load
        self._result: tuple[T] | None = None

    def get(self) -> T:
        if self._result is None:
            self._result = (self._load(),)
        return self._result[0]


class TrackRecordReads:
    """ARCH-16: rozliczenia i próbki kalibracji okna `window_days`, czytane
    leniwie i najwyżej raz na dispatch.

    Odczyt pada przy pierwszym wołającym, który go potrzebuje — sekcja za
    wyłączoną flagą nie płaci za zapytanie. Wiersze rozliczeń (`resolved_rows`)
    bierze też pokrycie danych (DAT-16), sparsowane (`resolved`) — Track Record,
    chipy per symbol i scorecard; próbki (`calibration`) — krzywa, Brier i ranking.
    """

    def __init__(self, repository: RepositoryPort, window_days: int) -> None:
        self.window_days = window_days
        self._rows = _Once(lambda: repository.get_resolved_predictions(window_days))
        self._resolved = _Once(lambda: parse_resolved_predictions(self._rows.get()))
        self._calibration = _Once(
            lambda: split_calibration_samples(repository.get_resolved_for_calibration(window_days))
        )

    def resolved_rows(self) -> list[dict[str, Any]]:
        """Surowe wiersze rozliczeń okna; błąd odczytu leci do wołającego.
        Każdy wołający dostaje własną listę, jak przy osobnym odczycie."""
        return list(self._rows.get())

    def resolved(self) -> list[ResolvedPrediction]:
        """Zamknięte predykcje okna (`parse_resolved_predictions`), własna lista."""
        return list(self._resolved.get())

    def calibration(self) -> CalibrationSamples:
        """Próbki kalibracji okna — kierunkowe po cyklach i SIDEWAYS (WIA-39)."""
        return self._calibration.get()


def window_benchmark(
    repository: RepositoryPort,
    resolved: list[ResolvedPrediction],
    costs: PaperCosts,
    *,
    window_days: int,
    settle_after_hours: int,
) -> BenchmarkComparison | None:
    """WIA-21: agent obok koszyka i IVV — te same okna, te same koszty.

    IVV czytamy z darmowych `price_snapshots` (zero płatnych portów), o dzień
    dłużej niż okno predykcji: snapshot startu najstarszego cyklu bywa zapisany
    minuty przed predykcją, czyli tuż przed odcięciem okna. Błąd odczytu IVV
    zostawia koszyk i jawny brak danych benchmarku.
    """
    try:
        history = [
            (stamp, float(price.amount))
            for stamp, price in repository.get_price_history(BENCHMARK_SYMBOL, window_days + 1)
        ]
    except Exception:
        logger.exception("Failed to read %s snapshots for the benchmark", BENCHMARK_SYMBOL)
        history = []
    try:
        return benchmark_comparison(
            resolved,
            costs,
            history,
            settle_after=timedelta(hours=settle_after_hours),
        )
    except Exception:
        logger.exception("Failed to compare the equity curve with benchmarks")
        return None


def window_coverage(
    repository: RepositoryPort,
    *,
    symbols: Sequence[str],
    crypto_symbols: Sequence[str],
    days: int,
    settled_rows: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
) -> CoverageReport | None:
    """DAT-16 — pokrycie danych okna track recordu z darmowych odczytów
    (snapshoty cen, rozliczenia). Błąd → None: zdanie się chowa, cykl leci dalej.
    `settled_rows` (ARCH-16) to rozliczenia okna już przeczytane w tym dispatchu."""
    try:
        return build_data_coverage(
            repository,
            symbols=symbols,
            crypto_symbols=crypto_symbols,
            days=days,
            now=datetime.now(UTC),
            settled_rows=settled_rows,
        )
    except Exception:
        logger.exception("Data coverage failed — scorecards without the coverage line")
        return None
