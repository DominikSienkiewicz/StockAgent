"""Odczyty okna track recordu wspólne dla maila i publicznego scorecardu.

Porównanie z koszykiem i IVV (WIA-21) oraz pokrycie danych (DAT-16) liczą te
same darmowe odczyty repozytorium dla sekcji Track Record w mailu
(`use_cases/assemble_report.py`) i dla publicznego scorecardu
(`use_cases/publish_public_scorecard.py`). Oba odczyty są best-effort: błąd
zostawia jawny brak danych, nigdy nie wywraca cyklu.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from src.application.data_coverage import build_data_coverage
from src.application.ports import RepositoryPort
from src.application.report_evidence import BENCHMARK_SYMBOL, PaperCosts, benchmark_comparison
from src.application.report_models import ResolvedPrediction
from src.domain.benchmark import BenchmarkComparison
from src.domain.coverage import CoverageReport

logger = logging.getLogger(__name__)


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
) -> CoverageReport | None:
    """DAT-16 — pokrycie danych okna track recordu z darmowych odczytów
    (snapshoty cen, rozliczenia). Błąd → None: zdanie się chowa, cykl leci dalej."""
    try:
        return build_data_coverage(
            repository,
            symbols=symbols,
            crypto_symbols=crypto_symbols,
            days=days,
            now=datetime.now(UTC),
        )
    except Exception:
        logger.exception("Data coverage failed — scorecards without the coverage line")
        return None
