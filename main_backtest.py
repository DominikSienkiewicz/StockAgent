"""Offline walk-forward backtest harness (T3 — Walk-Forward Backtest Harness).

Replayuje materializowany widok `ml_feature_store` przez foldy expanding-window
(reużywa logiki walk-forward z `XGBoostAdapter.walk_forward_report`, finding #23)
i raportuje metryki out-of-sample per symbol — RMSE kandydata vs baseline „zwrot=0"
oraz trafność kierunku. Czysta EWALUACJA: nie persystuje modelu, nie zmienia
żadnego stanu. Zero płatnych portów (czyta tylko zebrane cechy).

Uruchamiany ręcznie / przez GHA na żądanie:
    uv run python main_backtest.py
"""

from __future__ import annotations

import logging
import sys

from src.application.ml_features import FEATURE_SET_VERSION
from src.application.ports import RepositoryPort
from src.application.use_cases.backtest import BacktestUseCase
from src.config import Settings
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter
from src.infrastructure.logging_redaction import configure_logging

logger = logging.getLogger(__name__)


def _fmt_optional(value: float | None, pattern: str) -> str:
    """Liczba w logu albo `n/a`, gdy miara jest niezdefiniowana (np. jeden fold)."""
    return "n/a" if value is None else pattern % value


def _feature_set_version(settings: Settings) -> int | None:
    """ML-51: ta sama próbka co trening — z `feature_set_filter_enabled` tylko
    wiersze bieżącej wersji zestawu cech (kolumna z migracji 027)."""
    return FEATURE_SET_VERSION if settings.feature_set_filter_enabled else None


def build_use_case(
    settings: Settings,
    *,
    repository: RepositoryPort | None = None,
    cost_bps: float | None = None,
) -> BacktestUseCase:
    """DI Container — backtest potrzebuje tylko bazy (cechy) + XGBoost (foldy).

    `cost_bps` (WIA-50) domyślnie koszt akcji/ETF; krypto podaje własny."""
    return BacktestUseCase(
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        db_port=repository
        or SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key),
        cost_bps=settings.paper_cost_bps if cost_bps is None else cost_bps,
        feature_set_version=_feature_set_version(settings),
    )


def backtest_symbols(settings: Settings) -> list[str]:
    """NIEZ-18: akcje, ETF-y i krypto — kolejność z configu, bez duplikatów."""
    return list(dict.fromkeys([*settings.symbols, *settings.crypto_symbols]))


def main(settings: Settings | None = None) -> int:
    settings = settings or Settings.from_env()
    configure_logging(settings.secret_values())
    repository = SupabaseRepository(
        url=settings.supabase_url, key=settings.supabase_key
    )
    symbols = backtest_symbols(settings)
    crypto = set(settings.crypto_symbols)
    stock_backtest = build_use_case(settings, repository=repository)
    # WIA-50: krypto ma własny koszt obrotu; bez krypto drugi use case nie powstaje.
    crypto_backtest = (
        build_use_case(
            settings, repository=repository, cost_bps=settings.crypto_paper_cost_bps
        )
        if crypto
        else stock_backtest
    )

    logger.info("Backtest start — symbols=%s", symbols)
    # Świeży widok raz na cały przebieg; pojedyncze symbole już go nie odświeżają.
    try:
        repository.refresh_feature_store()
    except Exception:
        logger.exception(
            "refresh_feature_store failed — backtest na ostatnim widoku"
        )

    failures = 0
    for symbol in symbols:
        is_crypto = symbol in crypto
        use_case = crypto_backtest if is_crypto else stock_backtest
        cost_bps = settings.crypto_paper_cost_bps if is_crypto else settings.paper_cost_bps
        try:
            summary = use_case.run(symbol, refresh_view=False)
            # ML-20: IC i jego t-stat obok RMSE — zawsze z liczbą foldów, bo przy
            # 4 foldach t-stat mówi niewiele. WIA-50: zwrot netto strategii
            # znakowej po koszcie obrotu i sam obrót.
            logger.info(
                "%s: %s | folds=%d cand_rmse=%.5f base_rmse=%.5f "
                "drift_rmse=%s llm_rmse=%s hit=%.1f%% impr=%.1f%% ic=%s sd=%s t=%s "
                "net=%s turnover=%s cost=%gbps",
                symbol,
                summary.verdict,
                summary.n_folds,
                summary.mean_candidate_rmse,
                summary.mean_baseline_rmse,
                _fmt_optional(summary.mean_drift_baseline_rmse, "%.5f"),
                _fmt_optional(summary.mean_llm_baseline_rmse, "%.5f"),
                summary.mean_hit_rate * 100.0,
                summary.improvement_pct,
                _fmt_optional(summary.mean_ic, "%.3f"),
                _fmt_optional(summary.ic_std, "%.3f"),
                _fmt_optional(summary.ic_t_stat, "%.2f"),
                _fmt_optional(summary.mean_net_return, "%.5f"),
                _fmt_optional(summary.turnover, "%.2f"),
                cost_bps,
            )
        except Exception:
            failures += 1
            logger.exception("Backtest failed for %s", symbol)

    logger.info("Backtest done — symbols=%d failures=%d", len(symbols), failures)
    # NIEZ-18: jak Fast Loop — exit 1 tylko, gdy padły wszystkie symbole.
    return 1 if symbols and failures == len(symbols) else 0


if __name__ == "__main__":
    sys.exit(main())
