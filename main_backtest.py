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

from src.application.use_cases.backtest import BacktestUseCase
from src.config import Settings
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter
from src.infrastructure.logging_redaction import configure_logging

logger = logging.getLogger(__name__)


def _fmt_optional(value: float | None, pattern: str) -> str:
    """Liczba w logu albo `n/a`, gdy miara jest niezdefiniowana (np. jeden fold)."""
    return "n/a" if value is None else pattern % value


def build_use_case(settings: Settings) -> BacktestUseCase:
    """DI Container — backtest potrzebuje tylko bazy (cechy) + XGBoost (foldy)."""
    return BacktestUseCase(
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        db_port=SupabaseRepository(
            url=settings.supabase_url, key=settings.supabase_key
        ),
        cost_bps=settings.paper_cost_bps,
    )


def main(settings: Settings | None = None) -> int:
    settings = settings or Settings.from_env()
    configure_logging(settings.secret_values())
    repository = SupabaseRepository(
        url=settings.supabase_url, key=settings.supabase_key
    )
    # WIA-50: backtest iteruje `symbols` (bez krypto), więc koszt akcji/ETF.
    use_case = BacktestUseCase(
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        db_port=repository,
        cost_bps=settings.paper_cost_bps,
    )

    logger.info("Backtest start — symbols=%s", settings.symbols)
    # Świeży widok raz na cały przebieg; pojedyncze symbole już go nie odświeżają.
    try:
        repository.refresh_feature_store()
    except Exception:
        logger.exception(
            "refresh_feature_store failed — backtest na ostatnim widoku"
        )

    for symbol in settings.symbols:
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
                settings.paper_cost_bps,
            )
        except Exception:
            logger.exception("Backtest failed for %s", symbol)

    logger.info("Backtest done — symbols=%d", len(settings.symbols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
