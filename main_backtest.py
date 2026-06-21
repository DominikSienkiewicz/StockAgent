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

logger = logging.getLogger(__name__)


def build_use_case(settings: Settings) -> BacktestUseCase:
    """DI Container — backtest potrzebuje tylko bazy (cechy) + XGBoost (foldy)."""
    return BacktestUseCase(
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        db_port=SupabaseRepository(
            url=settings.supabase_url, key=settings.supabase_key
        ),
    )


def main(settings: Settings | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    settings = settings or Settings.from_env()
    repository = SupabaseRepository(
        url=settings.supabase_url, key=settings.supabase_key
    )
    use_case = BacktestUseCase(
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        db_port=repository,
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
            logger.info(
                "%s: %s | folds=%d cand_rmse=%.5f base_rmse=%.5f "
                "hit=%.1f%% impr=%.1f%%",
                symbol,
                summary.verdict,
                summary.n_folds,
                summary.mean_candidate_rmse,
                summary.mean_baseline_rmse,
                summary.mean_hit_rate * 100.0,
                summary.improvement_pct,
            )
        except Exception:
            logger.exception("Backtest failed for %s", symbol)

    logger.info("Backtest done — symbols=%d", len(settings.symbols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
