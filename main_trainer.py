"""Slow Loop entry point — uruchamiany przez GHA co tydzień.

Retrenuje model XGBoost na zebranych logach predykcyjnych.
Po treningu plik wag (.ubj) jest commitowany do repo przez workflow GHA.

Użycie:
    uv run python main_trainer.py
"""

from __future__ import annotations

import logging
import sys

from src.application.use_cases.train_model import TrainModelUseCase
from src.config import Settings
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter

logger = logging.getLogger(__name__)


def build_use_case(settings: Settings) -> TrainModelUseCase:
    """DI Container — Slow Loop potrzebuje tylko bazy + XGBoost."""
    return TrainModelUseCase(
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        db_port=SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key),
    )


def main(settings: Settings | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    settings = settings or Settings.from_env()
    use_case = build_use_case(settings)

    logger.info("Slow Loop start — symbols=%s", settings.symbols)
    try:
        use_case.refresh_feature_store()
    except Exception:
        logger.exception("Failed to refresh feature store")
        return 1

    failures = 0
    for symbol in settings.symbols:
        try:
            result = use_case.run(symbol, refresh_view=False)
            logger.info("%s: %s", symbol, result)
        except Exception:
            logger.exception("Training failed for symbol %s", symbol)
            failures += 1

    logger.info("Slow Loop done — failures=%d/%d", failures, len(settings.symbols))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
