"""Slow Loop entry point — uruchamiany przez GHA co tydzień.

Retrenuje model XGBoost na zebranych logach predykcyjnych.
Po treningu plik wag (.ubj) jest commitowany do repo przez workflow GHA.

Użycie:
    uv run python main_trainer.py
"""

from __future__ import annotations

import logging
import sys

from src.application.quota_monitor import QuotaMonitor
from src.application.use_cases.train_model import TrainModelUseCase
from src.config import Settings
from src.infrastructure.adapters.alpha_vantage_fundamentals import (
    AlphaVantageFundamentalsAdapter,
)
from src.infrastructure.adapters.cached_fundamentals import CachedFundamentalsAdapter
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
    quota_monitor = QuotaMonitor()
    supabase_repo = SupabaseRepository(
        url=settings.supabase_url, key=settings.supabase_key
    )
    use_case = build_use_case(settings)

    logger.info("Slow Loop start — symbols=%s", settings.symbols)
    try:
        use_case.refresh_feature_store()
    except Exception:
        logger.exception("Failed to refresh feature store")
        return 1

    # Slow loop płaci za Alpha Vantage requesty i odświeża cache fundamentów.
    # Fast loop czyta wyłącznie z cache — nigdy nie woła AV bezpośrednio.
    fundamentals_port = CachedFundamentalsAdapter(
        repo=supabase_repo,
        delegate=AlphaVantageFundamentalsAdapter(
            api_keys=settings.alpha_vantage_api_keys,
            quota_monitor=quota_monitor,
        ),
    )
    # Odświeżenie fundamentów dla spółek (STOCK) — ETF-y nie mają EPS/P/E.
    # Pojedynczy błąd nie wywala całego cyklu.
    stock_symbols = [s for s in settings.symbols if s not in settings.symbols_etf]
    for symbol in stock_symbols:
        try:
            fundamentals_port.get_fundamentals(symbol)
        except Exception:
            logger.exception("Fundamentals refresh failed for %s", symbol)

    # Persystencja alertów kwoty zebranych podczas odświeżania fundamentów
    # (np. wyczerpany dzienny limit AV) — banner w mailu czyta ostatnie 24h,
    # więc wyczerpanie limitu w Slow Loop będzie widoczne w najbliższym raporcie.
    for alert in quota_monitor.alerts:
        try:
            supabase_repo.save_quota_alert(alert)
        except Exception:
            logger.exception("Failed to persist quota alert from %s", alert.source)

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
