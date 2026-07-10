"""Slow Loop entry point — uruchamiany przez GHA co tydzień.

Retrenuje model XGBoost na zebranych logach predykcyjnych.
Po treningu plik wag (.ubj) jest commitowany do repo przez workflow GHA.

Użycie:
    uv run python main_trainer.py
"""

from __future__ import annotations

import logging
import sys

from main_agent import build_notifier
from src.application.quota_monitor import QuotaMonitor
from src.application.report_weekly_recap import (
    render_weekly_recap_html,
    render_weekly_recap_text,
)
from src.application.use_cases.train_model import TrainModelUseCase
from src.application.use_cases.weekly_recap import WeeklyRecapUseCase
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

    # BUG: trener widział tylko `settings.symbols`, więc BTC/ETH NIGDY nie
    # przechodziły retrainu — model krypto zostawał na zawsze przy cold-starcie.
    # `dict.fromkeys` zachowuje kolejność i deduplikuje symbol obecny w obu listach.
    trainable = list(dict.fromkeys([*settings.symbols, *settings.crypto_symbols]))

    failures = 0
    for symbol in trainable:
        try:
            result = use_case.run(symbol, refresh_view=False)
            logger.info("%s: %s", symbol, result)
            # #12: karta kondycji modelu. Zapis best-effort (wzorzec
            # `save_quota_alert`) — brak migracji 019 nie może wywalić treningu.
            if settings.model_scorecard_enabled:
                try:
                    supabase_repo.save_model_scorecard(symbol, result)
                except Exception:
                    logger.exception("Failed to persist model scorecard for %s", symbol)
        except Exception:
            logger.exception("Training failed for symbol %s", symbol)
            failures += 1

    # #4: LLM-as-Judge kalibracji pewności (Slow Loop, poza budżetem Fast Loopa).
    if settings.calibration_judge_enabled:
        try:
            from main_agent import build_llm_adapter
            from src.application.use_cases.calibrate_confidence import (
                CalibrateConfidenceUseCase,
            )

            cal_result = CalibrateConfidenceUseCase(
                repository_port=supabase_repo,
                llm_port=build_llm_adapter(settings, quota_monitor=quota_monitor),
            ).run(days=30)
            logger.info("Calibration judge — %s", cal_result.get("status"))
        except Exception:
            logger.exception("Calibration judge failed — pomijam")

    # #17: niedzielna retrospektywa. Slow Loop chodzi co tydzień i dotąd NIE
    # WYSYŁAŁ użytkownikowi nic. Koszt: dokładnie jedno wywołanie LLM tygodniowo.
    # Poniżej progu 5 zamkniętych predykcji use case zwraca `skipped_*` i mail
    # NIE wychodzi — przy chudym tygodniu recap robiłby dramat z szumu.
    if settings.weekly_recap_enabled:
        try:
            from main_agent import build_llm_adapter

            recap_result = WeeklyRecapUseCase(
                repository_port=supabase_repo,
                llm_port=build_llm_adapter(settings, quota_monitor=quota_monitor),
            ).run(days=7)
            if recap_result["status"] == "recap_ready":
                recap = recap_result["recap"]
                narrative = recap_result["narrative"]
                build_notifier(settings, quota_monitor=quota_monitor).send_report(
                    subject="📅 Tydzień StockAgenta",
                    html_body=render_weekly_recap_html(recap, narrative),
                    plain_text=render_weekly_recap_text(recap, narrative),
                )
            logger.info("Weekly recap — %s", recap_result.get("status"))
        except Exception:
            logger.exception("Weekly recap failed — pomijam")

    logger.info("Slow Loop done — failures=%d/%d", failures, len(trainable))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
