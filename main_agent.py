"""Fast Loop entry point — uruchamiany przez GHA co 12h (lub lokalnie).

Wczytuje konfigurację z `.env` lub zmiennych środowiskowych,
instancjuje wszystkie adaptery (DI Container), iteruje po liście symboli
i wywołuje AnalyzeMarketUseCase dla każdego z nich.

Po cyklu — jeśli włączone — wysyła raport mailem przez Resend.com.

Użycie:
    uv run python main_agent.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import UTC, datetime

from src.application.ports import LLMPort, ReportNotifierPort, RepositoryPort
from src.application.report_builder import (
    build_html_report,
    parse_resolved_predictions,
    to_symbol_result,
)
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
from src.config import Settings
from src.domain.value_objects import Threshold
from src.infrastructure.adapters.alpha_vantage_adapters import (
    AlphaVantageNewsAdapter,
    AlphaVantageSentimentAdapter,
)
from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient
from src.infrastructure.adapters.finnhub_api import FinnhubAdapter
from src.infrastructure.adapters.resend_notifier import NullNotifier, ResendNotifier
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter

ACCURACY_STATS_DAYS = 30
RESOLVED_PREDICTIONS_HOURS = 24

logger = logging.getLogger(__name__)


def build_llm_adapter(settings: Settings) -> LLMPort:
    """Factory LLM — wybiera providera na podstawie konfiguracji."""
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Install with: uv sync --extra anthropic"
            )
        from src.infrastructure.llm.anthropic_client import AnthropicAdapter

        return AnthropicAdapter(api_key=settings.anthropic_api_key)

    from src.infrastructure.llm.openai_client import OpenAIAdapter

    return OpenAIAdapter(api_key=settings.openai_api_key)


def build_notifier(settings: Settings) -> ReportNotifierPort:
    """Factory notifier — Resend gdy włączone i skonfigurowane, inaczej Null."""
    if (
        settings.notifications_enabled
        and settings.resend_api_key
        and settings.digest_to_email
    ):
        return ResendNotifier(
            api_key=settings.resend_api_key,
            sender=settings.digest_from_email,
            recipient=settings.digest_to_email,
        )
    return NullNotifier()


def build_repository(settings: Settings) -> RepositoryPort:
    return SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key)


def build_use_case(
    settings: Settings,
    repository: RepositoryPort | None = None,
) -> AnalyzeMarketUseCase:
    """DI Container — buduje wszystkie adaptery i wstrzykuje do Use Case'a."""
    av_client = AlphaVantageClient(
        api_keys=settings.alpha_vantage_api_keys,
        symbols=settings.symbols,
    )
    return AnalyzeMarketUseCase(
        market_port=FinnhubAdapter(api_key=settings.finnhub_api_key),
        sentiment_port=AlphaVantageSentimentAdapter(av_client),
        news_port=AlphaVantageNewsAdapter(av_client),
        repository_port=repository or build_repository(settings),
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        llm_port=build_llm_adapter(settings),
        threshold=Threshold(settings.volatility_threshold),
    )


def main(settings: Settings | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    settings = settings or Settings.from_env()
    repository = build_repository(settings)
    use_case = build_use_case(settings, repository=repository)
    notifier = build_notifier(settings)

    started_at = datetime.now(UTC)
    started_perf = time.perf_counter()
    logger.info("Fast Loop start — symbols=%s", settings.symbols)

    results = []
    failures = 0
    for symbol in settings.symbols:
        try:
            raw = use_case.run(symbol)
            logger.info(
                "%s: status=%s delta=%s prediction_id=%s",
                symbol,
                raw.get("status"),
                raw.get("delta"),
                raw.get("prediction_id"),
            )
            results.append(to_symbol_result(symbol, raw))
        except Exception as exc:
            logger.exception("Failed to process symbol %s", symbol)
            failures += 1
            results.append(to_symbol_result(symbol, None, error=str(exc)))

    duration = time.perf_counter() - started_perf
    logger.info("Fast Loop done — failures=%d/%d", failures, len(settings.symbols))

    # ---- Wysyłka raportu ----
    try:
        accuracy_stats = None
        resolved = []
        try:
            accuracy_stats = repository.get_accuracy_stats(ACCURACY_STATS_DAYS)
        except Exception:
            logger.exception("Failed to fetch accuracy stats — report without them")
        try:
            resolved_rows = repository.get_recently_resolved_predictions(
                RESOLVED_PREDICTIONS_HOURS
            )
            resolved = parse_resolved_predictions(resolved_rows)
        except Exception:
            logger.exception("Failed to fetch resolved predictions")

        html, text = build_html_report(
            results, started_at, duration,
            accuracy_stats=accuracy_stats,
            resolved_predictions=resolved,
        )
        analyzed = sum(1 for r in results if r.status == "saved")
        subject = (
            f"StockAgent — {analyzed} predykcji, {failures} błędów "
            f"({started_at.strftime('%Y-%m-%d %H:%M UTC')})"
        )
        notifier.send_report(subject=subject, html_body=html, plain_text=text)
    except Exception:
        logger.exception("Failed to send report")

    # Exit code 1 tylko gdy wszystkie symbole padły (catastrophic failure).
    # Pojedyncze błędy per-symbol (np. ticker niewspierany przez Finnhub free)
    # są raportowane w mailu i nie powinny psuć całego cyklu w GHA.
    total = len(settings.symbols)
    if total > 0 and failures == total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
