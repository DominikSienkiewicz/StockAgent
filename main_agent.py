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
from decimal import Decimal

from src.application.ports import (
    AdvisoryCouncilPort,
    EmbeddingPort,
    LLMPort,
    ReportNotifierPort,
    RepositoryPort,
)
from src.application.report_builder import (
    SymbolResult,
    build_html_report,
    parse_resolved_predictions,
    to_symbol_result,
)
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
from src.application.use_cases.monitor_macro_risk import (
    MacroRiskReport,
    MonitorMacroRiskUseCase,
)
from src.config import Settings
from src.domain.asset import Asset
from src.domain.value_objects import AssetType, Threshold
from src.infrastructure.adapters.advisory_council import LLMAdvisoryCouncil
from src.infrastructure.adapters.alpha_vantage_adapters import (
    AlphaVantageNewsAdapter,
    AlphaVantageSentimentAdapter,
)
from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient
from src.infrastructure.adapters.cached_fundamentals import (
    CachedFundamentalsAdapter,
    NullFundamentalsAdapter,
)
from src.infrastructure.adapters.coingecko import CoinGeckoAdapter
from src.infrastructure.adapters.finnhub_api import FinnhubAdapter
from src.infrastructure.adapters.nbp_client import NbpClient
from src.infrastructure.adapters.resend_notifier import NullNotifier, ResendNotifier
from src.infrastructure.adapters.routing_market_data import RoutingMarketDataPort
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter

ACCURACY_STATS_DAYS = 30
RESOLVED_PREDICTIONS_HOURS = 24


def _ignored_result(symbol: str) -> SymbolResult:
    """Tworzy SymbolResult dla symbolu pre-filtrowanego z SYMBOLS_UNSUPPORTED_PRICE.

    Ląduje w mailu jako "ignored" (nie błąd) — żeby ograniczenia free tier
    Finnhub nie zaśmiecały sekcji błędów cyklu po cyklu.
    """
    return SymbolResult(
        symbol=symbol,
        status="ignored",
        error_message="unsupported by current price adapter (Finnhub free)",
    )

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


def build_council_llm_adapter(settings: Settings, main_llm: LLMPort) -> LLMPort:
    """Wybiera LLM dla rady doradczej.

    Domyślnie rada korzysta z `main_llm` (jedna instancja klienta SDK,
    wstecznie kompatybilne). Gdy `council_llm_provider` lub
    `council_llm_model` są ustawione → buduje OSOBNY adapter z innym
    modelem/providerem. Pozwala to puścić ~94% callsów cyklu (15 person × N
    symboli) na tańszym modelu, zachowując frontier model dla głównej analizy.
    """
    if settings.council_llm_provider is None and settings.council_llm_model is None:
        return main_llm

    provider = settings.council_llm_provider or settings.llm_provider
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "council_llm_provider=anthropic but ANTHROPIC_API_KEY is not set."
            )
        from src.infrastructure.llm.anthropic_client import (
            DEFAULT_MODEL as ANTHROPIC_DEFAULT,
        )
        from src.infrastructure.llm.anthropic_client import AnthropicAdapter

        return AnthropicAdapter(
            api_key=settings.anthropic_api_key,
            model=settings.council_llm_model or ANTHROPIC_DEFAULT,
        )

    from src.infrastructure.llm.openai_client import DEFAULT_MODEL as OPENAI_DEFAULT
    from src.infrastructure.llm.openai_client import OpenAIAdapter

    return OpenAIAdapter(
        api_key=settings.openai_api_key,
        model=settings.council_llm_model or OPENAI_DEFAULT,
    )


def build_council_adapter(
    settings: Settings, llm_port: LLMPort
) -> AdvisoryCouncilPort:
    """Pakuje LLM port + załadowane persony w `LLMAdvisoryCouncil`.

    Persony czytane z `settings.council_personas_dir` przy starcie agenta —
    fail-fast: jeśli katalog pusty/zepsuty, agent w ogóle nie startuje
    zamiast cicho lecieć z mniejszą/inną radą.
    """
    from pathlib import Path

    from src.infrastructure.persona_loader import load_council_personas

    personas = load_council_personas(Path(settings.council_personas_dir))
    return LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)


def build_embedding_adapter(settings: Settings) -> EmbeddingPort | None:
    """Embeddingi — OpenAI niezależnie od LLM_PROVIDER.

    Anthropic nie udostępnia natywnego embeddings API; OpenAI to nasz jedyny
    provider w stacku. `openai_api_key` jest wymaganym polem w `Settings`
    (bez default), więc jest zawsze obecne — embeddingi działają również
    gdy LLM_PROVIDER=anthropic. Dzięki temu kolumna `embedding` w
    `prediction_logs` (pgvector) jest wypełniona dla wszystkich deploymentów,
    a similarity search nad newsami nie zależy od wyboru LLM.
    """
    from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingAdapter

    return OpenAIEmbeddingAdapter(api_key=settings.openai_api_key)


def build_macro_risk_use_case(
    settings: Settings,
    repository: RepositoryPort,
) -> MonitorMacroRiskUseCase | None:
    """Buduje MonitorMacroRisk gdy są skonfigurowane tickery ryzyka.

    Brak `risk_symbols` lub `risk_symbol_types` = wyłączony Risk Watch.
    NBP wpinamy tylko gdy NBP_ENABLED=true (osobne źródło, osobny port).
    """
    if not settings.risk_symbols or not settings.risk_symbol_types:
        return None
    macro_port = NbpClient() if settings.nbp_enabled else None
    return MonitorMacroRiskUseCase(
        market_port=FinnhubAdapter(api_key=settings.finnhub_api_key),
        repository_port=repository,
        macro_port=macro_port,
    )


def build_use_case(
    settings: Settings,
    repository: RepositoryPort | None = None,
) -> AnalyzeMarketUseCase:
    """DI Container — buduje wszystkie adaptery i wstrzykuje do Use Case'a.

    SYMBOLS przekazujemy do AlphaVantageClient jako sumę akcji + krypto,
    z osobną listą crypto_symbols → adapter wewnętrznie prefiksuje BTC →
    CRYPTO:BTC w requestach NEWS_SENTIMENT.
    """
    all_av_symbols = list(settings.symbols) + list(settings.crypto_symbols)
    av_client = AlphaVantageClient(
        api_keys=settings.alpha_vantage_api_keys,
        symbols=all_av_symbols,
        crypto_symbols=settings.crypto_symbols,
    )
    llm_port = build_llm_adapter(settings)
    council_llm_port = build_council_llm_adapter(settings, llm_port)
    supabase_repo = repository or build_repository(settings)
    # Fast loop: delegate = Null — nie woła płatnego AV, czyta tylko cache Supabase.
    fundamentals_port = CachedFundamentalsAdapter(
        repo=supabase_repo,
        delegate=NullFundamentalsAdapter(),
    )
    # Routing: krypto → CoinGecko, reszta → Finnhub. Use case widzi jeden port.
    market_port = RoutingMarketDataPort(
        equity=FinnhubAdapter(api_key=settings.finnhub_api_key),
        crypto=CoinGeckoAdapter(),
        crypto_symbols=settings.crypto_symbols,
    )
    crypto_threshold = (
        Threshold(settings.crypto_volatility_threshold)
        if settings.crypto_symbols
        else None
    )
    # Próg rady dla krypto — używamy tego samego co głównego (5%), bo i tak
    # bramka volatility już odsiała pierwszy poziom; council ma jedynie
    # drugie sito kosztowe i nie ma sensu robić go ostrzejszego.
    crypto_council_threshold = (
        Threshold(settings.crypto_volatility_threshold)
        if settings.crypto_symbols
        else None
    )
    return AnalyzeMarketUseCase(
        market_port=market_port,
        sentiment_port=AlphaVantageSentimentAdapter(av_client),
        news_port=AlphaVantageNewsAdapter(av_client),
        repository_port=supabase_repo,
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        llm_port=llm_port,
        threshold=Threshold(settings.volatility_threshold),
        embedding_port=build_embedding_adapter(settings),
        council_port=build_council_adapter(settings, council_llm_port),
        # 0.0 → wyłączony (rada zawsze leci gdy główna bramka przepuści).
        council_threshold=(
            Threshold(settings.council_volatility_threshold)
            if settings.council_volatility_threshold > Decimal("0")
            else None
        ),
        fundamentals_port=fundamentals_port,
        crypto_threshold=crypto_threshold,
        crypto_council_threshold=crypto_council_threshold,
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
    logger.info(
        "Fast Loop start — symbols=%s crypto=%s",
        settings.symbols,
        settings.crypto_symbols,
    )

    results = []
    failures = 0
    unsupported = set(settings.symbols_unsupported_price)
    crypto_set = set(settings.crypto_symbols)
    etf_set = set(settings.symbols_etf)
    # Krypto dochodzi do puli iterowanych symboli — ma własny adapter ceny
    # (CoinGecko) i własny próg volatility (5%).
    all_symbols = list(settings.symbols) + list(settings.crypto_symbols)
    # Eligible — symbole, dla których pójdzie pełna analiza. Unsupported tickery
    # są oznaczone "ignored" przed pętlą, więc throttle policzy je poprawnie.
    eligible = [s for s in all_symbols if s not in unsupported]
    for skipped in (s for s in all_symbols if s in unsupported):
        logger.info(
            "%s: skipped — in SYMBOLS_UNSUPPORTED_PRICE (Finnhub free / EU-listed)",
            skipped,
        )
        results.append(_ignored_result(skipped))

    for index, symbol in enumerate(eligible):
        try:
            # Klasyfikacja STOCK/ETF/CRYPTO — każdy typ ma własne reguły
            # (pomijanie fundamentali dla ETF/CRYPTO, osobne progi dla CRYPTO).
            if symbol in crypto_set:
                asset_type = AssetType.CRYPTO
            elif symbol in etf_set:
                asset_type = AssetType.ETF
            else:
                asset_type = AssetType.STOCK
            asset = Asset(symbol=symbol, asset_type=asset_type)
            raw = use_case.run(symbol, asset=asset)
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

        # Throttle MIĘDZY symbolami — chroni OpenAI TPM. Po ostatnim brak sleepa.
        if settings.symbol_throttle_seconds > 0 and index < len(eligible) - 1:
            time.sleep(settings.symbol_throttle_seconds)

    duration = time.perf_counter() - started_perf
    logger.info("Fast Loop done — failures=%d/%d", failures, len(settings.symbols))

    # ---- Risk Watch (osobny pass, nie miesza się z predykcjami) ----
    macro_risk_report: MacroRiskReport | None = None
    risk_use_case = build_macro_risk_use_case(settings, repository)
    if risk_use_case is not None:
        try:
            symbol_types = {
                sym: settings.risk_symbol_types[sym]
                for sym in settings.risk_symbols
                if sym in settings.risk_symbol_types
            }
            macro_risk_report = risk_use_case.run(symbol_types)
            logger.info(
                "Risk Watch — overall=%s signals=%d nbp=%s",
                macro_risk_report.overall_alert.value,
                len(macro_risk_report.signals),
                macro_risk_report.polish_macro is not None,
            )
        except Exception:
            logger.exception("Risk Watch failed — pomijam sekcję w raporcie")

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
            macro_risk_report=macro_risk_report,
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
    total = len(eligible)
    if total > 0 and failures == total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
