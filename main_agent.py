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
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, NamedTuple

from src.application.alpha_signals import AlphaSignals
from src.application.ports import (
    AdvisoryCouncilPort,
    AnalystConsensusPort,
    EarningsCalendarPort,
    EmbeddingPort,
    InsiderFlowPort,
    LLMPort,
    MacroRatesPort,
    OptionsFlowPort,
    ReportNotifierPort,
    RepositoryPort,
    SocialVelocityPort,
    ToolUseLLMPort,
    WebPublisherPort,
)
from src.application.quota_monitor import QuotaMonitor
from src.application.report_builder import (
    SymbolResult,
    build_html_report,
    parse_resolved_predictions,
    to_symbol_result,
)
from src.application.report_council_history import InvestorHistory
from src.application.report_lessons import LessonsReport, build_lessons
from src.application.report_models import ResolvedPrediction
from src.application.tools import build_research_tools
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
from src.application.use_cases.enrich_alpha import AlphaEnrichmentUseCase
from src.application.use_cases.monitor_macro_risk import (
    MacroRiskReport,
    MonitorMacroRiskUseCase,
)
from src.application.use_cases.portfolio_risk import (
    PortfolioRiskReport,
    PortfolioRiskUseCase,
)
from src.config import Settings
from src.domain.asset import Asset
from src.domain.calibration_curve import CalibrationBucket, reliability_curve
from src.domain.earnings import EarningsEvent, earnings_threshold_multiplier
from src.domain.equity_curve import EquityCurve
from src.domain.equity_curve import equity_curve as build_equity_curve
from src.domain.macro_rates import YieldCurveSnapshot, yield_curve_alert_level
from src.domain.macro_risk import MacroAlertLevel
from src.domain.peer_group import build_peer_context
from src.domain.persona_track_record import PersonaTrackRecord, rank_personas
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.scenarios import Scenario
from src.domain.subscriber import Subscriber, symbols_for
from src.domain.value_objects import AssetType, Threshold
from src.infrastructure.adapters.advisory_council import LLMAdvisoryCouncil
from src.infrastructure.adapters.alert_notifier import DelegatingAlertNotifier
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
from src.infrastructure.adapters.composite_notifier import CompositeNotifier
from src.infrastructure.adapters.finnhub_api import FinnhubAdapter
from src.infrastructure.adapters.nbp_client import NbpClient
from src.infrastructure.adapters.resend_notifier import NullNotifier, ResendNotifier
from src.infrastructure.adapters.routing_market_data import RoutingMarketDataPort
from src.infrastructure.adapters.subscriber_repo import SupabaseSubscriberRepository
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter

# R2: domyślne scenariusze stress-testu (szok rynku jako ułamek). Szok stosowany
# do portfela równoważonego przez β-to-portfel każdego symbolu (domena liczy resztę).
DEFAULT_STRESS_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="Korekta rynku -10%", market_shock=-0.10),
    Scenario(name="Krach -20%", market_shock=-0.20),
    Scenario(name="Rajd +10%", market_shock=0.10),
)

ACCURACY_STATS_DAYS = 30
RESOLVED_PREDICTIONS_HOURS = 24
# Okno track recordu person (#3) — 90 dni, tak jak domyślne okno RPC z migracji
# 018. Dłuższe niż okno accuracy: hit-rate per persona potrzebuje próbki.
PERSONA_TRACK_RECORD_DAYS = 90

logger = logging.getLogger(__name__)


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


def _build_single_llm(
    settings: Settings,
    provider: str,
    model: str,
    quota_monitor: QuotaMonitor | None,
) -> LLMPort:
    """Buduje pojedynczy adapter LLM dla danego providera/modelu."""
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("router_frontier_provider=anthropic but ANTHROPIC_API_KEY unset.")
        from src.infrastructure.llm.anthropic_client import AnthropicAdapter

        return AnthropicAdapter(
            api_key=settings.anthropic_api_key, model=model, quota_monitor=quota_monitor
        )
    from src.infrastructure.llm.openai_client import OpenAIAdapter

    return OpenAIAdapter(
        api_key=settings.openai_api_key, model=model, quota_monitor=quota_monitor
    )


def build_llm_adapter(
    settings: Settings, quota_monitor: QuotaMonitor | None = None
) -> LLMPort:
    """Factory LLM — wybiera providera na podstawie konfiguracji.

    #2: gdy `model_routing_enabled`, zwraca `CascadingLLMAdapter` — tani model
    domyślnie, eskalacja do frontier tylko gdy `confidence < routing_floor`.
    """
    if settings.model_routing_enabled:
        from src.infrastructure.llm.cascading import CascadingLLMAdapter
        from src.infrastructure.llm.openai_client import OpenAIAdapter

        cheap = OpenAIAdapter(
            api_key=settings.openai_api_key,
            model=settings.router_cheap_model,
            quota_monitor=quota_monitor,
        )
        frontier = _build_single_llm(
            settings,
            settings.router_frontier_provider,
            settings.router_frontier_model,
            quota_monitor,
        )
        return CascadingLLMAdapter(
            cheap=cheap, frontier=frontier, routing_floor=settings.routing_floor
        )

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Install with: uv sync --extra anthropic"
            )
        from src.infrastructure.llm.anthropic_client import AnthropicAdapter

        return AnthropicAdapter(
            api_key=settings.anthropic_api_key, quota_monitor=quota_monitor
        )

    from src.infrastructure.llm.openai_client import OpenAIAdapter

    return OpenAIAdapter(
        api_key=settings.openai_api_key, quota_monitor=quota_monitor
    )


def build_tool_use_adapter(
    settings: Settings, quota_monitor: QuotaMonitor | None = None
) -> ToolUseLLMPort | None:
    """#6: adapter LLM z pętlą tool-use. None gdy wyłączony (tool_use_enabled=false).

    Provider wybierany jak w `build_llm_adapter` (LLM_PROVIDER). `max_rounds` to
    twardy cap rund tool-call (FinOps): worst-case wywołań płatnych na analizę =
    max_rounds + 1. Read-only toole budujemy osobno w `build_use_case`.
    """
    if not settings.tool_use_enabled:
        return None
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "TOOL_USE_ENABLED=true i LLM_PROVIDER=anthropic, ale "
                "ANTHROPIC_API_KEY nie jest ustawiony."
            )
        from src.infrastructure.llm.tool_calling_anthropic import (
            ToolCallingAnthropicAdapter,
        )

        return ToolCallingAnthropicAdapter(
            api_key=settings.anthropic_api_key,
            max_rounds=settings.tool_use_max_rounds,
            quota_monitor=quota_monitor,
        )
    from src.infrastructure.llm.tool_calling_openai import ToolCallingOpenAIAdapter

    return ToolCallingOpenAIAdapter(
        api_key=settings.openai_api_key,
        max_rounds=settings.tool_use_max_rounds,
        quota_monitor=quota_monitor,
    )


def build_push_channels(
    settings: Settings, quota_monitor: QuotaMonitor | None = None
) -> list[ReportNotifierPort]:
    """Buduje kanały push (Telegram/Slack) dla obecnych sekretów (U1).

    Kanał jest dołączany TYLKO gdy jego sekrety są ustawione — analogicznie do
    Resend. Lista może być pusta (brak skonfigurowanych kanałów)."""
    from src.infrastructure.adapters.slack_notifier import SlackNotifier
    from src.infrastructure.adapters.telegram_notifier import TelegramNotifier

    channels: list[ReportNotifierPort] = []
    if settings.telegram_bot_token and settings.telegram_chat_id:
        channels.append(
            TelegramNotifier(
                bot_token=settings.telegram_bot_token,
                chat_id=settings.telegram_chat_id,
                quota_monitor=quota_monitor,
            )
        )
    if settings.slack_webhook_url:
        channels.append(
            SlackNotifier(
                webhook_url=settings.slack_webhook_url,
                quota_monitor=quota_monitor,
            )
        )
    return channels


def build_notifier(
    settings: Settings, quota_monitor: QuotaMonitor | None = None
) -> ReportNotifierPort:
    """Factory notifier — Resend + kanały push (Telegram/Slack) jako jeden
    `ReportNotifierPort` (CompositeNotifier). Każdy skonfigurowany kanał dostaje
    kopię dziennego raportu; brak jakiegokolwiek kanału → Null.

    Gdy wpada w Null mimo `notifications_enabled=true`, loguje WARNING z nazwą
    brakującego sekretu — inaczej "Notifications disabled" nie mówi, czego
    brakuje (typowo: DIGEST_TO_EMAIL / RESEND_API_KEY nie ustawione jako
    GitHub repo *Secret*, nie Variable).
    """
    channels: list[ReportNotifierPort] = []
    if (
        settings.notifications_enabled
        and settings.resend_api_key
        and settings.digest_to_email
    ):
        channels.append(
            ResendNotifier(
                api_key=settings.resend_api_key,
                sender=settings.digest_from_email,
                recipient=settings.digest_to_email,
                quota_monitor=quota_monitor,
            )
        )
    elif not settings.notifications_enabled:
        logger.info("Email wyłączony (notifications_enabled=false w config.toml).")
    else:
        missing = []
        if not settings.resend_api_key:
            missing.append("RESEND_API_KEY")
        if not settings.digest_to_email:
            missing.append("DIGEST_TO_EMAIL")
        logger.warning(
            "Email WŁĄCZONY (notifications_enabled=true), ale brakuje sekretów: "
            "%s — ustaw je jako GitHub repo Secrets (nie Variables) / w .env.",
            ", ".join(missing),
        )

    # Kanały push (Telegram/Slack) dochodzą do dziennego raportu obok maila.
    channels.extend(build_push_channels(settings, quota_monitor=quota_monitor))

    if not channels:
        return NullNotifier()
    if len(channels) == 1:
        return channels[0]
    return CompositeNotifier(channels)


def build_repository(settings: Settings) -> RepositoryPort:
    return SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key)


def build_web_publisher(settings: Settings) -> WebPublisherPort:
    """U3 — publisher statycznego digestu web. File gdy włączone, inaczej Null."""
    from src.infrastructure.adapters.web_publisher import (
        FileWebPublisher,
        NullWebPublisher,
    )

    if settings.web_digest_enabled:
        return FileWebPublisher(output_path=settings.web_digest_path)
    return NullWebPublisher()


def build_council_llm_adapter(
    settings: Settings,
    main_llm: LLMPort,
    quota_monitor: QuotaMonitor | None = None,
) -> LLMPort:
    """Wybiera LLM dla rady doradczej.

    Domyślnie rada korzysta z `main_llm` (jedna instancja klienta SDK,
    wstecznie kompatybilne). Gdy `council_llm_provider` lub
    `council_llm_model` są ustawione → buduje OSOBNY adapter z innym
    modelem/providerem. Pozwala to puścić większość callsów cyklu (N person ×
    M symboli) na tańszym modelu, zachowując frontier model dla głównej analizy.
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
            quota_monitor=quota_monitor,
        )

    from src.infrastructure.llm.openai_client import DEFAULT_MODEL as OPENAI_DEFAULT
    from src.infrastructure.llm.openai_client import OpenAIAdapter

    return OpenAIAdapter(
        api_key=settings.openai_api_key,
        model=settings.council_llm_model or OPENAI_DEFAULT,
        quota_monitor=quota_monitor,
    )


def build_council_adapter(
    settings: Settings,
    llm_port: LLMPort,
    repository: RepositoryPort | None = None,
) -> AdvisoryCouncilPort:
    """Pakuje LLM port + załadowane persony w `LLMAdvisoryCouncil`.

    Persony czytane z `settings.council_personas_dir` przy starcie agenta —
    fail-fast: jeśli katalog pusty/zepsuty, agent w ogóle nie startuje
    zamiast cicho lecieć z mniejszą/inną radą.

    #3: gdy `persona_weighting_enabled` i jest repozytorium, wstrzykuje dostawcę
    wag person (cache per asset_type, jeden odczyt na klasę aktywa/cykl).
    #1: `debate_mode` z configu — jedna runda rewizji przy podzielonym round-1.
    """
    from collections.abc import Mapping
    from pathlib import Path

    from src.infrastructure.persona_loader import load_council_personas

    personas = load_council_personas(Path(settings.council_personas_dir))

    weights_provider = None
    if settings.persona_weighting_enabled and repository is not None:
        repo = repository
        _cache: dict[AssetType, Mapping[str, float]] = {}

        def weights_provider(asset_type: AssetType) -> Mapping[str, float]:
            if asset_type not in _cache:
                _cache[asset_type] = repo.get_persona_accuracy(asset_type)
            return _cache[asset_type]

    return LLMAdvisoryCouncil(
        llm_port=llm_port,
        personas=personas,
        weights_provider=weights_provider,
        debate_mode=settings.debate_mode_enabled,
    )


def build_embedding_adapter(
    settings: Settings, quota_monitor: QuotaMonitor | None = None
) -> EmbeddingPort | None:
    """Embeddingi — OpenAI niezależnie od LLM_PROVIDER.

    Anthropic nie udostępnia natywnego embeddings API; OpenAI to nasz jedyny
    provider w stacku. `openai_api_key` jest wymaganym polem w `Settings`
    (bez default), więc jest zawsze obecne — embeddingi działają również
    gdy LLM_PROVIDER=anthropic. Dzięki temu kolumna `embedding` w
    `prediction_logs` (pgvector) jest wypełniona dla wszystkich deploymentów,
    a similarity search nad newsami nie zależy od wyboru LLM.
    """
    from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingAdapter

    return OpenAIEmbeddingAdapter(
        api_key=settings.openai_api_key, quota_monitor=quota_monitor
    )


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
    # R3: hedge-effectiveness liczy korelację instrumentów ryzyka vs portfel
    # bazowy (akcje + krypto). Flag-gated; book = analizowane symbole.
    book_symbols = list(settings.symbols) + list(settings.crypto_symbols)
    return MonitorMacroRiskUseCase(
        market_port=FinnhubAdapter(api_key=settings.finnhub_api_key),
        repository_port=repository,
        macro_port=macro_port,
        book_symbols=book_symbols,
        hedge_enabled=settings.hedge_effectiveness_enabled,
    )


def alpha_per_symbol_enabled(settings: Settings) -> bool:
    """Czy włączone JAKIEKOLWIEK per-symbol źródło alpha (Dane)."""
    return (
        settings.insider_flow_enabled
        or settings.earnings_calendar_enabled
        or settings.options_flow_enabled
        or settings.social_velocity_enabled
        or settings.analyst_consensus_enabled
    )


def build_alpha_enrichment(settings: Settings) -> AlphaEnrichmentUseCase:
    """DI dla nowych źródeł alpha (Dane). Każdy port to realny adapter (gdy
    flaga on) albo Null-adapter (default → zero I/O). EDGAR/Reddit są darmowe;
    earnings reużywa klucza AV; opcje/analitycy reużywają klucza Finnhub."""
    from src.infrastructure.adapters.alpha_vantage_earnings import (
        AlphaVantageEarningsAdapter,
        NullEarningsCalendarAdapter,
    )
    from src.infrastructure.adapters.edgar_insider import (
        EdgarInsiderAdapter,
        NullInsiderFlowAdapter,
    )
    from src.infrastructure.adapters.finnhub_analyst import (
        FinnhubAnalystAdapter,
        NullAnalystConsensusAdapter,
    )
    from src.infrastructure.adapters.finnhub_options import (
        FinnhubOptionsAdapter,
        NullOptionsFlowAdapter,
    )
    from src.infrastructure.adapters.reddit_social import (
        NullSocialVelocityAdapter,
        RedditSocialAdapter,
    )

    insider: InsiderFlowPort = (
        EdgarInsiderAdapter(user_agent=settings.edgar_user_agent)
        if settings.insider_flow_enabled
        else NullInsiderFlowAdapter()
    )
    earnings: EarningsCalendarPort = (
        AlphaVantageEarningsAdapter(api_key=settings.alpha_vantage_api_keys[0])
        if settings.earnings_calendar_enabled and settings.alpha_vantage_api_keys
        else NullEarningsCalendarAdapter()
    )
    options: OptionsFlowPort = (
        FinnhubOptionsAdapter(api_key=settings.finnhub_api_key)
        if settings.options_flow_enabled
        else NullOptionsFlowAdapter()
    )
    social: SocialVelocityPort = (
        RedditSocialAdapter(user_agent=settings.edgar_user_agent)
        if settings.social_velocity_enabled
        else NullSocialVelocityAdapter()
    )
    analyst: AnalystConsensusPort = (
        FinnhubAnalystAdapter(api_key=settings.finnhub_api_key)
        if settings.analyst_consensus_enabled
        else NullAnalystConsensusAdapter()
    )
    return AlphaEnrichmentUseCase(
        insider_port=insider,
        earnings_port=earnings,
        options_port=options,
        social_port=social,
        analyst_port=analyst,
    )


def build_macro_rates_port(settings: Settings) -> MacroRatesPort:
    """FRED yield-curve port — realny gdy włączone + jest klucz, inaczej Null."""
    if settings.yield_curve_enabled and settings.fred_api_key:
        from src.infrastructure.adapters.fred_client import FredClient

        return FredClient(api_key=settings.fred_api_key)
    from src.infrastructure.adapters.fred_client import NullMacroRatesAdapter

    return NullMacroRatesAdapter()


def build_use_case(
    settings: Settings,
    repository: RepositoryPort | None = None,
    quota_monitor: QuotaMonitor | None = None,
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
        quota_monitor=quota_monitor,
    )
    llm_port = build_llm_adapter(settings, quota_monitor=quota_monitor)
    council_llm_port = build_council_llm_adapter(
        settings, llm_port, quota_monitor=quota_monitor
    )
    supabase_repo = repository or build_repository(settings)
    # Fast loop: delegate = Null — nie woła płatnego AV, czyta tylko cache Supabase.
    fundamentals_port = CachedFundamentalsAdapter(
        repo=supabase_repo,
        delegate=NullFundamentalsAdapter(),
    )
    # Routing: krypto → CoinGecko, reszta → Finnhub. Use case widzi jeden port.
    market_port = RoutingMarketDataPort(
        equity=FinnhubAdapter(
            api_key=settings.finnhub_api_key, quota_monitor=quota_monitor
        ),
        crypto=CoinGeckoAdapter(quota_monitor=quota_monitor),
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
    # #6: tool-use research agent — adapter + read-only toole (fundamenty z cache,
    # makro NBP). Budowane tylko gdy włączony. Toole NIE robią nowych płatnych
    # wywołań: fast loop czyta fundamenty z cache, a NBP jest darmowe i bezkluczowe.
    tool_use_port = build_tool_use_adapter(settings, quota_monitor=quota_monitor)
    research_tools = (
        build_research_tools(fundamentals_port, NbpClient())
        if tool_use_port is not None
        else []
    )
    tool_use_threshold = (
        Threshold(settings.tool_use_volatility_threshold)
        if tool_use_port is not None
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
        embedding_port=build_embedding_adapter(settings, quota_monitor=quota_monitor),
        council_port=build_council_adapter(
            settings, council_llm_port, repository=supabase_repo
        ),
        # 0.0 → wyłączony (rada zawsze leci gdy główna bramka przepuści).
        council_threshold=(
            Threshold(settings.council_volatility_threshold)
            if settings.council_volatility_threshold > Decimal("0")
            else None
        ),
        fundamentals_port=fundamentals_port,
        crypto_threshold=crypto_threshold,
        crypto_council_threshold=crypto_council_threshold,
        reflection_min_age_hours=settings.reflection_min_age_hours,
        rag_outcome_weight=settings.rag_outcome_weight,
        tool_use_port=tool_use_port,
        research_tools=research_tools,
        tool_use_threshold=tool_use_threshold,
        vector_memory_enabled=settings.vector_memory_enabled,
    )


def _run_risk_watch(
    settings: Settings, repository: RepositoryPort
) -> MacroRiskReport | None:
    """Risk Watch przed pętlą (zasila też reżim rynku #7). Off / błąd → None."""
    risk_use_case = build_macro_risk_use_case(settings, repository)
    if risk_use_case is None:
        return None
    try:
        symbol_types = {
            sym: settings.risk_symbol_types[sym]
            for sym in settings.risk_symbols
            if sym in settings.risk_symbol_types
        }
        report = risk_use_case.run(symbol_types)
        logger.info(
            "Risk Watch — overall=%s signals=%d nbp=%s",
            report.overall_alert.value,
            len(report.signals),
            report.polish_macro is not None,
        )
        return report
    except Exception:
        logger.exception("Risk Watch failed — pomijam sekcję w raporcie")
        return None


def _resolve_regime(
    settings: Settings,
    macro_risk_report: MacroRiskReport | None,
    yield_curve: YieldCurveSnapshot | None = None,
) -> tuple[str, float]:
    """#7: reżim rynku z overall_alert (zacieśnia próg + label do promptu).
    Zwraca (regime_context, regime_multiplier). Off / brak raportu → ("", 1.0).

    Inwersja krzywej rentowności wchodzi tu jako GŁOS w liczniku ELEVATED
    (przez `risk_signals`), a NIE przez `_compute_overall_alert` — max dałby złą
    semantykę, bo inwersja podbiłaby overall_alert i sama wymusiła RISK_OFF.
    Reguła `RegimeDetector` (">= 2 głosy ELEVATED → RISK_OFF") sprawia, że sama
    inwersja daje tylko NEUTRAL. To celowe: inwersje bywają wielomiesięczne,
    a weto trzymałoby agenta w risk-off przez cały ten czas.
    Snapshot jest już pobrany przed tym wywołaniem — zero dodatkowego fetchu.
    """
    if not (settings.regime_aware_enabled and macro_risk_report is not None):
        return "", 1.0
    from src.domain.regime import RegimeDetector

    # Głos krzywej wymaga JEDNOCZEŚNIE yield_curve_enabled i regime_aware_enabled.
    risk_signals: list[MacroAlertLevel] = []
    if settings.yield_curve_enabled and yield_curve is not None:
        risk_signals.append(yield_curve_alert_level(yield_curve.state()))

    detector = RegimeDetector()
    regime = detector.classify(macro_risk_report.overall_alert, risk_signals)
    regime_multiplier = detector.threshold_multiplier(
        regime, settings.regime_threshold_max_multiplier
    )
    regime_context = detector.label(regime)
    logger.info(
        "Market regime — %s (mnożnik progu %.2f)", regime.value, regime_multiplier
    )
    return regime_context, regime_multiplier


def _classify_asset(symbol: str, crypto_set: set[str], etf_set: set[str]) -> AssetType:
    """STOCK/ETF/CRYPTO — pomijanie fundamentali dla ETF/CRYPTO, osobne progi CRYPTO."""
    if symbol in crypto_set:
        return AssetType.CRYPTO
    if symbol in etf_set:
        return AssetType.ETF
    return AssetType.STOCK


class _ProcessedSymbol(NamedTuple):
    """Wynik przetworzenia jednego symbolu — niezależny od ścieżki (seq/parallel)."""

    result: SymbolResult
    is_failure: bool
    alpha_signal: AlphaSignals | None
    alpha_price: Decimal | None


def _process_one_symbol(
    symbol: str,
    *,
    peer_ctx: tuple[tuple[str, str], ...],
    use_case: AnalyzeMarketUseCase,
    crypto_set: set[str],
    etf_set: set[str],
    regime_context: str,
    regime_multiplier: float,
    collect_alpha: bool,
    alpha_use_case: AlphaEnrichmentUseCase,
    earnings_gate_enabled: bool = False,
) -> _ProcessedSymbol:
    """Przetwarza jeden symbol. NIGDY nie rzuca — łapie wyjątek i zwraca wynik
    z `is_failure=True`. Dzięki temu pula wątków (ścieżka równoległa
    `_analyze_symbols`) nigdy nie widzi wyjątku, a semantyka resilience jest
    identyczna z dawną pętlą sekwencyjną.

    `earnings_gate_enabled` (#6) domyślnie WYŁĄCZONE — wtedy ścieżka jest
    bit-w-bit taka jak przed wprowadzeniem bramki: zero prefetchu, `enrich`
    wołany dokładnie jak dawniej. Włącza je `settings.earnings_calendar_enabled`.
    """
    try:
        asset = Asset(
            symbol=symbol,
            asset_type=_classify_asset(symbol, crypto_set, etf_set),
        )
        # #6: bramka earnings. Event pobieramy PRZED grafem, bo mnożnik progu
        # musi być znany zanim graf zdecyduje o odpaleniu płatnych portów.
        # Gdy kalendarz earnings jest wyłączony, nie ruszamy portu w ogóle.
        earnings_event: EarningsEvent | None = None
        earnings_multiplier = 1.0
        if earnings_gate_enabled:
            earnings_event = alpha_use_case.fetch_earnings(symbol)
            if earnings_event is not None:
                earnings_multiplier = earnings_threshold_multiplier(
                    earnings_event.proximity()
                )
        raw = use_case.run(
            symbol,
            asset=asset,
            regime_context=regime_context,
            regime_multiplier=regime_multiplier,
            earnings_multiplier=earnings_multiplier,
            peer_context=peer_ctx,
        )
        logger.info(
            "%s: status=%s delta=%s prediction_id=%s",
            symbol,
            raw.get("status"),
            raw.get("delta"),
            raw.get("prediction_id"),
        )
        sr = to_symbol_result(symbol, raw)
        alpha_signal: AlphaSignals | None = None
        alpha_price: Decimal | None = None
        if collect_alpha:
            # De-dup: gdy bramka earnings pobrała już event, wstrzykujemy go,
            # zamiast płacić za drugie wywołanie Alpha Vantage na ten sam symbol.
            # Z wyłączoną bramką wołamy `enrich(symbol)` dokładnie jak dawniej.
            prefetch: dict[str, Any] = (
                {"earnings": earnings_event, "earnings_prefetched": True}
                if earnings_gate_enabled
                else {}
            )
            alpha_signal = alpha_use_case.enrich(symbol, **prefetch)
            if sr.current_price is not None:
                alpha_price = sr.current_price
        return _ProcessedSymbol(sr, False, alpha_signal, alpha_price)
    except Exception as exc:
        logger.exception("Failed to process symbol %s", symbol)
        return _ProcessedSymbol(
            to_symbol_result(symbol, None, error=str(exc)), True, None, None
        )


def _analyze_symbols_parallel(
    eligible: list[str],
    *,
    max_workers: int,
    use_case: AnalyzeMarketUseCase,
    crypto_set: set[str],
    etf_set: set[str],
    regime_context: str,
    regime_multiplier: float,
    collect_alpha: bool,
    alpha_use_case: AlphaEnrichmentUseCase,
    earnings_gate_enabled: bool = False,
) -> tuple[list[SymbolResult], int, dict[str, AlphaSignals], dict[str, Decimal]]:
    """Ścieżka równoległa: concurrency > 1 ORAZ contagion off (contagion #5 zależy
    od kolejności przetwarzania, więc jest niekompatybilny z out-of-order).
    executor.map zachowuje kolejność wejścia → wyniki deterministyczne."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed = list(
            executor.map(
                lambda sym: _process_one_symbol(
                    sym,
                    peer_ctx=(),
                    use_case=use_case,
                    crypto_set=crypto_set,
                    etf_set=etf_set,
                    regime_context=regime_context,
                    regime_multiplier=regime_multiplier,
                    collect_alpha=collect_alpha,
                    alpha_use_case=alpha_use_case,
                    earnings_gate_enabled=earnings_gate_enabled,
                ),
                eligible,
            )
        )
    return (
        [p.result for p in processed],
        sum(1 for p in processed if p.is_failure),
        {
            sym: p.alpha_signal
            for sym, p in zip(eligible, processed, strict=True)
            if p.alpha_signal is not None
        },
        {
            sym: p.alpha_price
            for sym, p in zip(eligible, processed, strict=True)
            if p.alpha_price is not None
        },
    )


def _analyze_symbols_sequential(
    eligible: list[str],
    *,
    use_case: AnalyzeMarketUseCase,
    settings: Settings,
    crypto_set: set[str],
    etf_set: set[str],
    regime_context: str,
    regime_multiplier: float,
    collect_alpha: bool,
    alpha_use_case: AlphaEnrichmentUseCase,
) -> tuple[list[SymbolResult], int, dict[str, AlphaSignals], dict[str, Decimal]]:
    """Ścieżka sekwencyjna: zachowuje kolejność, buduje peer_context dla contagion
    (#5) i throttluje między symbolami (ochrona OpenAI TPM)."""
    results: list[SymbolResult] = []
    failures = 0
    # #5: akumulator werdyktów tego cyklu (symbol → stance) dla contagion.
    processed_stances: dict[str, str] = {}
    alpha_signals: dict[str, AlphaSignals] = {}
    alpha_prices: dict[str, Decimal] = {}
    for index, symbol in enumerate(eligible):
        peer_ctx = (
            build_peer_context(symbol, settings.peer_groups, processed_stances)
            if settings.contagion_enabled
            else ()
        )
        proc = _process_one_symbol(
            symbol,
            peer_ctx=peer_ctx,
            use_case=use_case,
            crypto_set=crypto_set,
            etf_set=etf_set,
            regime_context=regime_context,
            regime_multiplier=regime_multiplier,
            collect_alpha=collect_alpha,
            alpha_use_case=alpha_use_case,
            earnings_gate_enabled=settings.earnings_calendar_enabled,
        )
        results.append(proc.result)
        if proc.is_failure:
            failures += 1
        elif proc.result.trend:  # zasil akumulator contagion dla kolejnych symboli
            processed_stances[symbol] = proc.result.trend
        if proc.alpha_signal is not None:
            alpha_signals[symbol] = proc.alpha_signal
        if proc.alpha_price is not None:
            alpha_prices[symbol] = proc.alpha_price
        # Throttle MIĘDZY symbolami — chroni OpenAI TPM. Po ostatnim brak sleepa.
        if settings.symbol_throttle_seconds > 0 and index < len(eligible) - 1:
            time.sleep(settings.symbol_throttle_seconds)
    return results, failures, alpha_signals, alpha_prices


def _analyze_symbols(
    eligible: list[str],
    *,
    use_case: AnalyzeMarketUseCase,
    settings: Settings,
    crypto_set: set[str],
    etf_set: set[str],
    regime_context: str,
    regime_multiplier: float,
    collect_alpha: bool,
    alpha_use_case: AlphaEnrichmentUseCase,
) -> tuple[list[SymbolResult], int, dict[str, AlphaSignals], dict[str, Decimal]]:
    """Dispatcher: wybiera ścieżkę równoległą lub sekwencyjną.
    Zwraca (wyniki, liczba_błędów, alpha_signals, alpha_prices)."""
    # Ścieżka równoległa: tylko gdy concurrency > 1 ORAZ contagion off (contagion
    # #5 zależy od kolejności przetwarzania, więc jest niekompatybilny z out-of-order).
    if settings.symbol_concurrency > 1 and not settings.contagion_enabled:
        return _analyze_symbols_parallel(
            eligible,
            max_workers=settings.symbol_concurrency,
            use_case=use_case,
            crypto_set=crypto_set,
            etf_set=etf_set,
            regime_context=regime_context,
            regime_multiplier=regime_multiplier,
            collect_alpha=collect_alpha,
            alpha_use_case=alpha_use_case,
            earnings_gate_enabled=settings.earnings_calendar_enabled,
        )

    if settings.symbol_concurrency > 1 and settings.contagion_enabled:
        logger.warning(
            "symbol_concurrency=%d zignorowane — contagion_enabled wymaga "
            "przetwarzania sekwencyjnego (peer_context zależy od kolejności).",
            settings.symbol_concurrency,
        )

    return _analyze_symbols_sequential(
        eligible,
        use_case=use_case,
        settings=settings,
        crypto_set=crypto_set,
        etf_set=etf_set,
        regime_context=regime_context,
        regime_multiplier=regime_multiplier,
        collect_alpha=collect_alpha,
        alpha_use_case=alpha_use_case,
    )


def _build_portfolio_radar(
    settings: Settings, repository: RepositoryPort, all_symbols: list[str]
) -> PortfolioRiskReport | None:
    """Q4 — korelacja/koncentracja watchlisty (+ R1/R2 VaR/CVaR/stress). Darmowe.
    Błąd → None (sekcja pomijana w raporcie)."""
    try:
        report = PortfolioRiskUseCase(
            repository_port=repository,
            # R1/R2: VaR/CVaR + stress-test (z tych samych darmowych zwrotów).
            var_enabled=settings.portfolio_var_enabled,
            stress_enabled=settings.stress_test_enabled,
            var_confidence=settings.var_confidence,
            scenarios=(
                DEFAULT_STRESS_SCENARIOS if settings.stress_test_enabled else ()
            ),
        ).run(all_symbols)
        logger.info(
            "Portfolio Radar — clusters=%d watchlist=%d",
            len(report.clusters),
            report.watchlist_size,
        )
        return report
    except Exception:
        logger.exception("Portfolio Radar failed — pomijam sekcję w raporcie")
        return None


def _build_signal_calibration(
    settings: Settings, repository: RepositoryPort
) -> list[CalibrationBucket] | None:
    """#9 — kubełki kalibracji pod RANKING sygnałów (nie pod sekcję Track Record).

    Świadomie osobne od `_build_track_record`: `calibration_curve_enabled` rysuje
    sekcję, a `calibrated_confidence_enabled` zmienia ranking. Sprzęgnięcie ich
    znaczyłoby, że włączenie rankingu po cichu dorysowuje sekcję w mailu.
    Best-effort: błąd odczytu → None → surowa pewność, cykl leci dalej.
    """
    if not settings.calibrated_confidence_enabled:
        return None
    try:
        rows = repository.get_resolved_for_calibration(settings.track_record_days)
    except Exception:
        logger.exception("Failed to fetch calibration samples for signal ranking")
        return None
    samples = [
        (float(row["confidence_score"]), bool(row["is_trend_correct"]))
        for row in rows
        if row.get("confidence_score") is not None
        and row.get("is_trend_correct") is not None
    ]
    return reliability_curve(samples) if samples else None


def _build_track_record(
    settings: Settings, repository: RepositoryPort
) -> tuple[EquityCurve | None, list[CalibrationBucket] | None, LessonsReport | None]:
    """T1/T2/T4 — krzywa kapitału, kalibracja, lessons (render-only, zero
    płatnych). Każda podsekcja flag-gated; wszystkie czytają zamknięte predykcje
    z okna track_record_days."""
    equity_curve: EquityCurve | None = None
    calibration_buckets: list[CalibrationBucket] | None = None
    lessons: LessonsReport | None = None
    if settings.equity_curve_enabled or settings.lessons_enabled:
        try:
            tr_resolved = parse_resolved_predictions(
                repository.get_resolved_predictions(settings.track_record_days)
            )
        except Exception:
            logger.exception("Failed to fetch resolved predictions for track record")
            tr_resolved = []
        if settings.equity_curve_enabled:
            trades = [
                (r.predicted_trend, float(r.actual_change_pct))
                for r in tr_resolved
                if r.actual_change_pct is not None
            ]
            equity_curve = build_equity_curve(trades)
        if settings.lessons_enabled:
            lessons = build_lessons(tr_resolved)
    if settings.calibration_curve_enabled:
        try:
            cal_rows = repository.get_resolved_for_calibration(
                settings.track_record_days
            )
            samples = [
                (float(row["confidence_score"]), bool(row["is_trend_correct"]))
                for row in cal_rows
                if row.get("confidence_score") is not None
                and row.get("is_trend_correct") is not None
            ]
            calibration_buckets = reliability_curve(samples) if samples else None
        except Exception:
            logger.exception("Failed to compute calibration curve")
    return equity_curve, calibration_buckets, lessons


def _fetch_report_context(
    repository: RepositoryPort, quota_monitor: QuotaMonitor, all_symbols: list[str]
) -> tuple[
    dict[str, Any] | None,
    list[ResolvedPrediction],
    list[QuotaAlert],
    list[InvestorHistory],
    list[PersonaTrackRecord],
]:
    """Best-effort dociągnięcie danych do raportu: accuracy, zamknięte predykcje,
    alerty kwot (persystencja bieżących + odczyt 24h), historia głosów rady
    i ranking wiarygodności person.
    Każdy odczyt osobno owinięty — błąd jednego nie blokuje pozostałych."""
    accuracy_stats: dict[str, Any] | None = None
    resolved: list[ResolvedPrediction] = []
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

    # Persystencja alertów z BIEŻĄCEGO cyklu (przed odczytem history, żeby też
    # tu trafiły) + odczyt z ostatnich 24h dla bannera.
    for alert in quota_monitor.alerts:
        try:
            repository.save_quota_alert(alert)
        except Exception:
            logger.exception("Failed to persist quota alert from %s", alert.source)
    try:
        recent_alerts = repository.get_recent_quota_alerts(hours=24)
    except Exception:
        logger.exception("Failed to fetch recent quota alerts — using cycle-only")
        recent_alerts = list(quota_monitor.alerts)

    # U5: historia głosów rady per inwestor (panel drill-down w raporcie).
    council_history: list[InvestorHistory] = []
    try:
        council_history = repository.get_council_vote_history(
            all_symbols, window_days=30
        )
    except Exception:
        logger.exception("Failed to fetch council vote history")

    # #3: leaderboard person. Surowe (hit_rate, votes) z RPC, ranking i próg
    # min_votes liczy domena. Brak migracji 018 / błąd RPC → pusty ranking
    # → sekcja sama się chowa, cykl leci dalej.
    persona_track_record: list[PersonaTrackRecord] = []
    try:
        persona_track_record = rank_personas(
            repository.get_persona_track_record(PERSONA_TRACK_RECORD_DAYS)
        )
    except Exception:
        logger.exception("Failed to fetch persona track record")
    return (
        accuracy_stats,
        resolved,
        recent_alerts,
        council_history,
        persona_track_record,
    )


def _dispatch_reports(
    *,
    results: list[SymbolResult],
    started_at: datetime,
    duration: float,
    failures: int,
    settings: Settings,
    repository: RepositoryPort,
    quota_monitor: QuotaMonitor,
    notifier: ReportNotifierPort,
    web_publisher: WebPublisherPort,
    subscribers: list[Subscriber],
    all_symbols: list[str],
    macro_risk_report: MacroRiskReport | None,
    portfolio_risk_report: PortfolioRiskReport | None,
    alpha_signals: dict[str, AlphaSignals],
    alpha_prices: dict[str, Decimal],
    yield_curve: YieldCurveSnapshot | None,
) -> None:
    """Faza wysyłki: dociągnięcie danych raportu, zbudowanie raportu i rozesłanie
    — pojedynczo albo per subskrybent (raport tnięty do jego watchlisty). Cała
    faza jest best-effort: błąd wysyłki nie psuje exit-code cyklu."""
    try:
        (
            accuracy_stats,
            resolved,
            recent_alerts,
            council_history,
            persona_track_record,
        ) = _fetch_report_context(repository, quota_monitor, all_symbols)

        # T1/T2/T4: Track Record (render-only, zero płatnych). Każda podsekcja
        # flag-gated; wszystkie czytają zamknięte predykcje z okna track_record_days.
        equity_curve, calibration_buckets, lessons = _build_track_record(
            settings, repository
        )
        # #9: kubełki pod ranking sygnałów (osobna flaga, osobny odczyt).
        signal_calibration_buckets = _build_signal_calibration(settings, repository)

        def _build(symbols_filter: frozenset[str] | None = None) -> tuple[str, str]:
            return build_html_report(
                results, started_at, duration,
                accuracy_stats=accuracy_stats,
                resolved_predictions=resolved,
                macro_risk_report=macro_risk_report,
                portfolio_risk_report=portfolio_risk_report,
                council_history=council_history,
                persona_track_record=persona_track_record,
                quota_alerts=recent_alerts,
                symbols_filter=symbols_filter,
                equity_curve=equity_curve,
                calibration_buckets=calibration_buckets,
                lessons=lessons,
                alpha_signals=alpha_signals,
                yield_curve=yield_curve,
                alpha_prices=alpha_prices,
                signal_calibration_buckets=signal_calibration_buckets,
            )

        html, text = _build()
        analyzed = sum(1 for r in results if r.status == "saved")
        # Subject prefix gdy są CRITICAL — wymusza zwrócenie uwagi w skrzynce.
        max_sev = quota_monitor.max_severity()
        prefix = "⚠️ [QUOTA] " if max_sev is QuotaSeverity.CRITICAL else ""
        subject = (
            f"{prefix}StockAgent — {analyzed} predykcji, {failures} błędów "
            f"({started_at.strftime('%Y-%m-%d %H:%M UTC')})"
        )

        # U3: publikacja statycznego digestu web (read-only dashboard).
        try:
            url = web_publisher.publish(html)
            if url:
                logger.info("Web digest opublikowany: %s", url)
        except Exception:
            logger.exception("Web digest publish failed")

        # U2: jeśli są subskrybenci, każdy dostaje raport TNIĘTY do swojej
        # watchlisty (analiza poszła RAZ nad unią — to tylko re-slicing). Bez
        # subskrybentów → dotychczasowa pojedyncza wysyłka.
        if subscribers and settings.resend_api_key:
            all_syms = frozenset(all_symbols)
            for sub in subscribers:
                sub_html, sub_text = _build(symbols_for(sub, all_syms))
                try:
                    ResendNotifier(
                        api_key=settings.resend_api_key,
                        sender=settings.digest_from_email,
                        recipient=sub.email,
                        quota_monitor=quota_monitor,
                    ).send_report(
                        subject=subject, html_body=sub_html, plain_text=sub_text
                    )
                except Exception:
                    logger.exception("Failed to send digest to %s", sub.email)
        else:
            notifier.send_report(subject=subject, html_body=html, plain_text=text)
    except Exception:
        logger.exception("Failed to send report")


def main(settings: Settings | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    settings = settings or Settings.from_env()
    # Real-time alerty (U4): kanał alertów budujemy BEZ quota_monitora, żeby
    # CRITICAL wyemitowany podczas pushu alertu nie wracał rekurencyjnie do
    # record() → push → record(). Hook montujemy tylko gdy włączony i jest dokąd.
    alert_channels = build_push_channels(settings, quota_monitor=None)
    alert_notifier = (
        DelegatingAlertNotifier(CompositeNotifier(alert_channels))
        if settings.realtime_alerts_enabled and alert_channels
        else None
    )
    quota_monitor = QuotaMonitor(alert_notifier=alert_notifier)
    repository = build_repository(settings)
    use_case = build_use_case(
        settings, repository=repository, quota_monitor=quota_monitor
    )
    notifier = build_notifier(settings, quota_monitor=quota_monitor)
    web_publisher = build_web_publisher(settings)  # U3
    # U2: lista subskrybentów (każdy z własną watchlistą). Brak / błąd → [].
    subscribers = []
    if settings.subscriptions_enabled:
        try:
            subscribers = SupabaseSubscriberRepository(
                url=settings.supabase_url, key=settings.supabase_key
            ).list_subscribers()
        except Exception:
            logger.exception("Failed to list subscribers — fallback do pojedynczej wysyłki")

    started_at = datetime.now(UTC)
    started_perf = time.perf_counter()
    logger.info(
        "Fast Loop start — symbols=%s crypto=%s",
        settings.symbols,
        settings.crypto_symbols,
    )

    # Dane: enrichment alpha (per-symbol) + krzywa rentowności (raz na cykl).
    # Wszystko domyślnie Null/None → zero I/O, póki flagi off.
    alpha_use_case = build_alpha_enrichment(settings)
    collect_alpha = alpha_per_symbol_enabled(settings)
    yield_curve = build_macro_rates_port(settings).fetch_yield_curve()
    results: list[SymbolResult] = []
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

    # ---- Risk Watch (PRZED pętlą — zasila też reżim rynku #7) ----
    macro_risk_report = _run_risk_watch(settings, repository)

    # #7: reżim rynku z overall_alert (zacieśnia próg + label do promptu). Off → neutral.
    regime_context, regime_multiplier = _resolve_regime(
        settings, macro_risk_report, yield_curve
    )

    loop_results, failures, alpha_signals, alpha_prices = _analyze_symbols(
        eligible,
        use_case=use_case,
        settings=settings,
        crypto_set=crypto_set,
        etf_set=etf_set,
        regime_context=regime_context,
        regime_multiplier=regime_multiplier,
        collect_alpha=collect_alpha,
        alpha_use_case=alpha_use_case,
    )
    results.extend(loop_results)

    duration = time.perf_counter() - started_perf
    # #34: mianownik MUSI być liczbą faktycznie przetworzonych symboli
    # (eligible = symbols + crypto − unsupported), a nie len(settings.symbols).
    # Pętla iteruje `eligible`, więc failures odnoszą się do niego — inaczej
    # przy obecności krypto/unsupported ratio się rozjeżdża (failures mogą nawet
    # przekroczyć mianownik). Spójne z logiką exit-code (też używa len(eligible)).
    logger.info("Fast Loop done — failures=%d/%d", failures, len(eligible))

    # ---- Portfolio Radar (Q4 — korelacja/koncentracja watchlisty, darmowe) ----
    portfolio_risk_report = _build_portfolio_radar(settings, repository, all_symbols)

    # ---- Wysyłka raportu ----
    _dispatch_reports(
        results=results,
        started_at=started_at,
        duration=duration,
        failures=failures,
        settings=settings,
        repository=repository,
        quota_monitor=quota_monitor,
        notifier=notifier,
        web_publisher=web_publisher,
        subscribers=subscribers,
        all_symbols=all_symbols,
        macro_risk_report=macro_risk_report,
        portfolio_risk_report=portfolio_risk_report,
        alpha_signals=alpha_signals,
        alpha_prices=alpha_prices,
        yield_curve=yield_curve,
    )

    # Exit code 1 tylko gdy wszystkie symbole padły (catastrophic failure).
    # Pojedyncze błędy per-symbol (np. ticker niewspierany przez Finnhub free)
    # są raportowane w mailu i nie powinny psuć całego cyklu w GHA.
    total = len(eligible)
    if total > 0 and failures == total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
