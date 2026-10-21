"""Fast Loop entry point — uruchamiany przez GHA co 12h (lub lokalnie).

Wczytuje konfigurację z `.env` lub zmiennych środowiskowych,
instancjuje wszystkie adaptery (DI Container), iteruje po liście symboli
i wywołuje AnalyzeMarketUseCase dla każdego z nich.

Po cyklu — jeśli włączone — wysyła raport mailem przez Resend.com.

Użycie:
    uv run python main_agent.py
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, NamedTuple

from src.application.agent_graph import AgentGraphDeps, AgentState
from src.application.alpha_signals import AlphaSignals
from src.application.cycle_telemetry import CycleTelemetry
from src.application.data_coverage import build_data_coverage
from src.application.metering import BudgetedLLMPort, MeteredLLMPort, PaidCallAllowance
from src.application.move_interval import (
    CALIBRATION_LOOKBACK_DAYS,
    calibration_windows,
    interval_coverage,
)
from src.application.portfolio_fx import collect_pln_rates, effective_pln_rates, nbp_today
from src.application.ports import (
    AdvisoryCouncilPort,
    AnalystConsensusPort,
    AttestationPublisherPort,
    CycleRun,
    EarningsCalendarPort,
    EmbeddingPort,
    FxRatePort,
    HeartbeatPort,
    InsiderFlowPort,
    LLMPort,
    MacroRatesPort,
    MarketDataPort,
    NewsPort,
    OptionsFlowPort,
    PortfolioPositionsPort,
    ReportNotifierPort,
    RepositoryPort,
    SentimentPort,
    SocialVelocityPort,
    ToolUseLLMPort,
    UsageSink,
    WebPublisherPort,
)
from src.application.quota_monitor import QuotaMonitor
from src.application.report_builder import (
    ReportContext,
    SymbolResult,
    build_cycle_lead,
    build_html_report,
    parse_resolved_predictions,
    to_symbol_result,
)
from src.application.report_council_history import InvestorHistory
from src.application.report_evidence import (
    BENCHMARK_SYMBOL,
    PaperCosts,
    benchmark_comparison,
    hit_rate_evidence,
    portfolio_curve,
)
from src.application.report_formatting import (
    DEFAULT_DISPLAY_TIMEZONE,
    build_sector_map,
    format_local,
)
from src.application.report_lessons import LessonsReport, build_lessons
from src.application.report_messenger import build_messenger_digest
from src.application.report_model_scorecard import (
    ModelScorecardView,
    build_scorecard_view,
)
from src.application.report_models import ResolvedPrediction
from src.application.report_onboarding import onboarding_subject
from src.application.report_ops import render_cycle_summary_markdown
from src.application.report_public_scorecard import render_public_scorecard_html
from src.application.report_regime import RegimeBanner
from src.application.report_symbol_record import SymbolTrackRecordView, symbol_track_record_view
from src.application.report_track_record import TrackRecordStats
from src.application.run_context import RunContext
from src.application.signal_calibration import (
    directional_calibration_curve,
    split_calibration_samples,
)
from src.application.tools import build_research_tools
from src.application.usage_ledger import UsageLedger
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase, PaidPhase
from src.application.use_cases.enrich_alpha import AlphaEnrichmentUseCase
from src.application.use_cases.monitor_macro_risk import (
    MacroRiskReport,
    MonitorMacroRiskUseCase,
)
from src.application.use_cases.portfolio_risk import (
    PortfolioRiskReport,
    PortfolioRiskUseCase,
    PositionRisk,
)
from src.config import Settings, load_evaluation_protocol
from src.domain.alpha_fusion import AlphaFusionScore, fuse_alpha_signals
from src.domain.asset import Asset
from src.domain.benchmark import BenchmarkComparison
from src.domain.calibration_curve import (
    CalibrationBucket,
    confidence_reorders_ranking,
    discrimination_evidence,
    probability_scores,
    reliability_curve,
)
from src.domain.conformal import CalibrationPoint
from src.domain.consensus_shift import ConsensusShift, detect_shift
from src.domain.coverage import CoverageReport
from src.domain.cycle_maturity import CycleMaturity, SkipReason, classify_cycle
from src.domain.digest_lead import LeadItem, lead_headline
from src.domain.earnings import EarningsEvent, earnings_threshold_multiplier
from src.domain.equity_curve import EquityCurve
from src.domain.finops import CycleCost, estimate_cycle_cost, rank_by_magnitude
from src.domain.hedge import equity_hedge_symbols
from src.domain.macro_rates import YieldCurveSnapshot, yield_curve_alert_level
from src.domain.macro_risk import MacroAlertLevel
from src.domain.market_calendar import closure_reason, to_nyse_time
from src.domain.model_scorecard import MODEL_SCORECARD_WINDOW_DAYS
from src.domain.number_format import format_money_pl, format_number_pl
from src.domain.peer_group import build_peer_context
from src.domain.persona_track_record import (
    PersonaTrackRecord,
    assess_against_track_record,
    rank_personas,
    sort_personas,
)
from src.domain.portfolio import PlnRates, Portfolio
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.recipe import PredictionRecipe, universe_version
from src.domain.regime import MarketRegime, RegimeDetector, apply_hysteresis
from src.domain.scenarios import Scenario
from src.domain.subscriber import Subscriber, is_owner, masked_email, symbols_for
from src.domain.value_objects import AssetType, Threshold
from src.infrastructure.adapters._finnhub_common import FINNHUB_REQUESTS_PER_MINUTE
from src.infrastructure.adapters._http import RateLimiter, redact_secrets
from src.infrastructure.adapters.advisory_council import (
    COUNCIL_MAX_PERSONAS,
    LLMAdvisoryCouncil,
)
from src.infrastructure.adapters.alert_notifier import DelegatingAlertNotifier
from src.infrastructure.adapters.alpha_vantage_adapters import (
    AlphaVantageNewsAdapter,
    AlphaVantageSentimentAdapter,
    PhaseTwoOnlyFeed,
)
from src.infrastructure.adapters.alpha_vantage_client import NEWS_WINDOW, AlphaVantageClient
from src.infrastructure.adapters.cached_fundamentals import (
    CachedFundamentalsAdapter,
    NullFundamentalsAdapter,
)
from src.infrastructure.adapters.coingecko import CoinGeckoAdapter
from src.infrastructure.adapters.composite_notifier import CompositeNotifier
from src.infrastructure.adapters.finnhub_api import FinnhubAdapter
from src.infrastructure.adapters.http_heartbeat import HttpHeartbeat, NullHeartbeat
from src.infrastructure.adapters.local_stubs import (
    LocalStubNewsAdapter,
    LocalStubSentimentAdapter,
)
from src.infrastructure.adapters.nbp_client import NbpClient, NbpFxRateClient
from src.infrastructure.adapters.resend_notifier import NullNotifier, ResendNotifier
from src.infrastructure.adapters.routing_market_data import RoutingMarketDataPort
from src.infrastructure.adapters.subscriber_repo import SupabaseSubscriberRepository
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from src.infrastructure.adapters.write_guard import (
    ReadOnlyAttestationPublisher,
    ReadOnlyRepository,
)
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter
from src.infrastructure.llm.local_stub import LocalStubLLMAdapter
from src.infrastructure.logging_redaction import configure_logging
from src.infrastructure.persona_loader import load_council_personas, personas_sha256
from src.infrastructure.recipe_fingerprints import ml_model_version, prompt_templates_sha256

# R2: domyślne scenariusze stress-testu (szok rynku jako ułamek). Szok stosowany
# do portfela równoważonego przez β-to-portfel każdego symbolu (domena liczy resztę).
DEFAULT_STRESS_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="Korekta rynku -10%", market_shock=-0.10),
    Scenario(name="Krach -20%", market_shock=-0.20),
    Scenario(name="Rajd +10%", market_shock=0.10),
)

ACCURACY_STATS_DAYS = 30
RESOLVED_PREDICTIONS_HOURS = 24
# NIEZ-01: banner alertów kwot sięga tydzień + 3 h wstecz, żeby niedzielny CRITICAL
# ze Slow Loopa (cron raz w tygodniu) był w najbliższym mailu, a nie znikał po 24 h.
QUOTA_ALERT_LOOKBACK_HOURS = 7 * 24 + 3
# Okno track recordu person (#3) — 90 dni, tak jak domyślne okno RPC z migracji
# 018. Dłuższe niż okno accuracy: hit-rate per persona potrzebuje próbki.
PERSONA_TRACK_RECORD_DAYS = 90

logger = logging.getLogger(__name__)

# DAT-12: limit Finnhub jest per klucz, więc limiter jest JEDEN na proces — każdy
# adapter Finnhub (/quote, analitycy, opcje) dostaje ten sam obiekt.
FINNHUB_RATE_LIMITER = RateLimiter(FINNHUB_REQUESTS_PER_MINUTE)


def _ignored_result(symbol: str) -> SymbolResult:
    """Tworzy SymbolResult dla symbolu pre-filtrowanego z SYMBOLS_UNSUPPORTED_PRICE.

    Ląduje w mailu jako "ignored" (nie błąd) — żeby ograniczenia free tier
    Finnhub nie zaśmiecały sekcji błędów cyklu po cyklu.
    """
    return SymbolResult(
        symbol=symbol,
        status="ignored",
        # #4: trzecia kategoria pominięcia — ticker nieobsługiwany cenowo.
        # Wykluczona z licznika cold-startu (nigdy nie dostanie punktu odniesienia).
        skip_reason=SkipReason.UNSUPPORTED_PRICE,
        error_message="unsupported by current price adapter (Finnhub free)",
    )


def _build_single_llm(
    settings: Settings,
    provider: str,
    model: str,
    quota_monitor: QuotaMonitor | None,
    usage_sink: UsageSink | None = None,
) -> LLMPort:
    """Buduje pojedynczy adapter LLM dla danego providera/modelu."""
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("router_frontier_provider=anthropic but ANTHROPIC_API_KEY unset.")
        from src.infrastructure.llm.anthropic_client import AnthropicAdapter

        return AnthropicAdapter(
            api_key=settings.anthropic_api_key,
            model=model,
            quota_monitor=quota_monitor,
            usage_sink=usage_sink,
        )
    from src.infrastructure.llm.openai_client import OpenAIAdapter

    return OpenAIAdapter(
        api_key=settings.require_openai_api_key(),
        model=model,
        quota_monitor=quota_monitor,
        usage_sink=usage_sink,
    )


def build_llm_adapter(
    settings: Settings,
    quota_monitor: QuotaMonitor | None = None,
    usage_sink: UsageSink | None = None,
    *,
    telemetry: CycleTelemetry | None = None,
    fallback_allowance: PaidCallAllowance | None = None,
) -> LLMPort:
    """Factory LLM — wybiera providera na podstawie konfiguracji.

    #2: gdy `model_routing_enabled`, zwraca `CascadingLLMAdapter` — tani model
    domyślnie, eskalacja do frontier tylko gdy `confidence < routing_floor`.
    FIN-13: `usage_sink` dostaje tokeny każdego requestu (oba modele kaskady).
    NIEZ-44: z `llm_fallback_provider` całość idzie za `FailoverLLMAdapter`
    (`_with_llm_fallback`); request do zapasowego liczy `telemetry` pod `llm`,
    a `fallback_allowance` (Slow Loop) odrzuca go ponad przydział joba.
    """
    primary = _build_primary_llm(settings, quota_monitor, usage_sink)
    return _with_llm_fallback(
        settings, primary, quota_monitor, usage_sink, telemetry, fallback_allowance
    )


def _with_llm_fallback(
    settings: Settings,
    primary: LLMPort,
    quota_monitor: QuotaMonitor | None,
    usage_sink: UsageSink | None,
    telemetry: CycleTelemetry | None,
    allowance: PaidCallAllowance | None = None,
) -> LLMPort:
    """NIEZ-44: zapasowy dostawca za głównym — tylko gdy skonfigurowany i ma klucz.

    Zapasowy dostaje ten sam monitor kwot i ten sam `usage_sink` (tokeny z jego
    dostawcą i modelem), a `MeteredLLMPort` liczy jego request w telemetrii cyklu
    pod `llm` — więc widzi go sufit FIN-12 i sekcja kosztu. Request do głównego
    liczy jak dotąd graf (Fast Loop) albo meter joba (Slow Loop). `allowance`
    (Slow Loop, FIN-30): request do zapasowego zużywa przydział joba, a ponad nim
    `BudgetedLLMPort` odrzuca go przed dostawcą i failover nie startuje."""
    provider = settings.llm_fallback_provider
    if provider is None:
        return primary
    key_name, key = _fallback_api_key(settings)
    if not key:
        logger.warning(
            "llm_fallback_provider=%s, ale %s nie jest ustawiony — LLM bez failovera",
            provider,
            key_name,
        )
        return primary
    from src.infrastructure.llm.failover import FailoverLLMAdapter

    model = settings.llm_fallback_model or _provider_default_model(provider)
    secondary = _build_single_llm(settings, provider, model, quota_monitor, usage_sink)
    if telemetry is not None:
        secondary = MeteredLLMPort(secondary, telemetry, "llm")
    if allowance is not None:
        secondary = BudgetedLLMPort(secondary, allowance=allowance)
    return FailoverLLMAdapter(primary, secondary, provider=provider, model=model)


def _fallback_api_key(settings: Settings) -> tuple[str, str | None]:
    """NIEZ-44: (nazwa zmiennej, wartość) klucza dostawcy zapasowego."""
    if settings.llm_fallback_provider == "anthropic":
        return "ANTHROPIC_API_KEY", settings.anthropic_api_key
    return "OPENAI_API_KEY", settings.openai_api_key


def llm_fallback_available(settings: Settings) -> bool:
    """NIEZ-44: czy `build_llm_adapter` zbuduje zapasowego dostawcę — ustawiony
    `llm_fallback_provider` i jego klucz. Slow Loop prosi wtedy budżet FIN-30
    o +1 wywołanie na job."""
    return settings.llm_fallback_provider is not None and bool(_fallback_api_key(settings)[1])


def _build_primary_llm(
    settings: Settings,
    quota_monitor: QuotaMonitor | None,
    usage_sink: UsageSink | None,
) -> LLMPort:
    """Główny LLM analizy: kaskada modeli (#2) albo adapter `llm_provider`."""
    if settings.model_routing_enabled:
        from src.infrastructure.llm.cascading import CascadingLLMAdapter
        from src.infrastructure.llm.openai_client import OpenAIAdapter

        cheap = OpenAIAdapter(
            api_key=settings.require_openai_api_key(),
            model=settings.router_cheap_model,
            quota_monitor=quota_monitor,
            usage_sink=usage_sink,
        )
        frontier = _build_single_llm(
            settings,
            settings.router_frontier_provider,
            settings.router_frontier_model,
            quota_monitor,
            usage_sink,
        )
        return CascadingLLMAdapter(
            cheap=cheap,
            frontier=frontier,
            routing_floor=settings.routing_floor,
            max_escalations=settings.routing_max_escalations,
            quota_monitor=quota_monitor,
        )

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Add it to .env (local) or the repository secrets (CI)."
            )
        from src.infrastructure.llm.anthropic_client import AnthropicAdapter

        return AnthropicAdapter(
            api_key=settings.anthropic_api_key,
            quota_monitor=quota_monitor,
            usage_sink=usage_sink,
        )

    from src.infrastructure.llm.openai_client import OpenAIAdapter

    return OpenAIAdapter(
        api_key=settings.require_openai_api_key(),
        quota_monitor=quota_monitor,
        usage_sink=usage_sink,
    )


def build_tool_use_adapter(
    settings: Settings,
    quota_monitor: QuotaMonitor | None = None,
    usage_sink: UsageSink | None = None,
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
            secret_values=settings.secret_values(),
            usage_sink=usage_sink,
        )
    from src.infrastructure.llm.tool_calling_openai import ToolCallingOpenAIAdapter

    return ToolCallingOpenAIAdapter(
        api_key=settings.require_openai_api_key(),
        max_rounds=settings.tool_use_max_rounds,
        quota_monitor=quota_monitor,
        secret_values=settings.secret_values(),
        usage_sink=usage_sink,
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
    settings: Settings,
    quota_monitor: QuotaMonitor | None = None,
    *,
    include_push: bool = True,
) -> ReportNotifierPort:
    """Factory notifier — Resend + kanały push (Telegram/Slack) jako jeden
    `ReportNotifierPort` (CompositeNotifier). Każdy skonfigurowany kanał dostaje
    kopię dziennego raportu; brak jakiegokolwiek kanału → Null.

    Gdy wpada w Null mimo `notifications_enabled=true`, loguje WARNING z nazwą
    brakującego sekretu — inaczej "Notifications disabled" nie mówi, czego
    brakuje (typowo: DIGEST_TO_EMAIL / RESEND_API_KEY nie ustawione jako
    GitHub repo *Secret*, nie Variable).

    `include_push=False` (Fast Loop, decyzja 2026-10-19) zostawia sam e-mail:
    stary push dostaje wtedy osobny tekst bez sekcji portfela
    (`build_legacy_push_notifier`), bo kwoty P/L nie mają trafiać do komunikatorów.
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

    # #5: gdy messenger digest jest włączony, push NIE wchodzi do tego notifiera
    # — dostaje osobno 5-linijkowy skrót (`_dispatch_push_digest`). Pełny
    # plain-text raportu przekracza limit 4096 znaków Telegrama → 400 i cichy
    # zgon pusha. Z flagą OFF zachowujemy dotychczasowe (wadliwe) sklejenie.
    if include_push and not settings.messenger_digest_enabled:
        channels.extend(build_push_channels(settings, quota_monitor=quota_monitor))

    if not channels:
        return NullNotifier()
    if len(channels) == 1:
        return channels[0]
    return CompositeNotifier(channels)


def build_legacy_push_notifier(
    settings: Settings, quota_monitor: QuotaMonitor | None = None
) -> ReportNotifierPort | None:
    """Stary wariant pushu (`messenger_digest_enabled=false`): pełny tekst raportu
    do Telegram/Slack, wysyłany OSOBNO od maila, żeby tekst mógł pominąć sekcję
    portfela. None, gdy działa skrót #5 albo nie ma sekretów żadnego kanału."""
    if settings.messenger_digest_enabled:
        return None
    channels = build_push_channels(settings, quota_monitor=quota_monitor)
    if not channels:
        return None
    if len(channels) == 1:
        return channels[0]
    return CompositeNotifier(channels)


def build_push_notifier(
    settings: Settings, quota_monitor: QuotaMonitor | None = None
) -> ReportNotifierPort:
    """#5 — kanały push jako JEDEN port (Telegram + Slack), z izolacją wyjątków
    per kanał w `CompositeNotifier`. Brak sekretów → Null (kanał nie powstaje)."""
    channels = build_push_channels(settings, quota_monitor=quota_monitor)
    if not channels:
        return NullNotifier()
    if len(channels) == 1:
        return channels[0]
    return CompositeNotifier(channels)


def _dispatch_push_digest(
    *,
    push_notifier: ReportNotifierPort,
    results: list[SymbolResult],
    started_at: datetime,
    resolved: list[ResolvedPrediction],
    subject: str,
    email_delivered: bool = True,
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE,
    resolved_unavailable: bool = False,
) -> None:
    """#5 — wysyła skrót push RAZ, niezależnie od fan-outu per subskrybent.

    Gałąź U2 (subskrybenci) w ogóle pomijała `notifier.send_report`, więc push
    nie szedł, gdy ktokolwiek był zapisany. Skrót liczymy nad PEŁNYMI wynikami
    cyklu — push to jeden kanał, nie personalizowana watchlista.
    Best-effort: błąd kanału nie może wywalić cyklu.
    """
    try:
        digest = build_messenger_digest(
            results,
            started_at,
            resolved,
            email_delivered=email_delivered,
            display_timezone=display_timezone,
            resolved_unavailable=resolved_unavailable,
        )
        push_notifier.send_report(
            subject=subject, html_body=digest, plain_text=digest
        )
    except Exception:
        logger.exception("Failed to send messenger digest")


def build_positions_port(settings: Settings) -> PortfolioPositionsPort:
    """#15 — realne pozycje użytkownika. Null gdy flaga off → sekcja się chowa,
    a warstwa ryzyka zostaje przy dotychczasowych równych wagach."""
    from src.infrastructure.adapters.positions_repo import (
        NullPositionsAdapter,
        SupabasePositionsAdapter,
    )

    if not settings.portfolio_positions_enabled:
        return NullPositionsAdapter()
    # PORT-18: waluta i prawdziwa data zakupu dopiero po migracji 035 (flaga).
    return SupabasePositionsAdapter(
        url=settings.supabase_url,
        key=settings.supabase_key,
        currency_columns=settings.positions_currency_enabled,
    )


def _build_user_portfolio(settings: Settings) -> Portfolio | None:
    """#15 — agregat portfela. Błąd / brak pozycji → None → sekcja się chowa
    (a nie: fałszywy P/L z niekompletnych danych)."""
    if not settings.portfolio_positions_enabled:
        return None
    port = build_positions_port(settings)
    try:
        positions = port.get_positions()
        as_of = port.get_as_of()
    except Exception:
        logger.exception("Failed to fetch portfolio positions")
        return None
    if not positions or as_of is None:
        return None
    if not settings.positions_currency_enabled:
        # PORT-11: bez migracji 035 adapter podaje `as_of` jako datę — to data
        # aktualizacji tabeli, nie zakupu. Nie udajemy kursu z dnia zakupu.
        positions = [dataclasses.replace(p, purchase_date=None) for p in positions]
    return Portfolio(positions=tuple(positions), as_of=as_of)


def build_fx_port(settings: Settings) -> FxRatePort:
    """PORT-11 — kurs NBP z dnia (darmowe). Cache `fx_rates` tylko za flagą
    `fx_rates_cache_enabled` (migracja 036); bez niej żadnego zapytania do tabeli.
    BEZ-08: bez `db_writes_enabled` cache jest tylko czytany — kurs z NBP nie
    trafia do wspólnej tabeli."""
    nbp = NbpFxRateClient()
    if not settings.fx_rates_cache_enabled:
        return nbp
    from src.infrastructure.adapters.fx_rates_cache import SupabaseCachedFxRates

    return SupabaseCachedFxRates(
        url=settings.supabase_url,
        key=settings.supabase_key,
        delegate=nbp,
        today=lambda: nbp_today(datetime.now(UTC)),
        write_through=settings.db_writes_enabled,
    )


def _collect_portfolio_fx(
    settings: Settings, portfolio: Portfolio | None, now: datetime
) -> PlnRates | None:
    """PORT-11 — kursy NBP do „P/L zł”: bieżące i z dni zakupu pozycji.

    Darmowe, w fazie raportu (poza grafem i bramką volatility), za `nbp_enabled`.
    Błąd → None: sekcja bierze wtedy same kursy bieżące z Risk Watch i mówi
    wprost, że kursu z dnia zakupu nie ma."""
    if portfolio is None or not settings.nbp_enabled:
        return None
    try:
        return collect_pln_rates(portfolio, build_fx_port(settings), today=nbp_today(now))
    except Exception:
        logger.exception("Failed to collect NBP rates for the portfolio")
        return None


def _quote_held_positions(
    settings: Settings, portfolio: Portfolio | None, watchlist: list[str]
) -> dict[str, Decimal]:
    """PORT-07 — darmowa cena pozycji spoza `watchlist` (symboli wycenionych
    w cyklu; w dzień zamknięcia NYSE samo krypto, FIN-34): SAM `/quote` przez
    `build_market_port` (Finnhub / CoinGecko).

    Bez `save_price_snapshot` (należy wyłącznie do Fast Loopa i bramki), bez
    grafu, bez LLM, Alpha Vantage i embeddingów — dopisanie symbolu do `symbols`
    uruchamiałoby płatne wywołania przy każdym ruchu ≥ progu. Symbole z
    `symbols_unsupported_price` pomijamy (darmowe źródło i tak ich nie wyceni).
    Zapytania idą sekwencyjnie, żeby nie dokładać równoległego ruchu do limitu
    Finnhub (60/min). Błąd symbolu → brak ceny; sekcja portfela pokaże go jako
    niewyceniony (PORT-03)."""
    if portfolio is None:
        return {}
    skipped = set(watchlist) | set(settings.symbols_unsupported_price)
    held = sorted({position.symbol for position in portfolio.positions} - skipped)
    if not held:
        return {}
    market = build_market_port(settings)
    prices: dict[str, Decimal] = {}
    for symbol in held:
        try:
            prices[symbol] = market.get_current_price(symbol).amount
        except Exception:
            logger.exception("Quote for held position %s failed — left unpriced", symbol)
    return prices


def _portfolio_clusters(settings: Settings) -> dict[str, str]:
    """#15 — mapa symbol → grupa z `peer_groups`, czyli sektor akcji z `sectors`
    w config.toml (PORT-05) — sektor, nie policzony klaster korelacji. Grup nie
    liczymy tutaj; domena tylko sumuje ekspozycję."""
    clusters: dict[str, str] = {}
    for cluster, symbols in settings.peer_groups.items():
        for symbol in symbols:
            clusters[symbol] = cluster
    return clusters


def build_attestation_publisher(settings: Settings) -> AttestationPublisherPort:
    """#16 — publisher commit-reveal. Null gdy `attestation_enabled=false`,
    tylko do odczytu bez `db_writes_enabled` (BEZ-08)."""
    from src.infrastructure.adapters.attestation_publisher import (
        FileAttestationPublisher,
        NullAttestationPublisher,
    )

    if not settings.attestation_enabled:
        return NullAttestationPublisher()
    publisher = FileAttestationPublisher(
        commitment_path=settings.attestation_commitments_path,
        reveal_path=settings.attestation_reveals_path,
    )
    if settings.db_writes_enabled:
        return publisher
    # BEZ-08: predykcja z runu tylko do odczytu ma id `dry-run-…` i nie istnieje
    # w bazie — jej commitment byłby fałszywym wpisem w publicznym logu.
    return ReadOnlyAttestationPublisher(publisher)


def build_options_port(
    settings: Settings, quota_monitor: QuotaMonitor | None = None
) -> OptionsFlowPort | None:
    """#18 — port opcji dla grafu (fetch IV w `_predict_node`, za bramką
    volatility). None gdy `options_flow_enabled=false` → sygnał nie powstaje."""
    if not settings.options_flow_enabled:
        return None
    from src.infrastructure.adapters.finnhub_options import FinnhubOptionsAdapter

    return FinnhubOptionsAdapter(
        api_key=settings.require_finnhub_api_key(),
        quota_monitor=quota_monitor,
        rate_limiter=FINNHUB_RATE_LIMITER,
    )


def build_market_port(settings: Settings) -> MarketDataPort:
    """Darmowe źródła ceny: krypto → CoinGecko, reszta → Finnhub.
    Wydzielone z `build_use_case`, bo `main_watch` (#11) potrzebuje samej ceny."""
    return RoutingMarketDataPort(
        equity=FinnhubAdapter(
            api_key=settings.require_finnhub_api_key(), rate_limiter=FINNHUB_RATE_LIMITER
        ),
        crypto=CoinGeckoAdapter(),
        crypto_symbols=settings.crypto_symbols,
    )


def build_public_scorecard_publisher(settings: Settings) -> WebPublisherPort:
    """#10 — DRUGA instancja `FileWebPublisher`, z własną ścieżką.

    Nie wolno reużyć instancji prywatnego digestu (`build_web_publisher`):
    obie piszą do pliku, więc jedna nadpisałaby drugą — publiczny scorecard
    zastąpiłby prywatny raport albo odwrotnie.
    """
    from src.infrastructure.adapters.web_publisher import (
        FileWebPublisher,
        NullWebPublisher,
    )

    if not settings.public_scorecard_enabled:
        return NullWebPublisher()
    return FileWebPublisher(output_path=settings.public_scorecard_path)


def _paper_costs(settings: Settings) -> PaperCosts:
    """WIA-50: założenia kosztowe paper-PnL z config.toml (akcje/ETF i krypto)."""
    return PaperCosts(
        cost_bps=settings.paper_cost_bps,
        crypto_cost_bps=settings.crypto_paper_cost_bps,
        crypto_symbols=frozenset(settings.crypto_symbols),
    )


def _benchmark_comparison(
    settings: Settings,
    repository: RepositoryPort,
    resolved: list[ResolvedPrediction],
    costs: PaperCosts,
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
            for stamp, price in repository.get_price_history(
                BENCHMARK_SYMBOL, settings.track_record_days + 1
            )
        ]
    except Exception:
        logger.exception("Failed to read %s snapshots for the benchmark", BENCHMARK_SYMBOL)
        history = []
    try:
        return benchmark_comparison(
            resolved,
            costs,
            history,
            settle_after=timedelta(hours=settings.reflection_min_age_hours),
        )
    except Exception:
        logger.exception("Failed to compare the equity curve with benchmarks")
        return None


def render_public_scorecard_page(
    settings: Settings, repository: RepositoryPort, costs: PaperCosts | None = None
) -> str:
    """#10 — strona z agregatami (N, hit-rate, Brier, ECE, krzywe) z okna
    `track_record_days`; "" gdy okno nie ma danych.

    Metryki liczone z `resolved`/`cal_rows` NIEZALEŻNIE od flag
    `equity_curve_enabled` / `calibration_curve_enabled` — tamte sterują
    sekcjami prywatnego maila, nie publicznym dowodem. Krzywa kapitału to
    portfel o równej wadze per cykl (WIA-17), po kosztach `costs` (WIA-50;
    None = brutto); strona ujawnia okno i datę generacji (WIA-33).
    Błąd odczytu leci wyżej: Fast Loop go połyka, `main_scorecard` kończy exit 1.
    """
    resolved = parse_resolved_predictions(
        repository.get_resolved_predictions(settings.track_record_days)
    )
    # WIA-39/WIA-58: kierunkowe próbki; cykle dają n_eff i bootstrap ECE.
    calibration = split_calibration_samples(
        repository.get_resolved_for_calibration(settings.track_record_days)
    )
    benchmark = (
        None
        if costs is None
        else _benchmark_comparison(settings, repository, resolved, costs)
    )
    return render_public_scorecard_html(
        resolved,
        portfolio_curve(resolved, costs),
        calibration.directional,
        generated_on=datetime.now(UTC).date(),
        window_days=settings.track_record_days,
        costs=costs,
        benchmark=benchmark,
        protocol=load_evaluation_protocol(),
        calibration_by_cycle=calibration.directional_by_cycle,
        coverage=_data_coverage(settings, repository),
    )


def _publish_public_scorecard(
    settings: Settings,
    repository: RepositoryPort,
    publisher: WebPublisherPort,
    costs: PaperCosts | None = None,
) -> None:
    """#10 — publikuje `render_public_scorecard_page` w cyklu Fast Loopa.

    Best-effort: błąd odczytu ani `OSError` przy zapisie nie wywalają cyklu.
    """
    if not settings.public_scorecard_enabled:
        return
    try:
        page = render_public_scorecard_page(settings, repository, costs)
        if page:
            publisher.publish(page)
    except Exception:
        logger.exception("Public scorecard publish failed — pomijam")


def build_repository(settings: Settings) -> RepositoryPort:
    """Repozytorium Fast Loopa (także `main_watch` i `main_deadman`).

    BEZ-08: bez `db_writes_enabled` owinięte w `ReadOnlyRepository` — lokalny run
    czyta wspólną bazę, ale niczego do niej nie zapisuje."""
    repository = SupabaseRepository(
        url=settings.supabase_url,
        key=settings.supabase_key,
        council_votes_upsert_enabled=settings.council_votes_upsert_enabled,
    )
    if settings.db_writes_enabled:
        return repository
    return ReadOnlyRepository(repository)


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
    usage_sink: UsageSink | None = None,
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
            usage_sink=usage_sink,
        )

    from src.infrastructure.llm.openai_client import DEFAULT_MODEL as OPENAI_DEFAULT
    from src.infrastructure.llm.openai_client import OpenAIAdapter

    return OpenAIAdapter(
        api_key=settings.require_openai_api_key(),
        model=settings.council_llm_model or OPENAI_DEFAULT,
        quota_monitor=quota_monitor,
        usage_sink=usage_sink,
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
    settings: Settings,
    quota_monitor: QuotaMonitor | None = None,
    usage_sink: UsageSink | None = None,
) -> EmbeddingPort | None:
    """Embeddingi — OpenAI niezależnie od LLM_PROVIDER.

    Anthropic nie udostępnia natywnego embeddings API; OpenAI to nasz jedyny
    provider w stacku. Fabryka żąda klucza OpenAI (`require_openai_api_key`,
    ARCH-05) niezależnie od LLM_PROVIDER — embeddingi działają również
    gdy LLM_PROVIDER=anthropic. Dzięki temu kolumna `embedding` w
    `prediction_logs` (pgvector) jest wypełniona dla wszystkich deploymentów,
    a similarity search nad newsami nie zależy od wyboru LLM.
    """
    from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingAdapter

    return OpenAIEmbeddingAdapter(
        api_key=settings.require_openai_api_key(),
        quota_monitor=quota_monitor,
        usage_sink=usage_sink,
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
        market_port=FinnhubAdapter(
            api_key=settings.require_finnhub_api_key(), rate_limiter=FINNHUB_RATE_LIMITER
        ),
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


def build_alpha_enrichment(
    settings: Settings,
    quota_monitor: QuotaMonitor | None = None,
    *,
    repository: RepositoryPort | None = None,
) -> AlphaEnrichmentUseCase:
    """DI dla nowych źródeł alpha (Dane). Każdy port to realny adapter (gdy
    flaga on) albo Null-adapter (default → zero I/O). EDGAR/Reddit są darmowe;
    earnings reużywa klucza AV; opcje/analitycy reużywają klucza Finnhub
    i jego limitera procesu (DAT-12).

    FIN-11: z `earnings_cache_enabled` kalendarz wyników czyta wyłącznie
    `earnings_cache` z `repository` (delegat Null) — Fast Loop nie woła AV."""
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
    earnings: EarningsCalendarPort = NullEarningsCalendarAdapter()
    if settings.earnings_calendar_enabled and settings.earnings_cache_enabled:
        # FIN-11: sam odczyt cache (delegat Null); bez repozytorium nie ma czego czytać.
        if repository is not None:
            from src.infrastructure.adapters.cached_earnings import CachedEarningsAdapter

            earnings = CachedEarningsAdapter(repo=repository, delegate=earnings)
    elif (
        settings.earnings_calendar_enabled
        and settings.alpha_vantage_api_keys
        and settings.paid_calls_enabled
    ):
        # FIN-35: kalendarz wyników to płatne zapytanie AV — lokalnie bez płatnych
        # wywołań zostaje Null-adapter.
        earnings = AlphaVantageEarningsAdapter(api_key=settings.alpha_vantage_api_keys[0])
    options: OptionsFlowPort = (
        FinnhubOptionsAdapter(
            api_key=settings.require_finnhub_api_key(),
            quota_monitor=quota_monitor,
            rate_limiter=FINNHUB_RATE_LIMITER,
        )
        if settings.options_flow_enabled
        else NullOptionsFlowAdapter()
    )
    social: SocialVelocityPort = (
        RedditSocialAdapter(user_agent=settings.edgar_user_agent)
        if settings.social_velocity_enabled
        else NullSocialVelocityAdapter()
    )
    analyst: AnalystConsensusPort = (
        FinnhubAnalystAdapter(
            api_key=settings.require_finnhub_api_key(),
            quota_monitor=quota_monitor,
            rate_limiter=FINNHUB_RATE_LIMITER,
        )
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


def _provider_default_model(provider: str) -> str:
    """Model, którego adapter danego providera używa bez jawnego `model=`."""
    if provider == "anthropic":
        from src.infrastructure.llm.anthropic_client import DEFAULT_MODEL as ANTHROPIC_DEFAULT

        return ANTHROPIC_DEFAULT
    from src.infrastructure.llm.openai_client import DEFAULT_MODEL as OPENAI_DEFAULT

    return OPENAI_DEFAULT


def _main_llm_identity(settings: Settings) -> tuple[str, str]:
    """(provider, model) głównego LLM — te same reguły co `build_llm_adapter`."""
    if settings.model_routing_enabled:
        return "routing", (
            f"openai:{settings.router_cheap_model}"
            f"|{settings.router_frontier_provider}:{settings.router_frontier_model}"
        )
    return settings.llm_provider, _provider_default_model(settings.llm_provider)


def _council_llm_identity(
    settings: Settings, main_identity: tuple[str, str]
) -> tuple[str, str]:
    """(provider, model) rady — te same reguły co `build_council_llm_adapter`."""
    if settings.council_llm_provider is None and settings.council_llm_model is None:
        return main_identity
    provider = settings.council_llm_provider or settings.llm_provider
    return provider, settings.council_llm_model or _provider_default_model(provider)


def build_prediction_recipe(settings: Settings, run_context: RunContext) -> PredictionRecipe:
    """WIA-20: przepis predykcji tego cyklu — liczony RAZ, wstrzykiwany do grafu
    jednym polem `AgentGraphDeps.recipe`. Czyta pliki promptów, person i wag,
    zero płatnych wywołań. Sekrety nie wchodzą (`Settings.config_hash`)."""
    main_identity = _main_llm_identity(settings)
    council_provider, council_model = _council_llm_identity(settings, main_identity)
    return PredictionRecipe(
        llm_provider=main_identity[0],
        llm_model=main_identity[1],
        council_provider=council_provider,
        council_model=council_model,
        prompt_sha256=prompt_templates_sha256(),
        personas_sha256=personas_sha256(Path(settings.council_personas_dir)),
        config_hash=settings.config_hash(),
        model_version=ml_model_version(settings.ml_model_path),
        universe_version=universe_version(settings.symbols, settings.crypto_symbols),
        git_sha=run_context.git_sha,
        run_id=run_context.run_id,
        protocol_sha256=load_evaluation_protocol().sha256,
    )


def build_move_calibration(
    settings: Settings, repository: RepositoryPort
) -> dict[AssetType, tuple[CalibrationPoint, ...]] | None:
    """ML-34 — okna kalibracyjne przedziału ruchu per klasa aktywa, RAZ na cykl.

    Darmowy odczyt rozliczonych predykcji (kolumny istniejące, bez migracji).
    Błąd → None: cykl leci bez przedziału zamiast bez maila."""
    crypto_set, etf_set = set(settings.crypto_symbols), set(settings.symbols_etf)
    try:
        rows = repository.get_settled_moves(CALIBRATION_LOOKBACK_DAYS)
    except Exception:
        logger.exception("Settled moves unavailable — cykl bez przedziałów ruchu")
        return None
    return calibration_windows(
        rows, lambda symbol: _classify_asset(symbol, crypto_set, etf_set)
    )


class _PaidPorts(NamedTuple):
    """FIN-35: porty Fast Loopa, które płacą za wywołanie — produkcyjne albo atrapy."""

    llm: LLMPort
    sentiment: SentimentPort
    news: NewsPort
    embedding: EmbeddingPort | None
    council: AdvisoryCouncilPort | None
    tool_use: ToolUseLLMPort | None


def _production_paid_ports(
    settings: Settings,
    telemetry: CycleTelemetry,
    repository: RepositoryPort,
    quota_monitor: QuotaMonitor | None,
    usage_sink: UsageSink | None = None,
) -> _PaidPorts:
    """Prawdziwe płatne adaptery (GHA: `PAID_CALLS_ENABLED=true`).

    Jeden `AlphaVantageClient` dla akcji i krypto: adapter wewnętrznie prefiksuje
    BTC → CRYPTO:BTC w requestach NEWS_SENTIMENT. FIN-13: każdy adapter LLM
    i embeddingów zgłasza tokeny do `usage_sink` (ledger przebiegu). EXP-27: w cyklu
    dwufazowym klient AV powstaje dopiero w fazie 2 (`build_phase_two_feeds`),
    a deps grafu dostają zaślepkę `PhaseTwoOnlyFeed`."""
    sentiment: SentimentPort
    news: NewsPort
    if settings.two_phase_cycle_enabled:
        sentiment = news = PhaseTwoOnlyFeed()
    else:
        av_client = _alpha_vantage_client(
            settings, [*settings.symbols, *settings.crypto_symbols], quota_monitor
        )
        sentiment = AlphaVantageSentimentAdapter(av_client)
        news = AlphaVantageNewsAdapter(av_client)
    llm_port = build_llm_adapter(
        settings, quota_monitor=quota_monitor, usage_sink=usage_sink, telemetry=telemetry
    )
    # FIN-02: każdy request persony, chairmana i rewizji liczy się u źródła.
    # Osobny obiekt niż `llm_port`, więc główna analiza nie liczy się podwójnie.
    council_llm_port = MeteredLLMPort(
        build_council_llm_adapter(
            settings, llm_port, quota_monitor=quota_monitor, usage_sink=usage_sink
        ),
        telemetry,
        "council_llm",
    )
    return _PaidPorts(
        llm=llm_port,
        sentiment=sentiment,
        news=news,
        embedding=build_embedding_adapter(
            settings, quota_monitor=quota_monitor, usage_sink=usage_sink
        ),
        council=build_council_adapter(settings, council_llm_port, repository=repository),
        tool_use=build_tool_use_adapter(
            settings, quota_monitor=quota_monitor, usage_sink=usage_sink
        ),
    )


def _alpha_vantage_client(
    settings: Settings, symbols: Sequence[str], quota_monitor: QuotaMonitor | None
) -> AlphaVantageClient:
    return AlphaVantageClient(
        api_keys=settings.require_alpha_vantage_api_keys(),
        symbols=list(symbols),
        crypto_symbols=settings.crypto_symbols,
        quota_monitor=quota_monitor,
        # `news_volume_24h` ma liczyć ostatnie 24 h, nie „50 najnowszych".
        news_window=NEWS_WINDOW,
        # FIN-04: request AV tylko dla symbolu za bramką (FEATURE_SET_VERSION 3).
        lazy_per_symbol=True,
    )


# EXP-27: porty feedu fazy 2 dla symboli za bramką (w kolejności fazy 2).
PhaseTwoFeeds = Callable[[Sequence[str]], tuple[SentimentPort, NewsPort]]


def build_phase_two_feeds(
    settings: Settings, quota_monitor: QuotaMonitor | None
) -> PhaseTwoFeeds:
    """EXP-27: fabryka feedu fazy 2 — `AlphaVantageClient` z samymi symbolami za
    bramką, w trybie leniwym FIN-04: request idzie tylko dla symbolu, który
    faktycznie dostaje analizę (odcięty sufitem FIN-12 nie kosztuje AV), a cechy
    newsów liczą własny feed symbolu jak w cyklu jednofazowym (FEATURE_SET_VERSION 3).
    FIN-35: bez `paid_calls_enabled` atrapy, bez klienta AV."""

    def feeds(passed: Sequence[str]) -> tuple[SentimentPort, NewsPort]:
        if not settings.paid_calls_enabled:
            return LocalStubSentimentAdapter(), LocalStubNewsAdapter()
        client = _alpha_vantage_client(settings, passed, quota_monitor)
        return AlphaVantageSentimentAdapter(client), AlphaVantageNewsAdapter(client)

    return feeds


def _local_stub_ports() -> _PaidPorts:
    """FIN-35: lokalny run bez płatnych wywołań. Neutralny LLM i pusty feed AV
    (z flagą `paid_calls_disabled`) przechodzą cały graf; rada, embeddingi
    i tool-use są wyłączone (None), więc nie udają werdyktu ani analogów RAG."""
    return _PaidPorts(
        llm=LocalStubLLMAdapter(),
        sentiment=LocalStubSentimentAdapter(),
        news=LocalStubNewsAdapter(),
        embedding=None,
        council=None,
        tool_use=None,
    )


# FIN-35: kod wyjścia odmowy startu — inny niż 1 („wszystkie symbole padły”).
STUB_WRITES_REFUSED_EXIT_CODE = 2


def refuses_stub_writes(settings: Settings) -> bool:
    """FIN-35: atrapy płatnych portów + zapis do bazy = neutralne prognozy atrapy
    w produkcyjnym track recordzie. True (z ERROR-em) = entry point nie startuje."""
    if not settings.db_writes_enabled or settings.paid_calls_enabled:
        return False
    logger.error(
        "DB_WRITES_ENABLED=true przy PAID_CALLS_ENABLED=false — odmowa startu: atrapy "
        "płatnych portów zapisałyby neutralne prognozy do produkcyjnej bazy. Ustaw "
        "PAID_CALLS_ENABLED=true (płatny przebieg) albo DB_WRITES_ENABLED=false "
        "(lokalny przebieg tylko do odczytu)."
    )
    return True


def warn_if_paid_calls_disabled(settings: Settings) -> None:
    """FIN-35: jeden jawny WARNING na przebieg, gdy płatne porty idą przez atrapy."""
    if settings.paid_calls_enabled:
        return
    logger.warning(
        "PAID_CALLS_ENABLED=false — lokalnie bez płatnych wywołań: LLM, rada, "
        "Alpha Vantage i embeddingi zastąpione atrapami (prognozy neutralne)"
    )


def build_use_case(
    settings: Settings,
    repository: RepositoryPort | None = None,
    quota_monitor: QuotaMonitor | None = None,
    run_context: RunContext | None = None,
    usage_ledger: UsageLedger | None = None,
) -> AnalyzeMarketUseCase:
    """DI Container — buduje wszystkie adaptery i wstrzykuje do Use Case'a.

    FIN-35: płatne porty (LLM, rada, Alpha Vantage, embeddingi, tool-use) są
    produkcyjne tylko przy `paid_calls_enabled`; inaczej graf dostaje atrapy
    (`_local_stub_ports`). Klient AV działa leniwie per symbol (FIN-04): symbol
    pod bramką volatility nie kosztuje żadnego requestu AV.

    `run_context` (MAIL-42) wchodzi do przepisu predykcji (WIA-20); brak = run
    lokalny. `usage_ledger` (FIN-13) dostaje tokeny płatnych adapterów; atrapy
    FIN-35 nie wysyłają requestów, więc niczego do niego nie zgłaszają.
    """
    # ARCH-11: jedna telemetria na cykl — dostają ją graf i licznik requestów rady.
    telemetry = CycleTelemetry()
    supabase_repo = repository or build_repository(settings)
    paid = (
        _production_paid_ports(
            settings, telemetry, supabase_repo, quota_monitor, usage_sink=usage_ledger
        )
        if settings.paid_calls_enabled
        else _local_stub_ports()
    )
    # Fast loop: delegate = Null — nie woła płatnego AV, czyta tylko cache Supabase.
    fundamentals_port = CachedFundamentalsAdapter(
        repo=supabase_repo,
        delegate=NullFundamentalsAdapter(),
    )
    # Routing: krypto → CoinGecko, reszta → Finnhub. Use case widzi jeden port.
    market_port = RoutingMarketDataPort(
        equity=FinnhubAdapter(
            api_key=settings.require_finnhub_api_key(),
            quota_monitor=quota_monitor,
            rate_limiter=FINNHUB_RATE_LIMITER,
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
    tool_use_port = paid.tool_use
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
    return AnalyzeMarketUseCase(AgentGraphDeps(
        market_port=market_port,
        sentiment_port=paid.sentiment,
        news_port=paid.news,
        repository_port=supabase_repo,
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        llm_port=paid.llm,
        threshold=Threshold(settings.volatility_threshold),
        embedding_port=paid.embedding,
        council_port=paid.council,
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
        # 0 = bez granicy wieku → None, żeby repo nie dostało kwarga (WIA-04).
        reflection_max_age_hours=settings.reflection_max_age_hours or None,
        rag_outcome_weight=settings.rag_outcome_weight,
        tool_use_port=tool_use_port,
        research_tools=research_tools,
        tool_use_threshold=tool_use_threshold,
        vector_memory_enabled=settings.vector_memory_enabled,
        regime_tagging_enabled=_regime_tagging_enabled(settings),
        receipts_enabled=settings.receipts_enabled,
        feature_set_version_enabled=settings.feature_set_version_enabled,
        # WIA-20: przepis liczymy tylko, gdy ktoś go zapisze (kwit v2, kolumna
        # `recipe_version` albo commitment v2 — WIA-41); flagi off → zero odczytów plików.
        recipe=(
            build_prediction_recipe(settings, run_context or RunContext())
            if settings.receipts_enabled
            or settings.recipe_version_enabled
            or settings.attestation_enabled
            else None
        ),
        recipe_version_enabled=settings.recipe_version_enabled,
        # ML-14: off → graf nie dotyka tabeli `feature_snapshots` (migracja 031).
        feature_snapshots_enabled=settings.feature_snapshots_enabled,
        # ML-34: przedział w mailu zawsze; kolumny `interval_*` (migracja 032) za flagą.
        move_calibration=build_move_calibration(settings, supabase_repo),
        interval_columns_enabled=settings.interval_coverage_enabled,
        # EXP-01: kolumna `gate_delta` (migracja 034) za flagą.
        gate_delta_enabled=settings.gate_delta_enabled,
        # ALR-10: bramka widzi dryf od ostatniej predykcji (darmowe odczyty); off.
        drift_gate_enabled=settings.drift_gate_enabled,
        # EXP-05: eksperymenty (ramię per symbol-dzień liczy use case); kolumna
        # `experiment_arms` (migracja 037) za flagą, kwit v2 za `receipts_enabled`.
        experiments=settings.experiment_definitions(),
        experiment_arms_enabled=settings.experiment_arms_enabled,
        # #18: port opcji trafia DO GRAFU (fetch za bramką volatility), a nie tylko
        # do enrichmentu po grafie. Off → None → sygnał nie powstaje.
        options_port=build_options_port(settings, quota_monitor),
        # #16: publisher tylko gdy flaga on — inaczej nowe kolumny przed
        # migracją 024 dałyby PGRST204 i zabiły zapis całej predykcji. WIA-70:
        # flaga on bez kolumn 024/043 → cykl bez atestacji, predykcje zapisane.
        attestation_publisher=(
            build_attestation_publisher(settings)
            if settings.attestation_enabled and supabase_repo.attestation_columns_available()
            else None
        ),
        telemetry=telemetry,
        # FIN-12: sufit płatnych wywołań LLM cyklu; None = bez limitu.
        paid_call_cap=settings.max_paid_llm_calls_per_cycle,
    ))


def cycle_cost(
    settings: Settings, telemetry: CycleTelemetry, ledger: UsageLedger
) -> CycleCost | None:
    """FinOps: koszt CAŁEGO cyklu z telemetrii grafu (ARCH-11).

    FIN-13: źródło z wycenionymi tokenami w ledgerze dostaje koszt z tokenów,
    reszta zostaje na `UNIT_COST_USD`. Cykl w całości odcięty bramką volatility
    ma zerowy licznik → sekcja mówi "cykl darmowy". FIN-35: atrapy liczą się
    w telemetrii jak prawdziwe wywołania, więc lokalnie sekcja kosztu by
    kłamała — chowamy ją (None)."""
    if not settings.paid_calls_enabled:
        return None
    return estimate_cycle_cost(
        telemetry.snapshot().paid_calls,
        token_costs=ledger.token_costs(settings.llm_price_table()),
        priced_as_of=settings.llm_pricing_as_of,
    )


def persist_llm_calls(
    *,
    settings: Settings,
    repository: RepositoryPort,
    ledger: UsageLedger,
    run_context: RunContext,
) -> None:
    """FIN-13: ledger `llm_calls` (migracja 040) jednym batchem po przebiegu.

    Tylko za `cost_ledger_enabled` — bez migracji tabela nie istnieje. Zapis idzie
    przez repozytorium przebiegu, więc lokalnie pomija go strażnik BEZ-08. Błąd
    zapisu nie wywraca przebiegu: ledger to księgowość, nie wynik cyklu."""
    if not settings.cost_ledger_enabled:
        return
    entries = ledger.entries(
        run_id=run_context.run_id,
        price_table=settings.llm_price_table(),
        price_as_of=settings.llm_pricing_as_of,
    )
    if not entries:
        return
    try:
        repository.save_llm_calls(entries)
    except Exception:
        logger.exception(
            "save_llm_calls failed — %d llm_calls rows of run %s not written",
            len(entries),
            run_context.run_id,
        )


def run_attempt_from_environ(environ: Mapping[str, str]) -> int | None:
    """NIEZ-29: numer próby runu (`GITHUB_RUN_ATTEMPT`) — ponowienie ma ten sam
    `run_id`. Jak `RunContext`: poza GitHub Actions zawsze None, a wartość, która
    nie jest liczbą ≥ 1, też None (kolumna ma CHECK >= 1)."""
    if environ.get("GITHUB_ACTIONS") != "true":
        return None
    raw = environ.get("GITHUB_RUN_ATTEMPT", "")
    if not raw.isdigit() or int(raw) < 1:
        return None
    return int(raw)


def persist_cycle_run(*, settings: Settings, repository: RepositoryPort, run: CycleRun) -> None:
    """NIEZ-29: wiersz `cycle_runs` (migracja 041) na końcu cyklu.

    Tylko za `cycle_runs_enabled` — bez migracji tabela nie istnieje. Zapis idzie
    przez repozytorium przebiegu, więc lokalnie pomija go strażnik BEZ-08. Błąd
    zapisu nie wywraca przebiegu: historia to ślad cyklu, nie jego wynik."""
    if not settings.cycle_runs_enabled:
        return
    try:
        repository.save_cycle_run(run)
    except Exception:
        logger.exception("save_cycle_run failed — cycle_runs row of run %s not written", run.run_id)


def _regime_tagging_enabled(settings: Settings) -> bool:
    """ML-38: brak wartości = tag reżimu idzie razem z `regime_aware_enabled`.

    Świadomy wyjątek od „nowy zapis za flagą domyślnie off”: kolumna
    `prediction_logs.regime` pochodzi z migracji 017, potwierdzonej na produkcji
    (2026-10-18). Jawne `regime_tagging_enabled` zawsze wygrywa."""
    if settings.regime_tagging_enabled is None:
        return settings.regime_aware_enabled
    return settings.regime_tagging_enabled


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
    *,
    repository: RepositoryPort | None = None,
    cycle_at: datetime | None = None,
    on_banner: Callable[[RegimeBanner], None] | None = None,
) -> tuple[str, float]:
    """#7: reżim rynku z overall_alert (zacieśnia próg + label do promptu).
    Zwraca (regime_context, regime_multiplier). Off / brak raportu → ("", 1.0).

    ALR-05: `on_banner` dostaje reżim cyklu dla nagłówka maila (reżim, mnożnik,
    próg akcji, powód); off / brak raportu → nie jest wołany.

    ALR-03: z `repository` i flagą `regime_hysteresis_enabled` surowy reżim
    przechodzi przez histerezę (stan z `regime_history`), patrz
    `_apply_regime_hysteresis`. Bez repozytorium — surowy reżim jak dotąd.

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

    # Głos krzywej wymaga JEDNOCZEŚNIE yield_curve_enabled i regime_aware_enabled.
    risk_signals: list[MacroAlertLevel] = []
    if settings.yield_curve_enabled and yield_curve is not None:
        risk_signals.append(yield_curve_alert_level(yield_curve.state()))

    detector = RegimeDetector()
    raw_regime = detector.classify(macro_risk_report.overall_alert, risk_signals)
    regime = (
        _apply_regime_hysteresis(
            settings, repository, raw_regime, cycle_at or datetime.now(UTC)
        )
        if repository is not None
        else raw_regime
    )
    regime_multiplier = detector.threshold_multiplier(
        regime, settings.regime_threshold_max_multiplier
    )
    regime_context = detector.label(regime)
    logger.info(
        "Market regime — %s (surowy %s, mnożnik progu %.2f)",
        regime.value,
        raw_regime.value,
        regime_multiplier,
    )
    if on_banner is not None:
        # Reżim efektywny różny od surowego = histereza trzyma RISK_OFF (ALR-03).
        reason = (
            detector.reason(macro_risk_report.overall_alert, risk_signals)
            if regime is raw_regime
            else f"histereza: trzymam {regime.value} (surowy odczyt {raw_regime.value})"
        )
        on_banner(
            RegimeBanner(
                regime=regime,
                multiplier=regime_multiplier,
                base_threshold=settings.volatility_threshold,
                reason=reason,
            )
        )
    return regime_context, regime_multiplier


def _apply_regime_hysteresis(
    settings: Settings,
    repository: RepositoryPort,
    raw_regime: MarketRegime,
    cycle_at: datetime,
) -> MarketRegime:
    """ALR-03: reżim efektywny po histerezie — odczyt ostatniego stanu i zapis
    nowego wiersza `regime_history`, raz na cykl.

    Flaga off albo błąd odczytu (np. brak migracji 033) → surowy reżim, czyli
    dokładnie zachowanie sprzed ALR-03; po błędzie odczytu nie zapisujemy, żeby
    stan bez historii nie wyzerował trwającej serii. Błąd zapisu tylko logujemy:
    następny cykl przeczyta starszy stan, co najwyżej opóźniając wyjście z
    RISK_OFF (wyższy próg — strona bezpieczna dla FinOps).
    """
    if not settings.regime_hysteresis_enabled:
        return raw_regime
    try:
        previous = repository.get_latest_regime_state()
    except Exception:
        logger.exception(
            "regime_history read failed — surowy reżim %s bez histerezy",
            raw_regime.value,
        )
        return raw_regime
    state = apply_hysteresis(previous, raw_regime, settings.regime_exit_confirmations)
    try:
        repository.save_regime_state(cycle_at, raw_regime, state)
    except Exception:
        logger.exception(
            "regime_history write failed — następny cykl użyje starszego stanu"
        )
    if state.regime is not raw_regime:
        logger.info(
            "Histereza reżimu — trzymam %s (surowy %s, potwierdzenia wyjścia %d/%d)",
            state.regime.value,
            raw_regime.value,
            state.streak,
            settings.regime_exit_confirmations,
        )
    return state.regime


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
    alpha_fusion_enabled: bool = False,
    cycle_at: datetime | None = None,
    secret_values: tuple[str, ...] = (),
) -> _ProcessedSymbol:
    """Przetwarza jeden symbol. NIGDY nie rzuca — łapie wyjątek i zwraca wynik
    z `is_failure=True`. Dzięki temu pula wątków (ścieżka równoległa
    `_analyze_symbols`) nigdy nie widzi wyjątku, a semantyka resilience jest
    identyczna z dawną pętlą sekwencyjną.

    `earnings_gate_enabled` (#6) domyślnie WYŁĄCZONE — wtedy ścieżka jest
    bit-w-bit taka jak przed wprowadzeniem bramki: zero prefetchu, `enrich`
    wołany dokładnie jak dawniej. Włącza je `settings.earnings_calendar_enabled`.

    `alpha_fusion_enabled` (#14) też domyślnie WYŁĄCZONE. Fuzja zmienia prompt
    LLM i zapisuje kandydata na cechę ML (`alpha_fusion_score`, poza
    `ML_FEATURE_COLUMNS` — model go nie widzi), więc jej wdrożenie ma być
    świadomą, odwracalną decyzją; o promocji do cech modelu rozstrzyga
    porównanie A/B (EXP-19), nie intuicja dobierającego wagi.
    Off → `alpha_fusion=None` → prompt i zapis jak przed #14.
    """
    try:
        inputs = _symbol_inputs(
            symbol,
            crypto_set=crypto_set,
            etf_set=etf_set,
            collect_alpha=collect_alpha,
            alpha_use_case=alpha_use_case,
            earnings_gate_enabled=earnings_gate_enabled,
            alpha_fusion_enabled=alpha_fusion_enabled,
        )
        raw = use_case.run(
            symbol,
            asset=inputs.asset,
            regime_context=regime_context,
            regime_multiplier=regime_multiplier,
            earnings_multiplier=inputs.earnings_multiplier,
            alpha_fusion=inputs.alpha_fusion,
            peer_context=peer_ctx,
            # FIN-14: wiek referencji ceny liczy się od startu cyklu.
            now=cycle_at,
        )
        return _finished_symbol(symbol, raw, inputs.alpha_signal, collect_alpha, cycle_at)
    except Exception as exc:
        return _failed_symbol(symbol, exc, secret_values)


class _SymbolInputs(NamedTuple):
    """Wejścia grafu liczone PRZED nim: klasa aktywa, mnożnik earnings, alfa."""

    asset: Asset
    earnings_multiplier: float
    alpha_signal: AlphaSignals | None
    alpha_fusion: AlphaFusionScore | None


def _symbol_inputs(
    symbol: str,
    *,
    crypto_set: set[str],
    etf_set: set[str],
    collect_alpha: bool,
    alpha_use_case: AlphaEnrichmentUseCase,
    earnings_gate_enabled: bool,
    alpha_fusion_enabled: bool,
) -> _SymbolInputs:
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
    # #14: enrichment biegnie PRZED grafem. Dawniej działał po nim, więc
    # LLM i cechy XGBoost nigdy nie widziały sygnałów alfa — 5 pobieranych
    # źródeł było render-only. Liczba płatnych wywołań się NIE zmienia:
    # `enrich` i tak leciał raz na symbol, tylko później.
    alpha_signal: AlphaSignals | None = None
    alpha_fusion: AlphaFusionScore | None = None
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
        if alpha_fusion_enabled and alpha_signal is not None:
            alpha_fusion = fuse_alpha_signals(
                insider=alpha_signal.insider,
                options=alpha_signal.options,
                social=alpha_signal.social,
                analyst=alpha_signal.analyst,
                earnings=alpha_signal.earnings,
            )
    return _SymbolInputs(asset, earnings_multiplier, alpha_signal, alpha_fusion)


def _finished_symbol(
    symbol: str,
    raw: dict[str, Any],
    alpha_signal: AlphaSignals | None,
    collect_alpha: bool,
    cycle_at: datetime | None,
) -> _ProcessedSymbol:
    logger.info(
        "%s: status=%s delta=%s prediction_id=%s",
        symbol,
        raw.get("status"),
        raw.get("delta"),
        raw.get("prediction_id"),
    )
    sr = to_symbol_result(symbol, raw, now=cycle_at)
    alpha_price: Decimal | None = None
    if collect_alpha and sr.current_price is not None:
        alpha_price = sr.current_price
    return _ProcessedSymbol(sr, False, alpha_signal, alpha_price)


def _failed_symbol(
    symbol: str, exc: Exception, secret_values: tuple[str, ...]
) -> _ProcessedSymbol:
    logger.exception("Failed to process symbol %s", symbol)
    # BEZ-07: komunikat idzie do maila — ani URL z kluczem, ani dosłowna
    # wartość sekretu z Settings (`secret_values`) nie może w nim zostać.
    result = to_symbol_result(symbol, None, error=redact_secrets(str(exc), secret_values))
    # NIEZ-22: typ wyjątku osobno — step summary liczy błędy per klasa, bez komunikatu.
    return _ProcessedSymbol(
        dataclasses.replace(result, error_type=type(exc).__name__), True, None, None
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
    alpha_fusion_enabled: bool = False,
    cycle_at: datetime | None = None,
    secret_values: tuple[str, ...] = (),
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
                    alpha_fusion_enabled=alpha_fusion_enabled,
                    cycle_at=cycle_at,
                    secret_values=secret_values,
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
    cycle_at: datetime | None = None,
    secret_values: tuple[str, ...] = (),
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
            alpha_fusion_enabled=settings.alpha_fusion_enabled,
            cycle_at=cycle_at,
            secret_values=secret_values,
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


class _GatedSymbol(NamedTuple):
    """EXP-27: symbol po fazie 1 — skończony (`done`) albo czekający na fazę 2."""

    symbol: str
    done: _ProcessedSymbol | None
    state: AgentState | None
    alpha_signal: AlphaSignals | None


def _gate_one_symbol(
    symbol: str,
    *,
    use_case: AnalyzeMarketUseCase,
    crypto_set: set[str],
    etf_set: set[str],
    regime_context: str,
    regime_multiplier: float,
    collect_alpha: bool,
    alpha_use_case: AlphaEnrichmentUseCase,
    earnings_gate_enabled: bool,
    alpha_fusion_enabled: bool,
    cycle_at: datetime | None,
    secret_values: tuple[str, ...],
) -> _GatedSymbol:
    """EXP-27 faza 1 jednego symbolu: te same wejścia co `_process_one_symbol`
    (earnings, alfa) i graf bramki. NIGDY nie rzuca — błąd to wynik `is_failure`."""
    try:
        inputs = _symbol_inputs(
            symbol,
            crypto_set=crypto_set,
            etf_set=etf_set,
            collect_alpha=collect_alpha,
            alpha_use_case=alpha_use_case,
            earnings_gate_enabled=earnings_gate_enabled,
            alpha_fusion_enabled=alpha_fusion_enabled,
        )
        outcome = use_case.run_gate(
            symbol,
            asset=inputs.asset,
            regime_context=regime_context,
            regime_multiplier=regime_multiplier,
            earnings_multiplier=inputs.earnings_multiplier,
            alpha_fusion=inputs.alpha_fusion,
            now=cycle_at,
        )
        if outcome.passed:
            return _GatedSymbol(symbol, None, outcome.state, inputs.alpha_signal)
        done = _finished_symbol(
            symbol, dict(outcome.state), inputs.alpha_signal, collect_alpha, cycle_at
        )
        return _GatedSymbol(symbol, done, None, inputs.alpha_signal)
    except Exception as exc:
        return _GatedSymbol(symbol, _failed_symbol(symbol, exc, secret_values), None, None)


def _analyze_gated_symbol(
    gated: _GatedSymbol,
    *,
    paid_phase: PaidPhase,
    peer_ctx: tuple[tuple[str, str], ...],
    collect_alpha: bool,
    cycle_at: datetime | None,
    secret_values: tuple[str, ...],
) -> _ProcessedSymbol:
    """EXP-27 faza 2 jednego symbolu za bramką. NIGDY nie rzuca."""
    assert gated.state is not None, "faza 2 dostaje tylko symbole za bramką"
    try:
        raw = paid_phase.run(gated.state, peer_context=peer_ctx)
        return _finished_symbol(gated.symbol, raw, gated.alpha_signal, collect_alpha, cycle_at)
    except Exception as exc:
        return _failed_symbol(gated.symbol, exc, secret_values)


def _parallel_paths(settings: Settings) -> bool:
    """Równolegle tylko przy concurrency > 1 i bez contagion (#5 zależy od kolejności)."""
    return settings.symbol_concurrency > 1 and not settings.contagion_enabled


def _run_paid_phase(
    order: list[str],
    gated: Mapping[str, _GatedSymbol],
    *,
    paid_phase: PaidPhase,
    settings: Settings,
    collect_alpha: bool,
    cycle_at: datetime | None,
    secret_values: tuple[str, ...],
) -> dict[str, _ProcessedSymbol]:
    """EXP-27 faza 2 w kolejności `order` (malejące |Δ|). Równolegle jak cykl
    jednofazowy (pula startuje symbole w tej kolejności); sekwencyjnie z contagion
    liczonym z werdyktów wcześniejszych symboli fazy 2 i throttle między nimi."""
    if _parallel_paths(settings):
        with ThreadPoolExecutor(max_workers=settings.symbol_concurrency) as executor:
            processed = list(
                executor.map(
                    lambda sym: _analyze_gated_symbol(
                        gated[sym],
                        paid_phase=paid_phase,
                        peer_ctx=(),
                        collect_alpha=collect_alpha,
                        cycle_at=cycle_at,
                        secret_values=secret_values,
                    ),
                    order,
                )
            )
        return dict(zip(order, processed, strict=True))

    results: dict[str, _ProcessedSymbol] = {}
    processed_stances: dict[str, str] = {}
    for index, symbol in enumerate(order):
        peer_ctx = (
            build_peer_context(symbol, settings.peer_groups, processed_stances)
            if settings.contagion_enabled
            else ()
        )
        proc = _analyze_gated_symbol(
            gated[symbol],
            paid_phase=paid_phase,
            peer_ctx=peer_ctx,
            collect_alpha=collect_alpha,
            cycle_at=cycle_at,
            secret_values=secret_values,
        )
        results[symbol] = proc
        if not proc.is_failure and proc.result.trend:
            processed_stances[symbol] = proc.result.trend
        # Throttle MIĘDZY płatnymi symbolami — chroni OpenAI TPM.
        if settings.symbol_throttle_seconds > 0 and index < len(order) - 1:
            time.sleep(settings.symbol_throttle_seconds)
    return results


def _analyze_symbols_two_phase(
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
    phase_two_feeds: PhaseTwoFeeds,
    cycle_at: datetime | None,
    secret_values: tuple[str, ...],
) -> tuple[list[SymbolResult], int, dict[str, AlphaSignals], dict[str, Decimal]]:
    """EXP-27: cykl dwufazowy. Faza 1 (wszystkie symbole): cena, reflect, bramka
    domenowa; symbol pod progiem kończy jak dziś. Faza 2: feed AV zbudowany dla
    symboli za bramką i płatna analiza w kolejności `rank_by_magnitude`, więc
    sufit FIN-12 odcina najmniejsze |Δ|. Wyniki w kolejności `eligible`."""

    def gate(symbol: str) -> _GatedSymbol:
        return _gate_one_symbol(
            symbol,
            use_case=use_case,
            crypto_set=crypto_set,
            etf_set=etf_set,
            regime_context=regime_context,
            regime_multiplier=regime_multiplier,
            collect_alpha=collect_alpha,
            alpha_use_case=alpha_use_case,
            earnings_gate_enabled=settings.earnings_calendar_enabled,
            alpha_fusion_enabled=settings.alpha_fusion_enabled,
            cycle_at=cycle_at,
            secret_values=secret_values,
        )

    # Faza 1 nie zależy od kolejności (contagion żyje w fazie 2), więc idzie
    # równolegle przy każdym concurrency > 1.
    if settings.symbol_concurrency > 1:
        with ThreadPoolExecutor(max_workers=settings.symbol_concurrency) as executor:
            gated_list = list(executor.map(gate, eligible))
    else:
        gated_list = [gate(symbol) for symbol in eligible]

    processed = {g.symbol: g.done for g in gated_list if g.done is not None}
    passed = {g.symbol: g for g in gated_list if g.state is not None}
    order = rank_by_magnitude(
        {symbol: g.state["delta"] for symbol, g in passed.items() if g.state is not None}
    )
    logger.info(
        "EXP-27 two-phase cycle: %d/%d symbols past the gate, paid phase by |Δ|: %s",
        len(order),
        len(eligible),
        order,
    )
    if order:
        sentiment_port, news_port = phase_two_feeds(order)
        processed.update(
            _run_paid_phase(
                order,
                passed,
                paid_phase=use_case.paid_phase(
                    sentiment_port=sentiment_port, news_port=news_port
                ),
                settings=settings,
                collect_alpha=collect_alpha,
                cycle_at=cycle_at,
                secret_values=secret_values,
            )
        )

    ordered = [processed[symbol] for symbol in eligible]
    return (
        [p.result for p in ordered],
        sum(1 for p in ordered if p.is_failure),
        {
            s: p.alpha_signal
            for s, p in zip(eligible, ordered, strict=True)
            if p.alpha_signal is not None
        },
        {
            s: p.alpha_price
            for s, p in zip(eligible, ordered, strict=True)
            if p.alpha_price is not None
        },
    )


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
    cycle_at: datetime | None = None,
    phase_two_feeds: PhaseTwoFeeds | None = None,
) -> tuple[list[SymbolResult], int, dict[str, AlphaSignals], dict[str, Decimal]]:
    """Dispatcher: wybiera ścieżkę równoległą lub sekwencyjną; z `phase_two_feeds`
    (EXP-27, `two_phase_cycle_enabled`) cykl dwufazowy.
    Zwraca (wyniki, liczba_błędów, alpha_signals, alpha_prices)."""
    secret_values = settings.secret_values()
    if phase_two_feeds is not None:
        return _analyze_symbols_two_phase(
            eligible,
            use_case=use_case,
            settings=settings,
            crypto_set=crypto_set,
            etf_set=etf_set,
            regime_context=regime_context,
            regime_multiplier=regime_multiplier,
            collect_alpha=collect_alpha,
            alpha_use_case=alpha_use_case,
            phase_two_feeds=phase_two_feeds,
            cycle_at=cycle_at,
            secret_values=secret_values,
        )
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
            alpha_fusion_enabled=settings.alpha_fusion_enabled,
            cycle_at=cycle_at,
            secret_values=secret_values,
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
        cycle_at=cycle_at,
        secret_values=secret_values,
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


def _build_position_risk(
    settings: Settings,
    repository: RepositoryPort,
    portfolio: Portfolio | None,
    rates: PlnRates | None = None,
) -> PositionRisk | None:
    """PORT-04 — VaR, obsunięcie i hedge REALNEGO portfela do sizingu.

    Tylko za `portfolio_var_enabled` i tylko przy realnych pozycjach: bez nich
    banda zostaje jak dotąd, a VaR radaru (równe wagi watchlisty) nigdy jej nie
    zwęża. Hedge'em są wyłącznie trzymane instrumenty typu INVERSE_EQUITY
    i VOLATILITY z `risk_symbol_types` (`EQUITY_HEDGE_TYPES`) — GLD czy EPOL to
    inwestycja. Darmowa historia cen; błąd → None (banda bez zmian).

    `rates` (kursy NBP) liczą wagi w złotych; pozycja bez kursu wypada."""
    if not settings.portfolio_var_enabled or portfolio is None:
        return None
    try:
        return PortfolioRiskUseCase(repository_port=repository).position_risk(
            portfolio,
            hedge_symbols=equity_hedge_symbols(settings.risk_symbol_types),
            rates=rates,
        )
    except Exception:
        logger.exception("Position risk failed — banda bez ryzyka portfela")
        return None


def _sector_map(settings: Settings) -> dict[str, str]:
    """PORT-05 — symbol → etykieta sektora z jednego źródła: `sectors`,
    `symbols_etf` i `crypto_symbols` z config.toml."""
    return build_sector_map(
        settings.sectors,
        etf_symbols=settings.symbols_etf,
        crypto_symbols=settings.crypto_symbols,
    )


def _portfolio_sectors(
    results: list[SymbolResult], sector_map: Mapping[str, str]
) -> dict[str, int]:
    """#4 — skład portfela wg sektorów (sektor → liczba instrumentów).

    Etykiety z mapy sektorów configu (PORT-05); symbole spoza mapy trafiają do
    "Inne", żeby licznik zawsze sumował się do liczby instrumentów.
    """
    counts: dict[str, int] = {}
    for r in results:
        sector = sector_map.get(r.symbol, "Inne")
        counts[sector] = counts.get(sector, 0) + 1
    return counts


def _classify_cycle_maturity(results: list[SymbolResult]) -> CycleMaturity:
    """#4 — dojrzałość cyklu z policzonych przyczyn pominięcia.

    Symbole nieobsługiwane cenowo są jawnie POZA mianownikiem (nigdy nie dostaną
    punktu odniesienia), a błędy liczone osobno — cykl zdominowany przez awarie
    źródeł nie może zostać powitaniem "Dzień 1". Reguła siedzi w domenie.
    """
    saved = sum(1 for r in results if r.status == "saved")
    errors = sum(1 for r in results if r.status == "error")
    cold_start = sum(1 for r in results if r.skip_reason is SkipReason.COLD_START)
    below = sum(1 for r in results if r.skip_reason is SkipReason.BELOW_THRESHOLD)
    stale = sum(1 for r in results if r.skip_reason is SkipReason.STALE_REFERENCE)
    # SkipReason.UNSUPPORTED_PRICE celowo nie ma tu swojej liczności — te symbole
    # wypadają z mianownika przez to, że nie trafiają do ŻADNEJ z powyższych sum.
    return classify_cycle(
        saved_count=saved,
        cold_start_count=cold_start,
        below_threshold_count=below,
        error_count=errors,
        stale_reference_count=stale,
    )


def _build_subject(
    *,
    lead_items: list[LeadItem],
    analyzed: int,
    failures: int,
    started_at: datetime,
    prefix: str,
    maturity: CycleMaturity = CycleMaturity.STEADY_STATE,
    analyzed_total: int = 0,
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE,
) -> str:
    """#1 — temat maila jak nagłówek gazety, nie jak licznik.

    Gdy cykl ma lead, jego pierwsza pozycja idzie do tematu ("NVDA -4,2%, rada
    PODZIELONA"); bez leadu zostaje dotychczasowy temat licznikowy. `prefix`
    (np. "⚠️ [QUOTA] ") jest sygnałem operacyjnym i przetrwa w obu wariantach.

    Liczone RAZ nad pełnymi danymi cyklu: sekcje raportu są re-slicowane per
    subskrybent, ale temat jest dla wszystkich ten sam. MAIL-14: znacznik czasu
    w strefie `display_timezone` (`started_at` jest w UTC).
    """
    stamp = format_local(started_at, display_timezone)
    # #4: pierwszy cykl ma własny, powitalny temat — lead i tak jest wtedy pusty
    # (brak punktu odniesienia = brak sygnałów).
    welcome = onboarding_subject(maturity, instrument_count=analyzed_total)
    if welcome:
        return f"{prefix}{welcome} ({stamp})"
    headline = lead_headline(lead_items)
    if headline:
        return f"{prefix}StockAgent — {headline} ({stamp})"
    return f"{prefix}StockAgent — {analyzed} predykcji, {failures} błędów ({stamp})"


def _reveal_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """#16 — rekord revealu: sól, hash i plaintext predykcji z wynikiem.

    WIA-70: wiersz z `commitment_payload` (043) ujawnia zapisany plaintext
    dosłownie (`payload`) — sceptyk liczy SHA-256(payload + sól) bez rekonstrukcji.
    Wiersz bez niego (sprzed 043) ujawnia kolumny, z których odtwarza się payload v1.
    """
    record: dict[str, Any] = {
        "symbol": row.get("symbol"),
        "commitment": row.get("commitment_hash"),
        # Sól wychodzi na jaw dopiero TERAZ — to jest sens revealu.
        "salt": row.get("commitment_salt"),
        "predicted_trend": row.get("predicted_trend"),
        "predicted_target_price": row.get("predicted_target_price"),
        "price_at_prediction": row.get("price_at_prediction"),
        "actual_price_after_12h": row.get("actual_price_after_12h"),
        "is_trend_correct": row.get("is_trend_correct"),
    }
    payload = row.get("commitment_payload")
    if payload is not None:
        record["payload"] = payload
    return record


def _sweep_reveals(
    repository: RepositoryPort, publisher: AttestationPublisherPort
) -> int:
    """#16 — ujawnia WSZYSTKIE zapadłe commitmenty. Zwraca liczbę revealów.

    Sweep, nie „reveal przy rozliczeniu": predykcje symboli usuniętych z
    `config.toml` nigdy nie przeszłyby przez reflect, więc ich commitmenty
    zostałyby na zawsze nieujawnione. Dla sceptyka wygląda to jak ukrywanie
    nietrafień — a to niszczy cały sens weryfikowalnego track recordu. WIA-03:
    dlatego odczyt nie ma ani filtra watchlisty, ani okna `track_record_days`.

    `revealed_at` stemplujemy DOPIERO po udanej publikacji. Odwrotna kolejność
    zgubiłaby nietrafienie na zawsze: oznaczone jako ujawnione, nigdy ujawnione.
    Pusty wynik `publish_reveal` to według kontraktu portu porażka, nie sukces.
    Best-effort per wiersz: jeden błąd nie przerywa sweepu.
    """
    try:
        rows = repository.get_unrevealed_commitments()
    except Exception:
        logger.exception("Failed to read unrevealed commitments — sweep skipped")
        return 0

    revealed = 0
    for row in rows:
        try:
            location = publisher.publish_reveal(_reveal_record(row))
        except Exception:
            logger.exception("Failed to publish reveal for %s", row.get("id"))
            continue
        if not location:
            logger.warning("Reveal for %s was not published — left unrevealed", row.get("id"))
            continue
        try:
            repository.mark_commitment_revealed(str(row["id"]))
        except Exception:
            logger.exception("Failed to mark reveal for %s", row.get("id"))
            continue
        revealed += 1
    return revealed


def _build_consensus_shifts(
    repository: RepositoryPort,
    results: list[SymbolResult],
    *,
    now: datetime,
) -> list[tuple[str, ConsensusShift]]:
    """#8 — wykrywa zmiany nastawienia rady względem poprzedniego cyklu.

    Tłumienie stęchłych flipów: symbol bez ŚWIEŻEGO werdyktu (odcięty bramką
    volatility) jest pomijany — inaczej stary flip pokazywałby się w kółko.
    `gap_hours` liczone z timestampu poprzedniego cyklu; domena degraduje
    porównanie starsze niż 48h do STALE_COMPARISON, zamiast robić dramaturgię.
    Graceful w CALLERZE (styl repo): błąd odczytu → brak sekcji, cykl żyje.
    """
    shifts: list[tuple[str, ConsensusShift]] = []
    for r in results:
        if r.council_verdict is None:
            continue
        try:
            previous = repository.get_previous_verdict(r.symbol)
        except Exception:
            logger.exception("Failed to fetch previous verdict for %s", r.symbol)
            continue
        if previous is None:
            continue
        gap_hours = (now - previous.timestamp).total_seconds() / 3600.0
        shifts.append(
            (
                r.symbol,
                detect_shift(
                    previous.verdict,
                    r.council_verdict,
                    previous_sentiment=previous.sentiment_score,
                    current_sentiment=r.sentiment_score or 0.0,
                    gap_hours=gap_hours,
                ),
            )
        )
    return shifts


def _build_model_scorecard(
    settings: Settings, repository: RepositoryPort
) -> ModelScorecardView | None:
    """#12 — widok karty kondycji modelu. Flaga OFF / brak migracji 019 / błąd
    odczytu → None → sekcja sama się chowa, cykl leci dalej. ML-15: okno odczytu
    to `MODEL_SCORECARD_WINDOW_DAYS`, nie `track_record_days` — karta starsza niż
    próg świeżości zostaje widoczna z odznaką STALE."""
    if not settings.model_scorecard_enabled:
        return None
    try:
        rows = repository.get_recent_model_scorecards(MODEL_SCORECARD_WINDOW_DAYS)
    except Exception:
        logger.exception("Failed to fetch model scorecards")
        return None
    view = build_scorecard_view(rows)
    if view is None:
        return None
    # DAT-16: karta mówi też, ile sesji okna ma dane i które symbole nie mają rozliczeń.
    return dataclasses.replace(view, coverage=_data_coverage(settings, repository))


def _data_coverage(settings: Settings, repository: RepositoryPort) -> CoverageReport | None:
    """DAT-16 — pokrycie danych okna track recordu z darmowych odczytów
    (snapshoty cen, rozliczenia). Błąd → None: zdanie się chowa, cykl leci dalej."""
    try:
        return build_data_coverage(
            repository,
            symbols=settings.symbols,
            crypto_symbols=settings.crypto_symbols,
            days=settings.track_record_days,
            now=datetime.now(UTC),
        )
    except Exception:
        logger.exception("Data coverage failed — scorecards without the coverage line")
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
    # WIA-39: ranking kalibrujemy krzywą kierunkową — SIDEWAYS trafia za darmo.
    split = split_calibration_samples(rows)
    # WIA-38: pewność, która nie odróżnia trafień od pudeł, nie przestawia rankingu.
    discrimination = discrimination_evidence(split.directional_by_cycle, n_boot=0)
    if discrimination is not None and not confidence_reorders_ranking(discrimination):
        logger.warning(
            "Confidence AUC %.2f at n_eff≈%.0f carries no ranking information — "
            "signals ranked on raw confidence",
            discrimination.auc,
            discrimination.n_eff,
        )
        return None
    return directional_calibration_curve(rows)


def _calibration_samples(rows: list[dict[str, Any]]) -> list[tuple[float, bool]]:
    """Wiersze kalibracji → kierunkowe próbki `(confidence, was_correct)` (WIA-39:
    SIDEWAYS mają osobną kalibrację); niepełne pomijamy."""
    return split_calibration_samples(rows).directional


def _build_track_record(
    settings: Settings, repository: RepositoryPort, costs: PaperCosts | None = None
) -> tuple[EquityCurve | None, list[CalibrationBucket] | None, LessonsReport | None]:
    """T1/T2/T4 — krzywa kapitału, kalibracja, lessons (render-only, zero
    płatnych). Każda podsekcja flag-gated; wszystkie czytają zamknięte predykcje
    z okna track_record_days. Krzywa kapitału to portfel o równej wadze per
    cykl (WIA-17), nie szeregowe składanie równoległych trade'ów, po kosztach
    obrotu `costs` (WIA-50; None = brutto)."""
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
            equity_curve = portfolio_curve(tr_resolved, costs)
        if settings.lessons_enabled:
            lessons = build_lessons(tr_resolved)
    if settings.calibration_curve_enabled:
        try:
            samples = _calibration_samples(
                repository.get_resolved_for_calibration(settings.track_record_days)
            )
            calibration_buckets = reliability_curve(samples) if samples else None
        except Exception:
            logger.exception("Failed to compute calibration curve")
    return equity_curve, calibration_buckets, lessons


def _expire_stale_predictions(
    settings: Settings, repository: RepositoryPort, symbols: Sequence[str]
) -> int:
    """WIA-49: oznacza nierozliczone predykcje starsze niż okno oceny reflect.

    `reflect_node` nie oceni już predykcji starszej niż `reflection_max_age_hours`
    (WIA-04), więc bez jawnego wygaszenia zostałaby po cichu bez oceny — w mailu
    wyglądałoby to jak brak predykcji. 0 wyłącza granicę wieku, więc nic nie
    wygasa. Zapis idzie przez strażnika BEZ-08; błąd symbolu nie psuje cyklu."""
    max_age = settings.reflection_max_age_hours
    if max_age <= 0:
        return 0
    expired = 0
    for symbol in symbols:
        try:
            expired += repository.expire_unverified_predictions(symbol, max_age)
        except Exception:
            logger.exception("Failed to expire stale predictions of %s", symbol)
    if expired:
        logger.info("Expired %d unverified prediction(s) older than %dh", expired, max_age)
    return expired


def _build_symbol_track_record(
    settings: Settings, repository: RepositoryPort
) -> SymbolTrackRecordView | None:
    """WIA-35: trafność kierunkowa per symbol do chipu przy predykcji i sygnale.

    Ta sama flaga i to samo okno co krzywa kapitału; darmowy odczyt zamkniętych
    predykcji, zero płatnych wywołań. Błąd odczytu → mail bez chipów."""
    if not settings.equity_curve_enabled:
        return None
    try:
        resolved = parse_resolved_predictions(
            repository.get_resolved_predictions(settings.track_record_days)
        )
    except Exception:
        logger.exception("Failed to fetch resolved predictions for per-symbol track record")
        return None
    return symbol_track_record_view(resolved, settings.track_record_days)


def _build_track_record_stats(
    settings: Settings, repository: RepositoryPort, costs: PaperCosts | None = None
) -> TrackRecordStats | None:
    """Dowód trafności (T1), Brier / BSS (T2) i pokrycie przedziałów 80% (ML-34)
    pod sekcją Track Record.

    Te same flagi co podsekcje, które statystyki uzupełniają; pokrycie za
    `interval_coverage_enabled` (kolumny z migracji 032). Osobna funkcja,
    bo `_build_track_record` zwraca stały kontrakt trzech podsekcji. Każdy
    odczyt best-effort: błąd → brak danej statystyki, cykl leci dalej.
    `costs` (WIA-50) to te same założenia, z którymi `_build_track_record`
    liczy krzywą — jadą do etykiety „paper-PnL (koszt X bps)” i do porównania
    z koszykiem i IVV (WIA-21), które bez kosztów nie powstaje.
    """
    evidence = None
    scores = None
    benchmark = None
    if settings.equity_curve_enabled:
        resolved: list[ResolvedPrediction] = []
        try:
            resolved = parse_resolved_predictions(
                repository.get_resolved_predictions(settings.track_record_days)
            )
            evidence = hit_rate_evidence(resolved)
        except Exception:
            logger.exception("Failed to compute hit-rate evidence for track record")
        if resolved and costs is not None:
            benchmark = _benchmark_comparison(settings, repository, resolved, costs)
    sideways_scores = None
    if settings.calibration_curve_enabled:
        try:
            split = split_calibration_samples(
                repository.get_resolved_for_calibration(settings.track_record_days)
            )
            scores = probability_scores(split.directional)
            # WIA-39: SIDEWAYS osobno — przy małym n wystarczą n i Brier.
            sideways_scores = probability_scores(split.sideways)
        except Exception:
            logger.exception("Failed to compute probability scores for track record")
    coverage = None
    if settings.interval_coverage_enabled:
        try:
            measured = interval_coverage(
                repository.get_settled_moves(settings.track_record_days, with_intervals=True)
            )
            # Bez rozliczonego przedziału nie ma czego mierzyć — podsekcja się chowa.
            coverage = measured if measured.n else None
        except Exception:
            logger.exception("Failed to measure interval coverage for track record")
    paper_costs = costs if settings.equity_curve_enabled else None
    if all(part is None for part in (evidence, scores, coverage, paper_costs, benchmark)):
        return None
    return TrackRecordStats(
        hit_rate=evidence,
        scores=scores,
        coverage=coverage,
        costs=paper_costs,
        benchmark=benchmark,
        sideways_scores=sideways_scores,
    )


def _persist_quota_alerts(repository: RepositoryPort, alerts: list[QuotaAlert]) -> None:
    """Zapis alertów kwot, każdy osobno — błąd jednego nie gubi pozostałych."""
    for alert in alerts:
        try:
            repository.save_quota_alert(alert)
        except Exception:
            logger.exception("Failed to persist quota alert from %s", alert.source)


def _fetch_report_context(
    repository: RepositoryPort,
    quota_monitor: QuotaMonitor,
    all_symbols: list[str],
    *,
    resolved_since: datetime | None = None,
) -> tuple[
    dict[str, Any] | None,
    list[ResolvedPrediction] | None,
    list[QuotaAlert],
    list[InvestorHistory],
    list[PersonaTrackRecord],
]:
    """Best-effort dociągnięcie danych do raportu: accuracy, zamknięte predykcje,
    alerty kwot (persystencja bieżących + odczyt z ostatniego tygodnia), historia głosów rady
    i ranking wiarygodności person.
    Każdy odczyt osobno owinięty — błąd jednego nie blokuje pozostałych.

    `resolved_since` (WIA-19) — zamknięte predykcje to te ROZLICZONE od tej
    chwili (start cyklu), a nie utworzone w ostatnich 24 h. None = okno 24 h.

    MAIL-25: nieudany odczyt zamkniętych predykcji zwraca None, nie pustą listę —
    pusta lista znaczy w mailu „brak rozliczeń”, czego po awarii nie wiemy."""
    accuracy_stats: dict[str, Any] | None = None
    resolved: list[ResolvedPrediction] | None = None
    try:
        accuracy_stats = repository.get_accuracy_stats(ACCURACY_STATS_DAYS)
    except Exception:
        logger.exception("Failed to fetch accuracy stats — report without them")
    try:
        resolved_rows = (
            repository.get_recently_resolved_predictions(RESOLVED_PREDICTIONS_HOURS)
            if resolved_since is None
            else repository.get_recently_resolved_predictions(
                RESOLVED_PREDICTIONS_HOURS, since=resolved_since
            )
        )
        resolved = parse_resolved_predictions(resolved_rows)
    except Exception:
        logger.exception("Failed to fetch resolved predictions")
        resolved = None

    # Persystencja alertów z BIEŻĄCEGO cyklu (przed odczytem history, żeby też
    # tu trafiły) + odczyt z okna bannera.
    _persist_quota_alerts(repository, quota_monitor.alerts)
    try:
        recent_alerts = repository.get_recent_quota_alerts(hours=QUOTA_ALERT_LOOKBACK_HOURS)
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
    # → sekcja sama się chowa, cykl leci dalej. WIA-30: po dolnej granicy Wilsona.
    persona_track_record: list[PersonaTrackRecord] = []
    try:
        persona_track_record = rank_personas(
            repository.get_persona_track_record(PERSONA_TRACK_RECORD_DAYS), rank_by="wilson"
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


def _assess_personas(
    records: list[PersonaTrackRecord], stats: TrackRecordStats | None
) -> list[PersonaTrackRecord]:
    """WIA-22: werdykt istotności person (BH-FDR) na parametrach z okna scorecardu,
    bez migracji; WIA-30: ranking po dolnej granicy Wilsona na n_eff — tym samym
    przedziale, który pokazuje wiersz. Bez dowodu trafności okna
    (`equity_curve_enabled` off, brak danych) ranking zostaje bez werdyktu."""
    evidence = None if stats is None else stats.hit_rate
    if not records or evidence is None:
        return records
    return sort_personas(assess_against_track_record(records, evidence), "wilson")


def _without_owner_portfolio(context: ReportContext) -> ReportContext:
    """BEZ-01: kontekst maila dla subskrybenta, który nie jest właścicielem —
    bez pozycji, klastrów i cen pozycji spoza watchlisty. Bez portfela lead
    (a więc i preheader) nie dostaje tierów „Twojej pozycji”."""
    return dataclasses.replace(
        context,
        user_portfolio=None,
        portfolio_clusters=None,
        held_prices=None,
        position_risk=None,
        portfolio_fx=None,
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
    context: ReportContext,
    legacy_push_notifier: ReportNotifierPort | None = None,
    cycle_symbols: list[str] | None = None,
) -> bool:
    """Faza wysyłki: dociągnięcie danych raportu, zbudowanie raportu i rozesłanie
    — pojedynczo albo per subskrybent (raport tnięty do jego watchlisty). Cała
    faza jest best-effort: błąd wysyłki nie psuje exit-code cyklu.

    Zwraca, czy mail wyszedł bez błędu (NIEZ-22/NIEZ-29: step summary i `cycle_runs`).
    False także wtedy, gdy raport nie powstał, bo padło coś przed wysyłką.

    `cycle_symbols` (FIN-34) to symbole, które TEN cykl wycenił w grafie; None =
    cała watchlista. W dzień zamknięcia NYSE to samo krypto, więc trzymane akcje
    z watchlisty dostają darmowy `/quote` jak pozycje spoza niej (PORT-07).

    `context` przynosi sekcje policzone JESZCZE PRZED wysyłką (ryzyko makro,
    radar portfela, alfa, koszt cyklu); resztę pól dokłada ta funkcja z odczytów
    repozytorium. Wcześniej te same wartości szły tędy jako osobne argumenty
    tylko po to, by wylądować w wywołaniu `build_html_report`."""
    # NIEZ-03: alerty do tego miejsca zapisuje `_fetch_report_context`; te, które
    # powstaną później (np. Resend przy samej wysyłce), dopisuje `finally`.
    persisted_alerts = len(quota_monitor.alerts)
    email_delivered = True
    # Wyjątek PRZED wysyłką = raportu nie ma; po wysyłce wynik maila już jest znany.
    report_sent = False
    try:
        # WIA-19: za flagą sekcja zamkniętych predykcji = rozliczone w TYM cyklu
        # (`settled_at` ≥ start cyklu, migracja 029). Trigger stempluje zegarem
        # bazy, a start liczy zegar runnera: oba chodzą po NTP, a od startu do
        # pierwszego rozliczenia mijają sekundy, więc rozjazd zegarów nie gubi wierszy.
        resolved_since = started_at if settings.settled_at_windows_enabled else None
        (
            accuracy_stats,
            resolved,
            recent_alerts,
            council_history,
            persona_track_record,
        ) = _fetch_report_context(
            repository, quota_monitor, all_symbols, resolved_since=resolved_since
        )

        # T1/T2/T4: Track Record (render-only, zero płatnych). Każda podsekcja
        # flag-gated; wszystkie czytają zamknięte predykcje z okna track_record_days.
        # WIA-50: krzywa, jej etykieta i scorecard z tymi samymi kosztami.
        paper_costs = _paper_costs(settings)
        equity_curve, calibration_buckets, lessons = _build_track_record(
            settings, repository, paper_costs
        )
        track_record_stats = _build_track_record_stats(settings, repository, paper_costs)
        persona_track_record = _assess_personas(persona_track_record, track_record_stats)
        # #9: kubełki pod ranking sygnałów (osobna flaga, osobny odczyt).
        signal_calibration_buckets = _build_signal_calibration(settings, repository)
        # #4: dojrzałość cyklu + skład sektorowy pod powitanie "Dzień 1".
        maturity = _classify_cycle_maturity(results)
        sector_map = _sector_map(settings)
        sectors = _portfolio_sectors(results, sector_map)
        # #12: karta kondycji modelu (flaga + migracja 019).
        model_scorecard = _build_model_scorecard(settings, repository)
        # #15: realny portfel użytkownika (flaga + migracja 023).
        user_portfolio = _build_user_portfolio(settings)
        # PORT-07: pozycje spoza symboli cyklu dostają sam darmowy /quote.
        held_prices = _quote_held_positions(
            settings, user_portfolio, all_symbols if cycle_symbols is None else cycle_symbols
        )
        # PORT-11: kursy NBP do „P/L zł” (darmowe, poza bramką volatility).
        portfolio_fx = _collect_portfolio_fx(settings, user_portfolio, started_at)
        # PORT-04: ryzyko realnego portfela do sizingu (flaga + pozycje), wagi
        # w złotych tymi samymi kursami co raport.
        macro = context.macro_risk_report
        position_risk = _build_position_risk(
            settings,
            repository,
            user_portfolio,
            rates=effective_pln_rates(
                portfolio_fx, macro.polish_macro if macro is not None else None
            ),
        )
        # #8: zmiany nastawienia rady vs poprzedni cykl (flaga, bez migracji).
        consensus_shifts = (
            _build_consensus_shifts(repository, results, now=started_at)
            if settings.cycle_diff_enabled
            else []
        )

        # Kontekst od wołającego (ryzyko, alfa, koszt) + to, co dociągnęliśmy tutaj.
        report_context = dataclasses.replace(
            context,
            accuracy_stats=accuracy_stats,
            resolved_predictions=resolved,
            resolved_this_cycle=resolved_since is not None,
            resolved_unavailable=resolved is None,
            council_history=council_history,
            persona_track_record=persona_track_record,
            quota_alerts=recent_alerts,
            equity_curve=equity_curve,
            calibration_buckets=calibration_buckets,
            lessons=lessons,
            track_record_stats=track_record_stats,
            symbol_track_record=_build_symbol_track_record(settings, repository),
            signal_calibration_buckets=signal_calibration_buckets,
            cycle_maturity=maturity,
            portfolio_sectors=sectors,
            model_scorecard=model_scorecard,
            consensus_shifts=consensus_shifts,
            user_portfolio=user_portfolio,
            held_prices=held_prices,
            portfolio_fx=portfolio_fx,
            portfolio_clusters=_portfolio_clusters(settings),
            position_risk=position_risk,
            sector_map=sector_map,
            sector_peers=settings.sector_peers,
            display_timezone=settings.display_timezone,
        )

        def _build(
            symbols_filter: frozenset[str] | None = None,
            *,
            user_portfolio_in_text: bool = True,
            include_portfolio: bool = True,
        ) -> tuple[str, str]:
            # Raport subskrybenta różni się filtrem symboli (i bez portfela, gdy to
            # nie właściciel — BEZ-01); stary push — brakiem sekcji portfela
            # w tekście (kwoty P/L nie idą do komunikatorów).
            recipient_context = dataclasses.replace(
                report_context,
                symbols_filter=symbols_filter,
                user_portfolio_in_text=user_portfolio_in_text,
            )
            return build_html_report(
                results, started_at, duration,
                recipient_context
                if include_portfolio
                else _without_owner_portfolio(recipient_context),
            )

        html, text = _build()
        analyzed = sum(1 for r in results if r.status == "saved")
        # Subject prefix gdy są CRITICAL — wymusza zwrócenie uwagi w skrzynce.
        max_sev = quota_monitor.max_severity()
        prefix = "⚠️ [QUOTA] " if max_sev is QuotaSeverity.CRITICAL else ""
        # #1: temat z leadu, liczony nad PEŁNYMI wynikami cyklu (nie per
        # subskrybent) — sekcje są re-slicowane, ale temat jest współdzielony.
        # PORT-10: ten sam lead co w treści, więc też świadomy portfela; obcy
        # subskrybenci dostają jeden wspólny wariant bez portfela (BEZ-01).
        def _subject(context: ReportContext) -> str:
            return _build_subject(
                lead_items=build_cycle_lead(results, context),
                analyzed=analyzed,
                failures=failures,
                started_at=started_at,
                prefix=prefix,
                maturity=maturity,
                analyzed_total=len(results),
                display_timezone=settings.display_timezone,
            )

        subject = _subject(report_context)

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
            # BEZ-01: temat obcego subskrybenta z leadu bez portfela — inaczej
            # niósłby „… na AAPL (25% portfela)” właściciela.
            public_subject = _subject(_without_owner_portfolio(report_context))
            for sub in subscribers:
                owner = is_owner(sub, settings.digest_to_email)
                if not owner and user_portfolio is not None:
                    logger.info(
                        "Digest dla %s bez portfela właściciela (adres ≠ DIGEST_TO_EMAIL)",
                        masked_email(sub.email),
                    )
                sub_html, sub_text = _build(
                    symbols_for(sub, all_syms), include_portfolio=owner
                )
                try:
                    ResendNotifier(
                        api_key=settings.resend_api_key,
                        sender=settings.digest_from_email,
                        recipient=sub.email,
                        quota_monitor=quota_monitor,
                    ).send_report(
                        subject=subject if owner else public_subject,
                        html_body=sub_html,
                        plain_text=sub_text,
                    )
                except Exception:
                    logger.exception("Failed to send digest to %s", masked_email(sub.email))
                    email_delivered = False
        else:
            # NIEZ-02: wyjątek Resend nie może wyciąć scorecardu, sweepu i pusha.
            try:
                notifier.send_report(subject=subject, html_body=html, plain_text=text)
            except Exception:
                logger.exception("Failed to send the email digest — dalsze kroki idą")
                email_delivered = False
            if legacy_push_notifier is not None:
                _send_legacy_push(
                    legacy_push_notifier, subject, _build(user_portfolio_in_text=False)
                )
        report_sent = True

        # #10: publiczny scorecard — osobny publisher, osobna ścieżka.
        _publish_public_scorecard(
            settings, repository, build_public_scorecard_publisher(settings), paper_costs
        )

        # #16: sweep revealów PO wysyłce — publikacja to artefakt poboczny,
        # nie może opóźnić ani zablokować raportu.
        if settings.attestation_enabled:
            _sweep_reveals(repository, build_attestation_publisher(settings))

        # #5: push idzie RAZ, skrótem, poza gałęzią fan-outu — inaczej albo
        # ginie (subskrybenci), albo przekracza limit Telegrama (pełny raport).
        if settings.messenger_digest_enabled:
            _dispatch_push_digest(
                push_notifier=build_push_notifier(settings, quota_monitor),
                results=results,
                started_at=started_at,
                resolved=resolved or [],
                subject=subject,
                email_delivered=email_delivered,
                resolved_unavailable=resolved is None,
                display_timezone=settings.display_timezone,
            )
    except Exception:
        logger.exception("Failed to send report")
    finally:
        _persist_quota_alerts(repository, quota_monitor.alerts[persisted_alerts:])
    return report_sent and email_delivered


def _send_legacy_push(
    push: ReportNotifierPort, subject: str, report: tuple[str, str]
) -> None:
    """Stary push: pełny raport bez sekcji portfela. Best-effort — błąd kanału
    nie może zatrzymać scorecardu, sweepu ani zapisu alertów."""
    html, text = report
    try:
        push.send_report(subject=subject, html_body=html, plain_text=text)
    except Exception:
        logger.exception("Failed to send the legacy push report")


def _record_av_budget_alert(settings: Settings, quota_monitor: QuotaMonitor) -> None:
    """FIN-32: najgorszy przypadek doby AV ponad pojemność kluczy → WARNING w bannerze
    maila. Log z liczbami wystawia już walidator `Settings`; tu tylko treść maila
    po polsku (MAIL-15), bo banner pokazuje wyłącznie Fast Loop."""
    budget = settings.av_budget()
    if budget.ok:
        return
    quota_monitor.record(
        QuotaAlert(
            source="Alpha Vantage",
            severity=QuotaSeverity.WARNING,
            message=(
                f"Budżet Alpha Vantage nie domyka się: najgorszy przypadek doby to "
                f"{budget.worst_case} zapytań (cała watchlista za bramką), a klucze "
                f"dają {budget.capacity} (25 na klucz). Gdy za bramkę przejdzie "
                "więcej symboli, część dostanie pusty feed newsów i sentymentu."
            ),
            action=(
                "Dodaj klucz w ALPHA_VANTAGE_API_KEYS (darmowy na alphavantage.co) "
                "albo zmniejsz watchlistę; av_budget_strict = true zatrzyma start."
            ),
            occurred_at=datetime.now(UTC),
        )
    )


def _record_weekly_budget_alert(settings: Settings, quota_monitor: QuotaMonitor) -> None:
    """WIA-11: najgorszy tydzień Fast Loopa ponad `fast_loop_weekly_budget_usd` →
    WARNING w logu i w bannerze maila (wzorzec FIN-32). Fail-open: katalog person,
    którego nie da się wczytać, pomija samą kontrolę — przebieg idzie dalej."""
    budget_usd = settings.fast_loop_weekly_budget_usd
    if budget_usd is None:
        return
    try:
        personas = load_council_personas(Path(settings.council_personas_dir))
    except (OSError, ValueError) as exc:
        logger.warning(
            "Fast Loop weekly budget not checked: council personas unreadable (%s)", exc
        )
        return
    weekly = settings.fast_loop_weekly_budget(personas=min(len(personas), COUNCIL_MAX_PERSONAS))
    if weekly.ok:
        return
    logger.warning(
        "Fast Loop weekly budget exceeded: the worst case is %.2f USD a week (%d cycles × "
        "%.4f USD, every eligible symbol past the volatility gate at the FIN-12 ceiling), "
        "the budget fast_loop_weekly_budget_usd is %.2f USD. Raise the budget or lower "
        "max_paid_llm_calls_per_cycle, the watchlist or the cron cadence.",
        weekly.worst_case_usd,
        weekly.cycles_per_week,
        weekly.cycle_worst_usd,
        budget_usd,
    )
    week_usd = format_money_pl(Decimal(str(weekly.worst_case_usd)), "USD")
    cycle_usd = format_number_pl(Decimal(str(weekly.cycle_worst_usd)), signed=False, decimals=4)
    budget = format_money_pl(Decimal(str(budget_usd)), "USD")
    quota_monitor.record(
        QuotaAlert(
            source="FinOps",
            severity=QuotaSeverity.WARNING,
            message=(
                f"Budżet tygodniowy Fast Loopa nie domyka się: najgorszy tydzień to "
                f"{week_usd} ({weekly.cycles_per_week} cykli po ${cycle_usd}, cała "
                f"watchlista za bramką przy suficie cyklu), a budżet to {budget}. "
                "Szacunek, nie faktura."
            ),
            action=(
                "Podnieś fast_loop_weekly_budget_usd w config.toml albo obniż "
                "max_paid_llm_calls_per_cycle, watchlistę lub kadencję crona."
            ),
            occurred_at=datetime.now(UTC),
        )
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


# NIEZ-22: plik podsumowania kroku, który GitHub Actions pokazuje na stronie runu.
STEP_SUMMARY_ENV = "GITHUB_STEP_SUMMARY"


def append_step_summary(markdown: str, environ: Mapping[str, str] | None = None) -> None:
    """NIEZ-22: dopisuje `markdown` do pliku z env `GITHUB_STEP_SUMMARY`.

    Brak zmiennej (lokalny run) = nic. Błąd zapisu tylko loguje: summary to podgląd
    przebiegu, nie jego wynik. Wspólne dla Fast Loopa, Slow Loopa i shock watcha."""
    path = (os.environ if environ is None else environ).get(STEP_SUMMARY_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as summary:
            summary.write(markdown.rstrip("\n") + "\n\n")
    except OSError:
        logger.warning("Step summary not written to %s", STEP_SUMMARY_ENV, exc_info=True)


def write_step_summary(render: Callable[[], str]) -> None:
    """NIEZ-22: renderuje i dopisuje summary tylko w GitHub Actions (jest zmienna).

    Wyjątek renderera też tylko loguje — podgląd nie może wywrócić przebiegu."""
    if not os.environ.get(STEP_SUMMARY_ENV):
        return
    try:
        markdown = render()
    except Exception:
        logger.warning("Step summary not rendered", exc_info=True)
        return
    append_step_summary(markdown)


def _report_delivery(
    delivered: bool,
    *,
    notifier: ReportNotifierPort,
    subscribers: Sequence[Subscriber],
    settings: Settings,
) -> bool | None:
    """NIEZ-22: czy mail cyklu wyszedł; None = przebieg bez kanału e-mail
    (`NullNotifier` i brak fan-outu do subskrybentów) — nic nie miało wyjść."""
    fan_out = bool(subscribers) and bool(settings.resend_api_key)
    if not fan_out and isinstance(notifier, NullNotifier):
        return None
    return delivered


def _write_cycle_summary(
    *,
    results: Sequence[SymbolResult],
    telemetry: CycleTelemetry,
    cost: CycleCost | None,
    failures: int,
    duration_s: float,
    quota_alerts: Sequence[QuotaAlert],
    delivered: bool | None,
) -> None:
    """NIEZ-22: agregaty cyklu na publicznej stronie runu — bez symboli i treści błędów."""
    write_step_summary(
        lambda: render_cycle_summary_markdown(
            results,
            paid_calls=telemetry.snapshot().paid_calls,
            cycle_cost=cost,
            failures=failures,
            duration_s=duration_s,
            quota_alerts=quota_alerts,
            delivered=delivered,
        )
    )


# FIN-34: wspólny log pominięcia akcji Fast Loopa i shock watcha (`main_watch`).
STOCKS_SKIPPED_LOG = "%s — akcje pominięte (%d), krypto idzie normalnie: %s"


def nyse_closure_label(settings: Settings, now: datetime) -> str | None:
    """FIN-34: „święto NYSE (…)” albo „NYSE zamknięta (weekend)” w dniu `now`
    (data ET); None, gdy NYSE handluje albo pomijanie jest wyłączone
    (`nyse_closure_skip_enabled`). Ta sama reguła dla Fast Loopa i shock watcha."""
    if not settings.nyse_closure_skip_enabled:
        return None
    reason = closure_reason(to_nyse_time(now).date())
    if reason is None:
        return None
    return "NYSE zamknięta (weekend)" if reason == "weekend" else f"święto NYSE ({reason})"


def build_heartbeat(settings: Settings) -> HeartbeatPort:
    """NIEZ-13: ping zewnętrznego watchdoga; brak / pusty `HEARTBEAT_URL` = Null."""
    url = (settings.heartbeat_url or "").strip()
    if not url:
        return NullHeartbeat()
    return HttpHeartbeat(url)


def ping_heartbeat(settings: Settings, *, ok: bool) -> None:
    """NIEZ-13: jeden ping na koniec przebiegu. Błąd pingu tylko loguje (bez
    tracebacku — komunikat adaptera i tak nie niesie URL-a): watchdog to sygnał
    o przebiegu, nie jego wynik."""
    try:
        build_heartbeat(settings).ping(ok=ok)
    except Exception as exc:
        logger.warning("Heartbeat ping failed — run result unchanged: %s", exc)


def run_with_heartbeat(settings: Settings, run: Callable[[], int]) -> int:
    """NIEZ-13: przebieg entry pointa z pingiem heartbeatu na końcu (ok = exit 0).

    Wyjątek przebiegu pinguje porażkę i leci dalej — job w Actions i tak pada,
    a watchdog dowiaduje się od razu, nie po oknie checku. Wspólne dla Fast
    Loopa, Slow Loopa i shock watcha."""
    try:
        exit_code = run()
    except Exception:
        ping_heartbeat(settings, ok=False)
        raise
    ping_heartbeat(settings, ok=exit_code == 0)
    return exit_code


def main(
    settings: Settings | None = None, *, clock: Callable[[], datetime] = _utc_now
) -> int:
    """`clock` to zegar cyklu: dzień NYSE (FIN-34) i `started_at` raportu."""
    resolved = settings or Settings.from_env()
    configure_logging(resolved.secret_values())
    return run_with_heartbeat(resolved, lambda: _run_fast_loop(resolved, clock))


def prediction_columns_missing(
    repository: RepositoryPort, use_case: AnalyzeMarketUseCase
) -> bool:
    """FIN-05: preflight kolumn `prediction_logs` raz na cykl, przed płatnymi węzłami.

    Brak kolumny, którą ten config zapisuje (niezaaplikowana migracja), to ERROR
    z instrukcją i koniec cyklu — inaczej każdy symbol za bramką płaciłby za analizę,
    której zapis padnie. Niekonkluzywny preflight (sieć, timeout) to WARNING i cykl
    biegnie dalej: blip nie może wyłączyć cyklu."""
    try:
        missing = sorted(
            repository.missing_prediction_columns(use_case.required_prediction_columns())
        )
    except Exception as exc:
        logger.warning(
            "prediction_logs column preflight inconclusive (%s) — the cycle continues",
            type(exc).__name__,
        )
        return False
    if not missing:
        return False
    logger.error(
        "prediction_logs lacks columns this config writes: %s — run the 'DB Migrate' "
        "workflow (or switch their flags off); cycle stopped before any paid call",
        ", ".join(missing),
    )
    return True


def _run_fast_loop(settings: Settings, clock: Callable[[], datetime]) -> int:
    """Jeden przebieg Fast Loopa; kod wyjścia jak `main()` (0 / 1 / 2)."""
    if refuses_stub_writes(settings):
        return STUB_WRITES_REFUSED_EXIT_CODE
    # FIN-34: w dzień zamknięcia NYSE ceny akcji stoją — akcje nie idą do grafu.
    # Krypto handluje w święta, więc idzie normalnie; bez krypto nie ma czego
    # analizować, więc przebieg kończy się przed budową adapterów i bez maila.
    closure = nyse_closure_label(settings, clock())
    if closure is not None and not settings.crypto_symbols:
        logger.info(
            "%s — akcje pominięte, brak krypto: przebieg kończy się bez analizy i bez maila",
            closure,
        )
        return 0
    if closure is not None:
        logger.info(STOCKS_SKIPPED_LOG, closure, len(settings.symbols), settings.crypto_symbols)
    stock_symbols = [] if closure is not None else list(settings.symbols)
    warn_if_paid_calls_disabled(settings)
    # MAIL-42: przebieg (run, commit, link do logów) — raz na cykl, z env GHA.
    run_context = RunContext.from_environ(os.environ)
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
    _record_av_budget_alert(settings, quota_monitor)
    _record_weekly_budget_alert(settings, quota_monitor)
    repository = build_repository(settings)
    # FIN-13: jeden ledger tokenów na przebieg — adaptery zgłaszają, mail wycenia.
    usage_ledger = UsageLedger()
    use_case = build_use_case(
        settings,
        repository=repository,
        quota_monitor=quota_monitor,
        run_context=run_context,
        usage_ledger=usage_ledger,
    )
    if settings.db_writes_enabled and prediction_columns_missing(repository, use_case):
        return 1
    # Mail i stary push osobno: tekst pushu nie niesie sekcji portfela.
    notifier = build_notifier(settings, quota_monitor=quota_monitor, include_push=False)
    legacy_push_notifier = build_legacy_push_notifier(settings, quota_monitor=quota_monitor)
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

    started_at = clock()
    started_perf = time.perf_counter()
    logger.info(
        "Fast Loop start — symbols=%s crypto=%s",
        stock_symbols,
        settings.crypto_symbols,
    )

    # Dane: enrichment alpha (per-symbol) + krzywa rentowności (raz na cykl).
    # Wszystko domyślnie Null/None → zero I/O, póki flagi off.
    alpha_use_case = build_alpha_enrichment(settings, quota_monitor, repository=repository)
    collect_alpha = alpha_per_symbol_enabled(settings)
    yield_curve = build_macro_rates_port(settings).fetch_yield_curve()
    results: list[SymbolResult] = []
    unsupported = set(settings.symbols_unsupported_price)
    crypto_set = set(settings.crypto_symbols)
    etf_set = set(settings.symbols_etf)
    # Krypto dochodzi do puli iterowanych symboli — ma własny adapter ceny
    # (CoinGecko) i własny próg volatility (5%).
    all_symbols = list(settings.symbols) + list(settings.crypto_symbols)
    # FIN-34: symbole TEGO cyklu — w dzień zamknięcia NYSE samo krypto. Watchlista
    # (`all_symbols`) zostaje pełna dla kontekstu raportu (trafność, radar, portfel).
    cycle_symbols = [*stock_symbols, *settings.crypto_symbols]
    # Eligible — symbole, dla których pójdzie pełna analiza. Unsupported tickery
    # są oznaczone "ignored" przed pętlą, więc throttle policzy je poprawnie.
    eligible = [s for s in cycle_symbols if s not in unsupported]
    for skipped in (s for s in cycle_symbols if s in unsupported):
        logger.info(
            "%s: skipped — in SYMBOLS_UNSUPPORTED_PRICE (Finnhub free / EU-listed)",
            skipped,
        )
        results.append(_ignored_result(skipped))

    # ---- Risk Watch (PRZED pętlą — zasila też reżim rynku #7) ----
    macro_risk_report = _run_risk_watch(settings, repository)

    # #7: reżim rynku z overall_alert (zacieśnia próg + label do promptu). Off → neutral.
    # ALR-05: ten sam reżim trafia do nagłówka maila.
    regime_banners: list[RegimeBanner] = []
    regime_context, regime_multiplier = _resolve_regime(
        settings,
        macro_risk_report,
        yield_curve,
        repository=repository,
        cycle_at=started_at,
        on_banner=regime_banners.append,
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
        # ARCH-26: odznaki proweniencji liczą wiek danych od startu cyklu —
        # tego samego zegara, który mail drukuje w nagłówku.
        cycle_at=started_at,
        # EXP-27: cykl dwufazowy tylko za flagą; None = cykl jednofazowy jak dotąd.
        phase_two_feeds=(
            build_phase_two_feeds(settings, quota_monitor)
            if settings.two_phase_cycle_enabled
            else None
        ),
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

    persist_llm_calls(
        settings=settings,
        repository=repository,
        ledger=usage_ledger,
        run_context=run_context,
    )
    cost = cycle_cost(settings, use_case.telemetry, usage_ledger)
    # WIA-49: po pętli (reflect już ocenił, co mógł) — darmowy zapis flag.
    expired_unverified = _expire_stale_predictions(settings, repository, all_symbols)

    # ---- Wysyłka raportu ----
    delivered = _dispatch_reports(
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
        cycle_symbols=cycle_symbols,
        legacy_push_notifier=legacy_push_notifier,
        context=ReportContext(
            macro_risk_report=macro_risk_report,
            portfolio_risk_report=portfolio_risk_report,
            alpha_signals=alpha_signals,
            alpha_prices=alpha_prices,
            yield_curve=yield_curve,
            cycle_cost=cost,
            ml_model_loaded=use_case.ml_model_loaded,
            run_context=run_context,
            expired_unverified=expired_unverified,
            regime=regime_banners[-1] if regime_banners else None,
        ),
    )
    report_delivery = _report_delivery(
        delivered, notifier=notifier, subscribers=subscribers, settings=settings
    )
    run_duration = time.perf_counter() - started_perf
    _write_cycle_summary(
        results=results,
        telemetry=use_case.telemetry,
        cost=cost,
        failures=failures,
        duration_s=run_duration,
        quota_alerts=quota_monitor.alerts,
        delivered=report_delivery,
    )
    persist_cycle_run(
        settings=settings,
        repository=repository,
        run=CycleRun(
            loop="fast",
            run_id=run_context.run_id,
            # Decyzja właściciela: numer próby czyta entry point, `RunContext`
            # (i przepis predykcji WIA-20) zostaje bez zmian.
            run_attempt=run_attempt_from_environ(os.environ),
            started_at=started_at,
            duration_s=run_duration,
            eligible=len(eligible),
            saved=sum(1 for result in results if result.status == "saved"),
            failures=failures,
            paid_calls=dict(use_case.telemetry.snapshot().paid_calls),
            report_delivered=report_delivery,
        ),
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
