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
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

from src.application.agent_graph import AgentGraphDeps
from src.application.attestation_cycle import (
    CycleCommitmentCollector,
    attestation_cycle_id,
    seal_attestation_cycle,
)
from src.application.circuit_alerts import record_open_circuit_alerts
from src.application.cycle_telemetry import CycleTelemetry
from src.application.errors import PortError
from src.application.http_metering import HttpCallMeter
from src.application.metering import BudgetedLLMPort, MeteredLLMPort, PaidCallAllowance
from src.application.move_interval import (
    CALIBRATION_LOOKBACK_DAYS,
    calibration_windows,
)
from src.application.portfolio_fx import collect_pln_rates, effective_pln_rates, nbp_today
from src.application.ports import (
    AdvisoryCouncilPort,
    AnalystConsensusPort,
    AttestationLedgerPort,
    AttestationPublisherPort,
    CircuitState,
    CycleRun,
    DigestArchivePort,
    DigestRecord,
    EarningsCalendarPort,
    EmbeddingPort,
    FundamentalsPort,
    FxRatePort,
    HeartbeatPort,
    InsiderFlowPort,
    LLMPort,
    MacroRatesPort,
    MacroSeriesPort,
    MarketDataPort,
    NewsPort,
    OptionsFlowPort,
    PortfolioPositionsPort,
    QuotePort,
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
    build_cycle_signals,
    build_html_report,
)
from src.application.report_delivery import (
    ChannelDelivery,
    DeliveryOutcome,
    DeliveryStatus,
    ReportChannel,
)
from src.application.report_evidence import (
    PaperCosts,
)
from src.application.report_formatting import (
    DEFAULT_DISPLAY_TIMEZONE,
    build_sector_map,
    format_local,
)
from src.application.report_lessons import LessonsReport
from src.application.report_messenger import build_messenger_digest
from src.application.report_model_scorecard import (
    ModelScorecardView,
)
from src.application.report_models import ResolvedPrediction, TradeSignal
from src.application.report_onboarding import onboarding_subject
from src.application.report_ops import render_cycle_summary_markdown
from src.application.report_regime import RegimeBanner
from src.application.report_symbol_record import SymbolTrackRecordView
from src.application.report_track_record import TrackRecordStats
from src.application.run_context import RunContext
from src.application.schema_gate import evaluate_schema_gate
from src.application.source_health import (
    SourceHealth,
    assess_source_health,
    degraded_symbols,
)
from src.application.tools import build_research_tools
from src.application.track_record_window import (
    TrackRecordReads,
)
from src.application.usage_ledger import UsageLedger
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
from src.application.use_cases.assemble_report import (
    RESOLVED_PREDICTIONS_HOURS as RESOLVED_PREDICTIONS_HOURS,
)
from src.application.use_cases.assemble_report import (
    AssembledReport,
    ReportAssembler,
    ReportFlags,
    assess_personas,
    build_consensus_shifts,
    build_model_scorecard,
    build_signal_calibration,
    build_symbol_track_record,
    build_track_record,
    build_track_record_stats,
    classify_cycle_maturity,
    fetch_report_context,
    persist_quota_alerts,
    portfolio_sectors,
)
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
from src.application.use_cases.publish_public_scorecard import (
    PublicScorecardOptions,
    PublishPublicScorecardUseCase,
)
from src.application.use_cases.publish_status_page import (
    PublishStatusPageUseCase,
    StatusPageOptions,
)
from src.application.use_cases.reveal_attestations import RevealAttestationsUseCase
from src.application.use_cases.run_cycle import (
    CycleOptions,
    CycleOutcome,
    PhaseTwoFeeds,
    ProcessedSymbol,
    RunFastLoopCycle,
    classify_asset,
)
from src.config import Settings, load_evaluation_protocol
from src.domain.calibration_curve import (
    CalibrationBucket,
)
from src.domain.conformal import CalibrationPoint
from src.domain.cycle_maturity import CycleMaturity, SkipReason
from src.domain.digest_lead import LeadItem, lead_headline
from src.domain.earnings import EarningsEvent, earnings_this_week
from src.domain.equity_curve import EquityCurve
from src.domain.feature_requirements import FLAG_REQUIRED_MIGRATION
from src.domain.finops import CycleCost, estimate_cycle_cost
from src.domain.hedge import equity_hedge_symbols
from src.domain.insider_flow import InsiderLeadWindow
from src.domain.liveness import FAST_LOOP_TRADING_DAY_SLOT, LivenessExpectation
from src.domain.macro_rates import YieldCurveSnapshot, yield_curve_alert_level
from src.domain.macro_risk import MacroAlertLevel
from src.domain.macro_series import MacroSeriesSnapshot, macro_features, macro_series_votes
from src.domain.market_calendar import closure_reason, to_nyse_time
from src.domain.number_format import format_money_pl, format_number_pl
from src.domain.portfolio import PlnRates, Portfolio
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.recipe import PredictionRecipe, universe_version
from src.domain.regime import MarketRegime, RegimeDetector, apply_hysteresis, risk_off_change
from src.domain.scenarios import Scenario
from src.domain.subscriber import Subscriber, is_owner, masked_email, symbols_for
from src.domain.value_objects import AssetType, Threshold
from src.infrastructure.adapters._finnhub_common import FINNHUB_REQUESTS_PER_MINUTE
from src.infrastructure.adapters._http import (
    RateLimiter,
    failed_requests_by_source,
    metered_http,
    process_circuits,
    redact_secrets,
)
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
from src.infrastructure.logging_setup import configure_logging, symbol_context
from src.infrastructure.persona_loader import load_council_personas, personas_sha256
from src.infrastructure.recipe_fingerprints import ml_model_version, prompt_templates_sha256

# R2: domyślne scenariusze stress-testu (szok rynku jako ułamek). Szok stosowany
# do portfela równoważonego przez β-to-portfel każdego symbolu (domena liczy resztę).
DEFAULT_STRESS_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="Korekta rynku -10%", market_shock=-0.10),
    Scenario(name="Krach -20%", market_shock=-0.20),
    Scenario(name="Rajd +10%", market_shock=0.10),
)

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


def build_subscriber_notifier(
    settings: Settings,
    email: str,
    quota_monitor: QuotaMonitor | None = None,
) -> ReportNotifierPort:
    """Kanał e-mail jednego subskrybenta fan-outu (MAIL-11).

    Ten sam guard co `build_notifier`: bez `notifications_enabled` albo bez
    `RESEND_API_KEY` subskrybent dostaje `NullNotifier`, więc flaga z
    config.toml wyłącza KAŻDĄ wysyłkę maila, nie tylko tę do właściciela.
    """
    if settings.notifications_enabled and settings.resend_api_key and email:
        return ResendNotifier(
            api_key=settings.resend_api_key,
            sender=settings.digest_from_email,
            recipient=email,
            quota_monitor=quota_monitor,
        )
    return NullNotifier()


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
    lead_items: Sequence[LeadItem] = (),
    ranked_signals: Sequence[TradeSignal] = (),
    archive_url: str | None = None,
) -> bool:
    """#5 — wysyła skrót push RAZ, niezależnie od fan-outu per subskrybent.

    MAIL-20: `lead_items` i `ranked_signals` to lead i ranking sygnałów maila
    właściciela — push mówi, co najważniejsze, w tej samej kolejności.

    Gałąź U2 (subskrybenci) w ogóle pomijała `notifier.send_report`, więc push
    nie szedł, gdy ktokolwiek był zapisany. Skrót liczymy nad PEŁNYMI wynikami
    cyklu — push to jeden kanał, nie personalizowana watchlista.
    Best-effort: błąd kanału nie może wywalić cyklu. Zwraca, czy skrót wyszedł
    (ARCH-08: status kanału w `DeliveryOutcome`).
    """
    try:
        digest = build_messenger_digest(
            results,
            started_at,
            resolved,
            email_delivered=email_delivered,
            display_timezone=display_timezone,
            resolved_unavailable=resolved_unavailable,
            lead_items=lead_items,
            ranked_signals=ranked_signals,
            archive_url=archive_url,
        )
        push_notifier.send_report(
            subject=subject, html_body=digest, plain_text=digest
        )
    except Exception:
        logger.exception("Failed to send messenger digest")
        return False
    return True


def build_positions_port(settings: Settings) -> PortfolioPositionsPort:
    """#15 — realne pozycje użytkownika. Null gdy flaga off → sekcja się chowa,
    a warstwa ryzyka zostaje przy dotychczasowych równych wagach."""
    from src.infrastructure.adapters.positions_repo import (
        NullPositionsAdapter,
        SupabasePositionsAdapter,
    )

    if not settings.portfolio_positions_enabled:
        return NullPositionsAdapter()
    return SupabasePositionsAdapter(url=settings.supabase_url, key=settings.supabase_key)


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


def build_attestation_ledger(settings: Settings) -> AttestationLedgerPort:
    """WIA-15 — trwały ledger: checkout gałęzi `attestation-log` w katalogu z tymi samymi
    ścieżkami, do których dopisuje publisher (krok workflowu przed agentem, WIA-14)."""
    from src.infrastructure.adapters.attestation_ledger import FileAttestationLedger

    return FileAttestationLedger(
        commitment_path=settings.attestation_commitments_path,
        reveal_path=settings.attestation_reveals_path,
    )


def attestation_reveals(
    settings: Settings, repository: RepositoryPort
) -> RevealAttestationsUseCase:
    """#16/WIA-15 — sweep revealów ze stemplem z trwałego ledgera: `reconcile` na starcie
    runu stempluje revealy, które przeżyły push poprzedniego runu, `run` po raporcie
    tylko publikuje."""
    return RevealAttestationsUseCase(
        repository, build_attestation_publisher(settings), build_attestation_ledger(settings)
    )


def build_options_port(
    settings: Settings, quota_monitor: QuotaMonitor | None = None
) -> OptionsFlowPort | None:
    """#18 — port opcji dla grafu (fetch IV w `_predict_node`, za bramką
    volatility). None, gdy `options_flow_enabled` i `deribit_dvol_enabled` są off
    → sygnał nie powstaje. DAT-18: z `deribit_dvol_enabled` krypto idzie do Deribit
    DVOL, a akcje do Finnhuba (albo Null bez `options_flow_enabled`)."""
    from src.infrastructure.adapters.finnhub_options import (
        FinnhubOptionsAdapter,
        NullOptionsFlowAdapter,
    )

    if not settings.options_flow_enabled and not settings.deribit_dvol_enabled:
        return None
    equity: OptionsFlowPort = (
        FinnhubOptionsAdapter(
            api_key=settings.require_finnhub_api_key(),
            quota_monitor=quota_monitor,
            rate_limiter=FINNHUB_RATE_LIMITER,
        )
        if settings.options_flow_enabled
        else NullOptionsFlowAdapter()
    )
    if not settings.deribit_dvol_enabled:
        return equity
    from src.infrastructure.adapters.deribit_dvol import DeribitDvolAdapter
    from src.infrastructure.adapters.routing_options_flow import RoutingOptionsFlowPort

    return RoutingOptionsFlowPort(
        equity=equity, crypto=DeribitDvolAdapter(), crypto_symbols=settings.crypto_symbols
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
    """#10 — `FileWebPublisher` publicznego scorecardu (same agregaty) na jego
    własnej ścieżce; bez flagi Null. Prywatny digest nie idzie już na dysk —
    leży w prywatnym archiwum (MAIL-24, BEZ-27)."""
    from src.infrastructure.adapters.web_publisher import (
        FileWebPublisher,
        NullWebPublisher,
    )

    if not settings.public_scorecard_enabled:
        return NullWebPublisher()
    return FileWebPublisher(output_path=settings.public_scorecard_path)


def public_status_enabled(settings: Settings) -> bool:
    """NIEZ-33: strona statusu leży w katalogu scorecardu i wychodzi jego deployem Pages,
    więc ma sens tylko przy obu flagach."""
    return settings.public_status_page_enabled and settings.public_scorecard_enabled


def build_public_status_publisher(settings: Settings) -> WebPublisherPort:
    """NIEZ-33 — TRZECIA instancja `FileWebPublisher` z własną ścieżką (strona statusu
    w podkatalogu scorecardu; `badge.json` ląduje obok niej)."""
    from src.infrastructure.adapters.web_publisher import (
        FileWebPublisher,
        NullWebPublisher,
    )

    if not public_status_enabled(settings):
        return NullWebPublisher()
    return FileWebPublisher(output_path=settings.public_status_path)


def status_page_use_case(
    settings: Settings, repository: RepositoryPort, clock: Callable[[], datetime]
) -> PublishStatusPageUseCase:
    """NIEZ-33 — strona statusu z `cycle_runs`; „wstrzymany / spóźniony” z tej samej
    reguły co deadman (`liveness_fast_loop_max_gap_h`, dni handlowe NYSE)."""
    fast_loop = LivenessExpectation(
        "Fast Loop",
        settings.liveness_fast_loop_max_gap_h,
        trading_day_slot=FAST_LOOP_TRADING_DAY_SLOT,
    )
    return PublishStatusPageUseCase(
        repository,
        build_public_status_publisher(settings),
        StatusPageOptions(fast_loop=fast_loop),
        clock=clock,
    )


def _paper_costs(settings: Settings) -> PaperCosts:
    """WIA-50: założenia kosztowe paper-PnL z config.toml (akcje/ETF i krypto)."""
    return PaperCosts(
        cost_bps=settings.paper_cost_bps,
        crypto_cost_bps=settings.crypto_paper_cost_bps,
        crypto_symbols=frozenset(settings.crypto_symbols),
    )


def _public_scorecard_options(settings: Settings) -> PublicScorecardOptions:
    """ARCH-02: okno i źródła publicznego scorecardu z `Settings` (bez konwersji —
    use case liczy z nich dokładnie to, co liczył composition root)."""
    return PublicScorecardOptions(
        window_days=settings.track_record_days,
        settle_after_hours=settings.reflection_min_age_hours,
        symbols=settings.symbols,
        crypto_symbols=settings.crypto_symbols,
        load_protocol=load_evaluation_protocol,
    )


def public_scorecard_use_case(
    settings: Settings, repository: RepositoryPort, publisher: WebPublisherPort
) -> PublishPublicScorecardUseCase:
    """#10/WIA-81 — publikacja strony i `scorecard.json` z okna `track_record_days`.
    `main_scorecard` woła `publish` (błąd odczytu = exit 1), Fast Loop — `run`."""
    return PublishPublicScorecardUseCase(repository, publisher, _public_scorecard_options(settings))


def _publish_public_scorecard(
    settings: Settings,
    repository: RepositoryPort,
    publisher: WebPublisherPort,
    costs: PaperCosts | None = None,
    *,
    reads: TrackRecordReads | None = None,
) -> None:
    """#10 — publikacja strony w cyklu Fast Loopa za `public_scorecard_enabled`
    (`PublishPublicScorecardUseCase`, best-effort). `reads` (ARCH-16) — odczyty
    okna z tego dispatchu; None = use case czyta sam."""
    if not settings.public_scorecard_enabled:
        return
    PublishPublicScorecardUseCase(
        repository, publisher, _public_scorecard_options(settings), reads=reads
    ).run(costs)


def build_repository(settings: Settings) -> RepositoryPort:
    """Repozytorium Fast Loopa (także `main_watch` i `main_deadman`).

    BEZ-08: bez `db_writes_enabled` owinięte w `ReadOnlyRepository` — lokalny run
    czyta wspólną bazę, ale niczego do niej nie zapisuje."""
    repository = SupabaseRepository(
        url=settings.supabase_url,
        key=settings.supabase_key,
        council_votes_upsert_enabled=settings.council_votes_upsert_enabled,
        persona_vote_outcomes_enabled=settings.persona_vote_outcomes_enabled,
        crypto_symbols=settings.crypto_symbols,
        correction_category_enabled=settings.correction_category_enabled,
        stale_quote_enabled=settings.stale_quote_enabled,
        session_close_settlement_enabled=settings.session_close_settlement_enabled,
        rag_exact_search_enabled=settings.rag_exact_search_enabled,
        receipts_enabled=settings.receipts_enabled,
    )
    if settings.db_writes_enabled:
        return repository
    return ReadOnlyRepository(repository)


def build_digest_archive(settings: Settings) -> DigestArchivePort | None:
    """MAIL-24: prywatne archiwum digestów (Supabase Storage + indeks `digests`).

    Tylko za `digest_archive_enabled` i z `db_writes_enabled` — lokalny run tylko
    do odczytu (BEZ-08) nie wrzuca plików do wspólnego bucketu. None = mail bez
    linku i bez kanału archiwum, bajt w bajt jak przed MAIL-24."""
    if not (settings.digest_archive_enabled and settings.db_writes_enabled):
        return None
    from src.infrastructure.adapters.supabase_storage_archive import (
        SupabaseStorageDigestArchive,
    )

    return SupabaseStorageDigestArchive.connect(settings.supabase_url, settings.supabase_key)


def digest_cycle_id(started_at: datetime, run_context: RunContext | None) -> str:
    """MAIL-24: identyfikator cyklu w archiwum — start UTC i numer runu, bez PII
    (`20261015T0530Z-18234567890`); lokalny run to `…-local`."""
    run_id = run_context.run_id if run_context is not None else "local"
    return f"{started_at.astimezone(UTC):%Y%m%dT%H%MZ}-{run_id}"


def _archive_digest(
    archive: DigestArchivePort,
    *,
    cycle_id: str,
    html: str,
    subject: str,
    lead_items: Sequence[LeadItem],
    top_signals: Sequence[TradeSignal],
) -> tuple[str | None, ChannelDelivery]:
    """MAIL-24: HTML maila właściciela do prywatnego archiwum i wiersz indeksu.

    Zwraca podpisany URL (albo None) i status kanału. Błąd magazynu = mail bez
    linku; błąd samego indeksu nie zabiera linku (plik już leży w buckecie)."""
    try:
        url = archive.store(cycle_id, html)
    except PortError:
        logger.warning("Digest archive upload failed — mail goes without the link", exc_info=True)
        return None, ChannelDelivery(ReportChannel.ARCHIVE, DeliveryStatus.FAILED)
    try:
        archive.record(
            DigestRecord(
                cycle_id=cycle_id,
                subject=subject,
                lead=tuple(
                    {"icon": item.icon, "headline": item.headline, "symbol": item.symbol}
                    for item in lead_items
                ),
                top_signals=tuple(
                    {
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "confidence": signal.confidence,
                    }
                    for signal in top_signals
                ),
                sent_at=_utc_now(),
            )
        )
    except PortError:
        logger.warning("Digest archive index write failed — link kept", exc_info=True)
    return url, ChannelDelivery(ReportChannel.ARCHIVE, DeliveryStatus.SENT)


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
        # BEZ-14: uzasadnienia person w prompcie przewodniczącego w fence'u; off.
        fence_opinion_reasoning=settings.prompt_taint_fence_enabled,
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
        # FIN-09: jeden kalendarz na cykl, cała lista kluczy (rotacja przy soft-limicie)
        # i alert kwoty, gdy limit odbiją wszystkie.
        first_key, *reserve_keys = settings.alpha_vantage_api_keys
        earnings = AlphaVantageEarningsAdapter(
            api_key=first_key, reserve_keys=reserve_keys, quota_monitor=quota_monitor
        )
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
    crypto_set, etf_set = set(settings.crypto_symbols), set(settings.symbols_etf)
    return AlphaEnrichmentUseCase(
        insider_port=insider,
        earnings_port=earnings,
        options_port=options,
        social_port=social,
        analyst_port=analyst,
        # DAT-21: kapitalizacja z cache fundamentów (zero requestów AV) do progu
        # znaczącego zakupu insidera; bez repozytorium próg stały.
        market_cap_port=(
            build_fast_loop_fundamentals_port(repository)
            if settings.insider_flow_enabled and repository is not None
            else None
        ),
        asset_type_of=lambda symbol: classify_asset(symbol, crypto_set, etf_set),
    )


def build_macro_rates_port(settings: Settings) -> MacroRatesPort:
    """FRED yield-curve port — realny gdy włączone + jest klucz, inaczej Null."""
    if settings.yield_curve_enabled and settings.fred_api_key:
        from src.infrastructure.adapters.fred_client import FredClient

        return FredClient(api_key=settings.fred_api_key)
    from src.infrastructure.adapters.fred_client import NullMacroRatesAdapter

    return NullMacroRatesAdapter()


def build_macro_series_port(settings: Settings) -> MacroSeriesPort:
    """DAT-22: zmienne stanu z FRED — realny port przy `macro_series_enabled` i kluczu
    FRED, inaczej Null (zero I/O)."""
    if settings.macro_series_enabled and settings.fred_api_key:
        from src.infrastructure.adapters.fred_client import FredClient

        return FredClient(api_key=settings.fred_api_key)
    from src.infrastructure.adapters.fred_client import NullMacroSeriesAdapter

    return NullMacroSeriesAdapter()


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
        rows, lambda symbol: classify_asset(symbol, crypto_set, etf_set)
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
        # FIN-15: bez `symbols_unsupported_price` — te symbole nie dochodzą do grafu.
        unsupported = set(settings.symbols_unsupported_price)
        av_client = _alpha_vantage_client(
            settings,
            [s for s in (*settings.symbols, *settings.crypto_symbols) if s not in unsupported],
            quota_monitor,
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


def build_fast_loop_fundamentals_port(repository: RepositoryPort) -> FundamentalsPort:
    """TEST-06 (G70): fundamenty Fast Loopa wyłącznie z cache Supabase.

    Port idzie przez graf PRZED bramką volatility, więc delegat jest Null — cache
    miss to brak danych, nie płatny request Alpha Vantage (odświeża Slow Loop)."""
    return CachedFundamentalsAdapter(repo=repository, delegate=NullFundamentalsAdapter())


def build_use_case(
    settings: Settings,
    repository: RepositoryPort | None = None,
    quota_monitor: QuotaMonitor | None = None,
    run_context: RunContext | None = None,
    usage_ledger: UsageLedger | None = None,
) -> AnalyzeMarketUseCase:
    """DI Container — Use Case Fast Loopa na deps z `build_agent_graph_deps`."""
    return AnalyzeMarketUseCase(
        build_agent_graph_deps(settings, repository, quota_monitor, run_context, usage_ledger)
    )


def build_agent_graph_deps(
    settings: Settings,
    repository: RepositoryPort | None = None,
    quota_monitor: QuotaMonitor | None = None,
    run_context: RunContext | None = None,
    usage_ledger: UsageLedger | None = None,
) -> AgentGraphDeps:
    """Buduje wszystkie adaptery Fast Loopa w jeden obiekt `AgentGraphDeps`.

    TEST-09 (G56): publiczna, żeby testy sprawdzały pola deps (który port pod
    którym polem, atrapy vs produkcja), a nie tylko to, że DI nie rzuca.

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
    fundamentals_port = build_fast_loop_fundamentals_port(supabase_repo)
    # Routing: krypto → CoinGecko, reszta → Finnhub. Use case widzi jeden port.
    equity_quotes = FinnhubAdapter(
        api_key=settings.require_finnhub_api_key(),
        quota_monitor=quota_monitor,
        rate_limiter=FINNHUB_RATE_LIMITER,
    )
    market_port = RoutingMarketDataPort(
        equity=equity_quotes,
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
    # FIN-08: osobny próg tool-use krypto; brak = krypto bierze `tool_use_threshold`.
    crypto_tool_use_threshold = (
        Threshold(settings.crypto_tool_use_volatility_threshold)
        if tool_use_port is not None
        and settings.crypto_tool_use_volatility_threshold is not None
        else None
    )
    return AgentGraphDeps(
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
        crypto_tool_use_threshold=crypto_tool_use_threshold,
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
        # ML-31: cechy rynkowe w wektorze i wersja zestawu cech 4 (bez migracji).
        market_features_enabled=settings.market_features_enabled,
        # ML-11: σ symbolu do prognozy modelu w jednostkach σ (darmowy odczyt); off.
        ml_sigma_target_enabled=settings.ml_sigma_target_enabled,
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
        # ML-29: rozkład trendu w prompcie i kolumna (migracja 051) za flagą.
        trend_probabilities_enabled=settings.trend_probabilities_enabled,
        # BEZ-14: lekcja z reflect w prompcie predykcji w fence'u; off.
        prompt_taint_fence_enabled=settings.prompt_taint_fence_enabled,
        # #18: port opcji trafia DO GRAFU (fetch za bramką volatility), a nie tylko
        # do enrichmentu po grafie. Off → None → sygnał nie powstaje.
        options_port=build_options_port(settings, quota_monitor),
        # WIA-28: horyzont edge z kalendarza NYSE zamiast stałych 24 h; off.
        edge_calendar_horizon_enabled=settings.edge_calendar_horizon_enabled,
        # DAT-14: akcje i ETF biorą cenę z /quote razem z czasem notowania (ten sam
        # request); kolumna `stale_quote` (migracja 055) za flagą. Źródło ceny bez
        # czasu notowania → cena jak przed DAT-14 (wzorzec `DailyChangePort` watcha).
        quote_port=equity_quotes if isinstance(equity_quotes, QuotePort) else None,
        stale_quote_enabled=settings.stale_quote_enabled,
        # WIA-61: zamknięcie sesji z tego samego notowania obok ceny /quote (migracja 059).
        session_close_settlement_enabled=settings.session_close_settlement_enabled,
        # DAT-23: podejrzenie splitu z `pc` notowania (zapis w `corporate_actions`, 062).
        split_detection_enabled=settings.split_detection_enabled,
        # DAT-15: strażnik anomalii ceny (darmowy odczyt historii snapshotów); off.
        price_outlier_guard_enabled=settings.price_outlier_guard_enabled,
        # #16: publisher tylko gdy flaga on — inaczej nowe kolumny przed
        # migracją 024 dałyby PGRST204 i zabiły zapis całej predykcji. WIA-70:
        # flaga on bez kolumn 024/043 → cykl bez atestacji, predykcje zapisane.
        # WIA-60: commitmenty cyklu idą do kolektora — ledger dostaje jedną linię rootu
        # Merkle'a na cykl (`seal_attestation_cycle` po przebiegu symboli).
        attestation_publisher=(
            CycleCommitmentCollector(build_attestation_publisher(settings))
            if settings.attestation_enabled and supabase_repo.attestation_columns_available()
            else None
        ),
        telemetry=telemetry,
        # FIN-12: sufit płatnych wywołań LLM cyklu; None = bez limitu.
        paid_call_cap=settings.max_paid_llm_calls_per_cycle,
    )


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
    macro_series: MacroSeriesSnapshot | None = None,
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

    DAT-22: z `macro_series_enabled` percentyl VIX z roku i 5-sesyjna zmiana
    spreadu HY dokładają po głosie (`macro_series_votes`) — tak samo jak krzywa:
    CRITICAL daje RISK_OFF, pojedynczy ELEVATED tylko NEUTRAL.
    """
    if not (settings.regime_aware_enabled and macro_risk_report is not None):
        return "", 1.0

    # Głos krzywej wymaga JEDNOCZEŚNIE yield_curve_enabled i regime_aware_enabled.
    risk_signals: list[MacroAlertLevel] = []
    if settings.yield_curve_enabled and yield_curve is not None:
        risk_signals.append(yield_curve_alert_level(yield_curve.state()))
    if settings.macro_series_enabled and macro_series is not None:
        risk_signals.extend(macro_series_votes(macro_series))

    detector = RegimeDetector()
    raw_regime = detector.classify(macro_risk_report.overall_alert, risk_signals)
    regime, previous_regime = (
        _apply_regime_hysteresis(
            settings, repository, raw_regime, cycle_at or datetime.now(UTC)
        )
        if repository is not None
        else (raw_regime, None)
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
                # ALR-05 krok 2: zmiana względem poprzedniego cyklu tylko z histerezy.
                change=risk_off_change(previous_regime, regime),
            )
        )
    return regime_context, regime_multiplier


def _apply_regime_hysteresis(
    settings: Settings,
    repository: RepositoryPort,
    raw_regime: MarketRegime,
    cycle_at: datetime,
) -> tuple[MarketRegime, MarketRegime | None]:
    """ALR-03: reżim efektywny po histerezie — odczyt ostatniego stanu i zapis
    nowego wiersza `regime_history`, raz na cykl. Drugi element to reżim
    efektywny poprzedniego cyklu (ALR-05: lead przy zmianie); None, gdy
    histereza nie zadziałała (flaga off, błąd odczytu, brak historii).

    Flaga off albo błąd odczytu (np. brak migracji 033) → surowy reżim, czyli
    dokładnie zachowanie sprzed ALR-03; po błędzie odczytu nie zapisujemy, żeby
    stan bez historii nie wyzerował trwającej serii. Błąd zapisu tylko logujemy:
    następny cykl przeczyta starszy stan, co najwyżej opóźniając wyjście z
    RISK_OFF (wyższy próg — strona bezpieczna dla FinOps).
    """
    if not settings.regime_hysteresis_enabled:
        return raw_regime, None
    try:
        previous = repository.get_latest_regime_state()
    except Exception:
        logger.exception(
            "regime_history read failed — surowy reżim %s bez histerezy",
            raw_regime.value,
        )
        return raw_regime, None
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
    return state.regime, previous.regime if previous is not None else None


# ARCH-07: przebieg symboli cyklu żyje w `use_cases/run_cycle.py`; `_ProcessedSymbol`
# i `_process_one_symbol` zostają dla testów wiringu (NIEZ-21, BEZ-07, #6, #14).
_ProcessedSymbol = ProcessedSymbol


def _secret_redactor(secret_values: tuple[str, ...]) -> Callable[[str], str]:
    """BEZ-07: komunikat błędu symbolu bez URL-a z kluczem i bez dosłownych
    wartości sekretów z `Settings` (`redact_secrets`)."""

    def redact(text: str) -> str:
        return redact_secrets(text, secret_values)

    return redact


def _cycle_runner(
    *,
    use_case: AnalyzeMarketUseCase,
    alpha_use_case: AlphaEnrichmentUseCase,
    options: CycleOptions,
    secret_values: tuple[str, ...],
    phase_two_feeds: PhaseTwoFeeds | None = None,
) -> RunFastLoopCycle:
    """ARCH-07: use case przebiegu z hookami świata — `time.sleep` (czytany przy
    wywołaniu, więc patch `time.sleep` w testach działa), redakcja sekretów
    i kontekst logu symbolu."""
    return RunFastLoopCycle(
        use_case=use_case,
        alpha_use_case=alpha_use_case,
        options=options,
        sleep=time.sleep,
        redact=_secret_redactor(secret_values),
        symbol_scope=symbol_context,
        phase_two_feeds=phase_two_feeds,
    )


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
) -> ProcessedSymbol:
    """Jeden symbol przez `RunFastLoopCycle.process_one`. NIGDY nie rzuca."""
    return _cycle_runner(
        use_case=use_case,
        alpha_use_case=alpha_use_case,
        options=CycleOptions(
            regime_context=regime_context,
            regime_multiplier=regime_multiplier,
            collect_alpha=collect_alpha,
            earnings_gate_enabled=earnings_gate_enabled,
            alpha_fusion_enabled=alpha_fusion_enabled,
            cycle_at=cycle_at,
            crypto_symbols=frozenset(crypto_set),
            etf_symbols=frozenset(etf_set),
        ),
        secret_values=secret_values,
    ).process_one(symbol, peer_ctx)


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
) -> CycleOutcome:
    """ARCH-07: przebieg symboli cyklu (`RunFastLoopCycle`) z `CycleOptions`
    zmapowanymi z `Settings`; z `phase_two_feeds` (EXP-27,
    `two_phase_cycle_enabled`) cykl dwufazowy.
    Zwraca (wyniki, liczba_błędów, alpha_signals, alpha_prices)."""
    options = CycleOptions(
        regime_context=regime_context,
        regime_multiplier=regime_multiplier,
        collect_alpha=collect_alpha,
        earnings_gate_enabled=settings.earnings_calendar_enabled,
        alpha_fusion_enabled=settings.alpha_fusion_enabled,
        cycle_at=cycle_at,
        crypto_symbols=frozenset(crypto_set),
        etf_symbols=frozenset(etf_set),
        symbol_concurrency=settings.symbol_concurrency,
        contagion_enabled=settings.contagion_enabled,
        peer_groups=settings.peer_groups,
        throttle_seconds=settings.symbol_throttle_seconds,
    )
    return _cycle_runner(
        use_case=use_case,
        alpha_use_case=alpha_use_case,
        options=options,
        secret_values=settings.secret_values(),
        phase_two_feeds=phase_two_feeds,
    ).run(eligible)


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


def enabled_sources(settings: Settings) -> dict[str, bool]:
    """MAIL-21: które źródła ten przebieg w ogóle odpytuje — z tych samych flag
    i kluczy, z których fabryki budują adaptery. Embeddingi są zawsze z OpenAI,
    więc OpenAI pracuje przy każdym płatnym przebiegu (`build_embedding_adapter`)."""
    paid = settings.paid_calls_enabled
    llm_providers = {settings.llm_provider, settings.council_llm_provider or settings.llm_provider}
    return {
        "Finnhub": bool(settings.symbols),
        "CoinGecko": bool(settings.crypto_symbols),
        "Alpha Vantage": paid,
        "EDGAR": settings.insider_flow_enabled,
        "Reddit": settings.social_velocity_enabled,
        "FRED": bool(settings.fred_api_key)
        and (settings.yield_curve_enabled or settings.macro_series_enabled),
        "NBP": settings.nbp_enabled,
        "Anthropic": paid and "anthropic" in llm_providers,
        "OpenAI": paid,
    }


def _llm_source(settings: Settings) -> str:
    """Dostawca głównej analizy LLM — jego wina, gdy symbol dostał atrapę albo failover."""
    return "Anthropic" if settings.llm_provider == "anthropic" else "OpenAI"


def cycle_source_health(
    settings: Settings,
    results: list[SymbolResult],
    *,
    circuits: Iterable[CircuitState],
    http_calls: Mapping[str, Mapping[str, int]],
) -> tuple[SourceHealth, ...]:
    """MAIL-21: linijka stanu każdego źródła — obwody HTTP procesu (NIEZ-31), flagi
    jakości symboli i błędne odpowiedzi z licznika HTTP przebiegu (NIEZ-24)."""
    return assess_source_health(
        enabled=enabled_sources(settings),
        circuits=circuits,
        degraded=degraded_symbols(results, llm_source=_llm_source(settings)),
        failed_requests=failed_requests_by_source(http_calls),
    )


def _report_flags(settings: Settings) -> ReportFlags:
    """ARCH-06: `Settings` → flagi i okna sekcji raportu (bez konwersji wartości)."""
    return ReportFlags(
        track_record_days=settings.track_record_days,
        reflection_min_age_hours=settings.reflection_min_age_hours,
        symbols=settings.symbols,
        crypto_symbols=settings.crypto_symbols,
        equity_curve_enabled=settings.equity_curve_enabled,
        lessons_enabled=settings.lessons_enabled,
        calibration_curve_enabled=settings.calibration_curve_enabled,
        calibrated_confidence_enabled=settings.calibrated_confidence_enabled,
        isotonic_calibration_enabled=settings.isotonic_calibration_enabled,
        interval_coverage_enabled=settings.interval_coverage_enabled,
        model_scorecard_enabled=settings.model_scorecard_enabled,
        cycle_diff_enabled=settings.cycle_diff_enabled,
        settled_at_windows_enabled=settings.settled_at_windows_enabled,
        shock_alerts_enabled=settings.shock_alerts_enabled,
        persona_brier_ranking_enabled=settings.persona_brier_ranking_enabled,
        council_redundancy_enabled=settings.council_redundancy_enabled,
        receipts_enabled=settings.receipts_enabled,
    )


# ARCH-06: sekcje raportu żyją w `use_cases/assemble_report.py`; poniższe aliasy
# przyjmują `Settings` (wołają je testy wiringu i `main_watch`).
_fetch_report_context = fetch_report_context
_persist_quota_alerts = persist_quota_alerts
_assess_personas = assess_personas
_build_consensus_shifts = build_consensus_shifts
_classify_cycle_maturity = classify_cycle_maturity
_portfolio_sectors = portfolio_sectors


def _build_track_record(
    settings: Settings,
    repository: RepositoryPort,
    costs: PaperCosts | None = None,
    *,
    reads: TrackRecordReads | None = None,
) -> tuple[EquityCurve | None, list[CalibrationBucket] | None, LessonsReport | None]:
    """T1/T2/T4 z `Settings` — `assemble_report.build_track_record`."""
    return build_track_record(_report_flags(settings), repository, costs, reads=reads)


def _build_track_record_stats(
    settings: Settings,
    repository: RepositoryPort,
    costs: PaperCosts | None = None,
    *,
    reads: TrackRecordReads | None = None,
) -> TrackRecordStats | None:
    """Statystyki Track Recordu z `Settings` — `assemble_report.build_track_record_stats`."""
    return build_track_record_stats(_report_flags(settings), repository, costs, reads=reads)


def _build_symbol_track_record(
    settings: Settings, repository: RepositoryPort, *, reads: TrackRecordReads | None = None
) -> SymbolTrackRecordView | None:
    """WIA-35 z `Settings` — `assemble_report.build_symbol_track_record`."""
    return build_symbol_track_record(_report_flags(settings), repository, reads=reads)


def _build_signal_calibration(
    settings: Settings, repository: RepositoryPort, *, reads: TrackRecordReads | None = None
) -> list[CalibrationBucket] | None:
    """#9 z `Settings` — `assemble_report.build_signal_calibration`."""
    return build_signal_calibration(_report_flags(settings), repository, reads=reads)


def _build_model_scorecard(
    settings: Settings, repository: RepositoryPort, *, reads: TrackRecordReads | None = None
) -> ModelScorecardView | None:
    """#12 z `Settings` — `assemble_report.build_model_scorecard`."""
    return build_model_scorecard(_report_flags(settings), repository, reads=reads)


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
        # MAIL-24: podpisany URL otwiera mail właściciela z portfelem.
        archive_url=None,
    )


# `date.weekday()` poniedziałku — dzień sekcji „📅 Ten tydzień” (DAT-20).
_MONDAY = 0


def _earnings_this_week(
    settings: Settings, started_at: datetime
) -> tuple[EarningsEvent, ...] | None:
    """DAT-20: w poniedziałek (strefa wyświetlania) wyniki spółek z watchlisty do
    niedzieli z kalendarza Finnhub — jedno zapytanie na watchlistę, darmowe, przez
    wspólny limiter. Bez flagi albo w inny dzień None (sekcji nie ma, zero ruchu)."""
    if not settings.finnhub_earnings_calendar_enabled:
        return None
    today = started_at.astimezone(ZoneInfo(settings.display_timezone)).date()
    if today.weekday() != _MONDAY:
        return None
    from src.infrastructure.adapters.finnhub_earnings import FinnhubEarningsCalendarAdapter

    stocks = [
        symbol
        for symbol in settings.symbols
        if symbol not in settings.symbols_etf and symbol not in settings.crypto_symbols
    ]
    calendar = FinnhubEarningsCalendarAdapter(
        settings.require_finnhub_api_key(),
        stocks,
        rate_limiter=FINNHUB_RATE_LIMITER,
        today=lambda: today,
    )
    return earnings_this_week((calendar.get_next_earnings(s) for s in stocks), today)


def _report_inputs(
    settings: Settings,
    repository: RepositoryPort,
    context: ReportContext,
    *,
    watchlist: list[str],
    started_at: datetime,
) -> ReportContext:
    """ARCH-06: część kontekstu raportu, którą składa composition root — portfel
    właściciela z adapterów (pozycje, `/quote`, kursy NBP) i konfiguracja
    prezentacji. Resztę sekcji dokłada `ReportAssembler` z repozytorium.

    `watchlist` (FIN-34) to symbole wycenione w tym cyklu: pozycje spoza niej
    dostają sam darmowy `/quote` (PORT-07)."""
    # #15: realny portfel użytkownika (flaga + migracja 023).
    user_portfolio = _build_user_portfolio(settings)
    held_prices = _quote_held_positions(settings, user_portfolio, watchlist)
    # PORT-11: kursy NBP do „P/L zł” (darmowe, poza bramką volatility).
    portfolio_fx = _collect_portfolio_fx(settings, user_portfolio, started_at)
    # PORT-04: ryzyko realnego portfela do sizingu (flaga + pozycje), wagi
    # w złotych tymi samymi kursami co raport.
    macro = context.macro_risk_report
    return dataclasses.replace(
        context,
        user_portfolio=user_portfolio,
        held_prices=held_prices,
        portfolio_fx=portfolio_fx,
        portfolio_clusters=_portfolio_clusters(settings),
        position_risk=_build_position_risk(
            settings,
            repository,
            user_portfolio,
            rates=effective_pln_rates(
                portfolio_fx, macro.polish_macro if macro is not None else None
            ),
        ),
        sector_map=_sector_map(settings),
        sector_peers=settings.sector_peers,
        display_timezone=settings.display_timezone,
        earnings_this_week=_earnings_this_week(settings, started_at),
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
    subscribers: list[Subscriber],
    all_symbols: list[str],
    context: ReportContext,
    legacy_push_notifier: ReportNotifierPort | None = None,
    cycle_symbols: list[str] | None = None,
) -> bool:
    """Faza wysyłki: złożenie raportu (`ReportAssembler`) i rozesłanie
    (`deliver_report`) — pojedynczo albo per subskrybent (raport tnięty do jego
    watchlisty). Cała faza jest best-effort: błąd wysyłki nie psuje exit-code cyklu.

    Zwraca, czy mail wyszedł bez błędu (NIEZ-22/NIEZ-29: step summary i `cycle_runs`).
    False także wtedy, gdy raport nie powstał, bo padło coś przed wysyłką.

    `cycle_symbols` (FIN-34) to symbole, które TEN cykl wycenił w grafie; None =
    cała watchlista. W dzień zamknięcia NYSE to samo krypto, więc trzymane akcje
    z watchlisty dostają darmowy `/quote` jak pozycje spoza niej (PORT-07).

    `context` przynosi sekcje policzone JESZCZE PRZED wysyłką (ryzyko makro,
    radar portfela, alfa, koszt cyklu); portfel właściciela i konfigurację
    prezentacji dokłada `_report_inputs`, a sekcje z odczytów repozytorium
    `ReportAssembler` (ARCH-06)."""
    # NIEZ-03: alerty do tego miejsca zapisuje `fetch_report_context`; te, które
    # powstaną później (np. Resend przy samej wysyłce), zapisuje `deliver_report`.
    persisted_alerts = len(quota_monitor.alerts)

    def assemble() -> AssembledReport:
        # WIA-50: krzywa, jej etykieta i scorecard z tymi samymi kosztami.
        assembler = ReportAssembler(repository, _report_flags(settings), _paper_costs(settings))
        return assembler.assemble(
            results=results,
            started_at=started_at,
            quota_monitor=quota_monitor,
            all_symbols=all_symbols,
            context=_report_inputs(
                settings,
                repository,
                context,
                watchlist=all_symbols if cycle_symbols is None else cycle_symbols,
                started_at=started_at,
            ),
        )

    return deliver_report(
        assemble,
        results=results,
        started_at=started_at,
        duration=duration,
        failures=failures,
        settings=settings,
        repository=repository,
        quota_monitor=quota_monitor,
        notifier=notifier,
        subscribers=subscribers,
        all_symbols=all_symbols,
        legacy_push_notifier=legacy_push_notifier,
        persisted_alerts=persisted_alerts,
    ).delivered


def deliver_report(
    assemble: Callable[[], AssembledReport],
    *,
    results: list[SymbolResult],
    started_at: datetime,
    duration: float,
    failures: int,
    settings: Settings,
    repository: RepositoryPort,
    quota_monitor: QuotaMonitor,
    notifier: ReportNotifierPort,
    subscribers: list[Subscriber],
    all_symbols: list[str],
    legacy_push_notifier: ReportNotifierPort | None = None,
    persisted_alerts: int = 0,
) -> DeliveryOutcome:
    """ARCH-08: render i rozesłanie złożonego raportu; wynik kanał po kanale.

    Kolejność: prywatne archiwum (MAIL-24, za flagą — link trafia do maila
    i pushu), mail — pojedynczy z opcjonalnym starym pushem albo fan-out per
    subskrybent (U2) — potem artefakty poboczne: publiczny scorecard (#10), sweep
    revealów (#16) i skrót push (#5). Błąd kanału go
    oznacza i nie zatrzymuje kolejnych; wyjątek spoza kanałów kończy fazę
    (przed końcem wysyłki `report_sent` zostaje False). Nigdy nie rzuca.

    Alerty kwot od indeksu `persisted_alerts` — powstałe po zapisie
    w `fetch_report_context`, np. przy samej wysyłce — zapisuje PO wysyłce."""
    channels: list[ChannelDelivery] = []
    report_sent = False
    try:
        # Złożenie raportu w tym samym `try` co wysyłka: wyjątek PRZED wysyłką =
        # raportu nie ma (report_sent False), a alerty i tak zapisuje `finally`.
        report = assemble()
        context = report.context
        html, text = _render_report(context, results, started_at, duration)
        # Subject prefix gdy są CRITICAL — wymusza zwrócenie uwagi w skrzynce.
        max_sev = quota_monitor.max_severity()
        prefix = "⚠️ [QUOTA] " if max_sev is QuotaSeverity.CRITICAL else ""

        # #1: temat z leadu, liczony nad PEŁNYMI wynikami cyklu (nie per
        # subskrybent) — sekcje są re-slicowane, ale temat jest współdzielony.
        # PORT-10: ten sam lead co w treści, więc też świadomy portfela; obcy
        # subskrybenci dostają jeden wspólny wariant bez portfela (BEZ-01).
        def subject_of(recipient_context: ReportContext) -> str:
            return _build_subject(
                lead_items=build_cycle_lead(results, recipient_context),
                analyzed=sum(1 for r in results if r.status == "saved"),
                failures=failures,
                started_at=started_at,
                prefix=prefix,
                maturity=recipient_context.cycle_maturity or CycleMaturity.STEADY_STATE,
                analyzed_total=len(results),
                display_timezone=settings.display_timezone,
            )

        subject = subject_of(context)
        # MAIL-24: archiwum dostaje mail właściciela BEZ linku do samego siebie,
        # potem mail i push renderują się z podpisanym URL-em.
        archive = build_digest_archive(settings)
        if archive is not None:
            archive_url, archive_channel = _archive_digest(
                archive,
                cycle_id=digest_cycle_id(started_at, context.run_context),
                html=html,
                subject=subject,
                lead_items=build_cycle_lead(results, context),
                top_signals=build_cycle_signals(results, context),
            )
            channels.append(archive_channel)
            if archive_url is not None:
                context = dataclasses.replace(context, archive_url=archive_url)
                html, text = _render_report(context, results, started_at, duration)

        # U2: jeśli są subskrybenci, każdy dostaje raport TNIĘTY do swojej
        # watchlisty (analiza poszła RAZ nad unią — to tylko re-slicing). Bez
        # subskrybentów → dotychczasowa pojedyncza wysyłka.
        if subscribers and settings.resend_api_key:
            all_syms = frozenset(all_symbols)
            # BEZ-01: temat obcego subskrybenta z leadu bez portfela — inaczej
            # niósłby „… na AAPL (25% portfela)” właściciela.
            public_subject = subject_of(_without_owner_portfolio(context))
            for sub in subscribers:
                owner = is_owner(sub, settings.digest_to_email)
                if not owner and context.user_portfolio is not None:
                    logger.info(
                        "Digest dla %s bez portfela właściciela (adres ≠ DIGEST_TO_EMAIL)",
                        masked_email(sub.email),
                    )
                sub_html, sub_text = _render_report(
                    context,
                    results,
                    started_at,
                    duration,
                    symbols_filter=symbols_for(sub, all_syms),
                    include_portfolio=owner,
                )
                channels.append(
                    _send_subscriber_digest(
                        settings,
                        sub,
                        quota_monitor,
                        subject=subject if owner else public_subject,
                        report=(sub_html, sub_text),
                    )
                )
        else:
            # NIEZ-02: wyjątek Resend nie może wyciąć scorecardu, sweepu i pusha.
            channels.append(_send_email_digest(notifier, subject, (html, text)))
            if legacy_push_notifier is not None:
                channels.append(
                    _send_legacy_push(
                        legacy_push_notifier,
                        subject,
                        _render_report(
                            context, results, started_at, duration, user_portfolio_in_text=False
                        ),
                    )
                )
        report_sent = True

        # #10: publiczny scorecard — osobny publisher, osobna ścieżka.
        _publish_public_scorecard(
            settings,
            repository,
            build_public_scorecard_publisher(settings),
            report.costs,
            reads=report.reads,
        )

        # #16: sweep revealów PO wysyłce — publikacja to artefakt poboczny,
        # nie może opóźnić ani zablokować raportu. WIA-15: bez stempla `revealed_at` —
        # ten przychodzi w następnym runie, gdy reveal przeżyje push na `attestation-log`.
        if settings.attestation_enabled:
            attestation_reveals(settings, repository).run()

        # #5: push idzie RAZ, skrótem, poza gałęzią fan-outu — inaczej albo
        # ginie (subskrybenci), albo przekracza limit Telegrama (pełny raport).
        if settings.messenger_digest_enabled:
            push_notifier = build_push_notifier(settings, quota_monitor)
            resolved = context.resolved_predictions
            pushed = _dispatch_push_digest(
                push_notifier=push_notifier,
                results=results,
                started_at=started_at,
                resolved=resolved or [],
                subject=subject,
                email_delivered=DeliveryOutcome(True, tuple(channels)).email_delivered,
                resolved_unavailable=resolved is None,
                display_timezone=settings.display_timezone,
                lead_items=build_cycle_lead(results, context),
                ranked_signals=build_cycle_signals(results, context),
                archive_url=context.archive_url,
            )
            channels.append(
                ChannelDelivery(ReportChannel.PUSH_DIGEST, _channel_status(push_notifier, pushed))
            )
    except Exception:
        logger.exception("Failed to send report")
    finally:
        _persist_quota_alerts(repository, quota_monitor.alerts[persisted_alerts:])
    return DeliveryOutcome(report_sent=report_sent, channels=tuple(channels))


def _render_report(
    context: ReportContext,
    results: list[SymbolResult],
    started_at: datetime,
    duration: float,
    *,
    symbols_filter: frozenset[str] | None = None,
    user_portfolio_in_text: bool = True,
    include_portfolio: bool = True,
) -> tuple[str, str]:
    """HTML i tekst raportu dla jednego odbiorcy. Raport subskrybenta różni się
    filtrem symboli (i bez portfela, gdy to nie właściciel — BEZ-01); stary
    push — brakiem sekcji portfela w tekście (kwoty P/L nie idą do komunikatorów)."""
    recipient_context = dataclasses.replace(
        context,
        symbols_filter=symbols_filter,
        user_portfolio_in_text=user_portfolio_in_text,
        # MAIL-24: stary push bez kwot portfela nie dostaje też linku do archiwum,
        # które je ma (obcego subskrybenta tnie `_without_owner_portfolio`).
        archive_url=context.archive_url if user_portfolio_in_text else None,
    )
    return build_html_report(
        results, started_at, duration,
        recipient_context
        if include_portfolio
        else _without_owner_portfolio(recipient_context),
    )


def _channel_status(channel: ReportNotifierPort, sent: bool) -> DeliveryStatus:
    """Wysłane przez `NullNotifier` (kanał wyłączony / bez sekretów) to SKIPPED."""
    if not sent:
        return DeliveryStatus.FAILED
    return DeliveryStatus.SKIPPED if isinstance(channel, NullNotifier) else DeliveryStatus.SENT


def _send_email_digest(
    notifier: ReportNotifierPort, subject: str, report: tuple[str, str]
) -> ChannelDelivery:
    """Pojedynczy mail właściciela (bez fan-outu)."""
    html, text = report
    try:
        notifier.send_report(subject=subject, html_body=html, plain_text=text)
    except Exception:
        logger.exception("Failed to send the email digest — dalsze kroki idą")
        return ChannelDelivery(ReportChannel.EMAIL, DeliveryStatus.FAILED)
    return ChannelDelivery(ReportChannel.EMAIL, _channel_status(notifier, sent=True))


def _send_subscriber_digest(
    settings: Settings,
    subscriber: Subscriber,
    quota_monitor: QuotaMonitor,
    *,
    subject: str,
    report: tuple[str, str],
) -> ChannelDelivery:
    """U2: mail jednego subskrybenta; `recipient` to adres zamaskowany jak w logu."""
    html, text = report
    recipient = masked_email(subscriber.email)
    try:
        channel = build_subscriber_notifier(settings, subscriber.email, quota_monitor)
        channel.send_report(subject=subject, html_body=html, plain_text=text)
    except Exception:
        logger.exception("Failed to send digest to %s", recipient)
        return ChannelDelivery(ReportChannel.EMAIL, DeliveryStatus.FAILED, recipient)
    return ChannelDelivery(ReportChannel.EMAIL, _channel_status(channel, sent=True), recipient)


def _send_legacy_push(
    push: ReportNotifierPort, subject: str, report: tuple[str, str]
) -> ChannelDelivery:
    """Stary push: pełny raport bez sekcji portfela. Best-effort — błąd kanału
    nie może zatrzymać scorecardu, sweepu ani zapisu alertów."""
    html, text = report
    try:
        push.send_report(subject=subject, html_body=html, plain_text=text)
    except Exception:
        logger.exception("Failed to send the legacy push report")
        return ChannelDelivery(ReportChannel.LEGACY_PUSH, DeliveryStatus.FAILED)
    return ChannelDelivery(ReportChannel.LEGACY_PUSH, _channel_status(push, sent=True))


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


# NIEZ-12: adnotacja GitHub Actions widoczna w UI przebiegu (exit code zostaje 0).
UNDELIVERED_REPORT_ANNOTATION = "::error title=StockAgent::Raport nie został dostarczony"


def flag_undelivered_report(
    quota_monitor: QuotaMonitor,
    *,
    now: datetime,
    repository: RepositoryPort | None = None,
) -> None:
    """NIEZ-12: raport z tego przebiegu nie dotarł — adnotacja `::error` na stdout
    i alert CRITICAL (banner następnego maila, push real-time, gdy włączony).

    Exit code zostaje 0 (CLAUDE.md, Resilience): niedostarczenie nie jest awarią
    cyklu, tylko wymaga uwagi właściciela. Z `repository` alert jest zapisany od
    razu (po fazie wysyłki nikt go już nie zapisze); Slow Loop zapisuje alerty sam."""
    print(UNDELIVERED_REPORT_ANNOTATION, flush=True)
    alert = QuotaAlert(
        source="Raport",
        severity=QuotaSeverity.CRITICAL,
        message="Raport z tego przebiegu nie dotarł do właściciela.",
        action=(
            "Sprawdź log przebiegu w GitHub Actions: klucz Resend, domenę nadawcy "
            "i kanały push."
        ),
        occurred_at=now,
    )
    quota_monitor.record(alert)
    if repository is not None:
        _persist_quota_alerts(repository, [alert])


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
            http_calls=telemetry.snapshot().http_calls,
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
    configure_logging("main_agent", resolved.secret_values())
    return run_with_heartbeat(resolved, lambda: _run_fast_loop_metered(resolved, clock))


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


SCHEMA_PROBE_UNAVAILABLE_SUMMARY = (
    "### Schema probe\n\n"
    "`schema_version()` did not answer (migration 071 not applied, or the database did not "
    "respond) — flags that need a migration were left on."
)


def _migration_bound_flags(settings: Settings) -> dict[str, bool]:
    """NIEZ-30: efektywna wartość każdej flagi z `FLAG_REQUIRED_MIGRATION`; niejawne
    `regime_tagging_enabled` idzie za `regime_aware_enabled` jak w `build_use_case`."""
    flags = {name: bool(getattr(settings, name)) for name in FLAG_REQUIRED_MIGRATION}
    flags["regime_tagging_enabled"] = _regime_tagging_enabled(settings)
    return flags


def gate_flags_by_schema(
    settings: Settings,
    repository: RepositoryPort,
    quota_monitor: QuotaMonitor,
    *,
    now: datetime,
    rebuild: Callable[[Settings], RepositoryPort],
) -> tuple[Settings, RepositoryPort]:
    """NIEZ-30: wyłącza na czas przebiegu flagi, których migracji baza jeszcze nie ma
    (inaczej zapis nieistniejącej kolumny kończy się PGRST204 i gubi predykcję).

    Wyłączone flagi to WARNING w logu i w bannerze maila (`quota_monitor`); repozytorium
    budowane jest od nowa z oczyszczonymi flagami (`rebuild`), bo część z nich trafia do
    jego konstruktora. Gdy wersji nie da się ustalić, nic się nie zmienia (fail-open,
    decyzja właściciela), a podsumowanie przebiegu dostaje notkę."""
    outcome = evaluate_schema_gate(repository, _migration_bound_flags(settings), now=now)
    if outcome.schema_version is None:
        append_step_summary(SCHEMA_PROBE_UNAVAILABLE_SUMMARY)
    if outcome.alert is None:
        return settings, repository
    quota_monitor.record(outcome.alert)
    gated = settings.model_copy(update=dict.fromkeys(outcome.disabled, False))
    return gated, rebuild(gated)


def _run_fast_loop_metered(settings: Settings, clock: Callable[[], datetime]) -> int:
    """NIEZ-24: przebieg z jednym miernikiem żądań HTTP — liczy każdą sesję
    `build_session()` adapterów, bez zmian w ich kodzie."""
    meter = HttpCallMeter()
    with metered_http(meter):
        return _run_fast_loop(settings, clock, http_meter=meter)


def _run_fast_loop(
    settings: Settings,
    clock: Callable[[], datetime],
    http_meter: HttpCallMeter | None = None,
) -> int:
    """Jeden przebieg Fast Loopa; kod wyjścia jak `main()` (0 / 1 / 2).
    `http_meter` (NIEZ-24) trafia do telemetrii cyklu, a z niej do step summary."""
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
    settings, repository = gate_flags_by_schema(
        settings, repository, quota_monitor, now=clock(), rebuild=build_repository
    )
    # FIN-13: jeden ledger tokenów na przebieg — adaptery zgłaszają, mail wycenia.
    usage_ledger = UsageLedger()
    use_case = build_use_case(
        settings,
        repository=repository,
        quota_monitor=quota_monitor,
        run_context=run_context,
        usage_ledger=usage_ledger,
    )
    if http_meter is not None:
        use_case.telemetry.bind_http_meter(http_meter)
    if settings.db_writes_enabled and prediction_columns_missing(repository, use_case):
        return 1
    # WIA-15: stempel `revealed_at` tylko dla revealów, które przeżyły push poprzedniego
    # runu — są w checkoucie `attestation-log` zrobionym przed krokiem agenta.
    if settings.attestation_enabled:
        attestation_reveals(settings, repository).reconcile()
    # Mail i stary push osobno: tekst pushu nie niesie sekcji portfela.
    notifier = build_notifier(settings, quota_monitor=quota_monitor, include_push=False)
    legacy_push_notifier = build_legacy_push_notifier(settings, quota_monitor=quota_monitor)
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
    # DAT-22: VIX, VXN, HY OAS, T10Y2Y, NFCI — raz na cykl (Null bez flagi: zero I/O).
    macro_series = build_macro_series_port(settings).fetch_macro_series(started_at.date())
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
        macro_series=macro_series,
    )

    # ML-31/ML-32/DAT-22: cechy rynkowe cyklu — raz, przed grafem (no-op bez flagi).
    use_case.prepare_market_features(
        eligible,
        asset_types={s: classify_asset(s, crypto_set, etf_set) for s in eligible},
        macro_features=macro_features(macro_series) if macro_series is not None else None,
        now=started_at,
    )

    try:
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
    finally:
        # WIA-60: jedna linia rootu Merkle'a na cykl, także po wyjątku — predykcje
        # zapisane do tej chwili mają sól w bazie, więc ich root wolno opublikować.
        seal_attestation_cycle(
            use_case.attestation_publisher,
            cycle_id=attestation_cycle_id(
                run_context.run_id, run_attempt_from_environ(os.environ), started_at
            ),
            committed_at=clock(),
        )
    results.extend(loop_results)

    duration = time.perf_counter() - started_perf
    # #34: mianownik MUSI być liczbą faktycznie przetworzonych symboli
    # (eligible = symbols + crypto − unsupported), a nie len(settings.symbols).
    # Pętla iteruje `eligible`, więc failures odnoszą się do niego — inaczej
    # przy obecności krypto/unsupported ratio się rozjeżdża (failures mogą nawet
    # przekroczyć mianownik). Spójne z logiką exit-code (też używa len(eligible)).
    logger.info("Fast Loop done — failures=%d/%d", failures, len(eligible))
    # NIEZ-31: otwarty obwód hosta (np. Finnhub) — jeden WARNING per host w bannerze.
    record_open_circuit_alerts(quota_monitor, process_circuits(), clock())
    # MAIL-21: stan źródeł do maila — ten sam odczyt obwodów i licznik HTTP przebiegu.
    source_health = cycle_source_health(
        settings,
        results,
        circuits=process_circuits().circuit_states(),
        http_calls=use_case.telemetry.snapshot().http_calls,
    )

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
            # DAT-21: daty filingów EDGAR są w kalendarzu NYSE, więc dzień cyklu też.
            insider_lead_window=InsiderLeadWindow(
                as_of=to_nyse_time(started_at).date(),
                max_age_days=settings.insider_lead_max_age_days,
            ),
            source_health=source_health,
        ),
    )
    report_delivery = _report_delivery(
        delivered, notifier=notifier, subscribers=subscribers, settings=settings
    )
    if report_delivery is False:
        flag_undelivered_report(quota_monitor, now=clock(), repository=repository)
    run_duration = time.perf_counter() - started_perf
    logger.info("HTTP requests per host: %s", dict(use_case.telemetry.snapshot().http_calls))
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

    # NIEZ-33: strona statusu po zapisie wiersza `cycle_runs` — widzi ten przebieg.
    if public_status_enabled(settings):
        status_page_use_case(settings, repository, clock).run()

    # Exit code 1 tylko gdy wszystkie symbole padły (catastrophic failure).
    # Pojedyncze błędy per-symbol (np. ticker niewspierany przez Finnhub free)
    # są raportowane w mailu i nie powinny psuć całego cyklu w GHA.
    total = len(eligible)
    if total > 0 and failures == total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
