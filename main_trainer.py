"""Slow Loop entry point — uruchamiany przez GHA co tydzień.

Ewaluuje każdy symbol osobno (bez zapisu wag), a potem trenuje JEDEN wspólny model
XGBoost na zebranych logach predykcyjnych wszystkich symboli (ML-13). Plik wag
(.ubj) powstaje tylko, gdy kandydat przejdzie bramkę walk-forward; workflow GHA
commituje go wtedy do repo.

Użycie:
    uv run python main_trainer.py
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from main_agent import build_notifier
from src.application.cycle_telemetry import CycleTelemetry
from src.application.metering import BudgetedLLMPort, MeteredLLMPort
from src.application.ml_features import FEATURE_SET_VERSION
from src.application.ports import LLMPort, RepositoryPort
from src.application.quota_monitor import QuotaMonitor
from src.application.report_weekly_recap import (
    render_weekly_recap_html,
    render_weekly_recap_text,
)
from src.application.use_cases.calibrate_confidence import (
    PAID_LLM_CALLS_PER_RUN as JUDGE_LLM_CALLS,
)
from src.application.use_cases.train_model import POOLED_MODEL_SYMBOL, TrainModelUseCase
from src.application.use_cases.weekly_recap import (
    PAID_LLM_CALLS_PER_RUN as RECAP_LLM_CALLS,
)
from src.application.use_cases.weekly_recap import WeeklyRecapUseCase
from src.config import Settings
from src.domain.finops import slow_loop_paid_budget
from src.infrastructure.adapters.alpha_vantage_fundamentals import (
    AlphaVantageFundamentalsAdapter,
)
from src.infrastructure.adapters.cached_fundamentals import CachedFundamentalsAdapter
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from src.infrastructure.adapters.write_guard import ReadOnlyRepository
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter
from src.infrastructure.llm.unavailable import UnavailableLLMAdapter
from src.infrastructure.logging_redaction import configure_logging

logger = logging.getLogger(__name__)

WEEKLY_RECAP_DAYS = 7

# FIN-30: joby Slow Loopa z płatnym LLM. Kolejność w `_requested_llm_calls` to
# pierwszeństwo przy przycinaniu budżetem tygodniowym.
RECAP_JOB = "recap"
JUDGE_JOB = "judge"


def build_repository(settings: Settings) -> RepositoryPort:
    """Repozytorium Slow Loopa — jedno źródło dla use case'u i kroków obok niego.

    ML-10: `model_scorecard_baselines_enabled` dokłada do karty baseline'y dryfu
    i „sam LLM" (kolumny z migracji 028). BEZ-08: bez `db_writes_enabled`
    owinięte w `ReadOnlyRepository` — lokalny trening niczego nie zapisuje."""
    repository = SupabaseRepository(
        url=settings.supabase_url,
        key=settings.supabase_key,
        scorecard_baselines_enabled=settings.model_scorecard_baselines_enabled,
    )
    if settings.db_writes_enabled:
        return repository
    return ReadOnlyRepository(repository)


def build_use_case(settings: Settings) -> TrainModelUseCase:
    """DI Container — Slow Loop potrzebuje tylko bazy + XGBoost.

    DAT-09: z `feature_set_filter_enabled` trening bierze wyłącznie wiersze
    bieżącej wersji zestawu cech (kolumna z migracji 027)."""
    return TrainModelUseCase(
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        db_port=build_repository(settings),
        feature_set_version=(
            FEATURE_SET_VERSION if settings.feature_set_filter_enabled else None
        ),
    )


def _refresh_fundamentals_cache(
    settings: Settings,
    repository: RepositoryPort,
    quota_monitor: QuotaMonitor,
) -> None:
    """Slow loop płaci za Alpha Vantage requesty i odświeża cache fundamentów.
    Fast loop czyta wyłącznie z cache — nigdy nie woła AV bezpośrednio.

    Odświeżamy tylko spółki (STOCK) — ETF-y nie mają EPS/P/E. Pojedynczy błąd
    nie wywala całego cyklu.
    """
    fundamentals_port = CachedFundamentalsAdapter(
        repo=repository,
        delegate=AlphaVantageFundamentalsAdapter(
            api_keys=settings.alpha_vantage_api_keys,
            quota_monitor=quota_monitor,
        ),
    )
    stock_symbols = [s for s in settings.symbols if s not in settings.symbols_etf]
    for symbol in stock_symbols:
        try:
            fundamentals_port.get_fundamentals(symbol)
        except Exception:
            logger.exception("Fundamentals refresh failed for %s", symbol)


def _persist_quota_alerts(
    repository: RepositoryPort, quota_monitor: QuotaMonitor, start: int = 0
) -> int:
    """Persystencja alertów kwoty od pozycji `start` (np. wyczerpany limit AV przy
    fundamentach, Resend przy recapie). Banner w mailu sięga tydzień wstecz
    (`QUOTA_ALERT_LOOKBACK_HOURS`), więc alert z niedzieli trafi do najbliższego
    raportu. Zwraca liczbę alertów już zapisanych — kolejne wywołanie zaczyna od niej.
    """
    alerts = quota_monitor.alerts
    for alert in alerts[start:]:
        try:
            repository.save_quota_alert(alert)
        except Exception:
            logger.exception("Failed to persist quota alert from %s", alert.source)
    return len(alerts)


def _persist_model_scorecard(
    repository: RepositoryPort, symbol: str, result: Mapping[str, Any]
) -> None:
    """#12: karta kondycji modelu. Zapis best-effort (wzorzec
    `save_quota_alert`) — brak migracji 019 nie może wywalić treningu."""
    try:
        repository.save_model_scorecard(symbol, result)
    except Exception:
        logger.exception("Failed to persist model scorecard for %s", symbol)


def _train_symbols(
    settings: Settings,
    repository: RepositoryPort,
    use_case: TrainModelUseCase,
    trainable: list[str],
) -> int:
    """Ewaluuje każdy symbol z osobna (bez zapisu wag, ML-13) i zwraca liczbę
    niepowodzeń. Błąd jednego symbolu nie przerywa ewaluacji pozostałych."""
    failures = 0
    for symbol in trainable:
        try:
            result = use_case.run(symbol, refresh_view=False)
            logger.info("%s: %s", symbol, result)
            if settings.model_scorecard_enabled:
                _persist_model_scorecard(repository, symbol, result)
        except Exception:
            logger.exception("Training failed for symbol %s", symbol)
            failures += 1
    return failures


def _train_pooled_model(
    settings: Settings, use_case: TrainModelUseCase, trainable: list[str]
) -> int:
    """ML-13: jeden wspólny model na wszystkich symbolach — jedyny zapis wag w
    przebiegu. Zwraca liczbę niepowodzeń (0 albo 1); wiersz karty modelu zapisuje
    use case jako `POOLED_MODEL_SYMBOL`."""
    try:
        result = use_case.run_pooled(
            trainable, record_scorecard=settings.model_scorecard_enabled
        )
    except Exception:
        logger.exception("Pooled training failed")
        return 1
    logger.info("%s (%d symbols): %s", POOLED_MODEL_SYMBOL, len(trainable), result)
    return 0


def _build_llm_or_unavailable(settings: Settings, quota_monitor: QuotaMonitor) -> LLMPort:
    """Adapter LLM dla kroków Slow Loopa z deterministycznym fallbackiem. Błąd BUDOWY
    (brak SDK albo klucza) nie może wywrócić całego kroku — use case dostaje adapter,
    który rzuca przy wywołaniu, i idzie ścieżką bez LLM."""
    from main_agent import build_llm_adapter

    try:
        return build_llm_adapter(settings, quota_monitor=quota_monitor)
    except Exception as exc:
        logger.exception("LLM adapter build failed — krok idzie bez LLM")
        return UnavailableLLMAdapter(type(exc).__name__)


def _requested_llm_calls(settings: Settings) -> dict[str, int]:
    """FIN-30: płatne wywołania LLM włączonych jobów przebiegu. Recap idzie przed
    sędzią kalibracji: to on trafia do użytkownika, więc przycięcie bierze sędziego."""
    requested: dict[str, int] = {}
    if settings.weekly_recap_enabled:
        requested[RECAP_JOB] = RECAP_LLM_CALLS
    if settings.calibration_judge_enabled:
        requested[JUDGE_JOB] = JUDGE_LLM_CALLS
    return requested


class _SlowLoopPaidLLM:
    """FIN-30: jeden meter i jeden przydział tygodniowego budżetu LLM na przebieg.

    Ten sam mechanizm co Fast Loop: `MeteredLLMPort` liczy każdy request w
    `CycleTelemetry` pod kluczem cennika `llm`, a `BudgetedLLMPort` odrzuca
    wywołanie ponad przydział domeny (`slow_loop_paid_budget`) przed dostawcą."""

    def __init__(self, settings: Settings, quota_monitor: QuotaMonitor) -> None:
        self._settings = settings
        self._quota_monitor = quota_monitor
        self._cap = settings.slow_loop_max_paid_llm_calls_per_week
        self._allowed = slow_loop_paid_budget(_requested_llm_calls(settings), self._cap)
        self.telemetry = CycleTelemetry()
        logger.info("Slow Loop LLM budget — allowed=%s cap=%s", self._allowed, self._cap)

    def port_for(self, job: str) -> LLMPort:
        provider = _build_llm_or_unavailable(self._settings, self._quota_monitor)
        # Adapter zastępczy nie wysyła requestu, więc nie trafia do licznika.
        metered = (
            provider
            if isinstance(provider, UnavailableLLMAdapter)
            else MeteredLLMPort(provider, self.telemetry, "llm")
        )
        return BudgetedLLMPort(metered, allowed=self._allowed.get(job, 0), job=job)

    def log_spend(self) -> None:
        spent = self.telemetry.snapshot().total_paid_calls
        ceiling = "no weekly cap" if self._cap is None else str(self._cap)
        logger.info("Slow Loop paid LLM calls: %d/%s", spent, ceiling)


def _run_calibration_judge(repository: RepositoryPort, llm_port: LLMPort) -> None:
    """#4: LLM-as-Judge kalibracji pewności (Slow Loop). FIN-30: jego wywołanie
    LLM liczy się do tygodniowego budżetu Slow Loopa, nie do budżetu Fast Loopa."""
    try:
        from src.application.use_cases.calibrate_confidence import (
            CalibrateConfidenceUseCase,
        )

        cal_result = CalibrateConfidenceUseCase(
            repository_port=repository,
            llm_port=llm_port,
        ).run(days=30)
        logger.info("Calibration judge — %s", cal_result.get("status"))
    except Exception:
        logger.exception("Calibration judge failed — pomijam")


def _send_weekly_recap(
    settings: Settings,
    repository: RepositoryPort,
    quota_monitor: QuotaMonitor,
    llm_port: LLMPort,
) -> None:
    """#17: niedzielna retrospektywa. Slow Loop chodzi co tydzień i dotąd NIE
    WYSYŁAŁ użytkownikowi nic. Koszt: dokładnie jedno wywołanie LLM tygodniowo
    (FIN-30: w tygodniowym budżecie Slow Loopa).
    Poniżej progu 5 zamkniętych predykcji use case zwraca `skipped_*` i mail
    NIE wychodzi — przy chudym tygodniu recap robiłby dramat z szumu.
    """
    # WIA-19: za flagą tydzień liczy się po chwili rozliczenia (migracja 029) —
    # piątkowa predykcja, w niedzielę jeszcze otwarta, trafia do kolejnego recapu.
    since = (
        datetime.now(UTC) - timedelta(days=WEEKLY_RECAP_DAYS)
        if settings.settled_at_windows_enabled
        else None
    )
    try:
        recap_result = WeeklyRecapUseCase(
            repository_port=repository,
            llm_port=llm_port,
            crypto_symbols=settings.crypto_symbols,
        ).run(days=WEEKLY_RECAP_DAYS, since=since)
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


def main(settings: Settings | None = None) -> int:
    settings = settings or Settings.from_env()
    configure_logging(settings.secret_values())
    quota_monitor = QuotaMonitor()
    repository = build_repository(settings)
    use_case = build_use_case(settings)

    logger.info("Slow Loop start — symbols=%s", settings.symbols)
    try:
        use_case.refresh_feature_store()
    except Exception:
        logger.exception("Failed to refresh feature store")
        return 1

    _refresh_fundamentals_cache(settings, repository, quota_monitor)
    persisted_alerts = _persist_quota_alerts(repository, quota_monitor)

    # BUG: trener widział tylko `settings.symbols`, więc BTC/ETH NIGDY nie
    # przechodziły retrainu — model krypto zostawał na zawsze przy cold-starcie.
    # `dict.fromkeys` zachowuje kolejność i deduplikuje symbol obecny w obu listach.
    trainable = list(dict.fromkeys([*settings.symbols, *settings.crypto_symbols]))
    failures = _train_symbols(settings, repository, use_case, trainable)
    failures += _train_pooled_model(settings, use_case, trainable)

    paid_llm = _SlowLoopPaidLLM(settings, quota_monitor)
    if settings.calibration_judge_enabled:
        _run_calibration_judge(repository, paid_llm.port_for(JUDGE_JOB))

    if settings.weekly_recap_enabled:
        _send_weekly_recap(settings, repository, quota_monitor, paid_llm.port_for(RECAP_JOB))
    paid_llm.log_spend()

    # NIEZ-03: alerty z sędziego i recapu (LLM, Resend) powstają po pierwszym zapisie.
    _persist_quota_alerts(repository, quota_monitor, start=persisted_alerts)

    logger.info("Slow Loop done — failures=%d/%d", failures, len(trainable) + 1)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
