"""Slow Loop entry point — uruchamiany przez GHA co tydzień.

Ewaluuje każdy symbol osobno (bez zapisu wag), a potem trenuje JEDEN wspólny model
XGBoost na zebranych logach predykcyjnych wszystkich symboli (ML-13). Plik wag
(.ubj) powstaje tylko, gdy kandydat przejdzie bramkę walk-forward; workflow GHA
wysyła go wtedy jako asset Release `model-latest` (CI-13), skąd pobiera go Fast Loop.

Użycie:
    uv run python main_trainer.py
"""

from __future__ import annotations

import logging
import os
import random
import sys
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from main_agent import (
    STUB_WRITES_REFUSED_EXIT_CODE,
    build_notifier,
    build_push_channels,
    flag_undelivered_report,
    gate_flags_by_schema,
    llm_fallback_available,
    persist_llm_calls,
    refuses_stub_writes,
    run_with_heartbeat,
    warn_if_paid_calls_disabled,
    write_step_summary,
)
from src.application.cycle_telemetry import CycleTelemetry
from src.application.errors import NotFound, PortError
from src.application.metering import BudgetedLLMPort, MeteredLLMPort, PaidCallAllowance
from src.application.ml_features import feature_set_version
from src.application.ports import LLMPort, ModelCatalogPort, RepositoryPort, UsageSink
from src.application.quota_monitor import QuotaMonitor
from src.application.report_ops import (
    render_challenger_verdicts_markdown,
    render_direction_variant_markdown,
    render_trainer_summary_markdown,
)
from src.application.report_snapshot_training import render_snapshot_models_markdown
from src.application.report_weekly_recap import (
    render_weekly_recap_html,
    render_weekly_recap_text,
)
from src.application.run_context import RunContext
from src.application.usage_ledger import UsageLedger
from src.application.use_cases.backfill_diagnoses import BackfillDiagnosesUseCase
from src.application.use_cases.batch_diagnosis import (
    PAID_LLM_CALLS_PER_RUN as BATCH_DIAGNOSIS_LLM_CALLS,
)
from src.application.use_cases.batch_diagnosis import BatchDiagnosisUseCase
from src.application.use_cases.export_evidence import ExportEvidenceUseCase
from src.application.use_cases.judge_diagnoses import (
    MAX_JUDGED_PER_RUN,
    JudgeDiagnosesUseCase,
)
from src.application.use_cases.monitor_output_drift import (
    MonitorOutputDriftUseCase,
    RowsReader,
    drift_alert,
)
from src.application.use_cases.probe_models import ModelProbe, ProbeModelsUseCase
from src.application.use_cases.reference_signal import RunReferenceSignalUseCase
from src.application.use_cases.review_challengers import ReviewChallengersUseCase
from src.application.use_cases.train_model import (
    POOLED_MODEL_SYMBOL,
    SnapshotModelsResult,
    TrainModelUseCase,
)
from src.application.use_cases.weekly_recap import (
    PAID_LLM_CALLS_PER_RUN as RECAP_LLM_CALLS,
)
from src.application.use_cases.weekly_recap import WeeklyRecapUseCase
from src.config import Settings, load_evaluation_protocol
from src.domain.db_size import BYTES_PER_MB, assess_db_size
from src.domain.direction_variant import DirectionReport
from src.domain.exit_policy import PARTIAL_FAILURE_LOG, is_partial_failure, run_exit_code
from src.domain.finops import slow_loop_paid_budget
from src.domain.fundamentals_refresh import refresh_budget, select_refresh_batch
from src.domain.promotion import ChallengerVerdict
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.infrastructure.adapters.age_archive import AgeEncryptedArchive, is_age_recipient
from src.infrastructure.adapters.alert_notifier import DelegatingAlertNotifier
from src.infrastructure.adapters.alpha_vantage_earnings import AlphaVantageEarningsAdapter
from src.infrastructure.adapters.alpha_vantage_fundamentals import (
    AlphaVantageFundamentalsAdapter,
)
from src.infrastructure.adapters.cached_earnings import CachedEarningsAdapter
from src.infrastructure.adapters.cached_fundamentals import CachedFundamentalsAdapter
from src.infrastructure.adapters.composite_notifier import CompositeNotifier
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from src.infrastructure.adapters.write_guard import ReadOnlyRepository
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter
from src.infrastructure.llm.anthropic_client import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from src.infrastructure.llm.local_stub import LocalStubLLMAdapter
from src.infrastructure.llm.model_catalog import (
    AnthropicModelCatalog,
    LocalStubModelCatalog,
    OpenAIModelCatalog,
)
from src.infrastructure.llm.openai_client import DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from src.infrastructure.llm.openai_embeddings import DEFAULT_MODEL as EMBEDDING_MODEL
from src.infrastructure.llm.unavailable import UnavailableLLMAdapter
from src.infrastructure.logging_setup import configure_logging

logger = logging.getLogger(__name__)

WEEKLY_RECAP_DAYS = 7

# FIN-30: joby Slow Loopa z płatnym LLM. Kolejność w `_requested_llm_calls` to
# pierwszeństwo przy przycinaniu budżetem tygodniowym.
RECAP_JOB = "recap"
# EXP-17: tygodniowa diagnoza zbiorcza — przed spłatą diagnoz (jedno wywołanie nad
# wszystkimi pudłami tygodnia jest tańsze niż wywołanie per pudło).
BATCH_DIAGNOSIS_JOB = "batch_diagnosis"
# WIA-53: spłata długu diagnoz — po diagnozach, które płacą za nowe lekcje.
DIAGNOSIS_JOB = "diagnosis"
# EXP-10: sędzia diagnoz z rubryką — audyt jakości, ostatni w kolejce, więc
# przycinany pierwszy.
DIAGNOSIS_JUDGE_JOB = "diagnosis_judge"
# NIEZ-44: rezerwa przydziału joba na request do zapasowego dostawcy LLM.
FALLBACK_LLM_CALLS_PER_JOB = 1
# NIEZ-32: katalog zaszyfrowanej kopii; workflow uploaduje z niego wyłącznie *.age.
EVIDENCE_BACKUP_DIR = Path("data/backup")


def build_repository(settings: Settings) -> RepositoryPort:
    """Repozytorium Slow Loopa — jedno źródło dla use case'u i kroków obok niego.

    ML-10: `model_scorecard_baselines_enabled` dokłada do karty baseline'y dryfu
    i „sam LLM" (kolumny z migracji 028). BEZ-08: bez `db_writes_enabled`
    owinięte w `ReadOnlyRepository` — lokalny trening niczego nie zapisuje."""
    repository = _supabase_repository(settings)
    if settings.db_writes_enabled:
        return repository
    return ReadOnlyRepository(repository)


def _supabase_repository(settings: Settings) -> SupabaseRepository:
    """Jedyne miejsce, gdzie Slow Loop konstruuje klienta Supabase."""
    return SupabaseRepository(
        url=settings.supabase_url,
        key=settings.supabase_key,
        scorecard_baselines_enabled=settings.model_scorecard_baselines_enabled,
        scorecard_run_id_enabled=settings.model_scorecard_run_id_enabled,
        scorecard_feature_stats_enabled=settings.model_scorecard_feature_stats_enabled,
        # CI-13: sha256 pliku wag w karcie (migracja 060) za flagą.
        scorecard_weights_sha256_enabled=settings.model_scorecard_weights_sha256_enabled,
        # ML-22: proweniencja treningu w karcie (migracja 064) za flagą.
        scorecard_provenance_enabled=settings.model_scorecard_provenance_enabled,
        correction_category_enabled=settings.correction_category_enabled,
        # DAT-21: kapitalizacja w cache fundamentów (migracja 056) za flagą.
        fundamentals_market_cap_enabled=settings.fundamentals_market_cap_enabled,
    )


def build_use_case(settings: Settings, *, run_id: str | None = None) -> TrainModelUseCase:
    """DI Container — Slow Loop potrzebuje tylko bazy + XGBoost.

    DAT-09: z `feature_set_filter_enabled` trening bierze wyłącznie wiersze
    bieżącej wersji zestawu cech (kolumna z migracji 027). ML-24: `run_id`
    przebiegu trafia do wiersza karty modelu wspólnego."""
    return TrainModelUseCase(
        ml_port=XGBoostAdapter(model_path=settings.ml_model_path),
        db_port=build_repository(settings),
        # ML-31: wersja z tej samej flagi co stempel Fast Loopa (4 z cechami rynkowymi).
        feature_set_version=(
            feature_set_version(settings.market_features_enabled)
            if settings.feature_set_filter_enabled
            else None
        ),
        run_id=run_id,
        # ML-52: wektory `feature_snapshots` i ceny do etykiety — sam odczyt,
        # więc bez strażnika zapisu (działa także lokalnie bez `db_writes_enabled`).
        training_data=_supabase_repository(settings),
        # ML-52 (4): model wspólny z wektorów zamiast z widoku (zmienia wagi; off).
        snapshot_source=settings.snapshot_training_enabled,
        # ML-11: target = zwrot / σ, karty w jednostkach σ (zmienia wagi i serving; off).
        sigma_target=settings.ml_sigma_target_enabled,
        # ML-22: commit kodu przebiegu w proweniencji treningu (karta i sidecar wag).
        git_sha=RunContext.from_environ(os.environ).git_sha,
        # ML-26: purge foldów po `settled_at` (zmienia werdykt bramki i wagi; off).
        settled_purge=settings.ml_settled_purge_enabled,
    )


def _refresh_fundamentals_cache(
    settings: Settings,
    repository: RepositoryPort,
    quota_monitor: QuotaMonitor,
) -> None:
    """Slow loop płaci za Alpha Vantage requesty i odświeża cache fundamentów.
    Fast loop czyta wyłącznie z cache — nigdy nie woła AV bezpośrednio.

    Odświeżamy tylko spółki (STOCK) — ETF-y nie mają EPS/P/E. Pojedynczy błąd
    nie wywala całego cyklu. FIN-35: lokalnie bez płatnych wywołań nie pytamy AV.
    DAT-19: tylko partia mieszcząca się w dziennym limicie kluczy, wybrana przez
    domenę (najpierw bez wpisu, potem najdawniej odświeżone).
    """
    if not settings.paid_calls_enabled:
        logger.info("Fundamentals refresh skipped — lokalnie bez płatnych wywołań")
        return
    fundamentals_port = CachedFundamentalsAdapter(
        repo=repository,
        delegate=AlphaVantageFundamentalsAdapter(
            api_keys=settings.require_alpha_vantage_api_keys(),
            quota_monitor=quota_monitor,
        ),
    )
    stock_symbols = [s for s in settings.symbols if s not in settings.symbols_etf]
    batch = select_refresh_batch(
        _fundamentals_fetched_at(repository, stock_symbols),
        refresh_budget(len(settings.alpha_vantage_api_keys)),
    )
    logger.info(
        "Fundamentals refresh — %d of %d companies this run (missing and oldest first)",
        len(batch),
        len(stock_symbols),
    )
    for symbol in batch:
        try:
            fundamentals_port.get_fundamentals(symbol)
        except Exception:
            logger.exception("Fundamentals refresh failed for %s", symbol)


def _fundamentals_fetched_at(
    repository: RepositoryPort, symbols: list[str]
) -> dict[str, datetime | None]:
    """DAT-19: wiek wpisu cache per spółka (None = brak wpisu). Błąd odczytu
    → wszystkie jak bez wpisu: partia idzie wtedy alfabetycznie, jak przed rotacją
    szła kolejnością listy."""
    try:
        known = repository.get_fundamentals_fetched_at(symbols)
    except PortError:
        logger.exception("Fundamentals cache ages unavailable — refresh without rotation")
        known = {}
    return {
        symbol: stamp if isinstance(stamp := known.get(symbol), datetime) else None
        for symbol in symbols
    }


def _refresh_earnings_cache(settings: Settings, repository: RepositoryPort) -> None:
    """FIN-11: raz w tygodniu odświeża `earnings_cache` (migracja 053) z Alpha
    Vantage — Fast Loop czyta datę raportu wyłącznie z cache, przed bramką
    volatility nie płaci za kalendarz.

    Wszystkie klucze AV (rotacja przy soft-limicie), bo fundamenty tego samego
    przebiegu zużywają już pierwszy klucz. Tylko spółki: ETF i krypto nie mają
    raportów. Wpis młodszy niż TTL (np. ponowny przebieg w tym samym tygodniu) nie
    kosztuje requestu. Pojedynczy błąd nie przerywa odświeżania."""
    if not (settings.earnings_calendar_enabled and settings.earnings_cache_enabled):
        return
    if not settings.paid_calls_enabled:
        logger.info("Earnings cache refresh skipped — lokalnie bez płatnych wywołań")
        return
    if not settings.alpha_vantage_api_keys:
        logger.warning("Earnings cache refresh skipped — ALPHA_VANTAGE_API_KEYS is empty")
        return
    first_key, *reserve_keys = settings.alpha_vantage_api_keys
    calendar = CachedEarningsAdapter(
        repo=repository,
        delegate=AlphaVantageEarningsAdapter(api_key=first_key, reserve_keys=reserve_keys),
    )
    stock_symbols = [s for s in settings.symbols if s not in settings.symbols_etf]
    for symbol in stock_symbols:
        try:
            calendar.get_next_earnings(symbol)
        except Exception:
            logger.exception("Earnings cache refresh failed for %s", symbol)


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
    repository: RepositoryPort,
    symbol: str,
    result: Mapping[str, Any],
    run_id: str | None = None,
) -> None:
    """#12: karta kondycji modelu. Zapis best-effort (wzorzec
    `save_quota_alert`) — brak migracji 019 nie może wywalić treningu."""
    try:
        repository.save_model_scorecard(symbol, result, run_id=run_id)
    except Exception:
        logger.exception("Failed to persist model scorecard for %s", symbol)


@dataclass
class _TrainingTally:
    """NIEZ-22: liczby przebiegu Slow Loopa do step summary — bez symboli.

    `statuses` — status ewaluacji per symbol (ile razy), `pooled` — wynik modelu
    wspólnego (None = rzucił albo nie ruszył)."""

    statuses: Counter[str] = field(default_factory=Counter)
    pooled: Mapping[str, Any] | None = None
    # ML-52: modele z `feature_snapshots` i raport selekcji (None = krok nie ruszył).
    snapshot_models: SnapshotModelsResult | None = None
    # ML-30: wariant binarny modelu (None = flaga off albo krok padł).
    direction: DirectionReport | None = None
    # ML-33: werdykt promocji challengerów (None = flaga off albo odczyt padł).
    challengers: tuple[ChallengerVerdict, ...] | None = None
    feature_store_refreshed: bool = True
    failures: int = 0
    total: int = 0

    def record_symbol(self, result: object) -> None:
        if isinstance(result, Mapping):
            self.statuses[str(result.get("status", "unknown"))] += 1

    def render(self) -> str:
        pooled = self.pooled or {}
        samples = pooled.get("n_samples")
        summary = render_trainer_summary_markdown(
            symbol_statuses=self.statuses,
            pooled_status=str(pooled["status"]) if "status" in pooled else None,
            pooled_samples=samples if isinstance(samples, int) else None,
            failures=self.failures,
            total=self.total,
            feature_store_refreshed=self.feature_store_refreshed,
            pooled_label_overlap=_label_overlap_of(pooled),
        )
        if self.snapshot_models is not None:
            summary += "\n" + render_snapshot_models_markdown(self.snapshot_models)
        if self.direction is not None:
            summary += "\n" + render_direction_variant_markdown(self.direction)
        if self.challengers is not None:
            summary += "\n" + render_challenger_verdicts_markdown(self.challengers)
        return summary


def _label_overlap_of(result: Mapping[str, Any]) -> tuple[int, int] | None:
    """ML-25: (nakładające się, zmierzone) etykiety z wyniku treningu wspólnego."""
    overlapping = result.get("label_overlap_count")
    measured = result.get("label_overlap_measured")
    if isinstance(overlapping, int) and isinstance(measured, int):
        return overlapping, measured
    return None


def _train_symbols(
    settings: Settings,
    repository: RepositoryPort,
    use_case: TrainModelUseCase,
    trainable: list[str],
    tally: _TrainingTally | None = None,
    run_id: str | None = None,
) -> int:
    """Ewaluuje każdy symbol z osobna (bez zapisu wag, ML-13) i zwraca liczbę
    niepowodzeń. Błąd jednego symbolu nie przerywa ewaluacji pozostałych."""
    failures = 0
    for symbol in trainable:
        try:
            result = use_case.run(symbol, refresh_view=False)
            logger.info("%s: %s", symbol, result)
            if tally is not None:
                tally.record_symbol(result)
            if settings.model_scorecard_enabled:
                _persist_model_scorecard(repository, symbol, result, run_id)
        except Exception:
            logger.exception("Training failed for symbol %s", symbol)
            failures += 1
    return failures


def _train_pooled_model(
    settings: Settings,
    use_case: TrainModelUseCase,
    trainable: list[str],
    tally: _TrainingTally | None = None,
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
    if tally is not None and isinstance(result, Mapping):
        tally.pooled = result
    return 0


def _evaluate_snapshot_models(
    settings: Settings,
    use_case: TrainModelUseCase,
    trainable: list[str],
    tally: _TrainingTally | None = None,
) -> None:
    """ML-52: modele z `feature_snapshots` (cechy darmowe na wszystkich cyklach,
    cechy LLM na cyklach gated) i raport selekcji bramki — sama ewaluacja, bez
    zapisu wag. Krok pomocniczy: błąd (np. brak tabeli) tylko loguje, exit code
    przebiegu się nie zmienia. Karty idą do `model_scorecards` jak karta `ALL`."""
    try:
        result = use_case.run_snapshot_models(
            trainable, record_scorecard=settings.model_scorecard_enabled
        )
    except Exception:
        logger.exception("Feature snapshot models failed — pomijam")
        return
    if not isinstance(result, SnapshotModelsResult):
        return
    logger.info(
        "Feature snapshot models — vectors=%d labeled=%d free=%s (n=%d) llm=%s (n=%d) "
        "gated/non-gated mean |return| ratio=%s",
        result.n_snapshots,
        result.n_labeled,
        result.free_model.status,
        result.free_model.n_samples,
        result.llm_model.status,
        result.llm_model.n_samples,
        result.selection.abs_return_ratio,
    )
    if tally is not None:
        tally.snapshot_models = result


def _evaluate_direction_variant(
    settings: Settings,
    use_case: TrainModelUseCase,
    trainable: list[str],
    tally: _TrainingTally | None = None,
) -> None:
    """ML-30: wariant binarny P(wzrost poza martwą strefą z protokołu ewaluacji,
    EXP-06) na próbce modelu wspólnego — sama ewaluacja, champion bez zmian.
    Krok pomocniczy: błąd odczytu albo treningu tylko loguje, exit code ten sam."""
    if not settings.ml_direction_classifier_enabled:
        return
    dead_band = float(load_evaluation_protocol().directional_dead_zone)
    try:
        report = use_case.run_direction_variant(trainable, dead_band=dead_band)
    except (PortError, LookupError, ValueError, RuntimeError):
        logger.exception("Direction variant failed — pomijam")
        return
    if not isinstance(report, DirectionReport):
        return
    logger.info(
        "Direction variant (ML-30) — %s, n=%d, dead band ±%g (%d rows inside): %s",
        report.outcome.value,
        report.n_samples,
        report.dead_band,
        report.n_in_dead_band,
        report.reason,
    )
    if tally is not None:
        tally.direction = report


def _review_challengers(
    settings: Settings, repository: RepositoryPort, tally: _TrainingTally | None = None
) -> None:
    """ML-33: werdykt promocji challengerów nogi ML z prognoz cienia rozliczonych
    predykcji (martwa strefa kierunku z protokołu, EXP-06) — sam raport w summary,
    bez zapisu i bez podmiany wag. Błąd odczytu (np. brak tabeli przed migracją 075)
    tylko loguje, exit code ten sam."""
    if not settings.shadow_predictions_enabled:
        return
    dead_band = float(load_evaluation_protocol().directional_dead_zone)
    try:
        verdicts = ReviewChallengersUseCase(
            repository, clock=lambda: datetime.now(UTC)
        ).run(dead_band=dead_band)
    except PortError:
        logger.warning("Champion/challenger review failed — pomijam", exc_info=True)
        return
    for entry in verdicts:
        logger.info(
            "Challenger (ML-33) %s [%s] — %s, pairs=%d, z=%s",
            entry.model_name,
            entry.kind.value,
            entry.verdict.status.value,
            entry.verdict.n_pairs,
            entry.verdict.paired_z,
        )
    if tally is not None:
        tally.challengers = verdicts


def _build_llm_or_unavailable(
    settings: Settings,
    quota_monitor: QuotaMonitor,
    usage_sink: UsageSink | None = None,
    telemetry: CycleTelemetry | None = None,
    allowance: PaidCallAllowance | None = None,
) -> LLMPort:
    """Adapter LLM dla kroków Slow Loopa z deterministycznym fallbackiem. Błąd BUDOWY
    (brak SDK albo klucza) nie może wywrócić całego kroku — use case dostaje adapter,
    który rzuca przy wywołaniu, i idzie ścieżką bez LLM. NIEZ-44: request do
    zapasowego dostawcy liczy `telemetry` przebiegu i zużywa `allowance` joba."""
    from main_agent import build_llm_adapter

    try:
        if settings.llm_fallback_provider is None:
            return build_llm_adapter(settings, quota_monitor=quota_monitor, usage_sink=usage_sink)
        # Telemetria i przydział dotyczą wyłącznie requestu do zapasowego — bez
        # failovera wywołanie fabryki zostaje takie jak dotąd.
        return build_llm_adapter(
            settings,
            quota_monitor=quota_monitor,
            usage_sink=usage_sink,
            telemetry=telemetry,
            fallback_allowance=allowance,
        )
    except Exception as exc:
        logger.exception("LLM adapter build failed — krok idzie bez LLM")
        return UnavailableLLMAdapter(type(exc).__name__)


def _requested_llm_calls(settings: Settings, extra_per_job: int = 0) -> dict[str, int]:
    """FIN-30: płatne wywołania LLM włączonych jobów przebiegu. Recap idzie pierwszy:
    to on trafia do użytkownika. Diagnoza zbiorcza (EXP-17) idzie przed spłatą diagnoz
    (WIA-53), a sędzia diagnoz (EXP-10) jest ostatni — przycięcie bierze go pierwszego.
    Kalibracja pewności nie woła LLM (WIA-65), więc nie prosi o przydział.
    `extra_per_job` (NIEZ-44) — rezerwa na request do zapasowego dostawcy."""
    requested: dict[str, int] = {}
    if settings.weekly_recap_enabled:
        requested[RECAP_JOB] = RECAP_LLM_CALLS + extra_per_job
    if settings.batch_diagnosis_enabled:
        requested[BATCH_DIAGNOSIS_JOB] = BATCH_DIAGNOSIS_LLM_CALLS + extra_per_job
    if settings.diagnosis_backfill_enabled:
        requested[DIAGNOSIS_JOB] = settings.diagnosis_backfill_max_calls + extra_per_job
    if settings.diagnosis_judge_enabled:
        requested[DIAGNOSIS_JUDGE_JOB] = MAX_JUDGED_PER_RUN + extra_per_job
    return requested


def _fallback_llm_calls(settings: Settings) -> int:
    """NIEZ-44 (decyzja 2026-10-20): z działającym zapasowym dostawcą każdy job prosi
    o +1 wywołanie — zużywa je wyłącznie request do zapasowego po awarii głównego.
    Atrapy FIN-35 nie mają failovera. Sufit tygodniowy przycina jak dotąd."""
    if settings.paid_calls_enabled and llm_fallback_available(settings):
        return FALLBACK_LLM_CALLS_PER_JOB
    return 0


class _SlowLoopPaidLLM:
    """FIN-30: jeden meter i jeden przydział tygodniowego budżetu LLM na przebieg.

    Ten sam mechanizm co Fast Loop: `MeteredLLMPort` liczy każdy request w
    `CycleTelemetry` pod kluczem cennika `llm`, a `BudgetedLLMPort` odrzuca
    wywołanie ponad przydział domeny (`slow_loop_paid_budget`) przed dostawcą.
    FIN-13: tokeny prawdziwego dostawcy trafiają do `ledger` przebiegu, z jobem
    jako węzłem."""

    def __init__(
        self, settings: Settings, quota_monitor: QuotaMonitor, ledger: UsageLedger
    ) -> None:
        self._settings = settings
        self._quota_monitor = quota_monitor
        self._ledger = ledger
        self._cap = settings.slow_loop_max_paid_llm_calls_per_week
        self._allowed = slow_loop_paid_budget(
            _requested_llm_calls(settings, _fallback_llm_calls(settings)), self._cap
        )
        self.telemetry = CycleTelemetry()
        logger.info("Slow Loop LLM budget — allowed=%s cap=%s", self._allowed, self._cap)

    def allowed(self, job: str) -> int:
        """Przydział joba z domeny (z rezerwą NIEZ-44 na zapasowego dostawcę)."""
        return self._allowed.get(job, 0)

    def port_for(self, job: str) -> LLMPort:
        # NIEZ-44: jeden przydział na job — dzieli go request do zapasowego dostawcy.
        allowance = PaidCallAllowance(self._allowed.get(job, 0), job)
        return BudgetedLLMPort(self._provider(job, allowance), allowance=allowance)

    def _provider(self, job: str, allowance: PaidCallAllowance) -> LLMPort:
        """Atrapa (FIN-35) i adapter zastępczy nie wysyłają requestu, więc nie
        trafiają do licznika; prawdziwy adapter idzie przez `MeteredLLMPort`."""
        if not self._settings.paid_calls_enabled:
            return LocalStubLLMAdapter()
        provider = _build_llm_or_unavailable(
            self._settings,
            self._quota_monitor,
            usage_sink=self._ledger,
            telemetry=self.telemetry,
            allowance=allowance,
        )
        if isinstance(provider, UnavailableLLMAdapter):
            return provider
        return MeteredLLMPort(provider, self.telemetry, "llm", node=job)

    def log_spend(self) -> None:
        spent = self.telemetry.snapshot().total_paid_calls
        ceiling = "no weekly cap" if self._cap is None else str(self._cap)
        logger.info("Slow Loop paid LLM calls: %d/%s", spent, ceiling)


def _reference_universe(settings: Settings) -> list[str]:
    """Protokół v1: symbole treningu (`symbols` + `crypto_symbols`) bez tych, dla
    których darmowe źródło nie podaje ceny."""
    unsupported = set(settings.symbols_unsupported_price)
    trained = dict.fromkeys([*settings.symbols, *settings.crypto_symbols])
    return [symbol for symbol in trained if symbol not in unsupported]


def _run_reference_signal(
    settings: Settings, repository: RepositoryPort, now: datetime | None = None
) -> None:
    """WIA-24: darmowy sygnał referencyjny raz na przebieg (za flagą, migracja 042).

    Cena z routera darmowych źródeł Fast Loopa (Finnhub /quote, CoinGecko) — zero
    płatnych portów. Zapis przez repozytorium Slow Loopa, więc lokalnie (BEZ-08)
    `ReadOnlyRepository` go pomija. Best-effort: błąd kroku nie zmienia exit code'u
    treningu."""
    if not settings.reference_signal_enabled:
        return
    # Import lokalny — fabryka darmowych cen żyje w DI Containerze Fast Loopa.
    from main_agent import build_market_port

    try:
        summary = RunReferenceSignalUseCase(
            market=build_market_port(settings), repository=repository
        ).run(_reference_universe(settings), now=now)
    except Exception:
        logger.exception("Reference signal failed — pomijam")
        return
    logger.info("Reference signal — %s", summary)


# NIEZ-19: pewność (migracja 016) jest zawsze; przepis (030) tylko przy jego zapisie.
_DRIFT_EXTRA_FIELDS = ("confidence_score",)


def _drift_rows_reader(settings: Settings) -> RowsReader:
    """Odczyt monitora wyjść LLM: rozliczone predykcje bez atrap po awarii LLM.
    `recipe_version` tylko z `recipe_version_enabled` — to ona oznacza migrację 030."""
    reader = _supabase_repository(settings)

    def read_rows(days: int) -> list[dict[str, Any]]:
        return reader.get_resolved_predictions_for_eval(
            days,
            with_recipe=settings.recipe_version_enabled,
            extra_fields=_DRIFT_EXTRA_FIELDS,
        )

    return read_rows


def _run_output_drift(
    settings: Settings,
    quota_monitor: QuotaMonitor,
    now: datetime | None = None,
    read_rows: RowsReader | None = None,
) -> None:
    """NIEZ-19: tydzień do tygodnia — udział klas, średnia pewność i bias wyjść LLM.
    Skok to WARNING w `quota_monitor` (banner maila po zapisie alertów). Zero
    płatnych wywołań; błąd kroku tylko loguje."""
    moment = now or datetime.now(UTC)
    try:
        report = MonitorOutputDriftUseCase(read_rows or _drift_rows_reader(settings)).run(
            now=moment
        )
    except Exception:
        logger.exception("Output drift monitor failed — pomijam")
        return
    alert = drift_alert(report, moment)
    if alert is None:
        logger.info("Output drift — %s", report.describe_pl())
        return
    logger.warning("Output drift — %s", alert.message)
    quota_monitor.record(alert)


def _default_model(provider: str) -> str:
    """Model adaptera dostawcy bez jawnego `model=` (jak `main_agent`)."""
    return ANTHROPIC_DEFAULT_MODEL if provider == "anthropic" else OPENAI_DEFAULT_MODEL


def _probed_models(settings: Settings) -> list[ModelProbe]:
    """NIEZ-25: modele, które agent faktycznie woła — te same reguły wyboru co
    fabryki `main_agent` (główny albo router, rada, zapasowy) plus embeddingi."""
    probes: list[ModelProbe] = []
    if settings.model_routing_enabled:
        probes.append(ModelProbe("openai", settings.router_cheap_model, "router (tani)"))
        probes.append(
            ModelProbe(
                settings.router_frontier_provider,
                settings.router_frontier_model,
                "router (frontier)",
            )
        )
    else:
        provider = settings.llm_provider
        probes.append(ModelProbe(provider, _default_model(provider), "główny LLM"))
    if settings.council_llm_provider is not None or settings.council_llm_model is not None:
        council = settings.council_llm_provider or settings.llm_provider
        model = settings.council_llm_model or _default_model(council)
        probes.append(ModelProbe(council, model, "rada"))
    if settings.llm_fallback_provider is not None:
        fallback = settings.llm_fallback_provider
        model = settings.llm_fallback_model or _default_model(fallback)
        probes.append(ModelProbe(fallback, model, "zapasowy LLM"))
    probes.append(ModelProbe("openai", EMBEDDING_MODEL, "embeddingi"))
    return probes


def _model_catalogs(settings: Settings) -> dict[str, ModelCatalogPort]:
    """Katalog per dostawca z kluczem; FIN-35: lokalnie atrapa bez requestów."""
    if not settings.paid_calls_enabled:
        stub = LocalStubModelCatalog()
        return {"anthropic": stub, "openai": stub}
    catalogs: dict[str, ModelCatalogPort] = {}
    if settings.anthropic_api_key:
        catalogs["anthropic"] = AnthropicModelCatalog(api_key=settings.anthropic_api_key)
    if settings.openai_api_key:
        catalogs["openai"] = OpenAIModelCatalog(api_key=settings.openai_api_key)
    return catalogs


def _push_critical_alerts(settings: Settings, alerts: list[QuotaAlert]) -> None:
    """NIEZ-25: CRITICAL sondy idzie od razu pushem, ZAWSZE gdy jest kanał
    (Telegram albo Slack), niezależnie od `realtime_alerts_enabled`: wycofany model
    to cichy total outage Fast Loopa, a bez pushu właściciel dowiedziałby się o nim
    dopiero z kolejnego maila. Błąd sieci sondy to WARNING i nie jest pushowany."""
    critical = [alert for alert in alerts if alert.severity is QuotaSeverity.CRITICAL]
    if not critical:
        return
    channels = build_push_channels(settings, quota_monitor=None)
    if not channels:
        logger.warning("Model probe: brak kanału push — CRITICAL tylko w quota_alerts")
        return
    try:
        DelegatingAlertNotifier(CompositeNotifier(channels)).send_alert(critical)
    except Exception:
        logger.exception("Model probe: push alertu nie wyszedł")


def _run_model_probe(
    settings: Settings,
    quota_monitor: QuotaMonitor,
    now: datetime | None = None,
    catalogs: dict[str, ModelCatalogPort] | None = None,
) -> None:
    """NIEZ-25: czy skonfigurowane modele LLM istnieją (Models API, bez tokenów).
    Brak modelu = CRITICAL (banner maila i push), błąd sondy = WARNING. Błąd
    kroku tylko loguje — trening i recap idą dalej."""
    if not settings.model_probe_enabled:
        return
    try:
        use_case = ProbeModelsUseCase(
            catalogs if catalogs is not None else _model_catalogs(settings)
        )
        alerts = use_case.run(_probed_models(settings), now=now or datetime.now(UTC))
    except Exception:
        logger.exception("Model probe failed — pomijam")
        return
    for alert in alerts:
        quota_monitor.record(alert)
    _push_critical_alerts(settings, alerts)


def _report_db_size(settings: Settings, repository: RepositoryPort) -> None:
    """NIEZ-37: rozmiar relacji `public` (RPC z migracji 077) i WARNING od 70%
    `db_size_budget_mb` — darmowy plan po limicie przechodzi w read-only. Sam odczyt,
    zero płatnych wywołań, bez retencji. Brak funkcji przed DB Migrate tylko loguje."""
    try:
        sizes = repository.get_db_size_report()
    except NotFound:
        logger.warning("DB size report unavailable — apply migration 077 (DB Migrate)")
        return
    except PortError:
        logger.warning("DB size report failed — pomijam", exc_info=True)
        return
    assessment = assess_db_size(sizes, settings.db_size_budget_mb)
    largest = ", ".join(
        f"{size.name}={size.total_bytes / BYTES_PER_MB:.1f} MB" for size in assessment.largest
    )
    total_mb = assessment.total_bytes / BYTES_PER_MB
    fraction = assessment.budget_fraction
    if assessment.needs_warning and fraction is not None:
        logger.warning(
            "DB size %.1f MB = %.0f%% of db_size_budget_mb=%s (NIEZ-37, retencja to "
            "osobna decyzja) — largest: %s",
            total_mb,
            fraction * 100,
            settings.db_size_budget_mb,
            largest,
        )
        return
    logger.info(
        "DB size %.1f MB (budget %s MB) — largest: %s",
        total_mb,
        settings.db_size_budget_mb,
        largest,
    )


def _run_evidence_export(
    settings: Settings, now: datetime | None = None, out_dir: Path = EVIDENCE_BACKUP_DIR
) -> None:
    """NIEZ-32: tygodniowa kopia tabel dowodowych, szyfrowana `age` przed zapisem.
    Bez poprawnego `BACKUP_AGE_RECIPIENT` krok jest pomijany z WARNING-iem —
    nieszyfrowana kopia nie powstaje nigdy. sha256 artefaktu idzie do logu.
    Sam odczyt bazy (bez zapisu), więc działa też bez `db_writes_enabled`."""
    recipient = (settings.backup_age_recipient or "").strip()
    if not recipient:
        logger.warning("Evidence export skipped — BACKUP_AGE_RECIPIENT nie ustawiony")
        return
    if not is_age_recipient(recipient):
        logger.warning("Evidence export skipped — BACKUP_AGE_RECIPIENT to nie klucz publiczny age")
        return
    try:
        receipt = ExportEvidenceUseCase(
            _supabase_repository(settings), AgeEncryptedArchive(recipient, out_dir)
        ).run(now=now or datetime.now(UTC))
    except Exception:
        logger.exception("Evidence export failed — pomijam")
        return
    logger.info(
        "Evidence export — %s sha256=%s (%d B)",
        receipt.path,
        receipt.sha256,
        receipt.size_bytes,
    )


def _run_calibration_scores(
    repository: RepositoryPort, *, brier_score_enabled: bool = False
) -> None:
    """#4: arytmetyczna ocena kalibracji pewności (Slow Loop), jednym zapisem
    zbiorczym (migracja 077). WIA-65: bez sędziego LLM — zero płatnych wywołań.
    WIA-37: `brier_score_enabled` = reguła Briera v2 (bez flagi liniowa v1)."""
    try:
        from src.application.use_cases.calibrate_confidence import (
            CalibrateConfidenceUseCase,
        )

        cal_result = CalibrateConfidenceUseCase(
            repository_port=repository,
            brier_score_enabled=brier_score_enabled,
        ).run(days=30)
        logger.info("Calibration scores — %s", cal_result.get("status"))
    except Exception:
        logger.exception("Calibration scores failed — pomijam")


def _run_batch_diagnosis(repository: RepositoryPort, llm_port: LLMPort) -> None:
    """EXP-17: pudła ostatnich 7 dni bez diagnozy — JEDNO wywołanie LLM w tygodniowym
    budżecie Slow Loopa. Przed spłatą diagnoz, żeby ta nie płaciła za nie osobno.
    Best-effort: błąd kroku nie zmienia exit code'u."""
    try:
        result = BatchDiagnosisUseCase(
            repository_port=repository, llm_port=llm_port, clock=lambda: datetime.now(UTC)
        ).run()
    except PortError:
        logger.warning("Batch diagnosis failed — pomijam", exc_info=True)
        return
    logger.info("Batch diagnosis — %s", result)


def _run_diagnosis_backfill(
    repository: RepositoryPort, llm_port: LLMPort, max_paid_calls: int
) -> None:
    """WIA-53: diagnozy błędnych predykcji, którym bramka volatility pominęła
    płatną diagnozę. Sekwencyjnie, po treningu — poza wyścigiem `symbol_concurrency`
    Fast Loopa. Best-effort: błąd kroku nie zmienia exit code'u."""
    try:
        result = BackfillDiagnosesUseCase(
            repository_port=repository, llm_port=llm_port, max_paid_calls=max_paid_calls
        ).run()
    except Exception:
        logger.exception("Diagnosis backfill failed — pomijam")
        return
    logger.info("Diagnosis backfill — %s", result)


def _run_diagnosis_judge(
    repository: RepositoryPort, llm_port: LLMPort, max_samples: int
) -> None:
    """EXP-10: losowa próbka najwyżej `max_samples` diagnoz tygodnia oceniona rubryką;
    diagnoza poniżej progu wypada z lessons i RAG. Po diagnozach przebiegu, w
    tygodniowym budżecie Slow Loopa. Best-effort: błąd kroku nie zmienia exit code'u."""
    try:
        result = JudgeDiagnosesUseCase(
            repository_port=repository,
            llm_port=llm_port,
            clock=lambda: datetime.now(UTC),
            rng=random.Random(),
            max_samples=max_samples,
        ).run()
    except PortError:
        logger.warning("Diagnosis judge failed — pomijam", exc_info=True)
        return
    logger.info("Diagnosis judge — %s", result)


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
            council_redundancy_enabled=settings.council_redundancy_enabled,
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
        # NIEZ-12: recap miał wyjść, a nie dotarł — widać to w UI Actions (exit 0 zostaje).
        flag_undelivered_report(quota_monitor, now=datetime.now(UTC))


def main(settings: Settings | None = None) -> int:
    resolved = settings or Settings.from_env()
    configure_logging("main_trainer", resolved.secret_values())
    # NIEZ-13: ping zewnętrznego heartbeatu na koniec (ok = exit 0).
    return run_with_heartbeat(resolved, lambda: _run_with_summary(resolved))


def _run_with_summary(settings: Settings) -> int:
    if refuses_stub_writes(settings):
        return STUB_WRITES_REFUSED_EXIT_CODE
    tally = _TrainingTally()
    exit_code = _run_slow_loop(settings, tally)
    # NIEZ-22: liczby przebiegu na stronie runu (GitHub Actions), także po porażce.
    write_step_summary(tally.render)
    return exit_code


def _run_slow_loop(settings: Settings, tally: _TrainingTally) -> int:
    """Jeden przebieg Slow Loopa; `tally` zbiera liczby do step summary (NIEZ-22)."""
    warn_if_paid_calls_disabled(settings)
    quota_monitor = QuotaMonitor()
    repository = build_repository(settings)
    settings, repository = gate_flags_by_schema(
        settings, repository, quota_monitor, now=datetime.now(UTC), rebuild=build_repository
    )
    # ML-24: jeden identyfikator na przebieg — wspólny dla wszystkich wierszy karty.
    run_id = str(uuid.uuid4())
    use_case = build_use_case(settings, run_id=run_id)

    logger.info("Slow Loop start — symbols=%s", settings.symbols)
    try:
        use_case.refresh_feature_store()
    except Exception:
        logger.exception("Failed to refresh feature store")
        tally.feature_store_refreshed = False
        tally.failures = tally.total = 1
        return 1

    _refresh_fundamentals_cache(settings, repository, quota_monitor)
    _refresh_earnings_cache(settings, repository)
    persisted_alerts = _persist_quota_alerts(repository, quota_monitor)

    # BUG: trener widział tylko `settings.symbols`, więc BTC/ETH NIGDY nie
    # przechodziły retrainu — model krypto zostawał na zawsze przy cold-starcie.
    # `dict.fromkeys` zachowuje kolejność i deduplikuje symbol obecny w obu listach.
    trainable = list(dict.fromkeys([*settings.symbols, *settings.crypto_symbols]))
    symbol_failures = _train_symbols(settings, repository, use_case, trainable, tally, run_id)
    failures = symbol_failures + _train_pooled_model(settings, use_case, trainable, tally)
    _evaluate_snapshot_models(settings, use_case, trainable, tally)
    _evaluate_direction_variant(settings, use_case, trainable, tally)
    _review_challengers(settings, repository, tally)
    tally.failures, tally.total = failures, len(trainable) + 1
    _run_reference_signal(settings, repository)
    _run_output_drift(settings, quota_monitor)
    _run_model_probe(settings, quota_monitor)
    _report_db_size(settings, repository)
    _run_evidence_export(settings)

    if settings.calibration_judge_enabled:
        _run_calibration_scores(
            repository, brier_score_enabled=settings.calibration_score_version_enabled
        )

    usage_ledger = UsageLedger()
    paid_llm = _SlowLoopPaidLLM(settings, quota_monitor, usage_ledger)

    if settings.weekly_recap_enabled:
        _send_weekly_recap(settings, repository, quota_monitor, paid_llm.port_for(RECAP_JOB))
    if settings.batch_diagnosis_enabled:
        _run_batch_diagnosis(repository, paid_llm.port_for(BATCH_DIAGNOSIS_JOB))
    if settings.diagnosis_backfill_enabled:
        # Rezerwa NIEZ-44 zostaje na zapasowego dostawcę — wierszy nie więcej niż cap.
        _run_diagnosis_backfill(
            repository,
            paid_llm.port_for(DIAGNOSIS_JOB),
            min(settings.diagnosis_backfill_max_calls, paid_llm.allowed(DIAGNOSIS_JOB)),
        )
    if settings.diagnosis_judge_enabled:
        # Rezerwa NIEZ-44 jak wyżej — próbka nie większa niż cap.
        _run_diagnosis_judge(
            repository,
            paid_llm.port_for(DIAGNOSIS_JUDGE_JOB),
            min(MAX_JUDGED_PER_RUN, paid_llm.allowed(DIAGNOSIS_JUDGE_JOB)),
        )
    paid_llm.log_spend()
    persist_llm_calls(
        settings=settings,
        repository=repository,
        ledger=usage_ledger,
        run_context=RunContext.from_environ(os.environ),
    )

    # NIEZ-03: alerty z recapu i diagnoz (LLM, Resend) powstają po pierwszym zapisie.
    _persist_quota_alerts(repository, quota_monitor, start=persisted_alerts)

    logger.info("Slow Loop done — failures=%d/%d", failures, len(trainable) + 1)
    return _slow_loop_exit_code(symbol_failures, failures - symbol_failures, len(trainable))


def _slow_loop_exit_code(symbol_failures: int, pooled_failures: int, symbols: int) -> int:
    """NIEZ-38: exit 1 tylko, gdy padły wszystkie symbole (jak Fast Loop). Część
    symboli albo sam model wspólny to częściowa porażka: WARNING i liczby w step
    summary, exit 0 — przy padniętym modelu wspólnym wagi zostają z poprzedniego
    tygodnia."""
    exit_code = run_exit_code(symbol_failures, symbols)
    if is_partial_failure(symbol_failures, symbols):
        logger.warning(PARTIAL_FAILURE_LOG, "Slow Loop", symbol_failures, symbols)
    if pooled_failures and exit_code == 0:
        logger.warning(
            "Slow Loop: model wspólny padł — wagi zostają z poprzedniego przebiegu, exit 0 "
            "(NIEZ-38)"
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
