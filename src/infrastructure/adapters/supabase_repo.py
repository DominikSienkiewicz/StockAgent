from __future__ import annotations

import functools
import inspect
import logging
import re
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, NamedTuple, cast

import httpx
from postgrest.exceptions import APIError
from supabase import create_client

from src.application.errors import MalformedPayload, NotFound, PortError, ProviderUnavailable
from src.application.ports import (
    CycleRun,
    DeferredDiagnosis,
    EvidenceSourcePort,
    JudgeableDiagnosis,
    LatestPrediction,
    OpenInvalidation,
    OpenReferenceCall,
    PreviousVerdict,
    RepositoryPort,
    StoredReceipt,
    StoredShockAlert,
    TrainingDataPort,
)
from src.application.report_council_history import InvestorHistory
from src.domain.attestation import StoredCommitment
from src.domain.corporate_actions import SplitSuspicion
from src.domain.council import (
    FIRST_COUNCIL_ROUND,
    CouncilVerdict,
    InvestorOpinion,
    Recommendation,
    parse_recommendation,
)
from src.domain.council_redundancy import CouncilVoteSample
from src.domain.db_size import RelationSize
from src.domain.earnings import EARNINGS_CACHE_TTL_HOURS, CachedEarningsDate
from src.domain.equity_curve import directional_hit_rate
from src.domain.feature_requirements import FLAG_REQUIRED_MIGRATION
from src.domain.finops import LedgerCost, LLMCallEntry
from src.domain.mistake_taxonomy import MistakeCategory
from src.domain.paper_ledger import (
    GENESIS_HASH,
    ExitReason,
    LedgerHead,
    LedgerRow,
    LedgerTrade,
    PaperTradeClose,
    PaperTradeEntry,
    TradeSide,
    chain_entry,
    utc_timestamp,
)
from src.domain.persona_track_record import (
    PersonaScoring,
    PersonaVoteOutcome,
    persona_accuracy_weights,
    persona_hit_stats,
)
from src.domain.prediction import (
    DIAGNOSIS_REJECTED_INSIGHT,
    SAME_SYMBOL_ANALOG_EXCLUSION_HOURS,
    UNDIAGNOSED_MISTAKE_PLACEHOLDERS,
    Prediction,
    TrendDirection,
    normalize_trend,
)
from src.domain.price_levels import LevelDirection, PriceLevel, PriceLevelAlert
from src.domain.promotion import SettledShadowForecast, ShadowForecast
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.reference_signal import (
    REFERENCE_RECIPE_VERSION,
    ReferenceCall,
    ReferenceOutcome,
    ReferenceRead,
)
from src.domain.regime import MarketRegime, RegimeState
from src.domain.regime_key import canonical_regime_key
from src.domain.session_close import SessionCloseSettlement
from src.domain.shadow_prompt import ShadowTrendForecast
from src.domain.shock import SentShockLevel, ShockDirection
from src.domain.significance import effective_n
from src.domain.track_record import group_by_cycle
from src.domain.training_labels import FeatureSnapshot
from src.domain.training_provenance import PROVENANCE_RECORD_KEYS
from src.domain.value_objects import (
    FUNDAMENTALS_CACHE_TTL_HOURS,
    AssetType,
    Fundamentals,
    Money,
)
from src.infrastructure.error_translation import as_port_error

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "prediction_logs"
DEFAULT_FEATURE_VIEW = "ml_feature_store"
DEFAULT_SNAPSHOT_TABLE = "price_snapshots"
DEFAULT_COUNCIL_VOTES_TABLE = "council_votes"

# ALR-16: kierunki szoku, które watch zapisuje w `shock_alerts.direction`.
_SHOCK_DIRECTIONS = frozenset(direction.value for direction in ShockDirection)
DEFAULT_QUOTA_ALERTS_TABLE = "quota_alerts"
DEFAULT_FEATURE_SNAPSHOT_TABLE = "feature_snapshots"
DEFAULT_REGIME_HISTORY_TABLE = "regime_history"
DEFAULT_SHADOW_PREDICTION_TABLE = "shadow_predictions"
# WIA-24 (migracja 042): sygnał referencyjny poza `prediction_logs`.
REFERENCE_PREDICTIONS_TABLE = "reference_predictions"
# FIN-13: ledger kosztów per request LLM / embeddingów (migracja 040).
LLM_CALLS_TABLE = "llm_calls"
# NIEZ-29: historia przebiegów pętli (migracja 041).
CYCLE_RUNS_TABLE = "cycle_runs"
# FIN-11: data najbliższego raportu per symbol (migracja 053).
EARNINGS_CACHE_TABLE = "earnings_cache"
# ALR-11 (migracja 057): alerty poziomów cenowych ustawiane przez użytkownika.
PRICE_LEVEL_ALERTS_TABLE = "price_level_alerts"
_PRICE_LEVEL_UNIQUE = "symbol,level,direction"
# DAT-23 (migracja 062): podejrzenia splitów, jeden wiersz na symbol, rodzaj i dzień.
CORPORATE_ACTIONS_TABLE = "corporate_actions"
_CORPORATE_ACTION_UNIQUE = "symbol,kind,detected_on"
_CORPORATE_ACTION_PRICE_COLUMNS = frozenset({"prev_close", "reference_price"})

# Retry parametry dla refresh_feature_store — Slow Loop wywołuje RPC przed
# treningiem; transient błąd sieci nie powinien wywalać całego treningu.
_REFRESH_MAX_ATTEMPTS = 3
_REFRESH_BACKOFF_S = 2.0

# Hosted Supabase / PostgREST tnie SELECT do ~1000 wierszy BEZ błędu (#8).
# Agregaty (get_accuracy_stats itd.) muszą paginować po .range(), inaczej
# trafność liczona jest na obciętym, niedeterministycznym podzbiorze.
_PAGE_SIZE = 1000


def _cycle_run(row: Mapping[str, Any]) -> CycleRun:
    """Wiersz `cycle_runs` (041) jako `CycleRun`; `paid_calls` to JSONB z liczbami."""
    attempt = row.get("run_attempt")
    delivered = row.get("report_delivered")
    paid = row.get("paid_calls")
    return CycleRun(
        loop=str(row["loop"]),
        run_id=str(row["run_id"]),
        run_attempt=None if attempt is None else int(attempt),
        started_at=datetime.fromisoformat(str(row["started_at"])),
        duration_s=float(row["duration_s"]),
        eligible=int(row["eligible"]),
        saved=int(row["saved"]),
        failures=int(row["failures"]),
        paid_calls=(
            {str(key): int(value) for key, value in paid.items()}
            if isinstance(paid, Mapping)
            else {}
        ),
        report_delivered=None if delivered is None else bool(delivered),
    )


def _paginate(query_builder: Any) -> list[dict[str, Any]]:
    """Iteruje .range(offset, offset+_PAGE_SIZE-1) po `query_builder` aż do
    zwrócenia krótkiej strony (<_PAGE_SIZE), kumulując wszystkie wiersze.

    `query_builder` to gotowy builder (po .order/.gte/...), na którym wywołanie
    .range(...).execute() zwraca kolejną stronę. Dla zbiorów <_PAGE_SIZE
    zachowanie jest identyczne jak pojedynczy select (jedna iteracja).

    postgrest-py `.range()` DOKLEJA `offset`/`limit` do parametrów tego samego
    buildera (`QueryParams.add`), więc bez resetu druga strona szłaby jako
    `offset=0&offset=1000&limit=1000&limit=1000` (NIEZ-15). Każda strona startuje
    od parametrów zapytania bazowego (filtry + sortowanie) i dostaje dokładnie
    jedną parę offset/limit."""
    base_params = query_builder.request.params
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query_builder.request.params = base_params
        response = query_builder.range(offset, offset + _PAGE_SIZE - 1).execute()
        page = _rows(response)
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


class _EvalField(NamedTuple):
    column: str  # kolumna `prediction_logs` (do rozpoznania braku migracji)
    select: str  # wyrażenie selecta PostgREST


# WIA-55 / EXP-01: pola opcjonalne ewaluacji (`main_tools.py evaluate --by/--echo`).
# Czytane tylko na żądanie: każda kolumna istnieje dopiero po swojej migracji
# (w komentarzu), a domyślny przebieg ma działać na każdej bazie.
_EVAL_EXTRA_FIELDS: dict[str, _EvalField] = {
    "symbol": _EvalField("symbol", "symbol"),
    # 005 — sama rekomendacja z JSONB, bez głosów i uzasadnień person.
    "council_recommendation": _EvalField(
        "council_verdict", "council_recommendation:council_verdict->>final_recommendation"
    ),
    "confidence_score": _EvalField("confidence_score", "confidence_score"),  # 016
    "regime": _EvalField("regime", "regime"),  # 017
    "alpha_fusion_score": _EvalField("alpha_fusion_score", "alpha_fusion_score"),  # 022
    "edge_sigma": _EvalField("edge_sigma", "edge_sigma"),  # 025
    "gate_delta": _EvalField("gate_delta", "gate_delta"),  # 034
    # WIA-47 — wejścia heurystyk RiskSignal (001, 004).
    "sentiment_score": _EvalField("sentiment_score", "sentiment_score"),
    "av_llm_agreement": _EvalField("av_llm_agreement", "av_llm_agreement"),
    "news_volume_24h": _EvalField("news_volume_24h", "news_volume_24h"),
    # EXP-05 — konsensus rady (005) i ramiona eksperymentów: kolumna (037)
    # oraz opcjonalny klucz kwitu v2 (020), z którego CLI czyta przed migracją.
    "council_consensus": _EvalField(
        "council_verdict", "council_consensus:council_verdict->>consensus_strength"
    ),
    "experiment_arms": _EvalField("experiment_arms", "experiment_arms"),  # 037
    "receipt_experiment_arms": _EvalField(
        "decision_receipts", "receipt_experiment_arms:decision_receipts->experiment_arms"
    ),
}

# Kody PostgREST/Postgres dla kolumny, której nie ma w bazie (select → 42703).
_MISSING_COLUMN_CODES = frozenset({"42703", "PGRST204"})

# ML-40: migracja, od której `council_votes` ma kolumnę `round` — ta sama, na którą
# czeka debata (`FLAG_REQUIRED_MIGRATION`), więc numer ma jedno źródło.
_COUNCIL_VOTE_ROUNDS_MIGRATION = FLAG_REQUIRED_MIGRATION["debate_mode_enabled"]

# WIA-70 (migracja 043): dosłowny plaintext commitmentu.
_COMMITMENT_PAYLOAD_COLUMN = "commitment_payload"


def _is_missing_column(exc: APIError, column: str) -> bool:
    """Czy błąd PostgREST zgłasza brak właśnie tej kolumny (migracja nie zaaplikowana)."""
    return exc.code in _MISSING_COLUMN_CODES and column in (exc.message or "")


# NIEZ-16 (decyzja 2026-10-23: ściśle cztery kody): brak funkcji RPC (PGRST202,
# 42883) albo tabeli (PGRST205, 42P01). Tylko wtedy odczyt graceful wskazuje
# migrację. Brak kolumny (`_MISSING_COLUMN_CODES`) ma osobną, tolerancyjną ścieżkę
# odczytu, a zerwane połączenie, timeout albo wstrzymany projekt to inna
# diagnoza — oba logowane z typem wyjątku i tracebackiem.
_MISSING_SCHEMA_CODES = frozenset({"PGRST202", "PGRST205", "42P01", "42883"})


def _log_degraded_read(source: str, skipped: str, migration: str, exc: Exception) -> None:
    """WARNING odczytu graceful: „apply migration” wyłącznie przy brakującym obiekcie
    schematu, każdy inny błąd z typem wyjątku i tracebackiem."""
    if isinstance(exc, APIError) and exc.code in _MISSING_SCHEMA_CODES:
        logger.warning("%s unavailable — %s (apply migration %s).", source, skipped, migration)
        return
    logger.warning("%s failed (%s) — %s.", source, type(exc).__name__, skipped, exc_info=True)


# ARCH-33: PostgREST nie dosięgnął bazy (PGRST000–003: brak połączenia, cache
# schematu, timeout puli), Postgres przerwał zapytanie (57014 statement_timeout)
# albo odrzucił połączenie (53300 too_many_connections).
_UNAVAILABLE_CODES = frozenset({"PGRST000", "PGRST001", "PGRST002", "PGRST003", "57014", "53300"})


def _port_error_kind(exc: APIError | httpx.HTTPError) -> type[PortError]:
    """Klasa taksonomii dla błędu PostgREST: brak bazy (kod albo zerwany transport
    httpx — połączenie, timeout) → niedostępny dostawca, brak obiektu schematu albo
    kolumny → NotFound, reszta → ogólny błąd portu."""
    if isinstance(exc, httpx.TransportError):
        return ProviderUnavailable
    if isinstance(exc, httpx.HTTPError):
        return PortError
    if exc.code in _UNAVAILABLE_CODES:
        return ProviderUnavailable
    if exc.code in _MISSING_SCHEMA_CODES or exc.code in _MISSING_COLUMN_CODES:
        return NotFound
    return PortError


def _translates_api_errors[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """ARCH-33: `APIError` i błąd transportu httpx uciekające z metody wychodzą
    przetłumaczone na taksonomię portów (wciąż `APIError` z kodem / typ httpx)."""

    @functools.wraps(method)
    def translating(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return method(*args, **kwargs)
        except (APIError, httpx.HTTPError) as exc:
            raise as_port_error(exc, _port_error_kind(exc)) from None

    return translating


def _translates_public_methods[C: type](cls: C) -> C:
    """ARCH-33: każda publiczna metoda klasy tłumaczy błędy dostawcy — także metoda
    dopisana później, bez pamiętania o dekoratorze. Metody z własną ścieżką
    degradacji działają jak dotąd (łapią wyjątek, zanim dojdzie do tłumaczenia)."""
    for name, member in list(vars(cls).items()):
        if inspect.isfunction(member) and not name.startswith("_"):
            setattr(cls, name, _translates_api_errors(member))
    return cls


class MissingColumnError(RuntimeError):
    """Odczyt prosił o kolumnę, której baza jeszcze nie ma — migracja nie
    została zaaplikowana. `column` to nazwa kolumny `prediction_logs`."""

    def __init__(self, column: str) -> None:
        super().__init__(f"column prediction_logs.{column} does not exist")
        self.column = column


# FIN-05: select po nieistniejącej kolumnie — Postgres `undefined_column`.
_UNDEFINED_COLUMN_CODE = "42703"
_UNDEFINED_COLUMN_MESSAGE = re.compile(r'column "?(?:\w+\.)?"?(\w+)"? does not exist')


def _undefined_column(exc: APIError) -> str | None:
    """Nazwa kolumny z komunikatu 42703 („column prediction_logs.x does not exist”)."""
    match = _UNDEFINED_COLUMN_MESSAGE.search(exc.message or "")
    return match.group(1) if match else None


def _missing_eval_column(exc: APIError, fields: Sequence[str]) -> str | None:
    """Kolumna z `fields`, której brak zgłasza błąd PostgREST; None, gdy to inny błąd."""
    if exc.code not in _MISSING_COLUMN_CODES:
        return None
    message = exc.message or ""
    for name in fields:
        column = _EVAL_EXTRA_FIELDS[name].column
        if column in message:
            return column
    return None


def _missing_column(exc: APIError, columns: Sequence[str]) -> str | None:
    """NIEZ-32: kolumna z `columns`, której brak zgłasza PostgREST (42703); None
    przy innym błędzie. Najdłuższe nazwy najpierw — `confidence` vs `confidence_score`."""
    if exc.code not in _MISSING_COLUMN_CODES:
        return None
    message = exc.message or ""
    for column in sorted(columns, key=len, reverse=True):
        if re.search(rf"\b{re.escape(column)}\b", message):
            return column
    return None


# Flaga, którą predict_node zostawia przy neutralnej atrapie po awarii głównego
# LLM („SIDEWAYS 50%”). Taki wiersz nie jest predykcją i nie może zasilać
# hit-rate ani kalibracji (WIA-09).
_LLM_FALLBACK_FLAG = "llm_analysis_failed"
# WIA-49: ślad wygaśnięcia predykcji, której reflect już nigdy nie oceni. Żyje
# w `data_quality_flags` (migracja 008), więc nie wymaga nowej kolumny; wiersz
# nie dostaje ceny, więc i tak nie wchodzi do ml_feature_store ani do trafności.
EXPIRED_UNVERIFIED_FLAG = "expired_unverified"
# Filtr PostgREST: brak flag (NULL, wiersze sprzed 008) albo lista bez flagi wygaśnięcia.
_NOT_EXPIRED_FILTER = (
    f'data_quality_flags.is.null,data_quality_flags.not.cs.["{EXPIRED_UNVERIFIED_FLAG}"]'
)


def _without_llm_fallback(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Odsiewa wiersze-atrapy po awarii LLM (flaga `llm_analysis_failed`).

    Filtr po stronie Pythona, nie PostgREST: `NOT (flags @> ...)` daje NULL dla
    wiersza z `data_quality_flags = NULL`, więc filtr w zapytaniu po cichu
    wyciąłby też takie wiersze. Wymaga `data_quality_flags` w SELECT.
    """
    return [row for row in rows if _LLM_FALLBACK_FLAG not in (row.get("data_quality_flags") or [])]


# DAT-14 (migracja 055): rozliczenie z notowania sprzed sesji po predykcji.
_STALE_QUOTE_COLUMN = "stale_quote"


def _without_stale_quote(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bez rozliczeń z nieaktualnego notowania (DAT-14, `stale_quote` = true).

    Bez kolumny w SELECT (flaga off) `stale_quote` jest None, więc filtr nic nie tnie."""
    return [row for row in rows if row.get(_STALE_QUOTE_COLUMN) is not True]


def _countable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wiersze, które liczą się do trafności: bez atrap LLM (WIA-09) i bez
    rozliczeń z nieaktualnego notowania (DAT-14)."""
    return _without_stale_quote(_without_llm_fallback(rows))


def _price_level_alert(row: Mapping[str, Any]) -> PriceLevelAlert:
    """Wiersz `price_level_alerts` → encja domeny (NUMERIC przez tekst do Decimal)."""
    last_seen = row.get("last_seen_price")
    return PriceLevelAlert(
        id=str(row["id"]),
        level=PriceLevel(
            symbol=str(row["symbol"]),
            level=Decimal(str(row["level"])),
            direction=LevelDirection(str(row["direction"])),
            note=_optional_text(row.get("note")),
        ),
        last_seen_price=Decimal(str(last_seen)) if last_seen is not None else None,
    )


def _optional_float(value: Any) -> float | None:
    """Liczba z kolumny NUMERIC (PostgREST oddaje float albo tekst) albo None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _settled_shadow(row: Mapping[str, Any]) -> SettledShadowForecast | None:
    """ML-33: wiersz cienia z osadzoną predykcją → prognoza ze zwrotem zrealizowanym.
    None, gdy predykcja nie ma dodatniej ceny albo rozliczenia, albo prognoza łamie
    kontrakt domeny (np. p spoza [0, 1] wpisane do bazy ręcznie)."""
    prediction = row.get("prediction_logs")
    if not isinstance(prediction, Mapping):
        return None
    price = _optional_float(prediction.get("price_at_prediction"))
    actual = _optional_float(prediction.get("actual_price_after_12h"))
    if price is None or actual is None or price <= 0:
        return None
    try:
        forecast = ShadowForecast(
            model_name=str(row.get("model_name") or ""),
            model_version=str(row.get("model_version") or ""),
            predicted_return=_optional_float(row.get("predicted_return")),
            proba=_optional_float(row.get("proba")),
        )
    except ValueError:
        return None
    return SettledShadowForecast(str(row.get("prediction_id")), forecast, actual / price - 1.0)


def _optional_timestamp(value: Any) -> datetime | None:
    """Znacznik czasu z wiersza PostgREST (ISO 8601) albo None, gdy go brak."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _feature_snapshot(row: Mapping[str, Any], feature_set_version: int) -> FeatureSnapshot | None:
    """Wiersz `feature_snapshots` jako domenowy wektor; bez symbolu, chwili albo
    obiektu JSON cech → None (wiersz pominięty, nie zgadujemy)."""
    symbol = row.get("symbol")
    as_of = _optional_timestamp(row.get("as_of"))
    features = row.get("features")
    if not isinstance(symbol, str) or not symbol or as_of is None:
        return None
    if not isinstance(features, Mapping):
        return None
    version = row.get("feature_set_version")
    return FeatureSnapshot(
        symbol=symbol,
        as_of=as_of,
        gated=row.get("gated") is True,
        features=dict(features),
        feature_set_version=version if isinstance(version, int) else feature_set_version,
    )


def _resolved_window(query: Any, cutoff: datetime, since: datetime | None) -> Any:
    """Okno zamkniętych predykcji (ocenionych kierunkowo), od najświeższej.

    Bez `since` — po dacie UTWORZENIA od `cutoff`, jak przed WIA-19. Z `since` —
    po chwili ROZLICZENIA (`settled_at`, migracja 029) od `since`: piątkowa
    predykcja rozliczona w poniedziałek wpada do poniedziałkowego okna. Kolumny
    przed migracją nie ma (filtr po niej → 400), więc bez `since` zapytanie jej
    nie dotyka."""
    column, start = ("timestamp", cutoff) if since is None else ("settled_at", since)
    return (
        query.gte(column, start.isoformat())
        .not_.is_("is_trend_correct", "null")
        .order(column, desc=True)
    )


def _utc_day_key(stamp: Any) -> str:
    """Klucz cyklu predykcji: dzień UTC w ISO (jak `report_evidence.cycle_key`).

    Brak albo nieczytelny znacznik → wspólny klucz "" — na niekorzyść dowodu,
    bo takie wiersze lądują w jednym cyklu."""
    if not isinstance(stamp, str):
        return ""
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return ""
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return moment.date().isoformat()


def _utc_hour_start(moment: datetime) -> datetime:
    """Początek godziny UTC chwili — ta sama granica co generowana kolumna
    `timestamp_hour` (`date_trunc('hour', timestamp, 'UTC')`, migracja 013).
    Chwila bez strefy to UTC."""
    utc = moment.astimezone(UTC) if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return utc.replace(minute=0, second=0, microsecond=0)


def _serialize(value: Any) -> Any:
    """Decimal → str (JSON-safe, zachowuje precyzję) — pozostałe typy bez zmian."""
    if isinstance(value, Decimal):
        return str(value)
    return value


# ARCH-38: kolumny cenowe `prediction_logs` w skali DECIMAL(14,4) (001, 032, 059).
_PREDICTION_PRICE_COLUMNS = frozenset(
    {
        "price_at_prediction",
        "predicted_target_price",
        "actual_price_after_12h",
        "interval_low",
        "interval_high",
        "settlement_close_price",
        "entry_price",
    }
)

# WIA-61 (migracja 059): trafność kierunku na oficjalnym zamknięciu sesji.
_SESSION_CLOSE_CORRECT_COLUMN = "is_trend_correct_session_close"


def _session_close_payload(settlement: SessionCloseSettlement) -> dict[str, Any]:
    """WIA-61: zamknięcie sesji z datą i źródłem obok ceny /quote — z nich
    weryfikator sprawdzi cenę w publicznym serwisie."""
    close = settlement.close
    return {
        "settlement_close_price": close.price.amount,
        "settlement_session_date": close.session_day.isoformat(),
        "settlement_source": close.source.value,
        _SESSION_CLOSE_CORRECT_COLUMN: settlement.is_trend_correct,
    }


def _session_close_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    """WIA-61: trafność na zamknięciu sesji i, dla porównania, trafność /quote na
    TYCH SAMYCH predykcjach — tylko wiersze z zapisanym zamknięciem."""
    with_close = [row for row in rows if row.get(_SESSION_CLOSE_CORRECT_COLUMN) is not None]
    return {
        "session_close_count": len(with_close),
        "session_close_correct_count": sum(
            1 for row in with_close if row[_SESSION_CLOSE_CORRECT_COLUMN]
        ),
        "session_close_quote_correct_count": sum(
            1 for row in with_close if row.get("is_trend_correct")
        ),
    }


def _storage_price(value: Decimal) -> Decimal:
    """ARCH-38: cena w skali kolumny DECIMAL(14,4), połówki od zera jak w Postgresie.

    Wartość, której kwantyzacja nie zmienia, zostaje z własnym literałem
    („99.0”, nie „99.0000”) — baza i tak przechowa ją identycznie."""
    stored = Money(value).for_storage().amount
    return value if stored == value else stored


def _serialize_record(
    record: dict[str, Any], price_columns: frozenset[str] = frozenset()
) -> dict[str, Any]:
    return {
        k: _serialize(_storage_price(v) if k in price_columns and isinstance(v, Decimal) else v)
        for k, v in record.items()
    }


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


# WIA-56: kolumny kwitu decyzyjnego z kontekstem predykcji (migracja 020).
_RECEIPT_COLUMNS = "id, symbol, timestamp, predicted_trend, is_trend_correct, decision_receipts"


def _stored_receipt(row: Mapping[str, Any]) -> StoredReceipt:
    """Wiersz `prediction_logs` → `StoredReceipt`; kwit niebędący obiektem JSON → None."""
    payload = row.get("decision_receipts")
    stamp = row.get("timestamp")
    correct = row.get("is_trend_correct")
    return StoredReceipt(
        prediction_id=str(row["id"]),
        symbol=str(row.get("symbol") or "?"),
        predicted_at=datetime.fromisoformat(stamp) if isinstance(stamp, str) else None,
        predicted_trend=_optional_text(row.get("predicted_trend")),
        is_trend_correct=correct if isinstance(correct, bool) else None,
        payload=payload if isinstance(payload, Mapping) else None,
    )


# WIA-53: wejście diagnozy błędnej predykcji (kwit `decision_receipts` osobno, 020).
_DEFERRED_COLUMNS = (
    "id, symbol, timestamp, predicted_trend, price_at_prediction, "
    "predicted_target_price, actual_price_after_12h, news_summary, reasoning_text, "
    "correction_insights, is_trend_correct, data_quality_flags"
)


def _deferred_diagnosis(row: Mapping[str, Any]) -> DeferredDiagnosis | None:
    """Wiersz `prediction_logs` → `DeferredDiagnosis`; bez cen nie ma czego diagnozować."""
    price = row.get("price_at_prediction")
    actual = row.get("actual_price_after_12h")
    if price is None or actual is None or row.get("id") is None:
        return None
    trend, _ = normalize_trend(row.get("predicted_trend"))
    target = row.get("predicted_target_price")
    receipt = row.get("decision_receipts")
    correct = row.get("is_trend_correct")
    return DeferredDiagnosis(
        prediction=Prediction(
            id=str(row["id"]),
            symbol=str(row.get("symbol") or "?"),
            predicted_trend=trend,
            price_at_prediction=Decimal(str(price)),
            predicted_target_price=Decimal(str(target if target is not None else price)),
            news_summary=_optional_text(row.get("news_summary")),
            reasoning=_optional_text(row.get("reasoning_text")),
        ),
        actual_price=Decimal(str(actual)),
        is_trend_correct=correct if isinstance(correct, bool) else None,
        placeholder=str(row.get("correction_insights") or ""),
        predicted_at=_optional_timestamp(row.get("timestamp")),
        receipt=receipt if isinstance(receipt, Mapping) else None,
    )


def _judgeable_diagnosis(row: Mapping[str, Any]) -> JudgeableDiagnosis | None:
    """EXP-10: wiersz `prediction_logs` → diagnoza do oceny (te same kolumny co WIA-53)."""
    deferred = _deferred_diagnosis(row)
    if deferred is None:
        return None
    return JudgeableDiagnosis(
        prediction=deferred.prediction,
        actual_price=deferred.actual_price,
        insight=deferred.placeholder,
        predicted_at=deferred.predicted_at,
    )


def _persona_vote_outcome(row: Mapping[str, Any]) -> PersonaVoteOutcome | None:
    """Wiersz RPC `persona_vote_outcomes` (045) → głos domeny; bez persony,
    rekomendacji albo cen nie ma czego oceniać → None."""
    name = row.get("investor_name")
    recommendation = row.get("recommendation")
    price = row.get("price_at_prediction")
    actual = row.get("actual_price_after_12h")
    if name is None or recommendation is None or price is None or actual is None:
        return None
    return PersonaVoteOutcome(
        prediction_id=str(row.get("prediction_id") or ""),
        investor_name=str(name),
        recommendation=str(recommendation),
        confidence=float(row.get("confidence") or 0.0),
        symbol=str(row.get("symbol") or ""),
        price_at_prediction=Decimal(str(price)),
        actual_price=Decimal(str(actual)),
        predicted_at=_optional_timestamp(row.get("predicted_at")),
        settled_at=_optional_timestamp(row.get("settled_at")),
    )


# WIA-59 (migracja 073): append-only paper ledger.
PAPER_TRADES_TABLE = "paper_trades"
_PAPER_LEDGER_APPEND_ATTEMPTS = 3
# Przegrany wyścig o głowę łańcucha: UNIQUE(prev_hash) albo trigger głowy (40001).
_PAPER_LEDGER_CONFLICT_CODES = frozenset({"23505", "40001"})
_PAPER_TRADE_COLUMNS = (
    "seq, prediction_id, symbol, side, entry_price, entry_ts, weight, cost_bps, "
    "prev_hash, row_hash, exit_price, exit_ts, pnl, exit_reason"
)


def _paper_trade_record(row: LedgerRow) -> dict[str, Any]:
    """Wiersz `paper_trades` z tych samych wartości, które weszły do hasha."""
    payload = row.entry.payload()
    payload.pop("ledger_version")
    return {**payload, "prev_hash": row.prev_hash, "row_hash": row.row_hash}


def _ledger_trade(raw: Mapping[str, Any]) -> LedgerTrade:
    """Wiersz bazy → wpis łańcucha domeny (ceny przez `str`, nigdy przez float)."""
    entry = PaperTradeEntry(
        prediction_id=str(raw["prediction_id"]),
        symbol=str(raw["symbol"]),
        side=TradeSide(raw["side"]),
        entry_price=Decimal(str(raw["entry_price"])),
        entry_ts=datetime.fromisoformat(str(raw["entry_ts"])),
        cost_bps=Decimal(str(raw["cost_bps"])),
        weight=Decimal(str(raw["weight"])),
    )
    row = LedgerRow(entry=entry, prev_hash=str(raw["prev_hash"]), row_hash=str(raw["row_hash"]))
    close = None
    if raw.get("exit_price") is not None:
        close = PaperTradeClose(
            exit_price=Decimal(str(raw["exit_price"])),
            exit_ts=datetime.fromisoformat(str(raw["exit_ts"])),
            pnl=Decimal(str(raw["pnl"])),
            exit_reason=ExitReason(raw["exit_reason"]),
        )
    return LedgerTrade(seq=int(raw["seq"]), row=row, close=close)


def _rows(response: Any) -> list[dict[str, Any]]:
    """Narrow supabase response.data (JSON union) do list[dict]."""
    data = response.data or []
    return cast(list[dict[str, Any]], data)


def _recommendation_or_hold(raw: Any, field: str) -> Recommendation:
    """ARCH-03: rekomendacja z JSONB bez rozróżniania wielkości liter; wartość spoza
    BUY/SELL/HOLD → HOLD z ostrzeżeniem, zamiast wpuścić ją do werdyktu rady."""
    parsed = parse_recommendation(raw)
    if parsed is None:
        logger.warning("council_verdict: unknown %s %r — read as HOLD", field, raw)
        return "HOLD"
    return parsed


def _deserialize_verdict(raw: dict[str, Any]) -> CouncilVerdict:
    """Odtwarza `CouncilVerdict` z JSONB (migracja 005). Opinie inwestorów są
    zapisane jako lista dictów — mapujemy je z powrotem na value objecty."""
    opinions = tuple(
        InvestorOpinion(
            investor_name=str(o.get("investor_name", "")),
            recommendation=_recommendation_or_hold(
                o.get("recommendation", "HOLD"), "recommendation"
            ),
            confidence=float(o.get("confidence") or 0.0),
            reasoning=str(o.get("reasoning", "")),
            key_factors=tuple(o.get("key_factors") or ()),
        )
        for o in raw.get("investor_opinions") or []
    )
    return CouncilVerdict(
        final_recommendation=_recommendation_or_hold(
            raw.get("final_recommendation", "HOLD"), "final_recommendation"
        ),
        consensus_strength=float(raw.get("consensus_strength") or 0.0),
        summary=str(raw.get("summary", "")),
        dissenting_views=tuple(raw.get("dissenting_views") or ()),
        investor_opinions=opinions,
        failed_personas=_failed_personas_of(raw.get("failed_personas")),
    )


def _failed_personas_of(raw: Any) -> int:
    """MAIL-12: liczba person bez głosu z JSONB. Brak pola (werdykt sprzed MAIL-12)
    albo wartość nieczytelna → 0; nie zgadujemy awarii z treści opinii."""
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


@_translates_public_methods
class SupabaseRepository(RepositoryPort, EvidenceSourcePort, TrainingDataPort):
    """Adapter dla Supabase (PostgreSQL via PostgREST).

    Tabela `prediction_logs` + zmaterializowany widok `ml_feature_store`
    — relacje, klasy danych, wymagany klucz i RLS opisuje docs/data-and-access.md.
    """

    def __init__(
        self,
        url: str,
        key: str,
        table_name: str = DEFAULT_TABLE,
        feature_view: str = DEFAULT_FEATURE_VIEW,
        snapshot_table: str = DEFAULT_SNAPSHOT_TABLE,
        council_votes_table: str = DEFAULT_COUNCIL_VOTES_TABLE,
        quota_alerts_table: str = DEFAULT_QUOTA_ALERTS_TABLE,
        *,
        scorecard_baselines_enabled: bool = False,
        scorecard_run_id_enabled: bool = False,
        council_votes_upsert_enabled: bool = False,
        scorecard_feature_stats_enabled: bool = False,
        persona_vote_outcomes_enabled: bool = False,
        crypto_symbols: Collection[str] = (),
        correction_category_enabled: bool = False,
        stale_quote_enabled: bool = False,
        fundamentals_market_cap_enabled: bool = False,
        session_close_settlement_enabled: bool = False,
        rag_exact_search_enabled: bool = False,
        rag_point_in_time_enabled: bool = False,
        receipts_enabled: bool = True,
        scorecard_weights_sha256_enabled: bool = False,
        scorecard_provenance_enabled: bool = False,
        executable_entry_enabled: bool = False,
        council_vote_rounds_enabled: bool = False,
        council_vote_rounds_from_schema: bool = False,
    ) -> None:
        self._client = create_client(url, key)
        # ML-23: dokładne KNN analogów RAG (funkcja z migracji 063) zamiast ivfflat
        # z probes = 1. Bez flagi RPC sprzed ML-23 — analogi w prompcie jak dziś.
        self._rag_function, self._rag_migration = (
            ("match_news_embeddings_exact", "063")
            if rag_exact_search_enabled
            else ("match_news_embeddings", "011")
        )
        # ML-41: analogi rozliczone przed chwilą predykcji, bez tego samego symbolu
        # w oknie horyzontu (funkcja z migracji 076, dokładne KNN). Ma pierwszeństwo
        # przed ML-23. Bez flagi RPC jak wyżej.
        self._rag_point_in_time_enabled = rag_point_in_time_enabled
        # WIA-56: odczyt rozliczeń niesie kwit (`decision_receipts`, 020) do linii pod
        # chybieniem tylko przy włączonych kwitach. Domyślnie true — bezpośrednia
        # konstrukcja (narzędzia, testy) czyta jak dotąd; Fast Loop podaje flagę z configu.
        self._receipts_enabled = receipts_enabled
        # ML-22: kolumny proweniencji treningu istnieją od migracji 064 — ta sama zasada.
        self._scorecard_provenance_enabled = scorecard_provenance_enabled
        # CI-13: kolumna `model_scorecards.weights_sha256` istnieje od migracji 060 —
        # bez flagi payload karty jej nie niesie (PGRST204 zgubiłby cały wiersz).
        self._scorecard_weights_sha256_enabled = scorecard_weights_sha256_enabled
        # DAT-14: kolumna `stale_quote` istnieje od migracji 055 — bez flagi odczyty
        # trafności jej nie wybierają (400 zabiłby raport), a zapis jej nie niesie.
        self._stale_quote_enabled = stale_quote_enabled
        # DAT-21: kolumna `fundamentals_cache.market_cap` istnieje od migracji 056 —
        # bez flagi upsert jej nie niesie (PGRST204 zgubiłby cały wpis cache).
        self._fundamentals_market_cap_enabled = fundamentals_market_cap_enabled
        # WIA-61: kolumny zamknięcia sesji istnieją od migracji 059 — bez flagi
        # odczyt trafności ich nie wybiera (400 zabiłby raport).
        self._session_close_settlement_enabled = session_close_settlement_enabled
        # WIA-54: kolumna `entry_price` istnieje od migracji 072 — bez flagi odczyt
        # track recordu jej nie wybiera (400 zabiłby raport).
        self._executable_entry_enabled = executable_entry_enabled
        # ML-10: kolumny baseline'ów dryfu i „sam LLM" istnieją od migracji 028 —
        # bez flagi payload karty ich nie niesie (PGRST204 zgubiłby cały wiersz).
        self._scorecard_baselines_enabled = scorecard_baselines_enabled
        # ML-24: kolumna `run_id` istnieje od migracji 049 — ta sama zasada.
        self._scorecard_run_id_enabled = scorecard_run_id_enabled
        # NIEZ-14: upsert głosów potrzebuje UNIQUE z migracji 054 — bez niego
        # PostgREST odrzuca `on_conflict` (42P10), więc do tego czasu goły insert.
        # ML-40: od migracji 082 klucz to (prediction_id, investor_name, round), więc
        # flaga czeka na 082 (`FLAG_REQUIRED_MIGRATION`), a cel upsertu niesie rundę.
        self._council_votes_upsert_enabled = council_votes_upsert_enabled
        # ML-40: z debatą `council_votes` ma też rundę rewizji (kolumna `round`,
        # migracja 082); odczyty głosów biorą wtedy niezależną rundę 1. Bez flagi,
        # z `council_vote_rounds_from_schema` (entry-pointy), o filtrze decyduje
        # `schema_version()`: wiersze rundy 2 z okresu z debatą zostają w tabeli, a
        # przed 082 filtr padłby (42703). Złożone wprost — bez sondy, jak dotąd.
        self._council_vote_rounds_enabled = council_vote_rounds_enabled
        self._council_vote_rounds_from_schema = council_vote_rounds_from_schema
        self._vote_rounds_in_schema: bool | None = None
        # ML-21: kolumna JSONB `feature_stats` istnieje od migracji 052 — ta sama zasada.
        self._scorecard_feature_stats_enabled = scorecard_feature_stats_enabled
        # WIA-51: wagi i ranking person z surowych głosów RPC 045 regułą domeny
        # (pasmo HOLD krypto z `crypto_symbols`). Bez migracji RPC nie istnieje.
        self._persona_vote_outcomes_enabled = persona_vote_outcomes_enabled
        self._crypto_symbols = frozenset(symbol.upper() for symbol in crypto_symbols)
        # WIA-57: kolumna `correction_category` istnieje od migracji 046 — zapis
        # i odczyt panelu lessons tylko z flagą (inaczej PGRST204 / 42703).
        self._correction_category_enabled = correction_category_enabled
        self._table = table_name
        self._view = feature_view
        self._snapshot_table = snapshot_table
        self._council_votes_table = council_votes_table
        self._quota_alerts_table = quota_alerts_table

    def _countable_columns(self, columns: str) -> str:
        """DAT-14: SELECT odczytu trafności z kolumną `stale_quote`, gdy ją mamy
        (flaga po migracji 055) — `_countable` odsiewa po niej rozliczenia."""
        if not self._stale_quote_enabled:
            return columns
        return f"{columns}, {_STALE_QUOTE_COLUMN}"

    # -----------------------------------------------------------------------
    # Reads
    # -----------------------------------------------------------------------

    def get_last_price(self, symbol: str) -> Money | None:
        # Czytamy z price_snapshots — zapisywane w KAŻDYM cyklu, więc nawet
        # gdy bot dotąd nie zrobił żadnej predykcji, ma punkt odniesienia.
        response = (
            self._client.table(self._snapshot_table)
            .select("price")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        return Money(Decimal(str(rows[0]["price"])))

    def get_price_history(self, symbol: str, days: int) -> list[tuple[datetime, Money]]:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        # PORT-22: radar czyta 90 dni — paginacja (#8), bo PostgREST tnie SELECT
        # do ~1000 wierszy bez błędu i najstarsze snapshoty znikałyby po cichu.
        query = (
            self._client.table(self._snapshot_table)
            .select("timestamp, price")
            .eq("symbol", symbol)
            .gte("timestamp", cutoff.isoformat())
            .order("timestamp", desc=False)
        )
        out: list[tuple[datetime, Money]] = []
        for row in _paginate(query):
            ts = row.get("timestamp")
            price = row.get("price")
            if not isinstance(ts, str) or price is None:
                continue
            out.append((datetime.fromisoformat(ts), Money(Decimal(str(price)))))
        return out

    def get_last_prediction_price(
        self, symbol: str, *, before: datetime | None = None
    ) -> Money | None:
        # Cena ostatniej zalogowanej predykcji — referencja dla cechy price_delta
        # w inference (musi pokrywać się z LAG(price_at_prediction) w widoku
        # ml_feature_store, inaczej cecha ma inny rozkład w treningu i predykcji).
        query = self._client.table(self._table).select("price_at_prediction").eq("symbol", symbol)
        if before is not None:
            # ML-18: bez wiersza z godziny, którą ten zapis nadpisze upsertem.
            query = query.lt("timestamp_hour", _utc_hour_start(before).isoformat())
        response = query.order("timestamp", desc=True).limit(1).execute()
        rows = _rows(response)
        if not rows or rows[0].get("price_at_prediction") is None:
            return None
        return Money(Decimal(str(rows[0]["price_at_prediction"])))

    def get_latest_prediction(self, symbol: str) -> LatestPrediction | None:
        # SUROWO, bez try/except — watch i bramka dryfu degradują same.
        response = (
            self._client.table(self._table)
            .select("price_at_prediction, timestamp, reasoning_text")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        row = rows[0]
        price, stamp = row.get("price_at_prediction"), row.get("timestamp")
        if price is None or not isinstance(stamp, str):
            return None
        reasoning = (_optional_text(row.get("reasoning_text")) or "").strip()
        return LatestPrediction(
            price=Money(Decimal(str(price))),
            timestamp=datetime.fromisoformat(stamp),
            reasoning=reasoning or None,
        )

    def get_unverified_prediction(
        self, symbol: str, min_age_hours: int = 0, max_age_hours: int | None = None
    ) -> Prediction | None:
        query = (
            self._client.table(self._table)
            .select("*")
            .eq("symbol", symbol)
            .is_("actual_price_after_12h", "null")
        )
        # Górne ograniczenie wieku — pomija predykcje zbyt świeże, by je
        # rzetelnie ocenić (patrz docstring portu). Filtr dokładamy tylko gdy
        # próg > 0, by nie zmieniać zachowania domyślnego.
        if min_age_hours > 0:
            cutoff = datetime.now(UTC) - timedelta(hours=min_age_hours)
            query = query.lte("timestamp", cutoff.isoformat())
        # WIA-04: dolna granica znacznika czasu — predykcji sprzed tygodni nie
        # oceniamy bieżącą ceną. Bez granicy (None) zachowanie bez zmian.
        if max_age_hours is not None and max_age_hours > 0:
            oldest = datetime.now(UTC) - timedelta(hours=max_age_hours)
            query = query.gte("timestamp", oldest.isoformat())
        # WIA-49: wiersz oznaczony `expired_unverified` jest już poza oceną — bez tego
        # filtra wracałby, gdy `max_age_hours` jest puste albo szersze niż przy
        # oznaczeniu. `OR ... IS NULL`: wiersze sprzed migracji 008 nie mają flag
        # (samo `NOT cs` wycięłoby NULL), a one zostają w kolejce.
        query = query.or_(_NOT_EXPIRED_FILTER)
        # ASC (oldest-first) — #19: opróżniamy backlog, rozliczając NAJSTARSZĄ
        # nierozliczoną predykcję, która przekroczyła próg min_age_hours. Przy
        # DESC starsze nierozliczone predykcje nigdy nie stałyby się znów
        # "najnowsze" i utknęłyby na zawsze (brak actual_price_after_12h →
        # nie wchodzą do ml_feature_store, nie liczą się do hit-rate).
        response = query.order("timestamp", desc=False).limit(1).execute()
        rows = _rows(response)
        if not rows:
            return None
        row = rows[0]
        return Prediction(
            id=str(row["id"]),
            symbol=str(row["symbol"]),
            # Stare wiersze sprzed ML-07 (lowercase, NULL, śmieć) nie mogą blokować
            # kolejki oldest-first — ta sama reguła co przy zapisie nowych.
            predicted_trend=normalize_trend(row.get("predicted_trend"))[0],
            price_at_prediction=Decimal(str(row["price_at_prediction"])),
            predicted_target_price=Decimal(str(row["predicted_target_price"])),
            # WIA-18: wejście predykcji dla diagnozy błędu (select("*") je zwraca).
            news_summary=_optional_text(row.get("news_summary")),
            reasoning=_optional_text(row.get("reasoning_text")),
            # DAT-14: reflect sprawdza, czy notowanie rozliczenia jest z sesji po niej.
            created_at=_optional_timestamp(row.get("timestamp")),
        )

    def get_feature_store_data(
        self, symbol: str, feature_set_version: int | None = None
    ) -> list[dict[str, Any]]:
        # NIEZ-15: paginacja — bez niej trening widział max ~1000 wierszy widoku.
        # `id` jako drugi klucz: granica strony deterministyczna przy równych timestampach.
        query = self._client.table(self._view).select("*").eq("symbol", symbol)
        if feature_set_version is not None:
            # DAT-09: filtr po stronie PostgREST — `eq` odrzuca też NULL (wersja
            # nieznana). Kolumna istnieje dopiero po migracji 027.
            query = query.eq("feature_set_version", feature_set_version)
        return _paginate(query.order("timestamp", desc=False).order("id", desc=False))

    def get_feature_snapshots(self, feature_set_version: int) -> list[FeatureSnapshot]:
        # ML-52: wektor czytany z JSONB bez żadnego przeliczania — trening dostaje
        # dokładnie to, co cykl zapisał (i podał modelowi). `id` jako drugi klucz:
        # granica strony deterministyczna przy równych `as_of` (NIEZ-15).
        query = (
            self._client.table(DEFAULT_FEATURE_SNAPSHOT_TABLE)
            .select("symbol, as_of, gated, features, feature_set_version")
            .eq("feature_set_version", feature_set_version)
        )
        rows = _paginate(query.order("as_of", desc=False).order("id", desc=False))
        return [
            snapshot
            for row in rows
            if (snapshot := _feature_snapshot(row, feature_set_version)) is not None
        ]

    def get_price_series(
        self, symbols: Sequence[str], *, since: datetime
    ) -> dict[str, list[tuple[datetime, float]]]:
        # ML-52: szereg cen do etykiety point-in-time — jeden odczyt dla wszystkich
        # symboli treningu zamiast `get_price_history` per symbol.
        wanted = list(dict.fromkeys(symbols))
        if not wanted:
            return {}
        query = (
            self._client.table(self._snapshot_table)
            .select("symbol, timestamp, price")
            .in_("symbol", wanted)
            .gte("timestamp", since.isoformat())
        )
        series: dict[str, list[tuple[datetime, float]]] = {}
        for row in _paginate(query.order("timestamp", desc=False).order("id", desc=False)):
            moment = _optional_timestamp(row.get("timestamp"))
            price = _optional_float(row.get("price"))
            symbol = row.get("symbol")
            if moment is None or price is None or not isinstance(symbol, str):
                continue
            series.setdefault(symbol, []).append((moment, price))
        return series

    def refresh_feature_store(self) -> None:
        """Wywołuje RPC `refresh_ml_feature_store` (zdef. w migrations/001_init.sql).

        Bez tego widok zmaterializowany pokazuje stan z momentu ostatniego
        REFRESH — co psuje Continual Learning (model uczy się na nieaktualnym
        snapshocie). Wywoływane przez Slow Loop przed treningiem.

        Retry z exponential backoff — transient błąd sieci/Supabase
        (gateway 5xx) nie powinien wywalać całego Slow Loop.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _REFRESH_MAX_ATTEMPTS + 1):
            try:
                self._client.rpc("refresh_ml_feature_store").execute()
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _REFRESH_MAX_ATTEMPTS:
                    delay = _REFRESH_BACKOFF_S * (2 ** (attempt - 1))
                    logger.warning(
                        "refresh_feature_store attempt %d/%d failed (%s) — retrying in %.1fs",
                        attempt,
                        _REFRESH_MAX_ATTEMPTS,
                        type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)
        assert last_exc is not None  # for type narrowing
        raise last_exc

    def get_recently_resolved_predictions(
        self, hours: int, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Predykcje zamknięte (z oceną kierunku) utworzone w ostatnich `hours`
        godzinach — albo, z `since`, rozliczone od `since` (WIA-19).

        Filtr po `is_trend_correct` (nie `accuracy_score`) — to ono napędza
        trafność raportu i naturalnie odsiewa wiersze sprzed migracji 009."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        # Pełen post-mortem w raporcie: oryginalne uzasadnienie + diagnoza + ceny
        # (a nie tylko kierunek), żeby pokazać "dlaczego wzrosło/spadło i czy
        # prognoza się potwierdziła".
        # Paginacja (#8) — bez niej raport pokazałby max ~1000 zamkniętych
        # predykcji (cichy cap PostgREST).
        columns = (
            "symbol, predicted_trend, is_trend_correct, timestamp, "
            "reasoning_text, correction_insights, price_at_prediction, "
            "actual_price_after_12h, data_quality_flags"
        )
        if self._receipts_enabled:
            # WIA-56: kwit (migracja 020, na produkcji) do linii pod chybieniem.
            columns += ", decision_receipts"
        query = _resolved_window(
            self._client.table(self._table).select(self._countable_columns(columns)),
            cutoff,
            since,
        )
        return _countable(_paginate(query))

    def get_resolved_predictions(self, days: int) -> list[dict[str, Any]]:
        """Zamknięte predykcje (oceniony kierunek) z ostatnich `days` dni.

        Ten sam zestaw kolumn co `get_recently_resolved_predictions` (godziny),
        ale szersze okno dniowe — zasila track-record (krzywa kapitału, panel
        lessons). `parse_resolved_predictions` parsuje oba kształty. Paginacja
        (#8) — pełen zbiór, nie obcięte ~1000 wierszy PostgREST."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        columns = (
            "symbol, predicted_trend, is_trend_correct, timestamp, "
            "reasoning_text, correction_insights, price_at_prediction, "
            "actual_price_after_12h, data_quality_flags"
        )
        if self._correction_category_enabled:
            columns += ", correction_category"
        if self._executable_entry_enabled:
            columns += ", entry_price"
        query = (
            self._client.table(self._table)
            .select(self._countable_columns(columns))
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .order("timestamp", desc=True)
        )
        return _countable(_paginate(query))

    def get_resolved_predictions_detailed(
        self, days: int, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """#17 — zamknięte predykcje z polami, których `get_resolved_predictions`
        nie zwraca: `id`, `predicted_target_price`, `confidence_score` i werdykt
        rady (JSONB). Recap tygodniowy liczy z nich accuracy, ECE i wybronionych
        dysydentów. Paginacja jak w pozostałych odczytach track-recordu.
        `since` — okno po chwili rozliczenia (WIA-19), jak w
        `get_recently_resolved_predictions`."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        query = _resolved_window(
            self._client.table(self._table).select(
                self._countable_columns(
                    "id, symbol, predicted_trend, is_trend_correct, timestamp, "
                    "reasoning_text, correction_insights, price_at_prediction, "
                    "predicted_target_price, actual_price_after_12h, "
                    "confidence_score, council_verdict"
                )
            ),
            cutoff,
            since,
        )
        # DAT-14: bez `data_quality_flags` w SELECT `_countable` tnie tylko `stale_quote`.
        return _countable(_paginate(query))

    def get_resolved_predictions_for_eval(
        self,
        days: int,
        *,
        with_recipe: bool = False,
        with_universe: bool = False,
        extra_fields: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Zamknięte predykcje z ostatnich `days` dni z polami do ewaluacji
        (ceny + kierunek). Read-only; używane przez `main_tools.py evaluate`.

        `with_recipe` (WIA-20) dokłada kolumnę `recipe_version` do segmentacji
        po przepisie — WYMAGA migracji 030 (inaczej PostgREST zwraca 400).
        `with_universe` (WIA-48) dokłada `universe_version` z kwitu v2 ścieżką
        JSON po `decision_receipts` (migracja 020) — bez nowej kolumny; wiersze
        bez kwitu v2 mają tu None.

        `extra_fields` (WIA-55, EXP-01) to pola z `_EVAL_EXTRA_FIELDS`, czytane
        tylko na żądanie: każde istnieje dopiero po swojej migracji. Nieznana
        nazwa → ValueError; brak kolumny w bazie → `MissingColumnError`.

        Atrapy po awarii LLM (`llm_analysis_failed`) są pomijane tym samym
        filtrem co mail i Track Record — to nie są predykcje.

        Metoda concrete-only (nie część `RepositoryPort`) — to narzędzie
        offline-analizy, nie część runtime'owego kontraktu use case'ów."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        columns = self._countable_columns(
            "predicted_trend, is_trend_correct, price_at_prediction, "
            "predicted_target_price, actual_price_after_12h, timestamp, "
            "data_quality_flags"
        )
        if with_recipe:
            columns += ", recipe_version"
        if with_universe:
            columns += ", universe_version:decision_receipts->>universe_version"
        for name in extra_fields:
            if name not in _EVAL_EXTRA_FIELDS:
                raise ValueError(f"unknown evaluation field: {name}")
            columns += f", {_EVAL_EXTRA_FIELDS[name].select}"
        # Paginacja (#8) — ewaluacja musi widzieć pełen zbiór, nie obcięte 1000.
        query = (
            self._client.table(self._table)
            .select(columns)
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("actual_price_after_12h", "null")
            .order("timestamp", desc=True)
        )
        try:
            return _countable(_paginate(query))
        except APIError as exc:
            missing = _missing_eval_column(exc, extra_fields)
            if missing is None:
                raise
            raise MissingColumnError(missing) from exc

    def get_settled_moves(self, days: int, *, with_intervals: bool = False) -> list[dict[str, Any]]:
        # ML-34: rosnąco po czasie, bo replay ACI idzie chronologicznie. Paginacja
        # (#8) — okno kalibracyjne nie może być po cichu obcięte do ~1000 wierszy.
        cutoff = datetime.now(UTC) - timedelta(days=days)
        columns = (
            "symbol, timestamp, price_at_prediction, predicted_target_price, actual_price_after_12h"
        )
        if with_intervals:
            columns += ", interval_low, interval_high"
        query = (
            self._client.table(self._table)
            # DAT-14: rozliczenie ceną wejścia (notowanie sprzed sesji) to nie ruch.
            .select(self._countable_columns(columns))
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("actual_price_after_12h", "null")
            .order("timestamp", desc=False)
        )
        return _without_stale_quote(_paginate(query))

    def get_accuracy_stats(self, days: int) -> dict[str, Any]:
        """Trafność kierunkowa predykcji z ostatnich `days` dni.

        `mean_accuracy` to hit-rate (udział trafionych kierunkowo predykcji),
        liczony z `is_trend_correct` — NIE z `accuracy_score` (bliskość ceny do
        celu, przy cold-starcie zawsze ~100% niezależnie od kierunku).

        Zwraca dict z kluczami: `mean_accuracy`, `sample_count`, `correct_count`,
        `days_window`. Gdy brak danych → mean_accuracy=None. Atrapy po awarii
        LLM (`llm_analysis_failed`) nie są predykcjami i nie wchodzą do próby.

        WIA-29: `mean_accuracy` liczy też prognozy SIDEWAYS („bez ruchu”) — na
        nim stoi sizing, więc semantyka zostaje. Obok, z tej samej próby,
        `directional_count` / `directional_correct_count`: tylko prognozy
        z zakładem na kierunek (`directional_hit_rate`, ta sama definicja co
        Track Record), żeby mail pokazał obie liczby z jasnymi etykietami.

        PORT-01: `n_eff` to efektywna liczność tej samej próby (`effective_n`
        po dniach UTC, jak w Track Record): predykcje jednego dnia ruszają się
        razem, więc sizing patrzy na liczbę niezależnych zakładów, nie wierszy.
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        # Paginacja (#8): hit-rate liczony na PEŁNYM zbiorze, nie obciętych 1000
        # wierszy. .order("timestamp") czyni podzbiór deterministycznym —
        # bez niego obcięta próbka była losowa przy każdym wywołaniu.
        columns = "is_trend_correct, predicted_trend, data_quality_flags, timestamp"
        if self._session_close_settlement_enabled:
            columns += f", {_SESSION_CLOSE_CORRECT_COLUMN}"
        query = (
            self._client.table(self._table)
            .select(self._countable_columns(columns))
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .order("timestamp", desc=True)
        )
        rows = [r for r in _countable(_paginate(query)) if r.get("is_trend_correct") is not None]
        calls = [(str(r.get("predicted_trend") or ""), bool(r["is_trend_correct"])) for r in rows]
        directional = directional_hit_rate(calls)
        n_eff = effective_n(
            group_by_cycle(
                (_utc_day_key(r.get("timestamp")), bool(r["is_trend_correct"])) for r in rows
            )
        )

        # WIA-61: za flagą druga trafność — na oficjalnym zamknięciu sesji.
        session_close = _session_close_stats(rows) if self._session_close_settlement_enabled else {}
        if not calls:
            return {
                "mean_accuracy": None,
                "sample_count": 0,
                "correct_count": 0,
                "days_window": days,
                "directional_count": 0,
                "directional_correct_count": 0,
                "n_eff": 0.0,
                **session_close,
            }
        correct_count = sum(1 for _, correct in calls if correct)
        return {
            "mean_accuracy": correct_count / len(calls),
            "sample_count": len(calls),
            "correct_count": correct_count,
            "days_window": days,
            "directional_count": directional.directional,
            "directional_correct_count": directional.hits,
            "n_eff": n_eff,
            **session_close,
        }

    def get_cached_fundamentals(self, symbol: str) -> Fundamentals | None:
        """Zwraca fundamentale z cache jeśli `fetched_at` mieści się w
        FUNDAMENTALS_CACHE_TTL_HOURS. Filtr TTL po stronie repo —
        adapter nie zna polityki świeżości."""
        cutoff = datetime.now(UTC) - timedelta(hours=FUNDAMENTALS_CACHE_TTL_HOURS)
        response = (
            self._client.table("fundamentals_cache")
            .select("*")
            .eq("symbol", symbol)
            .gte("fetched_at", cutoff.isoformat())
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        row = rows[0]
        return Fundamentals(
            trailing_pe=row.get("trailing_pe"),
            forward_pe=row.get("forward_pe"),
            peg_ratio=row.get("peg_ratio"),
            eps_growth_yoy=row.get("eps_growth_yoy"),
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
            # DAT-21: `select("*")` — przed migracją 056 klucza po prostu nie ma.
            market_cap=_optional_float(row.get("market_cap")),
        )

    def get_fundamentals_fetched_at(self, symbols: Sequence[str]) -> dict[str, datetime]:
        """DAT-19: wiek wpisów cache bez filtra TTL (rotacja refreshu)."""
        if not symbols:
            return {}
        response = (
            self._client.table("fundamentals_cache")
            .select("symbol, fetched_at")
            .in_("symbol", list(symbols))
            .execute()
        )
        fetched: dict[str, datetime] = {}
        for row in _rows(response):
            stamp = row.get("fetched_at")
            if isinstance(stamp, str):
                fetched[str(row.get("symbol"))] = datetime.fromisoformat(stamp)
        return fetched

    def save_fundamentals(self, symbol: str, fundamentals: Fundamentals) -> None:
        """Upsert jednego wiersza per symbol (nadpisuje poprzedni snapshot)."""
        payload: dict[str, Any] = {
            "symbol": symbol,
            "trailing_pe": fundamentals.trailing_pe,
            "forward_pe": fundamentals.forward_pe,
            "peg_ratio": fundamentals.peg_ratio,
            "eps_growth_yoy": fundamentals.eps_growth_yoy,
            "fetched_at": fundamentals.fetched_at.isoformat(),
        }
        if self._fundamentals_market_cap_enabled:
            payload["market_cap"] = fundamentals.market_cap
        self._client.table("fundamentals_cache").upsert(payload).execute()

    def get_cached_earnings(self, symbol: str) -> CachedEarningsDate | None:
        # FIN-11: TTL po stronie repo, jak fundamenty — adapter nie zna świeżości.
        cutoff = datetime.now(UTC) - timedelta(hours=EARNINGS_CACHE_TTL_HOURS)
        response = (
            self._client.table(EARNINGS_CACHE_TABLE)
            .select("symbol, report_date, fetched_at")
            .eq("symbol", symbol)
            .gte("fetched_at", cutoff.isoformat())
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        row = rows[0]
        return CachedEarningsDate(
            symbol=str(row["symbol"]),
            report_date=date.fromisoformat(row["report_date"]),
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
        )

    def save_earnings(self, entry: CachedEarningsDate) -> None:
        # Jeden wiersz per symbol (PK `symbol`) — upsert nadpisuje poprzednią datę.
        self._client.table(EARNINGS_CACHE_TABLE).upsert(
            {
                "symbol": entry.symbol,
                "report_date": entry.report_date.isoformat(),
                "fetched_at": entry.fetched_at.isoformat(),
            }
        ).execute()

    # -----------------------------------------------------------------------
    # Writes
    # -----------------------------------------------------------------------

    def save_price_snapshot(self, symbol: str, price: Money) -> None:
        # Upsert na (symbol, timestamp_hour) (#7) — re-run GHA / nakładający się
        # workflow_dispatch / retry LangGraph aktualizuje istniejący wiersz w tej
        # samej godzinie zamiast duplikować (duplikaty zaburzały trafność).
        record = _serialize_record({"symbol": symbol, "price": _storage_price(price.amount)})
        (
            self._client.table(self._snapshot_table)
            .upsert(record, on_conflict="symbol,timestamp_hour")
            .execute()
        )

    def save_feature_snapshot(
        self,
        symbol: str,
        features: Mapping[str, Any],
        *,
        gated: bool,
        feature_set_version: int,
    ) -> None:
        # ML-14 (migracja 031): upsert na (symbol, as_of_hour) jak snapshot ceny —
        # re-run w tej samej godzinie aktualizuje wektor zamiast go duplikować.
        # `as_of` stawia baza (DEFAULT NOW()), etykieta dochodzi później.
        record: dict[str, Any] = {
            "symbol": symbol,
            "features": dict(features),
            "gated": gated,
            "feature_set_version": feature_set_version,
        }
        (
            self._client.table(DEFAULT_FEATURE_SNAPSHOT_TABLE)
            .upsert(record, on_conflict="symbol,as_of_hour")
            .execute()
        )

    @_translates_api_errors
    def save_shadow_predictions(
        self, prediction_id: str, symbol: str, forecasts: Sequence[ShadowForecast]
    ) -> None:
        # ML-33 (migracja 075): jeden wiersz na model; upsert po (prediction_id,
        # model_name), bo re-run w tej samej godzinie nadpisuje predykcję upsertem.
        if not forecasts:
            return
        records = [
            {
                "prediction_id": prediction_id,
                "symbol": symbol,
                "model_name": forecast.model_name,
                "model_version": forecast.model_version,
                "predicted_return": forecast.predicted_return,
                "proba": forecast.proba,
            }
            for forecast in forecasts
        ]
        (
            self._client.table(DEFAULT_SHADOW_PREDICTION_TABLE)
            .upsert(records, on_conflict="prediction_id,model_name")
            .execute()
        )

    @_translates_api_errors
    def save_shadow_trend_prediction(
        self, prediction_id: str, symbol: str, forecast: ShadowTrendForecast
    ) -> None:
        # EXP-26 (migracja 083): wiersz wariantu LLM w tej samej tabeli co noga ML.
        # Bez kluczy `predicted_return` / `proba` — upsert ich nie dotyka, zostają NULL.
        record = {
            "prediction_id": prediction_id,
            "symbol": symbol,
            "model_name": forecast.model_name,
            "model_version": forecast.model_version,
            "predicted_trend": forecast.predicted_trend.value,
            "confidence_score": forecast.confidence,
        }
        (
            self._client.table(DEFAULT_SHADOW_PREDICTION_TABLE)
            .upsert(record, on_conflict="prediction_id,model_name")
            .execute()
        )

    @_translates_api_errors
    def get_settled_shadow_predictions(self, since: datetime) -> list[SettledShadowForecast]:
        # ML-33: `!inner` + filtr na osadzonej predykcji — tylko rozliczone. `id`
        # jako drugi klucz: granica strony deterministyczna przy równych `created_at`.
        query = (
            self._client.table(DEFAULT_SHADOW_PREDICTION_TABLE)
            .select(
                "prediction_id, model_name, model_version, predicted_return, proba, "
                "prediction_logs!inner(price_at_prediction, actual_price_after_12h)"
            )
            .gte("created_at", since.isoformat())
            .not_.is_("prediction_logs.actual_price_after_12h", "null")
        )
        rows = _paginate(query.order("created_at", desc=False).order("id", desc=False))
        return [settled for row in rows if (settled := _settled_shadow(row)) is not None]

    def get_latest_regime_state(self) -> RegimeState | None:
        # ALR-03 (migracja 033): `id` rozstrzyga remis dwóch przebiegów o tym
        # samym `cycle_at`. Kandydata do wyjścia z RISK_OFF tabela nie trzyma
        # osobno — przy trwającej serii to surowy odczyt tego wiersza.
        response = (
            self._client.table(DEFAULT_REGIME_HISTORY_TABLE)
            .select("raw_regime, regime, streak")
            .order("cycle_at", desc=True)
            .order("id", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        row = rows[0]
        streak = int(row["streak"])
        return RegimeState(
            regime=MarketRegime(row["regime"]),
            pending=MarketRegime(row["raw_regime"]) if streak > 0 else None,
            streak=streak,
        )

    def save_regime_state(self, cycle_at: datetime, raw: MarketRegime, state: RegimeState) -> None:
        self._client.table(DEFAULT_REGIME_HISTORY_TABLE).insert(
            {
                "cycle_at": cycle_at.isoformat(),
                "raw_regime": raw.value,
                "regime": state.regime.value,
                "streak": state.streak,
            }
        ).execute()

    def save_prediction(self, prediction: dict[str, Any]) -> str:
        # Upsert na (symbol, timestamp_hour) (#7) — idempotentny zapis predykcji.
        # Domyślny merge upsertu zwraca wiersz, więc nadal mamy z czego odczytać
        # id (graf używa go jako prediction_id).
        record = _serialize_record(prediction, _PREDICTION_PRICE_COLUMNS)
        response = (
            self._client.table(self._table)
            .upsert(record, on_conflict="symbol,timestamp_hour")
            .execute()
        )
        rows = _rows(response)
        if not rows:
            raise RuntimeError(f"Supabase upsert returned no data for {record.get('symbol')}")
        return str(rows[0]["id"])

    def save_quota_alert(self, alert: QuotaAlert) -> None:
        row = {
            "source": alert.source,
            "severity": alert.severity.value,
            "message": alert.message,
            "action": alert.action,
            "occurred_at": alert.occurred_at.isoformat(),
        }
        self._client.table(self._quota_alerts_table).insert(row).execute()

    def get_recent_quota_alerts(self, hours: int) -> list[QuotaAlert]:
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        response = (
            self._client.table(self._quota_alerts_table)
            .select("*")
            .gte("occurred_at", cutoff.isoformat())
            .order("occurred_at", desc=True)
            .execute()
        )
        rows = _rows(response)
        out: list[QuotaAlert] = []
        for row in rows:
            try:
                severity = QuotaSeverity(row["severity"])
            except (KeyError, ValueError):
                continue
            occurred = row.get("occurred_at")
            if not isinstance(occurred, str):
                continue
            out.append(
                QuotaAlert(
                    source=str(row.get("source", "?")),
                    severity=severity,
                    message=str(row.get("message", "")),
                    action=str(row.get("action", "")),
                    occurred_at=datetime.fromisoformat(occurred),
                )
            )
        return out

    def find_similar_predictions(
        self,
        embedding: list[float],
        limit: int = 3,
        regime: str | None = None,
        *,
        symbol: str | None = None,
        before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Similarity search nad `prediction_logs.embedding` (pgvector) przez
        RPC `match_news_embeddings` (migracja 011; filtr reżimu z migracji 017),
        a z `rag_exact_search_enabled` przez `match_news_embeddings_exact` (063, ML-23).

        Graceful: gdy RPC nie istnieje (migracja niezaaplikowana) lub błąd
        sieci/Supabase — zwraca `[]`, żeby RAG był czysto opcjonalnym
        wzbogaceniem, a nie twardą zależnością predykcji.

        `regime`: gdy podany, dokłada parametr `filter_regime` do RPC (migracja
        017 filtruje analogi po reżimie; NULL/UNKNOWN w bazie przepuszcza).

        ML-41: z `rag_point_in_time_enabled` RPC `match_news_embeddings_pit` (076) —
        `before` jako `before_ts` (tylko analogi rozliczone wcześniej), `symbol` jako
        `exclude_symbol` z oknem `SAME_SYMBOL_ANALOG_EXCLUSION_HOURS`."""
        params: dict[str, Any] = {
            "query_embedding": embedding,
            "match_count": limit,
        }
        if regime is not None:
            params["filter_regime"] = canonical_regime_key(regime)
        function, migration = self._rag_function, self._rag_migration
        if self._rag_point_in_time_enabled:
            function, migration = "match_news_embeddings_pit", "076"
            if before is not None:
                params["before_ts"] = before.isoformat()
            if symbol is not None:
                params["exclude_symbol"] = symbol
                params["exclude_window_hours"] = SAME_SYMBOL_ANALOG_EXCLUSION_HOURS
        try:
            response = self._client.rpc(function, params).execute()
        except Exception as exc:
            _log_degraded_read(f"{function} RPC", "RAG retrieval skipped", migration, exc)
            return []
        return _rows(response)

    def get_resolved_for_calibration(self, days: int) -> list[dict[str, Any]]:
        # NIEZ-15: paginacja + `id` jako drugi klucz — przy równych timestampach
        # granica strony musi być deterministyczna (bez gubienia/dublowania wierszy).
        # WIA-39/WIA-38: trend oddziela SIDEWAYS od kalibracji kierunkowej, a
        # `timestamp` daje cykle pod n_eff i bootstrap (obie kolumny od 001).
        # ML-54: `symbol` (od 001) wyznacza klasę aktywa mapy kalibracji.
        cutoff = datetime.now(UTC) - timedelta(days=days)
        query = (
            self._client.table(self._table)
            .select(
                self._countable_columns(
                    "id, symbol, confidence_score, is_trend_correct, predicted_trend, "
                    "timestamp, reasoning_text, data_quality_flags"
                )
            )
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .not_.is_("confidence_score", "null")
            .order("timestamp", desc=True)
            .order("id", desc=True)
        )
        return _countable(_paginate(query))

    def save_calibration_scores(self, scores: Mapping[str, float], *, score_version: int) -> int:
        # WIA-65: jedno wywołanie RPC (migracja 077) zamiast UPDATE wiersz po wierszu.
        if not scores:
            return 0
        payload = [{"id": prediction_id, "score": score} for prediction_id, score in scores.items()]
        data = (
            self._client.rpc(
                "save_calibration_scores",
                {"p_scores": payload, "p_score_version": score_version},
            )
            .execute()
            .data
        )
        if isinstance(data, bool) or not isinstance(data, int):
            raise MalformedPayload(
                f"save_calibration_scores zwróciło {type(data).__name__}, nie liczbę"
            )
        return data

    def get_persona_accuracy(
        self, asset_type: AssetType, window_days: int = 90
    ) -> dict[str, float]:
        # Liczenie trafności per persona (rekomendacja vs realny ruch ceny) robi
        # RPC `persona_accuracy_weights` w SQL (migracja 015) — Python tylko mapuje
        # wynik. `asset_type` zarezerwowany na przyszłe ważenie per klasa aktywa
        # (dziś council_votes nie nosi typu → waga globalna). Graceful: brak RPC
        # / błąd → {} (ważenie wyłączone, każda persona waży 1.0).
        _ = asset_type
        if self._persona_vote_outcomes_enabled:
            outcomes = self._graceful_vote_outcomes(window_days, "persona weighting skipped")
            return persona_accuracy_weights(outcomes, self._crypto_symbols)
        try:
            response = self._client.rpc(
                "persona_accuracy_weights", {"window_days": window_days}
            ).execute()
        except Exception as exc:
            _log_degraded_read(
                "persona_accuracy_weights RPC", "persona weighting skipped", "015", exc
            )
            return {}
        out: dict[str, float] = {}
        for row in _rows(response):
            name = row.get("investor_name")
            weight = row.get("weight")
            if name is None or weight is None:
                continue
            out[str(name)] = float(weight)
        return out

    def get_persona_track_record(self, window_days: int = 90) -> dict[str, tuple[float, int]]:
        # RPC `persona_accuracy_stats` (migracja 018) — ten sam JOIN co 015, ale
        # zwraca surowy hit-rate ORAZ COUNT(*) głosów. RPC z 015 niesie tylko
        # przeskalowaną wagę, z której nie da się odtworzyć wielkości próbki.
        # Graceful: brak RPC / błąd → {} (leaderboard sam się chowa).
        if self._persona_vote_outcomes_enabled:
            outcomes = self._graceful_vote_outcomes(window_days, "persona leaderboard skipped")
            return persona_hit_stats(outcomes, self._crypto_symbols)
        try:
            response = self._client.rpc(
                "persona_accuracy_stats", {"window_days": window_days}
            ).execute()
        except Exception as exc:
            _log_degraded_read(
                "persona_accuracy_stats RPC", "persona leaderboard skipped", "018", exc
            )
            return {}
        out: dict[str, tuple[float, int]] = {}
        for row in _rows(response):
            name = row.get("investor_name")
            hit_rate = row.get("hit_rate")
            votes = row.get("vote_count")
            if name is None or hit_rate is None or votes is None:
                continue
            out[str(name)] = (float(hit_rate), int(votes))
        return out

    def get_persona_scoring(self, window_days: int = 90) -> dict[str, PersonaScoring]:
        # WIA-44: RPC `persona_scoring_stats` (migracja 065) liczy hit-rate, Brier
        # i log loss pewności tą samą regułą co domena; pasmo HOLD krypto idzie
        # parametrem (jak `crypto_symbols` adaptera). Graceful: brak RPC → {}.
        try:
            response = self._client.rpc(
                "persona_scoring_stats",
                {"window_days": window_days, "crypto_symbols": sorted(self._crypto_symbols)},
            ).execute()
        except APIError as exc:
            _log_degraded_read(
                "persona_scoring_stats RPC", "persona Brier ranking skipped", "065", exc
            )
            return {}
        out: dict[str, PersonaScoring] = {}
        for row in _rows(response):
            name = row.get("investor_name")
            hit_rate = row.get("hit_rate")
            votes = row.get("vote_count")
            brier = row.get("brier")
            loss = row.get("log_loss")
            if name is None or hit_rate is None or votes is None or brier is None or loss is None:
                continue
            out[str(name)] = PersonaScoring(
                hit_rate=float(hit_rate), votes=int(votes), brier=float(brier), log_loss=float(loss)
            )
        return out

    def get_deferred_diagnoses(self, limit: int) -> list[DeferredDiagnosis]:
        # WIA-53: kandydatów jest kilkaset (dług z wstrzymanego Fast Loopa), więc
        # czytamy całość stronami i odsiewamy atrapy po awarii LLM w Pythonie
        # (`_without_llm_fallback`), zanim przytniemy do `limit`. Graceful 020:
        # bez kolumny kwitu czytamy bez niej, a prompt idzie bez bloku kwitu.
        try:
            rows = self._deferred_rows(with_receipt=True)
        except APIError as exc:
            if not _is_missing_column(exc, "decision_receipts"):
                raise
            rows = self._deferred_rows(with_receipt=False)
        deferred = [
            item for row in _without_llm_fallback(rows) if (item := _deferred_diagnosis(row))
        ]
        return deferred[: max(0, limit)]

    def _deferred_rows(self, *, with_receipt: bool) -> list[dict[str, Any]]:
        columns = _DEFERRED_COLUMNS + (", decision_receipts" if with_receipt else "")
        query = (
            self._client.table(self._table)
            .select(columns)
            .in_("correction_insights", list(UNDIAGNOSED_MISTAKE_PLACEHOLDERS))
            .eq("is_trend_correct", False)
            .order("timestamp", desc=True)
        )
        return _paginate(query)

    def save_backfilled_diagnosis(
        self,
        prediction_id: str,
        insight: str,
        *,
        replaces: str,
        correction_category: MistakeCategory | None = None,
    ) -> None:
        # WIA-53: warunek na placeholder — wiersz zdiagnozowany w międzyczasie
        # (ręczny run, drugi przebieg) nie zostaje nadpisany.
        payload = {"correction_insights": insight, **self._category_payload(correction_category)}
        (
            self._client.table(self._table)
            .update(payload)
            .eq("id", prediction_id)
            .eq("correction_insights", replaces)
            .execute()
        )

    def get_diagnoses_to_judge(self, days: int) -> list[JudgeableDiagnosis]:
        # EXP-10: kolumna oceny z migracji 078; flaga sędziego czeka na nią (schema gate).
        cutoff = datetime.now(UTC) - timedelta(days=days)
        query = (
            self._client.table(self._table)
            .select(_DEFERRED_COLUMNS)
            .eq("is_trend_correct", False)
            .is_("diagnosis_judge_score", "null")
            .not_.is_("correction_insights", "null")
            .gte("timestamp", cutoff.isoformat())
            .order("timestamp", desc=True)
            .order("id", desc=True)
        )
        return [
            item
            for row in _without_llm_fallback(_paginate(query))
            if (item := _judgeable_diagnosis(row))
        ]

    def save_diagnosis_judgement(
        self, prediction_id: str, *, score: int, rejected_insight: str | None = None
    ) -> None:
        # EXP-10: odrzucenie przenosi tekst do `rejected_diagnosis` tylko, gdy diagnoza
        # wciąż stoi w `correction_insights` (spłata diagnoz mogła ją w międzyczasie zmienić).
        if rejected_insight is None:
            (
                self._client.table(self._table)
                .update({"diagnosis_judge_score": score})
                .eq("id", prediction_id)
                .execute()
            )
            return
        payload: dict[str, Any] = {
            "diagnosis_judge_score": score,
            "correction_insights": DIAGNOSIS_REJECTED_INSIGHT,
            "rejected_diagnosis": rejected_insight,
        }
        (
            self._client.table(self._table)
            .update(payload)
            .eq("id", prediction_id)
            .eq("correction_insights", rejected_insight)
            .execute()
        )

    def get_persona_vote_outcomes(self, window_days: int = 90) -> list[PersonaVoteOutcome]:
        # WIA-51: surowe wiersze, więc czytamy stronami — PostgREST obcina wynik
        # RPC do `max_rows`, a 90 dni × watchlista × 7 person to dziesiątki tysięcy.
        query = (
            self._client.rpc("persona_vote_outcomes", {"window_days": window_days})
            .order("predicted_at")
            .order("prediction_id")
            .order("investor_name")
        )
        return [
            outcome
            for row in _paginate(query)
            if (outcome := _persona_vote_outcome(row)) is not None
        ]

    def _graceful_vote_outcomes(self, window_days: int, skipped: str) -> list[PersonaVoteOutcome]:
        """Głosy RPC 045 albo pusta lista z WARNING-iem (brak migracji, błąd sieci)."""
        try:
            return self.get_persona_vote_outcomes(window_days)
        except Exception as exc:
            _log_degraded_read("persona_vote_outcomes RPC", skipped, "045", exc)
            return []

    def get_unrevealed_commitments(self, days: int | None = None) -> list[dict[str, Any]]:
        # Graceful: brak migracji 024 → [] (sweep nie ma czego ujawnić). WIA-70:
        # brak samej kolumny z 043 → stare wiersze ujawniamy z kolumn, jak dotąd.
        try:
            try:
                return self._unrevealed_commitments(days, with_payload=True)
            except APIError as exc:
                if not _is_missing_column(exc, _COMMITMENT_PAYLOAD_COLUMN):
                    raise
            return self._unrevealed_commitments(days, with_payload=False)
        except Exception as exc:
            _log_degraded_read("Attestation columns", "reveal sweep skipped", "024", exc)
            return []

    def _unrevealed_commitments(
        self, days: int | None, *, with_payload: bool
    ) -> list[dict[str, Any]]:
        columns = (
            "id, symbol, predicted_trend, predicted_target_price, "
            "price_at_prediction, actual_price_after_12h, "
            "is_trend_correct, commitment_hash, commitment_salt"
        )
        if with_payload:
            columns += f", {_COMMITMENT_PAYLOAD_COLUMN}"
        query = self._client.table(self._table).select(columns)
        # WIA-03: sweep nie podaje okna — commitment starszy niż okno track recordu
        # też musi zostać ujawniony, inaczej wygląda jak ukryte pudło.
        if days is not None:
            query = query.gte("timestamp", (datetime.now(UTC) - timedelta(days=days)).isoformat())
        # NIEZ-15: bez paginacji sweep widział max ~1000 commitmentów.
        query = (
            query.not_.is_("commitment_hash", "null")
            .not_.is_("actual_price_after_12h", "null")
            .is_("revealed_at", "null")
            .order("timestamp", desc=False)
            .order("id", desc=False)
        )
        return _paginate(query)

    def get_current_hour_commitment_salt(self, symbol: str) -> str | None:
        # Godzina jak w kolumnie generowanej z 013: date_trunc('hour', timestamp, 'UTC').
        hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        try:
            response = (
                self._client.table(self._table)
                .select("commitment_salt")
                .eq("symbol", symbol)
                .eq("timestamp_hour", hour.isoformat())
                .not_.is_("commitment_salt", "null")
                .limit(1)
                .execute()
            )
        except Exception:
            logger.warning("Commitment salt of %s unavailable — świeża sól", symbol, exc_info=True)
            return None
        rows = _rows(response)
        return str(rows[0]["commitment_salt"]) if rows else None

    def get_current_hour_commitment(self, symbol: str) -> StoredCommitment | None:
        # Ta sama godzina co w `get_current_hour_commitment_salt` (klucz upsertu z 013).
        hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        try:
            response = (
                self._client.table(self._table)
                .select(f"commitment_hash, commitment_salt, {_COMMITMENT_PAYLOAD_COLUMN}")
                .eq("symbol", symbol)
                .eq("timestamp_hour", hour.isoformat())
                .not_.is_("commitment_hash", "null")
                .limit(1)
                .execute()
            )
        except Exception:
            logger.warning(
                "Stored commitment of %s unavailable — bez revealu nadpisania",
                symbol,
                exc_info=True,
            )
            return None
        rows = _rows(response)
        if not rows:
            return None
        row = rows[0]
        commitment, salt = row.get("commitment_hash"), row.get("commitment_salt")
        if not commitment or not salt:
            return None
        payload = row.get(_COMMITMENT_PAYLOAD_COLUMN)
        return StoredCommitment(
            commitment=str(commitment),
            salt=str(salt),
            payload=None if payload is None else str(payload),
        )

    def attestation_columns_available(self) -> bool:
        try:
            self._client.table(self._table).select(
                f"commitment_hash, commitment_salt, {_COMMITMENT_PAYLOAD_COLUMN}"
            ).limit(1).execute()
        except Exception:
            logger.warning(
                "Kolumny atestacji niedostępne — cykl bez atestacji "
                "(zaaplikuj migracje 024 i 043).",
                exc_info=True,
            )
            return False
        return True

    def mark_commitment_revealed(self, prediction_id: str) -> None:
        self._client.table(self._table).update({"revealed_at": datetime.now(UTC).isoformat()}).eq(
            "id", prediction_id
        ).execute()

    def _latest_timestamp(self, table: str) -> datetime | None:
        # Bez łagodzenia błędów: deadman musi odróżnić „job milczy” od „nie da się
        # sprawdzić” — wyjątek kończy jego przebieg exit 1 i alertem workflowu.
        response = (
            self._client.table(table)
            .select("timestamp")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        stamp = rows[0].get("timestamp") if rows else None
        return datetime.fromisoformat(stamp) if isinstance(stamp, str) else None

    def get_latest_snapshot_at(self) -> datetime | None:
        return self._latest_timestamp("price_snapshots")

    def get_last_scorecard_at(self) -> datetime | None:
        return self._latest_timestamp("model_scorecards")

    def probe_database(self) -> None:
        # Bez łagodzenia błędów: wyjątek to dla deadmana „baza nie odpowiada”.
        # Wstrzymany projekt potrafi odpowiedzieć stroną HTML zamiast listy wierszy.
        response = self._client.table("price_snapshots").select("timestamp").limit(1).execute()
        if not isinstance(response.data, list):
            raise RuntimeError("Supabase: odpowiedź bez listy wierszy price_snapshots")

    @_translates_api_errors
    def schema_version(self) -> int | None:
        """NIEZ-30: RPC `schema_version` (migracja 071) — najwyższa zaaplikowana migracja.
        NULL z bazy (rejestr Supabase CLI nieobecny albo pusty) to None, nie 0. Brak
        funkcji (PGRST202) wychodzi jako `NotFound`, awaria bazy jako `ProviderUnavailable`."""
        data = self._client.rpc("schema_version").execute().data
        if data is None:
            return None
        if isinstance(data, bool) or not isinstance(data, int):
            raise MalformedPayload(f"schema_version zwróciło {type(data).__name__}, nie liczbę")
        return data

    def get_db_size_report(self) -> list[RelationSize]:
        # NIEZ-37: RPC z migracji 077; PGRST202 przed migracją tłumaczy klasa (NotFound).
        data = self._client.rpc("db_size_report").execute().data
        if not isinstance(data, list):
            raise MalformedPayload(f"db_size_report zwróciło {type(data).__name__}, nie listę")
        sizes: list[RelationSize] = []
        for row in data:
            name = row.get("relation_name") if isinstance(row, dict) else None
            total = row.get("total_bytes") if isinstance(row, dict) else None
            if isinstance(name, str) and isinstance(total, int) and not isinstance(total, bool):
                sizes.append(RelationSize(name, total))
        return sizes

    def missing_prediction_columns(self, columns: frozenset[str]) -> frozenset[str]:
        """FIN-05: select `limit 0` po `columns`. PostgREST zgłasza jedną brakującą
        kolumnę na raz (42703), więc po każdej usuwamy ją z selecta i pytamy dalej —
        najwyżej jedno zapytanie na brakującą kolumnę. Inny błąd albo 42703 bez
        rozpoznanej kolumny z `columns` leci dalej: to nie dowód braku migracji."""
        remaining = set(columns)
        missing: set[str] = set()
        while remaining:
            try:
                (
                    self._client.table(self._table)
                    .select(",".join(sorted(remaining)))
                    .limit(0)
                    .execute()
                )
            except APIError as exc:
                column = _undefined_column(exc)
                if exc.code != _UNDEFINED_COLUMN_CODE or column not in remaining:
                    raise
                remaining.discard(column)
                missing.add(column)
                continue
            break
        return frozenset(missing)

    def read_rows(self, table: str, columns: Sequence[str]) -> list[dict[str, Any]]:
        """NIEZ-32 (`EvidenceSourcePort`): wszystkie wiersze `table`, tylko `columns`,
        stronami po `id` (paginacja #8). Kolumna bez swojej migracji (42703) wypada
        z selecta z WARNING-iem — kopia reszty kolumn i tak powstaje."""
        remaining = list(columns)
        while remaining:
            query = self._client.table(table).select(", ".join(remaining)).order("id")
            try:
                return _paginate(query)
            except APIError as exc:
                missing = _missing_column(exc, remaining)
                if missing is None:
                    raise
                logger.warning(
                    "Evidence export: %s.%s missing (migration not applied) — skipped",
                    table,
                    missing,
                )
                remaining.remove(missing)
        return []

    def get_last_price_snapshot(self, symbol: str) -> tuple[Money, datetime] | None:
        response = (
            self._client.table("price_snapshots")
            .select("price, timestamp")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        row = rows[0]
        price, stamp = row.get("price"), row.get("timestamp")
        if price is None or not isinstance(stamp, str):
            return None
        return Money(Decimal(str(price))), datetime.fromisoformat(stamp)

    def save_shock_alert(
        self, symbol: str, alert_date: date, delta: float, direction: str, *, level: int = 1
    ) -> None:
        # At-least-once (ALR-14, ALR-23): watch woła ten zapis PO wysłaniu pusha, więc
        # UNIQUE(symbol, alert_date) z 021 — od 081 UNIQUE(symbol, alert_date, level) —
        # odrzuca tylko drugi WIERSZ szczebla, nie drugi push. Dwa równoległe przebiegi
        # mogą wysłać ten sam alert dwa razy; kolizja klucza wraca jako PortError, który
        # watch loguje. Komentarz 021 („przed wysłaniem”) opisuje odwrotną kolejność.
        row: dict[str, Any] = {
            "symbol": symbol,
            "alert_date": alert_date.isoformat(),
            "delta": delta,
            "direction": direction,
        }
        # ALR-16: szczebel 1 to DEFAULT kolumny z 081, więc zapis bez drabiny wygląda
        # jak przed migracją; klucz `level` niesie tylko eskalacja (za flagą po 081).
        if level != 1:
            row["level"] = level
        self._client.table("shock_alerts").insert(row).execute()

    def get_shock_alert_levels(self, alert_date: date) -> dict[str, tuple[SentShockLevel, ...]]:
        # SUROWO (ARCH-33): bez migracji 081 leci `PortError`, a watch na ten przebieg
        # wraca do „jeden alert dziennie”.
        response = (
            self._client.table("shock_alerts")
            .select("symbol, level, direction")
            .eq("alert_date", alert_date.isoformat())
            .execute()
        )
        sent: dict[str, list[SentShockLevel]] = {}
        for row in _rows(response):
            symbol, level, direction = row.get("symbol"), row.get("level"), row.get("direction")
            if symbol is None or not isinstance(level, int) or direction not in _SHOCK_DIRECTIONS:
                # Pominięty szczebel mógłby wyjść drugi raz — lepiej cały przebieg bez drabiny.
                raise MalformedPayload(f"shock_alerts: nieczytelny szczebel {row!r}")
            sent.setdefault(str(symbol), []).append(
                SentShockLevel(level=level, direction=ShockDirection(direction))
            )
        return {
            symbol: tuple(sorted(levels, key=lambda rung: rung.level))
            for symbol, levels in sent.items()
        }

    def get_sent_shock_alerts(self, since: date) -> set[tuple[str, date]]:
        # Graceful: brak migracji 021 → pusty zbiór (debounce zostaje tylko w pamięci
        # przebiegu; UNIQUE odrzuci najwyżej drugi wiersz, push i tak już poszedł).
        try:
            response = (
                self._client.table("shock_alerts")
                .select("symbol, alert_date")
                .gte("alert_date", since.isoformat())
                .execute()
            )
        except Exception as exc:
            _log_degraded_read("shock_alerts", "debounce kept in memory only", "021", exc)
            return set()
        sent: set[tuple[str, date]] = set()
        for row in _rows(response):
            symbol, raw_date = row.get("symbol"), row.get("alert_date")
            if symbol is None or not isinstance(raw_date, str):
                continue
            sent.add((str(symbol), date.fromisoformat(raw_date)))
        return sent

    def get_shock_alert_times(self, since: datetime) -> dict[str, datetime]:
        # Graceful jak `get_sent_shock_alerts`: brak migracji 021 → {} (watch nie
        # tłumi wtedy krypto oknem 24 h; debounce dnia działa dalej).
        try:
            response = (
                self._client.table("shock_alerts")
                .select("symbol, created_at")
                .gte("created_at", since.isoformat())
                .execute()
            )
        except Exception as exc:
            _log_degraded_read("shock_alerts", "no 24 h crypto window this run", "021", exc)
            return {}
        latest: dict[str, datetime] = {}
        for row in _rows(response):
            symbol, raw_stamp = row.get("symbol"), row.get("created_at")
            if symbol is None or not isinstance(raw_stamp, str):
                continue
            stamp = datetime.fromisoformat(raw_stamp)
            key = str(symbol)
            if key not in latest or stamp > latest[key]:
                latest[key] = stamp
        return latest

    def get_shock_alerts_since(self, since: datetime) -> list[StoredShockAlert]:
        # SUROWO (ARCH-33): błąd bazy leci do ReportAssemblera, który chowa sekcję.
        response = (
            self._client.table("shock_alerts")
            .select("symbol, delta, direction, created_at")
            .gte("created_at", since.isoformat())
            .order("created_at", desc=False)
            .execute()
        )
        alerts: list[StoredShockAlert] = []
        for row in _rows(response):
            symbol, delta = row.get("symbol"), row.get("delta")
            direction, raw_stamp = row.get("direction"), row.get("created_at")
            if symbol is None or delta is None or not isinstance(raw_stamp, str):
                continue
            alerts.append(
                StoredShockAlert(
                    symbol=str(symbol),
                    delta=float(delta),
                    direction=str(direction),
                    created_at=datetime.fromisoformat(raw_stamp),
                )
            )
        return alerts

    def get_last_report_at(self, before: datetime) -> datetime | None:
        response = (
            self._client.table("cycle_runs")
            .select("started_at")
            .eq("loop", "fast")
            .eq("report_delivered", True)
            .lt("started_at", before.isoformat())
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        stamp = rows[0].get("started_at") if rows else None
        return datetime.fromisoformat(stamp) if isinstance(stamp, str) else None

    def get_active_price_levels(self) -> list[PriceLevelAlert]:
        # SUROWO, bez try/except — watch czyta za flagą i sam degraduje (ALR-11).
        response = (
            self._client.table(PRICE_LEVEL_ALERTS_TABLE)
            .select("id, symbol, level, direction, note, last_seen_price")
            .eq("active", True)
            .order("id", desc=False)
            .execute()
        )
        return [_price_level_alert(row) for row in _rows(response)]

    def add_price_levels(self, levels: Sequence[PriceLevel]) -> None:
        if not levels:
            return
        rows = [
            {
                "symbol": level.symbol,
                "level": str(level.level),
                "direction": level.direction.value,
                "note": level.note,
            }
            for level in levels
        ]
        # ON CONFLICT DO NOTHING: zapisany poziom (także wyłączony po trafieniu)
        # zostaje, jaki był — lista z config.toml go nie wskrzesza.
        self._client.table(PRICE_LEVEL_ALERTS_TABLE).upsert(
            rows, on_conflict=_PRICE_LEVEL_UNIQUE, ignore_duplicates=True
        ).execute()

    def record_price_level_reading(self, alert_id: str, price: Decimal) -> None:
        (
            self._client.table(PRICE_LEVEL_ALERTS_TABLE)
            .update({"last_seen_price": str(price)})
            .eq("id", alert_id)
            .eq("active", True)
            .execute()
        )

    def mark_price_level_triggered(
        self, alert_id: str, price: Decimal, triggered_at: datetime
    ) -> None:
        (
            self._client.table(PRICE_LEVEL_ALERTS_TABLE)
            .update(
                {
                    "active": False,
                    "triggered_at": triggered_at.isoformat(),
                    "last_seen_price": str(price),
                }
            )
            .eq("id", alert_id)
            .execute()
        )

    def get_previous_verdict(self, symbol: str) -> PreviousVerdict | None:
        # SUROWO, bez try/except — graceful degradation robi caller
        # (`_fetch_report_context`), tak jak reszta odczytów raportowych.
        response = (
            self._client.table(self._table)
            .select("council_verdict, sentiment_score, timestamp")
            .eq("symbol", symbol)
            .not_.is_("council_verdict", "null")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        row = rows[0]
        raw_verdict = row.get("council_verdict")
        stamp = row.get("timestamp")
        if not isinstance(raw_verdict, dict) or not isinstance(stamp, str):
            return None
        return PreviousVerdict(
            verdict=_deserialize_verdict(raw_verdict),
            sentiment_score=float(row.get("sentiment_score") or 0.0),
            timestamp=datetime.fromisoformat(stamp),
        )

    def save_model_scorecard(
        self, symbol: str, result: Mapping[str, Any], *, run_id: str | None = None
    ) -> None:
        # Jeden wiersz na przebieg treningu (migracja 019). Kolumna `symbol` jest
        # kluczowa: ewaluacja leci per symbol, a model wspólny ma wiersz "ALL"
        # (ML-13) — bez niej nie wiadomo, czego dotyczy werdykt bramki.
        payload = {
            "symbol": symbol,
            "status": result.get("status"),
            "candidate_holdout_rmse": result.get("candidate_holdout_rmse"),
            "baseline_rmse": result.get("baseline_rmse"),
            "candidate_holdout_directional_hit_rate": result.get(
                "candidate_holdout_directional_hit_rate"
            ),
            "n_folds": result.get("n_folds"),
            "n_samples": result.get("n_samples"),
        }
        if self._scorecard_baselines_enabled:
            payload["drift_baseline_rmse"] = result.get("drift_baseline_rmse")
            payload["llm_baseline_rmse"] = result.get("llm_baseline_rmse")
        if self._scorecard_run_id_enabled and run_id is not None:
            payload["run_id"] = run_id
        feature_stats = result.get("feature_stats")
        if self._scorecard_feature_stats_enabled and feature_stats is not None:
            payload["feature_stats"] = feature_stats
        weights_sha256 = result.get("weights_sha256")
        if self._scorecard_weights_sha256_enabled and weights_sha256 is not None:
            payload["weights_sha256"] = weights_sha256
        # ML-22: proweniencja tylko z treningu, który się odbył (ma odcisk zbioru) —
        # pominięcie niesie w extras samą wersję zestawu cech, nie proweniencję.
        if self._scorecard_provenance_enabled and result.get("dataset_sha256") is not None:
            payload.update({key: result.get(key) for key in PROVENANCE_RECORD_KEYS})
        self._client.table("model_scorecards").insert(payload).execute()

    def get_recent_model_scorecards(self, days: int = 30) -> list[dict[str, Any]]:
        # Graceful: brak migracji 019 / błąd → [] (sekcja raportu się chowa).
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        try:
            response = (
                self._client.table("model_scorecards")
                .select("*")
                .gte("timestamp", cutoff)
                .order("timestamp", desc=True)
                .execute()
            )
        except Exception as exc:
            _log_degraded_read("model_scorecards", "model health card skipped", "019", exc)
            return []
        return list(_rows(response))

    def _first_round_only(self, query: Any) -> Any:
        """ML-40: odczyt głosów zawęża do rundy 1, gdy tabela ma rundy — z debatą albo
        po migracji 082 bez niej, bo wyłączenie debaty nie usuwa zapisanych rewizji."""
        if self._has_vote_rounds():
            return query.eq("round", FIRST_COUNCIL_ROUND)
        return query

    def _has_vote_rounds(self) -> bool:
        """Flaga debaty przesądza bez zapytania (czeka na 082 w bramce schematu); bez
        niej, przy włączonej sondzie, `schema_version()` raz na instancję. Wersja
        nieznana albo RPC bez odpowiedzi = bez filtra, jak bramka schematu (fail-open);
        przed 082 kolumny nie ma."""
        if self._council_vote_rounds_enabled:
            return True
        if not self._council_vote_rounds_from_schema:
            return False
        if self._vote_rounds_in_schema is None:
            try:
                version = self.schema_version()
            except PortError:
                version = None
            self._vote_rounds_in_schema = (
                version is not None and version >= _COUNCIL_VOTE_ROUNDS_MIGRATION
            )
        return self._vote_rounds_in_schema

    def get_council_vote_history(
        self,
        symbols: list[str],
        *,
        window_days: int = 30,
        investors: list[str] | None = None,
    ) -> list[InvestorHistory]:
        if not symbols:
            return []
        cutoff = datetime.now(UTC) - timedelta(days=window_days)
        query = (
            self._client.table(self._council_votes_table)
            .select("investor_name, symbol, recommendation, timestamp")
            .in_("symbol", symbols)
            .gte("timestamp", cutoff.isoformat())
        )
        query = self._first_round_only(query)
        if investors:
            query = query.in_("investor_name", investors)
        # NIEZ-15: paginacja z `id` jako drugim kluczem (jak w odczytach Slow Loopa).
        rows = _paginate(query.order("timestamp", desc=True).order("id", desc=True))
        # Grupowanie per (inwestor, symbol); kolejność malejąca po czasie
        # zachowana (rekomendacje od najnowszej).
        grouped: dict[tuple[str, str], list[Recommendation]] = {}
        for row in rows:
            name = str(row.get("investor_name") or "")
            sym = str(row.get("symbol") or "")
            rec = parse_recommendation(row.get("recommendation"))
            if not name or not sym or rec is None:
                continue
            grouped.setdefault((name, sym), []).append(rec)
        return [
            InvestorHistory(
                investor_name=name,
                symbol=sym,
                recommendations=tuple(recs),
                window_days=window_days,
            )
            for (name, sym), recs in grouped.items()
        ]

    def get_council_vote_samples(self, window_days: int = 30) -> list[CouncilVoteSample]:
        # WIA-45: tylko trzy kolumny, paginacja z `id` jako drugim kluczem (jak historia
        # głosów). Graceful: baza niedostępna → [] z WARNING-iem, mail bez redundancji.
        cutoff = datetime.now(UTC) - timedelta(days=window_days)
        query = (
            self._first_round_only(
                self._client.table(self._council_votes_table)
                .select("prediction_id, investor_name, recommendation")
                .gte("timestamp", cutoff.isoformat())
            )
            .order("timestamp", desc=True)
            .order("id", desc=True)
        )
        try:
            rows = _paginate(query)
        except (APIError, httpx.HTTPError) as exc:
            _log_degraded_read("council_votes", "council redundancy skipped", "007", exc)
            return []
        samples: list[CouncilVoteSample] = []
        for row in rows:
            prediction_id = row.get("prediction_id")
            name = str(row.get("investor_name") or "")
            rec = parse_recommendation(row.get("recommendation"))
            if prediction_id is None or not name or rec is None:
                continue
            samples.append(CouncilVoteSample(str(prediction_id), name, rec))
        return samples

    def get_council_votes_for_prediction(self, prediction_id: str) -> list[InvestorOpinion]:
        response = self._first_round_only(
            self._client.table(self._council_votes_table)
            .select("investor_name, recommendation, confidence, reasoning, key_factors")
            .eq("prediction_id", prediction_id)
        ).execute()
        out: list[InvestorOpinion] = []
        for row in _rows(response):
            rec = parse_recommendation(row.get("recommendation"))
            if rec is None:
                continue
            out.append(
                InvestorOpinion(
                    investor_name=str(row.get("investor_name", "?")),
                    recommendation=rec,
                    confidence=float(row.get("confidence") or 0.0),
                    reasoning=str(row.get("reasoning") or ""),
                    key_factors=tuple(row.get("key_factors") or []),
                )
            )
        return out

    def get_receipt(self, prediction_id: str) -> StoredReceipt | None:
        response = (
            self._client.table(self._table)
            .select(_RECEIPT_COLUMNS)
            .eq("id", prediction_id)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        return _stored_receipt(rows[0]) if rows else None

    def find_receipts(self, symbol: str, day: date) -> list[StoredReceipt]:
        start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        response = (
            self._client.table(self._table)
            .select(_RECEIPT_COLUMNS)
            .eq("symbol", symbol)
            .gte("timestamp", start.isoformat())
            .lt("timestamp", (start + timedelta(days=1)).isoformat())
            .order("timestamp", desc=False)
            .execute()
        )
        return [_stored_receipt(row) for row in _rows(response)]

    def save_council_votes(
        self,
        prediction_id: str,
        symbol: str,
        votes: list[InvestorOpinion],
        *,
        debate_round: int = FIRST_COUNCIL_ROUND,
    ) -> None:
        if not votes:
            return
        rows = [
            _serialize_record(
                {
                    "prediction_id": prediction_id,
                    "symbol": symbol,
                    "investor_name": vote.investor_name,
                    "recommendation": vote.recommendation,
                    "confidence": vote.confidence,
                    "reasoning": vote.reasoning,
                    "key_factors": vote.key_factors,
                }
            )
            for vote in votes
        ]
        # ML-40: runda 1 to DEFAULT kolumny z 082, więc bez debaty zapis jak przed
        # migracją; klucz `round` niesie tylko rewizja (debata czeka na 082).
        if debate_round != FIRST_COUNCIL_ROUND:
            rows = [{**row, "round": debate_round} for row in rows]
        table = self._client.table(self._council_votes_table)
        if self._council_votes_upsert_enabled:
            # Re-run w tej samej godzinie nadpisuje głosy zamiast je dublować.
            table.upsert(rows, on_conflict="prediction_id,investor_name,round").execute()
            return
        table.insert(rows).execute()

    def save_llm_calls(self, calls: Sequence[LLMCallEntry]) -> None:
        # Jeden insert na przebieg (batch po cyklu), nie request na wywołanie LLM.
        if not calls:
            return
        rows = [
            {
                "run_id": call.run_id,
                "symbol": call.symbol,
                "node": call.node,
                "provider": call.usage.provider,
                "model": call.usage.model,
                "input_tokens": call.usage.input_tokens,
                "output_tokens": call.usage.output_tokens,
                "cache_read_tokens": call.usage.cache_read_tokens,
                "cache_write_tokens": call.usage.cache_write_tokens,
                "cost_usd": call.cost_usd,
                "price_as_of": (
                    call.price_as_of.isoformat() if call.price_as_of is not None else None
                ),
            }
            for call in calls
        ]
        self._client.table(LLM_CALLS_TABLE).insert(rows).execute()

    def get_llm_cost_since(self, since: datetime) -> float:
        # Paginacja (#8): miesiąc ledgera to tysiące wierszy. Suma po stronie klienta,
        # bo hosted PostgREST nie ma włączonych agregatów; NULL (model spoza cennika)
        # pomijamy — kosztu nie zgadujemy.
        query = (
            self._client.table(LLM_CALLS_TABLE)
            .select("cost_usd")
            .gte("created_at", since.isoformat())
            .order("id", desc=False)
        )
        return sum(
            (float(cost) for row in _paginate(query) if (cost := row.get("cost_usd")) is not None),
            0.0,
        )

    def get_llm_costs_since(self, since: datetime) -> list[LedgerCost]:
        # Paginacja (#8): tydzień ledgera to tysiące wierszy; agregację robi domena.
        query = (
            self._client.table(LLM_CALLS_TABLE)
            .select("symbol, cost_usd")
            .gte("created_at", since.isoformat())
            .order("id", desc=False)
        )
        return [
            LedgerCost(
                symbol=None if (symbol := row.get("symbol")) is None else str(symbol),
                cost_usd=None if (cost := row.get("cost_usd")) is None else float(cost),
            )
            for row in _paginate(query)
        ]

    def save_cycle_run(self, run: CycleRun) -> None:
        # Jeden insert na przebieg; `paid_calls` idzie jako JSONB (dict, nie string).
        row = {
            "loop": run.loop,
            "run_id": run.run_id,
            "run_attempt": run.run_attempt,
            "started_at": run.started_at.isoformat(),
            "duration_s": run.duration_s,
            "eligible": run.eligible,
            "saved": run.saved,
            "failures": run.failures,
            "paid_calls": dict(run.paid_calls),
            "report_delivered": run.report_delivered,
        }
        self._client.table(CYCLE_RUNS_TABLE).insert(row).execute()

    def get_cycle_runs_since(self, since: datetime) -> list[CycleRun]:
        # Paginacja (#8) i `id` jako drugi klucz, jak w pozostałych odczytach historii.
        query = (
            self._client.table(CYCLE_RUNS_TABLE)
            .select(
                "loop, run_id, run_attempt, started_at, duration_s, eligible, saved, "
                "failures, paid_calls, report_delivered"
            )
            .gte("started_at", since.isoformat())
            .order("started_at", desc=False)
            .order("id", desc=False)
        )
        return [_cycle_run(row) for row in _paginate(query)]

    def get_reference_reads(self, since: datetime) -> list[ReferenceRead]:
        # Paginacja (#8) i `id` jako drugi klucz, jak w pozostałych odczytach historii.
        query = (
            self._client.table(REFERENCE_PREDICTIONS_TABLE)
            .select("symbol, price, read_at")
            .gte("read_at", since.isoformat())
            .order("read_at", desc=False)
            .order("id", desc=False)
        )
        return [
            ReferenceRead(
                symbol=str(row["symbol"]),
                price=Decimal(str(row["price"])),
                read_at=datetime.fromisoformat(str(row["read_at"])),
            )
            for row in _paginate(query)
        ]

    def get_open_reference_calls(self) -> list[OpenReferenceCall]:
        query = (
            self._client.table(REFERENCE_PREDICTIONS_TABLE)
            .select("id, symbol, predicted_trend, price, read_at")
            .not_.is_("predicted_trend", "null")
            .is_("outcome", "null")
            .order("read_at", desc=False)
            .order("id", desc=False)
        )
        return [
            OpenReferenceCall(
                id=str(row["id"]),
                call=ReferenceCall(
                    symbol=str(row["symbol"]),
                    trend=TrendDirection(str(row["predicted_trend"])),
                    price_at_call=Decimal(str(row["price"])),
                    called_at=datetime.fromisoformat(str(row["read_at"])),
                ),
            )
            for row in _paginate(query)
        ]

    def save_reference_read(self, read: ReferenceRead, call: ReferenceCall | None) -> None:
        # Etykietę stawia jawnie kod (stała domeny), nie tylko DEFAULT kolumny.
        record = _serialize_record(
            {
                "symbol": read.symbol,
                "read_at": read.read_at.isoformat(),
                "price": read.price,
                "predicted_trend": None if call is None else call.trend.value,
                "recipe_version": REFERENCE_RECIPE_VERSION,
            }
        )
        self._client.table(REFERENCE_PREDICTIONS_TABLE).insert(record).execute()

    def settle_reference_call(
        self,
        record_id: str,
        outcome: ReferenceOutcome,
        settled_price: Decimal | None,
        settled_at: datetime,
    ) -> None:
        record = _serialize_record(
            {
                "outcome": outcome.value,
                "settled_price": settled_price,
                "settled_at": settled_at.isoformat(),
            }
        )
        self._client.table(REFERENCE_PREDICTIONS_TABLE).update(record).eq("id", record_id).execute()

    def save_corporate_action(self, action: SplitSuspicion) -> None:
        record = _serialize_record(
            {
                "symbol": action.symbol,
                "kind": action.kind.value,
                "ratio": action.ratio,
                "prev_close": action.prev_close,
                "reference_price": action.reference_price,
                "detected_at": action.detected_at.isoformat(),
                "detected_on": action.detected_on.isoformat(),
            },
            _CORPORATE_ACTION_PRICE_COLUMNS,
        )
        # ON CONFLICT DO NOTHING: powtórka cyklu tego dnia nie dubluje zdarzenia.
        self._client.table(CORPORATE_ACTIONS_TABLE).upsert(
            record, on_conflict=_CORPORATE_ACTION_UNIQUE, ignore_duplicates=True
        ).execute()

    @_translates_api_errors
    def get_open_invalidations(self, since: datetime) -> list[OpenInvalidation]:
        # ML-27 (migracja 074): indeks częściowy obsługuje dokładnie ten filtr.
        response = (
            self._client.table(self._table)
            .select(
                "id, symbol, predicted_trend, price_at_prediction, invalidation_price, timestamp"
            )
            .not_.is_("invalidation_price", "null")
            .is_("actual_price_after_12h", "null")
            .is_("invalidation_alerted_at", "null")
            .gte("timestamp", since.isoformat())
            .order("timestamp", desc=True)
            .execute()
        )
        return [
            OpenInvalidation(
                prediction_id=str(row["id"]),
                symbol=str(row["symbol"]),
                predicted_trend=str(row["predicted_trend"]),
                price_at_prediction=Decimal(str(row["price_at_prediction"])),
                invalidation_price=Decimal(str(row["invalidation_price"])),
                predicted_at=datetime.fromisoformat(str(row["timestamp"])),
            )
            for row in _rows(response)
        ]

    @_translates_api_errors
    def mark_invalidation_alerted(self, prediction_id: str, at: datetime) -> None:
        # Filtr NULL: drugi przebieg watcha nie przesuwa stempla pierwszego alarmu.
        (
            self._client.table(self._table)
            .update({"invalidation_alerted_at": at.isoformat()})
            .eq("id", prediction_id)
            .is_("invalidation_alerted_at", "null")
            .execute()
        )

    @_translates_api_errors
    def append_paper_trade(self, entry: PaperTradeEntry) -> LedgerRow | None:
        # WIA-59: hash liczy domena za głową odczytaną tuż przed zapisem. Równoległy
        # symbol z tą samą głową przegrywa w bazie (UNIQUE(prev_hash) → 23505 albo
        # trigger głowy → 40001) i ponawia od nowej głowy. ON CONFLICT (prediction_id)
        # DO NOTHING: re-run tej samej predykcji nie dopisuje drugiej pozycji.
        conflict: APIError | None = None
        for _attempt in range(_PAPER_LEDGER_APPEND_ATTEMPTS):
            head = self.get_paper_ledger_head()
            row = chain_entry(entry, GENESIS_HASH if head is None else head.row_hash)
            try:
                response = (
                    self._client.table(PAPER_TRADES_TABLE)
                    .upsert(
                        _paper_trade_record(row),
                        on_conflict="prediction_id",
                        ignore_duplicates=True,
                    )
                    .execute()
                )
            except APIError as exc:
                if exc.code not in _PAPER_LEDGER_CONFLICT_CODES:
                    raise
                conflict = exc
                continue
            return row if _rows(response) else None
        assert conflict is not None
        raise conflict

    @_translates_api_errors
    def close_paper_trade(self, prediction_id: str, close: PaperTradeClose) -> None:
        # Trigger 073 pozwala zamknąć pozycję raz; filtr `exit_price IS NULL` robi
        # z powtórnego rozliczenia (retry cyklu) no-op zamiast błędu triggera.
        (
            self._client.table(PAPER_TRADES_TABLE)
            .update(
                {
                    "exit_price": str(Money(close.exit_price).for_storage().amount),
                    "exit_ts": utc_timestamp(close.exit_ts),
                    "pnl": str(close.pnl),
                    "exit_reason": close.exit_reason.value,
                }
            )
            .eq("prediction_id", prediction_id)
            .is_("exit_price", "null")
            .execute()
        )

    @_translates_api_errors
    def get_paper_trades(self, days: int) -> list[LedgerTrade]:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        query = (
            self._client.table(PAPER_TRADES_TABLE)
            .select(_PAPER_TRADE_COLUMNS)
            .gte("entry_ts", cutoff.isoformat())
            .order("seq")
        )
        return [_ledger_trade(row) for row in _paginate(query)]

    @_translates_api_errors
    def get_paper_ledger_head(self) -> LedgerHead | None:
        response = (
            self._client.table(PAPER_TRADES_TABLE)
            .select("seq, row_hash")
            .order("seq", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        return LedgerHead(seq=int(rows[0]["seq"]), row_hash=str(rows[0]["row_hash"]))

    def expire_unverified_predictions(self, symbol: str, older_than_hours: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(hours=older_than_hours)
        # Ostra granica (`lt`) dopełnia `gte` z `get_unverified_prediction`: wiersz
        # albo jeszcze czeka na ocenę, albo wygasa — nigdy jedno i drugie.
        query = (
            self._client.table(self._table)
            .select("id, data_quality_flags")
            .eq("symbol", symbol)
            .is_("actual_price_after_12h", "null")
            .lt("timestamp", cutoff.isoformat())
        )
        expired = 0
        for row in _paginate(query.order("timestamp", desc=False).order("id", desc=False)):
            flags = list(row.get("data_quality_flags") or [])
            if EXPIRED_UNVERIFIED_FLAG in flags:
                continue
            # Flagi per wiersz: dopisujemy do istniejących (np. av_keys_exhausted),
            # a strażnik `actual_price IS NULL` pomija wiersz rozliczony w międzyczasie.
            response = (
                self._client.table(self._table)
                .update({"data_quality_flags": [*flags, EXPIRED_UNVERIFIED_FLAG]})
                .eq("id", row["id"])
                .is_("actual_price_after_12h", "null")
                .execute()
            )
            if _rows(response):
                expired += 1
        return expired

    def count_expired_unverified(self, days: int) -> int:
        # WIA-49: flaga żyje w `data_quality_flags` (JSONB, 008), więc zliczamy wiersze
        # okna z `cs ["expired_unverified"]`; `actual_price IS NULL` odsiewa wiersz
        # rozliczony po oznaczeniu. Paginacja po `id` — okno 30 dni może mieć >1000 wierszy.
        cutoff = datetime.now(UTC) - timedelta(days=days)
        query = (
            self._client.table(self._table)
            .select("id")
            .gte("timestamp", cutoff.isoformat())
            .is_("actual_price_after_12h", "null")
            .contains("data_quality_flags", f'["{EXPIRED_UNVERIFIED_FLAG}"]')
        )
        try:
            return len(_paginate(query.order("id", desc=False)))
        except (APIError, httpx.HTTPError) as exc:
            _log_degraded_read(
                "expired predictions count",
                "count of expired predictions taken from this run only",
                "008",
                exc,
            )
            return 0

    def update_prediction_accuracy(
        self,
        prediction_id: str,
        actual_price: Decimal,
        accuracy_score: float,
        is_trend_correct: bool,
        insight: str,
        correction_category: MistakeCategory | None = None,
        *,
        stale_quote: bool | None = None,
        session_close: SessionCloseSettlement | None = None,
        entry_price: Decimal | None = None,
    ) -> None:
        settlement: dict[str, Any] = {
            "actual_price_after_12h": actual_price,
            "accuracy_score": accuracy_score,
            "is_trend_correct": is_trend_correct,
            "correction_insights": insight,
            **self._category_payload(correction_category),
        }
        # DAT-14: klucz tylko, gdy graf go podał (flaga po migracji 055) — PGRST204.
        if stale_quote is not None:
            settlement[_STALE_QUOTE_COLUMN] = stale_quote
        # WIA-61: klucze tylko, gdy graf podał zamknięcie (flaga po migracji 059).
        if session_close is not None:
            settlement.update(_session_close_payload(session_close))
        # WIA-54: klucz tylko, gdy graf podał cenę wykonalną (flaga po migracji 072).
        if entry_price is not None:
            settlement["entry_price"] = entry_price
        payload = _serialize_record(settlement, _PREDICTION_PRICE_COLUMNS)
        # Idempotency guard: aktualizujemy tylko gdy actual_price_after_12h jest
        # NULL. Bez tego dwa równoczesne cykle (np. ręczny workflow_dispatch
        # nakładający się na scheduled run) mogłyby nadpisać już ocenioną
        # predykcję drugą oceną z lekko innej ceny.
        (
            self._client.table(self._table)
            .update(payload)
            .eq("id", prediction_id)
            .is_("actual_price_after_12h", "null")
            .execute()
        )

    def _category_payload(self, category: MistakeCategory | None) -> dict[str, str]:
        """WIA-57: klucz `correction_category` (migracja 046) tylko za flagą."""
        if category is None or not self._correction_category_enabled:
            return {}
        return {"correction_category": category.value}
