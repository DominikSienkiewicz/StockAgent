from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, NamedTuple, cast

from postgrest.exceptions import APIError
from supabase import create_client

from src.application.ports import PreviousVerdict, RepositoryPort
from src.application.report_council_history import InvestorHistory
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.equity_curve import directional_hit_rate
from src.domain.prediction import Prediction, normalize_trend
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.regime import MarketRegime, RegimeState
from src.domain.regime_key import canonical_regime_key
from src.domain.significance import effective_n
from src.domain.track_record import group_by_cycle
from src.domain.value_objects import (
    FUNDAMENTALS_CACHE_TTL_HOURS,
    AssetType,
    Fundamentals,
    Money,
)

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "prediction_logs"
DEFAULT_FEATURE_VIEW = "ml_feature_store"
DEFAULT_SNAPSHOT_TABLE = "price_snapshots"
DEFAULT_COUNCIL_VOTES_TABLE = "council_votes"
DEFAULT_QUOTA_ALERTS_TABLE = "quota_alerts"
DEFAULT_FEATURE_SNAPSHOT_TABLE = "feature_snapshots"
DEFAULT_REGIME_HISTORY_TABLE = "regime_history"

# Retry parametry dla refresh_feature_store — Slow Loop wywołuje RPC przed
# treningiem; transient błąd sieci nie powinien wywalać całego treningu.
_REFRESH_MAX_ATTEMPTS = 3
_REFRESH_BACKOFF_S = 2.0

# Hosted Supabase / PostgREST tnie SELECT do ~1000 wierszy BEZ błędu (#8).
# Agregaty (get_accuracy_stats itd.) muszą paginować po .range(), inaczej
# trafność liczona jest na obciętym, niedeterministycznym podzbiorze.
_PAGE_SIZE = 1000


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


class MissingColumnError(RuntimeError):
    """Odczyt prosił o kolumnę, której baza jeszcze nie ma — migracja nie
    została zaaplikowana. `column` to nazwa kolumny `prediction_logs`."""

    def __init__(self, column: str) -> None:
        super().__init__(f"column prediction_logs.{column} does not exist")
        self.column = column


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


# Flaga, którą predict_node zostawia przy neutralnej atrapie po awarii głównego
# LLM („SIDEWAYS 50%”). Taki wiersz nie jest predykcją i nie może zasilać
# hit-rate ani kalibracji (WIA-09).
_LLM_FALLBACK_FLAG = "llm_analysis_failed"


def _without_llm_fallback(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Odsiewa wiersze-atrapy po awarii LLM (flaga `llm_analysis_failed`).

    Filtr po stronie Pythona, nie PostgREST: `NOT (flags @> ...)` daje NULL dla
    wiersza z `data_quality_flags = NULL`, więc filtr w zapytaniu po cichu
    wyciąłby też takie wiersze. Wymaga `data_quality_flags` w SELECT.
    """
    return [row for row in rows if _LLM_FALLBACK_FLAG not in (row.get("data_quality_flags") or [])]


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


def _serialize(value: Any) -> Any:
    """Decimal → str (JSON-safe, zachowuje precyzję) — pozostałe typy bez zmian."""
    if isinstance(value, Decimal):
        return str(value)
    return value


def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: _serialize(v) for k, v in record.items()}


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _rows(response: Any) -> list[dict[str, Any]]:
    """Narrow supabase response.data (JSON union) do list[dict]."""
    data = response.data or []
    return cast(list[dict[str, Any]], data)


def _deserialize_verdict(raw: dict[str, Any]) -> CouncilVerdict:
    """Odtwarza `CouncilVerdict` z JSONB (migracja 005). Opinie inwestorów są
    zapisane jako lista dictów — mapujemy je z powrotem na value objecty."""
    opinions = tuple(
        InvestorOpinion(
            investor_name=str(o.get("investor_name", "")),
            recommendation=cast(Any, o.get("recommendation", "HOLD")),
            confidence=float(o.get("confidence") or 0.0),
            reasoning=str(o.get("reasoning", "")),
            key_factors=tuple(o.get("key_factors") or ()),
        )
        for o in raw.get("investor_opinions") or []
    )
    return CouncilVerdict(
        final_recommendation=cast(Any, raw.get("final_recommendation", "HOLD")),
        consensus_strength=float(raw.get("consensus_strength") or 0.0),
        summary=str(raw.get("summary", "")),
        dissenting_views=tuple(raw.get("dissenting_views") or ()),
        investor_opinions=opinions,
    )


class SupabaseRepository(RepositoryPort):
    """Adapter dla Supabase (PostgreSQL via PostgREST).

    Tabela `prediction_logs` + zmaterializowany widok `ml_feature_store`
    — schema patrz docs/08-database-schema.md.
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
    ) -> None:
        self._client = create_client(url, key)
        # ML-10: kolumny baseline'ów dryfu i „sam LLM" istnieją od migracji 028 —
        # bez flagi payload karty ich nie niesie (PGRST204 zgubiłby cały wiersz).
        self._scorecard_baselines_enabled = scorecard_baselines_enabled
        self._table = table_name
        self._view = feature_view
        self._snapshot_table = snapshot_table
        self._council_votes_table = council_votes_table
        self._quota_alerts_table = quota_alerts_table

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

    def get_price_history(
        self, symbol: str, days: int
    ) -> list[tuple[datetime, Money]]:
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
            out.append(
                (datetime.fromisoformat(ts), Money(Decimal(str(price))))
            )
        return out

    def get_last_prediction_price(self, symbol: str) -> Money | None:
        # Cena ostatniej zalogowanej predykcji — referencja dla cechy price_delta
        # w inference (musi pokrywać się z LAG(price_at_prediction) w widoku
        # ml_feature_store, inaczej cecha ma inny rozkład w treningu i predykcji).
        response = (
            self._client.table(self._table)
            .select("price_at_prediction")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        if not rows or rows[0].get("price_at_prediction") is None:
            return None
        return Money(Decimal(str(rows[0]["price_at_prediction"])))

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
        query = _resolved_window(
            self._client.table(self._table).select(
                "symbol, predicted_trend, is_trend_correct, timestamp, "
                "reasoning_text, correction_insights, price_at_prediction, "
                "actual_price_after_12h, data_quality_flags"
            ),
            cutoff,
            since,
        )
        return _without_llm_fallback(_paginate(query))

    def get_resolved_predictions(self, days: int) -> list[dict[str, Any]]:
        """Zamknięte predykcje (oceniony kierunek) z ostatnich `days` dni.

        Ten sam zestaw kolumn co `get_recently_resolved_predictions` (godziny),
        ale szersze okno dniowe — zasila track-record (krzywa kapitału, panel
        lessons). `parse_resolved_predictions` parsuje oba kształty. Paginacja
        (#8) — pełen zbiór, nie obcięte ~1000 wierszy PostgREST."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        query = (
            self._client.table(self._table)
            .select(
                "symbol, predicted_trend, is_trend_correct, timestamp, "
                "reasoning_text, correction_insights, price_at_prediction, "
                "actual_price_after_12h, data_quality_flags"
            )
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .order("timestamp", desc=True)
        )
        return _without_llm_fallback(_paginate(query))

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
                "id, symbol, predicted_trend, is_trend_correct, timestamp, "
                "reasoning_text, correction_insights, price_at_prediction, "
                "predicted_target_price, actual_price_after_12h, "
                "confidence_score, council_verdict"
            ),
            cutoff,
            since,
        )
        return _paginate(query)

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
        columns = (
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
            return _without_llm_fallback(_paginate(query))
        except APIError as exc:
            missing = _missing_eval_column(exc, extra_fields)
            if missing is None:
                raise
            raise MissingColumnError(missing) from exc

    def get_settled_moves(
        self, days: int, *, with_intervals: bool = False
    ) -> list[dict[str, Any]]:
        # ML-34: rosnąco po czasie, bo replay ACI idzie chronologicznie. Paginacja
        # (#8) — okno kalibracyjne nie może być po cichu obcięte do ~1000 wierszy.
        cutoff = datetime.now(UTC) - timedelta(days=days)
        columns = (
            "symbol, timestamp, price_at_prediction, predicted_target_price, "
            "actual_price_after_12h"
        )
        if with_intervals:
            columns += ", interval_low, interval_high"
        query = (
            self._client.table(self._table)
            .select(columns)
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("actual_price_after_12h", "null")
            .order("timestamp", desc=False)
        )
        return _paginate(query)

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
        query = (
            self._client.table(self._table)
            .select("is_trend_correct, predicted_trend, data_quality_flags, timestamp")
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .order("timestamp", desc=True)
        )
        rows = [
            r
            for r in _without_llm_fallback(_paginate(query))
            if r.get("is_trend_correct") is not None
        ]
        calls = [
            (str(r.get("predicted_trend") or ""), bool(r["is_trend_correct"]))
            for r in rows
        ]
        directional = directional_hit_rate(calls)
        n_eff = effective_n(
            group_by_cycle(
                (_utc_day_key(r.get("timestamp")), bool(r["is_trend_correct"]))
                for r in rows
            )
        )

        if not calls:
            return {
                "mean_accuracy": None,
                "sample_count": 0,
                "correct_count": 0,
                "days_window": days,
                "directional_count": 0,
                "directional_correct_count": 0,
                "n_eff": 0.0,
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
        )

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
        self._client.table("fundamentals_cache").upsert(payload).execute()

    # -----------------------------------------------------------------------
    # Writes
    # -----------------------------------------------------------------------

    def save_price_snapshot(self, symbol: str, price: Money) -> None:
        # Upsert na (symbol, timestamp_hour) (#7) — re-run GHA / nakładający się
        # workflow_dispatch / retry LangGraph aktualizuje istniejący wiersz w tej
        # samej godzinie zamiast duplikować (duplikaty zaburzały trafność).
        record = _serialize_record({"symbol": symbol, "price": price.amount})
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

    def save_regime_state(
        self, cycle_at: datetime, raw: MarketRegime, state: RegimeState
    ) -> None:
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
        record = _serialize_record(prediction)
        response = (
            self._client.table(self._table)
            .upsert(record, on_conflict="symbol,timestamp_hour")
            .execute()
        )
        rows = _rows(response)
        if not rows:
            raise RuntimeError(
                f"Supabase upsert returned no data for {record.get('symbol')}"
            )
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
        self, embedding: list[float], limit: int = 3, regime: str | None = None
    ) -> list[dict[str, Any]]:
        """Similarity search nad `prediction_logs.embedding` (pgvector) przez
        RPC `match_news_embeddings` (migracja 011; filtr reżimu z migracji 017).

        Graceful: gdy RPC nie istnieje (migracja niezaaplikowana) lub błąd
        sieci/Supabase — zwraca `[]`, żeby RAG był czysto opcjonalnym
        wzbogaceniem, a nie twardą zależnością predykcji.

        `regime`: gdy podany, dokłada parametr `filter_regime` do RPC (migracja
        017 filtruje analogi po reżimie; NULL/UNKNOWN w bazie przepuszcza)."""
        params: dict[str, Any] = {
            "query_embedding": embedding,
            "match_count": limit,
        }
        if regime is not None:
            params["filter_regime"] = canonical_regime_key(regime)
        try:
            response = self._client.rpc("match_news_embeddings", params).execute()
        except Exception:
            logger.warning(
                "match_news_embeddings RPC unavailable — RAG retrieval skipped "
                "(apply migration 011 to enable).",
                exc_info=True,
            )
            return []
        return _rows(response)

    def get_resolved_for_calibration(self, days: int) -> list[dict[str, Any]]:
        # NIEZ-15: paginacja + `id` jako drugi klucz — przy równych timestampach
        # granica strony musi być deterministyczna (bez gubienia/dublowania wierszy).
        cutoff = datetime.now(UTC) - timedelta(days=days)
        query = (
            self._client.table(self._table)
            .select("id, confidence_score, is_trend_correct, reasoning_text, data_quality_flags")
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .not_.is_("confidence_score", "null")
            .order("timestamp", desc=True)
            .order("id", desc=True)
        )
        return _without_llm_fallback(_paginate(query))

    def save_calibration(
        self, prediction_id: str, score: float, insight: str
    ) -> None:
        (
            self._client.table(self._table)
            .update(
                {"confidence_calibration": score, "calibration_insight": insight}
            )
            .eq("id", prediction_id)
            .execute()
        )

    def get_persona_accuracy(
        self, asset_type: AssetType, window_days: int = 90
    ) -> dict[str, float]:
        # Liczenie trafności per persona (rekomendacja vs realny ruch ceny) robi
        # RPC `persona_accuracy_weights` w SQL (migracja 015) — Python tylko mapuje
        # wynik. `asset_type` zarezerwowany na przyszłe ważenie per klasa aktywa
        # (dziś council_votes nie nosi typu → waga globalna). Graceful: brak RPC
        # / błąd → {} (ważenie wyłączone, każda persona waży 1.0).
        _ = asset_type
        try:
            response = self._client.rpc(
                "persona_accuracy_weights", {"window_days": window_days}
            ).execute()
        except Exception:
            logger.warning(
                "persona_accuracy_weights RPC unavailable — ważenie person pominięte "
                "(zaaplikuj migrację 015).",
                exc_info=True,
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

    def get_persona_track_record(
        self, window_days: int = 90
    ) -> dict[str, tuple[float, int]]:
        # RPC `persona_accuracy_stats` (migracja 018) — ten sam JOIN co 015, ale
        # zwraca surowy hit-rate ORAZ COUNT(*) głosów. RPC z 015 niesie tylko
        # przeskalowaną wagę, z której nie da się odtworzyć wielkości próbki.
        # Graceful: brak RPC / błąd → {} (leaderboard sam się chowa).
        try:
            response = self._client.rpc(
                "persona_accuracy_stats", {"window_days": window_days}
            ).execute()
        except Exception:
            logger.warning(
                "persona_accuracy_stats RPC unavailable — leaderboard person "
                "pominięty (zaaplikuj migrację 018).",
                exc_info=True,
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

    def get_unrevealed_commitments(self, days: int = 30) -> list[dict[str, Any]]:
        # Graceful: brak migracji 024 → [] (sweep nie ma czego ujawnić).
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        try:
            # NIEZ-15: bez paginacji sweep widział max ~1000 commitmentów.
            query = (
                self._client.table(self._table)
                .select(
                    "id, symbol, predicted_trend, predicted_target_price, "
                    "price_at_prediction, actual_price_after_12h, "
                    "is_trend_correct, commitment_hash, commitment_salt"
                )
                .gte("timestamp", cutoff)
                .not_.is_("commitment_hash", "null")
                .not_.is_("actual_price_after_12h", "null")
                .is_("revealed_at", "null")
                .order("timestamp", desc=False)
                .order("id", desc=False)
            )
            return _paginate(query)
        except Exception:
            logger.warning(
                "Attestation columns unavailable — reveal sweep pominięty "
                "(zaaplikuj migrację 024).",
                exc_info=True,
            )
            return []

    def mark_commitment_revealed(self, prediction_id: str) -> None:
        self._client.table(self._table).update(
            {"revealed_at": datetime.now(UTC).isoformat()}
        ).eq("id", prediction_id).execute()

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
        self, symbol: str, alert_date: date, delta: float, direction: str
    ) -> None:
        # UNIQUE(symbol, alert_date) w migracji 021 jest twardym debounce'em —
        # nawet gdy dwa przebiegi watcha wystartują równolegle.
        self._client.table("shock_alerts").insert(
            {
                "symbol": symbol,
                "alert_date": alert_date.isoformat(),
                "delta": delta,
                "direction": direction,
            }
        ).execute()

    def get_sent_shock_alerts(self, since: date) -> set[tuple[str, date]]:
        # Graceful: brak migracji 021 → pusty zbiór (debounce zostaje tylko
        # w pamięci przebiegu, a UNIQUE i tak zablokuje duplikat przy zapisie).
        try:
            response = (
                self._client.table("shock_alerts")
                .select("symbol, alert_date")
                .gte("alert_date", since.isoformat())
                .execute()
            )
        except Exception:
            logger.warning(
                "shock_alerts niedostępne — debounce tylko w pamięci "
                "(zaaplikuj migrację 021).",
                exc_info=True,
            )
            return set()
        sent: set[tuple[str, date]] = set()
        for row in _rows(response):
            symbol, raw_date = row.get("symbol"), row.get("alert_date")
            if symbol is None or not isinstance(raw_date, str):
                continue
            sent.add((str(symbol), date.fromisoformat(raw_date)))
        return sent

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

    def save_model_scorecard(self, symbol: str, result: Mapping[str, Any]) -> None:
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
        except Exception:
            logger.warning(
                "model_scorecards niedostępne — karta kondycji modelu pominięta "
                "(zaaplikuj migrację 019).",
                exc_info=True,
            )
            return []
        return list(_rows(response))

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
        if investors:
            query = query.in_("investor_name", investors)
        # NIEZ-15: paginacja z `id` jako drugim kluczem (jak w odczytach Slow Loopa).
        rows = _paginate(query.order("timestamp", desc=True).order("id", desc=True))
        # Grupowanie per (inwestor, symbol); kolejność malejąca po czasie
        # zachowana (rekomendacje od najnowszej).
        grouped: dict[tuple[str, str], list[Literal["BUY", "SELL", "HOLD"]]] = {}
        for row in rows:
            name = str(row.get("investor_name") or "")
            sym = str(row.get("symbol") or "")
            rec_raw = row.get("recommendation")
            if not name or not sym or rec_raw not in ("BUY", "SELL", "HOLD"):
                continue
            rec = cast("Literal['BUY', 'SELL', 'HOLD']", rec_raw)
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

    def get_council_votes_for_prediction(
        self, prediction_id: str
    ) -> list[InvestorOpinion]:
        response = (
            self._client.table(self._council_votes_table)
            .select("investor_name, recommendation, confidence, reasoning, key_factors")
            .eq("prediction_id", prediction_id)
            .execute()
        )
        out: list[InvestorOpinion] = []
        for row in _rows(response):
            rec = row.get("recommendation")
            if rec not in ("BUY", "SELL", "HOLD"):
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

    def save_council_votes(
        self,
        prediction_id: str,
        symbol: str,
        votes: list[InvestorOpinion],
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
        self._client.table(self._council_votes_table).insert(rows).execute()

    def update_prediction_accuracy(
        self,
        prediction_id: str,
        actual_price: Decimal,
        accuracy_score: float,
        is_trend_correct: bool,
        insight: str,
    ) -> None:
        payload = _serialize_record(
            {
                "actual_price_after_12h": actual_price,
                "accuracy_score": accuracy_score,
                "is_trend_correct": is_trend_correct,
                "correction_insights": insight,
            }
        )
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
