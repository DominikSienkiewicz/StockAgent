from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, StateGraph

from src.application.cycle_telemetry import CycleTelemetry
from src.application.decision_receipt import RECEIPT_SCHEMA_V1, RECEIPT_SCHEMA_V2
from src.application.errors import LLMOutputIncomplete, PortError
from src.application.ml_features import (
    CORPORATE_EVENT_SNAPSHOT_FEATURES,
    FREE_SNAPSHOT_FEATURES,
    GATED_SNAPSHOT_EXTRAS,
    MARKET_SNAPSHOT_FEATURES,
    ML_FEATURE_COLUMNS,
    feature_set_version,
)
from src.application.ports import (
    LLM_FAILOVER_MODEL_KEY,
    LLM_FAILOVER_PROVIDER_KEY,
    LLM_ROUTED_MODEL_KEY,
    AdvisoryCouncilPort,
    AttestationPublisherPort,
    EmbeddingPort,
    FundamentalsPort,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    OptionsFlowPort,
    QuotePort,
    RepositoryPort,
    ReturnScaleUnavailableError,
    SentimentPort,
    ShadowModelPort,
    Tool,
    ToolUseLLMPort,
)
from src.application.prompts import (
    format_analog_line,
    format_corporate_events,
    format_prompt_delta,
    get_mistake_diagnosis_prompt,
    get_prediction_prompt,
)
from src.application.rag_rerank import rerank_analogs
from src.application.usage_ledger import usage_scope
from src.domain.alpha_fusion import AlphaFusionScore
from src.domain.asset import Asset, PriceDelta, effective_threshold
from src.domain.attestation import (
    REVEAL_STATUS_SUPERSEDED,
    StoredCommitment,
    canonicalize,
    commit_canonical,
    commitment_ledger_record,
    new_salt,
    prediction_commitment_payload,
)
from src.domain.attestation import SCHEMA_VERSION as ATTESTATION_SCHEMA_VERSION
from src.domain.conformal import CalibrationPoint, MoveInterval, move_interval
from src.domain.corporate_actions import SplitSuspicion, detect_split, to_pre_split_price
from src.domain.corporate_events import CorporateEvent
from src.domain.council import (
    CouncilInput,
    CouncilVerdict,
    Recommendation,
    vindicated_dissenters,
)
from src.domain.experiment import (
    ARM_BLIND_COUNCIL,
    ARM_NO_MEMORY,
    Experiment,
    has_treatment,
)
from src.domain.feature_attribution import FeatureContribution, top_contributions
from src.domain.finops import (
    PAID_LLM_SOURCES,
    SHADOW_LLM_SOURCE,
    paid_budget_allows,
    shadow_call_allowed,
)
from src.domain.forecast import ProbabilityForecast
from src.domain.implied_edge import (
    ImpliedEdge,
    evaluate_edge,
    model_move_from_prices,
)
from src.domain.invalidation import Invalidation, is_valid_invalidation
from src.domain.market_calendar import hours_to_next_trading_day
from src.domain.mistake_taxonomy import parse_mistake_diagnosis
from src.domain.paper_ledger import (
    ExitReason,
    LedgerCosts,
    PaperTradeEntry,
    close_trade,
    side_of,
)
from src.domain.prediction import (
    CORRECT_PREDICTION_INSIGHT,
    DIAGNOSIS_BATCH_PENDING_INSIGHT,
    DIAGNOSIS_SKIPPED_INSIGHT,
    STALE_QUOTE_INSIGHT,
    Prediction,
    TrendDirection,
    is_substantive_insight,
    normalize_trend,
    sideways_tolerance_for,
)
from src.domain.price_features import SIGMA_LOOKBACK, return_sigma
from src.domain.price_validation import OUTLIER_HISTORY_DAYS, is_unconfirmed_outlier
from src.domain.promotion import CHAMPION_MODEL_NAME, ShadowForecast
from src.domain.recipe import PredictionRecipe
from src.domain.regime_key import canonical_regime_key
from src.domain.session_close import SessionClose, SessionCloseSettlement, session_close_of
from src.domain.shadow_prompt import (
    SHADOW_PROMPT_MAX_PER_CYCLE,
    ShadowTrendForecast,
    in_shadow_sample,
)
from src.domain.value_objects import (
    AssetType,
    Fundamentals,
    Money,
    Quote,
    Ratio,
    Threshold,
    ValuationVerdict,
    clamp_unit,
    is_stale,
    settles_on_stale_quote,
    to_percent,
)

logger = logging.getLogger(__name__)


@contextmanager
def _timed_node(node_name: str, symbol: str, telemetry: CycleTelemetry) -> Iterator[None]:
    """#14: lekka instrumentacja czasu wykonania płatnego węzła.

    Mierzy elapsed_ms i loguje go na INFO (z nazwą węzła i symbolem). Czas
    jest raportowany ZAWSZE, nawet gdy węzeł rzuci wyjątek (FinOps: chcemy
    wiedzieć, jak długo trwało wywołanie, które padło). ARCH-11: ten sam czas
    trafia do telemetrii cyklu, a wyjątek liczy się tam jako awaria węzła.
    """
    start = time.perf_counter()
    failed = True
    try:
        yield
        failed = False
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "node=%s symbol=%s elapsed_ms=%.1f", node_name, symbol, elapsed_ms
        )
        telemetry.record_node_timing(node_name, symbol, elapsed_ms, failed=failed)


def _log_paid_call_line(node_name: str, symbol: str, detail: str) -> None:
    """Jawna linia w logu, gdy odpala się PŁATNE wywołanie zewnętrzne
    (LLM / sentyment / news / rada). Pozwala audytować FinOps — ile płatnych
    wywołań poszło w danym cyklu i z którego węzła."""
    logger.info(
        "paid call: node=%s symbol=%s %s", node_name, symbol, detail
    )


def _log_paid_call(
    deps: AgentGraphDeps, node_name: str, symbol: str, source: str, detail: str
) -> None:
    """Linia w logu + inkrement licznika płatnych wywołań cyklu.

    `source` to klucz cennika z `src/domain/finops.py` (`llm`, `sentiment`,
    `news`, `embedding`). Licznik (ARCH-11: `deps.telemetry`) jest współdzielony
    przez wszystkie symbole cyklu, bo raportujemy KOSZT CYKLU, nie symbolu.
    Rady tu nie liczymy: jej requesty liczy u źródła `MeteredLLMPort` (FIN-02).
    """
    _log_paid_call_line(node_name, symbol, detail)
    deps.telemetry.record_paid_call(source)


class SentimentPayload(TypedDict, total=False):
    """Wynik `SentimentPort.get_social_score` w stanie grafu (ARCH-10).

    Kontrakt adaptera Alpha Vantage NEWS_SENTIMENT (i jego atrapy): sentyment
    ważony relevance i liczniki artykułów. Pola są opcjonalne, bo graf czyta je
    przez `.get` z wartością domyślną, a brak pola flaguje jako `_missing`.
    """

    av_sentiment_score: float
    av_sentiment_label: str
    av_relevance_avg: float
    news_volume_24h: int
    high_relevance_count: int
    # Feed zdegradowany (np. wyczerpane klucze AV, atrapa lokalnego runu) — graf
    # dopisuje powód do `data_quality_flags`, bo zera nie znaczą spokojnego dnia.
    degraded_reason: str


class AgentState(TypedDict, total=False):
    symbol: str
    previous_price: Decimal
    # FIN-14: moment zapisu referencji ceny i werdykt domeny o jej stęchłości
    # (`is_reference_stale`). Stęchła referencja zamyka bramkę volatility.
    reference_at: datetime | None
    stale_reference: bool
    current_price: Decimal
    # DAT-14: chwila notowania ceny bieżącej (Finnhub `t`) i werdykt domeny, czy jest
    # sprzed ostatniej rozpoczętej sesji. Brak kluczy = cena bez czasu (krypto, graf
    # bez `quote_port`) — stęchłości nie oceniamy, jak przed DAT-14.
    quoted_at: datetime
    stale_quote: bool
    # WIA-61: zamknięcie ostatniej zakończonej sesji z tego samego notowania (za
    # `session_close_settlement_enabled`); None = notowanie go nie wyznacza.
    session_close: SessionClose | None
    # WIA-54: notowanie cyklu z otwarciem sesji (`o`) — reflect bierze z niego cenę
    # wykonalną rozliczanej predykcji (za `executable_entry_enabled`).
    session_quote: Quote
    # ARCH-04: ruch ceny od referencji jako UŁAMEK (`PriceDelta.fraction`), nie
    # procent. Prompt i rada dostają procent przez `to_percent`.
    delta: Ratio
    # ALR-10: cena odniesienia dryfu (ostatnia predykcja z okna albo początek okna,
    # podaje use case za `drift_gate_enabled`) i dryf od niej (ułamek, z check_price).
    # Brak klucza = bramka jak przed ALR-10.
    drift_reference: Decimal
    drift: Ratio
    status: str
    # FIN-12: ruch przeszedł bramkę volatility, ale cykl wyczerpał sufit płatnych
    # wywołań LLM — symbol kończy jako 'ignored' bez analizy i bez snapshotu ceny.
    budget_exhausted: bool
    # MAIL-18: próg EFEKTYWNY bramki (ułamek, po mnożnikach reżimu i earnings) —
    # wynik 'ignored' niesie go do sekcji ⏸ Pominięte maila. Tylko prezentacja.
    effective_threshold: Ratio

    # Opcjonalny obiekt Asset (przekazywany z zewnątrz, gdy dostępny asset_type)
    asset: Asset

    # Self-Reflection
    reflection_context: str

    # Zewnętrzne dane
    sentiment: SentimentPayload | None
    news: list[dict[str, Any]]

    # Fundamenty i wycena
    fundamentals: Fundamentals | None
    valuation_verdict: ValuationVerdict

    # Predykcja
    llm_analysis: dict[str, Any]
    # NIEZ-44: (dostawca, model) zapasowego LLM, gdy analizę zrobił on, a nie główny.
    # Brak klucza = główny LLM. Zmienia przepis zapisany przy predykcji.
    llm_failover: tuple[str, str]
    # FIN-21: model, który odpowiedział w kaskadzie routingu (`dostawca:model`), do
    # kwitu v2. Brak klucza = analiza bez routingu modeli.
    llm_model_used: str
    # ML-29: rozkład P(BULLISH/BEARISH/SIDEWAYS) z odpowiedzi LLM po normalizacji.
    # None = flaga off albo LLM nie podał poprawnego rozkładu (trend ze starych pól).
    trend_probabilities: dict[str, float] | None
    # ML-27: poziom unieważnienia tezy po walidacji domeny. None = flaga on, ale LLM
    # nie podał poprawnego poziomu; brak klucza = flaga off.
    invalidation: Invalidation | None
    ml_target_price: Decimal
    # ML-14: wektor cech dokładnie w postaci podanej modelowi (`ML_FEATURE_COLUMNS`)
    # — trafia do `feature_snapshots`, żeby trening nie odtwarzał go widokiem SQL.
    ml_features: dict[str, float]
    # G03: True = XGBoost nie ma wag, a `ml_target_price` to baseline "cena bez
    # zmian", nie prognoza modelu. Rada i raport nie mogą go podawać jako sygnału.
    ml_cold_start: bool
    # ML-33: prognozy cienia nogi ML (champion i challengerzy) do `shadow_predictions`.
    # Brak klucza = bez `ShadowSetup` w deps; wynik cyklu ich nie czyta.
    shadow_forecasts: tuple[ShadowForecast, ...]
    # EXP-26: trend wariantu promptu albo modelu w cieniu do `shadow_predictions`
    # (migracja 083). Brak klucza = bez wariantu albo symbol poza podpróbką.
    shadow_prompt_forecast: ShadowTrendForecast
    # ML-34: przedział 80% zwrotu do rozliczenia (conformal + ACI) z okna klasy
    # aktywa. Mail pokazuje go od razu; kolumny `interval_*` idą za flagą.
    move_interval: MoveInterval | None

    # Embedding podsumowania newsów — liczony raz w predict_node (RAG retrieval)
    # i reużyty w save_node, żeby nie embedować dwa razy na cykl.
    news_embedding: list[float] | None

    # Q5: precedent receipts — zwięzła lista top-3 analogów RAG użytych w tym
    # cyklu (`{summary, predicted_trend, is_trend_correct}`). Render-only:
    # przepływa do SymbolResult i raportu, NIE jest persystowana (brak migracji).
    similar_precedents: list[dict[str, Any]]

    # Q3: atrybucja cech ML (SHAP-lite) — render-only do raportu.
    feature_attribution: tuple[FeatureContribution, ...]

    # #14: composite smart-money z 5 źródeł alfa (ważona suma klasyfikacji
    # domenowych, renormalizowana po dostępnych źródłach). None = wszystkie flagi
    # alfa off → prompt i cechy identyczne jak przed #14.
    alpha_fusion: AlphaFusionScore | None

    # DAT-23: mnożnik ceny po podejrzanym splicie względem ceny przed nim (10:1 → 0,1),
    # z `check_price` za `split_detection_enabled`. Brak klucza = bez splitu.
    split_ratio: Decimal
    # DAT-15: odczyt ceny anomalny bez potwierdzenia — bramka zamknięta, bez
    # rozliczenia i bez snapshotu. Brak klucza = odczyt wiarygodny.
    price_outlier: bool

    # #18: dywergencja predykcji modelu od implied move rynku opcji.
    # None = brak IV / options_flow_enabled=false → sygnał nie powstaje.
    implied_edge: ImpliedEdge | None

    # #7 reżim rynku (label do promptu + mnożnik progu, ≥1.0) i #5 contagion
    # (werdykty skorelowanych spółek z tego cyklu) — przekazywane per symbol.
    regime_context: str
    regime_multiplier: float
    peer_context: tuple[tuple[str, str], ...]
    # #6 bramka earnings — mnożnik progu (≥1.0) zależny od bliskości raportu
    # kwartalnego. Mnoży się z regime_multiplier w `_pick_threshold`.
    earnings_multiplier: float

    # ML-31: cechy rynkowe cyklu policzone przed grafem (use case, flaga
    # `market_features_enabled`). Brak klucza = symbol bez cech rynkowych (None).
    market_features: dict[str, Any]
    # DAT-17: świeże raporty 8-K symbolu, wybrane przez use case przed grafem (flaga
    # `corporate_events_enabled`). Pusta krotka = wiadomo, że świeżego 8-K nie ma;
    # brak klucza = nie wiadomo (flaga off, ETF/krypto, EDGAR niedostępny).
    fresh_corporate_events: tuple[CorporateEvent, ...]

    # EXP-05: ramiona AKTYWNYCH eksperymentów tego symbolu-dnia `{id: ramię}`,
    # przydzielone przez use case. Brak klucza = cykl jak bez eksperymentów.
    experiment_arms: dict[str, str]

    # Persystencja
    prediction_id: str | None

    # Rada doradcza
    council_verdict: CouncilVerdict | None

    # Audit jakości danych wejściowych — flagi typu "av_sentiment_score_missing",
    # "news_volume_24h_invalid". Pusta lista = wszystko OK. Trafia do
    # prediction_logs.data_quality_flags i pozwala odsiać śmieciowe rekordy
    # przy treningu / analizach. Bez tego cichych zer w features nie da się
    # odróżnić od prawdziwych zer.
    data_quality_flags: list[str]


# Ile pierwszych newsów (po relevance filtering) czyta prompt, rada i zapis.
_NEWS_TITLES_LIMIT = 5


def _news_titles(news: list[dict[str, Any]], limit: int) -> list[str]:
    """Niepuste tytuły z `limit` pierwszych newsów — jedno źródło dla promptu
    predykcji, rady i `news_summary`.

    ML-09: prompt dostaje LISTĘ, bo fence przycina każdy element do 300 znaków;
    sklejony string był jednym elementem i gubił nagłówki 4–5."""
    return [str(title) for title in (item.get("title") for item in news[:limit]) if title]


def _summarize_news(news: list[dict[str, Any]]) -> str:
    """Konkatenacja tytułów najnowszych newsów w jednej linijce (zapis i embedding)."""
    return " | ".join(_news_titles(news, _NEWS_TITLES_LIMIT))


# Q5: ile analogów wystawiamy jako "precedent receipts" w raporcie. Trzymamy
# zwięźle (top-3), żeby blok "Na podstawie analogów" był czytelny i mailowalny.
_PRECEDENT_RECEIPT_LIMIT = 3


# Ile analogów RAG trafia do promptu (top-N z RPC po reranku).
_ANALOG_LIMIT = 3
# ML-41: po odsianiu nierozliczonych zostaje mniej kandydatów, więc retrieval
# punktu w czasie bierze ich tyle razy więcej (ta sama jedna darmowa RPC).
_POINT_IN_TIME_OVERFETCH = 4


def _format_analog(item: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Mapuje jeden analog RAG na `(linia_promptu, rekord_precedensu)`.

    Zwraca None, gdy analog nie ma streszczenia (nieprzydatny do promptu).
    Rekord precedensu trzyma surowe pola — escapowanie robi report_builder
    przez `_html` przy renderze.
    """
    summary = str(item.get("news_summary") or "").strip()
    if not summary:
        return None
    trend = item.get("predicted_trend") or "?"
    correct = item.get("is_trend_correct")
    insight = str(item.get("correction_insights") or "").strip()
    # Treść linii składa `format_analog_line` w prompts.py — tylko moduły
    # promptów wchodzą do `prompt_sha256`, więc tu zostają wyłącznie dane.
    # WIA-10: placeholder reflect_node (np. historyczne „diagnoza odłożona”)
    # nie jest wnioskiem — analog zostaje, ale bez fałszywej lekcji.
    line = format_analog_line(
        source=str(item.get("symbol") or "").strip(),
        trend=str(trend),
        is_trend_correct=correct,
        lesson=insight if is_substantive_insight(insight) else "",
        headlines=summary,
    )
    # #13: RPC zwraca `id` i `similarity`, a kod je gubił. Bez nich kwit
    # decyzyjny nie pozwala zaudytować, KTÓRY analog stał za predykcją.
    precedent = {
        "summary": summary,
        "predicted_trend": str(trend),
        "is_trend_correct": correct,
        "id": item.get("id"),
        "similarity": item.get("similarity"),
    }
    return line, precedent


def _build_similar_context(
    repository_port: RepositoryPort,
    embedding: list[float],
    symbol: str,
    outcome_weight: float = 0.0,
    regime: str | None = None,
    *,
    point_in_time: bool = False,
    before: datetime | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Buduje blok 'podobne historyczne sytuacje' do promptu (RAG) ORAZ zwraca
    rerankowane rekordy do reużycia jako precedent receipts (Q5).

    Zwraca `(similar_context, precedents)`:
      - `similar_context` — jedna linia na analog; prompt ogradza je fence'em
        ANALOGS (BEZ-06),
      - `precedents` — zwięzła lista top-N analogów `{summary, predicted_trend,
        is_trend_correct}` dla raportu ("dlaczego ta decyzja").

    W pełni graceful: brak RPC / pgvector / błąd → `("", [])` (predykcja działa
    bez RAG i bez receipts). Jeden retrieval + jeden rerank — zero dodatkowych
    płatnych wywołań ponad to, co RAG już robił. `outcome_weight` (#9): rerank
    analogów po realnym wyniku; 0.0 = sama kolejność RPC. `point_in_time` (ML-41,
    `rag_point_in_time_enabled`): tylko analogi rozliczone, z większej puli kandydatów;
    repozytorium dostaje symbol i `before` (chwila predykcji) — RPC 076 bierze analogi
    rozliczone przed nią i pomija ten sam symbol w oknie horyzontu.
    """
    try:
        if point_in_time:
            similar = repository_port.find_similar_predictions(
                embedding,
                limit=_ANALOG_LIMIT * _POINT_IN_TIME_OVERFETCH,
                regime=regime,
                symbol=symbol,
                before=before,
            )
        else:
            similar = repository_port.find_similar_predictions(
                embedding, limit=_ANALOG_LIMIT, regime=regime
            )
        candidates = list(similar or [])
        if point_in_time:
            # ML-41: tylko analogi z wynikiem — nierozliczony (także zapisany w TYM
            # cyklu, echo bieżących newsów) nie zajmuje miejsca w top-N ani w reranku.
            candidates = [item for item in candidates if item.get("is_trend_correct") is not None]
        similar = rerank_analogs(
            candidates,
            outcome_weight=outcome_weight,
            top_k=_ANALOG_LIMIT if point_in_time else None,
        )
        lines: list[str] = []
        precedents: list[dict[str, Any]] = []
        for item in similar or []:
            formatted = _format_analog(item)
            if formatted is None:
                continue
            line, precedent = formatted
            lines.append(line)
            if len(precedents) < _PRECEDENT_RECEIPT_LIMIT:
                precedents.append(precedent)
        return "\n".join(lines), precedents
    except Exception:
        logger.exception(
            "find_similar_predictions failed for %s — predicting without RAG", symbol
        )
        return "", []


def _float_or_default(value: Any, default: float) -> float:
    """Liczba albo `default` — także dla NaN/Inf (ML-35), tą samą polityką co
    `_validate_float`, tylko bez flagi; zapis predykcji flaguje przez `_saved_float`."""
    parsed, _flag = _validate_float(value, "", default)
    return parsed


def _validate_float(value: Any, field_name: str, default: float) -> tuple[float, str | None]:
    """Parsuje pole liczbowe i jednocześnie wykrywa skażone wejście.

    Zwraca `(parsed_value, flag_or_None)`. Flaga sygnalizuje *jaki* problem
    wykryto — None = pole brak danych (`_missing`), nie-numeryczne lub NaN/Inf
    = `_invalid`. Bez tego rozróżnienia 0.0 z padłego wejścia było
    nieodróżnialne od prawdziwego 0.0 i model uczył się na śmieciach.
    """
    if value is None:
        return default, f"{field_name}_missing"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default, f"{field_name}_invalid"
    if math.isnan(parsed) or math.isinf(parsed):
        return default, f"{field_name}_invalid"
    return parsed, None


def _saved_float(value: Any, field_name: str, default: float) -> tuple[float, str | None]:
    """ML-35: pole liczbowe zapisu predykcji — wartość jak `_validate_float`, ale flaga
    tylko dla wartości obecnej i skażonej (`_invalid`); brak pola to cichy default."""
    parsed, flag = _validate_float(value, field_name, default)
    return parsed, None if value is None else flag


# Dozwolone kierunki trendu z LLM → sygnał liczbowy dla cechy ML. Mapowanie
# jest case-insensitive (normalizacja do upper-case), bo LLM bywa niespójny
# w wielkości liter ("bullish" vs "BULLISH").
_TREND_SIGNALS: dict[str, int] = {"BULLISH": 1, "BEARISH": -1, "SIDEWAYS": 0}


def _normalize_trend(value: Any) -> tuple[str, str | None]:
    """Kanoniczny trend_direction z LLM i ewentualna flaga `trend_direction_invalid`.

    Reguła żyje w domenie (`normalize_trend`): brak → SIDEWAYS bez flagi,
    śmieć (np. "UP") → SIDEWAYS z flagą, analogicznie do `_validate_float`.
    Bez tej flagi śmieciowy kierunek udawał prawdziwy SIDEWAYS i model uczył
    się na cichym zerze. ML-07 (decyzja właściciela 2026-10-18): SIDEWAYS jest
    też ZAPISYWANY — bez NULL-a i bez pomijania predykcji.
    """
    trend, invalid = normalize_trend(value)
    return trend.value, "trend_direction_invalid" if invalid else None


def _parse_number(value: Any) -> float | None:
    """Liczba z odpowiedzi LLM albo None (brak, nie-liczba, NaN).
    Nieskończoność zostaje liczbą — `clamp_unit` przytnie ją do granicy."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


# ML-07: skalary jednostkowe z odpowiedzi LLM → nazwa flagi `*_out_of_range`.
# Flaga nosi nazwę kolumny prediction_logs / cechy ML (jak `_missing`/`_invalid`).
_LLM_UNIT_FIELDS: tuple[tuple[str, str], ...] = (
    ("confidence_score", "confidence_score"),
    ("av_agreement", "av_llm_agreement"),
)


# NIEZ-44: flaga jakości predykcji od zapasowego dostawcy LLM (track record ją odsiewa).
LLM_FAILOVER_FLAG = "llm_failover"
# ML-11: model w jednostkach σ bez σ symbolu (za krótka historia snapshotów albo błąd
# odczytu) — cykl idzie na baseline'ie „cena bez zmian”, jak przy cold-starcie.
ML_SIGMA_UNAVAILABLE_FLAG = "ml_sigma_unavailable"
# MAIL-12: rada miała zagłosować, ale nie dała werdyktu (żadna persona nie
# odpowiedziała albo adapter rzucił). Predykcja zostaje w track recordzie.
COUNCIL_UNAVAILABLE_FLAG = "council_unavailable"
# NIEZ-26: odpowiedź LLM ucięta limitem tokenów albo odrzucona przez model — obok
# `llm_analysis_failed` (atrapa dalej poza track recordem), odróżnia to od awarii dostawcy.
LLM_OUTPUT_TRUNCATED_FLAG = "llm_output_truncated"
# Znacznik failovera bez modelu — przepis i tak nie może udawać głównego modelu.
_UNKNOWN_FAILOVER_MODEL = "unknown"


def _split_failover_marker(
    raw: dict[str, Any],
) -> tuple[tuple[str, str] | None, dict[str, Any]]:
    """NIEZ-44: zdejmuje znacznik adaptera failover z wyniku LLM, zanim ktokolwiek
    go przeczyta. Zwraca `(dostawca, model)` zapasowego (None = odpowiedział główny)
    i analizę bez kluczy znacznika."""
    analysis = {
        key: value
        for key, value in raw.items()
        if key not in (LLM_FAILOVER_PROVIDER_KEY, LLM_FAILOVER_MODEL_KEY)
    }
    provider = raw.get(LLM_FAILOVER_PROVIDER_KEY)
    if not provider:
        return None, analysis
    model = raw.get(LLM_FAILOVER_MODEL_KEY)
    return (str(provider), str(model) if model else _UNKNOWN_FAILOVER_MODEL), analysis


def _split_routed_model(raw: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """FIN-21: zdejmuje znacznik kaskady routingu z wyniku LLM, zanim ktokolwiek go
    przeczyta. Zwraca model, który odpowiedział (None = bez routingu), i analizę bez
    znacznika."""
    if LLM_ROUTED_MODEL_KEY not in raw:
        return None, raw
    analysis = {key: value for key, value in raw.items() if key != LLM_ROUTED_MODEL_KEY}
    model = raw[LLM_ROUTED_MODEL_KEY]
    return (str(model) if model else None), analysis


def _effective_recipe(deps: AgentGraphDeps, state: AgentState) -> PredictionRecipe | None:
    """WIA-20 + NIEZ-44: przepis, z którego FAKTYCZNIE powstała predykcja.

    Po failoverze główny LLM w przepisie podmienia dostawca i model zapasowego,
    więc kwit i `recipe_version` niosą prawdziwy model, a wersja przepisu wydziela
    takie predykcje w osobny segment track recordu."""
    failover = state.get("llm_failover")
    if deps.recipe is None or failover is None:
        return deps.recipe
    provider, model = failover
    return dataclasses.replace(deps.recipe, llm_provider=provider, llm_model=model)


def _normalize_llm_analysis(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """ML-07: kanonizuje wyjście głównego LLM, zanim ktokolwiek je przeczyta.

    Zwraca dict o tym samym kształcie (z `trend_direction` zawsze obecnym)
    oraz flagi jakości. Trend → kanon albo SIDEWAYS (`_normalize_trend`);
    confidence i av_agreement spoza [0, 1] →
    `clamp_unit` + flaga `*_out_of_range`. Braku i nie-liczb NIE ruszamy:
    te przypadki flaguje dalej `_validate_float` (`_missing` / `_invalid`).
    Surowy tekst modelu był wcześniej zapisywany wprost — „bullish” dawał
    OBSERWUJ zamiast KUP, a później wywracał reflect przy TrendDirection(...).
    """
    analysis = dict(raw)
    flags: list[str] = []
    trend, trend_flag = _normalize_trend(raw.get("trend_direction"))
    analysis["trend_direction"] = trend
    if trend_flag is not None:
        flags.append(trend_flag)
    for key, flag_name in _LLM_UNIT_FIELDS:
        number = _parse_number(raw.get(key))
        if number is None:
            continue
        clamped, out_of_range = clamp_unit(number)
        if out_of_range:
            analysis[key] = clamped
            flags.append(f"{flag_name}_out_of_range")
    return analysis, flags


def _apply_probability_forecast(
    analysis: dict[str, Any], symbol: str
) -> tuple[dict[str, Any], dict[str, float] | None]:
    """ML-29: trend i pewność z rozkładu P(BULLISH/BEARISH/SIDEWAYS) odpowiedzi.

    Reguła żyje w domenie (`ProbabilityForecast`): argmax to trend, max p to
    pewność; brak albo bezużyteczny rozkład → analiza bez zmian (stare pola).
    Zwraca `(analiza, rozkład_albo_None)`. Rozjazd z deklarowanym trendem LLM
    tylko logujemy — obowiązuje rozkład, bo to on trafia do punktacji."""
    forecast = ProbabilityForecast.from_llm(
        analysis.get("probabilities"),
        fallback_trend=TrendDirection(analysis["trend_direction"]),
        fallback_confidence=_float_or_default(analysis.get("confidence_score"), 0.5),
    )
    distribution = forecast.as_dict()
    if distribution is None:
        return analysis, None
    if forecast.trend.value != analysis["trend_direction"]:
        logger.info(
            "trend probabilities for %s point to %s, the declared trend was %s — "
            "the distribution wins",
            symbol,
            forecast.trend.value,
            analysis["trend_direction"],
        )
    derived = {
        **analysis,
        "trend_direction": forecast.trend.value,
        "confidence_score": forecast.confidence,
    }
    return derived, distribution


# ML-27: flagi jakości poziomu unieważnienia — poziom obecny, ale nieczytelny (nie
# liczba, brak ceny) albo po złej stronie ceny / dalej niż `INVALIDATION_BAND`.
INVALIDATION_INVALID_FLAG = "invalidation_invalid"
INVALIDATION_OUT_OF_BAND_FLAG = "invalidation_out_of_band"
# Powód trafia do maila — jedno zdanie, nie esej.
_INVALIDATION_REASON_MAX_LEN = 200


def _parse_invalidation(
    analysis: dict[str, Any], current_price: Decimal
) -> tuple[Invalidation | None, str | None]:
    """ML-27: poziom unieważnienia tezy z odpowiedzi LLM, po walidacji domeny.

    Zwraca `(poziom_albo_None, flaga_albo_None)`. Brak pola to nie skażone wejście
    (bez flagi); pole nieczytelne → `invalidation_invalid`, poziom po złej stronie
    ceny albo dalej niż 15% → `invalidation_out_of_band` (`is_valid_invalidation`).
    Trend z analizy jest już kanoniczny (ML-07, ML-29)."""
    raw = analysis.get("invalidation")
    if raw is None:
        return None, None
    price = _parse_number(raw.get("price")) if isinstance(raw, dict) else None
    if price is None or math.isinf(price):
        return None, INVALIDATION_INVALID_FLAG
    level = Decimal(str(price))
    trend = TrendDirection(analysis["trend_direction"])
    if not is_valid_invalidation(trend, current_price, level):
        return None, INVALIDATION_OUT_OF_BAND_FLAG
    reason = " ".join(str(raw.get("reason") or "").split())[:_INVALIDATION_REASON_MAX_LEN]
    return Invalidation(price=level, reason=reason), None


_TREND_TO_REC: dict[str, Recommendation] = {
    "BULLISH": "BUY",
    "BEARISH": "SELL",
    "SIDEWAYS": "HOLD",
}


def _vindicated_context(
    repository_port: RepositoryPort,
    last: Prediction,
    current_price: Decimal,
    symbol: str,
) -> str:
    """Buduje blok 'wybroniony dysydent' do promptu diagnozy (#8 dissent-replay).

    W pełni graceful: brak id / błąd odczytu / brak głosów / brak dysydenta →
    pusty string (diagnoza działa bez bloku). Zero płatnych wywołań."""
    if last.id is None:
        return ""
    try:
        votes = repository_port.get_council_votes_for_prediction(last.id)
    except Exception:
        logger.exception(
            "get_council_votes_for_prediction failed for %s — diagnoza bez dissent",
            symbol,
        )
        return ""
    actual_dir: Literal["UP", "DOWN", "FLAT"]
    if current_price > last.price_at_prediction:
        actual_dir = "UP"
    elif current_price < last.price_at_prediction:
        actual_dir = "DOWN"
    else:
        actual_dir = "FLAT"
    final_rec = _TREND_TO_REC.get(str(last.predicted_trend), "HOLD")
    vindicated = vindicated_dissenters(votes, actual_dir, final_rec)
    return ", ".join(f"{op.investor_name} ({op.recommendation})" for op in vindicated)


def _utc_now() -> datetime:
    """Domyślny zegar grafu (`AgentGraphDeps.clock`)."""
    return datetime.now(UTC)


def _build_fetch_fundamentals_node(
    port: FundamentalsPort,
) -> Callable[[AgentState], AgentState]:
    """Buduje węzeł grafu z domknięciem nad portem (DI-friendly i testowalny).

    Pozycja w grafie: po reflect_node, PRZED bramką volatility.
    Czytanie cache jest darmowe; płatne wywołania AV dzieją się tylko
    w slow loop (fast loop dostaje NullFundamentalsAdapter jako delegate).
    """

    def fetch_fundamentals_node(state: AgentState) -> AgentState:
        # Asset zawsze obecny w state gdy fundamentals_port != None
        # (main_agent.py klasyfikuje i przekazuje asset przed wywołaniem grafu).
        asset: Asset = state["asset"]
        if asset.asset_type is not AssetType.STOCK:
            return {
                "fundamentals": None,
                "valuation_verdict": ValuationVerdict.UNKNOWN,
            }

        try:
            fundamentals = port.get_fundamentals(asset.symbol)
        except Exception:
            return {
                "fundamentals": None,
                "valuation_verdict": ValuationVerdict.UNKNOWN,
            }

        verdict = asset.evaluate_valuation(fundamentals)
        return {"fundamentals": fundamentals, "valuation_verdict": verdict}

    return fetch_fundamentals_node


@dataclasses.dataclass(frozen=True)
class ShadowSetup:
    """ML-33: champion/challenger nogi ML w Fast Loopie, składany przez DI za flagą
    `shadow_predictions_enabled`.

    `champion_version` — sha256 pliku wag championa (jak `model_version` przepisu);
    `challengers` — do dwóch modeli z własnymi wagami. Ich prognozy cienia idą
    wyłącznie do `shadow_predictions` (migracja 075): wynik cyklu, prompt, mail
    i rekord predykcji są te same co bez nich."""

    champion_version: str
    challengers: tuple[ShadowModelPort, ...] = ()


@dataclasses.dataclass(frozen=True)
class PromptVariant:
    """EXP-26: opcje promptu predykcji, które wariant w cieniu nadpisuje względem
    produkcji. None = jak produkcja; `without_memory` = bez lekcji z reflect i bez
    analogów RAG (jak ramię EXP-03)."""

    prompt_delta_rounding: bool | None = None
    prompt_taint_fence: bool | None = None
    trend_probabilities: bool | None = None
    without_memory: bool = False


@dataclasses.dataclass(frozen=True)
class ShadowPromptSetup:
    """EXP-26: wariant promptu albo modelu LLM w cieniu, składany przez DI za flagą
    `shadow_prompt_variant_enabled`.

    `name` — nazwa wariantu (wiersz `shadow_predictions`), `version` — sha256 jego
    tożsamości, `llm_port` — model wariantu (bez kaskady i failovera), `variant` —
    nadpisane opcje promptu. Wołany tylko dla stałej podpróbki symboli
    (`sample_share`) za bramką volatility, najwyżej `max_calls_per_cycle` razy na
    cykl i tylko w luzie sufitu cyklu; wynik idzie wyłącznie do `shadow_predictions`."""

    name: str
    version: str
    llm_port: LLMPort
    variant: PromptVariant = PromptVariant()
    sample_share: float = 1.0
    max_calls_per_cycle: int = SHADOW_PROMPT_MAX_PER_CYCLE


@dataclasses.dataclass(frozen=True)
class AgentGraphDeps:
    """Zależności wstrzykiwane do węzłów grafu — zebrane w jeden obiekt, żeby
    węzły były funkcjami modułowymi (testowalnymi osobno), a fabryka
    create_agent_graph tylko je składała i okablowywała.

    Ten sam obiekt jest WEJŚCIEM `create_agent_graph` i `AnalyzeMarketUseCase`.
    Wcześniej obie brały te 23 zależności jako osobne kwargi i przepisywały je
    jedna do drugiej, więc każdy nowy port wymagał zgodnej zmiany w trzech
    sygnaturach. Pierwsze siedem pól jest wymagane; reszta to opcjonalne
    rozszerzenia Fast Loopa, których brak wyłącza odpowiedni węzeł albo bramkę.
    """

    market_port: MarketDataPort
    sentiment_port: SentimentPort
    news_port: NewsPort
    repository_port: RepositoryPort
    ml_port: MLPredictionPort
    llm_port: LLMPort
    threshold: Threshold
    embedding_port: EmbeddingPort | None = None
    council_port: AdvisoryCouncilPort | None = None
    council_threshold: Threshold | None = None
    fundamentals_port: FundamentalsPort | None = None
    crypto_threshold: Threshold | None = None
    crypto_council_threshold: Threshold | None = None
    reflection_min_age_hours: int = 0
    # WIA-04: górna granica wieku rozliczanej predykcji. None = bez granicy
    # (kwarg nie idzie do repozytorium — zachowanie sprzed WIA-04).
    reflection_max_age_hours: int | None = None
    rag_outcome_weight: float = 0.0
    tool_use_port: ToolUseLLMPort | None = None
    research_tools: Sequence[Tool] = ()
    tool_use_threshold: Threshold | None = None
    # FIN-08: osobny próg tool-use dla krypto (`crypto_tool_use_volatility_threshold`).
    # None = krypto bierze `tool_use_threshold`, jak przed FIN-08.
    crypto_tool_use_threshold: Threshold | None = None
    vector_memory_enabled: bool = False
    # ML-38: tag reżimu w `prediction_logs.regime` (migracja 017) przy KAŻDEJ
    # predykcji, niezależnie od filtra RAG. DI ustawia go jak `regime_aware_enabled`;
    # default False, żeby graf bez DI nie dopisywał klucza (PGRST204 przed 017).
    regime_tagging_enabled: bool = False
    receipts_enabled: bool = False
    # DAT-09: stempel `feature_set_version` na predykcji (migracja 027). Off =
    # klucza nie ma w rekordzie — przed migracją dałby PGRST204.
    feature_set_version_enabled: bool = False
    # WIA-20: przepis predykcji policzony RAZ na cykl w DI. Jest → kwit v2 (pola
    # przepisu w JSONB, za `receipts_enabled`); None → kwit v1 jak przed WIA-20.
    recipe: PredictionRecipe | None = None
    # WIA-20: stempel `recipe_version` (migracja 030). Off = klucza nie ma
    # w rekordzie — przed migracją dałby PGRST204.
    recipe_version_enabled: bool = False
    # ML-14: wektor cech w KAŻDYM cyklu do `feature_snapshots` (migracja 031).
    # Off = graf nie dotyka tabeli, której przed migracją nie ma.
    feature_snapshots_enabled: bool = False
    # ML-31: cechy rynkowe cyklu (stan `market_features`) w `feature_snapshots`
    # i wersja zestawu cech 4 w obu stemplach. Use case owija wtedy port rynku
    # w `CycleQuoteCache`. Off = zestaw i wersja 3 bez zmian.
    market_features_enabled: bool = False
    # ML-11: predict liczy σ symbolu (`return_sigma`, historia snapshotów ceny) i podaje
    # ją modelowi; model w jednostkach σ mnoży przez nią prognozę. Off = bez odczytu
    # historii i bez σ w wywołaniu predict, jak przed ML-11.
    ml_sigma_target_enabled: bool = False
    # ML-34: okna kalibracyjne przedziału ruchu per klasa aktywa (rozliczone
    # predykcje, rosnąco po czasie), czytane RAZ na cykl w DI. None = bez przedziału.
    move_calibration: Mapping[AssetType, Sequence[CalibrationPoint]] | None = None
    # ML-34: kolumny `interval_low/high` (migracja 032). Off = klucza nie ma
    # w rekordzie — przed migracją dałby PGRST204.
    interval_columns_enabled: bool = False
    # EXP-01: delta, która otworzyła bramkę, w `gate_delta` (migracja 034). Off =
    # klucza nie ma w rekordzie — przed migracją dałby PGRST204.
    gate_delta_enabled: bool = False
    # ALR-10: bramka widzi dryf od ostatniej predykcji (darmowe odczyty referencji
    # w use case). Off = stan bez `drift_reference`, bramka jak przed ALR-10.
    drift_gate_enabled: bool = False
    # EXP-05: definicje eksperymentów; use case liczy z nich ramiona per
    # symbol-dzień i podaje je w stanie (`experiment_arms`).
    experiments: tuple[Experiment, ...] = ()
    # EXP-05: kolumna `experiment_arms` (migracja 037). Off = klucza nie ma
    # w rekordzie — przed migracją dałby PGRST204.
    experiment_arms_enabled: bool = False
    # ML-29: prompt prosi o rozkład trzech klas, trend = argmax, pewność = max p,
    # a rozkład idzie do kolumny `trend_probabilities` (migracja 051). Off = prompt,
    # analiza i rekord jak przed ML-29 — przed migracją klucz dałby PGRST204.
    trend_probabilities_enabled: bool = False
    # BEZ-14: lekcja z reflect w prompcie predykcji w fence'u DATA-ONLY. Off = prompt
    # bajt w bajt jak przed BEZ-14.
    prompt_taint_fence_enabled: bool = False
    options_port: OptionsFlowPort | None = None
    # WIA-28: horyzont edge z kalendarza NYSE (godziny do następnego dnia handlowego)
    # dla akcji i ETF zamiast stałych 24 h. Off = stałe 24 h, jak przed WIA-28.
    edge_calendar_horizon_enabled: bool = False
    # DAT-14: notowanie z czasem (Finnhub /quote) dla akcji i ETF zamiast gołej ceny
    # z `market_port` — ten sam jeden request. None = cena bez czasu, jak przed DAT-14.
    quote_port: QuotePort | None = None
    # DAT-14: kolumna `stale_quote` (migracja 055) w predykcji i rozliczeniu. Off =
    # klucza nie ma w rekordzie — przed migracją dałby PGRST204.
    stale_quote_enabled: bool = False
    # WIA-61: rozliczenie zapisuje obok ceny /quote oficjalne zamknięcie sesji
    # (kolumny z migracji 059). Off = klucze nie idą do repozytorium (PGRST204).
    session_close_settlement_enabled: bool = False
    # WIA-54: rozliczenie zapisuje obok ceny /quote cenę wykonalną — otwarcie pierwszej
    # sesji po predykcji z tego samego notowania (kolumna `entry_price`, migracja 072).
    # Off = klucz nie idzie do repozytorium (PGRST204).
    executable_entry_enabled: bool = False
    # WIA-59: append-only paper ledger (`paper_trades`, migracja 073) — save otwiera
    # pozycję, reflect ją zamyka; koszty obrotu jak w paper-PnL (WIA-50). None = graf
    # nie dotyka tabeli, której przed migracją nie ma.
    paper_ledger_costs: LedgerCosts | None = None
    # ML-27: prompt prosi o poziom unieważnienia tezy, normalizacja go waliduje
    # (strona ceny, ±15%), a zapis niesie `invalidation_price` (migracja 074). Off =
    # prompt, analiza i rekord jak przed ML-27 — przed migracją klucz dałby PGRST204.
    invalidation_level_enabled: bool = False
    # ML-36: ruch ceny w prompcie predykcji z dwoma miejscami i znakiem („+3.46”) zamiast
    # pełnej precyzji `Decimal`. Off = tekst promptu jak przed ML-36.
    prompt_delta_rounding_enabled: bool = False
    # ML-41: analogi RAG tylko rozliczone (z wynikiem), z większej puli kandydatów.
    # Off = retrieval i analogi w prompcie jak przed ML-41.
    rag_point_in_time_enabled: bool = False
    # DAT-17: świeży 8-K ze stanu (`fresh_corporate_events`) w prompcie predykcji
    # i cecha `fresh_8k` w `feature_snapshots`. Off = prompt i wektor jak przed DAT-17.
    corporate_events_enabled: bool = False
    # DAT-23: `pc` notowania vs ostatni snapshot — podejrzenie splitu koryguje
    # referencję bramki, rozliczenie i `price_delta`, a zdarzenie idzie do
    # `corporate_actions` (migracja 062). Off = jak przed DAT-23.
    split_detection_enabled: bool = False
    # DAT-15: odczyt anomalny (MAD względem historii snapshotów, bez potwierdzenia
    # w `pc`) nie otwiera bramki, nie rozlicza i nie nadpisuje referencji. Off = bez
    # odczytu historii, jak przed DAT-15.
    price_outlier_guard_enabled: bool = False
    # EXP-17: reflect nie płaci za diagnozę błędu, tylko zostawia placeholder dla
    # tygodniowej diagnozy zbiorczej Slow Loopa. Off = diagnoza za bramką, jak dotąd.
    batch_diagnosis_enabled: bool = False
    attestation_publisher: AttestationPublisherPort | None = None
    # ML-33: prognozy cienia championa i challengerów do `shadow_predictions`
    # (migracja 075). None = flaga off: graf nie liczy cieni i nie dotyka tabeli.
    shadow: ShadowSetup | None = None
    # EXP-26: wariant promptu albo modelu LLM w cieniu (kolumny z migracji 083). None =
    # flaga off: graf nie woła wariantu i nie dotyka jego wierszy.
    shadow_prompt: ShadowPromptSetup | None = None
    # ARCH-11: metryki CYKLU (płatne wywołania, czasy węzłów, decyzje bramki),
    # współdzielone między symbolami. Własny egzemplarz per deps, gdy wołający
    # nie poda go z DI — `use_case.paid_calls` działa i wtedy.
    telemetry: CycleTelemetry = dataclasses.field(default_factory=CycleTelemetry)
    # Zgodność wstecz: dawny goły licznik płatnych wywołań. To NIE drugi licznik —
    # po `__post_init__` jest magazynem `telemetry.paid_calls` (podany Counter
    # zostaje przejęty przez telemetrię, brak = Counter telemetrii).
    paid_call_meter: Counter[str] | None = None
    # FIN-12: sufit płatnych wywołań LLM cyklu (`PAID_LLM_SOURCES`). None = bez
    # limitu. Decyzję podejmuje domena (`paid_budget_allows`) po bramce volatility.
    paid_call_cap: int | None = None
    # FIN-25: limit USD doby albo miesiąca z ledgera `llm_calls` wyczerpany przed
    # cyklem (odczyt raz, w entry poincie). True = cykl nie płaci za nic: symbol za
    # bramką idzie ścieżką `budget_exhausted` (bez snapshotu), diagnoza w reflect
    # jest pominięta. Domyślnie False — zachowanie jak przed FIN-25.
    llm_budget_spent: bool = False
    # ML-18: chwila predykcji. Referencja `price_delta` pomija wiersz z jej godziny
    # UTC, który zapis tej predykcji nadpisze upsertem. Testy wstrzykują stały zegar.
    clock: Callable[[], datetime] = _utc_now

    def __post_init__(self) -> None:
        missing_tool_use = [
            name
            for name, present in (
                ("research_tools", bool(self.research_tools)),
                ("tool_use_threshold", self.tool_use_threshold is not None),
            )
            if not present
        ]
        if self.tool_use_port is not None and missing_tool_use:
            # ARCH-18: niepełna trójka cicho spadała do zwykłego `llm_port.analyze`.
            raise ValueError(
                f"tool-use: tool_use_port requires {' and '.join(missing_tool_use)}"
            )
        if self.paid_call_meter is None:
            object.__setattr__(self, "paid_call_meter", self.telemetry.paid_calls)
        else:
            self.telemetry.bind_paid_calls(self.paid_call_meter)


# DAT-14: klasy aktywów notowane w sesjach NYSE — krypto handluje 24/7 (CoinGecko).
_SESSION_QUOTED_ASSETS = frozenset({AssetType.STOCK, AssetType.ETF})


def _session_quote_port(deps: AgentGraphDeps, state: AgentState) -> QuotePort | None:
    """DAT-14: port notowania z czasem, gdy symbol jest akcją albo ETF-em.
    Symbol bez klasy aktywa (graf bez use case'u) zostaje przy `market_port`."""
    asset = state.get("asset")
    if deps.quote_port is None or asset is None:
        return None
    return deps.quote_port if asset.asset_type in _SESSION_QUOTED_ASSETS else None


def _detect_split(
    deps: AgentGraphDeps, state: AgentState, quote: Quote | None
) -> SplitSuspicion | None:
    """DAT-23: podejrzenie splitu z `pc` notowania vs ostatni snapshot. Tylko za flagą,
    z notowaniem niosącym `pc` i ze świeżą referencją — stęchły snapshot (FIN-14)
    nie jest ceną sprzed jednej sesji, więc stosunek nic by nie mówił. Zdarzenie
    idzie do `corporate_actions`; błąd zapisu nie zatrzymuje korekty cyklu."""
    if not deps.split_detection_enabled or quote is None or quote.prev_close is None:
        return None
    if _reference_is_stale(state):
        return None
    suspicion = detect_split(
        state["symbol"], quote.prev_close.amount, state["previous_price"], at=deps.clock()
    )
    if suspicion is None:
        return None
    logger.warning(
        "Suspected split for %s: previous close %s vs last snapshot %s → ratio %s; "
        "the gate, the settlement and price_delta use split-adjusted prices",
        suspicion.symbol,
        suspicion.prev_close,
        suspicion.reference_price,
        suspicion.ratio,
    )
    try:
        deps.repository_port.save_corporate_action(suspicion)
    except PortError:
        logger.warning(
            "save_corporate_action failed for %s — split adjusted without a record",
            suspicion.symbol,
            exc_info=True,
        )
    return suspicion


def _is_outlier_reading(
    deps: AgentGraphDeps,
    state: AgentState,
    quote: Quote | None,
    current: Money,
    split: SplitSuspicion | None,
) -> bool:
    """DAT-15: odczyt anomalny bez potwierdzenia w `pc` (domena `price_validation`).

    Tylko za flagą i ze świeżą referencją: po pauzie (FIN-14) cykl i tak tylko
    zapisuje nowy punkt odniesienia, a strażnik nie może go blokować — inaczej trwały
    ruch nie dostałby nigdy referencji. Historia snapshotów jest w cenach sprzed
    splitu (DAT-23), więc odczyt i `pc` cofamy o jego stosunek. Błąd odczytu
    historii przepuszcza odczyt (fail-open, jak przed DAT-15)."""
    if not deps.price_outlier_guard_enabled or _reference_is_stale(state):
        return False
    symbol = state["symbol"]
    try:
        history = deps.repository_port.get_price_history(symbol, OUTLIER_HISTORY_DAYS)
    except PortError:
        logger.warning(
            "DAT-15: price history of %s unavailable — reading not validated",
            symbol,
            exc_info=True,
        )
        return False
    ratio = split.ratio if split is not None else Decimal(1)
    prev_close = quote.prev_close if quote is not None else None
    outlier = is_unconfirmed_outlier(
        to_pre_split_price(current.amount, ratio),
        [price.amount for _, price in history],
        prev_close=None if prev_close is None else to_pre_split_price(prev_close.amount, ratio),
    )
    if outlier:
        logger.warning(
            "Price reading %s for %s is an unconfirmed outlier against %d snapshots — "
            "gate closed, no settlement, reference kept for the next reading",
            current.amount,
            symbol,
            len(history),
        )
    return outlier


def _check_price_node(deps: AgentGraphDeps, state: AgentState) -> AgentState:
    symbol = state["symbol"]
    quote_port = _session_quote_port(deps, state)
    quote = quote_port.get_quote(symbol) if quote_port is not None else None
    current = quote.price if quote is not None else deps.market_port.get_current_price(symbol)
    split = _detect_split(deps, state, quote)
    # DAT-23: po splicie referencja (snapshot sprzed niego) w cenach po splicie.
    ratio = split.ratio if split is not None else Decimal(1)
    previous = Money(state["previous_price"] * ratio)
    delta = PriceDelta.calculate(previous, current)
    # #6: snapshot ceny NIE jest już zapisywany tutaj. Zapis przeniesiony do
    # węzłów TERMINALNYCH (save_node / ignore_node), żeby cykl był idempotentny:
    # cykl, który padł w połowie, nie zostawia snapshotu, więc retry reużyje
    # referencji z poprzedniego UKOŃCZONEGO cyklu (zamiast porównywać się do
    # świeżego snapshotu padłej próby → delta≈0 → cichy "ignore").
    update: AgentState = {
        "current_price": current.amount,
        "delta": Ratio(delta.fraction),
    }
    if split is not None:
        update["previous_price"] = previous.amount
        update["split_ratio"] = split.ratio
    if _is_outlier_reading(deps, state, quote, current, split):
        update["price_outlier"] = True
    if quote is not None:
        now = deps.clock()
        stale = is_stale(quote.quoted_at, now)
        if stale:
            logger.warning(
                "Quote for %s is from %s, before the last started NYSE session",
                symbol,
                quote.quoted_at.isoformat(),
            )
        update["quoted_at"] = quote.quoted_at
        update["stale_quote"] = stale
        if deps.session_close_settlement_enabled:
            update["session_close"] = session_close_of(quote, now)
        if deps.executable_entry_enabled:
            update["session_quote"] = quote
    reference = state.get("drift_reference")
    if reference is not None and reference > 0:
        update["drift"] = Ratio(
            PriceDelta.calculate(Money(reference * ratio), current).fraction
        )
    return update



def _persist_price_snapshot(deps: AgentGraphDeps, state: AgentState) -> None:
    """#6/#15: zapis snapshotu ceny — wołany wyłącznie z węzłów terminalnych.

    Cold-start nadal działa, bo KAŻDY ukończony cykl (saved albo ignored)
    zapisuje snapshot, więc następny cykl ma punkt odniesienia. Zapis jest
    owinięty w try/except (jak save_council_votes / embedding), żeby
    przejściowy błąd Supabase nie wywalił całego grafu.
    """
    repository_port = deps.repository_port
    try:
        repository_port.save_price_snapshot(
            state["symbol"], Money(state["current_price"])
        )
    except Exception:
        logger.exception(
            "save_price_snapshot failed for %s — cykl kończy się bez "
            "zapisu referencji ceny; następny cykl użyje starszego snapshotu",
            state["symbol"],
        )


def _free_snapshot_features(deps: AgentGraphDeps, state: AgentState) -> dict[str, Any]:
    """ML-14 — cechy DARMOWE (`FREE_SNAPSHOT_FEATURES`) wyłącznie ze stanu cyklu.

    Zero wywołań portów: wszystko policzyły już check_price, bramka i DI przed
    grafem (reżim, mnożniki, Alpha Fusion). Brak źródła → None, nie 0.0 —
    cisza nie może udawać neutralnego rynku w raporcie selekcji. ML-31: z flagą
    `market_features_enabled` dochodzą `MARKET_SNAPSHOT_FEATURES` ze stanu."""
    asset = state.get("asset") or Asset(symbol=state["symbol"])
    delta = state["delta"]
    fusion = state.get("alpha_fusion")
    verdict = state.get("valuation_verdict")
    threshold = _pick_threshold(state, deps.threshold, deps.crypto_threshold)
    raw: dict[str, Any] = {
        "price": float(state["current_price"]),
        "previous_price": float(state["previous_price"]),
        "snapshot_delta": float(delta),
        "abs_snapshot_delta": abs(float(delta)),
        "effective_threshold": float(threshold.value),
        "regime": canonical_regime_key(state.get("regime_context")),
        "regime_multiplier": float(state.get("regime_multiplier", 1.0) or 1.0),
        "earnings_multiplier": float(state.get("earnings_multiplier", 1.0) or 1.0),
        "alpha_fusion_score": fusion.score if fusion is not None else None,
        "asset_type": asset.asset_type.value,
        "valuation_verdict": verdict.value if verdict is not None else None,
    }
    free = {name: raw[name] for name in FREE_SNAPSHOT_FEATURES}
    if deps.market_features_enabled:
        market = state.get("market_features") or {}
        free.update({name: market.get(name) for name in MARKET_SNAPSHOT_FEATURES})
    if deps.corporate_events_enabled:
        free.update(_corporate_event_features(state))
    return free


def _corporate_event_features(state: AgentState) -> dict[str, Any]:
    """DAT-17: `fresh_8k` = 1.0 przy świeżym 8-K, 0.0 gdy wiadomo, że go nie ma, None,
    gdy nie wiadomo (brak klucza w stanie) — cisza nie udaje braku zdarzenia."""
    events = state.get("fresh_corporate_events")
    fresh = None if events is None else float(bool(events))
    return dict.fromkeys(CORPORATE_EVENT_SNAPSHOT_FEATURES, fresh)


def _gated_snapshot_features(state: AgentState) -> dict[str, Any]:
    """ML-14 — cechy LLM cyklu za bramką: wektor podany modelowi, deklarowana
    pewność i flagi jakości (trening odsieje nimi np. atrapę po awarii LLM)."""
    llm_analysis = state.get("llm_analysis") or {}
    extras: dict[str, Any] = {
        "confidence_score": _float_or_default(llm_analysis.get("confidence_score"), 0.5),
        "data_quality_flags": list(state.get("data_quality_flags") or []),
    }
    return {
        **state.get("ml_features", {}),
        **{name: extras[name] for name in GATED_SNAPSHOT_EXTRAS},
    }


def _persist_feature_snapshot(
    deps: AgentGraphDeps, state: AgentState, *, gated: bool
) -> None:
    """ML-14 — wektor cech point-in-time, wołany z węzłów terminalnych obok
    snapshotu ceny (także z ignore, więc trening widzi też dni bez ruchu).

    Etykiety nie zapisujemy — zwrot dołącza się później z `price_snapshots`.
    Błąd zapisu nie wywala cyklu (jak `_persist_price_snapshot`)."""
    if not deps.feature_snapshots_enabled:
        return
    features = _free_snapshot_features(deps, state)
    if gated:
        features.update(_gated_snapshot_features(state))
    try:
        deps.repository_port.save_feature_snapshot(
            state["symbol"],
            features,
            gated=gated,
            feature_set_version=feature_set_version(deps.market_features_enabled),
        )
    except Exception:
        logger.exception(
            "save_feature_snapshot failed for %s — cykl kończy się bez wektora cech",
            state["symbol"],
        )


def _build_decision_receipts(deps: AgentGraphDeps, state: AgentState) -> dict[str, Any]:
    """#13 — składa kwit decyzyjny z artefaktów JUŻ policzonych w tym cyklu.

    Zero nowej logiki i zero płatnych wywołań: persystujemy to, za co już
    zapłaciliśmy. `schema_version` od pierwszego commita — kwity są czytane
    miesiącami po zapisie, a JSONB nie ma migracji kolumn.

    MVP celowo bez odznak proweniencji: `AgentState` ich nie niesie (liczy je
    dopiero `to_symbol_result` z innych wejść), a dokładanie ich tutaj byłoby
    duplikacją logiki. WIA-66 dołoży je w v2 jako opcjonalny klucz.

    WIA-20: z przepisem w deps kwit ma `schema_version: 2` — wszystkie klucze v1
    plus płaskie pola przepisu i `recipe_version` (czytnik: `decision_receipt`).

    Wszystkie wartości muszą być JSON-serializowalne: `_serialize` w adapterze
    konwertuje Decimal TYLKO na najwyższym poziomie, więc zagnieżdżony Decimal
    wywaliłby zapis.
    """
    threshold = _pick_threshold(state, deps.threshold, deps.crypto_threshold)
    attribution: tuple[FeatureContribution, ...] = state.get("feature_attribution", ())
    precedents: list[dict[str, Any]] = list(state.get("similar_precedents") or [])
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_V1,
        # Próg EFEKTYWNY (po mnożnikach reżimu i earnings), nie surowy z configu
        # — inaczej post-mortem nie wyjaśni, czemu symbol przeszedł bramkę.
        "effective_threshold": str(threshold.value),
        "precedents": precedents,
        "feature_attribution": [
            {"feature": c.feature, "contribution": float(c.contribution)}
            for c in attribution
        ],
    }
    recipe = _effective_recipe(deps, state)
    if recipe is not None:
        receipt["schema_version"] = RECEIPT_SCHEMA_V2
        receipt.update(recipe.receipt_fields())
        # EXP-05: opcjonalny klucz v2 — tylko gdy jakiś eksperyment jest aktywny.
        arms = state.get("experiment_arms")
        if arms:
            receipt["experiment_arms"] = dict(arms)
        # FIN-21: opcjonalny klucz v2 — tylko gdy odpowiedź przyszła z routingu modeli.
        model_used = state.get("llm_model_used")
        if model_used:
            receipt["model_used"] = model_used
    return receipt


def _pick_threshold(
    state: AgentState,
    default: Threshold,
    crypto_override: Threshold | None,
) -> Threshold:
    """Efektywny próg bramki dla symbolu ze stanu. Regułę (próg klasy aktywa
    i mnożniki reżimu #7 oraz earnings #6) liczy domena (`effective_threshold`,
    FIN-08); graf podaje tylko klasę aktywa i mnożniki cyklu."""
    asset = state.get("asset")
    return effective_threshold(
        default,
        asset.asset_type if asset is not None else None,
        crypto_override,
        regime_multiplier=state.get("regime_multiplier", 1.0) or 1.0,
        earnings_multiplier=state.get("earnings_multiplier", 1.0) or 1.0,
    )



def _reference_is_stale(state: AgentState) -> bool:
    """FIN-14: werdykt domeny policzony przez use case przy odczycie referencji.
    Brak klucza (graf wołany bez use case'u) = referencja świeża, jak przed FIN-14."""
    return bool(state.get("stale_reference", False))


def _drift_of(state: AgentState) -> PriceDelta | None:
    """ALR-10: dryf od referencji policzony w check_price; None = bez dryfu."""
    drift = state.get("drift")
    return PriceDelta(drift) if drift is not None else None


def _gate_reason(passes: bool, stale: bool, *, by_drift: bool = False) -> str:
    """ARCH-11: powód decyzji bramki do telemetrii cyklu. Stęchła referencja
    zamyka bramkę niezależnie od ruchu (FIN-14), więc ma własny powód; bramka
    otwarta samym dryfem (ALR-10) też."""
    if passes:
        return "drift_above_threshold" if by_drift else "above_threshold"
    return "stale_reference" if stale else "below_threshold"


def _budget_exhausted(deps: AgentGraphDeps, symbol: str) -> bool:
    """FIN-12: czy cykl wyczerpał sufit płatnych wywołań LLM. Licznik jest wspólny
    dla wszystkich symboli cyklu (także równoległych), więc czytamy go pod lockiem
    telemetrii; reguła miękkiego sufitu żyje w domenie. FIN-25: limit USD doby albo
    miesiąca wyczerpany przed cyklem zamyka tę samą ścieżkę dla każdego symbolu."""
    if deps.llm_budget_spent:
        logger.warning(
            "LLM budget of the day or month spent before %s — symbol skipped without "
            "analysis, price reference kept for the next cycle",
            symbol,
        )
        return True
    spent = deps.telemetry.paid_calls_of(PAID_LLM_SOURCES)
    if paid_budget_allows(spent, deps.paid_call_cap):
        return False
    logger.warning(
        "paid LLM budget exhausted before %s: %d/%d calls spent this cycle — "
        "symbol skipped without analysis, price reference kept for the next cycle",
        symbol,
        spent,
        deps.paid_call_cap,
    )
    return True


@dataclasses.dataclass(frozen=True)
class _GateVerdict:
    """Werdykt domenowej bramki volatility dla symbolu (bez sufitu budżetu)."""

    passes: bool
    stale: bool
    threshold: Threshold
    asset: Asset
    # ALR-10: bramkę otworzył sam dryf — ruch cyklu był pod progiem.
    by_drift: bool = False


def _gate_verdict(
    state: AgentState,
    default: Threshold,
    crypto_override: Threshold | None,
    *,
    reference_stale: bool = False,
    drift: PriceDelta | None = None,
) -> _GateVerdict:
    """FIN-08: jeden helper każdej bramki volatility (główna, diagnoza reflect,
    tool-use, rada) — efektywny próg z domeny i werdykt `Asset.evaluate_volatility`."""
    asset = state.get("asset") or Asset(symbol=state["symbol"])
    chosen = _pick_threshold(state, default, crypto_override)
    delta = PriceDelta(state["delta"])
    passes = asset.evaluate_volatility(
        delta, chosen, reference_stale=reference_stale, drift=drift
    )
    # ALR-10: bramkę otworzył sam dryf, gdy bez niego ruch cyklu byłby pod progiem.
    by_drift = (
        drift is not None
        and passes
        and not asset.evaluate_volatility(delta, chosen, reference_stale=reference_stale)
    )
    return _GateVerdict(
        passes=passes, stale=reference_stale, threshold=chosen, asset=asset, by_drift=by_drift
    )


def _volatility_verdict(deps: AgentGraphDeps, state: AgentState) -> _GateVerdict:
    return _gate_verdict(
        state,
        deps.threshold,
        deps.crypto_threshold,
        reference_stale=_reference_is_stale(state),
        drift=_drift_of(state),
    )


def _log_gate_decision(state: AgentState, verdict: _GateVerdict, decision: str) -> None:
    # #13: decyzja bramki volatility musi być widoczna w logach — bez tego
    # nie dało się odróżnić zamierzonego "ignore" (FinOps) od cichego buga.
    drift = state.get("drift")
    logger.info(
        "volatility gate for %s: |Δ|=%.4f threshold=%.4f asset_type=%s "
        "stale_reference=%s drift=%s → %s",
        state["symbol"],
        abs(float(state["delta"])),
        float(verdict.threshold.value),
        verdict.asset.asset_type.value,
        verdict.stale,
        "-" if drift is None else f"{float(drift):.4f}",
        decision,
    )


# DAT-15: powód bramki dla odczytu anomalnego (telemetria cyklu).
PRICE_OUTLIER_REASON = "price_outlier"


def _closed_by_outlier(deps: AgentGraphDeps, state: AgentState) -> bool:
    """DAT-15: odczyt anomalny zamyka bramkę niezależnie od ruchu — delta od złego
    ticka nie może płacić za analizę."""
    if not state.get("price_outlier"):
        return False
    deps.telemetry.record_gate(state["symbol"], False, PRICE_OUTLIER_REASON)
    logger.info("volatility gate for %s: price outlier → ignore", state["symbol"])
    return True


def _should_analyze(deps: AgentGraphDeps, state: AgentState) -> str:
    if _closed_by_outlier(deps, state):
        return "ignore"
    verdict = _volatility_verdict(deps, state)
    passes = verdict.passes
    over_budget = passes and _budget_exhausted(deps, state["symbol"])
    if over_budget:
        decision = "budget_exhausted"
        deps.telemetry.record_gate(state["symbol"], False, "budget_exhausted")
    else:
        decision = "analyze" if passes else "ignore"
        deps.telemetry.record_gate(
            state["symbol"],
            passes,
            _gate_reason(passes, verdict.stale, by_drift=verdict.by_drift),
        )
    _log_gate_decision(state, verdict, decision)
    if over_budget:
        return "budget_exhausted"
    return "analyze_sentiment" if passes else "ignore"


# EXP-27: wynik bramki fazy 1 dla symbolu, który czeka na fazę płatną.
GATE_PASSED = "paid_phase"


def _gate_phase_route(deps: AgentGraphDeps, state: AgentState) -> str:
    """EXP-27 faza 1: sama bramka domenowa, bez sufitu budżetu. Symbol pod progiem
    kończy w `ignore` jak dziś; symbol za bramką kończy fazę 1 bez terminala
    (bez snapshotu) — o analizie albo cięciu budżetem decyduje faza 2, więc
    dopiero ona zapisuje decyzję w telemetrii (jedna decyzja na symbol)."""
    if _closed_by_outlier(deps, state):
        return "ignore"
    verdict = _volatility_verdict(deps, state)
    if not verdict.passes:
        deps.telemetry.record_gate(state["symbol"], False, _gate_reason(False, verdict.stale))
        _log_gate_decision(state, verdict, "ignore")
        return "ignore"
    _log_gate_decision(state, verdict, GATE_PASSED)
    return GATE_PASSED


def _paid_phase_route(deps: AgentGraphDeps, state: AgentState) -> str:
    """EXP-27 faza 2: sufit płatnych wywołań LLM cyklu (FIN-12) sprawdzany przed
    płatną częścią symbolu. DI podaje symbole malejąco po |Δ|, więc sufit odcina
    najmniejsze ruchy, a nie ostatnie w kolejności z config.toml."""
    symbol = state["symbol"]
    if _budget_exhausted(deps, symbol):
        deps.telemetry.record_gate(symbol, False, "budget_exhausted")
        return "budget_exhausted"
    by_drift = _volatility_verdict(deps, state).by_drift
    deps.telemetry.record_gate(symbol, True, _gate_reason(True, stale=False, by_drift=by_drift))
    return "analyze_sentiment"



def _settles_on_stale_quote(state: AgentState, last: Prediction) -> bool:
    """DAT-14: notowanie cyklu nie pochodzi z sesji po predykcji (weekend, święto,
    drugi przebieg przed otwarciem). Bez czasu notowania albo predykcji — False."""
    quoted_at = state.get("quoted_at")
    if quoted_at is None or last.created_at is None:
        return False
    return settles_on_stale_quote(quoted_at, last.created_at)


def _settle_on_stale_quote(
    deps: AgentGraphDeps,
    last: Prediction,
    current_price: Decimal,
    accuracy: float,
    trend_correct: bool,
) -> AgentState:
    """DAT-14: rozliczenie ceną równą cenie wejścia nie mówi nic o trafności —
    za flagą wiersz dostaje `stale_quote` i wypada z trafności, a płatna diagnoza
    LLM nie leci (nie ma prawdziwego błędu do wyjaśnienia)."""
    assert last.id is not None, "Prediction from repository must have id"
    logger.warning(
        "Settling %s prediction %s on a quote from before the next session — "
        "excluded from accuracy%s",
        last.symbol,
        last.id,
        "" if deps.stale_quote_enabled else " once stale_quote_enabled is on",
    )
    deps.repository_port.update_prediction_accuracy(
        prediction_id=last.id,
        actual_price=current_price,
        accuracy_score=accuracy,
        is_trend_correct=trend_correct,
        insight=STALE_QUOTE_INSIGHT,
        stale_quote=True if deps.stale_quote_enabled else None,
    )
    return {
        "reflection_context": "Ostatnie rozliczenie na nieaktualnym notowaniu — bez oceny."
    }


class _SettlementExtras(TypedDict, total=False):
    """WIA-61, WIA-54: opcjonalne pola rozliczenia — klucz tylko, gdy jest wartość."""

    session_close: SessionCloseSettlement
    entry_price: Decimal


def _session_close_extras(
    deps: AgentGraphDeps, state: AgentState, last: Prediction, tolerance: Decimal
) -> _SettlementExtras:
    """WIA-61 (okres przejściowy): zamknięcie sesji z trafnością na nim, gdy
    pochodzi z pierwszej sesji po predykcji albo późniejszej. Cena /quote zostaje
    rozliczeniem — trafność raportu, diagnoza i trening bez zmian. DAT-23: po
    splicie zamknięcie też w jednostkach predykcji."""
    close = state.get("session_close")
    if not deps.session_close_settlement_enabled or close is None or last.created_at is None:
        return {}
    if not close.settles(last.created_at):
        return {}
    close = dataclasses.replace(close, price=Money(_settlement_price(state, close.price.amount)))
    correct = last.is_trend_correct(close.price.amount, tolerance=tolerance)
    return {"session_close": SessionCloseSettlement(close, is_trend_correct=correct)}


def _executable_entry_extras(
    deps: AgentGraphDeps, state: AgentState, last: Prediction
) -> _SettlementExtras:
    """WIA-54: cena wykonalna rozliczanej predykcji z notowania cyklu — otwarcie
    pierwszej sesji po niej (domena `Prediction.executable_entry`). Brak (zaległe
    rozliczenie, przed sesją, bez `o`) = bez klucza, nigdy zero. DAT-23: otwarcie
    po splicie cofnięte do jednostek predykcji, jak cena rozliczenia."""
    quote = state.get("session_quote")
    if not deps.executable_entry_enabled or quote is None:
        return {}
    if quote.open_price is not None:
        open_price = Money(_settlement_price(state, quote.open_price.amount))
        quote = dataclasses.replace(quote, open_price=open_price)
    entry = last.executable_entry(quote, now=deps.clock())
    return {} if entry is None else {"entry_price": entry}


def _settlement_price(state: AgentState, price: Decimal) -> Decimal:
    """DAT-23: cena rozliczenia w jednostkach predykcji. Po podejrzanym splicie
    predykcja sprzed niego rozlicza się ceną cofniętą o stosunek splitu — to samo,
    co przeskalowanie `price_at_prediction` i celu, a zapisana cena zostaje
    w jednostkach `price_at_prediction`. Bez splitu cena bez zmian."""
    ratio = state.get("split_ratio")
    return price if ratio is None else to_pre_split_price(price, ratio)


def _close_paper_trade(
    deps: AgentGraphDeps, last: Prediction, exit_price: Decimal, reason: ExitReason
) -> None:
    """WIA-59: zamyka pozycję papierową rozliczonej predykcji (za `paper_ledger_costs`).

    Po rozliczeniu, więc ledger nigdy nie wyprzedza `prediction_logs`. Błąd zapisu
    tylko loguje — rozliczenie już jest, a pozycja zostaje otwarta w ledgerze."""
    if deps.paper_ledger_costs is None:
        return
    assert last.id is not None, "Prediction from repository must have id"
    side = side_of(str(last.predicted_trend))
    close = close_trade(side, last.price_at_prediction, exit_price, deps.clock(), reason)
    try:
        deps.repository_port.close_paper_trade(last.id, close)
    except PortError:
        logger.warning(
            "close_paper_trade failed for %s %s — settlement kept, the paper trade stays open",
            last.symbol,
            last.id,
            exc_info=True,
        )


def _reflect_node(deps: AgentGraphDeps, state: AgentState) -> AgentState:
    repository_port = deps.repository_port
    reflection_min_age_hours = deps.reflection_min_age_hours
    threshold = deps.threshold
    crypto_threshold = deps.crypto_threshold
    llm_port = deps.llm_port
    symbol = state["symbol"]
    if state.get("price_outlier"):
        # DAT-15: anomalny odczyt nie rozlicza — predykcja czeka na następny odczyt.
        return {"reflection_context": "Odczyt ceny anomalny — rozliczenie odłożone."}
    current_price = _settlement_price(state, state["current_price"])
    # min_age_hours odsiewa predykcje zbyt świeże, by je rzetelnie ocenić
    # (nakładające się cykle: manualny dispatch + scheduled). 0 = bez filtra.
    # max_age_hours (WIA-04) odsiewa predykcje zbyt stare — idzie tylko, gdy ustawione.
    if deps.reflection_max_age_hours is None:
        last = repository_port.get_unverified_prediction(
            symbol, min_age_hours=reflection_min_age_hours
        )
    else:
        last = repository_port.get_unverified_prediction(
            symbol,
            min_age_hours=reflection_min_age_hours,
            max_age_hours=deps.reflection_max_age_hours,
        )

    if last is None:
        return {"reflection_context": "Brak danych historycznych do oceny."}

    # Predykcje odczytane z repozytorium muszą mieć id (kontrakt RepositoryPort).
    assert last.id is not None, "Prediction from repository must have id"

    # Domena liczy accuracy_score (0.0-1.0) — bliskość ceny do celu; czyta go
    # recap tygodnia, a trening XGBoost uczy się na `target_return` z
    # ml_feature_store. is_trend_correct (zgodność kierunku) jest niezależną
    # miarą i liczy trafność raportu.
    accuracy = float(last.accuracy_score(current_price))
    # Pasmo SIDEWAYS zależy od klasy aktywa: ±0.5% to sensowny "brak ruchu" dla
    # akcji, ale przy dziennej zmienności BTC 3–5% uznawałoby niemal każdy dzień
    # za ruch i systematycznie zaniżało trafność krypto. Korekta POMIARU.
    asset = state.get("asset")
    tolerance = sideways_tolerance_for(asset.asset_type if asset is not None else None)
    trend_correct = last.is_trend_correct(current_price, tolerance=tolerance)

    if _settles_on_stale_quote(state, last):
        settled = _settle_on_stale_quote(deps, last, current_price, accuracy, trend_correct)
        _close_paper_trade(deps, last, current_price, ExitReason.STALE_QUOTE)
        return settled
    extras: _SettlementExtras = {
        **_session_close_extras(deps, state, last, tolerance),
        **_executable_entry_extras(deps, state, last),
    }

    if trend_correct:
        repository_port.update_prediction_accuracy(
            prediction_id=last.id,
            actual_price=current_price,
            accuracy_score=accuracy,
            is_trend_correct=trend_correct,
            insight=CORRECT_PREDICTION_INSIGHT,
            **extras,
        )
        _close_paper_trade(deps, last, current_price, ExitReason.SETTLED)
        return {
            "reflection_context": "Ostatnie prognozy były trafne. Kontynuuj strategię."
        }

    # #16: prior był BŁĘDNY. Tania księgowość (accuracy/trend/update) leci
    # ZAWSZE — to nie kosztuje płatnego API. PŁATNE analyze_mistake (LLM) jest
    # natomiast bramkowane tą samą logiką progu co główna bramka volatility
    # (_pick_threshold + Asset.evaluate_volatility), bo reflect biegnie po
    # check_price i ma już state["delta"]. Inaczej płaski cykl płaciłby za
    # diagnozę za każdym razem, gdy wisi nieoceniona błędna predykcja.
    #
    # UWAGA: to zawęża dotychczasową regułę "reflect jest w pełni niezależny
    # od zmienności" — niezależna pozostaje TYLKO tania część (ocena);
    # kosztowna diagnoza LLM jest teraz związana z progiem volatility.
    # FIN-14: ta sama bramka — ruch od stęchłej referencji nie płaci za diagnozę.
    # ALR-10: i ten sam dryf — powolny zjazd też odsłania błąd predykcji.
    # FIN-25: wyczerpany limit USD doby albo miesiąca — cykl nie płaci także za
    # diagnozę; rozliczenie (tania część) idzie jak zawsze.
    diagnose_paid = not deps.llm_budget_spent and _gate_verdict(
        state,
        threshold,
        crypto_threshold,
        reference_stale=_reference_is_stale(state),
        drift=_drift_of(state),
    ).passes

    if deps.batch_diagnosis_enabled:
        # EXP-17: diagnozę tego błędu zrobi raz w tygodniu Slow Loop, jednym
        # wywołaniem nad wszystkimi pudłami tygodnia — reflect nie płaci za nią.
        insight, category = DIAGNOSIS_BATCH_PENDING_INSIGHT, None
    elif diagnose_paid:
        # Counterfactual dissent-replay (#8): który dysydent rady trafił ruch,
        # mimo że agent poszedł inaczej? Wzbogaca diagnozę o stłumiony, trafny
        # głos mniejszości. W pełni graceful — brak głosów / błąd → bez bloku.
        vindicated_ctx = _vindicated_context(
            repository_port, last, current_price, symbol
        )
        # WIA-18: diagnoza widzi prawdziwe wejście predykcji (nagłówki,
        # uzasadnienie, cenę wejścia i cel). Prompt ogradza dane z bazy.
        prompt = get_mistake_diagnosis_prompt(
            last_trend=str(last.predicted_trend),
            last_news_summary=last.news_summary or "",
            actual_price=str(current_price),
            vindicated_context=vindicated_ctx,
            entry_price=str(last.price_at_prediction),
            target_price=str(last.predicted_target_price),
            last_reasoning=last.reasoning or "",
        )
        # #14: płatne wywołanie LLM diagnozy — log jawny + timing.
        _log_paid_call(deps, "reflect", symbol, "llm", "llm_port.analyze_mistake")
        with (
            _timed_node("reflect_analyze_mistake", symbol, deps.telemetry),
            usage_scope(source="llm", node="reflect", symbol=symbol),
        ):
            answer = llm_port.analyze_mistake(prompt)
        # WIA-57: JSON {category, insight}; odpowiedź bez niego → OTHER + cały tekst.
        diagnosis = parse_mistake_diagnosis(answer)
        insight, category = diagnosis.insight, diagnosis.category
    else:
        # Niska zmienność — nie palimy budżetu LLM na diagnozę. Predykcja
        # i tak zostaje zamknięta z accuracy/trend, więc diagnoza jest
        # POMINIĘTA, nie odłożona: nic jej później nie ponawia.
        insight, category = DIAGNOSIS_SKIPPED_INSIGHT, None

    repository_port.update_prediction_accuracy(
        prediction_id=last.id,
        actual_price=current_price,
        accuracy_score=accuracy,
        is_trend_correct=trend_correct,
        insight=insight,
        correction_category=category,
        **extras,
    )
    _close_paper_trade(deps, last, current_price, ExitReason.SETTLED)
    return {"reflection_context": f"Ostatni błąd: {insight}"}



def _analyze_sentiment_node(deps: AgentGraphDeps, state: AgentState) -> AgentState:
    sentiment_port = deps.sentiment_port
    symbol = state["symbol"]
    # #14: węzeł płatny (Alpha Vantage NEWS_SENTIMENT) — instrumentujemy czas
    # i logujemy jawnie fakt płatnego wywołania.
    _log_paid_call(
        deps, "analyze_sentiment", symbol, "sentiment", "sentiment_port.get_social_score"
    )
    with _timed_node("analyze_sentiment", symbol, deps.telemetry):
        # Port zwraca luźny dict; jego kształt opisuje `SentimentPayload`.
        payload = cast(SentimentPayload, sentiment_port.get_social_score(symbol))
        return {"sentiment": payload}



def _fetch_news_node(deps: AgentGraphDeps, state: AgentState) -> AgentState:
    news_port = deps.news_port
    symbol = state["symbol"]
    # #14: węzeł płatny (Alpha Vantage NEWS feed) — instrumentacja jak wyżej.
    _log_paid_call(deps, "fetch_news", symbol, "news", "news_port.get_news_context")
    with _timed_node("fetch_news", symbol, deps.telemetry):
        return {"news": news_port.get_news_context(symbol)}



def _compute_rag_context(
    deps: AgentGraphDeps, state: AgentState, news_summary: str
) -> tuple[list[float] | None, str, list[dict[str, Any]]]:
    """RAG: embedujemy podsumowanie newsów RAZ (reużyte w save_node) i
    wyszukujemy podobne historyczne sytuacje do wstrzyknięcia w prompt. W pełni
    graceful — błąd embeddingu/retrievalu zwraca puste konteksty.
    Zwraca ``(news_embedding, similar_context, similar_precedents)``."""
    news_embedding: list[float] | None = None
    similar_context = ""
    # Q5: precedent receipts — te same analogi, które trafiają do promptu,
    # reużyte jako audytowalny blok "Na podstawie analogów" w raporcie.
    similar_precedents: list[dict[str, Any]] = []
    if deps.embedding_port is not None and news_summary:
        # FIN-01: embedding to płatny request (OpenAI) — liczony w koszcie cyklu.
        _log_paid_call(
            deps, "predict", state["symbol"], "embedding", "embedding_port.embed"
        )
        try:
            with usage_scope(source="embedding", node="predict", symbol=state["symbol"]):
                news_embedding = deps.embedding_port.embed(news_summary)
        except Exception:
            logger.exception(
                "Embedding failed for %s — predicting without RAG", state["symbol"]
            )
        if news_embedding:
            # Vector Memory: gdy włączone, retrieval preferuje analogi z tego
            # samego reżimu rynku (filtr w RPC, migracja 017).
            regime_filter = (
                state.get("regime_context") if deps.vector_memory_enabled else None
            )
            similar_context, similar_precedents = _build_similar_context(
                deps.repository_port,
                news_embedding,
                state["symbol"],
                deps.rag_outcome_weight,
                regime=regime_filter,
                point_in_time=deps.rag_point_in_time_enabled,
                before=deps.clock() if deps.rag_point_in_time_enabled else None,
            )
    return news_embedding, similar_context, similar_precedents


def _run_llm_analysis(
    deps: AgentGraphDeps, state: AgentState, prompt: str, symbol: str
) -> tuple[dict[str, Any], list[str]]:
    """Płatna analiza jakościowa LLM. #6: dla NAJWIĘKSZYCH ruchów (osobny próg
    ``tool_use_threshold``) idzie kosztowna pętla tool-use; inaczej zwykły
    ``llm_port.analyze``. #12-guard: błąd parsowania/API degradujemy do
    neutralnego fallbacku. Zwraca ``(llm_analysis, flagi_jakości)``."""
    try:
        with usage_scope(source="llm", node="predict", symbol=symbol):
            return _call_prediction_llm(deps, state, prompt, symbol), []
    except LLMOutputIncomplete as exc:
        logger.warning(
            "llm_port.analyze returned an incomplete answer for %s (%s: %s) — "
            "degraduję do neutralnej analizy",
            state["symbol"],
            type(exc).__name__,
            exc,
        )
        return _neutral_llm_analysis(), ["llm_analysis_failed", LLM_OUTPUT_TRUNCATED_FLAG]
    except Exception:
        logger.exception(
            "llm_port.analyze failed for %s — degraduję do neutralnej analizy",
            state["symbol"],
        )
        return _neutral_llm_analysis(), ["llm_analysis_failed"]


def _neutral_llm_analysis() -> dict[str, Any]:
    return {
        "trend_direction": "SIDEWAYS",
        "confidence_score": 0.5,
        "av_agreement": 0.5,
        "reasoning": "Analiza LLM niedostępna — neutralny fallback.",
    }


def _call_prediction_llm(
    deps: AgentGraphDeps, state: AgentState, prompt: str, symbol: str
) -> dict[str, Any]:
    """Jedno płatne wywołanie predykcji: pętla tool-use albo `llm_port.analyze`."""
    # FIN-08: próg tool-use mnożą reżim i earnings jak każdą bramkę — w RISK-OFF
    # mniej symboli odpala pętlę. Krypto bierze własny próg tool-use, gdy jest
    # (None = wspólny `tool_use_threshold`).
    if (
        deps.tool_use_port is not None
        and deps.research_tools
        and deps.tool_use_threshold is not None
        and _gate_verdict(
            state, deps.tool_use_threshold, deps.crypto_tool_use_threshold
        ).passes
    ):
        _log_paid_call(
            deps, "predict", symbol, "llm", "tool_use_port.analyze_with_tools"
        )
        with _timed_node("predict_tool_use", symbol, deps.telemetry):
            return deps.tool_use_port.analyze_with_tools(prompt, deps.research_tools)
    _log_paid_call(deps, "predict", symbol, "llm", "llm_port.analyze")
    return deps.llm_port.analyze(prompt)


def _run_ml_prediction(
    ml_port: MLPredictionPort,
    features: dict[str, float],
    symbol: str,
    current_price: Decimal,
    *,
    return_scale: float | None = None,
) -> tuple[Decimal, tuple[FeatureContribution, ...]] | None:
    """Twarda predykcja liczbowa ML + atrybucja cech (SHAP-lite, lokalny CPU).
    Cold-start (model nietrenowany) → baseline 'cena bez zmian', brak atrybucji.
    Atrybucja jest graceful (błąd explain → sama predykcja). Zwraca
    ``(ml_target, feature_attribution)``.

    ML-11: `return_scale` (σ symbolu) idzie do modelu tylko, gdy jest; model
    w jednostkach σ bez niej → None (wołający idzie na baseline z flagą)."""
    if not ml_port.is_trained:
        return current_price, ()
    scale: dict[str, float] = {} if return_scale is None else {"return_scale": return_scale}
    # Model przewiduje zwrot do następnego cyklu; predict() rekonstruuje cenę bezwzględną
    # z bieżącej ceny (cena*(1+zwrot)).
    try:
        ml_target = ml_port.predict(features, current_price=current_price, **scale).amount
    except ReturnScaleUnavailableError:
        logger.warning("ML-11: brak σ dla %s — model w jednostkach σ pominięty", symbol)
        return None
    feature_attribution: tuple[FeatureContribution, ...] = ()
    try:
        contribs = {
            k: v for k, v in ml_port.explain(features, **scale).items() if k != "_bias"
        }
        feature_attribution = top_contributions(contribs)
    except Exception:
        logger.exception("explain() failed for %s — bez atrybucji", symbol)
    return ml_target, feature_attribution


def _ml_return_scale(deps: AgentGraphDeps, symbol: str) -> float | None:
    """ML-11: σ zwrotów symbolu z poprzednich cykli w chwili cyklu — ta sama funkcja
    domenowa co w treningu (`return_sigma`). Darmowy odczyt `price_snapshots`; błąd
    odczytu albo za krótka historia → None."""
    try:
        history = deps.repository_port.get_price_history(symbol, SIGMA_LOOKBACK.days)
    except PortError:
        logger.warning("ML-11: brak historii snapshotów %s — bez σ", symbol, exc_info=True)
        return None
    points = [(moment, float(price.amount)) for moment, price in history]
    return return_sigma(points, deps.clock())


def _shadow_forecasts(
    setup: ShadowSetup,
    features: dict[str, float],
    symbol: str,
    current_price: Decimal,
    ml_target: Decimal,
    *,
    ml_cold_start: bool,
    return_scale: float | None,
) -> tuple[ShadowForecast, ...]:
    """ML-33: prognoza championa (zwrot, który poszedł do maila; bez niej przy cold
    starcie) i prognozy cienia challengerów z TEGO SAMEGO wektora cech i σ.
    Challenger, który nie umie prognozować (cechy spoza wektora, model w σ bez σ),
    wypada z WARNING-iem — reszta cyklu go nie widzi."""
    forecasts: list[ShadowForecast] = []
    if not ml_cold_start and current_price > 0:
        forecasts.append(
            ShadowForecast(
                CHAMPION_MODEL_NAME,
                setup.champion_version,
                predicted_return=float((ml_target - current_price) / current_price),
            )
        )
    for challenger in setup.challengers:
        try:
            forecasts.append(challenger.forecast(features, return_scale=return_scale))
        except (PortError, ReturnScaleUnavailableError):
            logger.warning(
                "ML-33: challenger %s bez prognozy cienia dla %s",
                challenger.name,
                symbol,
                exc_info=True,
            )
    return tuple(forecasts)


def _persist_shadow_predictions(
    deps: AgentGraphDeps, state: AgentState, prediction_id: str
) -> None:
    """ML-33: prognozy cienia pod id zapisanej predykcji. Błąd zapisu (np. brak
    tabeli przed migracją 075) zostawia predykcję zapisaną, bez cieni."""
    forecasts = state.get("shadow_forecasts") or ()
    if deps.shadow is None or not forecasts:
        return
    try:
        deps.repository_port.save_shadow_predictions(prediction_id, state["symbol"], forecasts)
    except PortError:
        logger.warning(
            "save_shadow_predictions failed for %s — predykcja zapisana bez prognoz cienia",
            state["symbol"],
            exc_info=True,
        )


def _persist_shadow_prompt_forecast(
    deps: AgentGraphDeps, state: AgentState, prediction_id: str
) -> None:
    """EXP-26: trend wariantu pod id zapisanej predykcji. Błąd zapisu (np. brak kolumn
    przed migracją 083) zostawia predykcję zapisaną, bez wiersza wariantu."""
    forecast = state.get("shadow_prompt_forecast")
    if deps.shadow_prompt is None or forecast is None:
        return
    try:
        deps.repository_port.save_shadow_trend_prediction(
            prediction_id, state["symbol"], forecast
        )
    except PortError:
        logger.warning(
            "save_shadow_trend_prediction failed for %s — predykcja zapisana bez wariantu",
            state["symbol"],
            exc_info=True,
        )


@dataclasses.dataclass(frozen=True)
class _SealedCommitment:
    """Hash do publikacji po zapisie i — przy nadpisaniu innej predykcji w tej
    samej godzinie — commitment, który trzeba wtedy ujawnić jako „superseded"."""

    commitment: str
    superseded: StoredCommitment | None


def _seal_commitment(
    deps: AgentGraphDeps,
    record: dict[str, Any],
    *,
    council: CouncilVerdict | None,
    recipe: PredictionRecipe | None,
) -> _SealedCommitment | None:
    """#16 — dokleja do rekordu SHA-256 commitment, sól i zahashowany plaintext.

    Commit-reveal: publikujemy sam HASH treści predykcji + losowej soli.
    Dopóki sól nie zostanie ujawniona (sweep po rozliczeniu), hash jest
    nieodwracalny — sceptyk nie pozna predykcji, ale po ujawnieniu zweryfikuje,
    że powstała PRZED ruchem ceny.

    UCZCIWOŚĆ: daty commitów Gita są fałszowalne. Claim to „tamper-evidence
    przy branch protection", NIE „niepodrabialny timestamp".

    Klucze dokładane TYLKO gdy publisher istnieje — nowe kolumny przed migracjami
    024 i 043 dałyby PGRST204 i zabiły zapis całej predykcji (gate jak
    `receipts_enabled`; DI daje publisher dopiero, gdy kolumny istnieją).
    Zwraca commitment do publikacji PO zapisie (WIA-01) albo None, gdy atestacja off.

    WIA-41: werdykt rady powstaje po predict, więc commitment liczymy tutaj,
    w save_node — razem z pewnością i przepisem, z którego FAKTYCZNIE powstała
    predykcja (po failoverze to przepis z modelem zapasowym).
    """
    if deps.attestation_publisher is None:
        return None
    # WIA-02: schemat v2 — ceny w skali kolumny DECIMAL(14,4), te same co w rekordzie.
    payload = prediction_commitment_payload(
        symbol=record["symbol"],
        predicted_trend=record.get("predicted_trend"),
        predicted_target_price=record.get("predicted_target_price"),
        price_at_prediction=record["price_at_prediction"],
        confidence_score=record.get("confidence_score"),
        council_recommendation=None if council is None else council.final_recommendation,
        council_consensus_strength=None if council is None else council.consensus_strength,
        recipe_version=None if recipe is None else recipe.version,
        # WIA-42: hash protokołu ewaluacji, którym ta predykcja zostanie oceniona.
        protocol_sha256=None if recipe is None else recipe.protocol_sha256 or None,
    )
    # WIA-70: hash liczony z tych samych bajtów, które zapisujemy i ujawnimy dosłownie.
    plaintext = canonicalize(payload, schema_version=ATTESTATION_SCHEMA_VERSION)
    salt, superseded = _commitment_salt(deps, record["symbol"], plaintext)
    commitment = commit_canonical(plaintext, salt)
    record["commitment_hash"] = commitment
    record["commitment_salt"] = salt
    record["commitment_payload"] = plaintext
    return _SealedCommitment(commitment=commitment, superseded=superseded)


def _commitment_salt(
    deps: AgentGraphDeps, symbol: str, plaintext: str
) -> tuple[str, StoredCommitment | None]:
    """WIA-01: sól wiersza z tej samej godziny albo świeża, plus commitment,
    który ten zapis nadpisze INNĄ predykcją.

    Upsert (symbol, timestamp_hour) nadpisuje wiersz przy retry LangGraph albo
    re-runie GHA. Ten sam payload z tą samą solą daje ten sam, już opublikowany
    hash — bez revealu. Inny payload zabrałby z bazy sól i plaintext starego,
    opublikowanego hasha (nie do ujawnienia, czyli jak ukryte pudło), więc stary
    commitment idzie do revealu „superseded", a nowa predykcja dostaje świeżą
    sól — stara będzie jawna. Bez odczytu hasha (błąd, brak kolumny) zostaje
    reuse soli. Wartość spoza kontraktu portu (nie tekst, pusta) = brak."""
    existing = deps.repository_port.get_current_hour_commitment_salt(symbol)
    if not (isinstance(existing, str) and existing):
        return new_salt(), None
    stored = deps.repository_port.get_current_hour_commitment(symbol)
    if not isinstance(stored, StoredCommitment):
        return existing, None
    if stored.is_superseded_by(plaintext):
        return new_salt(), stored
    return stored.salt, None


def _superseded_reveal_record(symbol: str, stored: StoredCommitment) -> dict[str, Any]:
    """WIA-01 — reveal nadpisanego commitmentu: sól i dosłowny plaintext od razu,
    bo po upsercie nie ma ich już w bazie (sweep nigdy by ich nie ujawnił)."""
    record: dict[str, Any] = {
        "symbol": symbol,
        "commitment": stored.commitment,
        "salt": stored.salt,
        "status": REVEAL_STATUS_SUPERSEDED,
    }
    if stored.payload is not None:
        record["payload"] = stored.payload
    return record


def _publish_commitment(
    deps: AgentGraphDeps, symbol: str, sealed: _SealedCommitment, prediction_id: str
) -> None:
    """#16 — publikuje sam hash (bez soli), z id predykcji, czasem i wersją schematu
    (WIA-25). WIA-01: wołane DOPIERO po udanym `save_prediction` — hash bez zapisanej
    soli nigdy nie zostałby ujawniony, a nieudany zapis zostawia nadpisywany wiersz
    nietknięty (ujawni go sweep). Awaria publikacji nigdy nie wywala cyklu."""
    if deps.attestation_publisher is None:
        return
    if sealed.superseded is not None:
        try:
            deps.attestation_publisher.publish_reveal(
                _superseded_reveal_record(symbol, sealed.superseded)
            )
        except Exception:
            logger.exception("Superseded commitment reveal failed for %s", symbol)
    try:
        # Publikujemy WYŁĄCZNIE hash — sól zostaje w bazie do czasu revealu.
        deps.attestation_publisher.publish_commitment(
            commitment_ledger_record(
                symbol=symbol,
                commitment=sealed.commitment,
                prediction_id=str(prediction_id),
                committed_at=deps.clock(),
                schema_version=ATTESTATION_SCHEMA_VERSION,
            )
        )
    except Exception:
        logger.exception("Attestation commitment publish failed for %s", symbol)


# Tolerancja porównań z zerem. Score i wkłady fuzji to wynik ważonej sumy —
# „zero" wychodzi z niej jako ±1e-17, więc `== 0.0` przepuszczałoby szum
# arytmetyczny do promptu jako `insider +0.00`.
_ZERO_TOLERANCE = 1e-12


def _format_alpha_fusion(fusion: AlphaFusionScore | None) -> str:
    """#14 — blok promptu z rozbiciem wkładów, np.
    "Alpha Fusion +0.42 (insider +0.30, options +0.20, social -0.08)".

    Rozbicie jest w promptcie CELOWO: LLM ma widzieć, KTÓRE źródło pcha score,
    a nie tylko sumę — inaczej composite jest nieaudytowalną czarną skrzynką.
    Brak fuzji albo score 0.0 → "" → blok w ogóle nie wchodzi do promptu.
    """
    if fusion is None or math.isclose(fusion.score, 0.0, abs_tol=_ZERO_TOLERANCE):
        return ""
    parts = ", ".join(
        f"{name} {value:+.2f}"
        for name, value in sorted(fusion.contributions.items())
        if not math.isclose(value, 0.0, abs_tol=_ZERO_TOLERANCE)
    )
    head = f"Alpha Fusion {fusion.score:+.2f}"
    return f"{head} ({parts})" if parts else head


def _alpha_fusion_value(fusion: AlphaFusionScore | None) -> float:
    """Kandydat na cechę ML (poza `ML_FEATURE_COLUMNS`). Brak fuzji → 0.0,
    dokładnie jak `COALESCE` w widoku."""
    return fusion.score if fusion is not None else 0.0


# #18, WIA-28: horyzont predykcji to zwrot do następnego cyklu — pętla chodzi
# raz na dobę handlową, więc ~24 h w dni robocze i ~72 h przez weekend. Nazwy
# `*_12h` (`target_price_12h`, `actual_price_after_12h`) są historyczne i zostają
# jako kontrakt parsera i kolumn. Stałe 24 h ZANIŻA horyzont predykcji z piątku,
# więc za `edge_calendar_horizon_enabled` akcje i ETF biorą go z kalendarza NYSE.
_EDGE_HORIZON_HOURS = 24.0


def _edge_horizon_hours(deps: AgentGraphDeps, state: AgentState) -> float:
    """WIA-28: godziny do cyklu, który rozliczy predykcję. Z flagą akcje i ETF liczą
    je z kalendarza NYSE (do następnego dnia handlowego); krypto handluje 24/7
    i ma cykl codziennie, więc zostaje przy stałych 24 h, jak wszystko bez flagi."""
    if not deps.edge_calendar_horizon_enabled:
        return _EDGE_HORIZON_HOURS
    asset = state.get("asset") or Asset(symbol=state["symbol"])
    if asset.asset_type not in _SESSION_QUOTED_ASSETS:
        return _EDGE_HORIZON_HOURS
    return hours_to_next_trading_day(deps.clock())


def _resolve_implied_edge(
    deps: AgentGraphDeps,
    symbol: str,
    current_price: Decimal,
    ml_target: Decimal,
    *,
    ml_cold_start: bool,
    horizon_hours: float,
) -> ImpliedEdge | None:
    """#18 — dywergencja ruchu modelu od implied move rynku opcji.

    Wołane z `_predict_node`, czyli NATURALNIE za bramką volatility: przy niskiej
    zmienności płatny port opcji w ogóle nie rusza. Brak portu
    (`options_flow_enabled=false`) → None → cykl identyczny jak dziś.
    Cold-start → None: ruch baseline'u to sztuczne 0, nie ruch modelu.
    Graceful: Finnhub 403 na darmowym planie jest normą, nie awarią.
    """
    if deps.options_port is None or ml_cold_start:
        return None
    try:
        snapshot = deps.options_port.get_options_flow(symbol)
    except Exception:
        logger.warning("Options flow unavailable for %s — edge skipped", symbol)
        return None
    iv = snapshot.implied_vol if snapshot is not None else None
    move = model_move_from_prices(float(current_price), float(ml_target))
    return evaluate_edge(move, iv, horizon_hours)


def _resolve_move_interval(
    deps: AgentGraphDeps,
    state: AgentState,
    current_price: Decimal,
    ml_target: Decimal,
    *,
    ml_cold_start: bool,
) -> MoveInterval | None:
    """ML-34 — przedział zwrotu z okna kalibracyjnego klasy aktywa symbolu.

    Cold-start → predyktor 0 i empiryczne kwantyle zwrotów; model aktywny →
    jego zwrot ± kwantyl |reszt|. Zero wywołań: okno przyszło w deps."""
    if deps.move_calibration is None or current_price <= 0:
        return None
    asset = state.get("asset") or Asset(symbol=state["symbol"])
    window = deps.move_calibration.get(asset.asset_type, ())
    predicted = 0.0 if ml_cold_start else float((ml_target - current_price) / current_price)
    return move_interval(window, predicted_return=predicted, cold_start=ml_cold_start)


def _storage_price(price: Decimal | None) -> Decimal | None:
    """Cena w skali kolumny DECIMAL(14,4) (ARCH-38); brak ceny zostaje brakiem."""
    return None if price is None else Money(price).for_storage().amount


def _interval_price(price: Decimal, move_return: float) -> Decimal:
    """Granica przedziału jako cena (skala `predicted_target_price`, 4 miejsca)."""
    return Money(price * (Decimal(1) + Decimal(str(move_return)))).for_storage().amount


@dataclasses.dataclass(frozen=True)
class _PromptOptions:
    """Opcje promptu predykcji z flag cyklu (produkcja) albo po nadpisaniu wariantem
    w cieniu (EXP-26). `memory` = lekcje z reflect i analogi RAG w prompcie."""

    memory: bool
    rounded_delta: bool
    fence_lessons: bool
    include_probabilities: bool

    def with_variant(self, variant: PromptVariant) -> _PromptOptions:
        def chosen(override: bool | None, current: bool) -> bool:
            return current if override is None else override

        return _PromptOptions(
            memory=self.memory and not variant.without_memory,
            rounded_delta=chosen(variant.prompt_delta_rounding, self.rounded_delta),
            fence_lessons=chosen(variant.prompt_taint_fence, self.fence_lessons),
            include_probabilities=chosen(variant.trend_probabilities, self.include_probabilities),
        )


def _render_prediction_prompt(
    deps: AgentGraphDeps,
    state: AgentState,
    options: _PromptOptions,
    *,
    news_summary: str,
    similar_context: str,
) -> str:
    """Prompt predykcji z wejść cyklu — jedno źródło dla produkcji i wariantu w cieniu."""
    sentiment = state.get("sentiment") or {}
    news = state.get("news", [])
    return get_prediction_prompt(
        symbol=state["symbol"],
        current_data={
            "price": str(state["current_price"]),
            "delta_percentage": format_prompt_delta(
                to_percent(state["delta"]), rounded=options.rounded_delta
            ),
            "av_sentiment_score": sentiment.get("av_sentiment_score"),
            "av_sentiment_label": sentiment.get("av_sentiment_label"),
            "news_volume_24h": sentiment.get("news_volume_24h"),
            "high_relevance_count": sentiment.get("high_relevance_count"),
            "news_summary": news_summary,
            "news_titles": _news_titles(news, _NEWS_TITLES_LIMIT),
        },
        reflection_context=state.get("reflection_context", "") if options.memory else "",
        similar_context=similar_context if options.memory else "",
        # #7 reżim + #5 contagion (oba z bieżącego cyklu, przekazane w state).
        regime_context=state.get("regime_context", ""),
        peer_context=", ".join(
            f"{sym}:{stance}" for sym, stance in state.get("peer_context", ())
        ),
        # #14: pusty string przy braku fuzji → blok nie wchodzi do promptu.
        alpha_fusion_context=_format_alpha_fusion(state.get("alpha_fusion")),
        include_memory=options.memory,
        include_probabilities=options.include_probabilities,
        fence_lessons=options.fence_lessons,
        include_invalidation=deps.invalidation_level_enabled,
        # DAT-17: świeże 8-K tylko z flagą; "" = prompt bez bloku, jak przed DAT-17.
        corporate_events_context=(
            format_corporate_events(state.get("fresh_corporate_events", ()))
            if deps.corporate_events_enabled
            else ""
        ),
    )


def _shadow_prompt_state(
    deps: AgentGraphDeps,
    symbol: str,
    prompt_for: Callable[[_PromptOptions], str],
    production: _PromptOptions,
    *,
    production_failed: bool,
) -> AgentState:
    """EXP-26: trend wariantu promptu albo modelu w cieniu dla symbolu za bramką.

    Pomija symbol spoza stałej podpróbki i predykcję z atrapą produkcji (para bez
    porównania). Wywołanie rezerwuje telemetria pod `SHADOW_LLM_SOURCE` regułą domeny
    `shadow_call_allowed`: liczy się do sufitu cyklu, ale tylko w jego luzie, więc nie
    zabiera przydziału produkcji. Błąd portu wariantu tylko loguje — predykcja idzie dalej."""
    setup = deps.shadow_prompt
    if (
        setup is None
        or production_failed
        or not in_shadow_sample(setup.name, symbol, setup.sample_share)
    ):
        return {}

    def admits(calls: Mapping[str, int]) -> bool:
        return shadow_call_allowed(
            calls, cap=deps.paid_call_cap, max_shadow_calls=setup.max_calls_per_cycle
        )

    if not deps.telemetry.reserve_paid_call(SHADOW_LLM_SOURCE, admits):
        logger.info(
            "EXP-26: wariant %s pominięty dla %s — limit wywołań cienia albo sufit cyklu",
            setup.name,
            symbol,
        )
        return {}
    options = production.with_variant(setup.variant)
    _log_paid_call_line("predict_shadow", symbol, "shadow_prompt.llm_port.analyze")
    try:
        with (
            _timed_node("predict_shadow", symbol, deps.telemetry),
            usage_scope(source=SHADOW_LLM_SOURCE, node="predict_shadow", symbol=symbol),
        ):
            raw = setup.llm_port.analyze(prompt_for(options))
    except PortError:
        logger.warning(
            "EXP-26: wariant %s bez odpowiedzi dla %s — predykcja bez cienia",
            setup.name,
            symbol,
            exc_info=True,
        )
        return {}
    return {"shadow_prompt_forecast": _shadow_trend_forecast(setup, raw, symbol, options)}


def _shadow_trend_forecast(
    setup: ShadowPromptSetup, raw: dict[str, Any], symbol: str, options: _PromptOptions
) -> ShadowTrendForecast:
    """Trend i pewność wariantu po tej samej normalizacji co odpowiedź produkcji."""
    _, raw = _split_failover_marker(raw)
    _, raw = _split_routed_model(raw)
    analysis, _ = _normalize_llm_analysis(raw)
    if options.include_probabilities:
        analysis, _ = _apply_probability_forecast(analysis, symbol)
    confidence, _ = _saved_float(analysis.get("confidence_score"), "confidence_score", 0.5)
    return ShadowTrendForecast(
        setup.name, setup.version, TrendDirection(analysis["trend_direction"]), confidence
    )


def _predict_node(deps: AgentGraphDeps, state: AgentState) -> AgentState:
    repository_port = deps.repository_port
    ml_port = deps.ml_port
    symbol = state["symbol"]
    # #14: predict to najdroższy węzeł (LLM + ewentualny embedding/RAG).
    # Mierzymy całkowity czas węzła; perf_counter zamknięty tuż przed return.
    predict_start = time.perf_counter()
    sentiment = state.get("sentiment") or {}
    news = state.get("news", [])
    news_summary = _summarize_news(news)
    # EXP-03: ramię `no_memory` — bez lekcji z reflect i bez zapytania RAG. Embedding
    # newsów powstaje wtedy w save_node (fallback), więc koszt jak w status quo.
    memory = _uses_memory(state)

    news_embedding: list[float] | None = None
    similar_context = ""
    similar_precedents: list[dict[str, Any]] = []
    if memory:
        news_embedding, similar_context, similar_precedents = _compute_rag_context(
            deps, state, news_summary
        )

    # 1. LLM — analiza jakościowa z cross-validation pre-computed AV sentymentu
    prompt_options = _PromptOptions(
        memory=memory,
        rounded_delta=deps.prompt_delta_rounding_enabled,
        fence_lessons=deps.prompt_taint_fence_enabled,
        include_probabilities=deps.trend_probabilities_enabled,
    )

    def prompt_for(options: _PromptOptions) -> str:
        return _render_prediction_prompt(
            deps, state, options, news_summary=news_summary, similar_context=similar_context
        )

    prompt = prompt_for(prompt_options)
    raw_analysis, llm_flags = _run_llm_analysis(deps, state, prompt, symbol)
    failover, raw_analysis = _split_failover_marker(raw_analysis)
    model_used, raw_analysis = _split_routed_model(raw_analysis)
    # ML-07: kanonizacja PRZED pierwszym odczytem — rada, zapis i raport
    # czytają już ten sam znormalizowany dict, nie surowy tekst modelu.
    llm_analysis, analysis_flags = _normalize_llm_analysis(raw_analysis)
    trend_probabilities: dict[str, float] | None = None
    if deps.trend_probabilities_enabled:
        llm_analysis, trend_probabilities = _apply_probability_forecast(llm_analysis, symbol)
    invalidation_state: AgentState = {}
    if deps.invalidation_level_enabled:
        invalidation, invalidation_flag = _parse_invalidation(
            llm_analysis, state["current_price"]
        )
        invalidation_state["invalidation"] = invalidation
        if invalidation_flag is not None:
            analysis_flags.append(invalidation_flag)

    # 2. ML — twarda predykcja liczbowa (multi-feature)
    # #21: trend_direction walidowany jak pola liczbowe — present-but-invalid
    # (np. "UP") daje neutralny sygnał, ALE zostawia flagę jakości (z
    # normalizacji). Genuinely-absent default'uje do SIDEWAYS bez flagi.
    llm_trend_signal = _TREND_SIGNALS[llm_analysis["trend_direction"]]

    # Walidacja pól liczbowych — każdy padły input zostawia ślad w
    # data_quality_flags (None → _missing, nie-numeryczne/NaN → _invalid,
    # poza skalą [0, 1] → _out_of_range).
    flags: list[str] = list(llm_flags)
    if failover is not None:
        flags.append(LLM_FAILOVER_FLAG)
    flags.extend(analysis_flags)
    # Adapter sentymentu (AV) może oznaczyć cały feed jako zdegradowany
    # (np. wszystkie klucze wyczerpane). Wtedy 0.0 we wszystkich polach
    # to nie "spokojny dzień" tylko "brak danych" — flaga to rozróżnia.
    degraded_reason = sentiment.get("degraded_reason")
    if degraded_reason:
        flags.append(str(degraded_reason))

    def _v(value: Any, name: str, default: float = 0.0) -> float:
        parsed, flag = _validate_float(value, name, default)
        if flag is not None:
            flags.append(flag)
        return parsed

    # Cecha price_delta MUSI być liczona względem ceny POPRZEDNIEJ zalogowanej
    # predykcji — tak jak LAG(price_at_prediction) w widoku ml_feature_store.
    # state["delta"] (zmiana od snapshotu) napędza bramkę volatility, ale ma
    # inny interwał niż trening → użycie go tutaj dawało train/serve skew.
    # ML-18: bez wiersza z bieżącej godziny — upsert tej predykcji go nadpisze,
    # więc w widoku LAG liczy się od predykcji z wcześniejszej godziny.
    current_price = state["current_price"]
    last_pred_price = repository_port.get_last_prediction_price(
        state["symbol"], before=deps.clock()
    )
    if last_pred_price is not None and last_pred_price.amount != 0:
        # DAT-23: po podejrzanym splicie ostatnia predykcja (sprzed niego) w cenach po nim.
        reference = last_pred_price.amount * state.get("split_ratio", Decimal(1))
        ml_price_delta = float((current_price - reference) / reference)
    else:
        # Pierwsza predykcja dla symbolu — widok i tak odsiewa takie wiersze
        # (price_prev_12h NULL → price_delta NULL), więc neutralne 0.0 + flaga.
        ml_price_delta = 0.0
        flags.append("price_delta_no_prior_prediction")

    raw_features = {
        "price_delta": ml_price_delta,
        "av_sentiment_score": _v(
            sentiment.get("av_sentiment_score"), "av_sentiment_score"
        ),
        "av_relevance_avg": _v(
            sentiment.get("av_relevance_avg"), "av_relevance_avg"
        ),
        "news_volume_24h": _v(
            sentiment.get("news_volume_24h"), "news_volume_24h"
        ),
        "high_relevance_count": _v(
            sentiment.get("high_relevance_count"), "high_relevance_count"
        ),
        "llm_trend_signal": float(llm_trend_signal),
        # av_agreement domyślnie 0.5 (neutralny) — brak agreement to nie jest
        # to samo co 0% zgody (które jest informacją samą w sobie).
        "av_llm_agreement": _v(
            llm_analysis.get("av_agreement"), "av_llm_agreement", default=0.5
        ),
    }
    features = {name: raw_features[name] for name in ML_FEATURE_COLUMNS}
    # Cold-start guard: dopóki XGBoost nie ma wytrenowanych wag, baseline
    # "cena bez zmian" (helper); gdy model się wytrenuje, przejmuje predykcję.
    ml_cold_start = not ml_port.is_trained
    return_scale = (
        _ml_return_scale(deps, symbol)
        if deps.ml_sigma_target_enabled and ml_port.is_trained
        else None
    )
    prediction = _run_ml_prediction(
        ml_port, features, symbol, current_price, return_scale=return_scale
    )
    if prediction is None:
        # ML-11: model w jednostkach σ bez σ — baseline, nie liczba w złych jednostkach.
        flags.append(ML_SIGMA_UNAVAILABLE_FLAG)
        ml_cold_start = True
        prediction = (current_price, ())
    ml_target, feature_attribution = prediction
    shadow: AgentState = (
        {}
        if deps.shadow is None
        else {
            "shadow_forecasts": _shadow_forecasts(
                deps.shadow,
                features,
                symbol,
                current_price,
                ml_target,
                ml_cold_start=ml_cold_start,
                return_scale=return_scale,
            )
        }
    )

    # #14: pojedyncze wyjście węzła — logujemy całkowity czas predict.
    elapsed_ms = (time.perf_counter() - predict_start) * 1000.0
    logger.info("node=predict symbol=%s elapsed_ms=%.1f", symbol, elapsed_ms)
    deps.telemetry.record_node_timing("predict", symbol, elapsed_ms)
    # EXP-26: wariant w cieniu PO pomiarze predict — jego czas idzie pod własny węzeł.
    shadow_prompt = _shadow_prompt_state(
        deps, symbol, prompt_for, prompt_options, production_failed=_LLM_FALLBACK_FLAG in llm_flags
    )

    implied_edge = _resolve_implied_edge(
        deps,
        symbol,
        current_price,
        ml_target,
        ml_cold_start=ml_cold_start,
        horizon_hours=_edge_horizon_hours(deps, state),
    )
    interval = _resolve_move_interval(
        deps, state, current_price, ml_target, ml_cold_start=ml_cold_start
    )

    failover_state: AgentState = {} if failover is None else {"llm_failover": failover}
    routing_state: AgentState = {} if model_used is None else {"llm_model_used": model_used}
    return {
        **failover_state,
        **routing_state,
        **shadow,
        **shadow_prompt,
        **invalidation_state,
        "llm_analysis": llm_analysis,
        "trend_probabilities": trend_probabilities,
        "ml_target_price": ml_target,
        "ml_features": features,
        "ml_cold_start": ml_cold_start,
        "move_interval": interval,
        "data_quality_flags": flags,
        # #18: sygnał INFORMACYJNY (bez roli decyzyjnej) — do walidacji na
        # zamkniętych predykcjach, zanim dostanie jakąkolwiek moc sprawczą.
        "implied_edge": implied_edge,
        # Reużywane w save_node — embedujemy newsy tylko raz na cykl.
        "news_embedding": news_embedding,
        # Q5: precedent receipts dla raportu (render-only, brak persystencji).
        "similar_precedents": similar_precedents,
        # Q3: atrybucja cech ML dla raportu (render-only).
        "feature_attribution": feature_attribution,
    }



def _council_node(deps: AgentGraphDeps, state: AgentState) -> AgentState:
    council_port = deps.council_port
    council_threshold = deps.council_threshold
    crypto_council_threshold = deps.crypto_council_threshold
    if council_port is None:
        return {"council_verdict": None}
    # WIA-08: po awarii głównego LLM analiza jest zastępczym „SIDEWAYS 50%”, a cykl
    # i tak wypada z track recordu — rada nie płaci 8 wywołań za udawane wejście.
    if "llm_analysis_failed" in (state.get("data_quality_flags") or []):
        logger.info("council_node skipped for %s — main LLM analysis failed", state["symbol"])
        return {"council_verdict": None}
    # Osobny, opcjonalny próg dla rady. Główna bramka volatility już
    # przepuściła (predict_node odpalił) — tu odsiewamy dodatkowo średnie
    # zmienności, dla których N wywołań LLM rady się ekonomicznie nie opłaca.
    if council_threshold is not None:
        gate = _gate_verdict(state, council_threshold, crypto_council_threshold)
        if not gate.passes:
            logger.info(
                "council_node skipped for %s — |Δ|=%.4f < threshold=%.4f",
                state["symbol"],
                abs(float(state["delta"])),
                float(gate.threshold.value),
            )
            return {"council_verdict": None}
    sentiment = state.get("sentiment") or {}
    news = state.get("news", [])
    llm_analysis = state.get("llm_analysis") or {}
    data = CouncilInput(
        symbol=state["symbol"],
        current_price=state["current_price"],
        price_delta_pct=to_percent(state["delta"]),
        sentiment_score=_float_or_default(sentiment.get("av_sentiment_score"), 0.0),
        news_articles=tuple(_news_titles(news, _NEWS_TITLES_LIMIT)),
        llm_trend=str(llm_analysis.get("trend_direction", "SIDEWAYS")),
        llm_confidence=_float_or_default(llm_analysis.get("confidence_score"), 0.5),
        ml_price_target=state.get("ml_target_price") or state["current_price"],
        ml_model_active=not state.get("ml_cold_start", False),
        fundamentals=state.get("fundamentals"),
        valuation_verdict=state.get("valuation_verdict", ValuationVerdict.UNKNOWN),
        # #3: typ aktywa napędza dobór wag person (per klasa aktywa).
        asset_type=(state.get("asset") or Asset(symbol=state["symbol"])).asset_type,
        # EXP-02: ramię `blind` — te same wywołania rady, prompt bez werdyktu LLM.
        blind=has_treatment(state.get("experiment_arms") or {}, ARM_BLIND_COUNCIL),
    )
    # #14: rada to N równoległych wywołań LLM — najdroższy płatny węzeł.
    # Jawny log płatnego wywołania + timing (mierzony też przy błędzie).
    # FIN-02: węzeł tylko loguje — każdy request person, chairmana i rewizji
    # liczy `MeteredLLMPort`, którym DI owija port LLM rady.
    _log_paid_call_line("council", state["symbol"], "council_port.analyze")
    try:
        with (
            _timed_node("council", state["symbol"], deps.telemetry),
            usage_scope(source="council_llm", node="council", symbol=state["symbol"]),
        ):
            verdict = council_port.analyze(state["symbol"], data)
    except Exception:
        logger.exception(
            "council_node failed for %s — continuing without verdict",
            state["symbol"],
        )
        # MAIL-12: brak werdyktu po awarii ≠ rada pominięta progiem — flaga trafia
        # do zapisu i do odznak proweniencji w mailu.
        return {
            "council_verdict": None,
            "data_quality_flags": [
                *(state.get("data_quality_flags") or []),
                COUNCIL_UNAVAILABLE_FLAG,
            ],
        }
    return {"council_verdict": verdict}



def _uses_memory(state: AgentState) -> bool:
    """EXP-03: czy prompt predykcji dostaje pamięć (lekcje z reflect i analogi
    RAG). Fałsz tylko w ramieniu `no_memory` aktywnego eksperymentu."""
    return not has_treatment(state.get("experiment_arms") or {}, ARM_NO_MEMORY)


def _resolve_news_vector(
    deps: AgentGraphDeps, state: AgentState, news_summary: str
) -> list[float] | None:
    """Wektor newsów do zapisu w pgvector. predict_node zwykle już go policzył
    (RAG) — reużywamy; fallback: embedujemy tu, jeśli predict go nie dostarczył.
    Graceful: brak portu / pusty tekst / błąd API → None (zapis bez wektora).
    EXP-03: ramię `no_memory` nie robi zapytania RAG, więc wektor powstaje tutaj —
    predykcja trafia do korpusu analogów jak w status quo, bez dziury po eksperymencie."""
    if deps.embedding_port is None or not news_summary:
        return None
    vector = state.get("news_embedding")
    if vector:
        return vector
    # FIN-01: liczymy tylko realny request — wektor reużyty z predict nie kosztuje.
    _log_paid_call(deps, "save", state["symbol"], "embedding", "embedding_port.embed")
    try:
        with usage_scope(source="embedding", node="save", symbol=state["symbol"]):
            return deps.embedding_port.embed(news_summary)
    except Exception:
        logger.exception(
            "Embedding failed for %s — saving without vector", state["symbol"]
        )
        return None


# FIN-05: kolumny `prediction_logs`, które `_save_node` zapisuje w każdym rekordzie.
_ALWAYS_SAVED_PREDICTION_COLUMNS: frozenset[str] = frozenset({
    "symbol",
    "price_at_prediction",
    "sentiment_score",
    "av_relevance_avg",
    "news_volume_24h",
    "high_relevance_count",
    "av_llm_agreement",
    "news_summary",
    "predicted_trend",
    "predicted_target_price",
    "confidence_score",
    "reasoning_text",
    "council_verdict",
    "data_quality_flags",
    "alpha_fusion_score",
})


# WIA-61 (migracja 059): kolumny zamknięcia sesji, które reflect dopisuje do rozliczenia.
SESSION_CLOSE_SETTLEMENT_COLUMNS = (
    "settlement_close_price",
    "settlement_session_date",
    "settlement_source",
    "is_trend_correct_session_close",
)


def required_prediction_columns(deps: AgentGraphDeps) -> frozenset[str]:
    """FIN-05: każda kolumna `prediction_logs`, którą `_save_node` może zapisać przy
    tych zależnościach — klucze opcjonalne wyłącznie za ich flagą albo portem.
    WIA-61: także kolumny zamknięcia sesji, które za flagą dopisuje rozliczenie —
    bez nich każdy symbol padałby w reflect.

    DI sprawdza je raz na cykl, przed płatnymi węzłami: brakująca kolumna
    (niezaaplikowana migracja) kończyłaby zapis błędem po opłaconej analizie,
    w każdym symbolu i w każdym cyklu. Test pilnuje, że lista pokrywa rekord."""
    optional = {
        "regime": deps.vector_memory_enabled or deps.regime_tagging_enabled,
        "decision_receipts": deps.receipts_enabled,
        "feature_set_version": deps.feature_set_version_enabled,
        "recipe_version": deps.recipe_version_enabled and deps.recipe is not None,
        "edge_sigma": deps.options_port is not None,
        "interval_low": deps.interval_columns_enabled,
        "interval_high": deps.interval_columns_enabled,
        "gate_delta": deps.gate_delta_enabled,
        "experiment_arms": deps.experiment_arms_enabled,
        # DAT-14 (migracja 055): werdykt notowania z `quote_port`, za flagą.
        "stale_quote": deps.stale_quote_enabled and deps.quote_port is not None,
        # WIA-61 (migracja 059): zamknięcie sesji z notowania `quote_port`, za flagą.
        **dict.fromkeys(
            SESSION_CLOSE_SETTLEMENT_COLUMNS,
            deps.session_close_settlement_enabled and deps.quote_port is not None,
        ),
        # WIA-54 (migracja 072): cena wykonalna z notowania `quote_port`, za flagą.
        "entry_price": deps.executable_entry_enabled and deps.quote_port is not None,
        "commitment_hash": deps.attestation_publisher is not None,
        "commitment_salt": deps.attestation_publisher is not None,
        # WIA-70 (migracja 043): dosłowny plaintext commitmentu, ta sama zasada.
        "commitment_payload": deps.attestation_publisher is not None,
        # ML-27 (migracja 074): poziom unieważnienia tezy, tylko z flagą.
        "invalidation_price": deps.invalidation_level_enabled,
        # ML-29 (migracja 051): rozkład trzech klas, tylko z flagą.
        "trend_probabilities": deps.trend_probabilities_enabled,
        "embedding": deps.embedding_port is not None,
    }
    return _ALWAYS_SAVED_PREDICTION_COLUMNS | {
        column for column, written in optional.items() if written
    }


# WIA-09: atrapa po awarii głównego LLM nie jest predykcją (poza track recordem).
_LLM_FALLBACK_FLAG = "llm_analysis_failed"


def _open_paper_trade(
    deps: AgentGraphDeps, state: AgentState, prediction_id: str, record: Mapping[str, Any]
) -> None:
    """WIA-59: pozycja papierowa zapisanej predykcji (za `paper_ledger_costs`).

    Wejście po cenie z rekordu (skala DECIMAL(14,4)), strona z kierunku. Atrapa LLM
    i predykcja na nieaktualnym notowaniu (DAT-14) nie liczą się do trafności, więc
    nie otwierają pozycji. Błąd zapisu tylko loguje — predykcja już jest w bazie."""
    costs = deps.paper_ledger_costs
    if costs is None:
        return
    if _LLM_FALLBACK_FLAG in (state.get("data_quality_flags") or []) or state.get("stale_quote"):
        return
    asset = state.get("asset")
    entry = PaperTradeEntry(
        prediction_id=prediction_id,
        symbol=state["symbol"],
        side=side_of(str(record["predicted_trend"])),
        entry_price=record["price_at_prediction"],
        entry_ts=deps.clock(),
        cost_bps=costs.for_asset(asset.asset_type if asset is not None else None),
    )
    try:
        deps.repository_port.append_paper_trade(entry)
    except PortError:
        logger.warning(
            "append_paper_trade failed for %s — prediction saved without a paper trade",
            state["symbol"],
            exc_info=True,
        )


def _save_node(deps: AgentGraphDeps, state: AgentState) -> AgentState:
    vector_memory_enabled = deps.vector_memory_enabled
    repository_port = deps.repository_port
    llm_analysis = state.get("llm_analysis", {})
    sentiment = state.get("sentiment") or {}
    news_summary = _summarize_news(state.get("news", []))
    data_quality_flags = state.get("data_quality_flags", []) or []
    if data_quality_flags:
        logger.warning(
            "data_quality issues for %s: %s — record saved with flags, "
            "downstream training should filter these out",
            state["symbol"],
            data_quality_flags,
        )

    # #17: zawężenie do zmiennej lokalnej WEWNĄTRZ gałęzi `is not None`, żeby
    # mypy widział `CouncilVerdict` (nie `CouncilVerdict | None`) w asdict().
    # Wcześniej maskował to szeroki override `disable_error_code=["arg-type"]`.
    # ML-35: pewność z LLM po polityce `_validate_float` — NaN/Inf/nie-liczba to
    # default z flagą `_invalid` (brak pola zostaje cichym defaultem, jak dotąd).
    confidence, confidence_flag = _saved_float(
        llm_analysis.get("confidence_score"), "confidence_score", 0.5
    )
    if confidence_flag is not None:
        data_quality_flags = [*data_quality_flags, confidence_flag]

    council_verdict = state.get("council_verdict")
    council_verdict_dict = (
        _council_verdict_blob(council_verdict) if council_verdict is not None else None
    )

    record: dict[str, Any] = {
        "symbol": state["symbol"],
        # WIA-02: ceny w skali kolumn DECIMAL(14,4) — commitment liczony jest z tych
        # samych wartości, które baza przechowa.
        "price_at_prediction": _storage_price(state["current_price"]),
        # Zapisujemy nowy główny sygnał sentymentu (AV) jako sentiment_score.
        # Schema bazy nie wymaga zmian — pole jest generyczne.
        "sentiment_score": sentiment.get("av_sentiment_score"),
        "av_relevance_avg": sentiment.get("av_relevance_avg"),
        "news_volume_24h": sentiment.get("news_volume_24h"),
        "high_relevance_count": sentiment.get("high_relevance_count"),
        "av_llm_agreement": _float_or_default(
            llm_analysis.get("av_agreement"), 0.5
        ),
        "news_summary": news_summary,
        "predicted_trend": llm_analysis.get("trend_direction"),
        "predicted_target_price": _storage_price(state.get("ml_target_price")),
        # #4: persystujemy deklarowaną pewność — bez niej Slow-Loopowy sędzia
        # nie oceni kalibracji (migracja 016).
        "confidence_score": confidence,
        "reasoning_text": llm_analysis.get("reasoning"),
        "council_verdict": council_verdict_dict,
        "data_quality_flags": data_quality_flags,
        # #14: kandydat na cechę ML (migracja 022) — kolumna widoku, ale poza
        # `ML_FEATURE_COLUMNS`, więc model jej nie używa; promocja po A/B (EXP-19).
        # Widok robi COALESCE(...,0.0), żeby po promocji historyczne NULL-e nie
        # wypadły przez `dropna` (train/serve-skew).
        "alpha_fusion_score": _alpha_fusion_value(state.get("alpha_fusion")),
    }

    # Tag reżimu rynku, w którym powstała predykcja (migracja 017): Vector
    # Memory filtruje po nim analogi, a ML-38 zbiera go także bez RAG — do
    # segmentacji trafności po reżimie. Bez obu flag kolumna zostaje NULL.
    if vector_memory_enabled or deps.regime_tagging_enabled:
        record["regime"] = canonical_regime_key(state.get("regime_context"))

    # #13: kwit decyzyjny — pełny łańcuch dowodowy predykcji (analogi RAG,
    # atrybucje cech, PRÓG EFEKTYWNY, odznaki proweniencji). Dopisujemy klucz
    # TYLKO za flagą: nowy klucz przed migracją 020 → PGRST204 i śmierć zapisu
    # całej predykcji (dokładnie ten sam gate co `vector_memory_enabled`).
    if deps.receipts_enabled:
        record["decision_receipts"] = _build_decision_receipts(deps, state)

    # DAT-09: wersja zestawu cech, z którą policzono wejścia tej predykcji —
    # trening oddziela nią wiersze sprzed zmiany znaczenia `news_volume_24h`.
    if deps.feature_set_version_enabled:
        record["feature_set_version"] = feature_set_version(deps.market_features_enabled)

    # WIA-20: wersja przepisu — track record segmentuje się po niej. NIEZ-44: po
    # failoverze to przepis z modelem zapasowego (osobny segment).
    recipe = _effective_recipe(deps, state)
    if deps.recipe_version_enabled and recipe is not None:
        record["recipe_version"] = recipe.version

    # #18: `edge_sigma` (migracja 025) tylko gdy sygnał realnie powstał —
    # bez portu opcji klucz nie istnieje, więc stary schemat zapisuje się jak dotąd.
    implied_edge = state.get("implied_edge")
    if implied_edge is not None:
        record["edge_sigma"] = implied_edge.edge_sigma

    # ML-34: przedział 80% jako ceny (migracja 032) — Track Record liczy z nich
    # pokrycie zrealizowane. Bez przedziału (za mało historii) klucza nie ma.
    interval = state.get("move_interval")
    if (
        deps.interval_columns_enabled
        and interval is not None
        and interval.low is not None
        and interval.high is not None
    ):
        record["interval_low"] = _interval_price(state["current_price"], interval.low)
        record["interval_high"] = _interval_price(state["current_price"], interval.high)

    # EXP-01: delta bramki do testu echa momentum. Cold-start (brak snapshotu,
    # previous_price 0) to brak punktu odniesienia, nie ruch — klucza nie ma.
    if deps.gate_delta_enabled and state["previous_price"] != 0:
        record["gate_delta"] = float(state["delta"])

    # EXP-05: ramiona aktywnych eksperymentów (migracja 037). Bez aktywnego
    # eksperymentu klucza nie ma — kolumna zostaje NULL.
    arms = state.get("experiment_arms")
    if deps.experiment_arms_enabled and arms:
        record["experiment_arms"] = dict(arms)

    # ML-29: rozkład trzech klas (migracja 051) — RPS i 3-klasowy Brier. Bez
    # poprawnego rozkładu od LLM klucza nie ma, kolumna zostaje NULL.
    trend_probabilities = state.get("trend_probabilities")
    if deps.trend_probabilities_enabled and trend_probabilities is not None:
        record["trend_probabilities"] = dict(trend_probabilities)

    # ML-27: poziom unieważnienia tezy (migracja 074) — watch alarmuje po jego
    # przekroczeniu. Bez poprawnego poziomu od LLM klucza nie ma, kolumna zostaje NULL.
    invalidation = state.get("invalidation")
    if deps.invalidation_level_enabled and invalidation is not None:
        record["invalidation_price"] = _storage_price(invalidation.price)

    # DAT-14: predykcja na notowaniu sprzed ostatniej sesji wypada z trafności po
    # rozliczeniu. Bez notowania z czasem (krypto) kolumna zostaje NULL.
    stale_quote = state.get("stale_quote")
    if deps.stale_quote_enabled and stale_quote is not None:
        record["stale_quote"] = stale_quote

    # #16: commit-reveal (migracje 024, 043); publikacja dopiero po zapisie.
    sealed = _seal_commitment(deps, record, council=council_verdict, recipe=recipe)

    # Embedding podsumowania newsów → pgvector (reużyty z predict_node lub
    # policzony tu w fallbacku; graceful — przy braku zapisujemy bez wektora).
    vector = _resolve_news_vector(deps, state, news_summary)
    if vector:
        record["embedding"] = vector

    prediction_id = repository_port.save_prediction(record)
    _open_paper_trade(deps, state, prediction_id, record)
    # WIA-01: hash trafia do publicznego logu dopiero, gdy sól jest w bazie.
    if sealed is not None:
        _publish_commitment(deps, state["symbol"], sealed, prediction_id)
    _persist_shadow_predictions(deps, state, prediction_id)
    _persist_shadow_prompt_forecast(deps, state, prediction_id)

    # Strukturalny audit trail rady — JSONB blob w prediction_logs zostaje
    # (wstecz-kompat), ale głosy lecą też do osobnej tabeli, żeby dało się
    # je odpytywać per inwestor/symbol bez parsowania JSON. Pusta lista =
    # no-op po stronie adaptera (rada padła lub pominięta progiem).
    verdict = state.get("council_verdict")
    if verdict is not None:
        try:
            repository_port.save_council_votes(
                prediction_id=prediction_id,
                symbol=state["symbol"],
                votes=list(verdict.investor_opinions),
            )
        except PortError:
            logger.exception(
                "save_council_votes failed for %s — JSONB blob in "
                "prediction_logs is still saved",
                state["symbol"],
            )

    # #6: węzeł terminalny — zapisujemy snapshot ceny dla następnego cyklu.
    _persist_price_snapshot(deps, state)
    _persist_feature_snapshot(deps, state, gated=True)

    return {"prediction_id": prediction_id, "status": "saved"}



def _council_verdict_blob(verdict: CouncilVerdict) -> dict[str, Any]:
    """JSONB `prediction_logs.council_verdict` w kształcie sprzed ML-40: głosy rundy 1
    debaty niesie `council_votes.round` (migracja 082), więc blob, kwit i czytniki
    werdyktu zostają bajt w bajt takie same z rewizją i bez niej."""
    blob = dataclasses.asdict(verdict)
    del blob["first_round_opinions"]
    return blob


def _ignore_node(deps: AgentGraphDeps, state: AgentState) -> AgentState:
    # DAT-15: anomalny odczyt nie nadpisuje referencji ani nie trafia do wektora cech —
    # następny cykl liczy ruch od ostatniego wiarygodnego snapshotu.
    if state.get("price_outlier"):
        return {"status": "ignored"}
    # #6: węzeł terminalny — zapisujemy snapshot ceny dla następnego cyklu.
    _persist_price_snapshot(deps, state)
    # FIN-14 (decyzja 2026-10-19): cykl re-baseline nie jest obserwacją cyklu —
    # `snapshot_delta` od stęchłej referencji to ruch z wielu dni, więc wektor
    # ML-14 nie powstaje. Zapisujemy tylko nowy punkt odniesienia.
    if not _reference_is_stale(state):
        _persist_feature_snapshot(deps, state, gated=False)
    threshold = _pick_threshold(state, deps.threshold, deps.crypto_threshold)
    return {"status": "ignored", "effective_threshold": Ratio(threshold.value)}


def _budget_exhausted_node(deps: AgentGraphDeps, state: AgentState) -> AgentState:
    """FIN-12: węzeł terminalny symbolu odciętego budżetem cyklu — jedyny wyjątek
    od zapisu snapshotu w każdym cyklu. Snapshot z bieżącą ceną zjadłby ruch ponad
    progiem: następny cykl liczyłby Δ od nowej ceny i ruch zniknąłby bez analizy.
    Wektor ML-14 też nie powstaje — ten sam ruch wróci jako obserwacja następnego
    cyklu, więc zapis tutaj zdublowałby go w treningu."""
    return {"status": "ignored", "budget_exhausted": True}



def create_agent_graph(deps: AgentGraphDeps) -> StateGraph[AgentState]:
    """Fabryka grafu LangGraph — Fast Loop kompletny.

    Topologia bez fundamentals_port:
        check_price → reflect → [delta ≥ threshold?]
                                  ├── nie → ignore → END
                                  ├── tak, budżet LLM cyklu wyczerpany → budget_exhausted → END
                                  └── tak → analyze_sentiment → fetch_news
                                                              → predict → save → END

    Topologia z fundamentals_port:
        check_price → reflect → fetch_fundamentals → [delta ≥ threshold?]
                                                        ├── nie → ignore → END
                                                        └── tak → analyze_sentiment → …

    `reflect` i `fetch_fundamentals` są PRZED bramką volatility — działają
    w każdym cyklu niezależnie od bieżącej zmienności. Czytanie cache
    fundamentów jest darmowe; płatne wywołania AV dzieją się tylko w slow loop.
    """

    # ---------- Topologia ----------
    workflow = StateGraph[AgentState](state_schema=AgentState)
    workflow.add_node("check_price", lambda s: _check_price_node(deps, s))
    workflow.add_node("reflect", lambda s: _reflect_node(deps, s))
    workflow.add_node("analyze_sentiment", lambda s: _analyze_sentiment_node(deps, s))
    workflow.add_node("fetch_news", lambda s: _fetch_news_node(deps, s))
    workflow.add_node("predict", lambda s: _predict_node(deps, s))
    workflow.add_node("council", lambda s: _council_node(deps, s))
    workflow.add_node("save", lambda s: _save_node(deps, s))
    workflow.add_node("ignore", lambda s: _ignore_node(deps, s))
    workflow.add_node("budget_exhausted", lambda s: _budget_exhausted_node(deps, s))

    workflow.set_entry_point("check_price")
    # check_price → reflect: bezwarunkowo. Ocena przeszłej predykcji odbywa się
    # w KAŻDYM cyklu, niezależnie od bieżącej zmienności.
    workflow.add_edge("check_price", "reflect")

    if deps.fundamentals_port is not None:
        # reflect → fetch_fundamentals → bramka volatility.
        # Węzeł fundamentów jest PRZED bramką — czytanie cache jest darmowe.
        workflow.add_node(
            "fetch_fundamentals",
            # #17: LangGraph 1.x stubs typują węzły jako `_Node[Never]` i nie
            # inferują schematu opartego o TypedDict (StateGraph[AgentState]).
            # Runtime jest poprawny; wąski ignore zamiast szerokiego override.
            _build_fetch_fundamentals_node(deps.fundamentals_port),  # type: ignore[arg-type]
        )
        workflow.add_edge("reflect", "fetch_fundamentals")
        # fetch_fundamentals → bramka volatility: dopiero TU decydujemy, czy robić
        # nową prognozę.
        workflow.add_conditional_edges(
            "fetch_fundamentals",
            lambda s: _should_analyze(deps, s),
            {
                "analyze_sentiment": "analyze_sentiment",
                "ignore": "ignore",
                "budget_exhausted": "budget_exhausted",
            },
        )
    else:
        # Tryb kompatybilności wstecznej: reflect → bramka volatility bezpośrednio.
        workflow.add_conditional_edges(
            "reflect",
            lambda s: _should_analyze(deps, s),
            {
                "analyze_sentiment": "analyze_sentiment",
                "ignore": "ignore",
                "budget_exhausted": "budget_exhausted",
            },
        )

    workflow.add_edge("analyze_sentiment", "fetch_news")
    workflow.add_edge("fetch_news", "predict")
    workflow.add_edge("predict", "council")
    workflow.add_edge("council", "save")
    workflow.add_edge("save", END)
    workflow.add_edge("ignore", END)
    workflow.add_edge("budget_exhausted", END)

    return workflow


def create_gate_graph(deps: AgentGraphDeps) -> StateGraph[AgentState]:
    """EXP-27 faza 1 cyklu dwufazowego — darmowa część grafu dla KAŻDEGO symbolu:

        check_price → reflect → [fetch_fundamentals] → [delta ≥ threshold?]
                                                          ├── nie → ignore → END
                                                          └── tak → END (czeka na fazę 2)

    `reflect` zostaje przed bramką (Self-Reflection), łącznie z płatną diagnozą
    pomyłki pod tą samą bramką. Symbol za bramką kończy bez snapshotu ceny: zapisze
    go terminal fazy 2 (`save`), a symbol odcięty sufitem zostaje bez snapshotu
    jak w FIN-12. Stan końcowy jest wejściem `create_paid_graph`."""
    workflow = StateGraph[AgentState](state_schema=AgentState)
    workflow.add_node("check_price", lambda s: _check_price_node(deps, s))
    workflow.add_node("reflect", lambda s: _reflect_node(deps, s))
    workflow.add_node("ignore", lambda s: _ignore_node(deps, s))
    workflow.set_entry_point("check_price")
    workflow.add_edge("check_price", "reflect")
    gate_source = "reflect"
    if deps.fundamentals_port is not None:
        fetch_fundamentals = _build_fetch_fundamentals_node(deps.fundamentals_port)
        workflow.add_node("fetch_fundamentals", lambda s: fetch_fundamentals(s))
        workflow.add_edge("reflect", "fetch_fundamentals")
        gate_source = "fetch_fundamentals"
    workflow.add_conditional_edges(
        gate_source,
        lambda s: _gate_phase_route(deps, s),
        {"ignore": "ignore", GATE_PASSED: END},
    )
    workflow.add_edge("ignore", END)
    return workflow


def create_paid_graph(deps: AgentGraphDeps) -> StateGraph[AgentState]:
    """EXP-27 faza 2 — płatna część grafu dla symbolu za bramką, od stanu fazy 1:

        [budżet cyklu?] ├── wyczerpany → budget_exhausted → END
                        └── jest → analyze_sentiment → fetch_news → predict
                                   → council → save → END

    Te same węzły co w `create_agent_graph`; DI podaje tu porty feedu klienta AV
    zbudowanego tylko dla symboli za bramką."""
    workflow = StateGraph[AgentState](state_schema=AgentState)
    workflow.add_node("analyze_sentiment", lambda s: _analyze_sentiment_node(deps, s))
    workflow.add_node("fetch_news", lambda s: _fetch_news_node(deps, s))
    workflow.add_node("predict", lambda s: _predict_node(deps, s))
    workflow.add_node("council", lambda s: _council_node(deps, s))
    workflow.add_node("save", lambda s: _save_node(deps, s))
    workflow.add_node("budget_exhausted", lambda s: _budget_exhausted_node(deps, s))
    workflow.set_conditional_entry_point(
        lambda s: _paid_phase_route(deps, s),
        {"analyze_sentiment": "analyze_sentiment", "budget_exhausted": "budget_exhausted"},
    )
    workflow.add_edge("analyze_sentiment", "fetch_news")
    workflow.add_edge("fetch_news", "predict")
    workflow.add_edge("predict", "council")
    workflow.add_edge("council", "save")
    workflow.add_edge("save", END)
    workflow.add_edge("budget_exhausted", END)
    return workflow

