from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.application.decision_receipt import RECEIPT_SCHEMA_V1, RECEIPT_SCHEMA_V2
from src.application.ml_features import (
    FEATURE_SET_VERSION,
    FREE_SNAPSHOT_FEATURES,
    GATED_SNAPSHOT_EXTRAS,
    ML_FEATURE_COLUMNS,
)
from src.application.ports import (
    AdvisoryCouncilPort,
    AttestationPublisherPort,
    EmbeddingPort,
    FundamentalsPort,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    OptionsFlowPort,
    RepositoryPort,
    SentimentPort,
    Tool,
    ToolUseLLMPort,
)
from src.application.prompts import get_mistake_diagnosis_prompt, get_prediction_prompt
from src.application.rag_rerank import rerank_analogs
from src.domain.alpha_fusion import AlphaFusionScore
from src.domain.asset import Asset, PriceDelta
from src.domain.attestation import commit as attestation_commit
from src.domain.attestation import new_salt
from src.domain.conformal import CalibrationPoint, MoveInterval, move_interval
from src.domain.council import CouncilInput, CouncilVerdict, vindicated_dissenters
from src.domain.feature_attribution import FeatureContribution, top_contributions
from src.domain.implied_edge import (
    ImpliedEdge,
    evaluate_edge,
    model_move_from_prices,
)
from src.domain.prediction import (
    CORRECT_PREDICTION_INSIGHT,
    DIAGNOSIS_SKIPPED_INSIGHT,
    Prediction,
    is_substantive_insight,
    normalize_trend,
    sideways_tolerance_for,
)
from src.domain.recipe import PredictionRecipe
from src.domain.regime_key import canonical_regime_key
from src.domain.value_objects import (
    AssetType,
    Fundamentals,
    Money,
    Threshold,
    ValuationVerdict,
    clamp_unit,
)

logger = logging.getLogger(__name__)


@contextmanager
def _timed_node(node_name: str, symbol: str) -> Iterator[None]:
    """#14: lekka instrumentacja czasu wykonania płatnego węzła.

    Mierzy elapsed_ms i loguje go na INFO (z nazwą węzła i symbolem). Czas
    jest raportowany ZAWSZE, nawet gdy węzeł rzuci wyjątek (FinOps: chcemy
    wiedzieć, jak długo trwało wywołanie, które padło). Trzyma się warstwy
    application — żadnego nowego portu, czysty stdlib.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "node=%s symbol=%s elapsed_ms=%.1f", node_name, symbol, elapsed_ms
        )


def _log_paid_call(
    deps: AgentGraphDeps, node_name: str, symbol: str, source: str, detail: str
) -> None:
    """Jawna linia w logu, gdy odpala się PŁATNE wywołanie zewnętrzne
    (LLM / sentyment / news). Pozwala audytować FinOps — ile płatnych wywołań
    poszło w danym cyklu i z którego węzła.

    `source` to klucz cennika z `src/domain/finops.py` (`llm`, `council_llm`,
    `sentiment`, `news`). Inkrementujemy licznik tutaj — jedno miejsce prawdy
    o tym, co jest płatne. Licznik jest współdzielony przez wszystkie symbole
    cyklu, bo raportujemy KOSZT CYKLU, nie symbolu.
    """
    logger.info(
        "paid call: node=%s symbol=%s %s", node_name, symbol, detail
    )
    deps.paid_call_meter[source] += 1


class AgentState(TypedDict, total=False):
    symbol: str
    previous_price: Decimal
    # FIN-14: moment zapisu referencji ceny i werdykt domeny o jej stęchłości
    # (`is_reference_stale`). Stęchła referencja zamyka bramkę volatility.
    reference_at: datetime | None
    stale_reference: bool
    current_price: Decimal
    delta: Decimal
    status: str

    # Opcjonalny obiekt Asset (przekazywany z zewnątrz, gdy dostępny asset_type)
    asset: Asset

    # Self-Reflection
    reflection_context: str

    # Zewnętrzne dane
    sentiment: dict[str, Any] | None
    news: list[dict[str, Any]]

    # Fundamenty i wycena
    fundamentals: Fundamentals | None
    valuation_verdict: ValuationVerdict

    # Predykcja
    llm_analysis: dict[str, Any]
    ml_target_price: Decimal
    # ML-14: wektor cech dokładnie w postaci podanej modelowi (`ML_FEATURE_COLUMNS`)
    # — trafia do `feature_snapshots`, żeby trening nie odtwarzał go widokiem SQL.
    ml_features: dict[str, float]
    # G03: True = XGBoost nie ma wag, a `ml_target_price` to baseline "cena bez
    # zmian", nie prognoza modelu. Rada i raport nie mogą go podawać jako sygnału.
    ml_cold_start: bool
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
    if correct:
        outcome = "prognoza trafiła"
    elif correct is False:
        outcome = "prognoza chybiła"
    else:
        outcome = "wynik nieznany"
    insight = str(item.get("correction_insights") or "").strip()
    line = f"  - [{trend}, {outcome}] {summary}"
    # WIA-10: placeholder reflect_node (np. historyczne „diagnoza odłożona”)
    # nie jest wnioskiem — analog zostaje, ale bez fałszywej lekcji.
    if is_substantive_insight(insight):
        line += f" → wniosek: {insight}"
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
) -> tuple[str, list[dict[str, Any]]]:
    """Buduje blok 'podobne historyczne sytuacje' do promptu (RAG) ORAZ zwraca
    rerankowane rekordy do reużycia jako precedent receipts (Q5).

    Zwraca `(similar_context, precedents)`:
      - `similar_context` — tekstowy blok wstrzykiwany do promptu (jak dotąd),
      - `precedents` — zwięzła lista top-N analogów `{summary, predicted_trend,
        is_trend_correct}` dla raportu ("dlaczego ta decyzja").

    W pełni graceful: brak RPC / pgvector / błąd → `("", [])` (predykcja działa
    bez RAG i bez receipts). Jeden retrieval + jeden rerank — zero dodatkowych
    płatnych wywołań ponad to, co RAG już robił. `outcome_weight` (#9): rerank
    analogów po realnym wyniku; 0.0 = sama kolejność RPC.
    """
    try:
        similar = repository_port.find_similar_predictions(
            embedding, limit=3, regime=regime
        )
        similar = rerank_analogs(list(similar or []), outcome_weight=outcome_weight)
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
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


_TREND_TO_REC: dict[str, Literal["BUY", "SELL", "HOLD"]] = {
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


def _build_fetch_fundamentals_node(
    port: FundamentalsPort,
) -> Callable[[AgentState], dict[str, Any]]:
    """Buduje węzeł grafu z domknięciem nad portem (DI-friendly i testowalny).

    Pozycja w grafie: po reflect_node, PRZED bramką volatility.
    Czytanie cache jest darmowe; płatne wywołania AV dzieją się tylko
    w slow loop (fast loop dostaje NullFundamentalsAdapter jako delegate).
    """

    def fetch_fundamentals_node(state: AgentState) -> dict[str, Any]:
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
    # ML-34: okna kalibracyjne przedziału ruchu per klasa aktywa (rozliczone
    # predykcje, rosnąco po czasie), czytane RAZ na cykl w DI. None = bez przedziału.
    move_calibration: Mapping[AssetType, Sequence[CalibrationPoint]] | None = None
    # ML-34: kolumny `interval_low/high` (migracja 032). Off = klucza nie ma
    # w rekordzie — przed migracją dałby PGRST204.
    interval_columns_enabled: bool = False
    # EXP-01: delta, która otworzyła bramkę, w `gate_delta` (migracja 034). Off =
    # klucza nie ma w rekordzie — przed migracją dałby PGRST204.
    gate_delta_enabled: bool = False
    options_port: OptionsFlowPort | None = None
    attestation_publisher: AttestationPublisherPort | None = None
    # Licznik płatnych wywołań CYKLU (współdzielony między symbolami). Pole jest
    # NIE-opcjonalne z własnym licznikiem per egzemplarz deps: wołający, który go
    # nie poda, i tak dostaje działający licznik pod `use_case.paid_calls`, więc
    # nikt nie musi doszywać go przez `dataclasses.replace` po fakcie.
    paid_call_meter: Counter[str] = dataclasses.field(default_factory=Counter)


def _check_price_node(deps: AgentGraphDeps, state: AgentState) -> dict[str, Any]:
    market_port = deps.market_port
    current = market_port.get_current_price(state["symbol"])
    previous = Money(state["previous_price"])
    delta = PriceDelta.calculate(previous, current)
    # #6: snapshot ceny NIE jest już zapisywany tutaj. Zapis przeniesiony do
    # węzłów TERMINALNYCH (save_node / ignore_node), żeby cykl był idempotentny:
    # cykl, który padł w połowie, nie zostawia snapshotu, więc retry reużyje
    # referencji z poprzedniego UKOŃCZONEGO cyklu (zamiast porównywać się do
    # świeżego snapshotu padłej próby → delta≈0 → cichy "ignore").
    return {
        "current_price": current.amount,
        "delta": delta.percentage,
    }



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
    cisza nie może udawać neutralnego rynku w raporcie selekcji."""
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
    return {name: raw[name] for name in FREE_SNAPSHOT_FEATURES}


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
            feature_set_version=FEATURE_SET_VERSION,
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
    if deps.recipe is not None:
        receipt["schema_version"] = RECEIPT_SCHEMA_V2
        receipt.update(deps.recipe.receipt_fields())
    return receipt


def _pick_threshold(
    state: AgentState,
    default: Threshold,
    crypto_override: Threshold | None,
) -> Threshold:
    """Wybiera próg per asset_type. CRYPTO ma osobną zmienność (3-5%
    dziennie natywnie), więc bramka 2% dla akcji byłaby bezsensowna."""
    asset = state.get("asset")
    if (
        asset is not None
        and asset.asset_type is AssetType.CRYPTO
        and crypto_override is not None
    ):
        base = crypto_override
    else:
        base = default
    # #7: reżim rynku może tylko ZACIEŚNIĆ próg (mnożnik ≥ 1.0 gwarantowany
    # przez RegimeDetector) — w RISK-OFF mniej cykli odpala płatne porty.
    # #6: bramka earnings działa tak samo i MNOŻY się z reżimem — predykcja 12h
    # tuż przed raportem kwartalnym to rzut monetą, więc zaostrzamy próg zamiast
    # palić tokeny na analizę, którą zaora raport. Oba mnożniki są >= 1.0
    # (kontrakt domenowy), więc iloczyn też — bramka nigdy się nie ROZLUŹNIA.
    # `max(..., 1.0)` to pas bezpieczeństwa, gdyby kiedyś przeciekła wartość < 1.
    regime_mult = state.get("regime_multiplier", 1.0) or 1.0
    earnings_mult = state.get("earnings_multiplier", 1.0) or 1.0
    mult = max(regime_mult * earnings_mult, 1.0)
    if not math.isclose(mult, 1.0):
        return Threshold(base.value * Decimal(str(mult)))
    return base



def _reference_is_stale(state: AgentState) -> bool:
    """FIN-14: werdykt domeny policzony przez use case przy odczycie referencji.
    Brak klucza (graf wołany bez use case'u) = referencja świeża, jak przed FIN-14."""
    return bool(state.get("stale_reference", False))


def _should_analyze(deps: AgentGraphDeps, state: AgentState) -> str:
    threshold = deps.threshold
    crypto_threshold = deps.crypto_threshold
    asset = state.get("asset") or Asset(symbol=state["symbol"])
    delta = PriceDelta(state["delta"])
    chosen = _pick_threshold(state, threshold, crypto_threshold)
    stale = _reference_is_stale(state)
    passes = asset.evaluate_volatility(delta, chosen, reference_stale=stale)
    decision = "analyze" if passes else "ignore"
    # #13: decyzja bramki volatility musi być widoczna w logach — bez tego
    # nie dało się odróżnić zamierzonego "ignore" (FinOps) od cichego buga.
    logger.info(
        "volatility gate for %s: |Δ|=%.4f threshold=%.4f asset_type=%s "
        "stale_reference=%s → %s",
        state["symbol"],
        abs(float(state["delta"])),
        float(chosen.value),
        asset.asset_type.value,
        stale,
        decision,
    )
    return "analyze_sentiment" if passes else "ignore"



def _reflect_node(deps: AgentGraphDeps, state: AgentState) -> dict[str, Any]:
    repository_port = deps.repository_port
    reflection_min_age_hours = deps.reflection_min_age_hours
    threshold = deps.threshold
    crypto_threshold = deps.crypto_threshold
    llm_port = deps.llm_port
    symbol = state["symbol"]
    current_price = state["current_price"]
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

    # Domena liczy accuracy_score (0.0-1.0) — bliskość ceny do celu,
    # to napędza trening XGBoost. is_trend_correct (zgodność kierunku)
    # jest niezależną miarą i napędza trafność raportu.
    accuracy = float(last.accuracy_score(current_price))
    # Pasmo SIDEWAYS zależy od klasy aktywa: ±0.5% to sensowny "brak ruchu" dla
    # akcji, ale przy dziennej zmienności BTC 3–5% uznawałoby niemal każdy dzień
    # za ruch i systematycznie zaniżało trafność krypto. Korekta POMIARU.
    asset = state.get("asset")
    tolerance = sideways_tolerance_for(asset.asset_type if asset is not None else None)
    trend_correct = last.is_trend_correct(current_price, tolerance=tolerance)

    if trend_correct:
        repository_port.update_prediction_accuracy(
            prediction_id=last.id,
            actual_price=current_price,
            accuracy_score=accuracy,
            is_trend_correct=trend_correct,
            insight=CORRECT_PREDICTION_INSIGHT,
        )
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
    gate_asset = state.get("asset") or Asset(symbol=symbol)
    gate_delta = PriceDelta(state["delta"])
    gate_threshold = _pick_threshold(state, threshold, crypto_threshold)
    # FIN-14: ta sama bramka — ruch od stęchłej referencji nie płaci za diagnozę.
    diagnose_paid = gate_asset.evaluate_volatility(
        gate_delta, gate_threshold, reference_stale=_reference_is_stale(state)
    )

    if diagnose_paid:
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
        with _timed_node("reflect_analyze_mistake", symbol):
            insight = llm_port.analyze_mistake(prompt)
    else:
        # Niska zmienność — nie palimy budżetu LLM na diagnozę. Predykcja
        # i tak zostaje zamknięta z accuracy/trend, więc diagnoza jest
        # POMINIĘTA, nie odłożona: nic jej później nie ponawia.
        insight = DIAGNOSIS_SKIPPED_INSIGHT

    repository_port.update_prediction_accuracy(
        prediction_id=last.id,
        actual_price=current_price,
        accuracy_score=accuracy,
        is_trend_correct=trend_correct,
        insight=insight,
    )
    return {"reflection_context": f"Ostatni błąd: {insight}"}



def _analyze_sentiment_node(deps: AgentGraphDeps, state: AgentState) -> dict[str, Any]:
    sentiment_port = deps.sentiment_port
    symbol = state["symbol"]
    # #14: węzeł płatny (Alpha Vantage NEWS_SENTIMENT) — instrumentujemy czas
    # i logujemy jawnie fakt płatnego wywołania.
    _log_paid_call(
        deps, "analyze_sentiment", symbol, "sentiment", "sentiment_port.get_social_score"
    )
    with _timed_node("analyze_sentiment", symbol):
        return {"sentiment": sentiment_port.get_social_score(symbol)}



def _fetch_news_node(deps: AgentGraphDeps, state: AgentState) -> dict[str, Any]:
    news_port = deps.news_port
    symbol = state["symbol"]
    # #14: węzeł płatny (Alpha Vantage NEWS feed) — instrumentacja jak wyżej.
    _log_paid_call(deps, "fetch_news", symbol, "news", "news_port.get_news_context")
    with _timed_node("fetch_news", symbol):
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
        try:
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
            )
    return news_embedding, similar_context, similar_precedents


def _run_llm_analysis(
    deps: AgentGraphDeps, state: AgentState, prompt: str, symbol: str
) -> tuple[dict[str, Any], bool]:
    """Płatna analiza jakościowa LLM. #6: dla NAJWIĘKSZYCH ruchów (osobny próg
    ``tool_use_threshold``) idzie kosztowna pętla tool-use; inaczej zwykły
    ``llm_port.analyze``. #12-guard: błąd parsowania/API degradujemy do
    neutralnego fallbacku. Zwraca ``(llm_analysis, llm_failed)``."""
    try:
        if (
            deps.tool_use_port is not None
            and deps.research_tools
            and deps.tool_use_threshold is not None
            and (state.get("asset") or Asset(symbol=symbol)).evaluate_volatility(
                PriceDelta(state["delta"]), deps.tool_use_threshold
            )
        ):
            _log_paid_call(
                deps, "predict", symbol, "llm", "tool_use_port.analyze_with_tools"
            )
            with _timed_node("predict_tool_use", symbol):
                return deps.tool_use_port.analyze_with_tools(
                    prompt, deps.research_tools
                ), False
        _log_paid_call(deps, "predict", symbol, "llm", "llm_port.analyze")
        return deps.llm_port.analyze(prompt), False
    except Exception:
        logger.exception(
            "llm_port.analyze failed for %s — degraduję do neutralnej analizy",
            state["symbol"],
        )
        return {
            "trend_direction": "SIDEWAYS",
            "confidence_score": 0.5,
            "av_agreement": 0.5,
            "reasoning": "Analiza LLM niedostępna — neutralny fallback.",
        }, True


def _run_ml_prediction(
    ml_port: MLPredictionPort,
    features: dict[str, float],
    symbol: str,
    current_price: Decimal,
) -> tuple[Decimal, tuple[FeatureContribution, ...]]:
    """Twarda predykcja liczbowa ML + atrybucja cech (SHAP-lite, lokalny CPU).
    Cold-start (model nietrenowany) → baseline 'cena bez zmian', brak atrybucji.
    Atrybucja jest graceful (błąd explain → sama predykcja). Zwraca
    ``(ml_target, feature_attribution)``."""
    if not ml_port.is_trained:
        return current_price, ()
    # Model przewiduje ZWROT 12h; predict() rekonstruuje cenę bezwzględną
    # z bieżącej ceny (cena*(1+zwrot)).
    ml_target = ml_port.predict(features, current_price=current_price).amount
    feature_attribution: tuple[FeatureContribution, ...] = ()
    try:
        contribs = {
            k: v for k, v in ml_port.explain(features).items() if k != "_bias"
        }
        feature_attribution = top_contributions(contribs)
    except Exception:
        logger.exception("explain() failed for %s — bez atrybucji", symbol)
    return ml_target, feature_attribution


def _attach_commitment(
    deps: AgentGraphDeps, record: dict[str, Any]
) -> None:
    """#16 — dokleja SHA-256 commitment do rekordu i publikuje go (bez soli).

    Commit-reveal: publikujemy sam HASH treści predykcji + losowej soli.
    Dopóki sól nie zostanie ujawniona (sweep po rozliczeniu), hash jest
    nieodwracalny — sceptyk nie pozna predykcji, ale po ujawnieniu zweryfikuje,
    że powstała PRZED ruchem ceny.

    UCZCIWOŚĆ: daty commitów Gita są fałszowalne. Claim to „tamper-evidence
    przy branch protection", NIE „niepodrabialny timestamp".

    Klucze dokładane TYLKO gdy publisher istnieje — nowe kolumny przed migracją
    024 dałyby PGRST204 i zabiły zapis całej predykcji (gate jak `receipts_enabled`).
    Awaria publikacji nigdy nie wywala cyklu.
    """
    if deps.attestation_publisher is None:
        return
    payload = {
        "symbol": record["symbol"],
        "predicted_trend": record.get("predicted_trend"),
        "predicted_target_price": str(record.get("predicted_target_price")),
        "price_at_prediction": str(record.get("price_at_prediction")),
    }
    salt = new_salt()
    commitment = attestation_commit(payload, salt)
    record["commitment_hash"] = commitment
    record["commitment_salt"] = salt
    try:
        # Publikujemy WYŁĄCZNIE hash — sól zostaje w bazie do czasu revealu.
        deps.attestation_publisher.publish_commitment(
            {"symbol": record["symbol"], "commitment": commitment}
        )
    except Exception:
        logger.exception("Attestation commitment publish failed for %s", record["symbol"])


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
    """Ósma cecha ML. Brak fuzji → 0.0, dokładnie jak `COALESCE` w widoku."""
    return fusion.score if fusion is not None else 0.0


# #18: horyzont predykcji. Nazwy `*_12h` są historyczne — pętla chodzi raz na
# dobę handlową, ale kontrakt predykcji (i kolumna `actual_price_after_12h`)
# pozostaje 12-godzinny, więc taki horyzont skalujemy sqrt-time.
_EDGE_HORIZON_HOURS = 12.0


def _resolve_implied_edge(
    deps: AgentGraphDeps,
    symbol: str,
    current_price: Decimal,
    ml_target: Decimal,
    *,
    ml_cold_start: bool,
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
    return evaluate_edge(move, iv, _EDGE_HORIZON_HOURS)


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


def _interval_price(price: Decimal, move_return: float) -> Decimal:
    """Granica przedziału jako cena (skala `predicted_target_price`, 4 miejsca)."""
    return (price * (Decimal(1) + Decimal(str(move_return)))).quantize(Decimal("0.0001"))


def _predict_node(deps: AgentGraphDeps, state: AgentState) -> dict[str, Any]:
    repository_port = deps.repository_port
    ml_port = deps.ml_port
    symbol = state["symbol"]
    # #14: predict to najdroższy węzeł (LLM + ewentualny embedding/RAG).
    # Mierzymy całkowity czas węzła; perf_counter zamknięty tuż przed return.
    predict_start = time.perf_counter()
    sentiment = state.get("sentiment") or {}
    news = state.get("news", [])
    news_summary = _summarize_news(news)

    news_embedding, similar_context, similar_precedents = _compute_rag_context(
        deps, state, news_summary
    )

    # 1. LLM — analiza jakościowa z cross-validation pre-computed AV sentymentu
    prompt = get_prediction_prompt(
        symbol=state["symbol"],
        current_data={
            "price": str(state["current_price"]),
            "delta_percentage": str(state["delta"] * Decimal("100")),
            "av_sentiment_score": sentiment.get("av_sentiment_score"),
            "av_sentiment_label": sentiment.get("av_sentiment_label"),
            "news_volume_24h": sentiment.get("news_volume_24h"),
            "high_relevance_count": sentiment.get("high_relevance_count"),
            "news_summary": news_summary,
            "news_titles": _news_titles(news, _NEWS_TITLES_LIMIT),
        },
        reflection_context=state.get("reflection_context", ""),
        similar_context=similar_context,
        # #7 reżim + #5 contagion (oba z bieżącego cyklu, przekazane w state).
        regime_context=state.get("regime_context", ""),
        peer_context=", ".join(
            f"{sym}:{stance}" for sym, stance in state.get("peer_context", ())
        ),
        # #14: pusty string przy braku fuzji → blok nie wchodzi do promptu.
        alpha_fusion_context=_format_alpha_fusion(state.get("alpha_fusion")),
    )
    raw_analysis, llm_failed = _run_llm_analysis(deps, state, prompt, symbol)
    # ML-07: kanonizacja PRZED pierwszym odczytem — rada, zapis i raport
    # czytają już ten sam znormalizowany dict, nie surowy tekst modelu.
    llm_analysis, analysis_flags = _normalize_llm_analysis(raw_analysis)

    # 2. ML — twarda predykcja liczbowa (multi-feature)
    # #21: trend_direction walidowany jak pola liczbowe — present-but-invalid
    # (np. "UP") daje neutralny sygnał, ALE zostawia flagę jakości (z
    # normalizacji). Genuinely-absent default'uje do SIDEWAYS bez flagi.
    llm_trend_signal = _TREND_SIGNALS[llm_analysis["trend_direction"]]

    # Walidacja pól liczbowych — każdy padły input zostawia ślad w
    # data_quality_flags (None → _missing, nie-numeryczne/NaN → _invalid,
    # poza skalą [0, 1] → _out_of_range).
    flags: list[str] = []
    if llm_failed:
        flags.append("llm_analysis_failed")
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
    current_price = state["current_price"]
    last_pred_price = repository_port.get_last_prediction_price(state["symbol"])
    if last_pred_price is not None and last_pred_price.amount != 0:
        ml_price_delta = float(
            (current_price - last_pred_price.amount) / last_pred_price.amount
        )
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
    ml_target, feature_attribution = _run_ml_prediction(
        ml_port, features, symbol, current_price
    )

    # #14: pojedyncze wyjście węzła — logujemy całkowity czas predict.
    elapsed_ms = (time.perf_counter() - predict_start) * 1000.0
    logger.info("node=predict symbol=%s elapsed_ms=%.1f", symbol, elapsed_ms)

    implied_edge = _resolve_implied_edge(
        deps, symbol, current_price, ml_target, ml_cold_start=ml_cold_start
    )
    interval = _resolve_move_interval(
        deps, state, current_price, ml_target, ml_cold_start=ml_cold_start
    )

    return {
        "llm_analysis": llm_analysis,
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



def _council_node(deps: AgentGraphDeps, state: AgentState) -> dict[str, Any]:
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
        asset = state.get("asset") or Asset(symbol=state["symbol"])
        delta = PriceDelta(state["delta"])
        chosen = _pick_threshold(
            state, council_threshold, crypto_council_threshold
        )
        if not asset.evaluate_volatility(delta, chosen):
            logger.info(
                "council_node skipped for %s — |Δ|=%.4f < threshold=%.4f",
                state["symbol"],
                abs(float(state["delta"])),
                float(chosen.value),
            )
            return {"council_verdict": None}
    sentiment = state.get("sentiment") or {}
    news = state.get("news", [])
    llm_analysis = state.get("llm_analysis") or {}
    data = CouncilInput(
        symbol=state["symbol"],
        current_price=state["current_price"],
        price_delta_pct=state["delta"] * Decimal("100"),
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
    )
    # #14: rada to N równoległych wywołań LLM — najdroższy płatny węzeł.
    # Jawny log płatnego wywołania + timing (mierzony też przy błędzie).
    _log_paid_call(
        deps, "council", state["symbol"], "council_llm", "council_port.analyze"
    )
    try:
        with _timed_node("council", state["symbol"]):
            verdict = council_port.analyze(state["symbol"], data)
    except Exception:
        logger.exception(
            "council_node failed for %s — continuing without verdict",
            state["symbol"],
        )
        verdict = None
    return {"council_verdict": verdict}



def _resolve_news_vector(
    deps: AgentGraphDeps, state: AgentState, news_summary: str
) -> list[float] | None:
    """Wektor newsów do zapisu w pgvector. predict_node zwykle już go policzył
    (RAG) — reużywamy; fallback: embedujemy tu, jeśli predict go nie dostarczył.
    Graceful: brak portu / pusty tekst / błąd API → None (zapis bez wektora)."""
    if deps.embedding_port is None or not news_summary:
        return None
    vector = state.get("news_embedding")
    if vector:
        return vector
    try:
        return deps.embedding_port.embed(news_summary)
    except Exception:
        logger.exception(
            "Embedding failed for %s — saving without vector", state["symbol"]
        )
        return None


def _save_node(deps: AgentGraphDeps, state: AgentState) -> dict[str, Any]:
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
    council_verdict = state.get("council_verdict")
    council_verdict_dict = (
        dataclasses.asdict(council_verdict) if council_verdict is not None else None
    )

    record: dict[str, Any] = {
        "symbol": state["symbol"],
        "price_at_prediction": state["current_price"],
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
        "predicted_target_price": state.get("ml_target_price"),
        # #4: persystujemy deklarowaną pewność — bez niej Slow-Loopowy sędzia
        # nie oceni kalibracji (migracja 016).
        "confidence_score": _float_or_default(
            llm_analysis.get("confidence_score"), 0.5
        ),
        "reasoning_text": llm_analysis.get("reasoning"),
        "council_verdict": council_verdict_dict,
        "data_quality_flags": data_quality_flags,
        # #14: ósma cecha ML (migracja 022). Widok robi COALESCE(...,0.0), więc
        # historyczne NULL-e nie wypadają przez `dropna` (train/serve-skew).
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
        record["feature_set_version"] = FEATURE_SET_VERSION

    # WIA-20: wersja przepisu — track record segmentuje się po niej.
    if deps.recipe_version_enabled and deps.recipe is not None:
        record["recipe_version"] = deps.recipe.version

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

    # #16: commit-reveal (migracja 024).
    _attach_commitment(deps, record)

    # Embedding podsumowania newsów → pgvector (reużyty z predict_node lub
    # policzony tu w fallbacku; graceful — przy braku zapisujemy bez wektora).
    vector = _resolve_news_vector(deps, state, news_summary)
    if vector:
        record["embedding"] = vector

    prediction_id = repository_port.save_prediction(record)

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
        except Exception:
            logger.exception(
                "save_council_votes failed for %s — JSONB blob in "
                "prediction_logs is still saved",
                state["symbol"],
            )

    # #6: węzeł terminalny — zapisujemy snapshot ceny dla następnego cyklu.
    _persist_price_snapshot(deps, state)
    _persist_feature_snapshot(deps, state, gated=True)

    return {"prediction_id": prediction_id, "status": "saved"}



def _ignore_node(deps: AgentGraphDeps, state: AgentState) -> dict[str, Any]:
    # #6: węzeł terminalny — zapisujemy snapshot ceny dla następnego cyklu.
    _persist_price_snapshot(deps, state)
    # FIN-14 (decyzja 2026-10-19): cykl re-baseline nie jest obserwacją cyklu —
    # `snapshot_delta` od stęchłej referencji to ruch z wielu dni, więc wektor
    # ML-14 nie powstaje. Zapisujemy tylko nowy punkt odniesienia.
    if not _reference_is_stale(state):
        _persist_feature_snapshot(deps, state, gated=False)
    return {"status": "ignored"}



def create_agent_graph(deps: AgentGraphDeps) -> StateGraph[AgentState]:
    """Fabryka grafu LangGraph — Fast Loop kompletny.

    Topologia bez fundamentals_port:
        check_price → reflect → [delta ≥ threshold?]
                                  ├── nie → ignore → END
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
            },
        )

    workflow.add_edge("analyze_sentiment", "fetch_news")
    workflow.add_edge("fetch_news", "predict")
    workflow.add_edge("predict", "council")
    workflow.add_edge("council", "save")
    workflow.add_edge("save", END)
    workflow.add_edge("ignore", END)

    return workflow

