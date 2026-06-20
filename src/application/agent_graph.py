from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable
from decimal import Decimal
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.application.ml_features import ML_FEATURE_COLUMNS
from src.application.ports import (
    AdvisoryCouncilPort,
    EmbeddingPort,
    FundamentalsPort,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.application.prompts import get_mistake_diagnosis_prompt, get_prediction_prompt
from src.domain.asset import Asset, PriceDelta
from src.domain.council import CouncilInput, CouncilVerdict
from src.domain.value_objects import AssetType, Fundamentals, Money, Threshold, ValuationVerdict

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    symbol: str
    previous_price: Decimal
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

    # Embedding podsumowania newsów — liczony raz w predict_node (RAG retrieval)
    # i reużyty w save_node, żeby nie embedować dwa razy na cykl.
    news_embedding: list[float] | None

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


def _summarize_news(news: list[dict[str, Any]]) -> str:
    """Konkatenacja tytułów najnowszych newsów w jednej linijce (dla promptu)."""
    titles = [item.get("title", "") for item in news[:5]]
    return " | ".join(t for t in titles if t)


def _build_similar_context(
    repository_port: RepositoryPort,
    embedding: list[float],
    symbol: str,
) -> str:
    """Buduje tekstowy blok 'podobne historyczne sytuacje' do promptu (RAG).

    W pełni graceful: brak RPC / pgvector / błąd → pusty string (predykcja
    działa bez RAG). Każdy rekord skracamy do: kierunek, czy trafiony, wniosek.
    """
    try:
        similar = repository_port.find_similar_predictions(embedding, limit=3)
        lines: list[str] = []
        for item in similar or []:
            summary = str(item.get("news_summary") or "").strip()
            if not summary:
                continue
            trend = item.get("predicted_trend") or "?"
            correct = item.get("is_trend_correct")
            outcome = (
                "prognoza trafiła" if correct
                else "prognoza chybiła" if correct is False
                else "wynik nieznany"
            )
            insight = str(item.get("correction_insights") or "").strip()
            line = f"  - [{trend}, {outcome}] {summary}"
            if insight:
                line += f" → wniosek: {insight}"
            lines.append(line)
        return "\n".join(lines)
    except Exception:
        logger.exception(
            "find_similar_predictions failed for %s — predicting without RAG", symbol
        )
        return ""


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


def create_agent_graph(
    *,
    market_port: MarketDataPort,
    sentiment_port: SentimentPort,
    news_port: NewsPort,
    repository_port: RepositoryPort,
    ml_port: MLPredictionPort,
    llm_port: LLMPort,
    threshold: Threshold,
    embedding_port: EmbeddingPort | None = None,
    council_port: AdvisoryCouncilPort | None = None,
    council_threshold: Threshold | None = None,
    fundamentals_port: FundamentalsPort | None = None,
    crypto_threshold: Threshold | None = None,
    crypto_council_threshold: Threshold | None = None,
    reflection_min_age_hours: int = 0,
) -> StateGraph[AgentState]:
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

    def check_price_node(state: AgentState) -> dict[str, Any]:
        current = market_port.get_current_price(state["symbol"])
        previous = Money(state["previous_price"])
        delta = PriceDelta.calculate(previous, current)
        # Snapshot ceny w KAŻDYM cyklu — następny cykl użyje go jako punktu
        # odniesienia (rozwiązuje cold-start deadlock).
        repository_port.save_price_snapshot(state["symbol"], current)
        return {
            "current_price": current.amount,
            "delta": delta.percentage,
        }

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
            return crypto_override
        return default

    def should_analyze(state: AgentState) -> str:
        asset = state.get("asset") or Asset(symbol=state["symbol"])
        delta = PriceDelta(state["delta"])
        chosen = _pick_threshold(state, threshold, crypto_threshold)
        if asset.evaluate_volatility(delta, chosen):
            return "analyze_sentiment"
        return "ignore"

    def reflect_node(state: AgentState) -> dict[str, Any]:
        symbol = state["symbol"]
        current_price = state["current_price"]
        # min_age_hours odsiewa predykcje zbyt świeże, by je rzetelnie ocenić
        # (nakładające się cykle: manualny dispatch + scheduled). 0 = bez filtra.
        last = repository_port.get_unverified_prediction(
            symbol, min_age_hours=reflection_min_age_hours
        )

        if last is None:
            return {"reflection_context": "Brak danych historycznych do oceny."}

        # Predykcje odczytane z repozytorium muszą mieć id (kontrakt RepositoryPort).
        assert last.id is not None, "Prediction from repository must have id"

        # Domena liczy accuracy_score (0.0-1.0) — bliskość ceny do celu,
        # to napędza trening XGBoost. is_trend_correct (zgodność kierunku)
        # jest niezależną miarą i napędza trafność raportu.
        accuracy = float(last.accuracy_score(current_price))
        trend_correct = last.is_trend_correct(current_price)

        if not trend_correct:
            prompt = get_mistake_diagnosis_prompt(
                last_trend=str(last.predicted_trend),
                last_news_summary="(zapisany w prediction_logs)",
                actual_price=str(current_price),
            )
            insight = llm_port.analyze_mistake(prompt)
            repository_port.update_prediction_accuracy(
                prediction_id=last.id,
                actual_price=current_price,
                accuracy_score=accuracy,
                is_trend_correct=trend_correct,
                insight=insight,
            )
            return {"reflection_context": f"Ostatni błąd: {insight}"}

        repository_port.update_prediction_accuracy(
            prediction_id=last.id,
            actual_price=current_price,
            accuracy_score=accuracy,
            is_trend_correct=trend_correct,
            insight="Trafiona predykcja.",
        )
        return {"reflection_context": "Ostatnie prognozy były trafne. Kontynuuj strategię."}

    def analyze_sentiment_node(state: AgentState) -> dict[str, Any]:
        return {"sentiment": sentiment_port.get_social_score(state["symbol"])}

    def fetch_news_node(state: AgentState) -> dict[str, Any]:
        return {"news": news_port.get_news_context(state["symbol"])}

    def predict_node(state: AgentState) -> dict[str, Any]:
        sentiment = state.get("sentiment") or {}
        news = state.get("news", [])
        news_summary = _summarize_news(news)

        # RAG: embedujemy podsumowanie newsów RAZ (reużyte w save_node) i
        # wyszukujemy podobne historyczne sytuacje, by wstrzyknąć je do promptu
        # ("jak skończyły się analogiczne układy"). W pełni graceful — błąd
        # embeddingu/retrievalu nie blokuje predykcji.
        news_embedding: list[float] | None = None
        similar_context = ""
        if embedding_port is not None and news_summary:
            try:
                news_embedding = embedding_port.embed(news_summary)
            except Exception:
                logger.exception(
                    "Embedding failed for %s — predicting without RAG", state["symbol"]
                )
            if news_embedding:
                similar_context = _build_similar_context(
                    repository_port, news_embedding, state["symbol"]
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
            },
            reflection_context=state.get("reflection_context", ""),
            similar_context=similar_context,
        )
        llm_analysis = llm_port.analyze(prompt)

        # 2. ML — twarda predykcja liczbowa (multi-feature)
        llm_trend_signal = {"BULLISH": 1, "BEARISH": -1, "SIDEWAYS": 0}.get(
            llm_analysis.get("trend_direction", "SIDEWAYS"), 0
        )

        # Walidacja pól liczbowych — każdy padły input zostawia ślad w
        # data_quality_flags (None → _missing, nie-numeryczne/NaN → _invalid).
        flags: list[str] = []
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
        # Cold-start guard: dopóki XGBoost nie ma wytrenowanych wag (pierwsze
        # tygodnie działania, przed pierwszym Slow Loop), używamy baseline
        # "cena bez zmian". Agent działa, LLM nadal daje trend; gdy model się
        # wytrenuje, automatycznie przejmuje predykcję liczbową.
        if ml_port.is_trained:
            # Model przewiduje ZWROT 12h; predict() rekonstruuje cenę bezwzględną
            # z bieżącej ceny (cena*(1+zwrot)).
            ml_target = ml_port.predict(features, current_price=current_price).amount
        else:
            ml_target = state["current_price"]

        return {
            "llm_analysis": llm_analysis,
            "ml_target_price": ml_target,
            "data_quality_flags": flags,
            # Reużywane w save_node — embedujemy newsy tylko raz na cykl.
            "news_embedding": news_embedding,
        }

    def council_node(state: AgentState) -> dict[str, Any]:
        if council_port is None:
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
            news_articles=[
                item.get("title", "")
                for item in news[:5]
                if item.get("title")
            ],
            llm_trend=str(llm_analysis.get("trend_direction", "SIDEWAYS")),
            llm_confidence=_float_or_default(llm_analysis.get("confidence_score"), 0.5),
            ml_price_target=state.get("ml_target_price") or state["current_price"],
            fundamentals=state.get("fundamentals"),
            valuation_verdict=state.get("valuation_verdict", ValuationVerdict.UNKNOWN),
        )
        try:
            verdict = council_port.analyze(state["symbol"], data)
        except Exception:
            logger.exception(
                "council_node failed for %s — continuing without verdict",
                state["symbol"],
            )
            verdict = None
        return {"council_verdict": verdict}

    def save_node(state: AgentState) -> dict[str, Any]:
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
            "reasoning_text": llm_analysis.get("reasoning"),
            "council_verdict": (
                dataclasses.asdict(state["council_verdict"])
                if state.get("council_verdict") is not None
                else None
            ),
            "data_quality_flags": data_quality_flags,
        }

        # Embedding podsumowania newsów → pgvector. Opcjonalne (graceful):
        # gdy brak portu lub pusty tekst / błąd API, zapisujemy bez wektora.
        # predict_node zwykle już policzył wektor (RAG) — reużywamy go, żeby
        # nie embedować dwa razy na cykl. Fallback: embedujemy tu, jeśli predict
        # nie dostarczył wektora (np. retrieval był pominięty).
        if embedding_port is not None and news_summary:
            vector = state.get("news_embedding")
            if not vector:
                try:
                    vector = embedding_port.embed(news_summary)
                except Exception:
                    logger.exception(
                        "Embedding failed for %s — saving without vector",
                        state["symbol"],
                    )
                    vector = None
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
                    votes=verdict.investor_opinions,
                )
            except Exception:
                logger.exception(
                    "save_council_votes failed for %s — JSONB blob in "
                    "prediction_logs is still saved",
                    state["symbol"],
                )

        return {"prediction_id": prediction_id, "status": "saved"}

    def ignore_node(_: AgentState) -> dict[str, Any]:
        return {"status": "ignored"}

    # ---------- Topologia ----------
    workflow = StateGraph[AgentState](state_schema=AgentState)
    workflow.add_node("check_price", check_price_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("fetch_news", fetch_news_node)
    workflow.add_node("predict", predict_node)
    workflow.add_node("council", council_node)
    workflow.add_node("save", save_node)
    workflow.add_node("ignore", ignore_node)

    workflow.set_entry_point("check_price")
    # check_price → reflect: bezwarunkowo. Ocena przeszłej predykcji odbywa się
    # w KAŻDYM cyklu, niezależnie od bieżącej zmienności.
    workflow.add_edge("check_price", "reflect")

    if fundamentals_port is not None:
        # reflect → fetch_fundamentals → bramka volatility.
        # Węzeł fundamentów jest PRZED bramką — czytanie cache jest darmowe.
        workflow.add_node(
            "fetch_fundamentals",
            _build_fetch_fundamentals_node(fundamentals_port),
        )
        workflow.add_edge("reflect", "fetch_fundamentals")
        # fetch_fundamentals → bramka volatility: dopiero TU decydujemy, czy robić
        # nową prognozę.
        workflow.add_conditional_edges(
            "fetch_fundamentals",
            should_analyze,
            {
                "analyze_sentiment": "analyze_sentiment",
                "ignore": "ignore",
            },
        )
    else:
        # Tryb kompatybilności wstecznej: reflect → bramka volatility bezpośrednio.
        workflow.add_conditional_edges(
            "reflect",
            should_analyze,
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
