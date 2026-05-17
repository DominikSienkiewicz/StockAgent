from __future__ import annotations

import dataclasses
import logging
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

    # Persystencja
    prediction_id: str | None

    # Rada doradcza
    council_verdict: CouncilVerdict | None


def _summarize_news(news: list[dict[str, Any]]) -> str:
    """Konkatenacja tytułów najnowszych newsów w jednej linijce (dla promptu)."""
    titles = [item.get("title", "") for item in news[:5]]
    return " | ".join(t for t in titles if t)


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    fundamentals_port: FundamentalsPort | None = None,
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

    def should_analyze(state: AgentState) -> str:
        asset = Asset(symbol=state["symbol"])
        delta = PriceDelta(state["delta"])
        if asset.evaluate_volatility(delta, threshold):
            return "analyze_sentiment"
        return "ignore"

    def reflect_node(state: AgentState) -> dict[str, Any]:
        symbol = state["symbol"]
        current_price = state["current_price"]
        last = repository_port.get_unverified_prediction(symbol)

        if last is None:
            return {"reflection_context": "Brak danych historycznych do oceny."}

        # Predykcje odczytane z repozytorium muszą mieć id (kontrakt RepositoryPort).
        assert last.id is not None, "Prediction from repository must have id"

        # Domena liczy accuracy_score (0.0-1.0) — zapis zamyka pętlę feedback:
        # to ten wskaźnik napędza get_accuracy_stats() i trening XGBoost.
        accuracy = float(last.accuracy_score(current_price))

        if not last.is_trend_correct(current_price):
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
                insight=insight,
            )
            return {"reflection_context": f"Ostatni błąd: {insight}"}

        repository_port.update_prediction_accuracy(
            prediction_id=last.id,
            actual_price=current_price,
            accuracy_score=accuracy,
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
        )
        llm_analysis = llm_port.analyze(prompt)

        # 2. ML — twarda predykcja liczbowa (multi-feature)
        llm_trend_signal = {"BULLISH": 1, "BEARISH": -1, "SIDEWAYS": 0}.get(
            llm_analysis.get("trend_direction", "SIDEWAYS"), 0
        )
        # Cross-validation: czy LLM zgodził się z AV (Phase B)
        av_llm_agreement = _float_or_default(llm_analysis.get("av_agreement"), 0.5)

        raw_features = {
            "price_delta": float(state["delta"]),
            "av_sentiment_score": _float_or_default(
                sentiment.get("av_sentiment_score"), 0.0
            ),
            "av_relevance_avg": _float_or_default(
                sentiment.get("av_relevance_avg"), 0.0
            ),
            "news_volume_24h": _float_or_default(
                sentiment.get("news_volume_24h"), 0.0
            ),
            "high_relevance_count": _float_or_default(
                sentiment.get("high_relevance_count"), 0.0
            ),
            "llm_trend_signal": float(llm_trend_signal),
            "av_llm_agreement": av_llm_agreement,
        }
        features = {name: raw_features[name] for name in ML_FEATURE_COLUMNS}
        # Cold-start guard: dopóki XGBoost nie ma wytrenowanych wag (pierwsze
        # tygodnie działania, przed pierwszym Slow Loop), używamy baseline
        # "cena bez zmian". Agent działa, LLM nadal daje trend; gdy model się
        # wytrenuje, automatycznie przejmuje predykcję liczbową.
        if ml_port.is_trained:
            ml_target = ml_port.predict(features).amount
        else:
            ml_target = state["current_price"]

        return {
            "llm_analysis": llm_analysis,
            "ml_target_price": ml_target,
        }

    def council_node(state: AgentState) -> dict[str, Any]:
        if council_port is None:
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
        }

        # Embedding podsumowania newsów → pgvector. Opcjonalne (graceful):
        # gdy brak portu lub pusty tekst / błąd API, zapisujemy bez wektora.
        if embedding_port is not None and news_summary:
            try:
                vector = embedding_port.embed(news_summary)
                if vector:
                    record["embedding"] = vector
            except Exception:
                logger.exception("Embedding failed for %s — saving without vector", state["symbol"])

        prediction_id = repository_port.save_prediction(record)
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
