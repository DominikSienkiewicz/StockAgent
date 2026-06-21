from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, Literal, TypedDict

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
    Tool,
    ToolUseLLMPort,
)
from src.application.prompts import get_mistake_diagnosis_prompt, get_prediction_prompt
from src.application.rag_rerank import rerank_analogs
from src.domain.asset import Asset, PriceDelta
from src.domain.council import CouncilInput, CouncilVerdict, vindicated_dissenters
from src.domain.feature_attribution import FeatureContribution, top_contributions
from src.domain.prediction import Prediction
from src.domain.regime_key import canonical_regime_key
from src.domain.value_objects import AssetType, Fundamentals, Money, Threshold, ValuationVerdict

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


def _log_paid_call(node_name: str, symbol: str, detail: str) -> None:
    """#14: jawna linia w logu, gdy odpala się PŁATNE wywołanie zewnętrzne
    (LLM / sentyment / news). Pozwala audytować FinOps — ile płatnych wywołań
    poszło w danym cyklu i z którego węzła."""
    logger.info(
        "paid call: node=%s symbol=%s %s", node_name, symbol, detail
    )


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

    # Q5: precedent receipts — zwięzła lista top-3 analogów RAG użytych w tym
    # cyklu (`{summary, predicted_trend, is_trend_correct}`). Render-only:
    # przepływa do SymbolResult i raportu, NIE jest persystowana (brak migracji).
    similar_precedents: list[dict[str, Any]]

    # Q3: atrybucja cech ML (SHAP-lite) — render-only do raportu.
    feature_attribution: tuple[FeatureContribution, ...]

    # #7 reżim rynku (label do promptu + mnożnik progu, ≥1.0) i #5 contagion
    # (werdykty skorelowanych spółek z tego cyklu) — przekazywane per symbol.
    regime_context: str
    regime_multiplier: float
    peer_context: tuple[tuple[str, str], ...]

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


# Q5: ile analogów wystawiamy jako "precedent receipts" w raporcie. Trzymamy
# zwięźle (top-3), żeby blok "Na podstawie analogów" był czytelny i mailowalny.
_PRECEDENT_RECEIPT_LIMIT = 3


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
            if len(precedents) < _PRECEDENT_RECEIPT_LIMIT:
                # Kompaktowy rekord do raportu — surowe pola (escapowanie robi
                # report_builder przez _html przy renderze).
                precedents.append(
                    {
                        "summary": summary,
                        "predicted_trend": str(trend),
                        "is_trend_correct": correct,
                    }
                )
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


def _validate_trend_direction(value: Any) -> tuple[int, str | None]:
    """Waliduje trend_direction z LLM przed mapowaniem na sygnał ML.

    Zwraca `(signal, flag_or_None)`. Genuinely-absent (None / pusty string)
    może bez flagi default'ować do SIDEWAYS — to nie jest skażony input, tylko
    brak pola. Wartość OBECNA, ale spoza zbioru {BULLISH, BEARISH, SIDEWAYS}
    (np. "UP", literówka) → neutralny sygnał 0 + flaga `trend_direction_invalid`
    (analogicznie do `_validate_float`). Bez tej flagi śmieciowy kierunek
    udawał prawdziwy SIDEWAYS i model uczył się na cichym zerze.
    """
    if value is None:
        return _TREND_SIGNALS["SIDEWAYS"], None
    normalized = str(value).strip().upper()
    if not normalized:
        return _TREND_SIGNALS["SIDEWAYS"], None
    if normalized in _TREND_SIGNALS:
        return _TREND_SIGNALS[normalized], None
    return _TREND_SIGNALS["SIDEWAYS"], "trend_direction_invalid"


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
    rag_outcome_weight: float = 0.0,
    tool_use_port: ToolUseLLMPort | None = None,
    research_tools: Sequence[Tool] = (),
    tool_use_threshold: Threshold | None = None,
    vector_memory_enabled: bool = False,
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
        # #6: snapshot ceny NIE jest już zapisywany tutaj. Zapis przeniesiony do
        # węzłów TERMINALNYCH (save_node / ignore_node), żeby cykl był idempotentny:
        # cykl, który padł w połowie, nie zostawia snapshotu, więc retry reużyje
        # referencji z poprzedniego UKOŃCZONEGO cyklu (zamiast porównywać się do
        # świeżego snapshotu padłej próby → delta≈0 → cichy "ignore").
        return {
            "current_price": current.amount,
            "delta": delta.percentage,
        }

    def _persist_price_snapshot(state: AgentState) -> None:
        """#6/#15: zapis snapshotu ceny — wołany wyłącznie z węzłów terminalnych.

        Cold-start nadal działa, bo KAŻDY ukończony cykl (saved albo ignored)
        zapisuje snapshot, więc następny cykl ma punkt odniesienia. Zapis jest
        owinięty w try/except (jak save_council_votes / embedding), żeby
        przejściowy błąd Supabase nie wywalił całego grafu.
        """
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
        mult = state.get("regime_multiplier", 1.0)
        if mult and mult != 1.0:
            return Threshold(base.value * Decimal(str(mult)))
        return base

    def should_analyze(state: AgentState) -> str:
        asset = state.get("asset") or Asset(symbol=state["symbol"])
        delta = PriceDelta(state["delta"])
        chosen = _pick_threshold(state, threshold, crypto_threshold)
        passes = asset.evaluate_volatility(delta, chosen)
        decision = "analyze" if passes else "ignore"
        # #13: decyzja bramki volatility musi być widoczna w logach — bez tego
        # nie dało się odróżnić zamierzonego "ignore" (FinOps) od cichego buga.
        logger.info(
            "volatility gate for %s: |Δ|=%.4f threshold=%.4f asset_type=%s → %s",
            state["symbol"],
            abs(float(state["delta"])),
            float(chosen.value),
            asset.asset_type.value,
            decision,
        )
        return "analyze_sentiment" if passes else "ignore"

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

        if trend_correct:
            repository_port.update_prediction_accuracy(
                prediction_id=last.id,
                actual_price=current_price,
                accuracy_score=accuracy,
                is_trend_correct=trend_correct,
                insight="Trafiona predykcja.",
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
        diagnose_paid = gate_asset.evaluate_volatility(gate_delta, gate_threshold)

        if diagnose_paid:
            # Counterfactual dissent-replay (#8): który dysydent rady trafił ruch,
            # mimo że agent poszedł inaczej? Wzbogaca diagnozę o stłumiony, trafny
            # głos mniejszości. W pełni graceful — brak głosów / błąd → bez bloku.
            vindicated_ctx = _vindicated_context(
                repository_port, last, current_price, symbol
            )
            prompt = get_mistake_diagnosis_prompt(
                last_trend=str(last.predicted_trend),
                last_news_summary="(zapisany w prediction_logs)",
                actual_price=str(current_price),
                vindicated_context=vindicated_ctx,
            )
            # #14: płatne wywołanie LLM diagnozy — log jawny + timing.
            _log_paid_call("reflect", symbol, "llm_port.analyze_mistake")
            with _timed_node("reflect_analyze_mistake", symbol):
                insight = llm_port.analyze_mistake(prompt)
        else:
            # Niska zmienność — nie palimy budżetu LLM na diagnozę. Predykcja
            # i tak zostaje zamknięta z accuracy/trend; diagnozę odkładamy.
            insight = "Niska zmienność — diagnoza LLM odłożona."

        repository_port.update_prediction_accuracy(
            prediction_id=last.id,
            actual_price=current_price,
            accuracy_score=accuracy,
            is_trend_correct=trend_correct,
            insight=insight,
        )
        return {"reflection_context": f"Ostatni błąd: {insight}"}

    def analyze_sentiment_node(state: AgentState) -> dict[str, Any]:
        symbol = state["symbol"]
        # #14: węzeł płatny (Alpha Vantage NEWS_SENTIMENT) — instrumentujemy czas
        # i logujemy jawnie fakt płatnego wywołania.
        _log_paid_call("analyze_sentiment", symbol, "sentiment_port.get_social_score")
        with _timed_node("analyze_sentiment", symbol):
            return {"sentiment": sentiment_port.get_social_score(symbol)}

    def fetch_news_node(state: AgentState) -> dict[str, Any]:
        symbol = state["symbol"]
        # #14: węzeł płatny (Alpha Vantage NEWS feed) — instrumentacja jak wyżej.
        _log_paid_call("fetch_news", symbol, "news_port.get_news_context")
        with _timed_node("fetch_news", symbol):
            return {"news": news_port.get_news_context(symbol)}

    def predict_node(state: AgentState) -> dict[str, Any]:
        symbol = state["symbol"]
        # #14: predict to najdroższy węzeł (LLM + ewentualny embedding/RAG).
        # Mierzymy całkowity czas węzła; perf_counter zamknięty tuż przed return.
        predict_start = time.perf_counter()
        sentiment = state.get("sentiment") or {}
        news = state.get("news", [])
        news_summary = _summarize_news(news)

        # RAG: embedujemy podsumowanie newsów RAZ (reużyte w save_node) i
        # wyszukujemy podobne historyczne sytuacje, by wstrzyknąć je do promptu
        # ("jak skończyły się analogiczne układy"). W pełni graceful — błąd
        # embeddingu/retrievalu nie blokuje predykcji.
        news_embedding: list[float] | None = None
        similar_context = ""
        # Q5: precedent receipts — te same analogi, które trafiają do promptu,
        # reużyte jako audytowalny blok "Na podstawie analogów" w raporcie.
        similar_precedents: list[dict[str, Any]] = []
        if embedding_port is not None and news_summary:
            try:
                news_embedding = embedding_port.embed(news_summary)
            except Exception:
                logger.exception(
                    "Embedding failed for %s — predicting without RAG", state["symbol"]
                )
            if news_embedding:
                # Vector Memory: gdy włączone, retrieval preferuje analogi z tego
                # samego reżimu rynku (filtr w RPC, migracja 017).
                regime_filter = (
                    state.get("regime_context") if vector_memory_enabled else None
                )
                similar_context, similar_precedents = _build_similar_context(
                    repository_port,
                    news_embedding,
                    state["symbol"],
                    rag_outcome_weight,
                    regime=regime_filter,
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
            # #7 reżim + #5 contagion (oba z bieżącego cyklu, przekazane w state).
            regime_context=state.get("regime_context", ""),
            peer_context=", ".join(
                f"{sym}:{stance}" for sym, stance in state.get("peer_context", ())
            ),
        )
        # #12-guard: w przeciwieństwie do council_node, predict_node wołał
        # llm_port.analyze bez zabezpieczenia. Czatliwa / niepoprawna odpowiedź
        # LLM (adapter rzuca ValueError przy parsowaniu JSON) wywalała cały
        # symbol. Degradujemy gracefully: neutralna analiza + flaga jakości,
        # żeby ML i zapis nadal się wykonały, a trening odsiał te rekordy.
        llm_failed = False
        # #6: tool-use research agent. Tylko NAJWIĘKSZE ruchy (osobny, wyższy próg
        # `tool_use_threshold`) uzasadniają kosztowną, wielorundową pętlę tool-use,
        # w której model może dociągnąć read-only toole (fundamenty/makro) przed
        # werdyktem. Poniżej tego progu (albo gdy tool-use wyłączony) — zwykły,
        # jednorazowy llm_port.analyze. Bramka liczona na tej samej delcie co
        # główna bramka volatility (state["delta"] względem snapshotu).
        # #14: jawny log płatnego wywołania LLM (główna analiza jakościowa).
        try:
            if (
                tool_use_port is not None
                and research_tools
                and tool_use_threshold is not None
                and (state.get("asset") or Asset(symbol=symbol)).evaluate_volatility(
                    PriceDelta(state["delta"]), tool_use_threshold
                )
            ):
                _log_paid_call("predict", symbol, "tool_use_port.analyze_with_tools")
                with _timed_node("predict_tool_use", symbol):
                    llm_analysis = tool_use_port.analyze_with_tools(
                        prompt, research_tools
                    )
            else:
                _log_paid_call("predict", symbol, "llm_port.analyze")
                llm_analysis = llm_port.analyze(prompt)
        except Exception:
            logger.exception(
                "llm_port.analyze failed for %s — degraduję do neutralnej analizy",
                state["symbol"],
            )
            llm_analysis = {
                "trend_direction": "SIDEWAYS",
                "confidence_score": 0.5,
                "av_agreement": 0.5,
                "reasoning": "Analiza LLM niedostępna — neutralny fallback.",
            }
            llm_failed = True

        # 2. ML — twarda predykcja liczbowa (multi-feature)
        # #21: trend_direction walidowany jak pola liczbowe — present-but-invalid
        # (np. "UP") daje neutralny sygnał, ALE zostawia flagę jakości. Genuinely
        # -absent default'uje do SIDEWAYS bez flagi.
        llm_trend_signal, trend_flag = _validate_trend_direction(
            llm_analysis.get("trend_direction")
        )

        # Walidacja pól liczbowych — każdy padły input zostawia ślad w
        # data_quality_flags (None → _missing, nie-numeryczne/NaN → _invalid).
        flags: list[str] = []
        if llm_failed:
            flags.append("llm_analysis_failed")
        if trend_flag is not None:
            flags.append(trend_flag)
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
        feature_attribution: tuple[FeatureContribution, ...] = ()
        if ml_port.is_trained:
            # Model przewiduje ZWROT 12h; predict() rekonstruuje cenę bezwzględną
            # z bieżącej ceny (cena*(1+zwrot)).
            ml_target = ml_port.predict(features, current_price=current_price).amount
            # Q3: atrybucja cech (SHAP-lite) — lokalny CPU, zero płatnych. Graceful:
            # gdyby explain padł, predykcja i tak leci (atrybucja jest dodatkiem).
            try:
                contribs = {
                    k: v for k, v in ml_port.explain(features).items() if k != "_bias"
                }
                feature_attribution = top_contributions(contribs)
            except Exception:
                logger.exception("explain() failed for %s — bez atrybucji", symbol)
        else:
            ml_target = state["current_price"]

        # #14: pojedyncze wyjście węzła — logujemy całkowity czas predict.
        elapsed_ms = (time.perf_counter() - predict_start) * 1000.0
        logger.info("node=predict symbol=%s elapsed_ms=%.1f", symbol, elapsed_ms)

        return {
            "llm_analysis": llm_analysis,
            "ml_target_price": ml_target,
            "data_quality_flags": flags,
            # Reużywane w save_node — embedujemy newsy tylko raz na cykl.
            "news_embedding": news_embedding,
            # Q5: precedent receipts dla raportu (render-only, brak persystencji).
            "similar_precedents": similar_precedents,
            # Q3: atrybucja cech ML dla raportu (render-only).
            "feature_attribution": feature_attribution,
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
            news_articles=tuple(
                item.get("title", "")
                for item in news[:5]
                if item.get("title")
            ),
            llm_trend=str(llm_analysis.get("trend_direction", "SIDEWAYS")),
            llm_confidence=_float_or_default(llm_analysis.get("confidence_score"), 0.5),
            ml_price_target=state.get("ml_target_price") or state["current_price"],
            fundamentals=state.get("fundamentals"),
            valuation_verdict=state.get("valuation_verdict", ValuationVerdict.UNKNOWN),
            # #3: typ aktywa napędza dobór wag person (per klasa aktywa).
            asset_type=(state.get("asset") or Asset(symbol=state["symbol"])).asset_type,
        )
        # #14: rada to N równoległych wywołań LLM — najdroższy płatny węzeł.
        # Jawny log płatnego wywołania + timing (mierzony też przy błędzie).
        _log_paid_call("council", state["symbol"], "council_port.analyze")
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
        }

        # Vector Memory: tagujemy embedding reżimem rynku, w którym powstała
        # predykcja (migracja 017), żeby przyszły RAG mógł filtrować analogi po
        # reżimie. Bez flagi kolumna zostaje NULL (RAG działa jak dotąd).
        if vector_memory_enabled:
            record["regime"] = canonical_regime_key(state.get("regime_context"))

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
                    votes=list(verdict.investor_opinions),
                )
            except Exception:
                logger.exception(
                    "save_council_votes failed for %s — JSONB blob in "
                    "prediction_logs is still saved",
                    state["symbol"],
                )

        # #6: węzeł terminalny — zapisujemy snapshot ceny dla następnego cyklu.
        _persist_price_snapshot(state)

        return {"prediction_id": prediction_id, "status": "saved"}

    def ignore_node(state: AgentState) -> dict[str, Any]:
        # #6: węzeł terminalny — zapisujemy snapshot ceny dla następnego cyklu.
        _persist_price_snapshot(state)
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
            # #17: LangGraph 1.x stubs typują węzły jako `_Node[Never]` i nie
            # inferują schematu opartego o TypedDict (StateGraph[AgentState]).
            # Runtime jest poprawny; wąski ignore zamiast szerokiego override.
            _build_fetch_fundamentals_node(fundamentals_port),  # type: ignore[arg-type]
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
