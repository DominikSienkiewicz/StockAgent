from unittest.mock import Mock

from src.application.agent_graph import _build_similar_context
from src.application.ports import RepositoryPort
from src.application.rag_rerank import rerank_analogs


class TestBuildSimilarContextRerank:
    def test_reranks_by_outcome_in_built_text(self):
        repo = Mock(spec=RepositoryPort)
        repo.find_similar_predictions.return_value = [
            {"news_summary": "ALPHA", "predicted_trend": "BULLISH",
             "is_trend_correct": False, "similarity": 0.9, "correction_insights": ""},
            {"news_summary": "BETA", "predicted_trend": "BEARISH",
             "is_trend_correct": True, "similarity": 0.85, "correction_insights": ""},
        ]
        # BETA trafiony (0.85+0.3=1.15) > ALPHA nietrafiony (0.9) → BETA wyżej.
        # _build_similar_context zwraca (tekst, rekordy) — receipts #Q5.
        text, _records = _build_similar_context(repo, [0.1], "AAPL", outcome_weight=0.3)
        assert text.index("BETA") < text.index("ALPHA")


class TestRerankAnalogs:
    def test_correct_analog_outranks_incorrect_at_equal_similarity(self):
        records = [
            {"id": "a", "similarity": 0.8, "is_trend_correct": False},
            {"id": "b", "similarity": 0.8, "is_trend_correct": True},
        ]
        result = rerank_analogs(records, outcome_weight=0.3)
        assert [r["id"] for r in result] == ["b", "a"]

    def test_none_outcome_is_neutral(self):
        # a: 0.9 + 0 ; b: 0.85 + 0.3 = 1.15 → b wygrywa mimo niższego similarity.
        records = [
            {"id": "a", "similarity": 0.9, "is_trend_correct": None},
            {"id": "b", "similarity": 0.85, "is_trend_correct": True},
        ]
        result = rerank_analogs(records, outcome_weight=0.3)
        assert [r["id"] for r in result] == ["b", "a"]

    def test_outcome_weight_zero_preserves_similarity_order(self):
        records = [
            {"id": "a", "similarity": 0.7, "is_trend_correct": True},
            {"id": "b", "similarity": 0.9, "is_trend_correct": False},
        ]
        result = rerank_analogs(records, outcome_weight=0.0)
        assert [r["id"] for r in result] == ["b", "a"]

    def test_top_k_truncates(self):
        records = [
            {"id": str(i), "similarity": i / 10, "is_trend_correct": None}
            for i in range(5)
        ]
        result = rerank_analogs(records, outcome_weight=0.3, top_k=2)
        assert len(result) == 2

    def test_missing_similarity_treated_as_zero(self):
        records = [
            {"id": "a", "is_trend_correct": True},
            {"id": "b", "similarity": 0.1, "is_trend_correct": False},
        ]
        # a: 0 + 0.3 = 0.3 ; b: 0.1 - 0.3 = -0.2 → a wygrywa.
        result = rerank_analogs(records, outcome_weight=0.3)
        assert [r["id"] for r in result] == ["a", "b"]


class TestPointInTimeAnalogs:
    """ML-41: za `rag_point_in_time_enabled` analogi RAG to wyłącznie predykcje
    rozliczone. Nierozliczony analog (w tym zapisany chwilę wcześniej w TYM cyklu —
    echo bieżących newsów) nie zajmuje miejsca w top-3 ani nie siedzi „neutralnie”
    w reranku. Mniej kandydatów po filtrze, więc retrieval bierze ich więcej."""

    @staticmethod
    def _analog(name: str, correct: bool | None, similarity: float) -> dict:
        return {
            "news_summary": name,
            "predicted_trend": "BULLISH",
            "is_trend_correct": correct,
            "similarity": similarity,
            "correction_insights": "",
        }

    def _repo(self) -> Mock:
        repo = Mock(spec=RepositoryPort)
        repo.find_similar_predictions.return_value = [
            self._analog("ECHO", None, 0.99),
            self._analog("ALPHA", True, 0.9),
            self._analog("OPEN", None, 0.88),
            self._analog("BETA", False, 0.85),
            self._analog("GAMMA", True, 0.8),
            self._analog("DELTA", True, 0.7),
        ]
        return repo

    def test_only_settled_analogs_reach_the_prompt(self) -> None:
        repo = self._repo()

        text, precedents = _build_similar_context(repo, [0.1], "AAPL", point_in_time=True)

        assert "ECHO" not in text
        assert "OPEN" not in text
        assert [p["summary"] for p in precedents] == ["ALPHA", "BETA", "GAMMA"]
        assert "DELTA" not in text
        assert repo.find_similar_predictions.call_args.kwargs["limit"] == 12

    def test_flag_off_keeps_the_retrieval_as_it_was(self) -> None:
        repo = self._repo()

        text, _ = _build_similar_context(repo, [0.1], "AAPL")

        assert "ECHO" in text
        assert repo.find_similar_predictions.call_args.kwargs["limit"] == 3

    def test_graph_flag_reaches_the_retrieval(self) -> None:
        from decimal import Decimal

        from src.application.agent_graph import AgentGraphDeps, _compute_rag_context
        from src.application.ports import (
            EmbeddingPort,
            LLMPort,
            MarketDataPort,
            MLPredictionPort,
            NewsPort,
            SentimentPort,
        )
        from src.domain.value_objects import Threshold

        repo = self._repo()
        embedding = Mock(spec=EmbeddingPort)
        embedding.embed.return_value = [0.1]
        deps = AgentGraphDeps(
            market_port=Mock(spec=MarketDataPort),
            sentiment_port=Mock(spec=SentimentPort),
            news_port=Mock(spec=NewsPort),
            repository_port=repo,
            ml_port=Mock(spec=MLPredictionPort),
            llm_port=Mock(spec=LLMPort),
            threshold=Threshold(Decimal("0.02")),
            embedding_port=embedding,
            rag_point_in_time_enabled=True,
        )

        _vector, context, _precedents = _compute_rag_context(deps, {"symbol": "AAPL"}, "news")

        assert "ECHO" not in context
        assert repo.find_similar_predictions.call_args.kwargs["limit"] == 12
