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
