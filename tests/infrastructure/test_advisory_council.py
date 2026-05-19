# tests/infrastructure/test_advisory_council.py
from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import AdvisoryCouncilPort, LLMPort
from src.domain.council import CouncilInput, CouncilVerdict
from src.infrastructure.adapters.advisory_council import LLMAdvisoryCouncil

ALL_INVESTORS = [
    "Warren Buffett", "Benjamin Graham", "George Soros", "Peter Lynch",
    "Ray Dalio", "Charlie Munger", "Philip Fisher", "Paul Tudor Jones",
    "Bill Gross", "Jesse Livermore",
    "Cathie Wood", "Michael Burry", "Howard Marks",
    "Stanley Druckenmiller", "Joel Greenblatt",
]
COUNCIL_SIZE = len(ALL_INVESTORS)


def _make_input() -> CouncilInput:
    return CouncilInput(
        symbol="AAPL",
        current_price=Decimal("180.00"),
        price_delta_pct=Decimal("3.5"),
        sentiment_score=0.6,
        news_articles=["Apple beats earnings"],
        llm_trend="BULLISH",
        llm_confidence=0.82,
        ml_price_target=Decimal("185.00"),
    )


def _opinion_json(rec: str = "BUY") -> dict:
    return {
        "recommendation": rec,
        "confidence": 0.8,
        "reasoning": "Silne fundamenty.",
        "key_factors": ["moat", "FCF", "wzrost"],
    }


def _verdict_json() -> dict:
    return {
        "final_recommendation": "BUY",
        "consensus_strength": 0.73,
        "summary": "Większość radzi kupować.",
        "dissenting_views": ["Soros: zbyt ryzykowne"],
    }


@pytest.fixture
def llm_port() -> Mock:
    mock = Mock(spec=LLMPort)
    mock.analyze.return_value = _opinion_json()
    return mock


@pytest.fixture
def council(llm_port: Mock) -> LLMAdvisoryCouncil:
    return LLMAdvisoryCouncil(llm_port=llm_port)


class TestLLMAdvisoryCouncilIsPort:
    def test_implements_port(self, council: LLMAdvisoryCouncil):
        assert isinstance(council, AdvisoryCouncilPort)


class TestAnalyzeCallCount:
    def _side_effect_factory(self, llm_port: Mock):
        call_count = 0
        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            return _opinion_json() if call_count <= COUNCIL_SIZE else _verdict_json()
        llm_port.analyze.side_effect = side_effect

    def test_calls_llm_once_per_investor_plus_chairman(
        self, council: LLMAdvisoryCouncil, llm_port: Mock
    ):
        self._side_effect_factory(llm_port)
        council.analyze("AAPL", _make_input())
        assert llm_port.analyze.call_count == COUNCIL_SIZE + 1

    def test_returns_council_verdict(self, council: LLMAdvisoryCouncil, llm_port: Mock):
        self._side_effect_factory(llm_port)
        result = council.analyze("AAPL", _make_input())
        assert isinstance(result, CouncilVerdict)
        assert result.final_recommendation == "BUY"

    def test_verdict_has_opinion_per_investor(
        self, council: LLMAdvisoryCouncil, llm_port: Mock
    ):
        self._side_effect_factory(llm_port)
        result = council.analyze("AAPL", _make_input())
        assert len(result.investor_opinions) == COUNCIL_SIZE

    def test_opinion_investor_names_all_present(self, council: LLMAdvisoryCouncil, llm_port: Mock):
        self._side_effect_factory(llm_port)
        result = council.analyze("AAPL", _make_input())
        names = {op.investor_name for op in result.investor_opinions}
        assert names == set(ALL_INVESTORS)


class TestFallbacks:
    def test_missing_recommendation_defaults_to_hold(
        self, council: LLMAdvisoryCouncil, llm_port: Mock
    ):
        call_count = 0
        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= COUNCIL_SIZE:
                return {"confidence": 0.5, "reasoning": "ok", "key_factors": []}
            return _verdict_json()
        llm_port.analyze.side_effect = side_effect
        result = council.analyze("AAPL", _make_input())
        for op in result.investor_opinions:
            assert op.recommendation == "HOLD"

    def test_missing_final_recommendation_defaults_to_hold(
        self, council: LLMAdvisoryCouncil, llm_port: Mock
    ):
        call_count = 0
        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= COUNCIL_SIZE:
                return _opinion_json()
            return {"consensus_strength": 0.5, "summary": "ok", "dissenting_views": []}
        llm_port.analyze.side_effect = side_effect
        result = council.analyze("AAPL", _make_input())
        assert result.final_recommendation == "HOLD"
