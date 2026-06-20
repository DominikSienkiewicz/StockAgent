# tests/infrastructure/test_advisory_council.py
from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import AdvisoryCouncilPort, LLMPort
from src.domain.council import CouncilInput, CouncilVerdict, InvestorPersona
from src.infrastructure.adapters.advisory_council import (
    COUNCIL_MAX_PERSONAS,
    LLMAdvisoryCouncil,
)

ALL_INVESTORS = [
    "Warren Buffett", "Benjamin Graham", "George Soros", "Peter Lynch",
    "Ray Dalio", "Charlie Munger", "Philip Fisher", "Paul Tudor Jones",
    "Bill Gross", "Jesse Livermore",
    "Cathie Wood", "Michael Burry", "Howard Marks",
    "Stanley Druckenmiller", "Joel Greenblatt",
]
COUNCIL_SIZE = len(ALL_INVESTORS)

_TEST_PERSONAS = tuple(
    InvestorPersona(name=name, style=f"Test style for {name}. " * 3)
    for name in ALL_INVESTORS
)


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
    # max_personas podniesione ponad rozmiar rady testowej (15 > domyślny cap 12),
    # żeby testy fan-outu sprawdzały wywołanie KAŻDEJ persony bez ucinania.
    return LLMAdvisoryCouncil(
        llm_port=llm_port,
        personas=_TEST_PERSONAS,
        max_personas=COUNCIL_SIZE,
    )


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

    def test_missing_final_recommendation_uses_votes_not_chairman(
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
        # Wszystkie opinie to BUY (domyślny mock) → werdykt z głosów to BUY,
        # niezależnie od tego, co (nie) zwrócił chairman.
        assert result.final_recommendation == "BUY"


class TestConsensusDerivedFromVotes:
    """Finding #3: consensus_strength + final_recommendation muszą pochodzić
    z REALNYCH głosów inwestorów, nie z liczby wymyślonej przez chairman-LLM."""

    def _council_3_personas(self, llm_port: Mock) -> LLMAdvisoryCouncil:
        personas = tuple(
            InvestorPersona(name=n, style=f"Styl testowy dla {n}. " * 3)
            for n in ("Inv A", "Inv B", "Inv C", "Inv D")
        )
        return LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, max_personas=10
        )

    def test_unanimous_buy_survives_chairman_failure(self, llm_port: Mock):
        # 4 inwestorów głosuje BUY; wywołanie chairmana RZUCA wyjątek.
        # Mimo to werdykt musi być BUY z wysokim consensus_strength — NIE 0.5/HOLD.
        n_personas = 4
        call_count = 0

        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= n_personas:
                return _opinion_json("BUY")
            raise RuntimeError("chairman down")

        llm_port.analyze.side_effect = side_effect
        council = self._council_3_personas(llm_port)
        result = council.analyze("AAPL", _make_input())

        assert result.final_recommendation == "BUY"
        assert result.consensus_strength == pytest.approx(1.0)

    def test_chairman_number_is_ignored_for_decision(self, llm_port: Mock):
        # Chairman zwraca SELL / 0.99, ale wszyscy inwestorzy głosują BUY.
        # Decyzja należy do głosów → BUY, a strength = z głosów (1.0), nie 0.99.
        n_personas = 4
        call_count = 0

        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= n_personas:
                return _opinion_json("BUY")
            return {
                "final_recommendation": "SELL",
                "consensus_strength": 0.99,
                "summary": "Chairman się myli.",
                "dissenting_views": [],
            }

        llm_port.analyze.side_effect = side_effect
        council = self._council_3_personas(llm_port)
        result = council.analyze("AAPL", _make_input())

        assert result.final_recommendation == "BUY"
        assert result.consensus_strength == pytest.approx(1.0)

    def test_chairman_summary_still_used(self, llm_port: Mock):
        # Tekstowe pola (summary, dissenting_views) NADAL pochodzą od chairmana.
        n_personas = 4
        call_count = 0

        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= n_personas:
                return _opinion_json("BUY")
            return _verdict_json()

        llm_port.analyze.side_effect = side_effect
        council = self._council_3_personas(llm_port)
        result = council.analyze("AAPL", _make_input())

        assert result.summary == "Większość radzi kupować."
        assert result.dissenting_views == ["Soros: zbyt ryzykowne"]


class TestPersonaCap:
    """Finding #11: liczba wywołań LLM ograniczona górnym capem person."""

    def _personas(self, count: int) -> tuple[InvestorPersona, ...]:
        return tuple(
            InvestorPersona(name=f"Inv {i:02d}", style=f"Styl testowy {i}. " * 3)
            for i in range(count)
        )

    def test_truncates_above_cap_and_warns(self, llm_port: Mock, caplog):
        over = COUNCIL_MAX_PERSONAS + 3
        personas = self._personas(over)
        council = LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)

        with caplog.at_level("WARNING"):
            result = council.analyze("AAPL", _make_input())

        # Tylko `cap` opinii inwestorów + jeden chairman.
        assert llm_port.analyze.call_count == COUNCIL_MAX_PERSONAS + 1
        assert len(result.investor_opinions) == COUNCIL_MAX_PERSONAS
        # Ostrzeżenie wymienia odrzucone persony (te ponad capem).
        warning_text = " ".join(r.getMessage() for r in caplog.records)
        dropped_names = [p.name for p in personas[COUNCIL_MAX_PERSONAS:]]
        for name in dropped_names:
            assert name in warning_text

    def test_at_cap_unchanged_no_warning(self, llm_port: Mock, caplog):
        personas = self._personas(COUNCIL_MAX_PERSONAS)
        council = LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)

        with caplog.at_level("WARNING"):
            result = council.analyze("AAPL", _make_input())

        assert len(result.investor_opinions) == COUNCIL_MAX_PERSONAS
        assert "persona" not in " ".join(
            r.getMessage().lower() for r in caplog.records
        )

    def test_below_cap_unchanged(self, llm_port: Mock):
        personas = self._personas(5)
        council = LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)
        result = council.analyze("AAPL", _make_input())
        assert len(result.investor_opinions) == 5

    def test_custom_cap_param(self, llm_port: Mock):
        personas = self._personas(8)
        council = LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, max_personas=3
        )
        result = council.analyze("AAPL", _make_input())
        assert len(result.investor_opinions) == 3

    def test_default_constant_value_is_12(self):
        assert COUNCIL_MAX_PERSONAS == 12

    def test_backward_compatible_signature(self, llm_port: Mock):
        # Istniejące wywołanie bez max_personas musi nadal działać.
        personas = self._personas(3)
        council = LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)
        assert isinstance(council, LLMAdvisoryCouncil)
