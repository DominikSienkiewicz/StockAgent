# tests/application/test_council_prompts.py
from decimal import Decimal

from src.application.council_prompts import (
    INVESTOR_PERSONAS,
    chairman_prompt,
    investor_prompt,
)
from src.domain.council import CouncilInput, InvestorOpinion


def _make_input() -> CouncilInput:
    return CouncilInput(
        symbol="AAPL",
        current_price=Decimal("180.00"),
        price_delta_pct=Decimal("3.5"),
        sentiment_score=0.6,
        news_articles=["Apple beats earnings", "iPhone sales up"],
        llm_trend="BULLISH",
        llm_confidence=0.82,
        ml_price_target=Decimal("185.00"),
    )


def _make_opinion(investor: str, rec: str = "BUY") -> InvestorOpinion:
    return InvestorOpinion(
        investor_name=investor,
        recommendation=rec,  # type: ignore[arg-type]
        confidence=0.75,
        reasoning="Silne podstawy.",
        key_factors=["moat"],
    )


class TestInvestorPersonas:
    def test_all_investors_present(self):
        expected = {
            "Warren Buffett",
            "Benjamin Graham",
            "George Soros",
            "Peter Lynch",
            "Ray Dalio",
            "Charlie Munger",
            "Philip Fisher",
            "Paul Tudor Jones",
            "Bill Gross",
            "Jesse Livermore",
            "Cathie Wood",
            "Michael Burry",
            "Howard Marks",
            "Stanley Druckenmiller",
            "Joel Greenblatt",
        }
        assert set(INVESTOR_PERSONAS.keys()) == expected

    def test_personas_sourced_from_domain(self):
        from src.domain.council import DEFAULT_INVESTOR_PERSONAS

        domain_names = {p.name for p in DEFAULT_INVESTOR_PERSONAS}
        assert set(INVESTOR_PERSONAS.keys()) == domain_names

    def test_each_persona_is_non_empty_string(self):
        for name, style in INVESTOR_PERSONAS.items():
            assert isinstance(style, str) and len(style) > 20, f"{name} persona too short"


class TestInvestorPrompt:
    def test_contains_investor_name(self):
        prompt = investor_prompt("Warren Buffett", _make_input())
        assert "Warren Buffett" in prompt

    def test_contains_symbol(self):
        prompt = investor_prompt("Peter Lynch", _make_input())
        assert "AAPL" in prompt

    def test_contains_price(self):
        prompt = investor_prompt("Ray Dalio", _make_input())
        assert "180" in prompt

    def test_contains_output_schema_keys(self):
        prompt = investor_prompt("George Soros", _make_input())
        assert "recommendation" in prompt
        assert "confidence" in prompt
        assert "key_factors" in prompt


class TestChairmanPrompt:
    def test_uses_dynamic_council_size_not_hardcoded(self):
        # Bug regression: chairman_prompt miał wpisane na sztywno "11 legendarnych
        # inwestorów" niezależnie od liczby opinii przekazanych do funkcji.
        opinions = [_make_opinion("Warren Buffett"), _make_opinion("Peter Lynch")]
        prompt = chairman_prompt(opinions, _make_input())
        assert "2 legendarnych" in prompt
        assert "11 legendarnych" not in prompt

    def test_contains_all_investor_names(self):
        opinions = [
            _make_opinion("Warren Buffett", "BUY"),
            _make_opinion("George Soros", "SELL"),
        ]
        prompt = chairman_prompt(opinions, _make_input())
        assert "Warren Buffett" in prompt
        assert "George Soros" in prompt

    def test_contains_output_schema_keys(self):
        opinions = [_make_opinion("Warren Buffett")]
        prompt = chairman_prompt(opinions, _make_input())
        assert "final_recommendation" in prompt
        assert "consensus_strength" in prompt
        assert "dissenting_views" in prompt
