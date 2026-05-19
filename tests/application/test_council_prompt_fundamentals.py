# tests/application/test_council_prompt_fundamentals.py
from datetime import UTC, datetime
from decimal import Decimal

from src.application.council_prompts import investor_prompt
from src.domain.council import CouncilInput, InvestorPersona
from src.domain.value_objects import Fundamentals, ValuationVerdict

_BUFFETT = InvestorPersona(
    name="Warren Buffett",
    style="Trwała przewaga konkurencyjna, margines bezpieczeństwa, horyzont 10+ lat.",
)


def _input(
    fundamentals: Fundamentals | None,
    verdict: ValuationVerdict,
) -> CouncilInput:
    return CouncilInput(
        symbol="AAPL",
        current_price=Decimal("190.00"),
        price_delta_pct=Decimal("0.012"),
        sentiment_score=0.5,
        news_articles=["headline A"],
        llm_trend="UP",
        llm_confidence=0.7,
        ml_price_target=Decimal("195.00"),
        fundamentals=fundamentals,
        valuation_verdict=verdict,
    )


def test_prompt_includes_valuation_when_known() -> None:
    f = Fundamentals(
        trailing_pe=25.0,
        forward_pe=20.0,
        peg_ratio=2.5,
        eps_growth_yoy=0.05,
        fetched_at=datetime(2026, 5, 17, tzinfo=UTC),
    )
    prompt = investor_prompt(_BUFFETT, _input(f, ValuationVerdict.OVERVALUED))
    assert "Valuation snapshot" in prompt
    assert "2.50" in prompt  # PEG
    assert "OVERVALUED" in prompt


def test_prompt_omits_valuation_when_unknown() -> None:
    prompt = investor_prompt(_BUFFETT, _input(None, ValuationVerdict.UNKNOWN))
    assert "Valuation snapshot" not in prompt
