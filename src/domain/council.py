# src/domain/council.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from src.domain.value_objects import Fundamentals, ValuationVerdict


@dataclass(frozen=True)
class CouncilInput:
    symbol: str
    current_price: Decimal
    price_delta_pct: Decimal
    sentiment_score: float
    news_articles: list[str]
    llm_trend: str
    llm_confidence: float
    ml_price_target: Decimal
    # Pola opcjonalne — domyślne wartości zapewniają wsteczną kompatybilność
    fundamentals: Fundamentals | None = field(default=None)
    valuation_verdict: ValuationVerdict = field(default=ValuationVerdict.UNKNOWN)


@dataclass(frozen=True)
class InvestorOpinion:
    investor_name: str
    recommendation: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    reasoning: str
    key_factors: list[str]


@dataclass(frozen=True)
class CouncilVerdict:
    final_recommendation: Literal["BUY", "SELL", "HOLD"]
    consensus_strength: float
    summary: str
    dissenting_views: list[str]
    investor_opinions: list[InvestorOpinion]
