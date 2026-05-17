from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from src.domain.council import CouncilVerdict
from src.domain.value_objects import ValuationVerdict


@dataclass(frozen=True)
class ValuationSection:
    trailing_pe: float | None
    forward_pe: float | None
    peg_ratio: float | None
    eps_growth_yoy: float | None
    verdict: ValuationVerdict
    fetched_at: datetime


@dataclass(frozen=True)
class TopNewsItem:
    title: str
    source: str | None
    url: str | None
    relevance: float
    sentiment: float


@dataclass(frozen=True)
class SymbolResult:
    """Pojedynczy wynik analizy dla symbolu."""

    symbol: str
    status: str  # "saved" | "ignored" | "error"
    delta: Decimal | None = None
    current_price: Decimal | None = None
    trend: str | None = None
    target_price: Decimal | None = None
    confidence_score: float | None = None
    reasoning: str | None = None
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    news_volume: int | None = None
    av_llm_agreement: float | None = None
    reflection_insight: str | None = None
    top_news: list[TopNewsItem] = field(default_factory=list)
    error_message: str | None = None
    council_verdict: CouncilVerdict | None = None
    valuation: ValuationSection | None = None

    @property
    def expected_change(self) -> Decimal | None:
        """Procentowa zmiana z obecnej ceny do prognozowanej (target)."""
        if self.current_price is None or self.target_price is None:
            return None
        if self.current_price == 0:
            return None
        return (self.target_price - self.current_price) / self.current_price


@dataclass(frozen=True)
class TradeSignal:
    symbol: str
    direction: str
    confidence: float
    expected_change: Decimal
    strength: float
    current_price: Decimal | None
    target_price: Decimal | None


@dataclass(frozen=True)
class RiskSignal:
    symbol: str
    type: str
    severity: str
    description: str


@dataclass(frozen=True)
class ResolvedPrediction:
    """Predykcja zamknięta w bieżącym cyklu (dostała accuracy_score)."""

    symbol: str
    predicted_trend: str
    accuracy_score: float
    is_correct: bool
