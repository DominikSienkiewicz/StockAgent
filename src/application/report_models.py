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
    # Klasa aktywa ("STOCK" | "ETF" | "CRYPTO" | ...) — z domeny (Asset.asset_type).
    # Pozwala raportowi rozpoznać krypto niezawodnie (nie przez słownik SECTORS)
    # i wyświetlić je w dedykowanej sekcji, też gdy cykl był "ignored".
    asset_class: str | None = None
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
    """Predykcja zamknięta — oceniona kierunkowo (is_trend_correct).

    Niesie pełen post-mortem: oryginalne uzasadnienie (dlaczego prognoza
    mówiła wzrost/spadek), faktyczny ruch ceny i diagnozę (dlaczego się nie
    sprawdziła). Pola wzbogacające są opcjonalne (wstecznie kompatybilne)."""

    symbol: str
    predicted_trend: str
    is_correct: bool
    # Oryginalne uzasadnienie prognozy ("dlaczego wzrośnie/spadnie").
    reasoning: str | None = None
    # Diagnoza po fakcie ("dlaczego się nie potwierdziła") — z reflect_node.
    insight: str | None = None
    # Ceny do policzenia faktycznego ruchu ("co się stało").
    price_at_prediction: Decimal | None = None
    actual_price: Decimal | None = None

    @property
    def actual_change_pct(self) -> Decimal | None:
        """Faktyczna procentowa zmiana ceny od momentu predykcji do zamknięcia."""
        if self.price_at_prediction is None or self.actual_price is None:
            return None
        if self.price_at_prediction == 0:
            return None
        return (self.actual_price - self.price_at_prediction) / self.price_at_prediction
