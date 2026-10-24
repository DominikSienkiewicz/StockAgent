from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from src.application.decision_receipt import DecisionReceipt
from src.domain.conformal import MoveInterval
from src.domain.council import CouncilVerdict
from src.domain.cycle_maturity import SkipReason
from src.domain.feature_attribution import FeatureContribution
from src.domain.mistake_taxonomy import MistakeCategory
from src.domain.provenance import ProvenanceBadge
from src.domain.sizing import PositionAction, SizeBand
from src.domain.value_objects import Ratio, ValuationVerdict


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
class SimilarPrecedent:
    """Q5 — pojedynczy analog RAG użyty w bieżącym cyklu (precedent receipt).

    Kompaktowy snapshot historycznej sytuacji: jej streszczenie, prognozowany
    wtedy trend i czy się sprawdził kierunkowo. Render-only — buduje blok
    'Na podstawie analogów' w raporcie. `is_trend_correct` może być None
    (analog jeszcze nieoceniony)."""

    summary: str
    predicted_trend: str
    is_trend_correct: bool | None = None


@dataclass(frozen=True)
class SymbolResult:
    """Pojedynczy wynik analizy dla symbolu."""

    symbol: str
    status: str  # "saved" | "ignored" | "error"
    delta: Decimal | None = None
    current_price: Decimal | None = None
    # #4: dlaczego symbol został pominięty. Rozdziela cold-start (brak punktu
    # odniesienia) od realnego odcięcia bramką volatility i od tickerów
    # niewspieranych cenowo. None = nie pominięto (saved) albo błąd.
    # Bez tego rozdziału powitalny ton "Dzień 1" przykryłby masową awarię źródeł.
    skip_reason: SkipReason | None = None
    # FIN-14: moment zapisu referencji ceny, od której liczy się `delta`. None =
    # brak referencji (cold-start) albo wynik spoza grafu. Przy STALE_REFERENCE
    # raport pisze „referencja sprzed <data> — nowy punkt odniesienia”.
    reference_at: datetime | None = None
    # Klasa aktywa ("STOCK" | "ETF" | "CRYPTO" | ...) — z domeny (Asset.asset_type).
    # Pozwala raportowi rozpoznać krypto niezawodnie (nie przez mapę sektorów)
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
    # NIEZ-22: nazwa klasy wyjątku symbolu, który padł — agregat w step summary.
    # Sama nazwa klasy (bez komunikatu), więc może trafić na publiczną stronę runu.
    error_type: str | None = None
    council_verdict: CouncilVerdict | None = None
    valuation: ValuationSection | None = None
    # Q5: analogi RAG użyte w tym cyklu (precedent receipts). Pusta lista =
    # brak retrievalu / brak analogów → raport nie renderuje bloku.
    similar_precedents: list[SimilarPrecedent] = field(default_factory=list)
    # Q6: odznaki proweniencji danych (FRESH / STALE / DEGRADED / FLAGGED).
    # Pusta lista = brak sygnału (renderer pomija chipy).
    provenance_badges: list[ProvenanceBadge] = field(default_factory=list)
    # Q3: feature attribution (SHAP-lite) — które cechy ML pchnęły prognozę w
    # górę/dół i o ile (znakowane wkłady, posortowane malejąco po module). Pusty
    # tuple = brak atrybucji (model nietrenowany / cykl bez predykcji) →
    # wstecznie kompatybilne, renderer pomija sekcję.
    feature_attribution: tuple[FeatureContribution, ...] = ()
    # G03: True = XGBoost bez wag, `target_price` to baseline "cena bez zmian".
    # Renderer podpisuje prognozę jako cold-start zamiast pokazywać +0.00%.
    ml_cold_start: bool = False
    # ML-34: przedział 80% zwrotu do rozliczenia (conformal + ACI). None = cykl
    # bez przedziału (brak okna kalibracyjnego) → renderer pomija linię.
    move_interval: MoveInterval | None = None
    # MAIL-18: próg EFEKTYWNY bramki (ułamek) dla wyniku 'ignored' i mnożniki,
    # które go podniosły (reżim #7, earnings #6). None = wynik spoza bramki.
    effective_threshold: Decimal | None = None
    regime_multiplier: float = 1.0
    earnings_multiplier: float = 1.0
    # MAIL-21: flagi jakości danych symbolu z grafu (`data_quality_flags`) —
    # z nich linijka stanu źródeł przypisuje awarie dostawcom.
    data_quality_flags: tuple[str, ...] = ()

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
    # Q7: sugerowana banda wielkości pozycji (Kelly-lite z konsensusu rady +
    # track recordu agenta). None = brak danych do sizingu (np. brak werdyktu
    # rady) — wsteczna kompatybilność, renderer pomija magnitudę.
    size_band: SizeBand | None = None
    # #9: pewność skorygowana historycznym hit-rate'em swojego kubełka kalibracji.
    # None = brak historii → `strength` policzone na surowej `confidence`
    # (zachowanie identyczne jak przed #9). Surowa `confidence` zostaje obok,
    # żeby raport mógł pokazać "pewność LLM: 85% → skalibrowana historią: 58%".
    calibrated_confidence: float | None = None
    # WIA-13 (G03): True = brak prognozy ruchu (XGBoost bez wag), `strength` to
    # sama (skalibrowana) pewność, a `expected_change` = 0 nie jest prognozą.
    ml_cold_start: bool = False
    # WIA-12: liczność kubełka, którym skalibrowano pewność (0 = kubełek pusty,
    # pewność bez korekty). None = brak kalibracji. Mała próba skacze, więc
    # raport pokazuje n obok pewności skalibrowanej.
    calibration_bucket_count: int | None = None
    # PORT-08: kierunek w kontekście portfela czytelnika (ZREDUKUJ / UNIKAJ / TRZYMAJ
    # zamiast gołego SPRZEDAJ) i waga pozycji. None = portfel nieznany — raport
    # pokazuje `direction` jak przed PORT-08.
    position_action: PositionAction | None = None
    position_weight: Decimal | None = None


@dataclass(frozen=True)
class PortfolioMood:
    """📊 Nastroje watchlisty — agregaty sentymentu jednego cyklu (MAIL-10, MAIL-31).

    `avg_sentiment` i `avg_sentiment_label` są None, gdy cykl nie ma ani jednego
    sentymentu: brak danych to nie neutralny nastrój „+0,00”."""

    avg_sentiment: float | None
    avg_sentiment_label: str | None
    sentiment_count: int
    most_bullish: SymbolResult | None
    most_bearish: SymbolResult | None
    high_confidence_count: int
    saved_count: int
    total_news: int


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
    # Moment predykcji (`prediction_logs.timestamp`). Jego dzień UTC to klucz
    # cyklu: predykcje jednego cyklu są skorelowane (WIA-17, WIA-23).
    predicted_at: datetime | None = None
    # WIA-56: kwit decyzyjny (migracja 020) do linii pod chybieniem; None = brak
    # kwitu albo nieznana wersja schematu.
    receipt: DecisionReceipt | None = None
    # WIA-57: kategoria przyczyny błędu (`correction_category`, migracja 046);
    # None = brak kategorii (wiersz sprzed migracji albo odczyt bez flagi).
    correction_category: MistakeCategory | None = None

    @property
    def actual_change_pct(self) -> Ratio | None:
        """Faktyczna zmiana ceny od momentu predykcji do zamknięcia jako UŁAMEK
        (`Ratio`: 0.02 = +2%), mimo sufiksu `_pct` — formatery raportu mnożą go
        przez 100 (ARCH-04)."""
        if self.price_at_prediction is None or self.actual_price is None:
            return None
        if self.price_at_prediction == 0:
            return None
        return Ratio((self.actual_price - self.price_at_prediction) / self.price_at_prediction)
