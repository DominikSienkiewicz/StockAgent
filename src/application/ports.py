from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.application.report_council_history import InvestorHistory
from src.domain.analyst_consensus import AnalystConsensus
from src.domain.council import CouncilInput, CouncilVerdict, InvestorOpinion
from src.domain.earnings import EarningsEvent
from src.domain.insider_flow import InsiderFlowSnapshot
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.options_flow import OptionsFlowSnapshot
from src.domain.polish_macro import PolishMacroSnapshot
from src.domain.prediction import Prediction
from src.domain.quota import QuotaAlert
from src.domain.social_velocity import SocialVelocitySnapshot
from src.domain.subscriber import Subscriber
from src.domain.value_objects import AssetType, Fundamentals, Money


class MarketDataPort(ABC):
    """Źródło aktualnych cen rynkowych (np. Finnhub, Alpha Vantage)."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Money: ...


class SentimentPort(ABC):
    """Źródło metryk sentymentu finansowego (Alpha Vantage NEWS_SENTIMENT)."""

    @abstractmethod
    def get_social_score(self, symbol: str) -> dict[str, Any]: ...


class NewsPort(ABC):
    """Źródło artykułów newsowych (Alpha Vantage NEWS_SENTIMENT feed)."""

    @abstractmethod
    def get_news_context(self, symbol: str) -> list[dict[str, Any]]: ...


class RepositoryPort(ABC):
    """Persystencja: zapis predykcji, odczyt historii, dostęp do feature store."""

    @abstractmethod
    def get_last_price(self, symbol: str) -> Money | None:
        """Najnowszy snapshot ceny z `price_snapshots` (nie z prediction_logs)."""

    @abstractmethod
    def get_price_history(
        self, symbol: str, days: int
    ) -> list[tuple[datetime, Money]]:
        """Chronologiczna historia cen z `price_snapshots` (rosnąco po czasie),
        za ostatnie `days` dni. Zasila analizy portfelowe (drawdown, korelacje) —
        wszystko liczone z DARMOWYCH snapshotów, bez płatnych portów. Pusta lista
        gdy brak historii."""

    @abstractmethod
    def get_last_prediction_price(self, symbol: str) -> Money | None:
        """Cena OSTATNIEJ zalogowanej predykcji (`prediction_logs.price_at_prediction`).

        Referencja dla cechy `price_delta` w inference. Widok `ml_feature_store`
        liczy `price_delta` jako `LAG(price_at_prediction)` — czyli zmianę względem
        POPRZEDNIEJ predykcji. Fast Loop musi użyć tej samej referencji (nie
        ostatniego snapshotu ceny), inaczej cecha ma inny rozkład w treningu i
        w predykcji (train/serve skew). Zwraca None gdy brak wcześniejszej
        predykcji dla symbolu (pierwszy cykl — widok i tak takie wiersze odsiewa)."""

    @abstractmethod
    def save_price_snapshot(self, symbol: str, price: Money) -> None:
        """Zapisuje bieżącą cenę — wywoływane w KAŻDYM cyklu (rozwiązuje
        cold-start deadlock: następny cykl ma punkt odniesienia do delty)."""

    @abstractmethod
    def save_prediction(self, prediction: dict[str, Any]) -> str:
        """Zwraca ID zapisanej predykcji."""

    @abstractmethod
    def get_unverified_prediction(
        self, symbol: str, min_age_hours: int = 0
    ) -> Prediction | None:
        """NAJSTARSZA nierozliczona predykcja (bez rzeczywistej ceny), starsza
        niż `min_age_hours`. Oldest-first, by drenować zaległości — inaczej
        starsze nierozliczone predykcje nigdy nie wejdą do feature store.

        `min_age_hours` > 0 → bierze tylko predykcje starsze niż próg
        (timestamp ≤ now − min_age_hours). Chroni przed przedwczesną oceną:
        gdy ręczny `workflow_dispatch` nałoży się na scheduled run, świeżo
        zapisana predykcja nie zostanie oceniona po cenie sprzed kilku minut
        (co zatruwałoby accuracy_score i zawyżało hit-rate raportu)."""

    @abstractmethod
    def update_prediction_accuracy(
        self,
        prediction_id: str,
        actual_price: Decimal,
        accuracy_score: float,
        is_trend_correct: bool,
        insight: str,
    ) -> None:
        """Zamyka predykcję feedbackiem. `accuracy_score` (bliskość ceny do
        celu) napędza trening; `is_trend_correct` (zgodność kierunku) napędza
        trafność raportu — to dwie różne miary, nie wolno ich mylić."""

    @abstractmethod
    def get_feature_store_data(self, symbol: str) -> list[dict[str, Any]]:
        """Zwraca rekordy ze zmaterializowanego widoku ml_feature_store."""

    @abstractmethod
    def refresh_feature_store(self) -> None:
        """Odświeża zmaterializowany widok ml_feature_store (Slow Loop, przed treningiem)."""

    @abstractmethod
    def get_accuracy_stats(self, days: int) -> dict[str, Any]:
        """Statystyki trafności ostatnich `days` dni z prediction_logs."""

    @abstractmethod
    def get_recently_resolved_predictions(self, hours: int) -> list[dict[str, Any]]:
        """Predykcje z ocenionym kierunkiem (`is_trend_correct`) z ostatnich
        `hours` godzin."""

    @abstractmethod
    def get_resolved_predictions(self, days: int) -> list[dict[str, Any]]:
        """Predykcje z ocenionym kierunkiem z ostatnich `days` dni — szersze okno
        niż `get_recently_resolved_predictions` (godziny), do analiz track-recordu.

        Ten sam kształt wierszy co `get_recently_resolved_predictions`
        (symbol, predicted_trend, is_trend_correct, timestamp, reasoning_text,
        correction_insights, price_at_prediction, actual_price_after_12h), więc
        `parse_resolved_predictions` parsuje oba. JEDEN odczyt zasila krzywą
        kapitału (paper-PnL), panel „lessons learned" (correction_insights) i
        inne sekcje track-recordu — bez dublowania zapytań. Pusta lista gdy brak
        danych / błąd po stronie wywołującego."""

    @abstractmethod
    def get_cached_fundamentals(self, symbol: str) -> Fundamentals | None:
        """Zwraca fundamentale z cache jeśli `fetched_at` mieści się w
        FUNDAMENTALS_CACHE_TTL_HOURS. Filtr TTL po stronie repo —
        adapter nie zna polityki świeżości."""

    @abstractmethod
    def save_fundamentals(self, symbol: str, fundamentals: Fundamentals) -> None:
        """Upsert jednego wiersza per symbol (nadpisuje poprzedni snapshot)."""

    @abstractmethod
    def save_quota_alert(self, alert: QuotaAlert) -> None:
        """Persystuje pojedynczy alert wyczerpania limitu / subskrypcji.

        main_agent woła po cyklu dla każdego alertu zebranego przez
        `QuotaMonitor`. Tabela `quota_alerts` (migracja 010) jest
        audit trail; banner w mailu używa `get_recent_quota_alerts`.
        """

    @abstractmethod
    def get_recent_quota_alerts(self, hours: int) -> list[QuotaAlert]:
        """Zwraca alerty z ostatnich `hours` godzin, posortowane malejąco
        po `occurred_at`. Banner w mailu używa tej listy, by pokazać też
        alerty z poprzednich cykli, których jeszcze nie naprawiono."""

    @abstractmethod
    def find_similar_predictions(
        self, embedding: list[float], limit: int = 3, regime: str | None = None
    ) -> list[dict[str, Any]]:
        """Wyszukuje historyczne predykcje o najbardziej podobnym kontekście
        newsowym (similarity search nad `prediction_logs.embedding`, pgvector).

        Zasila RAG w `predict_node`: wstrzykuje "jak podobne sytuacje skończyły
        się w przeszłości" do promptu. Implementacja MUSI być graceful — gdy
        pgvector / RPC niedostępne, zwraca `[]` (predykcja działa bez RAG).
        Zwracane rekordy zawierają m.in. `news_summary`, `predicted_trend`,
        `is_trend_correct`, `correction_insights`.

        `regime` (Cross-Asset Vector Memory): gdy podany, retrieval preferuje
        analogi z tego samego reżimu rynku (tag zapisany przy embeddingu,
        migracja 017). None = bez filtra (zachowanie wsteczne)."""

    @abstractmethod
    def get_council_votes_for_prediction(
        self, prediction_id: str
    ) -> list[InvestorOpinion]:
        """Głosy rady zapisane dla danej predykcji (`council_votes`).

        Napędza counterfactual dissent-replay: gdy predykcja okazała się błędna,
        self-reflection sprawdza, który dysydent rady miał rację. Pusta lista =
        brak zapisanych głosów (rada padła / pominięta progiem)."""

    @abstractmethod
    def get_resolved_for_calibration(self, days: int) -> list[dict[str, Any]]:
        """#4 — rozliczone predykcje z `id`, `confidence_score`,
        `is_trend_correct`, `reasoning_text` do oceny kalibracji (Slow Loop).
        Pomija wiersze bez `confidence_score`/`is_trend_correct`."""

    @abstractmethod
    def save_calibration(
        self, prediction_id: str, score: float, insight: str
    ) -> None:
        """#4 — zapisuje ocenę kalibracji pewności (sędzia-LLM, Slow Loop):
        `confidence_calibration` (0-1, czy deklarowana pewność była uzasadniona)
        + `calibration_insight` na `prediction_logs` (migracja 016)."""

    @abstractmethod
    def get_persona_accuracy(
        self, asset_type: AssetType, window_days: int = 90
    ) -> dict[str, float]:
        """#3 — waga każdej persony rady wg jej historycznej trafności
        (rekomendacja vs realny ruch ceny), zmapowana do ~[0.5, 1.5].
        `{investor_name: waga}`. Pusta mapa = brak danych / RPC niedostępne
        → ważenie wyłączone (każda persona waży 1.0)."""

    @abstractmethod
    def get_council_vote_history(
        self,
        symbols: list[str],
        *,
        window_days: int = 30,
        investors: list[str] | None = None,
    ) -> list[InvestorHistory]:
        """U5 — historia głosów rady per inwestor na obserwowanych symbolach
        z ostatnich `window_days` dni (jeden `InvestorHistory` na parę
        inwestor×symbol, rekomendacje od najnowszej). Napędza panel drill-down
        w raporcie. `investors=None` = wszyscy. Pusta lista gdy brak danych."""

    @abstractmethod
    def save_council_votes(
        self,
        prediction_id: str,
        symbol: str,
        votes: list[InvestorOpinion],
    ) -> None:
        """Strukturalny audit trail rady — jedna linia per inwestor.

        Umożliwia odpytywanie "jak Burry głosował na NVDA w ostatnim miesiącu"
        bez parsowania JSONB blob `prediction_logs.council_verdict`. Pusta lista
        głosów to no-op (rada padła całkowicie / została pominięta progiem).
        """


class MLPredictionPort(ABC):
    """Lokalny model predykcyjny (np. XGBoost)."""

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Czy model jest gotów do predykcji (wagi załadowane / wytrenowane).

        Fast Loop sprawdza to przed wywołaniem `predict()` — przy cold-starcie
        (brak pliku modelu) używa baseline zamiast crashować."""

    @abstractmethod
    def predict(
        self, current_features: dict[str, float], current_price: Decimal
    ) -> Money:
        """Predykcja ceny docelowej 12h. Model przewiduje ZWROT — `current_price`
        jest potrzebna do rekonstrukcji ceny bezwzględnej (`cena*(1+zwrot)`)."""

    @abstractmethod
    def explain(self, current_features: dict[str, float]) -> dict[str, float]:
        """Q3 — znakowane wkłady per-cecha (SHAP-lite) w surową predykcję.

        Zwraca `{nazwa_cechy: wkład, ..., "_bias": base}`. Lokalny CPU, zero
        płatnych wywołań. Rzuca RuntimeError gdy model nietrenowany (jak predict)."""

    @abstractmethod
    def train(self, features: Any, target: Any) -> dict[str, Any]:
        """`features`/`target` to `pandas.DataFrame`/`Series` — typujemy luźno,
        by nie wprowadzać zależności od pandas w warstwie application."""

    @abstractmethod
    def walk_forward_report(
        self, features: Any, target: Any
    ) -> list[dict[str, float]]:
        """Per-fold metryki out-of-sample (walk-forward, expanding window) BEZ
        persystencji — zasila offline backtest harness (`main_backtest.py`).

        Reużywa tę samą logikę foldów co `train()` (finding #23), ale zwraca
        SZCZEGÓŁ każdego foldu zamiast średniej: lista dictów z kluczami
        `train_end`, `valid_end`, `n_train`, `n_valid`, `candidate_rmse`,
        `baseline_rmse` (zwrot=0), `hit_rate` (trafność kierunku). Czysto
        ewaluacyjna — nie zapisuje modelu, nie zmienia stanu adaptera."""


class LLMPort(ABC):
    """Model językowy do rozumowania i samokorekty (np. OpenAI, Anthropic)."""

    @abstractmethod
    def analyze(self, prompt: str) -> dict[str, Any]:
        """Główna analiza — zwraca ustrukturyzowany JSON wg output_schema z promptu."""

    @abstractmethod
    def analyze_mistake(self, prompt: str) -> str:
        """Diagnoza błędnej predykcji — zwraca tekstowy correction_insight."""


class EmbeddingPort(ABC):
    """Generowanie wektorów embeddingu dla tekstu (np. OpenAI embeddings).

    Wektor podsumowania newsów trafia do `prediction_logs.embedding` (pgvector),
    co umożliwia przyszły similarity search nad historycznymi sytuacjami."""

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...


class ReportNotifierPort(ABC):
    """Wysyłka raportów z cyklu (np. email przez Resend)."""

    @abstractmethod
    def send_report(self, subject: str, html_body: str, plain_text: str) -> None: ...


class AlertNotifierPort(ABC):
    """Real-time push pojedynczych alertów CRITICAL — odsprzężony od dziennego maila.

    Pozwala wystawić alert (np. wyczerpanie limitu, email nie poszedł) natychmiast
    w cyklu, nie czekając na następny raport dobowy. Adapter formatuje alerty i
    deleguje do dowolnego transportu (Telegram / Slack / Resend)."""

    @abstractmethod
    def send_alert(self, alerts: list[QuotaAlert]) -> None: ...


class AdvisoryCouncilPort(ABC):
    """Rada doradcza inwestorów — N równoległych analiz person + konsensus.

    Liczba person jest data-driven (pliki JSON w `council_personas_dir`),
    więc nie podajemy jej tu na sztywno."""

    @abstractmethod
    def analyze(self, symbol: str, data: CouncilInput) -> CouncilVerdict: ...


class FundamentalsPort(ABC):
    """Źródło danych fundamentalnych spółek (P/E, PEG, EPS growth).

    Zwraca None dla aktywów bez sensownych fundamentów (ETF-y, błąd API,
    brak danych w źródle)."""

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Fundamentals | None: ...


class MacroIndicatorsPort(ABC):
    """Wskaźniki makroekonomiczne polskiego rynku (kurs PLN, ew. rentowności).

    Zwraca None, gdy źródło niedostępne — wywołujący traktuje brak danych
    jako "neutralne tło" i nie blokuje raportu."""

    @abstractmethod
    def fetch_polish_macro(self) -> PolishMacroSnapshot | None: ...


class SubscriberRepositoryPort(ABC):
    """Źródło listy subskrybentów raportu (U2 — personalizowany fan-out).

    Każdy subskrybent ma własny podzbiór symboli; analiza leci RAZ nad unią,
    a raport jest tylko tnięty per odbiorca. Zwraca [] gdy brak / błąd."""

    @abstractmethod
    def list_subscribers(self) -> list[Subscriber]: ...


class WebPublisherPort(ABC):
    """Publikacja raportu HTML jako statyczny artefakt web (U3 — read-only
    dashboard, alternatywa dla maila). Zwraca URL/ścieżkę albo "" gdy
    wyłączone / błąd (publikacja nigdy nie wywala cyklu)."""

    @abstractmethod
    def publish(self, html: str) -> str: ...


@dataclass(frozen=True)
class Tool:
    """#6 — read-only narzędzie udostępniane modelowi w pętli tool-use.

    `func` przyjmuje argumenty wywołania (dict) i zwraca JSON-safe wynik (dict).
    `parameters` to JSON Schema argumentów (dla SDK function-calling)."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[[dict[str, Any]], dict[str, Any]]


class ToolUseLLMPort(ABC):
    """#6 — LLM z pętlą tool-use: model może na żądanie wywołać read-only toole
    (fundamenty, RAG, makro) zanim wyda werdykt. Implementacja MUSI mieć twardy
    cap rund (FinOps). Zwraca ustrukturyzowany JSON jak `LLMPort.analyze`."""

    @abstractmethod
    def analyze_with_tools(
        self, prompt: str, tools: Sequence[Tool]
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Nowe źródła alpha (kategoria „Dane"). Każdy port ma adapter (wolne źródło)
# i Null-adapter (default Fast Loop / brak klucza → zero płatnych wywołań).
# ---------------------------------------------------------------------------


class InsiderFlowPort(ABC):
    """Przepływy insiderów (SEC Form 4 / 13F) per symbol.

    Zwraca None gdy brak danych / źródło niedostępne. Null-adapter (Fast Loop /
    brak konfiguracji) zwraca None bez żadnego wywołania sieciowego."""

    @abstractmethod
    def get_insider_flow(self, symbol: str) -> InsiderFlowSnapshot | None: ...


class EarningsCalendarPort(ABC):
    """Najbliższe zdarzenie wyników (earnings) per symbol.

    Zwraca None gdy brak nadchodzącego raportu / brak danych."""

    @abstractmethod
    def get_next_earnings(self, symbol: str) -> EarningsEvent | None: ...


class OptionsFlowPort(ABC):
    """Przepływy opcji i IV per symbol (put/call ratio, implied vol).

    Brak darmowego źródła → domyślnie Null-adapter. Zwraca None gdy brak danych."""

    @abstractmethod
    def get_options_flow(self, symbol: str) -> OptionsFlowSnapshot | None: ...


class SocialVelocityPort(ABC):
    """Prędkość społeczna (Reddit/X) per symbol — wzmianki 24h vs baseline.

    Domyślnie Null-adapter (free-tier/OAuth opcjonalne). Zwraca None gdy brak."""

    @abstractmethod
    def get_social_velocity(self, symbol: str) -> SocialVelocitySnapshot | None: ...


class MacroRatesPort(ABC):
    """Stopy i krzywa rentowności (FRED) — raz na cykl, nie per symbol.

    Zwraca None gdy źródło niedostępne / brak klucza."""

    @abstractmethod
    def fetch_yield_curve(self) -> YieldCurveSnapshot | None: ...


class AnalystConsensusPort(ABC):
    """Konsensus analityków i cel cenowy per symbol.

    Zwraca None gdy brak pokrycia analitycznego / źródło niedostępne."""

    @abstractmethod
    def get_consensus(self, symbol: str) -> AnalystConsensus | None: ...
