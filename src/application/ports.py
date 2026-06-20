from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.domain.council import CouncilInput, CouncilVerdict, InvestorOpinion
from src.domain.polish_macro import PolishMacroSnapshot
from src.domain.prediction import Prediction
from src.domain.quota import QuotaAlert
from src.domain.value_objects import Fundamentals, Money


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
        self, embedding: list[float], limit: int = 3
    ) -> list[dict[str, Any]]:
        """Wyszukuje historyczne predykcje o najbardziej podobnym kontekście
        newsowym (similarity search nad `prediction_logs.embedding`, pgvector).

        Zasila RAG w `predict_node`: wstrzykuje "jak podobne sytuacje skończyły
        się w przeszłości" do promptu. Implementacja MUSI być graceful — gdy
        pgvector / RPC niedostępne, zwraca `[]` (predykcja działa bez RAG).
        Zwracane rekordy zawierają m.in. `news_summary`, `predicted_trend`,
        `is_trend_correct`, `correction_insights`."""

    @abstractmethod
    def get_council_votes_for_prediction(
        self, prediction_id: str
    ) -> list[InvestorOpinion]:
        """Głosy rady zapisane dla danej predykcji (`council_votes`).

        Napędza counterfactual dissent-replay: gdy predykcja okazała się błędna,
        self-reflection sprawdza, który dysydent rady miał rację. Pusta lista =
        brak zapisanych głosów (rada padła / pominięta progiem)."""

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
    def train(self, features: Any, target: Any) -> dict[str, Any]:
        """`features`/`target` to `pandas.DataFrame`/`Series` — typujemy luźno,
        by nie wprowadzać zależności od pandas w warstwie application."""


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
