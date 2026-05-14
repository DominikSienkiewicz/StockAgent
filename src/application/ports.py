from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

from src.domain.prediction import Prediction
from src.domain.value_objects import Money


class MarketDataPort(ABC):
    """Źródło aktualnych cen rynkowych (np. Finnhub, Alpha Vantage)."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Money: ...


class SentimentPort(ABC):
    """Źródło metryk sentymentu społecznego (np. LunarCrush)."""

    @abstractmethod
    def get_social_score(self, symbol: str) -> dict[str, Any]: ...


class NewsPort(ABC):
    """Źródło artykułów newsowych (np. NewsAPI, SerpApi)."""

    @abstractmethod
    def get_news_context(self, symbol: str) -> list[dict[str, Any]]: ...


class RepositoryPort(ABC):
    """Persystencja: zapis predykcji, odczyt historii, dostęp do feature store."""

    @abstractmethod
    def get_last_price(self, symbol: str) -> Money | None:
        """Najnowszy snapshot ceny z `price_snapshots` (nie z prediction_logs)."""

    @abstractmethod
    def save_price_snapshot(self, symbol: str, price: Money) -> None:
        """Zapisuje bieżącą cenę — wywoływane w KAŻDYM cyklu (rozwiązuje
        cold-start deadlock: następny cykl ma punkt odniesienia do delty)."""

    @abstractmethod
    def save_prediction(self, prediction: dict[str, Any]) -> str:
        """Zwraca ID zapisanej predykcji."""

    @abstractmethod
    def get_unverified_prediction(self, symbol: str) -> Prediction | None:
        """Najnowsza predykcja sprzed 12-24h bez przypisanej rzeczywistej ceny."""

    @abstractmethod
    def update_prediction_accuracy(
        self,
        prediction_id: str,
        actual_price: Decimal,
        accuracy_score: float,
        insight: str,
    ) -> None: ...

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
        """Predykcje z wypełnionym `accuracy_score` w ostatnich `hours` godzin."""


class MLPredictionPort(ABC):
    """Lokalny model predykcyjny (np. XGBoost)."""

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Czy model jest gotów do predykcji (wagi załadowane / wytrenowane).

        Fast Loop sprawdza to przed wywołaniem `predict()` — przy cold-starcie
        (brak pliku modelu) używa baseline zamiast crashować."""

    @abstractmethod
    def predict(self, current_features: dict[str, float]) -> Money: ...

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
