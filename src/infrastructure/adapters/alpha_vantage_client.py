"""Wspólny klient dla Alpha Vantage NEWS_SENTIMENT.

Cel: jedno wywołanie REST per cykl decyzyjny (Fast Loop) dla wszystkich
symboli — odpowiedź zawiera artykuły wraz z `ticker_sentiment` per ticker.
Dwa cienkie adaptery (News + Sentiment) konsumują ten sam buffer.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.infrastructure.adapters._http import build_session

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://www.alphavantage.co"
DEFAULT_TIMEOUT = 30
DEFAULT_RELEVANCE_THRESHOLD = 0.5   # odsiewamy artykuły gdzie ticker jest tylko wzmianką
DEFAULT_LIMIT = 50                   # max artykułów per batch (free tier OK)

# AV NEWS_SENTIMENT z parametrem `tickers=A,B,C` filtruje artykuły wspominające
# WSZYSTKIE tickery (AND, nie OR — z dokumentacji). Heterogeniczna lista
# (AAPL, AMZN, MSFT...) zwróci pusty feed. Dlatego batchujemy PER TICKER:
# robimy osobny request dla każdego symbolu i scalamy wyniki.
DEFAULT_BATCH_SIZE = 1
# AV free tier: max 1 req/sec (per klucz).
INTER_REQUEST_DELAY_SECONDS = 1.1

HIGH_RELEVANCE_BAR = 0.8

# Alpha Vantage akceptuje tylko alphanumeric + ':' '_' '-' (brak kropki!).
# Symbole z kropką (np. CSPX.L = LSE notation) zostają pominięte z warning.
_VALID_TICKER_RE = re.compile(r"^[A-Za-z0-9:_-]+$")


class AlphaVantageClient:
    """Lazy single-batch fetcher z in-memory cache na cały lifecycle instancji.

    Pierwsza wywołanie `articles_for` / `sentiment_for` triggeruje JEDEN
    request REST do AV z `tickers=AAPL,AMD,NVDA,...`. Wszystkie kolejne
    wywołania (różne symbole, różne adaptery) korzystają z buffera.
    """

    def __init__(
        self,
        api_keys: Iterable[str] | str,
        symbols: Iterable[str],
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        limit: int = DEFAULT_LIMIT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        crypto_symbols: Iterable[str] | None = None,
        quota_monitor: QuotaMonitor | None = None,
    ) -> None:
        # Akceptujemy pojedynczy klucz (str) lub listę — rotacja przy rate-limit.
        if isinstance(api_keys, str):
            self._api_keys = [api_keys]
        else:
            self._api_keys = list(api_keys)
        if not self._api_keys:
            raise ValueError("AlphaVantageClient requires at least one API key.")
        # Krypto: AV oczekuje prefiksu `CRYPTO:` w `tickers`. Trzymamy mapowanie
        # plain → wire (BTC → CRYPTO:BTC) i zamieniamy zarówno w requestach
        # jak i przy lookupie ticker_sentiment.
        self._crypto_symbols: set[str] = set(crypto_symbols or ())
        self._symbols = [
            self._to_wire(s) for s in self._filter_supported(list(symbols))
        ]
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._threshold = relevance_threshold
        self._limit = limit
        self._batch_size = batch_size
        self._cached_feed: list[dict[str, Any]] | None = None
        # Indeks aktualnie używanego klucza — przesuwany przy rate-limit.
        self._active_key_idx = 0
        self._quota_monitor = quota_monitor
        # Sygnał degradacji wystawiany na zewnątrz: gdy wszystkie klucze
        # zwróciły rate-limit, feed jest (potencjalnie) niekompletny. Bez
        # tego sygnału pusty feed wygląda identycznie jak "spokojny dzień".
        self._degraded_reason: str | None = None
        self._session = build_session()

    @property
    def degraded_reason(self) -> str | None:
        """Niesie informację o degradacji feedu w bieżącym cyklu.

        Wartości: `"av_keys_exhausted"` gdy rotacja wyczerpała wszystkie klucze
        i co najmniej jeden batch nie został pobrany. `None` gdy wszystko OK.
        Konsumowane przez `AlphaVantageSentimentAdapter` → graph
        → `prediction_logs.data_quality_flags`.
        """
        return self._degraded_reason

    @staticmethod
    def _filter_supported(symbols: list[str]) -> list[str]:
        """AV odrzuca cały batch przy choć jednym niewspieranym tickerze.

        Filtrujemy tickery zawierające kropkę (np. CSPX.L) — i tak Finnhub
        free tier ich nie obsługuje, więc to spójne ograniczenie.
        """
        valid: list[str] = []
        skipped: list[str] = []
        for sym in symbols:
            if _VALID_TICKER_RE.match(sym):
                valid.append(sym)
            else:
                skipped.append(sym)
        if skipped:
            logger.warning(
                "Alpha Vantage doesn't support tickers with non-alphanumeric chars "
                "(np. dot-notation LSE/ETF jak CSPX.L); sentiment & news będą puste "
                "dla: %s. Aby ich uniknąć w portfolio, usuń z SYMBOLS env var.",
                skipped,
            )
        return valid

    # -----------------------------------------------------------------------
    # Fetch & cache
    # -----------------------------------------------------------------------

    def _fetch(self) -> list[dict[str, Any]]:
        if self._cached_feed is not None:
            return self._cached_feed

        combined: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        batches = [
            self._symbols[i : i + self._batch_size]
            for i in range(0, len(self._symbols), self._batch_size)
        ]

        for idx, batch in enumerate(batches):
            # Short-circuit: jeśli rotacja w poprzednim batchu wyczerpała
            # WSZYSTKIE klucze, kolejne batche i tak dostałyby None — nie
            # re-chodzimy po martwych kluczach i nie śpimy między nimi.
            if self._active_key_idx >= len(self._api_keys):
                self._mark_keys_exhausted()
                break

            # Pacing 1 req/sec dotyczy tylko realnych requestów: śpimy przed
            # batchem >0 dopiero gdy mamy żywy klucz, do którego pójdzie request.
            if idx > 0:
                time.sleep(INTER_REQUEST_DELAY_SECONDS)

            payload = self._fetch_batch_with_rotation(batch)
            if payload is None:
                # Wszystkie klucze wyczerpane — graceful degradation:
                # zostawiamy częściowy feed z poprzednich batchów.
                self._mark_keys_exhausted()
                break

            for article in payload.get("feed", []):
                url = article.get("url")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                combined.append(article)

        self._cached_feed = combined
        return self._cached_feed

    def _mark_keys_exhausted(self) -> None:
        """Wystawia sygnał degradacji + CRITICAL alert (idempotentnie).

        Wołane zarówno gdy batch zwrócił None (rotacja wyczerpała klucze
        w trakcie), jak i przy short-circuit kolejnych batchów po wcześniejszym
        wyczerpaniu. Guard na `_degraded_reason` zapobiega podwójnemu alertowi.
        """
        if self._degraded_reason is not None:
            return
        logger.error(
            "All %d Alpha Vantage API keys exhausted — returning partial feed.",
            len(self._api_keys),
        )
        self._degraded_reason = "av_keys_exhausted"
        self._emit_quota_alert(
            severity=QuotaSeverity.CRITICAL,
            message=(
                f"All {len(self._api_keys)} Alpha Vantage API keys hit "
                "the daily 25 req/day limit. News + sentiment feed is "
                "incomplete for this cycle."
            ),
            action=(
                "Add another key in ALPHA_VANTAGE_API_KEYS (free at "
                "alphavantage.co), or wait for UTC midnight reset."
            ),
        )

    def _fetch_batch_with_rotation(
        self, tickers: list[str]
    ) -> dict[str, Any] | None:
        """Próbuje wszystkich kluczy od `_active_key_idx` w górę.

        Zwraca payload przy sukcesie (i aktualizuje `_active_key_idx`),
        albo `None` gdy wszystkie klucze wyczerpane.
        """
        while self._active_key_idx < len(self._api_keys):
            key = self._api_keys[self._active_key_idx]
            payload = self._fetch_batch(tickers, key)

            if _is_rate_limited(payload):
                logger.warning(
                    "Alpha Vantage key #%d rate-limited: %s — rotating to next.",
                    self._active_key_idx + 1,
                    str(payload.get("Information", ""))[:160],
                )
                self._active_key_idx += 1
                # Bez sleepa: do martwego klucza nie poszedł realny request,
                # więc pacing 1 req/sec go nie dotyczy. Re-chodzenie po
                # wyczerpanych kluczach z 1.1s przerwą marnowałoby wall-clock
                # na quota-wyczerpany dzień. Pacing pilnuje tylko realnych
                # requestów (w `_fetch`, między batchami).
                continue

            return payload

        return None

    def _emit_quota_alert(
        self, severity: QuotaSeverity, message: str, action: str
    ) -> None:
        """Wystawia alert do QuotaMonitor jeśli ten jest skonfigurowany."""
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source="Alpha Vantage",
                severity=severity,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )

    def _fetch_batch(self, tickers: list[str], api_key: str) -> dict[str, Any]:
        response = self._session.get(
            f"{self._base_url}/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ",".join(tickers),
                "limit": str(self._limit),
                "apikey": api_key,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def _to_wire(self, symbol: str) -> str:
        """Plain ticker (BTC) → wire format wymagany przez AV (CRYPTO:BTC).

        Dla zwykłych akcji zwraca ticker bez zmian. Mapowanie idzie tylko
        w jedną stronę — odwrotne przejście robi się przez `self._crypto_symbols`
        przy parsowaniu ticker_sentiment.
        """
        if symbol in self._crypto_symbols:
            return f"CRYPTO:{symbol}"
        return symbol

    def articles_for(self, symbol: str) -> list[dict[str, Any]]:
        """Zwraca artykuły gdzie `symbol` ma relevance ≥ threshold,
        wzbogacone o per-ticker fields (sentyment, relevance)."""
        wire_symbol = self._to_wire(symbol)
        out: list[dict[str, Any]] = []
        for item in self._fetch():
            ticker_data = self._ticker_block(item, wire_symbol)
            if ticker_data is None:
                continue
            if ticker_data["relevance"] < self._threshold:
                continue
            out.append(
                {
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "published_at": item.get("time_published"),
                    "description": item.get("summary"),
                    "relevance_score": ticker_data["relevance"],
                    "ticker_sentiment_score": ticker_data["sentiment"],
                }
            )
        return out

    def sentiment_for(self, symbol: str) -> dict[str, Any]:
        """Wieloargumentowa agregata sentymentu dla pojedynczego symbolu.

        Zwraca multi-feature dict — wszystkie pola są bezpośrednio
        konsumowalne przez warstwę aplikacji i jako features ML.
        """
        articles = self.articles_for(symbol)

        if not articles:
            return {
                "av_sentiment_score": 0.0,
                "av_relevance_avg": 0.0,
                "news_volume_24h": 0,
                "high_relevance_count": 0,
                "av_sentiment_label": "Neutral",
            }

        total_weight = sum(a["relevance_score"] for a in articles) or 1.0
        weighted_sentiment = (
            sum(a["ticker_sentiment_score"] * a["relevance_score"] for a in articles)
            / total_weight
        )
        avg_relevance = sum(a["relevance_score"] for a in articles) / len(articles)
        high_rel = sum(1 for a in articles if a["relevance_score"] >= HIGH_RELEVANCE_BAR)

        return {
            "av_sentiment_score": round(weighted_sentiment, 4),
            "av_relevance_avg": round(avg_relevance, 4),
            "news_volume_24h": len(articles),
            "high_relevance_count": high_rel,
            "av_sentiment_label": _score_to_label(weighted_sentiment),
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _ticker_block(item: dict[str, Any], symbol: str) -> dict[str, float] | None:
        for ts in item.get("ticker_sentiment", []) or []:
            if ts.get("ticker") == symbol:
                try:
                    return {
                        "relevance": float(ts.get("relevance_score", 0)),
                        "sentiment": float(ts.get("ticker_sentiment_score", 0)),
                    }
                except (TypeError, ValueError):
                    return None
        return None


def _is_rate_limited(payload: dict[str, Any]) -> bool:
    """AV w free tier sygnalizuje rate-limit polem `Information` w 200 OK,
    bez wypełnionego `feed`. Wykrywamy ten wzorzec, by rotować klucz."""
    return "Information" in payload and "feed" not in payload


def _score_to_label(score: float) -> str:
    """Mapowanie AV score → label (zgodne z konwencją AV: bins co 0.15)."""
    if score <= -0.35:
        return "Bearish"
    if score <= -0.15:
        return "Somewhat-Bearish"
    if score < 0.15:
        return "Neutral"
    if score < 0.35:
        return "Somewhat-Bullish"
    return "Bullish"
