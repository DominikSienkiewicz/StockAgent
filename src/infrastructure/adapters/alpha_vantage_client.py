"""Wspólny klient dla Alpha Vantage NEWS_SENTIMENT.

Cel: jedno wywołanie REST per cykl decyzyjny (Fast Loop) dla wszystkich
symboli — odpowiedź zawiera artykuły wraz z `ticker_sentiment` per ticker.
Dwa cienkie adaptery (News + Sentiment) konsumują ten sam buffer.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.infrastructure.adapters._http import (
    build_session,
    raise_for_status_redacted,
    safe_get,
)

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

# Okno, które obiecuje nazwa cechy `news_volume_24h`. Bez `time_from` AV zwraca
# `limit` najnowszych artykułów z dowolnego okresu — dla NVDA to kilka godzin,
# dla rzadko opisywanego ETF-u miesiące — więc licznik nie mierzył 24 h.
NEWS_WINDOW = timedelta(hours=24)
# Format parametru `time_from` w NEWS_SENTIMENT (minuty) i pola `time_published`.
_TIME_FROM_FORMAT = "%Y%m%dT%H%M"
_TIME_PUBLISHED_FORMAT = "%Y%m%dT%H%M%S"

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
        news_window: timedelta | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """`news_window` ogranicza feed do artykułów nie starszych niż okno:
        idzie do AV jako `time_from` (te same requesty) i jest powtórzone
        filtrem po `time_published`. `None` = bez okna (dotychczasowe zachowanie).
        """
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
        self._news_window = news_window
        self._clock = clock if clock is not None else _utc_now
        # Początek okna liczony RAZ na fetch — `time_from` i filtr muszą mieć tę
        # samą granicę, inaczej artykuł z krawędzi wpadałby tylko do jednego.
        self._window_start: datetime | None = None
        # Sygnał degradacji wystawiany na zewnątrz: gdy wszystkie klucze
        # zwróciły rate-limit, feed jest (potencjalnie) niekompletny. Bez
        # tego sygnału pusty feed wygląda identycznie jak "spokojny dzień".
        self._degraded_reason: str | None = None
        self._session = build_session()
        # Serializuje leniwy fetch feedu — bez tego współbieżni callerzy (równoległe
        # symbole) startują N duplikatów cold-startu, zjadając dzienny limit AV.
        self._fetch_lock = threading.Lock()

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
        # Fast path bez locka: feed już wypełniony.
        if self._cached_feed is not None:
            return self._cached_feed
        with self._fetch_lock:
            # Re-check pod lockiem: inny wątek mógł wypełnić feed, gdy czekaliśmy.
            if self._cached_feed is not None:
                return self._cached_feed
            return self._fetch_locked()

    def _fetch_locked(self) -> list[dict[str, Any]]:
        if self._news_window is not None:
            self._window_start = self._clock() - self._news_window
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

            self._collect_unique_articles(payload, combined, seen_urls)

        self._cached_feed = combined
        return self._cached_feed

    @staticmethod
    def _collect_unique_articles(
        payload: dict[str, Any],
        combined: list[dict[str, Any]],
        seen_urls: set[str],
    ) -> None:
        """Dokleja artykuły z batcha do `combined`, deduplikując po URL."""
        for article in payload.get("feed", []):
            url = article.get("url")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            combined.append(article)

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
            # MAIL-15: treść alertu idzie do maila po polsku; angielska
            # diagnoza jest w logger.error powyżej.
            message=(
                f"Wszystkie klucze Alpha Vantage ({len(self._api_keys)}) wyczerpały "
                "dzienny limit 25 zapytań. Feed newsów i sentymentu jest w tym "
                "cyklu niepełny."
            ),
            action=(
                "Dodaj kolejny klucz w ALPHA_VANTAGE_API_KEYS (darmowy na "
                "alphavantage.co) albo poczekaj na reset o północy UTC."
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
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(tickers),
            "limit": str(self._limit),
            "sort": "LATEST",
            "apikey": api_key,
        }
        if self._window_start is not None:
            params["time_from"] = self._window_start.strftime(_TIME_FROM_FORMAT)
        response = safe_get(
            self._session,
            f"{self._base_url}/query",
            params=params,
            timeout=self._timeout,
        )
        raise_for_status_redacted(response)
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
            if not self._within_window(item):
                continue
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

    def _within_window(self, item: dict[str, Any]) -> bool:
        """Drugi bezpiecznik okna: AV mógłby zignorować `time_from`.

        Artykuł bez czytelnego `time_published` odpada — nie da się wykazać,
        że jest z okna, a liczy się do cechy `news_volume_24h`."""
        if self._window_start is None:
            return True
        published = _parse_time_published(item.get("time_published"))
        if published is None:
            logger.warning(
                "Alpha Vantage article without parsable time_published (%r) — "
                "dropped from the news window.",
                item.get("time_published"),
            )
            return False
        return published >= self._window_start

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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_time_published(raw: object) -> datetime | None:
    """`20261018T093000` (czas UTC wg AV) → aware datetime; None gdy nieczytelny."""
    try:
        return datetime.strptime(str(raw), _TIME_PUBLISHED_FORMAT).replace(tzinfo=UTC)
    except ValueError:
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
