"""Wspólny klient dla Alpha Vantage NEWS_SENTIMENT.

Dwa tryby (FIN-04): wsadowy (domyślny) — pierwszy symbol za bramką pobiera
feed WSZYSTKICH tickerów, kolejne czytają bufor; leniwy per symbol
(`lazy_per_symbol=True`) — request idzie tylko dla symbolu, o który pyta graf,
czyli za bramką volatility. Dwa cienkie adaptery (News + Sentiment)
konsumują ten sam bufor.

ARCH-33: błąd sieci, statusu i nieparsowalna treść wychodzą w taksonomii portów
(`ProviderUnavailable`, `QuotaExhausted`, `MalformedPayload`), wciąż jako typy
`requests` — ścieżka częściowego feedu (DAT-08) łapie `PortError`.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from src.application.errors import PortError
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.infrastructure.adapters._alpha_vantage_common import (
    NOTE_BACKOFF_SECONDS,
    is_frequency_limit,
    is_soft_limit,
    soft_limit_text,
)
from src.infrastructure.adapters._http import (
    build_session,
    json_payload,
    raise_for_status_translated,
    translated_get,
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

# Klucz danych NEWS_SENTIMENT — jego obecność odróżnia feed od soft-limitu (DAT-06).
_NEWS_DATA_KEYS = ("feed",)

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


def supported_tickers(symbols: Iterable[str]) -> list[str]:
    """Tickery, które AV przyjmuje w `tickers` (bez kropki, np. CSPX.L odpada).
    Z tej listy klient pobiera feed; FIN-06 liczy z niej requesty cyklu."""
    return [sym for sym in symbols if _VALID_TICKER_RE.match(sym)]


class AlphaVantageClient:
    """Fetcher NEWS_SENTIMENT z in-memory cache na cały lifecycle instancji.

    Tryb wsadowy: pierwsze `articles_for` / `sentiment_for` pobiera feed dla
    wszystkich skonfigurowanych tickerów (request per ticker, AND w `tickers`),
    a kolejne wywołania czytają scalony bufor. Tryb leniwy (`lazy_per_symbol`):
    jeden request per pytany symbol, cache per wire-symbol — symbol pod bramką
    nie kosztuje nic, a `news_volume_24h` liczy tylko jego własny feed.
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
        lazy_per_symbol: bool = False,
    ) -> None:
        """`news_window` ogranicza feed do artykułów nie starszych niż okno:
        idzie do AV jako `time_from` (te same requesty) i jest powtórzone
        filtrem po `time_published`. `None` = bez okna (dotychczasowe zachowanie).

        `lazy_per_symbol=True` (FIN-04): request tylko dla pytanego symbolu ze
        skonfigurowanej listy (inny ticker → `[]` bez HTTP), globalny pacing
        `INTER_REQUEST_DELAY_SECONDS` między requestami wszystkich wątków
        i degradacja per symbol w `sentiment_for` zamiast `degraded_reason`.
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
        # DAT-07: degradacja PER SYMBOL (wire ticker → powód), np. payload bez
        # `feed` i bez markera limitu. `sentiment_for` dokłada ją tylko temu
        # symbolowi — flaga cykliczna oflagowałaby wszystkie.
        self._symbol_degraded_reasons: dict[str, str] = {}
        # DAT-06: ostatni soft-limit przed wyczerpaniem był minutowy (`Note`) —
        # wtedy alert nie może mówić o dziennym limicie 25 zapytań.
        self._exhausted_by_frequency_limit = False
        self._keys_exhausted_reported = False
        self._session = build_session()
        # Serializuje leniwy fetch feedu — bez tego współbieżni callerzy (równoległe
        # symbole) startują N duplikatów cold-startu, zjadając dzienny limit AV.
        self._fetch_lock = threading.Lock()
        # Rotacja klucza i alert o wyczerpaniu: dwa wątki z limitem na tym samym
        # kluczu przesuwają indeks RAZ — inaczej przeskoczyłyby żywy klucz.
        self._rotation_lock = threading.Lock()
        # FIN-04: tryb leniwy — cache per wire-symbol z lockiem per symbol
        # (jeden request mimo współbieżnych callerów) i jeden pacer na instancję.
        self._lazy_per_symbol = lazy_per_symbol
        self._symbol_set = frozenset(self._symbols)
        self._symbol_feeds: dict[str, list[dict[str, Any]]] = {}
        self._symbol_locks: dict[str, threading.Lock] = {}
        self._symbol_locks_guard = threading.Lock()
        self._window_lock = threading.Lock()
        self._pace_lock = threading.Lock()
        self._last_request_at: float | None = None

    @property
    def degraded_reason(self) -> str | None:
        """Niesie informację o degradacji feedu w bieżącym cyklu.

        Wartości: `"av_keys_exhausted"` gdy rotacja wyczerpała wszystkie klucze
        i co najmniej jeden batch nie został pobrany, `"av_fetch_error"` gdy błąd
        sieci/HTTP przerwał fetch (feed częściowy, DAT-08). `None` gdy wszystko OK.
        Konsumowane przez `AlphaVantageSentimentAdapter` → graph
        → `prediction_logs.data_quality_flags`. W trybie leniwym zawsze `None`:
        symbol pobrany przed wyczerpaniem kluczy ma pełne dane, więc powód
        jedzie per symbol w `sentiment_for`.
        """
        return self._degraded_reason

    @staticmethod
    def _filter_supported(symbols: list[str]) -> list[str]:
        """AV odrzuca cały batch przy choć jednym niewspieranym tickerze.

        Filtrujemy tickery zawierające kropkę (np. CSPX.L) — i tak Finnhub
        free tier ich nie obsługuje, więc to spójne ograniczenie.
        """
        valid = supported_tickers(symbols)
        skipped = [sym for sym in symbols if sym not in valid]
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

    def _feed_for(self, wire_symbol: str) -> list[dict[str, Any]]:
        if self._lazy_per_symbol:
            return self._fetch_symbol(wire_symbol)
        return self._fetch()

    def _fetch_symbol(self, wire_symbol: str) -> list[dict[str, Any]]:
        """FIN-04: feed jednego symbolu — fast path bez locka, potem lock symbolu."""
        cached = self._symbol_feeds.get(wire_symbol)
        if cached is not None:
            return cached
        if wire_symbol not in self._symbol_set:
            # Ticker spoza listy albo odrzucony przez AV (CSPX.L): zero HTTP.
            return []
        with self._symbol_lock(wire_symbol):
            cached = self._symbol_feeds.get(wire_symbol)
            if cached is not None:
                return cached
            return self._fetch_symbol_locked(wire_symbol)

    def _symbol_lock(self, wire_symbol: str) -> threading.Lock:
        with self._symbol_locks_guard:
            return self._symbol_locks.setdefault(wire_symbol, threading.Lock())

    def _fetch_symbol_locked(self, wire_symbol: str) -> list[dict[str, Any]]:
        self._ensure_window_start()
        feed: list[dict[str, Any]] = []
        try:
            # Klucze wyczerpane wcześniej w cyklu → None bez HTTP na martwym kluczu.
            payload = self._fetch_batch_with_rotation([wire_symbol])
        except PortError as exc:
            # DAT-08 per symbol: bez ponownego requestu w tym cyklu.
            logger.error(
                "Alpha Vantage NEWS_SENTIMENT request failed for %s (%s) — "
                "the symbol keeps an empty feed for the rest of the cycle.",
                wire_symbol,
                type(exc).__name__,
            )
            self._symbol_degraded_reasons.setdefault(wire_symbol, "av_fetch_error")
            self._symbol_feeds[wire_symbol] = feed
            raise
        if payload is None:
            self._mark_keys_exhausted()
            self._symbol_degraded_reasons.setdefault(wire_symbol, "av_keys_exhausted")
        else:
            if "feed" not in payload:
                self._mark_unexpected_payload([wire_symbol], payload)
            feed = list(payload.get("feed", []))
        self._symbol_feeds[wire_symbol] = feed
        return feed

    def _ensure_window_start(self) -> None:
        """Jedno okno na cykl: wszystkie symbole dostają ten sam `time_from`,
        liczony przy pierwszym requeście (jak w trybie wsadowym)."""
        if self._news_window is None:
            return
        with self._window_lock:
            if self._window_start is None:
                self._window_start = self._clock() - self._news_window

    def _pace(self) -> None:
        """Globalny pacing trybu leniwego: odstęp ≥ `INTER_REQUEST_DELAY_SECONDS`
        między realnymi requestami WSZYSTKICH wątków. Bez niego równoległe
        symbole (`symbol_concurrency` > 1) strzelałyby burstem i paliły klucze."""
        with self._pace_lock:
            if self._last_request_at is not None:
                wait = INTER_REQUEST_DELAY_SECONDS - (time.monotonic() - self._last_request_at)
                if wait > 0:
                    time.sleep(wait)
            self._last_request_at = time.monotonic()

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

            try:
                payload = self._fetch_batch_with_rotation(batch)
            except PortError as exc:
                self._cache_partial_feed_after_error(batch, exc, combined)
                raise
            if payload is None:
                # Wszystkie klucze wyczerpane — graceful degradation:
                # zostawiamy częściowy feed z poprzednich batchów.
                self._mark_keys_exhausted()
                break

            if "feed" not in payload:
                self._mark_unexpected_payload(batch, payload)
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

    def _cache_partial_feed_after_error(
        self,
        tickers: list[str],
        exc: PortError,
        combined: list[dict[str, Any]],
    ) -> None:
        """DAT-08: błąd sieci/HTTP w środku fetchu zostawia w cache to, co już
        przyszło. Bez tego `_cached_feed` zostaje None i KAŻDY kolejny symbol
        startuje pełny fetch od nowa, paląc dzienny limit kluczy. Wołający
        rzuca wyjątek dalej (kontrakt bez zmian); log niesie tylko typ wyjątku,
        bo jego treść bywa z URL-em zapytania."""
        logger.error(
            "Alpha Vantage NEWS_SENTIMENT request failed for %s (%s) — caching the "
            "partial feed of %d articles; the rest of the cycle reads it without refetching.",
            ",".join(tickers),
            type(exc).__name__,
            len(combined),
        )
        self._cached_feed = combined
        if self._degraded_reason is None:
            self._degraded_reason = "av_fetch_error"

    def _mark_unexpected_payload(
        self, tickers: list[str], payload: dict[str, Any]
    ) -> None:
        """Payload bez `feed` i bez markera limitu (`Error Message`, `{}`) —
        flaga tylko dla tickerów tego batcha. Log niesie same klucze payloadu:
        wartości bywają długie, a do diagnozy kształtu wystarczą nazwy pól."""
        logger.error(
            "Alpha Vantage NEWS_SENTIMENT returned an unexpected payload for %s "
            "(keys: %s) — no feed and no rate-limit marker.",
            ",".join(tickers),
            sorted(payload),
        )
        for ticker in tickers:
            self._symbol_degraded_reasons.setdefault(ticker, "av_unexpected_payload")

    def _mark_keys_exhausted(self) -> None:
        """Wystawia sygnał degradacji + CRITICAL alert (idempotentnie).

        Wołane zarówno gdy batch zwrócił None (rotacja wyczerpała klucze
        w trakcie), jak i przy short-circuit kolejnych batchów po wcześniejszym
        wyczerpaniu. Własny guard zapobiega podwójnemu alertowi — guard na
        `_degraded_reason` wyciszałby CRITICAL po wcześniejszym `av_fetch_error`.
        """
        with self._rotation_lock:
            if self._keys_exhausted_reported:
                return
            self._keys_exhausted_reported = True
        logger.error(
            "All %d Alpha Vantage API keys exhausted — returning partial feed.",
            len(self._api_keys),
        )
        if not self._lazy_per_symbol:
            self._degraded_reason = "av_keys_exhausted"
        # MAIL-15: treść alertu idzie do maila po polsku; angielska
        # diagnoza jest w logger.error powyżej.
        if self._exhausted_by_frequency_limit:
            message = (
                f"Wszystkie klucze Alpha Vantage ({len(self._api_keys)}) odbiły limit "
                "częstotliwości zapytań (pole Note) mimo ponowienia. Feed newsów "
                "i sentymentu jest w tym cyklu niepełny."
            )
            action = (
                "Rozłóż zapytania w czasie albo dodaj kolejny klucz "
                "w ALPHA_VANTAGE_API_KEYS (darmowy na alphavantage.co)."
            )
        else:
            message = (
                f"Wszystkie klucze Alpha Vantage ({len(self._api_keys)}) wyczerpały "
                "dzienny limit 25 zapytań. Feed newsów i sentymentu jest w tym "
                "cyklu niepełny."
            )
            action = (
                "Dodaj kolejny klucz w ALPHA_VANTAGE_API_KEYS (darmowy na "
                "alphavantage.co) albo poczekaj na reset o północy UTC."
            )
        self._emit_quota_alert(
            severity=QuotaSeverity.CRITICAL, message=message, action=action
        )

    def _fetch_batch_with_rotation(
        self, tickers: list[str]
    ) -> dict[str, Any] | None:
        """Próbuje wszystkich kluczy od `_active_key_idx` w górę.

        Zwraca payload przy sukcesie (i aktualizuje `_active_key_idx`),
        albo `None` gdy wszystkie klucze wyczerpane. `Note` (limit minutowy)
        dostaje jeden backoff i ponowienie tym samym kluczem, `Information`
        (limit dzienny) od razu rotuje (DAT-06).
        """
        backed_off = False
        while (key_idx := self._active_key_idx) < len(self._api_keys):
            payload = self._fetch_batch(tickers, self._api_keys[key_idx])

            if not is_soft_limit(payload, _NEWS_DATA_KEYS):
                return payload

            marker = soft_limit_text(payload, self._api_keys)
            if is_frequency_limit(payload) and not backed_off:
                logger.warning(
                    "Alpha Vantage key #%d hit the frequency limit: %s — retrying "
                    "the same key in %.1f s.",
                    key_idx + 1,
                    marker,
                    NOTE_BACKOFF_SECONDS,
                )
                backed_off = True
                time.sleep(NOTE_BACKOFF_SECONDS)
                continue

            logger.warning(
                "Alpha Vantage key #%d rate-limited: %s — rotating to next.",
                key_idx + 1,
                marker,
            )
            self._rotate_past(key_idx, frequency_limit=is_frequency_limit(payload))
            backed_off = False
            # Tryb wsadowy: bez sleepa — do martwego klucza nie poszedł realny
            # request, więc pacing 1 req/sec go nie dotyczy, a re-chodzenie po
            # wyczerpanych kluczach z 1.1s przerwą marnowałoby wall-clock na
            # quota-wyczerpany dzień (pacing między batchami jest w `_fetch`).
            # Tryb leniwy: każdy request i tak przechodzi przez `_pace()`.

        return None

    def _rotate_past(self, key_idx: int, *, frequency_limit: bool) -> None:
        """Przesuwa aktywny klucz za `key_idx`, o ile nikt tego jeszcze nie zrobił:
        drugi wątek z limitem na tym samym kluczu nie przeskoczy żywego następnego."""
        with self._rotation_lock:
            if self._active_key_idx == key_idx:
                self._active_key_idx = key_idx + 1
                self._exhausted_by_frequency_limit = frequency_limit

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
        if self._lazy_per_symbol:
            self._pace()
        response = translated_get(
            self._session,
            f"{self._base_url}/query",
            params=params,
            timeout=self._timeout,
        )
        raise_for_status_translated(response)
        data: dict[str, Any] = json_payload(response)
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
        for item in self._feed_for(wire_symbol):
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
        features = _sentiment_features(self.articles_for(symbol))
        # DAT-07: degradacja tego symbolu (nie całego cyklu) jedzie w payloadzie.
        reason = self._symbol_degraded_reasons.get(self._to_wire(symbol))
        if reason is not None:
            features["degraded_reason"] = reason
        return features

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


def _sentiment_features(articles: list[dict[str, Any]]) -> dict[str, Any]:
    """Relevance-weighted agregata artykułów jednego symbolu (cechy ML i promptu)."""
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
