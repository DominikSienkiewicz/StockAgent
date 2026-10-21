"""Wspólny HTTP client z retry/backoff dla transientnych błędów.

Adaptery (Finnhub, Alpha Vantage, Resend) używają `build_session()` zamiast
gołego `requests` — dzięki temu pojedynczy timeout albo chwilowy 5xx nie
wywala całego symbolu z cyklu.

BEZ-07: `requests` wkleja pełny URL (z `?token=` / `&apikey=`) do treści
`HTTPError` i błędów sieci, a ten tekst trafia do maila, `quota_alerts`
i publicznych logów GHA. Adaptery z sekretem w URL-u wołają więc
`safe_get` i `raise_for_status_redacted` zamiast gołych metod `requests`.

NIEZ-24: każda sesja ma hook odpowiedzi, który liczy żądanie per host w
`HttpCallMeter` — jawnie podanym (`build_session(meter=...)`) albo zainstalowanym
na przebieg przez `metered_http`. Dzięki temu licznik obejmuje wszystkie adaptery
bez zmian w ich kodzie.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.application.http_metering import HttpCallMeter

logger = logging.getLogger(__name__)

# Retry dla błędów, które zwykle są przejściowe:
#   429 — rate limit, 500/502/503/504 — chwilowa awaria po stronie API.
# backoff_factor=0.5 → przerwy 0.5s, 1s, 2s między próbami. Retry-After serwera jest
# respektowany, ale najwyżej RETRY_AFTER_MAX_SECONDS — nagłówek potrafi żądać godzin,
# a job ma 15 minut limitu.
RETRY_TOTAL = 3
RETRY_BACKOFF_FACTOR = 0.5
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)
RETRY_AFTER_MAX_SECONDS = 10


# NIEZ-24: miernik przebiegu dla sesji bez jawnego `meter`. Czytany przy KAŻDEJ
# odpowiedzi, nie przy budowie sesji — liczą się też sesje zbudowane wcześniej.
_installed_meter: HttpCallMeter | None = None


@contextmanager
def metered_http(meter: HttpCallMeter) -> Iterator[HttpCallMeter]:
    """Na czas bloku żądania sesji bez własnego miernika liczą się w `meter`.
    Po wyjściu wraca poprzedni miernik (zagnieżdżenie i testy niczego nie gubią)."""
    global _installed_meter
    previous, _installed_meter = _installed_meter, meter
    try:
        yield meter
    finally:
        _installed_meter = previous


def _response_hook(meter: HttpCallMeter | None) -> Callable[..., requests.Response]:
    def hook(response: requests.Response, *_args: Any, **_kwargs: Any) -> requests.Response:
        target = meter if meter is not None else _installed_meter
        if target is None:
            return response
        try:
            target.record(
                urlsplit(response.url or "").hostname or "unknown",
                response.status_code,
                retries=_retry_count(response),
                elapsed_s=response.elapsed.total_seconds(),
            )
        except Exception:
            # Pomiar to obserwowalność — jego błąd nie może zepsuć żądania adaptera.
            logger.debug("HTTP meter failed to record a response", exc_info=True)
        return response

    return hook


def _retry_count(response: requests.Response) -> int:
    """Liczba ponowień urllib3 za tą odpowiedzią (`raw.retries.history`)."""
    retries = getattr(response.raw, "retries", None)
    return len(getattr(retries, "history", ()) or ())


def build_session(
    *, retry_post: bool = True, meter: HttpCallMeter | None = None
) -> requests.Session:
    """Zwraca `requests.Session` z zamontowanym retry na https://.

    https-only: wszystkie API agenta (Finnhub, Alpha Vantage, CoinGecko, NBP,
    Resend, Supabase, OpenAI/Anthropic) komunikują się po https, więc montujemy
    retry-adapter wyłącznie na schemacie https:// — nie ma ścieżki http://.

    `retry_post=False` dla API bez klucza idempotencji (Telegram, webhook Slacka):
    POST powtórzony po 5xx, który serwer jednak przyjął, dubluje wiadomość.

    `meter` (NIEZ-24) — miernik tej sesji; None = miernik z `metered_http`, o ile
    jest zainstalowany.
    """
    session = requests.Session()
    session.hooks["response"].append(_response_hook(meter))
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=frozenset({"GET", "POST"} if retry_post else {"GET"}),
        raise_on_status=False,  # status sprawdzamy sami przez raise_for_status()
        retry_after_max=RETRY_AFTER_MAX_SECONDS,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


class RateLimiter:
    """Token bucket współdzielony przez wszystkie adaptery jednego API (DAT-12).

    Limit jest per klucz, więc limiter działa tylko jako JEDEN obiekt na proces
    (DI tworzy go raz) — limiter na instancji adaptera nie widziałby ruchu
    pozostałych. Pojemność = `rate_per_minute` (burst), dolewka liniowa.
    `acquire()` blokuje wątek do czasu dostępnego tokenu; lock trzyma kolejkę
    przy równoległych symbolach."""

    def __init__(
        self,
        rate_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError(f"rate_per_minute must be positive (got {rate_per_minute})")
        self._capacity = float(rate_per_minute)
        self._per_second = rate_per_minute / 60.0
        self._clock = clock
        self._sleep = sleep
        self._tokens = self._capacity
        self._updated = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            elapsed = max(0.0, now - self._updated)
            self._tokens = min(self._capacity, self._tokens + elapsed * self._per_second)
            self._updated = now
            if self._tokens < 1.0:
                self._sleep((1.0 - self._tokens) / self._per_second)
                self._tokens = 1.0
                self._updated = self._clock()
            self._tokens -= 1.0


REDACTED = "***"
# Krótsze wartości nie są redagowane dosłownie: testowe `chat_id=42` albo
# atrapa klucza zjadałyby zwykły tekst logu, a prawdziwe sekrety są dłuższe.
MIN_LITERAL_SECRET_LENGTH = 8

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Parametr query stringa: ?apikey=…, &token=…, &api_key=…, &key=…
    (
        re.compile(r"(?i)([?&;](?:apikey|api_key|token|key)=)[^&#\s'\"]+"),
        rf"\g<1>{REDACTED}",
    ),
    # Nagłówek autoryzacji (także w reprze dicta), z opcjonalnym schematem.
    (
        re.compile(
            r"(?i)((?:proxy-)?authorization['\"]?\s*[:=]\s*['\"]?"
            r"(?:(?:bearer|basic|token)\s+)?)[^\s'\",}&]+"
        ),
        rf"\g<1>{REDACTED}",
    ),
    # Nagłówki z kluczem: `apikey` (Supabase), `x-api-key` (Anthropic).
    (
        re.compile(r"(?i)(['\"]?(?:apikey|api_key|x-api-key)['\"]?\s*[:=]\s*['\"]?)[^\s'\",}&]+"),
        rf"\g<1>{REDACTED}",
    ),
    # Goły token okaziciela (np. zrzut nagłówków bez nazwy).
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+"), rf"\g<1> {REDACTED}"),
    # Telegram: token bota siedzi w ścieżce `/bot<id>:<token>/`.
    (re.compile(r"(/bot)\d+:[A-Za-z0-9_-]+"), rf"\g<1>{REDACTED}"),
    # Slack: cały URL incoming webhooka jest sekretem.
    (re.compile(r"(?i)(hooks\.slack\.com/)[^\s'\"<>]+"), rf"\g<1>{REDACTED}"),
)

def redact_secrets(text: str, secret_values: Iterable[str] = ()) -> str:
    """Zwraca `text` bez sekretów: każdej wartości z `secret_values` (np. kluczy
    z `Settings`, najdłuższe najpierw) i rozpoznawalnych kształtów — parametrów
    `apikey`/`token`/`api_key`/`key`, nagłówków autoryzacji, ścieżki bota
    Telegrama i URL-a webhooka Slacka."""
    literals = sorted(
        {value for value in secret_values if len(value) >= MIN_LITERAL_SECRET_LENGTH},
        key=len,
        reverse=True,
    )
    for value in literals:
        text = text.replace(value, REDACTED)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def raise_for_status_redacted(response: requests.Response) -> None:
    """`response.raise_for_status()`, ale `HTTPError` niesie URL bez sekretów.

    Rzuca ten sam typ (z tym samym `response`), więc wywołujący łapią go jak
    dotąd; `from None` odcina oryginał, który `logger.exception` dokleiłby do
    tracebacku razem z kluczem."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise _redacted_copy(exc) from None


def safe_get(session: requests.Session, url: str, **kwargs: Any) -> requests.Response:
    """`session.get(url, **kwargs)`, ale błąd sieci (timeout, DNS, retry) ma
    komunikat bez sekretów z query stringa — ten sam typ wyjątku, `from None`."""
    try:
        return session.get(url, **kwargs)
    except requests.RequestException as exc:
        raise _redacted_copy(exc) from None


def _redacted_copy[E: requests.RequestException](exc: E) -> E:
    return type(exc)(redact_secrets(str(exc)), request=exc.request, response=exc.response)
