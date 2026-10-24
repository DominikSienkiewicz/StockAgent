"""Wspólny HTTP client z retry/backoff dla transientnych błędów.

Adaptery (Finnhub, Alpha Vantage, Resend) używają `build_session()` zamiast
gołego `requests` — dzięki temu pojedynczy timeout albo chwilowy 5xx nie
wywala całego symbolu z cyklu.

BEZ-07: `requests` wkleja pełny URL (z `?token=` / `&apikey=`) do treści
`HTTPError` i błędów sieci, a ten tekst trafia do maila, `quota_alerts`
i publicznych logów GHA. Adaptery z sekretem w URL-u wołają więc
`safe_get` i `raise_for_status_redacted` zamiast gołych metod `requests`.

ARCH-33: `translate_request_error` / `translate_http_error` zamieniają wyjątek
`requests` na taksonomię portów (`src.application.errors`), zachowując jego typ.

NIEZ-31: każda sesja ma circuit breaker per host na czas procesu
(`CircuitBreakerRegistry`, wspólny `process_circuits()`): po
`CIRCUIT_FAILURE_THRESHOLD` kolejnych porażkach (5xx, timeout, zerwane połączenie)
obwód hosta zostaje otwarty i żądanie od razu rzuca `CircuitOpenError` bez ruchu
sieciowego. Awaria Finnhuba nie zjada już 15 minut joba na 43 × 4 próby.

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
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.application.errors import PortError, ProviderUnavailable, QuotaExhausted
from src.application.http_metering import RATE_LIMITED, HttpCallMeter
from src.application.ports import CircuitState, CircuitStatePort
from src.infrastructure.error_translation import as_port_error

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


# NIEZ-31: tyle KOLEJNYCH porażek hosta (każda po pełnym retry urllib3) otwiera obwód.
CIRCUIT_FAILURE_THRESHOLD = 5

# Nazwy dostawców dla człowieka (banner kwot, MAIL-21); nieznany host = sam host.
_HOST_SOURCES = {
    "finnhub.io": "Finnhub",
    "www.alphavantage.co": "Alpha Vantage",
    "api.coingecko.com": "CoinGecko",
    "www.sec.gov": "EDGAR",
    "data.sec.gov": "EDGAR",
    "www.reddit.com": "Reddit",
    "api.stlouisfed.org": "FRED",
    "api.nbp.pl": "NBP",
    "api.resend.com": "Resend",
    "api.telegram.org": "Telegram",
    "hooks.slack.com": "Slack",
}


# MAIL-21: źródła, które odcinają klienta przez 403 (EDGAR bez nagłówka SEC, Reddit
# bez OAuth), a 404 nie zwracają — u nich każde 4xx to błędna odpowiedź. U reszty
# 4xx bywa zwykłym brakiem danych (404 na tickerze EU), więc się nie liczy.
_SOURCES_FAILING_ON_4XX = frozenset({"EDGAR", "Reddit"})
_FAILED_CLASSES: tuple[str, ...] = (RATE_LIMITED, "5xx")


def source_of_host(host: str) -> str:
    """Nazwa dostawcy dla człowieka; nieznany host = sam host."""
    return _HOST_SOURCES.get(host, host)


def failed_requests_by_source(http_calls: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    """MAIL-21: błędne odpowiedzi per dostawca z migawki `HttpCallMeter` — 429 i 5xx
    każdego hosta, a u źródeł z `_SOURCES_FAILING_ON_4XX` także 4xx. Bez zer."""
    failed: dict[str, int] = {}
    for host, counts in http_calls.items():
        source = source_of_host(host)
        classes = _FAILED_CLASSES + (("4xx",) if source in _SOURCES_FAILING_ON_4XX else ())
        count = sum(counts.get(status_class, 0) for status_class in classes)
        if count:
            failed[source] = failed.get(source, 0) + count
    return failed


class CircuitOpenError(ProviderUnavailable, requests.ConnectionError):
    """NIEZ-31: obwód hosta otwarty — żądanie nie wyszło do sieci.

    `requests.ConnectionError` zostaje bazą: adaptery, które łapią dziś
    `requests.RequestException` (EDGAR, Reddit, NBP, Alpha Vantage), prowadzą
    ten błąd na swoje istniejące ścieżki degradacji bez zmian w kodzie."""


def _current_symbol() -> str | None:
    # Import w funkcji: logging_setup → logging_redaction → _http (cykl importów).
    from src.infrastructure.logging_setup import current_symbol

    return current_symbol()


class _HostCircuit:
    def __init__(self) -> None:
        self.consecutive_failures = 0
        self.is_open = False
        self.skipped_calls = 0
        self.skipped_symbols: set[str] = set()


class CircuitBreakerRegistry(CircuitStatePort):
    """Stan obwodów per host na czas życia procesu, bezpieczny dla wątków puli
    symboli. Otwarty obwód nie zamyka się do końca procesu (jeden przebieg joba).

    Liczy porażki transportu (5xx, timeout, zerwane połączenie), nie 4xx — 404
    na tickerze EU wyłączyłby źródło. Każda inna odpowiedź zeruje licznik."""

    def __init__(self, failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD) -> None:
        self._threshold = failure_threshold
        self._circuits: dict[str, _HostCircuit] = {}
        self._lock = threading.Lock()

    def before_request(self, host: str) -> None:
        """Rzuca `CircuitOpenError`, gdy obwód hosta jest otwarty (i liczy pominięcie)."""
        with self._lock:
            circuit = self._circuits.get(host)
            if circuit is None or not circuit.is_open:
                return
            circuit.skipped_calls += 1
            symbol = _current_symbol()
            if symbol is not None:
                circuit.skipped_symbols.add(symbol)
        raise CircuitOpenError(
            f"{host}: circuit open after {self._threshold} consecutive failures — "
            "request skipped without network traffic"
        )

    def record_failure(self, host: str) -> None:
        with self._lock:
            circuit = self._circuits.setdefault(host, _HostCircuit())
            circuit.consecutive_failures += 1
            opens = not circuit.is_open and circuit.consecutive_failures >= self._threshold
            circuit.is_open = circuit.is_open or opens
        if opens:
            logger.warning(
                "%s: circuit opened after %d consecutive failures — further requests "
                "are skipped until the process ends",
                host,
                self._threshold,
            )

    def record_success(self, host: str) -> None:
        with self._lock:
            circuit = self._circuits.get(host)
            if circuit is not None and not circuit.is_open:
                circuit.consecutive_failures = 0

    def circuit_states(self) -> tuple[CircuitState, ...]:
        with self._lock:
            return tuple(
                CircuitState(
                    source=source_of_host(host),
                    host=host,
                    is_open=circuit.is_open,
                    consecutive_failures=circuit.consecutive_failures,
                    skipped_calls=circuit.skipped_calls,
                    skipped_symbols=frozenset(circuit.skipped_symbols),
                )
                for host, circuit in sorted(self._circuits.items())
            )

    def reset(self) -> None:
        """Zamyka wszystkie obwody — dla testów; przebieg produkcyjny nie resetuje."""
        with self._lock:
            self._circuits.clear()


_PROCESS_CIRCUITS = CircuitBreakerRegistry()


def process_circuits() -> CircuitBreakerRegistry:
    """Obwody tego procesu — wspólne dla wszystkich sesji `build_session()`."""
    return _PROCESS_CIRCUITS


class _CircuitBreakingAdapter(HTTPAdapter):
    """`HTTPAdapter` z retry urllib3, który pyta obwód hosta przed wysyłką i zgłasza
    mu wynik. Jedna porażka = jedno logiczne żądanie po wyczerpaniu ponowień."""

    def __init__(self, breakers: CircuitBreakerRegistry, *, max_retries: Retry) -> None:
        super().__init__(max_retries=max_retries)
        self._breakers = breakers

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: float | tuple[float | None, float | None] | None = None,
        verify: bool | str = True,
        cert: str | tuple[str, str] | None = None,
        proxies: dict[str, str] | None = None,
    ) -> requests.Response:
        host = urlsplit(request.url or "").hostname or "unknown"
        self._breakers.before_request(host)
        try:
            response = super().send(
                request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies
            )
        except (requests.ConnectionError, requests.Timeout):
            self._breakers.record_failure(host)
            raise
        if response.status_code >= _SERVER_ERROR:
            self._breakers.record_failure(host)
        else:
            self._breakers.record_success(host)
        return response


def build_session(
    *,
    retry_post: bool = True,
    meter: HttpCallMeter | None = None,
    breakers: CircuitBreakerRegistry | None = None,
) -> requests.Session:
    """Zwraca `requests.Session` z zamontowanym retry na https://.

    https-only: wszystkie API agenta (Finnhub, Alpha Vantage, CoinGecko, NBP,
    Resend, Supabase, OpenAI/Anthropic) komunikują się po https, więc montujemy
    retry-adapter wyłącznie na schemacie https:// — nie ma ścieżki http://.

    `retry_post=False` dla API bez klucza idempotencji (Telegram, webhook Slacka):
    POST powtórzony po 5xx, który serwer jednak przyjął, dubluje wiadomość.

    `meter` (NIEZ-24) — miernik tej sesji; None = miernik z `metered_http`, o ile
    jest zainstalowany.

    `breakers` (NIEZ-31) — obwody hostów; None = wspólne obwody procesu.
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
    adapter = _CircuitBreakingAdapter(
        breakers if breakers is not None else _PROCESS_CIRCUITS, max_retries=retry
    )
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


_RATE_LIMITED = 429
_SERVER_ERROR = 500


def http_error_kind(status_code: int) -> type[PortError]:
    """ARCH-33: klasa taksonomii dla statusu HTTP błędnej odpowiedzi — 429 to kwota,
    5xx to niedostępny dostawca, reszta 4xx to ogólny błąd portu."""
    if status_code == _RATE_LIMITED:
        return QuotaExhausted
    if status_code >= _SERVER_ERROR:
        return ProviderUnavailable
    return PortError


def translate_http_error(exc: requests.HTTPError, status_code: int) -> requests.HTTPError:
    """`exc` z `raise_for_status_redacted` jako klasa taksonomii statusu (wciąż `HTTPError`)."""
    return as_port_error(exc, http_error_kind(status_code))


def translate_request_error(exc: requests.RequestException) -> requests.RequestException:
    """Błąd sieci z `safe_get` w taksonomii: zerwane połączenie i timeout to niedostępny
    dostawca, reszta (np. zły URL z konfiguracji) to ogólny błąd portu."""
    transport = isinstance(exc, requests.ConnectionError | requests.Timeout)
    return as_port_error(exc, ProviderUnavailable if transport else PortError)
