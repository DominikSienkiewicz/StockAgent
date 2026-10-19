"""Konfiguracja logów entry-pointów z redakcją sekretów (BEZ-02).

Logi GHA są publiczne, a traceback z URL-em (`?apikey=…`) albo klucz z listy
`alpha_vantage_api_keys` trafiał do nich jawnym tekstem. Redagujemy w
Formatterze, nie w Filtrze: traceback (`exc_text`) dokleja dopiero
`Formatter.format`, więc Filtr widzi rekord bez niego.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from src.infrastructure.adapters._http import redact_secrets

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s — %(message)s"


class RedactingFormatter(logging.Formatter):
    """Formatuje rekord formatterem `inner`, a gotowy tekst (z tracebackiem)
    przepuszcza przez `redact_secrets` z wartościami sekretów z `Settings`."""

    def __init__(
        self, inner: logging.Formatter | None = None, secret_values: Iterable[str] = ()
    ) -> None:
        super().__init__()
        self.inner = inner if inner is not None else logging.Formatter()
        self._secret_values = tuple(secret_values)

    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(self.inner.format(record), self._secret_values)


def configure_logging(
    secret_values: Iterable[str] = (),
    *,
    level: int = logging.INFO,
    fmt: str = DEFAULT_LOG_FORMAT,
) -> None:
    """Jedyna konfiguracja logów `main_*.py`: `logging.basicConfig(level, fmt)`
    (no-op, gdy root ma już handlery), po czym każdy handler roota dostaje
    `RedactingFormatter` z `secret_values`. Ponowne wywołanie podmienia
    wartości zamiast owijać formatter drugi raz."""
    logging.basicConfig(level=level, format=fmt)
    secrets = tuple(secret_values)
    for handler in logging.getLogger().handlers:
        inner = handler.formatter
        if isinstance(inner, RedactingFormatter):
            inner = inner.inner
        handler.setFormatter(RedactingFormatter(inner, secrets))
