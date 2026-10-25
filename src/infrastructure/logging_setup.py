"""NIEZ-21: logi entry-pointów z kontekstem przebiegu (run_id, entry_point, symbol).

Przy `symbol_concurrency > 1` linie kilku symboli przeplatają się w jednym logu
GHA, a symbol niesie tylko część komunikatów. `RunContextFilter` dokleja do
każdego rekordu run_id (`GITHUB_RUN_ID` albo uuid4 lokalnie), próbę
(`GITHUB_RUN_ATTEMPT`), commit (`GITHUB_SHA`), entry_point i symbol z ContextVar
ustawianego przez `symbol_context`. W GitHub Actions linia jest JSON-em
(`LOG_FORMAT=json|text`, json domyślnie przy `GITHUB_ACTIONS=true`), lokalnie
zostaje czytelny tekst.

ContextVar nie przechodzi sam do wątków puli: worker symbolu ustawia go u siebie
(`RunFastLoopCycle.process_one` przez `symbol_scope`), a rada person startuje wątki przez
`contextvars.copy_context()` (`advisory_council._run_round`).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import IO, Literal

from src.infrastructure.logging_redaction import (
    DEFAULT_LOG_FORMAT,
    RedactingFormatter,
)
from src.infrastructure.logging_redaction import (
    configure_logging as configure_redacted_logging,
)

LogFormat = Literal["json", "text"]

# Nazwa handlera, który entry-point tworzy sam. JSON dostaje tylko on — handlery
# zastane w roocie (np. pytesta) zachowują swój format.
OWNED_HANDLER_NAME = "stockagent"

_SYMBOL: ContextVar[str | None] = ContextVar("stockagent_log_symbol", default=None)


@contextmanager
def symbol_context(symbol: str) -> Iterator[None]:
    """Symbol w polu `symbol` każdej linii logu z tego bloku (i z wątków
    startowanych przez `copy_context`). Po wyjściu wraca poprzednia wartość."""
    token = _SYMBOL.set(symbol)
    try:
        yield
    finally:
        _SYMBOL.reset(token)


def current_symbol() -> str | None:
    return _SYMBOL.get()


def resolve_log_format(environ: Mapping[str, str]) -> LogFormat:
    """`LOG_FORMAT` (json/text, bez rozróżniania wielkości liter); bez niego JSON
    tylko w GitHub Actions. Nieznana wartość daje tekst, nie nieczytelny log."""
    explicit = environ.get("LOG_FORMAT", "").strip().lower()
    if explicit == "json":
        return "json"
    if explicit == "text":
        return "text"
    if explicit:
        return "text"
    return "json" if environ.get("GITHUB_ACTIONS") == "true" else "text"


class RunContextFilter(logging.Filter):
    """Dokleja do rekordu pola przebiegu i bieżący symbol. Nie odrzuca rekordów."""

    def __init__(
        self, *, entry_point: str, run_id: str, run_attempt: str | None, git_sha: str | None
    ) -> None:
        super().__init__()
        self.entry_point = entry_point
        self.run_id = run_id
        self.run_attempt = run_attempt
        self.git_sha = git_sha

    @classmethod
    def from_environ(cls, entry_point: str, environ: Mapping[str, str]) -> RunContextFilter:
        """run_id z `GITHUB_RUN_ID` (ten sam co w stopce maila i `cycle_runs`),
        a lokalnie losowy uuid4 — dwa lokalne przebiegi da się rozdzielić."""
        return cls(
            entry_point=entry_point,
            run_id=environ.get("GITHUB_RUN_ID") or uuid.uuid4().hex,
            run_attempt=environ.get("GITHUB_RUN_ATTEMPT") or None,
            git_sha=environ.get("GITHUB_SHA") or None,
        )

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        record.run_attempt = self.run_attempt
        record.git_sha = self.git_sha
        record.entry_point = self.entry_point
        record.symbol = _SYMBOL.get()
        return True


_CONTEXT_FIELDS = ("run_id", "run_attempt", "git_sha", "entry_point", "symbol")


class JsonFormatter(logging.Formatter):
    """Jedna linia JSON na rekord; traceback w polu `exc_info`, nie w nowych liniach."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in _CONTEXT_FIELDS:
            payload[name] = getattr(record, name, None)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    entry_point: str,
    secret_values: Iterable[str] = (),
    *,
    level: int = logging.INFO,
    environ: Mapping[str, str] | None = None,
    stream: IO[str] | None = None,
) -> None:
    """Jedyna konfiguracja logów `main_*.py`: handler entry-pointu (stderr albo
    `stream`) z redakcją sekretów (BEZ-02), filtrem kontekstu przebiegu na każdym
    handlerze roota i JSON-em na handlerze entry-pointu, gdy `resolve_log_format`
    tak mówi. Ponowne wywołanie podmienia filtr i sekrety, niczego nie dubluje."""
    env = os.environ if environ is None else environ
    root = logging.getLogger()
    if not root.handlers:
        owned = logging.StreamHandler(stream)
        owned.set_name(OWNED_HANDLER_NAME)
        owned.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        root.addHandler(owned)
        # Jak `basicConfig`: poziom roota tylko wtedy, gdy to my go konfigurujemy.
        root.setLevel(level)
    configure_redacted_logging(secret_values, level=level)
    context_filter = RunContextFilter.from_environ(entry_point, env)
    use_json = resolve_log_format(env) == "json"
    for handler in root.handlers:
        for stale in [f for f in handler.filters if isinstance(f, RunContextFilter)]:
            handler.removeFilter(stale)
        handler.addFilter(context_filter)
        formatter = handler.formatter
        if use_json and handler.get_name() == OWNED_HANDLER_NAME and isinstance(
            formatter, RedactingFormatter
        ):
            formatter.inner = JsonFormatter()
