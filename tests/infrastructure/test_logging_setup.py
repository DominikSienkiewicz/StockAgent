"""NIEZ-21: logi JSON z run_id, entry_point i symbolem (contextvars).

Przy `symbol_concurrency = 4` linie czterech symboli przeplatają się w jednym
logu GHA, a symbol niesie tylko część komunikatów. Każda linia JSON ma więc
run_id przebiegu i symbol z ContextVar ustawianego w workerze symbolu.
"""

from __future__ import annotations

import contextvars
import io
import json
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from src.infrastructure.logging_setup import (
    JsonFormatter,
    RunContextFilter,
    configure_logging,
    resolve_log_format,
    symbol_context,
)

GHA_ENV = {
    "GITHUB_ACTIONS": "true",
    "GITHUB_RUN_ID": "42",
    "GITHUB_RUN_ATTEMPT": "2",
    "GITHUB_SHA": "abc1234def",
}


def _json_logger(environ: dict[str, str]) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RunContextFilter.from_environ("main_agent", environ))
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(f"test.niez21.{id(stream)}")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, stream


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_json_formatter_emits_run_id_and_symbol() -> None:
    logger, stream = _json_logger({"GITHUB_RUN_ID": "42", "LOG_FORMAT": "json"})

    with symbol_context("NVDA"):
        logger.info("x")

    (line,) = _lines(stream)
    assert line["run_id"] == "42"
    assert line["symbol"] == "NVDA"
    assert line["message"] == "x"
    assert line["level"] == "INFO"
    assert line["entry_point"] == "main_agent"


def test_json_line_carries_attempt_and_commit_from_actions_env() -> None:
    logger, stream = _json_logger(GHA_ENV)

    logger.warning("cycle %s done", "A1")

    (line,) = _lines(stream)
    assert line["run_id"] == "42"
    assert line["run_attempt"] == "2"
    assert line["git_sha"] == "abc1234def"
    assert line["message"] == "cycle A1 done"
    # Poza zakresem symbolu pole jest jawnie puste, nie znika.
    assert line["symbol"] is None


def test_local_run_without_actions_env_gets_a_random_run_id() -> None:
    first = RunContextFilter.from_environ("main_agent", {})
    second = RunContextFilter.from_environ("main_agent", {})

    assert first.run_id != second.run_id
    assert len(first.run_id) == 32


def test_symbol_context_restores_previous_symbol_on_exit() -> None:
    logger, stream = _json_logger(GHA_ENV)

    with symbol_context("NVDA"):
        with symbol_context("AAPL"):
            logger.info("inner")
        logger.info("outer")
    logger.info("after")

    assert [line["symbol"] for line in _lines(stream)] == ["AAPL", "NVDA", None]


def test_symbol_reaches_threads_started_with_a_copied_context() -> None:
    # Rada person (council_service._run_round) startuje wątki przez copy_context.
    logger, stream = _json_logger(GHA_ENV)

    with symbol_context("MSFT"), ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(contextvars.copy_context().run, logger.info, "persona %d", i)
            for i in range(2)
        ]
        for future in futures:
            future.result()

    assert {line["symbol"] for line in _lines(stream)} == {"MSFT"}


def test_json_formatter_keeps_traceback_in_one_line() -> None:
    logger, stream = _json_logger(GHA_ENV)

    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")

    (line,) = _lines(stream)
    assert "ValueError: boom" in str(line["exc_info"])
    assert len(stream.getvalue().splitlines()) == 1


def test_log_format_defaults_to_json_only_in_github_actions() -> None:
    assert resolve_log_format({"GITHUB_ACTIONS": "true"}) == "json"
    assert resolve_log_format({}) == "text"
    assert resolve_log_format({"GITHUB_ACTIONS": "true", "LOG_FORMAT": "text"}) == "text"
    assert resolve_log_format({"LOG_FORMAT": "JSON"}) == "json"
    # Literówka nie przełącza formatu po cichu na nieczytelny.
    assert resolve_log_format({"LOG_FORMAT": "yaml"}) == "text"


@contextmanager
def _bare_root() -> Iterator[logging.Logger]:
    """Root bez handlerów (także pytesta) na czas bloku, potem stan sprzed niego."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    for handler in saved_handlers:
        root.removeHandler(handler)
    try:
        yield root
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


def test_configure_logging_emits_redacted_json_with_context() -> None:
    stream = io.StringIO()
    with _bare_root():
        configure_logging(
            "main_trainer",
            ["AVKEY-literal-0001"],
            environ={**GHA_ENV, "LOG_FORMAT": "json"},
            stream=stream,
        )
        with symbol_context("BTC"):
            logging.getLogger("svc").info("key %s", "AVKEY-literal-0001")

    (line,) = _lines(stream)
    assert line["entry_point"] == "main_trainer"
    assert line["symbol"] == "BTC"
    assert line["message"] == "key ***"


def test_configure_logging_keeps_text_format_locally() -> None:
    stream = io.StringIO()
    with _bare_root():
        configure_logging("main_agent", (), environ={}, stream=stream)
        logging.getLogger("svc").info("hello")

    assert stream.getvalue().rstrip().endswith("INFO svc — hello")


def test_configure_logging_twice_keeps_one_context_filter() -> None:
    with _bare_root() as root:
        configure_logging("main_agent", (), environ={}, stream=io.StringIO())
        configure_logging("main_agent", (), environ={}, stream=io.StringIO())

        filters = [
            f
            for handler in root.handlers
            for f in handler.filters
            if isinstance(f, RunContextFilter)
        ]
    assert len(filters) == 1
