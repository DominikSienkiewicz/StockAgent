"""BEZ-02: formatter logów redagujący sekrety.

Traceback z URL-em (`?apikey=…`) i klucze z `alpha_vantage_api_keys` szły do
publicznych logów GHA. Filtr logów tego nie łapie — `exc_text` dokleja dopiero
`Formatter.format` — więc redagujemy gotowy tekst rekordu.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager

from src.infrastructure.logging_redaction import (
    DEFAULT_LOG_FORMAT,
    RedactingFormatter,
    configure_logging,
)

AV_KEYS = ["AVKEY-literal-0001", "AVKEY-literal-0002"]


def _capturing_logger(formatter: logging.Formatter) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    logger = logging.getLogger(f"test.bez02.{id(stream)}")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, stream


def test_formatter_redacts_apikey_in_exception_text() -> None:
    formatter = RedactingFormatter(logging.Formatter("%(levelname)s %(message)s"), AV_KEYS)
    logger, stream = _capturing_logger(formatter)

    try:
        raise ValueError(
            "401 Client Error for url: https://www.alphavantage.co/query?apikey=SECRET123 "
            f"(key {AV_KEYS[1]} rejected)"
        )
    except ValueError:
        logger.exception("Alpha Vantage fetch failed with key %s", AV_KEYS[0])

    output = stream.getvalue()
    assert "ERROR Alpha Vantage fetch failed" in output
    assert "Traceback" in output and "ValueError" in output
    assert "SECRET123" not in output
    for key in AV_KEYS:
        assert key not in output


def test_formatter_keeps_the_inner_format_of_clean_records() -> None:
    formatter = RedactingFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    record = logging.LogRecord("svc", logging.INFO, __file__, 1, "cycle %s done", ("A1",), None)

    assert formatter.format(record) == "[INFO] svc: cycle A1 done"


@contextmanager
def _bare_root() -> Iterator[logging.Logger]:
    """Root bez handlerów (także pytesta) na czas bloku, potem stan sprzed niego.

    W ciele testu, nie w fixturze: pytest dokłada swoje handlery do roota
    dopiero w fazie wywołania testu."""
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


def test_configure_logging_installs_a_redacting_handler_like_basic_config() -> None:
    with _bare_root() as root:
        configure_logging(AV_KEYS, fmt="%(levelname)s %(message)s")
        level, handlers = root.level, root.handlers[:]

    assert level == logging.INFO
    assert len(handlers) == 1
    formatter = handlers[0].formatter
    assert isinstance(formatter, RedactingFormatter)
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "key=%s", (AV_KEYS[0],), None)
    assert formatter.format(record) == "INFO key=***"


def test_configure_logging_defaults_to_the_agent_format() -> None:
    with _bare_root() as root:
        configure_logging()
        formatter = root.handlers[0].formatter

    assert isinstance(formatter, RedactingFormatter)
    assert formatter.inner._fmt == DEFAULT_LOG_FORMAT


def test_configure_logging_wraps_existing_handlers_once() -> None:
    # Root skonfigurowany wcześniej (inna biblioteka, pytest) też nie może puścić sekretu,
    # a ponowne wywołanie podmienia wartości zamiast owijać formatter drugi raz.
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    original = logging.Formatter("%(message)s")
    handler.setFormatter(original)

    with _bare_root() as root:
        root.addHandler(handler)
        configure_logging(["first-secret-value"])
        configure_logging(["second-secret-value"])
        logging.getLogger("test.bez02.wrap").warning("second-secret-value via ?token=abc")
        handlers = root.handlers[:]

    formatter = handler.formatter
    assert isinstance(formatter, RedactingFormatter)
    assert formatter.inner is original
    assert handlers == [handler]
    assert stream.getvalue() == "*** via ?token=***\n"
