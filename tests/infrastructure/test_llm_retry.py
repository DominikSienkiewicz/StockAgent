"""ARCH-29: wspólna pętla ponowień (`_retry.py`) i sygnalizacja kwot (`_quota.py`)
warstwy LLM — jedna kopia zamiast trzech pętli i czterech `_emit_quota_alert`."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.infrastructure.llm._quota import QuotaEmitter
from src.infrastructure.llm._retry import (
    MAX_RETRY_WAIT_S,
    RetryPolicy,
    call_with_retry,
    retry_after_seconds,
)

LOGGER = logging.getLogger("tests.llm_retry")


class Transient(Exception):
    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__("transient")
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = SimpleNamespace(headers=headers)


class Permanent(Exception):
    pass


class Translated(Exception):
    pass


def _policy(
    *,
    max_retries: int = 2,
    label: str = "Test",
    translate: Callable[[Exception], Exception] | None = None,
) -> RetryPolicy:
    return RetryPolicy(
        retryable=(Transient, Permanent),
        max_retries=max_retries,
        backoff_base=2.0,
        logger=LOGGER,
        label=label,
        is_transient=lambda exc: isinstance(exc, Transient),
        translate=translate,
    )


class _Script:
    """Kolejne wyniki `fn`: wyjątek jest rzucany, reszta zwracana."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Recorder:
    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self.recovered: list[int] = []
        self.exhausted: list[Exception] = []

    def run(self, fn: _Script, policy: RetryPolicy) -> object:
        return call_with_retry(
            fn,
            policy,
            sleep=self.sleeps.append,
            on_recovered=self.recovered.append,
            on_exhausted=self.exhausted.append,
        )


class TestRetryAfterSeconds:
    def test_server_retry_after_wins_over_the_backoff(self) -> None:
        assert retry_after_seconds(Transient(retry_after="7"), 0, 2.0) == 7.0

    def test_retry_after_is_capped(self) -> None:
        assert retry_after_seconds(Transient(retry_after="3600"), 0, 2.0) == MAX_RETRY_WAIT_S

    def test_unparseable_retry_after_falls_back_to_exponential_backoff(self) -> None:
        assert retry_after_seconds(Transient(retry_after="soon"), 2, 2.0) == 8.0

    def test_backoff_is_capped(self) -> None:
        assert retry_after_seconds(Permanent(), 10, 2.0) == MAX_RETRY_WAIT_S


class TestCallWithRetry:
    def test_clean_call_neither_sleeps_nor_reports(self) -> None:
        recorder = _Recorder()

        assert recorder.run(_Script("ok"), _policy()) == "ok"
        assert (recorder.sleeps, recorder.recovered, recorder.exhausted) == ([], [], [])

    def test_transient_error_is_retried_and_the_recovery_reported(self) -> None:
        recorder = _Recorder()
        script = _Script(Transient(), Transient(retry_after="3"), "ok")

        assert recorder.run(script, _policy()) == "ok"
        assert script.calls == 3
        assert recorder.sleeps == [2.0, 3.0]
        assert recorder.recovered == [2]
        assert recorder.exhausted == []

    def test_exhausted_retries_report_and_raise_the_translated_error(self) -> None:
        recorder = _Recorder()
        script = _Script(Transient(), Transient(), Transient())
        policy = _policy(translate=lambda exc: Translated(str(exc)))

        with pytest.raises(Translated):
            recorder.run(script, policy)

        assert script.calls == 3
        assert len(recorder.exhausted) == 1
        assert recorder.recovered == []

    def test_without_translation_the_sdk_error_itself_escapes(self) -> None:
        recorder = _Recorder()
        error = Transient()

        with pytest.raises(Transient) as excinfo:
            recorder.run(_Script(error), _policy(max_retries=0))

        assert excinfo.value is error
        assert recorder.exhausted == [error]

    def test_permanent_error_escapes_at_once_without_a_report(self) -> None:
        recorder = _Recorder()
        script = _Script(Permanent(), "never")

        with pytest.raises(Translated):
            recorder.run(script, _policy(translate=lambda exc: Translated(str(exc))))

        assert script.calls == 1
        assert (recorder.sleeps, recorder.exhausted) == ([], [])

    def test_error_outside_the_policy_is_not_touched(self) -> None:
        recorder = _Recorder()

        with pytest.raises(KeyError):
            recorder.run(_Script(KeyError("x")), _policy(translate=Translated))

    def test_retry_is_logged_by_the_adapter_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=LOGGER.name):
            _Recorder().run(_Script(Transient(), "ok"), _policy(label="OpenAI embeddings"))

        assert [record.name for record in caplog.records] == [LOGGER.name]
        assert "OpenAI embeddings Transient (attempt 1/2) — sleeping 2.0s" in caplog.text


class TestQuotaEmitter:
    MOMENT = datetime(2026, 10, 26, 6, 0, tzinfo=UTC)

    def test_alert_goes_to_the_monitor_and_the_adapter_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        monitor = QuotaMonitor()
        emitter = QuotaEmitter("OpenAI (gpt-4o)", monitor, LOGGER, clock=lambda: self.MOMENT)

        with caplog.at_level(logging.WARNING, logger=LOGGER.name):
            emitter.emit(QuotaSeverity.CRITICAL, "po polsku", "zrób to", "in English")

        assert monitor.alerts == [
            QuotaAlert(
                source="OpenAI (gpt-4o)",
                severity=QuotaSeverity.CRITICAL,
                message="po polsku",
                action="zrób to",
                occurred_at=self.MOMENT,
            )
        ]
        assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
            (LOGGER.name, logging.ERROR, "in English")
        ]

    def test_warning_without_a_monitor_is_only_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=LOGGER.name):
            QuotaEmitter("Anthropic (claude-sonnet-4-6)", None, LOGGER).emit(
                QuotaSeverity.WARNING, "po polsku", "zrób to", "retry succeeded"
            )

        assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
            (logging.WARNING, "retry succeeded")
        ]
