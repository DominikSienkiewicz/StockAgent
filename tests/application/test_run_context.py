"""MAIL-42: `RunContext` — tożsamość przebiegu z env GitHub Actions."""

from __future__ import annotations

import dataclasses

import pytest

from src.application.run_context import RunContext

_GHA_ENV = {
    "GITHUB_ACTIONS": "true",
    "GITHUB_RUN_ID": "1234",
    "GITHUB_SHA": "6eff942aa1b2c3d4e5f60718293a4b5c6d7e8f90",
    "GITHUB_EVENT_NAME": "schedule",
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "owner/StockAgent",
}


def test_github_actions_env_builds_run_identity_and_logs_url() -> None:
    context = RunContext.from_environ(_GHA_ENV)

    assert context == RunContext(
        run_id="1234",
        git_sha="6eff942aa1b2c3d4e5f60718293a4b5c6d7e8f90",
        trigger="schedule",
        run_url="https://github.com/owner/StockAgent/actions/runs/1234",
    )
    assert context.short_sha == "6eff942"


def test_local_run_has_local_sha_and_no_logs_url() -> None:
    context = RunContext.from_environ({})

    assert context.git_sha == "local"
    assert context.run_id == "local"
    assert context.trigger == "local"
    assert context.run_url is None


def test_github_variables_outside_actions_do_not_fake_a_run() -> None:
    """Link do logów tylko w GHA: lokalnie wyeksportowane GITHUB_* to nie przebieg."""
    env = {key: value for key, value in _GHA_ENV.items() if key != "GITHUB_ACTIONS"}

    assert RunContext.from_environ(env) == RunContext()


def test_actions_without_repository_has_no_logs_url() -> None:
    env = {key: value for key, value in _GHA_ENV.items() if key != "GITHUB_REPOSITORY"}

    context = RunContext.from_environ(env)

    assert context.run_id == "1234"
    assert context.run_url is None


def test_run_context_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        RunContext().git_sha = "x"  # type: ignore[misc]
