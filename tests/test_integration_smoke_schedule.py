"""TEST-16: cotygodniowy test kształtu dostawców w `integration_smoke.yml`.

Smoke na żywym API jest ręczny (CI-05): płaci za trzy wywołania LLM i wysyła mail.
Cron dokłada do tego samego workflowu osobny job, który puszcza wyłącznie
`tests/infrastructure/test_provider_shapes.py` (darmowe źródła + jedno zapytanie
Alpha Vantage z limitu klucza smoke). Strażnik pilnuje, że cron nigdy nie odpali
płatnego smoke'a i że porażka testu kształtu nie przejdzie po cichu.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parent.parent
SMOKE_WORKFLOW = ROOT / ".github" / "workflows" / "integration_smoke.yml"
SHAPES_TEST = "tests/infrastructure/test_provider_shapes.py"
DISPATCH_ONLY = "github.event_name == 'workflow_dispatch'"
SCHEDULE_ONLY = "github.event_name == 'schedule'"


def _spec() -> dict[str, Any]:
    spec: dict[str, Any] = yaml.safe_load(SMOKE_WORKFLOW.read_text(encoding="utf-8"))
    return spec


def _triggers() -> dict[str, Any]:
    spec: dict[Any, Any] = _spec()
    # PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True.
    triggers = spec.get("on", spec.get(True))
    assert isinstance(triggers, dict), triggers
    return triggers


def _pytest_commands(job: dict[str, Any]) -> list[list[str]]:
    commands = [str(step.get("run", "")) for step in job.get("steps", [])]
    return [shlex.split(command) for command in commands if "pytest" in command]


def _scheduled_jobs() -> dict[str, dict[str, Any]]:
    """Joby, które cron może uruchomić — każdy bez warunku tylko na dispatch."""
    return {
        name: job
        for name, job in _spec()["jobs"].items()
        if str(job.get("if", "")).strip() != DISPATCH_ONLY
    }


def test_smoke_runs_weekly_on_a_single_cron() -> None:
    schedule = _triggers().get("schedule")

    assert isinstance(schedule, list) and len(schedule) == 1, schedule
    minute, hour, day_of_month, month, day_of_week = str(schedule[0]["cron"]).split()
    assert (day_of_month, month) == ("*", "*")
    assert day_of_week.isdigit(), "raz w tygodniu = jeden dzień tygodnia"
    assert minute.isdigit() and hour.isdigit()


def test_paid_smoke_job_runs_only_on_manual_dispatch() -> None:
    jobs = _spec()["jobs"]

    assert str(jobs["live-api-smoke"].get("if", "")).strip() == DISPATCH_ONLY


def test_scheduled_jobs_run_only_the_provider_shape_test() -> None:
    scheduled = _scheduled_jobs()

    assert scheduled, "cron bez jobu — test kształtu nigdy nie biegnie"
    for name, job in scheduled.items():
        assert str(job.get("if", "")).strip() == SCHEDULE_ONLY, name
        commands = _pytest_commands(job)
        assert commands, name
        for tokens in commands:
            selected = [token for token in tokens if token.startswith("tests")]
            assert selected == [SHAPES_TEST], (name, tokens)
            assert tokens[tokens.index("-m") + 1] == "integration", (name, tokens)


def test_scheduled_jobs_alert_on_failure_through_the_smoke_channel() -> None:
    for name, job in _scheduled_jobs().items():
        last = job["steps"][-1]
        condition = str(last.get("if", ""))
        assert "failure()" in condition and "cancelled()" in condition, name
        assert "scripts/notify-failure.sh" in str(last.get("run", "")), name
        env = last.get("env", {})
        # Sekrety smoke'a (CI-05), nie produkcyjne: alert idzie na adres smoke.
        assert env.get("RESEND_API_KEY") == "${{ secrets.SMOKE_RESEND_API_KEY }}", name
        assert env.get("DIGEST_TO_EMAIL") == "${{ secrets.SMOKE_DIGEST_TO_EMAIL }}", name
