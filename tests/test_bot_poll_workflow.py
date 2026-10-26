"""Strażnik `bot_poll.yml` (MAIL-48): bot właściciela jest darmowy z konstrukcji
i bez sekretów TELEGRAM_* jest no-opem.

Workflow odpytuje Telegram co 15 minut, czyta bazę i zapisuje offset oraz
wyciszenia. Płatne klucze nie mają tu czego szukać (`Settings` wymaga ich pól, więc
dostają zaślepki), a przebieg co kwadrans nie może stać w kolejce `supabase-writers`
ani pingować heartbeatu, który pilnuje ciszy pętli.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "bot_poll.yml"

PAID_SECRET_KEYS = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEYS",
}
ALLOWED_SECRET_KEYS = {"SUPABASE_URL", "SUPABASE_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"}


def _spec() -> dict[Any, Any]:
    spec: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return spec


def _steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = _spec()["jobs"]["poll-owner-commands"]["steps"]
    return steps


def _bot_env() -> dict[str, str]:
    step = next(s for s in _steps() if str(s.get("run", "")).endswith("main_bot.py"))
    return {str(key): str(value) for key, value in dict(step.get("env") or {}).items()}


def test_bot_polls_every_fifteen_minutes_and_can_be_run_by_hand() -> None:
    spec = _spec()
    # PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True.
    triggers: dict[str, Any] = spec.get("on", spec.get(True)) or {}

    assert triggers["schedule"] == [{"cron": "*/15 * * * *"}]
    assert "workflow_dispatch" in triggers


def test_bot_reads_only_the_free_secrets() -> None:
    from_secrets = {key for key, value in _bot_env().items() if "secrets." in value}

    assert from_secrets == ALLOWED_SECRET_KEYS


def test_bot_maps_no_paid_secret_but_builds_settings_with_placeholders() -> None:
    env = _bot_env()

    assert not [key for key in PAID_SECRET_KEYS if "secrets." in env.get(key, "")]
    for key in ("OPENAI_API_KEY", "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEYS"):
        assert env.get(key), f"brak zaślepki {key} w kroku bota"


def test_bot_does_not_ping_the_loop_heartbeat() -> None:
    assert "HEARTBEAT_URL" not in _bot_env()


def test_without_secrets_every_later_step_is_skipped() -> None:
    first, *rest = _steps()
    work = [step for step in rest if "failure()" not in str(step.get("if", ""))]

    assert first.get("id") == "secrets"
    assert work
    for step in work:
        condition = str(step.get("if", ""))
        assert "steps.secrets.outputs.configured == 'true'" in condition or (
            "steps.flag.outputs.enabled == 'true'" in condition
        ), step.get("name")


def test_overlapping_runs_queue_instead_of_answering_twice() -> None:
    assert _spec()["concurrency"] == {"group": "bot-poll", "cancel-in-progress": False}


def test_bot_stays_out_of_the_writers_queue() -> None:
    job = _spec()["jobs"]["poll-owner-commands"]

    assert "concurrency" not in job
    assert _spec()["concurrency"]["group"] != "supabase-writers"
