"""Strażnik alertu o porażce jobu (NIEZ-23).

GitHub mailuje o porażce crona tylko autora ostatniej zmiany crona i nie odróżnia
timeoutu od crasha, a anulowanie przez timeout jest całkowicie ciche. Każdy
workflow z cronem albo zapisem kończy się więc krokiem, który przy porażce lub
anulowaniu wysyła mail przez Resend — kanał, którego sekrety już są w repo.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
SCRIPT = ROOT / "scripts" / "notify-failure.sh"
ALERTED_WORKFLOWS = (
    "fast_loop_12h.yml",
    "slow_loop_weekly.yml",
    "shock_watch_hourly.yml",
    "db_migrate.yml",
    "deadman_daily.yml",
    "bot_poll.yml",
)

FAKE_CURL = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$FAKE_DIR/args"
cat > "$FAKE_DIR/stdin"
for arg in "$@"; do
  case "$arg" in @*) cp "${arg#@}" "$FAKE_DIR/body" ;; esac
done
"""


@pytest.mark.parametrize("name", ALERTED_WORKFLOWS)
def test_every_job_ends_with_a_failure_alert(name: str) -> None:
    spec = yaml.safe_load((WORKFLOWS_DIR / name).read_text(encoding="utf-8"))
    for job_name, job in spec["jobs"].items():
        last = job["steps"][-1]
        condition = str(last.get("if", ""))
        assert "failure()" in condition and "cancelled()" in condition, (name, job_name)
        assert "scripts/notify-failure.sh" in str(last.get("run", "")), (name, job_name)
        env = last.get("env", {})
        assert env.get("RESEND_API_KEY") == "${{ secrets.RESEND_API_KEY }}", (name, job_name)
        assert env.get("DIGEST_TO_EMAIL") == "${{ secrets.DIGEST_TO_EMAIL }}", (name, job_name)


def _run_script(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    if shutil.which("jq") is None:
        pytest.skip("skrypt wymaga jq (jest na runnerach GitHuba)")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(FAKE_CURL, encoding="utf-8")
    curl.chmod(0o755)
    full_env = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_DIR": str(tmp_path),
        **env,
    }
    result = subprocess.run(
        [str(SCRIPT), "🟢 Fast Loop"], env=full_env, capture_output=True, text=True, check=False
    )
    return result, tmp_path


def test_failure_alert_mails_the_run_with_the_key_on_stdin(tmp_path: Path) -> None:
    result, out = _run_script(
        tmp_path,
        RESEND_API_KEY="re_test_key",
        DIGEST_TO_EMAIL="owner@example.com",
        RUN_URL="https://github.com/o/r/actions/runs/42",
        JOB_STATUS="cancelled",
    )

    assert result.returncode == 0, result.stderr
    args = (out / "args").read_text(encoding="utf-8")
    assert "https://api.resend.com/emails" in args
    assert "re_test_key" not in args, "klucz nie może trafić do argv"
    assert "Authorization: Bearer re_test_key" in (out / "stdin").read_text(encoding="utf-8")
    body = json.loads((out / "body").read_text(encoding="utf-8"))
    assert body["to"] == ["owner@example.com"]
    assert "Fast Loop" in body["subject"] and "cancelled" in body["subject"]
    assert "https://github.com/o/r/actions/runs/42" in body["text"]
    assert body["from"]


def test_missing_key_warns_without_failing_or_calling_resend(tmp_path: Path) -> None:
    result, out = _run_script(tmp_path, DIGEST_TO_EMAIL="owner@example.com")

    assert result.returncode == 0
    assert "::warning::" in result.stdout
    assert not (out / "args").exists()
