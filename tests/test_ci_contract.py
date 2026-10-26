"""Kontrakt jobów bramki w `ci.yml` poza samymi komendami (te pilnuje test_quality_gate_sync).

CI-16: liczby testów z commitów (np. 9d98e16) były nieodtwarzalne. Oba kroki pytest
joba `quality` piszą raport JUnit, a krok `!cancelled()` dopisuje z niego tests,
skipped i failures do podsumowania joba — w commitach linkujemy run zamiast liczb.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parent.parent
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
JUNIT_SUMMARY = "main_tools.py junit-summary"


def _quality_steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"][
        "quality"
    ]["steps"]
    return steps


def _pytest_tokens(step: dict[str, Any]) -> list[str] | None:
    tokens = shlex.split(str(step.get("run", "")).split("#", 1)[0])
    return tokens if "pytest" in tokens else None


def _junit_path(tokens: list[str]) -> str | None:
    paths = [
        token.removeprefix("--junitxml=") for token in tokens if token.startswith("--junitxml=")
    ]
    return paths[0] if len(paths) == 1 else None


def test_pytest_steps_emit_junit() -> None:
    steps = _quality_steps()
    pytest_steps = [
        (index, tokens) for index, step in enumerate(steps) if (tokens := _pytest_tokens(step))
    ]
    reports = [_junit_path(tokens) for _, tokens in pytest_steps]
    summaries = [
        (index, step)
        for index, step in enumerate(steps)
        if JUNIT_SUMMARY in str(step.get("run", ""))
    ]

    assert len(pytest_steps) == 2, "job quality ma dwa kroki pytest (jednostkowe i kontenerowe)"
    assert None not in reports and len(set(reports)) == 2, f"--junitxml w krokach pytest: {reports}"
    assert len(summaries) == 1, f"jeden krok `{JUNIT_SUMMARY}` w jobie quality"
    index, summary = summaries[0]
    run = str(summary["run"])
    assert index > max(position for position, _ in pytest_steps), "podsumowanie po krokach pytest"
    assert "!cancelled()" in str(summary.get("if", "")), "podsumowanie także po czerwonym pytest"
    assert all(str(report) in run for report in reports), run
    assert '>> "$GITHUB_STEP_SUMMARY"' in run


# CI-23: workflowy z sekretami i skrypty bash operujące na sekretach repo nie miały żadnej
# analizy statycznej. Job `lint-infra` uruchamia actionlint (składnia workflowów i shellcheck
# bloków `run:`), zizmor (template injection, uprawnienia, nieprzypięte akcje, artipacked)
# i shellcheck skryptów — każde narzędzie w przypiętej wersji (pilnuje test_workflow_policy).
LINT_INFRA_TOOLS = ("actionlint", "zizmor", "shellcheck")
SECRET_SCRIPTS = ("scripts/gh-secrets-check.sh", "scripts/gh-secrets-sync.sh")


def _lint_infra_runs() -> str:
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    assert "lint-infra" in jobs, "ci.yml bez joba lint-infra"
    return "\n".join(str(step.get("run", "")) for step in jobs["lint-infra"]["steps"])


@pytest.mark.parametrize("tool", LINT_INFRA_TOOLS)
def test_lint_infra_runs_each_linter(tool: str) -> None:
    assert re.search(rf"\b{tool}\b", _lint_infra_runs()), f"lint-infra nie uruchamia {tool}"


def test_lint_infra_shellchecks_the_secret_scripts() -> None:
    runs = _lint_infra_runs()
    shellchecked = {
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "scripts").glob("*.sh"))
        if "scripts/*.sh" in runs or str(path.relative_to(ROOT)) in runs
    }

    assert set(SECRET_SCRIPTS) <= shellchecked, runs
