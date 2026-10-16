"""Strażnik szwu workflow ↔ `config.toml` ↔ adapter LLM.

`config.toml` ustawia `llm_provider = "anthropic"`, a adapter LLM budują
`main_agent.py` (Fast Loop) i `main_trainer.py` (niedzielny recap). Gdy SDK
`anthropic` siedział tylko w extra, Slow Loop synchronizował środowisko bez
niego: `ModuleNotFoundError` przy budowie adaptera połykał `except Exception`,
job kończył się na zielono, a mail nie wychodził.

Kontrakt: każdy job uruchamiający entry-point z LLM ma SDK — albo jako
zależność rdzeniową w `[project].dependencies`, albo przez `--extra anthropic`
w każdym swoim `uv sync`.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
LLM_ENTRYPOINTS = ("main_agent.py", "main_trainer.py")


def _llm_jobs() -> list[tuple[str, list[dict[str, object]]]]:
    """(plik workflow, kroki joba) dla każdego joba, który odpala entry-point z LLM."""
    jobs: list[tuple[str, list[dict[str, object]]]] = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in spec.get("jobs", {}).values():
            steps = job.get("steps", [])
            runs = " ".join(str(s.get("run", "")) for s in steps)
            if any(entrypoint in runs for entrypoint in LLM_ENTRYPOINTS):
                jobs.append((path.name, steps))
    return jobs


def _core_dependency_names() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return {
        re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].strip().lower()
        for requirement in project["dependencies"]
    }


LLM_JOBS = _llm_jobs()


def test_llm_entrypoints_run_in_some_workflow() -> None:
    # Bez tego parametryzacja poniżej mogłaby po cichu przejść na pustej liście.
    assert LLM_JOBS, "żaden workflow nie uruchamia main_agent.py ani main_trainer.py"


@pytest.mark.parametrize(
    "workflow, steps", LLM_JOBS, ids=[workflow for workflow, _ in LLM_JOBS]
)
def test_llm_entrypoint_workflow_installs_anthropic_sdk(
    workflow: str, steps: list[dict[str, object]]
) -> None:
    if "anthropic" in _core_dependency_names():
        return
    sync_cmds = [str(s.get("run", "")) for s in steps if "uv sync" in str(s.get("run", ""))]
    assert sync_cmds and all("--extra anthropic" in cmd for cmd in sync_cmds), (
        f"{workflow} uruchamia entry-point z LLM (llm_provider = anthropic), ale "
        "`uv sync` nie instaluje SDK: dodaj `anthropic` do [project].dependencies "
        "albo `--extra anthropic` do każdego `uv sync` w jobie."
    )
