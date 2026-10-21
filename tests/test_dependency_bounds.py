"""Strażnik dolnych granic zależności (CI-12): deklaracja w `pyproject.toml` ↔ `uv.lock`.

`langgraph>=0.0.39` obiecywał zgodność z API sprzed 1.0, której kod nie ma, a lock
trzymał 1.2.0, więc nikt tego nie widział; tak samo `xgboost>=2`, `pandas>=2.2`
i `numpy>=1.26` przy locku 3.x / 3.x / 2.x. Heurystyka: dolna granica każdej
zależności rdzeniowej ma ten sam major co wersja z locka. To nie dowód zgodności
(część pakietów łamie API w minorach) — dowodem jest ręczny job `lowest-direct`
w `ci.yml`, który instaluje dolne granice i puszcza testy jednostkowe.
Bez `packaging`: strażnik nie może zależeć od niezadeklarowanych pakietów.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parent.parent
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_LOWER_BOUND = re.compile(r"(?:>=|==|~=|>)\s*(\d+)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lower_bound_major(requirement: str) -> int | None:
    """Major dolnej granicy wpisu zależności (`>=`, `==`, `~=`, `>`), None bez niej."""
    specifiers = requirement.split(";", 1)[0]
    match = _LOWER_BOUND.search(specifiers)
    return int(match.group(1)) if match else None


def _core_requirements() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements: dict[str, str] = {}
    for requirement in project["dependencies"]:
        match = _NAME.match(requirement.strip())
        assert match, f"nieczytelny wpis zależności: {requirement!r}"
        requirements[_normalize(match.group(0))] = requirement
    return requirements


def _locked_versions() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {_normalize(package["name"]): package["version"] for package in lock["package"]}


@pytest.mark.parametrize(
    ("requirement", "major"),
    [
        ("langgraph>=1.2,<2", 1),
        ("supabase>=2.30", 2),
        ("foo[extra] >= 3.0 ; python_version < '3.13'", 3),
        ("bar~=0.102.0", 0),
        ("baz<3", None),
        ("qux", None),
    ],
)
def test_lower_bound_reader(requirement: str, major: int | None) -> None:
    assert lower_bound_major(requirement) == major


def test_every_core_dependency_declares_a_lower_bound() -> None:
    unbounded = [req for req in _core_requirements().values() if lower_bound_major(req) is None]

    assert unbounded == []


def test_lower_bounds_share_major_with_lock() -> None:
    locked = _locked_versions()
    mismatched = {
        requirement: locked[name]
        for name, requirement in _core_requirements().items()
        if (major := lower_bound_major(requirement)) is not None
        and major != int(locked[name].split(".", 1)[0])
    }

    assert mismatched == {}, (
        f"dolna granica w innym majorze niż lock: {mismatched} — deklaracja obiecuje API, "
        "którego kod nie testuje. Podnieś granicę i uruchom job `lowest-direct` w ci.yml."
    )


def _jobs() -> dict[str, Any]:
    jobs: dict[str, Any] = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    return jobs


def test_ci_has_a_manual_lowest_direct_job() -> None:
    lowest = {
        name: job
        for name, job in _jobs().items()
        if any("--resolution lowest-direct" in str(step.get("run", "")) for step in job["steps"])
    }

    assert len(lowest) == 1, f"oczekiwany jeden job `--resolution lowest-direct`: {sorted(lowest)}"
    (job,) = lowest.values()
    assert job.get("if") == "github.event_name == 'workflow_dispatch'"
    runs = [str(step.get("run", "")) for step in job["steps"]]
    assert any('pytest -m "not integration and not containers"' in run for run in runs), runs
