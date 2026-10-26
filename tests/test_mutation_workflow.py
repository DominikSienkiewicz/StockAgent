"""TEST-35: ręczny pomiar mutation testingu domeny (mutmut), bez progu.

Mutation score jest twardszym dowodem jakości testów niż coverage: ujawnia testy-wydmuszki
typu „is not None”. Pomiar obejmuje bramkę volatility (`asset`), reżim, sizing, VaR
i FinOps; każdy moduł to osobny job macierzy z testami domeny, które go importują
(pełne tests/domain na mutanta to godziny). Próg właściciel ustali po pierwszym pomiarze,
więc przeżyte mutanty nie czerwienią joba — wynik idzie do podsumowania.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "mutation_testing.yml"
MUTATED_MODULES = frozenset({"asset", "regime", "sizing", "var", "finops"})


def _spec() -> dict[Any, Any]:
    assert WORKFLOW.is_file(), f"brak {WORKFLOW.relative_to(ROOT)}"
    spec: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return spec


def _job() -> dict[str, Any]:
    (job,) = _spec()["jobs"].values()
    assert isinstance(job, dict)
    return job


def _matrix() -> dict[str, list[str]]:
    entries = _job()["strategy"]["matrix"]["include"]
    return {str(entry["module"]): str(entry["tests"]).split() for entry in entries}


def _domain_tests_importing(module: str) -> list[str]:
    pattern = re.compile(rf"\bsrc\.domain\.{module}\b|from src\.domain import [^\n]*\b{module}\b")
    return sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests" / "domain").glob("test_*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    )


def test_mutation_workflow_runs_only_by_hand() -> None:
    spec = _spec()
    # PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True.
    triggers = spec.get("on", spec.get(True))

    assert triggers == {"workflow_dispatch": None}, triggers


def test_mutation_matrix_covers_the_five_modules() -> None:
    assert set(_matrix()) == MUTATED_MODULES
    assert _job()["strategy"].get("fail-fast") is False, "jeden moduł nie może zabić pomiaru reszty"


@pytest.mark.parametrize("module", sorted(MUTATED_MODULES))
def test_mutation_job_runs_every_domain_test_of_its_module(module: str) -> None:
    tests = _matrix().get(module, [])

    assert tests == _domain_tests_importing(module), (
        f"testy mutanta {module} ≠ testy tests/domain, które importują src.domain.{module} — "
        "nowy test modułu dopisz do macierzy mutation_testing.yml"
    )


def test_mutation_results_land_in_the_job_summary_without_a_threshold() -> None:
    runs = [str(step.get("run", "")) for step in _job()["steps"]]
    summaries = [
        step for step in _job()["steps"] if "junit-summary --mutation" in str(step.get("run", ""))
    ]

    assert any("mutmut run" in run for run in runs)
    assert any("mutmut junitxml" in run for run in runs)
    assert len(summaries) == 1
    assert "!cancelled()" in str(summaries[0].get("if", ""))
    assert '>> "$GITHUB_STEP_SUMMARY"' in str(summaries[0]["run"])
