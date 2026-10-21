"""Strażnik read-only dla `scripts/prod-state-check.sh` (NIEZ-47).

Skrypt czyta produkcję: GitHub przez `gh` i Supabase kluczem aplikacji. Każdy zapis
zmieniałby stan, który skrypt ma tylko zmierzyć, a wypisany klucz trafiłby do
pliku z wynikiem. Polityka jest więc sprawdzana statycznie na treści skryptu,
bez sieci i bez kluczy.

Tu też polityka workflowów bez kodu aplikacji: ręczny smoke testów na żywym API
(CI-05) i grupy Dependabota (CI-07).
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "prod-state-check.sh"

# (wzorzec, dlaczego zakazany) — każdy oznacza zapis albo wyciek sekretu.
FORBIDDEN = (
    (r"\bgh\s+(secret|variable)\s+(set|remove)\b", "zapis sekretów albo zmiennych"),
    (r"\bgh\s+workflow\s+(enable|disable|run)\b", "zmiana stanu workflowu"),
    (r"\bgh\s+run\s+(rerun|cancel)\b", "sterowanie przebiegiem"),
    (
        r"\bgh\s+api\b[^\n]*(\s-X\b|--method\b|\s-[fF]\b|--field\b|--raw-field\b|--input\b)",
        "gh api z metodą albo ciałem żądania",
    ),
    (r"rest/v1/rpc", "wywołanie funkcji bazy przez RPC"),
    (r"(?i)\b(insert|update|delete|alter|drop|truncate|grant|revoke|create)\b", "zapis SQL"),
    (r"\bset\s+-[a-z]*x|xtrace", "śledzenie powłoki wypisałoby klucze"),
)


def _code_lines() -> list[str]:
    return [
        line
        for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]


def test_prod_state_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


@pytest.mark.parametrize("pattern, reason", FORBIDDEN)
def test_prod_state_script_is_read_only(pattern: str, reason: str) -> None:
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    found = [line for line in lines if re.search(pattern, line)]
    assert not found, f"{reason}: {found}"


def test_prod_state_script_requests_are_get_only() -> None:
    calls = [line for line in _code_lines() if re.search(r"\bcurl\s+-", line)]
    assert calls, "skrypt nie robi żadnego żądania do Supabase"
    for call in calls:
        assert "--get" in call, call
        assert "-K -" in call, f"klucz ma iść przez stdin, nie przez argv: {call}"
        assert not re.search(r"\s(-X|--request|-d|--data\S*)\s", call), call


def test_prod_state_script_never_prints_the_key() -> None:
    for line in _code_lines():
        if re.search(r"\b(echo|printf)\b", line) and re.search(r"\$\{?SB_KEY\b", line):
            assert "| curl -K -" in line, f"klucz wypisany poza konfiguracją curl: {line}"


# CI-05: testy `-m integration` (prawdziwe API, ~5 płatnych wywołań i jeden mail)
# nie biegły w żadnym workflowie. Smoke jest WYŁĄCZNIE ręczny (decyzja właściciela):
# bez crona, push i PR-ów, z osobnymi sekretami `SMOKE_*`, żeby nie zjadał limitów
# kluczy produkcyjnych i nie mailował na adres digestu.
SMOKE_WORKFLOW = ROOT / ".github" / "workflows" / "integration_smoke.yml"


def _smoke_spec() -> dict[str, Any]:
    spec: dict[str, Any] = yaml.safe_load(SMOKE_WORKFLOW.read_text(encoding="utf-8"))
    return spec


def _pytest_marker_expression(command: str) -> str | None:
    """Wyrażenie `-m` komendy pytest ('' bez `-m`), None gdy to nie pytest."""
    tokens = shlex.split(command.split("#", 1)[0])
    if "pytest" not in tokens:
        return None
    args = tokens[tokens.index("pytest") + 1 :]
    return args[args.index("-m") + 1] if "-m" in args[:-1] else ""


def test_integration_smoke_runs_only_integration_marker() -> None:
    spec = _smoke_spec()
    # PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True.
    triggers = spec.get("on", spec.get(True))
    steps = [step for job in spec["jobs"].values() for step in job.get("steps", [])]
    markers = [
        marker
        for step in steps
        if (marker := _pytest_marker_expression(str(step.get("run", "")))) is not None
    ]

    assert isinstance(triggers, dict) and set(triggers) == {"workflow_dispatch"}, triggers
    assert markers == ["integration"], markers


def test_integration_smoke_uses_only_its_own_secrets() -> None:
    secrets = set(
        re.findall(r"secrets\.([A-Za-z0-9_]+)", SMOKE_WORKFLOW.read_text(encoding="utf-8"))
    )

    assert secrets, "smoke bez sekretów pominie każdy test na żywym API"
    assert all(name.startswith("SMOKE_") for name in secrets), sorted(secrets)
