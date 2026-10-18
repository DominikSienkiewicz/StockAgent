"""Strażnik read-only dla `scripts/prod-state-check.sh` (NIEZ-47).

Skrypt czyta produkcję: GitHub przez `gh` i Supabase kluczem aplikacji. Każdy zapis
zmieniałby stan, który skrypt ma tylko zmierzyć, a wypisany klucz trafiłby do
pliku z wynikiem. Polityka jest więc sprawdzana statycznie na treści skryptu,
bez sieci i bez kluczy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

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
