"""Strażnik kanonicznych skryptów cyklu gałęzi (worktree → status → merge).

`scripts/merge.sh`, `scripts/worktree-new.sh` i `scripts/worktree-status.sh` to
kopie bajt w bajt z szablonów Platform (`Platform/templates/`). Edytuje się je
w kanonie, nie tutaj — linia `CANON:` w nagłówku mówi, skąd przyszła kopia.
Test pilnuje, że każdy skrypt jest, da się go uruchomić i parsuje się w bashu
(`bash -n` nie wykonuje skryptu, tylko sprawdza składnię).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
CANONICAL_SCRIPTS = ("merge.sh", "worktree-new.sh", "worktree-status.sh")
HEADER_LINES = 10


@pytest.mark.parametrize("name", CANONICAL_SCRIPTS)
def test_canonical_script_is_present_and_executable(name: str) -> None:
    script = ROOT / "scripts" / name

    assert script.is_file(), f"brak scripts/{name}"
    assert os.access(script, os.X_OK), f"scripts/{name} nie ma bitu wykonywalności"


@pytest.mark.parametrize("name", CANONICAL_SCRIPTS)
def test_canonical_script_names_its_platform_template(name: str) -> None:
    header = (ROOT / "scripts" / name).read_text(encoding="utf-8").splitlines()[:HEADER_LINES]

    assert any(
        re.search(rf"CANON: .*Platform/templates/{re.escape(name)}\b", line) for line in header
    ), f"scripts/{name} nie wskazuje kanonu Platform/templates/{name} w nagłówku"


@pytest.mark.parametrize("name", CANONICAL_SCRIPTS)
def test_canonical_script_parses_in_bash(name: str) -> None:
    bash = shutil.which("bash")
    assert bash is not None, "brak bash w PATH"

    result = subprocess.run(
        [bash, "-n", str(ROOT / "scripts" / name)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
