"""Strażnik odznak wersji w README — odznaka to twierdzenie, a `uv.lock` to dowód.

Odznaka „LangGraph 1.3" wisiała przy locku 1.2.0. Recenzent porównuje to
w minutę, więc wersja z odznaki musi być prefiksem wersji zablokowanej.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

# Nazwa na odznace → nazwa pakietu w uv.lock.
VERSIONED_BADGES = {"LangGraph": "langgraph", "XGBoost": "xgboost"}


def _locked_version(package: str) -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return next(p["version"] for p in lock["package"] if p["name"] == package)


@pytest.mark.parametrize("badge, package", VERSIONED_BADGES.items())
def test_readme_version_badge_matches_the_lock(badge: str, package: str) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(rf"img\.shields\.io/badge/{badge}-([0-9.]+)-", readme)
    assert match, f"README nie ma odznaki wersji {badge}"

    locked = _locked_version(package)
    assert f"{locked}.".startswith(f"{match.group(1)}."), (
        f"odznaka {badge} {match.group(1)} ≠ uv.lock {package} {locked}"
    )
