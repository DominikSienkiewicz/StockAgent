"""Strażnik twierdzeń docs o wagach XGBoost — tekst to twierdzenie, `git ls-files` to dowód.

Docs mówiły, że model „żyje w pliku `.ubj` w repo”, a Slow Loop „commituje nowe
wagi”, podczas gdy w gicie nie było żadnego pliku wag: trener pomija symbole z
za małą próbką, a workflow commituje plik tylko wtedy, gdy kandydat pobije
baseline. Dopóki żaden `.ubj` nie jest śledzony, docs nie mogą tego twierdzić.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

PUBLIC_DOCS = ("README.md", "docs/how-it-works.md", "docs/running-locally.md")

# Twierdzenia prawdziwe tylko przy commitowanych wagach.
COMMITTED_WEIGHTS_CLAIMS = (
    r"\.ubj` file inside the repo",
    r"commits the new weights back to the repo",
    r"\+ commit new weights",
)


def _tracked_weight_files() -> list[str]:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "*.ubj"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("brak repozytorium git — nie da się sprawdzić śledzonych wag")
    return [line for line in listed.stdout.splitlines() if line]


@pytest.mark.parametrize("doc", PUBLIC_DOCS)
def test_docs_do_not_claim_committed_weights_when_none_tracked(doc: str) -> None:
    if _tracked_weight_files():
        pytest.skip("wagi są śledzone — twierdzenie o modelu w repo jest prawdziwe")

    text = (ROOT / doc).read_text(encoding="utf-8")
    found = [claim for claim in COMMITTED_WEIGHTS_CLAIMS if re.search(claim, text)]
    assert not found, f"{doc} twierdzi o commitowanych wagach, których nie ma: {found}"
