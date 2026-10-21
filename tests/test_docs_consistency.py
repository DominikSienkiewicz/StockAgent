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


def test_keys_called_unused_are_not_referenced_by_any_workflow() -> None:
    # DOC-03: docs nazywały TELEGRAM_*, SLACK_WEBHOOK_URL i FRED_API_KEY kluczami,
    # których „żaden workflow nie używa”, choć workflowy je czytają — operator
    # zostawiał przez to kanały push puste.
    text = (ROOT / "docs" / "running-locally.md").read_text(encoding="utf-8")
    workflows = "".join(
        path.read_text(encoding="utf-8") for path in (ROOT / ".github" / "workflows").glob("*.yml")
    )
    # Fraza aż do „are reported” (albo końca zdania) — „e.g.” w środku nie może jej uciąć.
    claims = re.findall(r"no workflow references(.{0,200}?)(?:are reported|\.\s+[A-Z])", text, re.S)
    named = {name for claim in claims for name in re.findall(r"`([A-Z][A-Z0-9_]*)\*?`", claim)}
    referenced = sorted(name for name in named if f"secrets.{name}" in workflows)
    assert not referenced, f"docs nazywają nieużywanymi klucze z workflowów: {referenced}"


# DOC-02: `alpha_fusion_score` trafia do `ml_feature_store` jako KANDYDAT — nie
# należy do `ML_FEATURE_COLUMNS`, więc nazwanie go „ósmą cechą” to fałsz, który
# recenzent wyłapie przy przejściu z config.py do ml_features.py. Migracje SQL
# zostają historyczne i ich nie sprawdzamy.
EIGHTH_FEATURE_CLAIM = re.compile(r"ósm\w* cech|eighth (?:ML |XGBoost )?feature", re.IGNORECASE)
EIGHTH_FEATURE_SOURCES = ("src/**/*.py", "main_*.py", "config.toml", "README.md", "docs/*.md")


def test_alpha_fusion_score_is_not_called_a_model_feature() -> None:
    from src.application.ml_features import ML_FEATURE_COLUMNS

    assert "alpha_fusion_score" not in ML_FEATURE_COLUMNS
    claims = [
        f"{path.relative_to(ROOT)}:{number}"
        for pattern in EIGHTH_FEATURE_SOURCES
        for path in sorted(ROOT.glob(pattern))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if EIGHTH_FEATURE_CLAIM.search(line)
    ]
    assert claims == []


def test_eighth_feature_pattern_matches_both_languages() -> None:
    assert EIGHTH_FEATURE_CLAIM.search("dokłada ósmą cechę ML")
    assert EIGHTH_FEATURE_CLAIM.search("as an eighth feature")
    assert not EIGHTH_FEATURE_CLAIM.search("kandydat na cechę ML")

def test_how_it_works_names_the_prediction_horizon_honestly() -> None:
    # WIA-28: pętla chodzi raz na dobę handlową, więc „12h return” to fałsz
    # widoczny dla quanta — horyzont to zwrot do następnego cyklu.
    text = (ROOT / "docs" / "how-it-works.md").read_text(encoding="utf-8")

    assert "predicts a 12h _return_" not in text
    assert "return to the next cycle" in text
