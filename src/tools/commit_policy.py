"""DOC-27: reguła TDD z CLAUDE.md jako sprawdzalna polityka gałęzi.

Poprawka błędu zaczyna się od testu, który go odtwarza — ale z historii nie dało się tego
sprawdzić. Polityka: commit `fix:` musi zmieniać coś w `tests/` albo mieć przed sobą na tej
samej gałęzi commit `test:`. Poprawki samego CI i dokumentacji (`fix(ci):`, `fix(docs):`)
nie mają testu do napisania, a merge commity nie są zmianą — oba są pomijane, inaczej
polityka byłaby fałszywie czerwona. Czyste funkcje; `git log` uruchamia
`main_tools.py commit-policy`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Separator rekordów (ASCII RS, `%x1e` w formacie `git log`) — nie występuje w temacie
# commita ani w ścieżce pliku.
RECORD_SEPARATOR = "\x1e"
GIT_LOG_FORMAT = "%x1e%s"
EXEMPT_FIX_SCOPES = frozenset({"ci", "docs"})
TEST_DIRECTORY = "tests/"
_CONVENTIONAL = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?!?:\s")


@dataclass(frozen=True)
class CommitInfo:
    """Temat commita i ścieżki plików, które zmienia."""

    subject: str
    files: Sequence[str]


def _type_and_scopes(subject: str) -> tuple[str, frozenset[str]] | None:
    match = _CONVENTIONAL.match(subject)
    if match is None:
        return None
    scopes = frozenset(scope.strip() for scope in (match.group("scope") or "").split(",") if scope)
    return match.group("type"), scopes


def _exempt(scopes: frozenset[str]) -> bool:
    """`fix(ci):`, `fix(docs):` albo `fix(ci,docs):` — poprawka bez kodu do przetestowania."""
    return bool(scopes) and scopes <= EXEMPT_FIX_SCOPES


def _changes_a_test(commit: CommitInfo) -> bool:
    return any(path.startswith(TEST_DIRECTORY) for path in commit.files)


def fix_commits_without_test(commits: Sequence[CommitInfo]) -> list[str]:
    """Tematy commitów `fix:` bez testu, w kolejności chronologicznej (najstarszy pierwszy).

    Commit `fix:` jest w porządku, gdy sam zmienia plik w `tests/` albo gdy wcześniej na
    gałęzi był commit `test:`. Zakres `ci` / `docs` (także oba naraz) jest zwolniony;
    merge commity i inne typy są pomijane.
    """
    flagged: list[str] = []
    tested_before = False
    for commit in commits:
        if commit.subject.startswith("Merge "):
            continue
        parsed = _type_and_scopes(commit.subject)
        if parsed is None:
            continue
        kind, scopes = parsed
        tested_before = tested_before or kind == "test"
        if kind == "fix" and not (_exempt(scopes) or tested_before or _changes_a_test(commit)):
            flagged.append(commit.subject)
    return flagged


def parse_git_log(output: str) -> list[CommitInfo]:
    """Commity z `git log --reverse --name-only --format=%x1e%s` (temat, pusta linia, pliki)."""
    commits: list[CommitInfo] = []
    for record in output.split(RECORD_SEPARATOR)[1:]:
        subject, _, rest = record.partition("\n")
        files = tuple(line for line in rest.splitlines() if line.strip())
        commits.append(CommitInfo(subject=subject.strip(), files=files))
    return commits
