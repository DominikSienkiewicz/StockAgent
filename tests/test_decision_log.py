"""Strażnik dziennika decyzji i dziennika inżynierskiego (DOC-40).

Kod, `config.toml` i komentarze odwoływały się do „roadmap #N” — numerów z
gitignorowanego backlogu, których recenzent nie ma jak rozwiązać (a dwie serie
numeracji, czerwcowa i lipcowa, nachodziły na siebie: `#5` znaczyło kontagion i
skrót na komunikator). Teraz numer ma postać `R-N`, a `docs/decisions.md` jest
jedynym miejscem, które go definiuje, razem z hashem commita. Decyzje (`D-N`)
też niosą hashe, a dziennik inżynierski (incydent → reguła → strażnik) wskazuje
testy, które naprawdę istnieją.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DECISIONS = ROOT / "docs" / "decisions.md"
ENGINEERING_LOG = ROOT / "docs" / "engineering-log.md"
THIS_FILE = Path(__file__).resolve()

ROADMAP_ROW = re.compile(r"^\| (R-\d+) \|(.*)\|\s*$", re.MULTILINE)
DECISION_HEADING = re.compile(r"^### (D-\d+) · .+$", re.MULTILINE)
COMMIT_HASH = re.compile(r"`([0-9a-f]{7,40})`")
R_REFERENCE = re.compile(r"(?<![\w-])R-(\d+)\b")
STATUS = re.compile(r"\*\*Status\*\* (in force|(?:partly )?superseded by D-\d+)")
MIN_DECISIONS = 8

REFERENCE_SOURCES = (
    "README.md",
    "config.toml",
    "docs/*.md",
    "src/**/*.py",
    "main_*.py",
    "tests/**/*.py",
    ".github/**/*.yml",
    "scripts/*.sh",
)

# Gdzie „roadmap #N” nie może już stać: czytelnik i konfiguracja.
PHRASE_SOURCES = ("README.md", "config.toml", "docs/*.md", "src/config.py", "main_*.py")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _files(patterns: tuple[str, ...]) -> list[Path]:
    return sorted(
        {
            path
            for pattern in patterns
            for path in ROOT.glob(pattern)
            if path.is_file() and path.resolve() != THIS_FILE
        }
    )


def _reference_files() -> list[Path]:
    return _files(REFERENCE_SOURCES)


def _roadmap_rows() -> dict[str, list[str]]:
    return {
        match.group(1): [cell.strip() for cell in match.group(2).split("|")]
        for match in ROADMAP_ROW.finditer(_text(DECISIONS))
    }


def _decision_sections() -> dict[str, str]:
    text = _text(DECISIONS)
    headings = list(DECISION_HEADING.finditer(text))
    sections = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        tail = text.find("\n## ", heading.end())
        if tail != -1:
            end = min(end, tail)
        sections[heading.group(1)] = text[heading.start() : end]
    return sections


def _history_available() -> bool:
    try:
        shallow = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return shallow.stdout.strip() == "false"


def _commit_exists(sha: str) -> bool:
    found = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return found.returncode == 0


def _test_ids_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            ids.add(node.name)
        elif isinstance(node, ast.ClassDef):
            ids.update(
                f"{node.name}::{item.name}"
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_")
            )
    return ids


def _unresolved_guards(text: str) -> list[str]:
    """Strażnicy z tekstu: linkowane pliki testów muszą istnieć i nazywać testy,
    które w nich są. Zwraca opisy braków."""
    problems = []
    files = [
        (DECISIONS.parent / target).resolve() for target in re.findall(r"\]\(([^)#]+\.py)\)", text)
    ]
    named = re.findall(r"`((?:\w+::)?test_\w+)`", text)
    known: set[str] = set()
    for file in files:
        if file.is_file():
            known |= _test_ids_in(file)
        else:
            problems.append(f"brak pliku {file.relative_to(ROOT)}")
    if not files or not named:
        problems.append("brak linku do testu albo nazwy testu")
    problems.extend(f"brak testu {name}" for name in sorted(set(named) - known))
    return problems


def test_decision_log_and_engineering_log_exist() -> None:
    assert DECISIONS.is_file(), "brak docs/decisions.md"
    assert ENGINEERING_LOG.is_file(), "brak docs/engineering-log.md"


def test_every_roadmap_reference_resolves() -> None:
    defined = set(_roadmap_rows())
    unresolved = [
        f"{path.relative_to(ROOT)}:{number} {found.group(0)}"
        for path in _reference_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        for found in R_REFERENCE.finditer(line)
        if f"R-{found.group(1)}" not in defined
    ]

    assert defined, "docs/decisions.md nie definiuje żadnego R-N"
    assert unresolved == [], (
        f"odwołania do R-N, których docs/decisions.md nie definiuje: {unresolved}"
    )


def test_roadmap_ids_are_consecutive_and_each_row_has_a_commit() -> None:
    rows = _roadmap_rows()
    assert rows, "docs/decisions.md nie ma tabeli R-N"
    numbers = sorted(int(identifier[2:]) for identifier in rows)

    assert numbers == list(range(1, len(numbers) + 1)), f"numeracja R-N z dziurą: {numbers}"
    without_commit = [
        identifier for identifier, cells in rows.items() if not COMMIT_HASH.search(cells[1])
    ]
    assert without_commit == [], f"wiersze R-N bez hasha commita: {without_commit}"


def test_roadmap_flags_are_settings_fields() -> None:
    from src.config import Settings

    named = {
        flag for cells in _roadmap_rows().values() for flag in re.findall(r"`(\w+)`", cells[2])
    }

    assert named, "tabela R-N nie wymienia żadnej flagi"
    assert sorted(named - set(Settings.model_fields)) == []


def test_no_roadmap_hash_numbers_outside_the_logs() -> None:
    # „roadmap #12” prowadziło do pliku, którego nie ma w repo; opis dawnej nazwy
    # zostaje tylko w samych dziennikach.
    left = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in _files(PHRASE_SOURCES)
        if path not in (DECISIONS, ENGINEERING_LOG)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if re.search(r"roadmap #\d+", line, re.IGNORECASE)
    ]

    assert left == [], f"odwołania „roadmap #N” zamiast R-N: {left}"


@pytest.mark.parametrize("path", ["config.toml", "src/config.py"])
def test_config_comments_use_r_numbers_not_bare_hash_numbers(path: str) -> None:
    bare = [
        f"{path}:{number}"
        for number, line in enumerate(
            (ROOT / path).read_text(encoding="utf-8").splitlines(), start=1
        )
        if re.search(r"(?<![\w&])#\d{1,2}\b", line)
    ]

    assert bare == [], f"numer pozycji bez rozwiązania („#N”) zamiast R-N: {bare}"


def test_decision_log_has_enough_entries_with_status_and_commits() -> None:
    sections = _decision_sections()

    assert len(sections) >= MIN_DECISIONS
    numbers = sorted(int(identifier[2:]) for identifier in sections)
    assert numbers == list(range(1, len(numbers) + 1)), f"numeracja D-N z dziurą: {numbers}"
    problems = []
    for identifier, section in sections.items():
        meta = next((line for line in section.splitlines() if "**Commits**" in line), "")
        if not STATUS.search(section):
            problems.append(f"{identifier}: brak statusu")
        if not COMMIT_HASH.search(meta):
            problems.append(f"{identifier}: brak hasha commita")

    assert problems == []


def test_superseding_decisions_point_at_existing_entries() -> None:
    sections = _decision_sections()
    assert sections, "docs/decisions.md nie ma wpisów D-N"
    targets = re.findall(r"superseded by (D-\d+)", _text(DECISIONS))

    assert [target for target in targets if target not in sections] == []


@pytest.mark.skipif(not _history_available(), reason="płytka kopia repo — nie ma historii gita")
def test_every_cited_commit_exists_in_the_history() -> None:
    cited = {
        sha
        for text in (_text(DECISIONS), _text(ENGINEERING_LOG))
        for sha in COMMIT_HASH.findall(text)
    }

    assert cited, "dzienniki nie cytują żadnego commita"
    assert sorted(sha for sha in cited if not _commit_exists(sha)) == []


def test_decision_guards_name_tests_that_exist() -> None:
    assert _decision_sections(), "docs/decisions.md nie ma wpisów D-N"
    problems = {
        identifier: _unresolved_guards(section[section.index("**Guards**") :])
        if "**Guards**" in section
        else ["brak linii Guards"]
        for identifier, section in _decision_sections().items()
    }

    assert {key: value for key, value in problems.items() if value} == {}


def test_engineering_log_rows_carry_rule_guard_and_commit() -> None:
    rows = [
        line
        for line in _text(ENGINEERING_LOG).splitlines()
        if line.startswith("| ") and not line.startswith(("| Incident", "|---"))
    ]

    assert len(rows) >= 8, "dziennik inżynierski ma za mało incydentów"
    problems = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if len(cells) != 4:
            problems.append((row[:40], "oczekiwano 4 kolumn"))
            continue
        _incident, _rule, guard, commits = cells
        if not COMMIT_HASH.search(commits):
            problems.append((row[:40], "brak hasha commita"))
        missing = _unresolved_guards(guard)
        if missing:
            problems.append((row[:40], missing))

    assert problems == []


@pytest.mark.parametrize("document", ["docs/decisions.md", "docs/engineering-log.md"])
def test_readme_routes_to_the_logs(document: str) -> None:
    assert f"]({document})" in _text(ROOT / "README.md")
