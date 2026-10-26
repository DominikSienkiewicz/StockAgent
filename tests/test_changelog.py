"""CI-21: CHANGELOG.md prowadzony ręcznie, bez release-please.

Wydania v0.1–v0.3 to kamienie milowe nałożone na istniejące commity; tagi tworzy
i pushuje właściciel. Test sprawdza wyłącznie plik: nie woła gita ani nie zakłada, że
tagi czy starsze commity istnieją w checkoucie CI (`actions/checkout` bierze jeden
commit). Pozycje roadmapy bierze z tabeli „Roadmap items” w `docs/decisions.md`, żeby
changelog i dziennik decyzji nie rozjechały się co do commitu.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
DECISIONS = ROOT / "docs" / "decisions.md"

# Pozycje, które trafiły do wydań v0.1–v0.3: R-1…R-18 (roadmapa lipca) i R-19…R-27
# (czerwcowe ulepszenia rady). Późniejsze wpisy idą do sekcji Unreleased bez wiersza.
MILESTONE_ITEMS = tuple(f"R-{number}" for number in range(1, 28))
RELEASES = ("v0.3", "v0.2", "v0.1")
EXPECTED_ORDER = ["Unreleased", *RELEASES]

_SECTION_HEADING = re.compile(r"^## \[(?P<name>[^\]]+)\](?: - (?P<date>\d{4}-\d{2}-\d{2}))?\s*$")
_TAG_TARGET = re.compile(r"^Tag target: `(?P<sha>[0-9a-f]{7,40})`", re.MULTILINE)
_ITEM_ROW = re.compile(r"^\|\s*(?P<id>R-\d+)\s*\|.*`(?P<sha>[0-9a-f]{7,40})`", re.MULTILINE)
_ROADMAP_TABLE_HEADING = "## Roadmap items"


def _text() -> str:
    assert CHANGELOG.is_file(), "brak CHANGELOG.md w korzeniu repo (CI-21)"
    return CHANGELOG.read_text(encoding="utf-8")


def _sections() -> dict[str, str]:
    """Nazwa wydania → treść sekcji `## [nazwa]`, w kolejności z pliku."""
    sections: dict[str, str] = {}
    name: str | None = None
    lines: list[str] = []
    for line in _text().splitlines():
        if line.startswith("## "):
            if name is not None:
                sections[name] = "\n".join(lines)
            match = _SECTION_HEADING.match(line)
            assert match, f"nagłówek wydania w formie `## [vX.Y] - RRRR-MM-DD`: {line!r}"
            name = match.group("name")
            lines = [line]
        elif name is not None:
            lines.append(line)
    if name is not None:
        sections[name] = "\n".join(lines)
    return sections


def _decisions_roadmap_rows() -> dict[str, str]:
    """R-N → commit z tabeli „Roadmap items” w docs/decisions.md."""
    _, _, after = DECISIONS.read_text(encoding="utf-8").partition(_ROADMAP_TABLE_HEADING)
    table = after.split("\n## ", 1)[0]
    return {match.group("id"): match.group("sha") for match in _ITEM_ROW.finditer(table)}


def test_changelog_starts_with_its_title() -> None:
    assert _text().startswith("# Changelog\n")


def test_changelog_sections_run_from_unreleased_down_to_v0_1() -> None:
    assert list(_sections()) == EXPECTED_ORDER


def test_each_release_has_a_date_and_a_tag_target_and_dates_do_not_rise() -> None:
    sections = _sections()
    dates: list[str] = []
    for release in RELEASES:
        heading = sections[release].splitlines()[0]
        match = _SECTION_HEADING.match(heading)
        assert match and match.group("date"), f"{release}: brak daty w nagłówku"
        dates.append(match.group("date"))
        assert _TAG_TARGET.search(sections[release]), (
            f"{release}: brak linii ``Tag target: `<sha>` `` — właściciel taguje ten commit"
        )

    assert dates == sorted(dates, reverse=True), dates
    assert "Tag target" not in sections["Unreleased"]


def test_each_release_says_where_the_history_differs_from_the_plan() -> None:
    sections = _sections()

    for release in RELEASES:
        assert "\n### Divergences\n" in sections[release], f"{release}: brak sekcji Divergences"


def test_every_roadmap_item_sits_in_one_release_with_the_decision_log_commit() -> None:
    decisions = _decisions_roadmap_rows()
    sections = _sections()
    assert set(MILESTONE_ITEMS) <= set(decisions), "docs/decisions.md nie ma całej tabeli R-1…R-27"

    for item in MILESTONE_ITEMS:
        rows = [
            (release, match.group("sha"))
            for release in RELEASES
            for match in _ITEM_ROW.finditer(sections[release])
            if match.group("id") == item
        ]
        assert len(rows) == 1, f"{item}: wiersz w {[release for release, _ in rows]}, ma być jeden"
        assert rows[0][1] == decisions[item], (
            f"{item}: CHANGELOG podaje {rows[0][1]}, docs/decisions.md {decisions[item]}"
        )


def test_changelog_is_written_in_english() -> None:
    polish = sorted(set(re.findall(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]", _text())))

    assert polish == [], f"polskie znaki w CHANGELOG.md (jest po angielsku): {polish}"
