"""Strażnik linków względnych w Markdown (CI-32) — w lokalnej bramce, nie tylko w CI.

README jest routerem do `docs/`, więc link, który się nie rozwiązuje, to publiczny
defekt: wystarczy commit routera, którego dokumenty docelowe `.gitignore` po cichu
wyciął, a 404 wiszą, dopóki ktoś ich nie kliknie. Dawniej sprawdzał to tylko krok
basha w `ci.yml`, więc agent zgłaszał „zielono” według bramki z AGENTS.md, a CI
padało na linkach. Logika kroku przeszła tutaj bez zmian:

- pliki: `git ls-files '*.md'` (gitignorowane `docs/concepts/` nie wchodzą);
- cel: każde `](…)` w linii, bez kotwicy `#…`; linki zewnętrzne (`http…`,
  `mailto:`) pomijamy, bo wymagałyby sieci i padałyby na cudzej awarii;
- sufiks `:<cyfry>` (np. `src/a.py:190`) wskazuje linię, nie plik o nazwie
  „a.py:190”, więc zdejmujemy go przed sprawdzeniem istnienia.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).parent.parent
# Jak `grep -oE '\]\([^)#][^)]*\)'` z dawnego kroku CI, linia po linii.
_LINK_TARGET = re.compile(r"\]\(([^)#][^)]*)\)")
_EXTERNAL_PREFIXES = ("http", "mailto:")
_DIGITS = frozenset("0123456789")


def _strip_line_suffix(target: str) -> str:
    """Najkrótszy sufiks `:<cyfra>…` precz — jak `${target%:[0-9]*}` w bashu."""
    for index in range(len(target) - 1, -1, -1):
        if target[index] == ":" and target[index + 1 : index + 2] in _DIGITS:
            return target[:index]
    return target


def _link_targets(text: str) -> list[str]:
    targets: list[str] = []
    for line in text.splitlines():
        for raw in _LINK_TARGET.findall(line):
            target = raw.split("#", 1)[0]
            if not target or target.startswith(_EXTERNAL_PREFIXES):
                continue
            targets.append(_strip_line_suffix(target))
    return targets


def dead_links(root: Path, files: Iterable[str]) -> list[str]:
    """`<plik> -> <cel>` dla każdego linku względnego, którego cel nie istnieje.

    `files` to ścieżki względem `root`; cel liczymy od katalogu pliku."""
    dead: list[str] = []
    for file in files:
        directory = Path(file).parent
        text = (root / file).read_text(encoding="utf-8")
        dead.extend(
            f"{file} -> {target}"
            for target in _link_targets(text)
            if not (root / directory / target).exists()
        )
    return dead


def _tracked_markdown(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def test_dead_relative_link_is_reported(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[x](docs/missing.md) and [y](src/a.py:12)\n", encoding="utf-8"
    )

    assert dead_links(tmp_path, ["README.md"]) == ["README.md -> docs/missing.md"]


def test_external_anchor_and_nested_links_follow_the_ci_rules(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "b.md").write_text("b\n", encoding="utf-8")
    (docs / "a.md").write_text(
        "[web](https://example.com/x) [mail](mailto:a@example.com) [self](#top)\n"
        "[sibling](b.md#part) [up](../README.md) [gone](c.md:7-9)\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("root\n", encoding="utf-8")

    assert dead_links(tmp_path, ["docs/a.md"]) == ["docs/a.md -> c.md"]


def test_repository_markdown_links_resolve() -> None:
    files = _tracked_markdown(ROOT)

    # Strażnik pustego wyniku: bez gita (albo ze złym wzorcem) lista byłaby pusta.
    assert "README.md" in files
    assert dead_links(ROOT, files) == []
