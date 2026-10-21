"""DOC-18: liczby repo cytowane w docs są odtwarzalne i aktualne.

`running-locally.md` mówiło „640+ passing tests”, gdy funkcji `def test_` było
kilka razy więcej. Liczby w docs piszemy jako dolną granicę „N+”; strażnik
liczy je na żywo (`main_tools.collect_repo_stats`) i pada, gdy cytat przekracza
stan repo albo zestarzał się o więcej niż 25%.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import main_tools
from src.tools.repo_stats import RepoStats, compute_repo_stats

ROOT = Path(__file__).resolve().parents[2]
DOCS = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
# „N+” starzeje się, gdy stan repo urośnie o więcej niż tyle względem cytatu.
STALE_GROWTH = 1.25

_CITED = re.compile(
    r"(\d{1,3}(?:,\d{3})+|\d+)\+ "
    r"(passing tests|tests|test functions|migrations|ports|adapter modules|domain modules)\b"
)
_PHRASE_TO_STAT = {
    "passing tests": "test_functions",
    "tests": "test_functions",
    "test functions": "test_functions",
    "migrations": "migrations",
    "ports": "ports",
    "adapter modules": "adapter_modules",
    "domain modules": "domain_modules",
}


@pytest.fixture(scope="module")
def live() -> RepoStats:
    return main_tools.collect_repo_stats(ROOT)


def _citations() -> list[tuple[str, int, str]]:
    return [
        (path.relative_to(ROOT).as_posix(), int(match.group(1).replace(",", "")), match.group(2))
        for path in DOCS
        for match in _CITED.finditer(path.read_text(encoding="utf-8"))
    ]


def _is_current(cited: int, actual: int) -> bool:
    return cited <= actual <= cited * STALE_GROWTH


def test_docs_quote_current_test_count(live: RepoStats) -> None:
    cited = [(doc, number) for doc, number, phrase in _citations() if "test" in phrase]

    assert cited, "docs nie cytują liczby testów — strażnik byłby pusty"
    stale = [(doc, number) for doc, number in cited if not _is_current(number, live.test_functions)]
    assert stale == [], f"liczba funkcji testowych to {live.test_functions}: {stale}"


def test_every_cited_repo_number_is_current(live: RepoStats) -> None:
    stale = [
        (doc, number, phrase)
        for doc, number, phrase in _citations()
        if not _is_current(number, getattr(live, _PHRASE_TO_STAT[phrase]))
    ]

    assert stale == [], f"nieaktualne liczby w docs (stan: {live}): {stale}"


def test_committed_repo_stats_json_is_current(live: RepoStats) -> None:
    recorded = json.loads((ROOT / "docs" / "repo-stats.json").read_text(encoding="utf-8"))

    assert set(recorded) == set(json.loads(live.to_json()))
    assert recorded["domain_stdlib_only"] is live.domain_stdlib_only
    stale = {
        key: (value, getattr(live, key))
        for key, value in recorded.items()
        if not isinstance(value, bool) and not _is_current(value, getattr(live, key))
    }
    assert stale == {}, "odśwież: uv run python main_tools.py repo-stats"


def test_counts_come_from_the_sources_not_from_imports() -> None:
    sources = {
        "tests/test_a.py": (
            "def test_one():\n    pass\n\n"
            "class TestX:\n    def test_two(self):\n        pass\n\n"
            "def helper():\n    pass\n"
        ),
        "src/application/ports.py": (
            "from abc import ABC\n\nclass APort(ABC):\n    pass\n\n"
            "class NotAPort:\n    pass\n"
        ),
        "src/infrastructure/adapters/a.py": "",
        "src/infrastructure/adapters/_http.py": "",
        "src/infrastructure/adapters/__init__.py": "",
        "src/infrastructure/llm/b.py": "",
        "src/domain/__init__.py": "",
        "src/domain/asset.py": "from __future__ import annotations\nimport math\n"
        "from src.domain.value_objects import Money\n",
    }
    migrations = ["001_init.sql", "002_x.sql", "README.md", "3_bad.sql"]

    assert compute_repo_stats(sources, migrations) == RepoStats(
        test_functions=2,
        migrations=2,
        ports=1,
        adapter_modules=2,
        domain_modules=1,
        domain_stdlib_only=True,
    )


def test_a_third_party_import_in_the_domain_is_reported() -> None:
    sources = {"src/domain/asset.py": "import requests\n"}

    assert compute_repo_stats(sources, []).domain_stdlib_only is False


def test_json_is_deterministic() -> None:
    stats = RepoStats(1, 2, 3, 4, 5, True)

    assert stats.to_json() == stats.to_json()
    assert json.loads(stats.to_json())["test_functions"] == 1
