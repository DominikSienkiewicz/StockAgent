"""Strażnik odwołań do dokumentacji (DOC-14): każda ścieżka `docs/….md` wpisana
w kod, skrypty, migracje, workflowy albo README musi istnieć.

Docstring `SupabaseRepository` odsyłał po schemat bazy do pliku w `docs/`, którego
nigdy nie było, więc czytelnik szukał modelu danych w próżni. `docs/concepts/`
jest lokalne i gitignorowane — odwołań do niego nie sprawdzamy.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

SOURCE_GLOBS = (
    "src/**/*.py",
    "main_*.py",
    "scripts/*",
    "supabase/migrations/*.sql",
    ".github/workflows/*.yml",
    "config.toml",
    "README.md",
)
DOCS_PATH = re.compile(r"\bdocs/[A-Za-z0-9_./-]+\.md\b")
LOCAL_ONLY_PREFIX = "docs/concepts/"


def _referenced_doc_paths() -> dict[str, set[str]]:
    """Ścieżka dokumentu → pliki źródłowe, które ją wymieniają."""
    referenced: dict[str, set[str]] = {}
    for pattern in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            for doc in DOCS_PATH.findall(path.read_text(encoding="utf-8")):
                if not doc.startswith(LOCAL_ONLY_PREFIX):
                    referenced.setdefault(doc, set()).add(str(path.relative_to(ROOT)))
    return referenced


def test_docs_paths_referenced_in_source_exist() -> None:
    referenced = _referenced_doc_paths()
    missing = {
        doc: sorted(sources) for doc, sources in referenced.items() if not (ROOT / doc).is_file()
    }

    # Strażnik pustego wyniku: README zawsze linkuje dokumenty, więc zero trafień
    # znaczyłoby zepsuty wzorzec, a nie czyste repo.
    assert "docs/how-it-works.md" in referenced
    assert missing == {}
