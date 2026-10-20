"""Strażnik twierdzeń README: teza → link → dowód (DOC-05, DOC-12).

README to pierwszy kontakt recenzenta. Trasa „Engineering evidence” ma linkować
pliki, które tezę dowodzą, a sekcja „What this does NOT prove” — uczciwe
docstringi, które ograniczają claim (commit-reveal to tamper-evidence, nie
znacznik czasu; metryki karty modelu dotyczą kandydata; koszt to szacunek).
Linkujemy pliki, nie numery linii: link-check zdejmuje sufiks `:NNN`, więc link
do linii zgniłby po cichu, gdy kod się przesunie.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
_LINK_TARGET = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")

ENGINEERING_EVIDENCE = (
    "tests/test_architecture.py",
    "tests/infrastructure/test_migrations.py",
    "src/infrastructure/adapters/xgboost_local.py",
    "src/application/ports.py",
    "src/infrastructure/adapters/edgar_insider.py",
    "tests/application/test_report_builder.py",
    "tests/application/test_report_public_scorecard.py",
    "tests/test_markdown_links.py",
)
HONEST_DOCSTRINGS = (
    "src/domain/attestation.py",
    "src/domain/model_scorecard.py",
    "src/application/report_finops.py",
)


def _sections(text: str) -> dict[str, str]:
    """Nagłówek `## …` → treść sekcji do następnego `## `."""
    sections: dict[str, str] = {}
    for chunk in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = chunk.partition("\n")
        sections[heading.strip()] = body
    return sections


def _section(prefix: str) -> str:
    matches = [
        body for heading, body in _sections(README.read_text(encoding="utf-8")).items()
        if heading.startswith(prefix)
    ]
    assert len(matches) == 1, f"README ma mieć dokładnie jedną sekcję „{prefix}…”"
    return matches[0]


def _links(markdown: str) -> set[str]:
    return set(_LINK_TARGET.findall(markdown))


def test_link_helper_reads_targets_without_anchors() -> None:
    assert _links("[a](docs/x.md#part) [b](src/a.py) [c](https://e.com)") == {
        "docs/x.md",
        "src/a.py",
        "https://e.com",
    }


def test_readme_links_attestation_docstring() -> None:
    assert "src/domain/attestation.py" in _links(README.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", HONEST_DOCSTRINGS)
def test_known_limitations_link_the_honest_docstrings(path: str) -> None:
    assert path in _links(_section("What this does NOT prove"))


@pytest.mark.parametrize("path", ENGINEERING_EVIDENCE)
def test_readme_links_engineering_evidence(path: str) -> None:
    assert path in _links(_section("Engineering evidence"))
    assert (ROOT / path).is_file()


def test_readme_links_files_not_line_numbers() -> None:
    line_links = sorted(
        target
        for target in _links(README.read_text(encoding="utf-8"))
        if not target.startswith("http") and re.search(r":\d+$", target)
    )

    assert line_links == []
