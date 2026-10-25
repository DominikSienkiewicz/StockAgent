"""MAIL-22: docs/report-email.md opisuje mail w kolejności rejestru sekcji.

Tabela „Sections in render order” ma jeden wiersz na każdy wpis `REPORT_SECTIONS`
(kolumna `Key` = `ReportSection.key`), w tej samej kolejności — dopisanie,
usunięcie albo przestawienie sekcji bez poprawki docs to czerwony test (G63).
"""

from __future__ import annotations

import re
from pathlib import Path

from src.application.reporting.registry import REPORT_SECTIONS

ROOT = Path(__file__).parent.parent
REPORT_DOC = ROOT / "docs" / "report-email.md"
_TABLE_HEADING = "## Sections in render order"
_KEY_CELL = re.compile(r"^\|\s*\d+\s*\|\s*`([a-z_]+)`\s*\|")


def documented_section_keys(markdown: str) -> list[str]:
    """Klucze z kolumny `Key` tabeli pod nagłówkiem „Sections in render order”."""
    _, _, after = markdown.partition(_TABLE_HEADING)
    table_and_rest = after.split("\n## ", 1)[0]
    return [
        match.group(1)
        for line in table_and_rest.splitlines()
        if (match := _KEY_CELL.match(line))
    ]


def test_extractor_reads_the_key_column() -> None:
    sample = (
        f"{_TABLE_HEADING}\n\n| # | Key | Section |\n|---|---|---|\n"
        "| 1 | `preheader` | Hidden preview |\n| 2 | `masthead` | Header |\n\n## Next\n"
        "| 3 | `ignored` | outside the table |\n"
    )

    assert documented_section_keys(sample) == ["preheader", "masthead"]


def test_report_email_doc_mentions_every_slot_section() -> None:
    documented = documented_section_keys(REPORT_DOC.read_text(encoding="utf-8"))

    assert documented == [section.key for section in REPORT_SECTIONS]
