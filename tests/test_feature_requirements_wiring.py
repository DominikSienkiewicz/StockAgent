"""NIEZ-30: mapa `FLAG_REQUIRED_MIGRATION` ↔ tabela flag ↔ pola `Settings` ↔ pliki migracji.

Mapa nie ma własnego źródła prawdy: kolumna „Migration” tabeli flag w
`docs/configuration.md` jest tym, co pilnuje `tests/test_docs_consistency.py`
(istnieje plik, zgadza się status). Ten test sprawdza, że mapa dla tej samej flagi
niesie najnowszą z wymienionych migracji — nowa flaga z migracją bez wpisu w mapie
wróciłaby do PGRST204 zamiast do wyłączenia na starcie.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from src.config import Settings
from src.domain.feature_requirements import FLAG_REQUIRED_MIGRATION

ROOT = Path(__file__).parent.parent
_FLAG_ROW = re.compile(r"^\| `(?P<flag>\w+)` \|")
_MIGRATION = re.compile(r"`(\d{3})`")


def _documented_migrations() -> dict[str, int]:
    """Flaga → najnowsza migracja z kolumny „Migration” (indeks 3 wiersza tabeli flag)."""
    documented: dict[str, int] = {}
    for line in (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8").splitlines():
        row = _FLAG_ROW.match(line)
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if row is None or len(cells) != 7:
            continue
        numbers = [int(number) for number in _MIGRATION.findall(cells[3])]
        if numbers:
            documented[row["flag"]] = max(numbers)
    return documented


def _is_boolean_field(name: str) -> bool:
    annotation = Settings.model_fields[name].annotation
    return bool in (get_args(annotation) or (annotation,))


def test_the_map_equals_the_migration_column_of_the_flag_table() -> None:
    assert dict(FLAG_REQUIRED_MIGRATION) == _documented_migrations()


def test_every_mapped_flag_is_a_boolean_setting() -> None:
    assert [name for name in FLAG_REQUIRED_MIGRATION if name not in Settings.model_fields] == []
    assert [name for name in FLAG_REQUIRED_MIGRATION if not _is_boolean_field(name)] == []


def test_every_mapped_migration_is_a_file_in_the_migrations_directory() -> None:
    numbers = {
        int(path.name[:3]) for path in (ROOT / "supabase" / "migrations").glob("*.sql")
    }

    assert sorted(set(FLAG_REQUIRED_MIGRATION.values()) - numbers) == []
