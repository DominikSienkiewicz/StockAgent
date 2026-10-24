"""Sondy migracji w `scripts/prod-state-check.sh` muszą nadążać za katalogiem (DOC-51).

Skrypt odpowiada właścicielowi na pytanie „które migracje są na produkcji”, zanim
włączy flagi, które ich wymagają. Lista sond kończyła się na 025, więc o 026–065
nie mówił nic. Test porównuje ją z `supabase/migrations/`: każda migracja od
pierwszej sondowanej ma dokładnie jedną sondę, a obiekt sondy (tabela, kolumna,
funkcja) naprawdę powstaje w pliku tej migracji. Migracje, których OpenAPI
PostgREST nie pokaże (sama definicja widoku, indeks, ograniczenie, RLS), mają
wiersz `opaque` — skrypt opisuje je w wyniku jako nie do sprawdzenia, zamiast
milczeć.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "prod-state-check.sh"
MIGRATIONS = ROOT / "supabase" / "migrations"

KINDS = {"table", "column", "function", "opaque"}
OPAQUE_REASONS = {"view-definition", "index", "constraint", "rls"}
FIRST_PROBED = 17


def _probes() -> list[tuple[int, str, str, str]]:
    text = SCRIPT.read_text(encoding="utf-8")
    body = re.search(r'MIGRATION_PROBES="(.*?)"\n', text, re.S)
    assert body is not None, "skrypt nie ma listy MIGRATION_PROBES"
    rows = []
    for line in body.group(1).splitlines():
        number, kind, name, column = line.split()
        rows.append((int(number), kind, name, column))
    return rows


def _migration_files() -> dict[int, Path]:
    return {int(path.name[:3]): path for path in sorted(MIGRATIONS.glob("*.sql"))}


def _sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_every_migration_from_the_first_probe_has_exactly_one_probe() -> None:
    probed = [number for number, *_ in _probes()]
    expected = [number for number in _migration_files() if number >= FIRST_PROBED]

    assert probed == expected, (
        f"sondy {sorted(set(probed) ^ set(expected))} rozjechane z katalogiem migracji"
    )


def test_probe_rows_use_known_kinds_and_reasons() -> None:
    problems = [
        (number, kind, column)
        for number, kind, _name, column in _probes()
        if kind not in KINDS or (kind == "opaque" and column not in OPAQUE_REASONS)
    ]

    assert problems == []


@pytest.mark.parametrize(
    "number, kind, name, column",
    _probes(),
    ids=[f"{number:03d}-{kind}-{name}" for number, kind, name, _ in _probes()],
)
def test_probed_object_is_defined_by_its_migration(
    number: int, kind: str, name: str, column: str
) -> None:
    sql = _sql(_migration_files()[number])
    flat = re.sub(r"\s+", " ", sql)

    if kind == "table":
        assert re.search(rf"CREATE TABLE (IF NOT EXISTS )?{name}\b", flat, re.I), (
            f"{number}: tabela {name} nie powstaje w tej migracji"
        )
    elif kind == "column":
        assert re.search(rf"\b{column}\b", flat), f"{number}: brak kolumny {column}"
        assert re.search(rf"\b{name}\b", flat), f"{number}: brak tabeli {name}"
        assert re.search(rf"ADD COLUMN (IF NOT EXISTS )?{column}\b", flat, re.I) or re.search(
            rf"AS {column}\b", flat, re.I
        ), f"{number}: kolumna {column} nie jest dodawana"
    elif kind == "function":
        assert re.search(rf"CREATE (OR REPLACE )?FUNCTION {name}\b", flat, re.I), (
            f"{number}: funkcja {name} nie powstaje w tej migracji"
        )
    else:
        spellings = {name, name.replace("_", " ")}
        assert any(spelling.casefold() in flat.casefold() for spelling in spellings), (
            f"{number}: obiekt {name} nie występuje w migracji"
        )


def test_script_describes_objects_openapi_cannot_show() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "opaque)" in text
    assert "OpenAPI nie pokazuje" in text


def test_script_heading_and_header_follow_the_probe_range() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    last = f"{max(_migration_files()):03d}"

    assert "obecność migracji 017–025" not in text, "nagłówek skryptu opisuje stary zakres"
    assert f"017–{last}" in text or "$PROBE_RANGE" in text
    assert "### Migracje 017–025" not in text


def test_running_locally_describes_the_current_probe_range() -> None:
    text = (ROOT / "docs" / "running-locally.md").read_text(encoding="utf-8")

    assert "migrations 017–025" not in text
    assert "migrations from 017" in text


def test_running_locally_lists_exactly_the_opaque_migrations() -> None:
    text = (ROOT / "docs" / "running-locally.md").read_text(encoding="utf-8")
    opaque = [f"{number:03d}" for number, kind, *_ in _probes() if kind == "opaque"]
    listed = f"({', '.join(opaque[:-1])} and {opaque[-1]})"

    assert listed in text, f"docs nie wymieniają migracji bez śladu w OpenAPI: {listed}"
