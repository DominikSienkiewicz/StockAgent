"""Tabela relacji `docs/data-and-access.md` nadąża za migracjami (DOC-37).

Dokument modelu danych jest ręczny i zdryfuje tak, jak lista migracji w
running-locally kończąca się kiedyś na 018. Bez Dockera: relacje (tabele,
widoki, widoki zmaterializowane) wyliczamy z tekstu `supabase/migrations/*.sql`
w kolejności aplikowania — `CREATE` dodaje, `DROP` zdejmuje — i porównujemy z
pierwszą kolumną tabeli w sekcji `## Relations`.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
MIGRATIONS_DIR = ROOT / "supabase" / "migrations"
DATA_AND_ACCESS = ROOT / "docs" / "data-and-access.md"

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_RELATION_NAME = r"(?:public\.)?\"?(?P<name>\w+)\"?"
_CREATE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:UNLOGGED\s+)?(?:MATERIALIZED\s+)?(?:TABLE|VIEW)\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?" + _RELATION_NAME,
    re.IGNORECASE,
)
_DROP = re.compile(
    r"\bDROP\s+(?:MATERIALIZED\s+)?(?:TABLE|VIEW)\s+(?:IF\s+EXISTS\s+)?" + _RELATION_NAME,
    re.IGNORECASE,
)
_RELATION_ROW = re.compile(r"^\| `(?P<name>\w+)` \|", re.MULTILINE)


def relations_after(sql_scripts: list[str]) -> set[str]:
    """Relacje, które zostają po zaaplikowaniu skryptów w podanej kolejności."""
    relations: set[str] = set()
    for script in sql_scripts:
        code = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", script))
        events = sorted(
            [(m.start(), True, m["name"]) for m in _CREATE.finditer(code)]
            + [(m.start(), False, m["name"]) for m in _DROP.finditer(code)]
        )
        for _, created, name in events:
            if created:
                relations.add(name.lower())
            else:
                relations.discard(name.lower())
    return relations


def documented_relations(markdown: str) -> set[str]:
    """Pierwsza kolumna tabeli w sekcji `## Relations` (do następnego `## `)."""
    section = re.split(r"^## Relations$", markdown, maxsplit=1, flags=re.MULTILINE)[1]
    section = re.split(r"^## ", section, maxsplit=1, flags=re.MULTILINE)[0]
    table = section.split("\n| Function |", 1)[0]
    return {m["name"] for m in _RELATION_ROW.finditer(table)}


def test_extractor_follows_create_and_drop_and_skips_comments() -> None:
    scripts = [
        "-- CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS.\n"
        "CREATE TABLE IF NOT EXISTS public.alpha (id int);\n"
        "/* CREATE TABLE ghost (id int); */\n"
        "create materialized view beta AS SELECT 1;",
        "DROP MATERIALIZED VIEW IF EXISTS beta;\nCREATE MATERIALIZED VIEW beta AS SELECT 2;\n"
        "CREATE OR REPLACE VIEW gamma AS SELECT 1;\nDROP TABLE alpha;",
    ]

    assert relations_after(scripts) == {"beta", "gamma"}


def test_extractor_reads_only_the_relations_table() -> None:
    markdown = (
        "## Relations\n\n| Relation | Migrations |\n|---|---|\n| `alpha` | `001` |\n\n"
        "| Function | Migration |\n|---|---|\n| `refresh()` | `001` |\n\n"
        "## Keys\n\n| `beta` | x |\n"
    )

    assert documented_relations(markdown) == {"alpha"}


def test_every_migrated_relation_has_a_row_in_data_and_access() -> None:
    scripts = [path.read_text(encoding="utf-8") for path in sorted(MIGRATIONS_DIR.glob("*.sql"))]
    migrated = relations_after(scripts)
    documented = documented_relations(DATA_AND_ACCESS.read_text(encoding="utf-8"))

    assert sorted(migrated - documented) == [], "relacja z migracji bez wiersza w docs"
    assert sorted(documented - migrated) == [], "wiersz w docs bez relacji w migracjach"
