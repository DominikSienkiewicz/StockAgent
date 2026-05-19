"""CLI walidator dla plików JSON z personami rady doradczej.

Uruchom:
    uv run python -m src.tools.validate_personas
    uv run python -m src.tools.validate_personas --dir custom/path

Exit code:
    0 — wszystkie pliki poprawne
    1 — błąd: brak katalogu, malformed JSON, brak pola, duplikat name itp.

Używaj przed commitem (pre-commit hook w .pre-commit-config.yaml) lub po
edycji ręcznej, by złapać literówki ZANIM agent wystartuje na produkcji.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.infrastructure.persona_loader import (
    DEFAULT_PERSONAS_DIR,
    PersonaSchemaError,
    load_council_personas,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate council persona JSON files.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_PERSONAS_DIR,
        help=f"Directory with persona JSON files (default: {DEFAULT_PERSONAS_DIR})",
    )
    args = parser.parse_args(argv)

    try:
        personas = load_council_personas(args.dir)
    except FileNotFoundError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    except PersonaSchemaError as exc:
        print(f"❌ Schema error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    print(f"✅ {len(personas)} persona(s) valid in {args.dir}/")
    for p in personas:
        print(f"   · {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
