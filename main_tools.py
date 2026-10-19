"""Composition root narzędzi jednorazowych (ARCH-39).

Jedyne miejsce, gdzie narzędzie CLI dostaje `Settings` i konkretne adaptery.
Czysta logika (parsowanie, walidacja, diff, metryki, formatowanie) żyje
w `src/tools/` i nie składa adapterów — tutaj tylko łączymy zależności i wołamy
czyste funkcje. Każde narzędzie to jedna podkomenda argparse. Osobny
`main_<x>.py` dostaje wyłącznie entry-point z własnym workflow (Fast Loop,
Slow Loop, watch, deadman, backtest).

Użycie:
    uv run python main_tools.py --help
    uv run python main_tools.py <podkomenda> [opcje]
    uv run python main_tools.py import-positions --csv pozycje.csv [--apply]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.tools.import_positions import (
    PositionsCsvError,
    format_plan,
    parse_positions_csv,
    plan_import,
)

Handler = Callable[[argparse.Namespace], int]


def _now() -> datetime:
    """Zegar narzędzi (UTC) — jedno miejsce do podmiany w testach."""
    return datetime.now(UTC)


def build_parser() -> argparse.ArgumentParser:
    """Parser z podkomendą na narzędzie; każda ustawia `handler` w defaults."""
    parser = argparse.ArgumentParser(
        prog="main_tools.py",
        description="StockAgent one-off tools (composition root, ARCH-39).",
    )
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    _add_import_positions(subcommands)
    return parser


def _add_import_positions(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subcommands.add_parser(
        "import-positions",
        help="Replace the positions table with a CSV file (dry-run unless --apply).",
        description=(
            "PORT-18: CSV columns symbol,quantity,avg_cost,currency[,purchase_date]. "
            "Without --apply prints the diff against the table and writes nothing."
        ),
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to the CSV file.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace the whole positions table (as_of = now).",
    )
    parser.set_defaults(handler=_run_import_positions)


def _run_import_positions(args: argparse.Namespace) -> int:
    """PORT-18: plik → walidacja → diff z tabelą → (z `--apply`) zamiana tabeli.

    Plik walidujemy przed dotknięciem bazy. Odczyt bieżącej tabeli jest ścisły,
    bo dry-run na pustej liście pokazałby „wszystko nowe”."""
    path: Path = args.csv
    now = _now()
    try:
        incoming = parse_positions_csv(path.read_text(encoding="utf-8-sig"), today=now.date())
    except OSError as exc:
        print(f"❌ nie da się odczytać {path}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    except PositionsCsvError as exc:
        print(f"❌ {path}: plik nie przeszedł walidacji:", file=sys.stderr)
        for error in exc.errors:
            print(f"   · {error}", file=sys.stderr)
        return 1

    from src.config import Settings
    from src.infrastructure.adapters.positions_repo import SupabasePositionsAdapter

    settings = Settings.from_env()
    currency_columns = settings.positions_currency_enabled
    adapter = SupabasePositionsAdapter(
        url=settings.supabase_url,
        key=settings.supabase_key,
        currency_columns=currency_columns,
    )
    plan = plan_import(adapter.current_positions(), incoming, currency_columns=currency_columns)
    print(format_plan(plan, file_lots=len(incoming)))
    if not args.apply:
        print(
            "To był dry-run — nic nie zapisano. "
            "--apply zastąpi tabelę positions (as_of = teraz)."
        )
        return 0
    if plan.blockers:
        for blocker in plan.blockers:
            print(f"❌ --apply odrzucony: {blocker}", file=sys.stderr)
        return 1
    adapter.replace_positions(incoming, as_of=now)
    print(f"✅ zapisano {len(incoming)} lot(ów) w positions, as_of {now.isoformat()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
