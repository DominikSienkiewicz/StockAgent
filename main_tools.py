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
    uv run python main_tools.py evaluate [--days N] [--by-recipe] [--by X] [--echo]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.tools.evaluate import (
    GATE_DELTA_FIELD,
    GATE_DELTA_UNAVAILABLE,
    SEGMENTATIONS,
    evaluation_reads,
    fields_kwarg,
    render_evaluation,
    with_asset_class,
)
from src.tools.import_positions import (
    PositionsCsvError,
    format_plan,
    parse_positions_csv,
    plan_import,
)

if TYPE_CHECKING:
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

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
    _add_evaluate(subcommands)
    _add_import_positions(subcommands)
    return parser


def _add_evaluate(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subcommands.add_parser(
        "evaluate",
        help="Evaluate prediction quality vs a naive baseline (read-only).",
        description="Evaluate StockAgent prediction quality vs a naive baseline.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Okno (dni) zamkniętych predykcji do ewaluacji (default: 30).",
    )
    parser.add_argument(
        "--by-recipe",
        action="store_true",
        help="Trafność per wersja przepisu (recipe_version) — wymaga migracji 030.",
    )
    parser.add_argument(
        "--by-universe",
        action="store_true",
        help="Trafność per wersja uniwersum symboli (z kwitu decyzyjnego v2).",
    )
    parser.add_argument(
        "--by",
        action="append",
        choices=SEGMENTATIONS,
        default=[],
        help=(
            "Ablacja komponentów albo segment kontekstu (WIA-55); można powtórzyć. "
            "Każda kolumna segmentu wymaga swojej migracji."
        ),
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Test echa momentum (EXP-01): trend LLM vs znak delty bramki — wymaga migracji 034.",
    )
    parser.set_defaults(handler=_run_evaluate)


def _run_evaluate(args: argparse.Namespace) -> int:
    """Ewaluacja offline (read-only, zero płatnych wywołań); exit 0 zawsze."""
    from src.config import Settings
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

    segmentations = tuple(dict.fromkeys(args.by))
    settings = Settings.from_env()
    repo = SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key)
    reads, fields = evaluation_reads(
        by_recipe=args.by_recipe,
        by_universe=args.by_universe,
        segmentations=segmentations,
        echo=args.echo,
    )
    rows, echo = _read_evaluation_rows(repo, args.days, reads, fields)
    if "asset_class" in segmentations:
        rows = with_asset_class(rows, settings.crypto_symbols, settings.symbols_etf)
    print(
        render_evaluation(
            rows,
            days=args.days,
            segmentations=segmentations,
            by_recipe=args.by_recipe,
            by_universe=args.by_universe,
            echo=echo,
        )
    )
    return 0


def _read_evaluation_rows(
    repo: SupabaseRepository, days: int, reads: dict[str, Any], fields: set[str]
) -> tuple[list[dict[str, Any]], bool]:
    """Odczyt wierszy ewaluacji i to, czy niosą `gate_delta`.

    Bez migracji 034 reszta raportu ma zostać: komunikat zamiast wyjątku
    i ponowny odczyt bez delty bramki (test echa wtedy pomijamy jawnie)."""
    from src.infrastructure.adapters.supabase_repo import MissingColumnError

    try:
        rows = repo.get_resolved_predictions_for_eval(days=days, **reads, **fields_kwarg(fields))
    except MissingColumnError as exc:
        if exc.column != GATE_DELTA_FIELD:
            raise
        print(GATE_DELTA_UNAVAILABLE)
        remaining = fields - {GATE_DELTA_FIELD}
        return (
            repo.get_resolved_predictions_for_eval(days=days, **reads, **fields_kwarg(remaining)),
            False,
        )
    return rows, GATE_DELTA_FIELD in fields


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
