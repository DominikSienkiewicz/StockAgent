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
        [--experiment ID]
    uv run python main_tools.py cost-envelope
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.tools.cost_envelope import EnvelopeInputs, cost_envelope, render_cost_envelope
from src.tools.evaluate import (
    GATE_DELTA_FIELD,
    OPTIONAL_COLUMNS,
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
    from src.application.ports import PositionsImportPort
    from src.config import Settings
    from src.domain.experiment import Experiment
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

Handler = Callable[[argparse.Namespace], int]


def _now() -> datetime:
    """Zegar narzędzi (UTC) — jedno miejsce do podmiany w testach."""
    return datetime.now(UTC)


def _load_settings() -> Settings:
    """`Settings` z env plus logi z redakcją jego sekretów (BEZ-02). Poziom
    WARNING jak dotąd: narzędzie pisze wynik na stdout, logi tylko przy kłopocie."""
    from src.config import Settings
    from src.infrastructure.logging_redaction import configure_logging

    settings = Settings.from_env()
    configure_logging(settings.secret_values(), level=logging.WARNING)
    return settings


def build_parser() -> argparse.ArgumentParser:
    """Parser z podkomendą na narzędzie; każda ustawia `handler` w defaults."""
    parser = argparse.ArgumentParser(
        prog="main_tools.py",
        description="StockAgent one-off tools (composition root, ARCH-39).",
    )
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    _add_evaluate(subcommands)
    _add_import_positions(subcommands)
    _add_cost_envelope(subcommands)
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
    parser.add_argument(
        "--experiment",
        metavar="ID",
        help=(
            "Porównanie ramion eksperymentu z config.toml (EXP-05): różnica metryk "
            "z CI blokowym; ramiona z kolumny (migracja 037) albo z kwitu v2."
        ),
    )
    parser.set_defaults(handler=_run_evaluate)


def _run_evaluate(args: argparse.Namespace) -> int:
    """Ewaluacja offline (read-only, zero płatnych wywołań); exit 0 zawsze."""
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

    segmentations = tuple(dict.fromkeys(args.by))
    settings = _load_settings()
    experiment: Experiment | None = None
    if args.experiment is not None:
        experiment = _find_experiment(settings.experiment_definitions(), args.experiment)
        if experiment is None:
            return 1
    repo = SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key)
    reads, fields = evaluation_reads(
        by_recipe=args.by_recipe,
        by_universe=args.by_universe,
        segmentations=segmentations,
        echo=args.echo,
        experiment=experiment is not None,
    )
    rows, read_fields = _read_evaluation_rows(repo, args.days, reads, fields)
    if "asset_class" in segmentations:
        rows = with_asset_class(rows, settings.crypto_symbols, settings.symbols_etf)
    print(
        render_evaluation(
            rows,
            days=args.days,
            segmentations=segmentations,
            by_recipe=args.by_recipe,
            by_universe=args.by_universe,
            echo=GATE_DELTA_FIELD in read_fields,
            experiment=experiment,
        )
    )
    return 0


def _find_experiment(experiments: Sequence[Experiment], experiment_id: str) -> Experiment | None:
    """Definicja eksperymentu z configu; nieznane id → komunikat na stderr i None."""
    for experiment in experiments:
        if experiment.id == experiment_id:
            return experiment
    known = ", ".join(experiment.id for experiment in experiments) or "brak"
    print(
        f"❌ nieznany eksperyment {experiment_id} — zdefiniowane w config.toml: {known}",
        file=sys.stderr,
    )
    return None


def _read_evaluation_rows(
    repo: SupabaseRepository, days: int, reads: dict[str, Any], fields: set[str]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Odczyt wierszy ewaluacji i zbiór pól, które naprawdę odczytano.

    Bez migracji kolumny opcjonalnej (`gate_delta` 034, `experiment_arms` 037)
    reszta raportu ma zostać: komunikat zamiast wyjątku i ponowny odczyt bez
    tej kolumny (test echa pomijamy wtedy jawnie, ramiona idą z kwitu v2)."""
    from src.infrastructure.adapters.supabase_repo import MissingColumnError

    remaining = set(fields)
    while True:
        try:
            rows = repo.get_resolved_predictions_for_eval(
                days=days, **reads, **fields_kwarg(remaining)
            )
        except MissingColumnError as exc:
            if exc.column not in OPTIONAL_COLUMNS or exc.column not in remaining:
                raise
            print(OPTIONAL_COLUMNS[exc.column])
            remaining.discard(exc.column)
            continue
        return rows, remaining


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

    settings = _load_settings()
    currency_columns = settings.positions_currency_enabled
    adapter = _positions_import_port(settings)
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
    if not settings.db_writes_enabled:
        print(
            "❌ --apply odrzucony: DB_WRITES_ENABLED=false (run tylko do odczytu) — "
            "nic nie zapisano w positions. Tabela jest we wspólnej bazie produkcyjnej; "
            "żeby ją zastąpić, uruchom świadomie z DB_WRITES_ENABLED=true.",
            file=sys.stderr,
        )
        return 1
    adapter.replace_positions(incoming, as_of=now)
    print(f"✅ zapisano {len(incoming)} lot(ów) w positions, as_of {now.isoformat()}")
    return 0


def _positions_import_port(settings: Settings) -> PositionsImportPort:
    """Adapter importu pozycji; bez `db_writes_enabled` tylko do odczytu (BEZ-08)."""
    from src.infrastructure.adapters.positions_repo import SupabasePositionsAdapter
    from src.infrastructure.adapters.write_guard import ReadOnlyPositionsImport

    adapter = SupabasePositionsAdapter(
        url=settings.supabase_url,
        key=settings.supabase_key,
        currency_columns=settings.positions_currency_enabled,
    )
    if settings.db_writes_enabled:
        return adapter
    return ReadOnlyPositionsImport(adapter)


def _add_cost_envelope(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subcommands.add_parser(
        "cost-envelope",
        help="Print the estimated min/max paid calls and cost of one Fast Loop cycle.",
        description=(
            "FIN-06: worst case = every eligible symbol of config.toml past the volatility "
            "gate; minimum = every symbol below it. Offline, zero paid calls."
        ),
    )
    parser.set_defaults(handler=_run_cost_envelope)


def _run_cost_envelope(args: argparse.Namespace) -> int:
    """FIN-06: koperta kosztu cyklu dla bieżącej konfiguracji (offline); exit 0."""
    from src.infrastructure.adapters.advisory_council import COUNCIL_MAX_PERSONAS
    from src.infrastructure.adapters.alpha_vantage_client import supported_tickers
    from src.infrastructure.persona_loader import load_council_personas

    settings = _load_settings()
    watchlist = [*settings.symbols, *settings.crypto_symbols]
    # Jak `main_agent.main`: bez ceny symbol nie wchodzi do grafu, a feed AV
    # (klient leniwy, FIN-04) pyta tylko o symbole za bramką, które AV przyjmuje.
    unsupported = set(settings.symbols_unsupported_price)
    personas = load_council_personas(Path(settings.council_personas_dir))
    inputs = EnvelopeInputs(
        n_gated=sum(1 for symbol in watchlist if symbol not in unsupported),
        n_av_symbols=len(supported_tickers(watchlist)),
        personas=min(len(personas), COUNCIL_MAX_PERSONAS),
        debate=settings.debate_mode_enabled,
        tool_rounds=settings.tool_use_max_rounds if settings.tool_use_enabled else 0,
    )
    print(render_cost_envelope(cost_envelope(inputs)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
