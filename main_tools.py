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
        [--experiment ID] [--discrimination]
    uv run python main_tools.py cost-envelope
    uv run python main_tools.py render-sample-report [--out-dir build/sample-report]
    uv run python main_tools.py render-graph-diagram [--out-dir docs]
    uv run python main_tools.py repo-stats [--out docs/repo-stats.json]
    uv run python main_tools.py explain <prediction_id> | --symbol X --date RRRR-MM-DD
    uv run python main_tools.py persona-ablation [--days N] [--window-days N]
    uv run python main_tools.py council-ablation [--days N]
    uv run python main_tools.py verify-attestation [--commitments PATH] [--reveals PATH]
    uv run python main_tools.py reproduce-training [--sidecar PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from src.domain.council_ablation import leave_one_out
from src.tools.cost_envelope import (
    EnvelopeInputs,
    cost_envelope,
    render_cost_envelope,
    render_weekly_budget,
)
from src.tools.council_ablation import format_council_ablation
from src.tools.evaluate import (
    GATE_DELTA_FIELD,
    OPTIONAL_COLUMNS,
    SEGMENTATIONS,
    evaluation_reads,
    fields_kwarg,
    render_evaluation,
    render_evaluation_json,
    with_asset_class,
)
from src.tools.explain import explain_prediction, explain_symbol_day
from src.tools.import_positions import (
    PositionsCsvError,
    format_plan,
    parse_positions_csv,
    plan_import,
)
from src.tools.persona_ablation import format_ablation, replay_persona_weighting
from src.tools.render_graph_diagram import graph_diagrams
from src.tools.render_sample_report import (
    SAMPLE_HTML_NAME,
    SAMPLE_TEXT_NAME,
    render_sample_report,
)
from src.tools.repo_stats import REPO_STATS_NAME, RepoStats, compute_repo_stats
from src.tools.reproduce_training import (
    SidecarError,
    assess_reproduction,
    fingerprint_of,
    format_reproduction,
    parse_sidecar,
    rows_in_range,
)
from src.tools.verify_attestation import (
    DEFAULT_COMMITMENTS_PATH,
    DEFAULT_REVEALS_PATH,
    format_report,
    parse_jsonl,
    verify_ledger,
)

if TYPE_CHECKING:
    from src.application.ports import PositionsImportPort
    from src.config import Settings
    from src.domain.experiment import Experiment
    from src.domain.persona_track_record import PersonaVoteOutcome
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

Handler = Callable[[argparse.Namespace], int]

_REPO_ROOT = Path(__file__).resolve().parent


def _now() -> datetime:
    """Zegar narzędzi (UTC) — jedno miejsce do podmiany w testach."""
    return datetime.now(UTC)


def _load_settings() -> Settings:
    """`Settings` z env plus logi z redakcją jego sekretów (BEZ-02). Poziom
    WARNING jak dotąd: narzędzie pisze wynik na stdout, logi tylko przy kłopocie."""
    from src.config import Settings
    from src.infrastructure.logging_setup import configure_logging

    settings = Settings.from_env()
    configure_logging("main_tools", settings.secret_values(), level=logging.WARNING)
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
    _add_render_sample_report(subcommands)
    _add_render_graph_diagram(subcommands)
    _add_repo_stats(subcommands)
    _add_explain(subcommands)
    _add_persona_ablation(subcommands)
    _add_council_ablation(subcommands)
    _add_verify_attestation(subcommands)
    _add_reproduce_training(subcommands)
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
    parser.add_argument(
        "--discrimination",
        action="store_true",
        help=(
            "Czy pewność coś znaczy (WIA-38): AUC pewności LLM i konsensusu rady "
            "z CI blokowym i trafnością top-k% wg pewności."
        ),
    )
    parser.add_argument(
        "--scores",
        action="store_true",
        help=(
            "RMSE w zwrotach per klasa aktywa oraz Brier i log loss pewności LLM (WIA-40) — "
            "czyta symbol i confidence_score (migracja 016)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Raport jako JSON do porównywania przebiegów (WIA-40); tylko JSON na stdout.",
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
        discrimination=args.discrimination,
        scores=args.scores,
    )
    rows, read_fields = _read_evaluation_rows(
        repo, args.days, reads, fields, notices=sys.stderr if args.json else sys.stdout
    )
    if "asset_class" in segmentations or args.scores:
        rows = with_asset_class(rows, settings.crypto_symbols, settings.symbols_etf)
    if args.json:
        print(render_evaluation_json(rows, days=args.days, segmentations=segmentations))
        return 0
    print(
        render_evaluation(
            rows,
            days=args.days,
            segmentations=segmentations,
            by_recipe=args.by_recipe,
            by_universe=args.by_universe,
            echo=GATE_DELTA_FIELD in read_fields,
            experiment=experiment,
            discrimination=args.discrimination,
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
    repo: SupabaseRepository,
    days: int,
    reads: dict[str, Any],
    fields: set[str],
    *,
    notices: TextIO = sys.stdout,
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
            print(OPTIONAL_COLUMNS[exc.column], file=notices)
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
    adapter = _positions_import_port(settings)
    plan = plan_import(adapter.current_positions(), incoming)
    print(format_plan(plan, file_lots=len(incoming)))
    if not args.apply:
        print(
            "To był dry-run — nic nie zapisano. "
            "--apply zastąpi tabelę positions (as_of = teraz)."
        )
        return 0
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

    adapter = SupabasePositionsAdapter(url=settings.supabase_url, key=settings.supabase_key)
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
            "gate; minimum = every symbol below it. WIA-11: plus the worst week at the "
            "FIN-12 ceiling against fast_loop_weekly_budget_usd. Offline, zero paid calls."
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
    # WIA-11: tydzień przy suficie FIN-12 — ta sama reguła co kontrola startu Fast Loopa.
    print()
    print(render_weekly_budget(settings.fast_loop_weekly_budget(personas=inputs.personas)))
    return 0


def _add_render_sample_report(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subcommands.add_parser(
        "render-sample-report",
        help="Render the report email of a fictional cycle (no keys, no database, no network).",
        description=(
            "DX-01: the HTML and plain-text email of a fixed, fictional Fast Loop cycle, "
            "built by the same build_html_report as the real run. Zero paid calls."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/sample-report"),
        help="Directory for report-email.html and report-email.txt (default: build/sample-report).",
    )
    parser.set_defaults(handler=_run_render_sample_report)


def _run_render_sample_report(args: argparse.Namespace) -> int:
    """DX-01: zapis przykładowego maila — bez `Settings`, kluczy i sieci; exit 0."""
    out_dir: Path = args.out_dir
    report = render_sample_report()
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / SAMPLE_HTML_NAME
    text_path = out_dir / SAMPLE_TEXT_NAME
    html_path.write_text(report.html, encoding="utf-8")
    text_path.write_text(report.text, encoding="utf-8")
    print(f"✅ przykładowy raport: {html_path} i {text_path}")
    return 0


def _add_render_graph_diagram(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subcommands.add_parser(
        "render-graph-diagram",
        help="Write the Fast Loop graph as Mermaid, generated from create_agent_graph.",
        description=(
            "DOC-17: docs/graph.mmd (the production graph, with fetch_fundamentals) and "
            "docs/graph-without-fundamentals.mmd, drawn by LangGraph from the code. "
            "No keys, no database, no network."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs"),
        help="Directory for the .mmd files (default: docs).",
    )
    parser.set_defaults(handler=_run_render_graph_diagram)


def _run_render_graph_diagram(args: argparse.Namespace) -> int:
    """DOC-17: zapis diagramów grafu — bez `Settings`, kluczy i sieci; exit 0."""
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, diagram in graph_diagrams().items():
        path = out_dir / name
        path.write_text(diagram, encoding="utf-8")
        print(f"✅ diagram grafu: {path}")
    return 0


def collect_repo_stats(root: Path = _REPO_ROOT) -> RepoStats:
    """DOC-18: liczby repo z plików `.py` w `src/` i `tests/` oraz z nazw migracji."""
    sources = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for directory in ("src", "tests")
        for path in sorted((root / directory).rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    migrations = [path.name for path in (root / "supabase" / "migrations").glob("*.sql")]
    return compute_repo_stats(sources, migrations)


def _add_repo_stats(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subcommands.add_parser(
        "repo-stats",
        help="Count tests, migrations, ports, adapters and domain modules into a JSON file.",
        description=(
            "DOC-18: the repository numbers the docs quote (test functions by AST, "
            "migrations, ABC ports, adapter modules, domain modules, a stdlib-only check "
            "of the domain). No keys, no database, no network."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs") / REPO_STATS_NAME,
        help="Output JSON file (default: docs/repo-stats.json).",
    )
    parser.set_defaults(handler=_run_repo_stats)


def _run_repo_stats(args: argparse.Namespace) -> int:
    """DOC-18: zapis liczb repo — bez `Settings`, kluczy i sieci; exit 0."""
    out: Path = args.out
    stats_json = collect_repo_stats().to_json()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stats_json, encoding="utf-8")
    print(stats_json, end="")
    return 0


def _add_explain(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subcommands.add_parser(
        "explain",
        help="Explain a prediction from its decision receipt (read-only).",
        description=(
            "WIA-56: RAG analogues with similarity, SHAP-lite attributions, the effective "
            "threshold and the council votes of one prediction (by id) or of every "
            "prediction of a symbol on one UTC day. Read-only, zero paid calls."
        ),
    )
    parser.add_argument("prediction_id", nargs="?", help="Id predykcji (prediction_logs.id).")
    parser.add_argument(
        "--symbol", type=str.upper, help="Symbol (razem z --date): predykcje z tego dnia."
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        metavar="RRRR-MM-DD",
        help="Dzień UTC predykcji (razem z --symbol).",
    )
    parser.set_defaults(handler=_run_explain, explain_parser=parser)


def _run_explain(args: argparse.Namespace) -> int:
    """WIA-56: „dlaczego?” — kwit po id albo po symbolu i dniu; exit 0 zawsze."""
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

    by_id = args.prediction_id is not None
    by_day = args.symbol is not None and args.date is not None
    partial_day = (args.symbol is None) != (args.date is None)
    if by_id == by_day or partial_day:
        parser: argparse.ArgumentParser = args.explain_parser
        parser.error("podaj <prediction_id> albo --symbol X razem z --date RRRR-MM-DD")
    settings = _load_settings()
    repo = SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key)
    if by_id:
        print(explain_prediction(repo, args.prediction_id))
    else:
        print(explain_symbol_day(repo, args.symbol, args.date))
    return 0


def _add_persona_ablation(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subcommands.add_parser(
        "persona-ablation",
        help="Does persona weighting help out of sample? (read-only, needs migration 045).",
        description=(
            "WIA-46: replays the council verdict of every past prediction with the persona "
            "weights known at that moment (votes settled before it) and without weights."
        ),
    )
    parser.add_argument(
        "--days", type=int, default=90, help="Okno (dni) ocenianych predykcji (default: 90)."
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=90,
        help="Okno wag person przed każdą predykcją, jak w Fast Loopie (default: 90).",
    )
    parser.set_defaults(handler=_run_persona_ablation)


def _run_persona_ablation(args: argparse.Namespace) -> int:
    """WIA-46: wagi prequential kontra brak wag; exit 1 tylko bez danych RPC 045."""
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

    settings = _load_settings()
    repo = SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key)
    since = _now() - timedelta(days=args.days)
    # Historia sięga okno wag dalej niż najstarsza oceniana predykcja.
    outcomes = _read_persona_votes(repo, window_days=args.days + args.window_days)
    if outcomes is None:
        return 1
    report = replay_persona_weighting(
        outcomes,
        since=since,
        window_days=args.window_days,
        crypto_symbols=settings.crypto_symbols,
    )
    print(format_ablation(report, days=args.days, window_days=args.window_days))
    return 0


def _read_persona_votes(
    repo: SupabaseRepository, *, window_days: int
) -> list[PersonaVoteOutcome] | None:
    """Surowe głosy person z RPC 045 albo None z komunikatem na stderr (brak
    migracji, błąd sieci) — wspólne dla ablacji ważenia i leave-one-out."""
    try:
        return repo.get_persona_vote_outcomes(window_days=window_days)
    except Exception as exc:
        print(
            f"❌ RPC persona_vote_outcomes niedostępne ({type(exc).__name__}) — "
            "uruchom migrację 045 (DB Migrate).",
            file=sys.stderr,
        )
        return None


def _add_council_ablation(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subcommands.add_parser(
        "council-ablation",
        help="Marginal contribution of each council persona, leave-one-out (read-only, needs 045).",
        description=(
            "WIA-43: for every settled prediction recomputes the council verdict without "
            "persona X and reports the change in hit rate and Brier of the consensus strength, "
            "with a block-bootstrap interval. Personas whose removal does not hurt are "
            "candidates for removal."
        ),
    )
    parser.add_argument(
        "--days", type=int, default=90, help="Okno (dni) ocenianych predykcji (default: 90)."
    )
    parser.set_defaults(handler=_run_council_ablation)


def _run_council_ablation(args: argparse.Namespace) -> int:
    """WIA-43: leave-one-out na zapisanych głosach; exit 1 tylko bez danych RPC 045."""
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

    settings = _load_settings()
    repo = SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key)
    outcomes = _read_persona_votes(repo, window_days=args.days)
    if outcomes is None:
        return 1
    report = leave_one_out(outcomes, crypto_symbols=settings.crypto_symbols)
    print(format_council_ablation(report, days=args.days))
    return 0


def _add_verify_attestation(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subcommands.add_parser(
        "verify-attestation",
        help="Check the published commit-reveal ledger offline (no keys, no database, no network).",
        description=(
            "WIA-26: for every reveal in reveals.jsonl recomputes SHA-256(payload + salt) and "
            "compares it with the commitment published in commitments.jsonl. Prints OK and "
            "exits 0, or FAIL and exits 1; exits 2 when a file cannot be read."
        ),
    )
    parser.add_argument(
        "--commitments",
        type=Path,
        default=Path(DEFAULT_COMMITMENTS_PATH),
        help=f"Commitments ledger (default: {DEFAULT_COMMITMENTS_PATH}).",
    )
    parser.add_argument(
        "--reveals",
        type=Path,
        default=Path(DEFAULT_REVEALS_PATH),
        help=f"Reveals ledger (default: {DEFAULT_REVEALS_PATH}).",
    )
    parser.set_defaults(handler=_run_verify_attestation)


def _run_verify_attestation(args: argparse.Namespace) -> int:
    """WIA-26: odczyt dwóch plików ledgera i werdykt — bez `Settings`, kluczy i sieci.

    Exit 0 = OK, 1 = FAIL (reveal nie odtwarza commitmentu albo ledger uszkodzony),
    2 = pliku nie da się odczytać (jak błąd użycia)."""
    texts: dict[str, str] = {}
    for label, path in (("commitments", args.commitments), ("reveals", args.reveals)):
        try:
            texts[label] = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"❌ nie da się odczytać {path}: {exc.strerror or exc}", file=sys.stderr)
            return 2
    commitments, bad_commitments = parse_jsonl(texts["commitments"], Path(args.commitments).name)
    reveals, bad_reveals = parse_jsonl(texts["reveals"], Path(args.reveals).name)
    report = verify_ledger(commitments, reveals, malformed=[*bad_commitments, *bad_reveals])
    print(format_report(report))
    return 0 if report.ok else 1


def _add_reproduce_training(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subcommands.add_parser(
        "reproduce-training",
        help="Do the sidecar's dataset fingerprint and seed give the saved booster (read-only)?",
        description=(
            "ML-22: rebuilds the shared model's sample the way the Slow Loop does, keeps the "
            "rows inside the sidecar's date range, compares their fingerprint with the "
            "sidecar, refits with the sidecar's parameters and compares the tree dump with "
            "the saved weights. Exit 0 = reproduced, 1 = not reproduced, 2 = no sidecar or "
            "weights."
        ),
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Sidecar JSON (default: ml_model_path with the .json suffix).",
    )
    parser.set_defaults(handler=_run_reproduce_training)


def _run_reproduce_training(args: argparse.Namespace) -> int:
    """ML-22: odczyt próbki (bez zapisu i bez odświeżania widoku), refit i werdykt."""
    from main_trainer import build_use_case
    from src.domain.model_scorecard import TrainingReport
    from src.infrastructure.adapters.xgboost_local import LIBRARY_VERSION, XGBoostAdapter

    settings = _load_settings()
    sidecar_path = args.sidecar or Path(settings.ml_model_path).with_suffix(".json")
    try:
        sidecar = parse_sidecar(Path(sidecar_path).read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"❌ nie da się odczytać {sidecar_path}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    except SidecarError as exc:
        print(f"❌ {sidecar_path}: {exc}", file=sys.stderr)
        return 2
    saved = XGBoostAdapter(model_path=settings.ml_model_path)
    if not saved.is_trained:
        print(f"❌ brak pliku wag {settings.ml_model_path}", file=sys.stderr)
        return 2
    trainable = list(dict.fromkeys([*settings.symbols, *settings.crypto_symbols]))
    prepared = build_use_case(settings).pooled_training_set(trainable)
    if isinstance(prepared, TrainingReport):
        print(f"NOT REPRODUCED: the sample is below the training minimum — {prepared.reason}")
        return 1
    sample = rows_in_range(prepared, sidecar)
    fingerprint = fingerprint_of(sample)
    refit_dump = (
        XGBoostAdapter.refit_booster_dump(sample.features, sample.target, params=sidecar.params)
        if fingerprint == sidecar.dataset_sha256
        else None
    )
    report = assess_reproduction(
        sidecar,
        fingerprint=fingerprint,
        n_rows=len(sample.target),
        refit_dump=refit_dump,
        saved_dump=saved.booster_dump(),
        library_version=LIBRARY_VERSION,
    )
    print(format_reproduction(report))
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
