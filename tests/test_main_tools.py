"""`main_tools.py` — composition root narzędzi jednorazowych (ARCH-39).

Narzędzia CLI dostają `Settings` i adaptery wyłącznie tutaj; `src/tools/` trzyma
czystą logikę. Każde narzędzie to podkomenda argparse.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_tools
from src.domain.portfolio import Position
from tests.settings_kwargs import NO_ENV_FILE


def test_main_tools_requires_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main_tools.main([])

    assert exit_info.value.code == 2
    assert "<command>" in capsys.readouterr().err


def test_main_tools_help_names_the_composition_root(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main_tools.main(["--help"])

    assert exit_info.value.code == 0
    assert "main_tools.py" in capsys.readouterr().out


# --- PORT-18: import-positions ------------------------------------------------

_CSV = (
    "symbol,quantity,avg_cost,currency,purchase_date\n"
    "NVDA,10,118.50,USD,2025-03-01\n"
    "SAP.DE,4,182.10,EUR,\n"
)
_NOW = datetime(2026, 10, 19, 8, 0, tzinfo=UTC)


@pytest.fixture
def positions_adapter(mocker: Any) -> MagicMock:
    mocker.patch("main_tools._now", return_value=_NOW)
    mocker.patch("src.config.Settings.from_env")
    adapter_cls: MagicMock = mocker.patch(
        "src.infrastructure.adapters.positions_repo.SupabasePositionsAdapter"
    )
    adapter_cls.return_value.current_positions.return_value = [
        Position(symbol="NVDA", quantity=Decimal("8"), purchase_price=Decimal("118"))
    ]
    return adapter_cls


def _csv_file(tmp_path: Path, text: str = _CSV) -> str:
    path = tmp_path / "positions.csv"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_import_positions_is_a_dry_run_without_apply(
    positions_adapter: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main_tools.main(["import-positions", "--csv", _csv_file(tmp_path)])

    assert code == 0
    positions_adapter.return_value.replace_positions.assert_not_called()
    out = capsys.readouterr().out
    assert "+ NVDA       10 × 118.5 USD, zakup 2025-03-01" in out
    assert "- NVDA       8 × 118 USD, zakup nieznany" in out
    assert "dry-run" in out


def test_import_positions_composes_the_adapter_from_settings(
    positions_adapter: MagicMock, tmp_path: Path
) -> None:
    main_tools.main(["import-positions", "--csv", _csv_file(tmp_path)])

    positions_adapter.assert_called_once()
    assert set(positions_adapter.call_args.kwargs) == {"url", "key"}


def test_import_positions_apply_replaces_the_table_with_as_of_now(
    positions_adapter: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main_tools.main(["import-positions", "--csv", _csv_file(tmp_path), "--apply"])

    assert code == 0
    replace = positions_adapter.return_value.replace_positions
    replace.assert_called_once()
    lots, as_of = replace.call_args.args[0], replace.call_args.kwargs["as_of"]
    assert [(lot.symbol, lot.currency, lot.purchase_date) for lot in lots] == [
        ("NVDA", "USD", date(2025, 3, 1)),
        ("SAP.DE", "EUR", None),
    ]
    assert as_of == _NOW
    assert "zapisano 2 lot(ów)" in capsys.readouterr().out


def test_import_positions_rejects_an_invalid_file_before_touching_the_database(
    positions_adapter: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _csv_file(tmp_path, "symbol,quantity,avg_cost,currency\nNVDA,-1,118,usd1\n")

    code = main_tools.main(["import-positions", "--csv", bad, "--apply"])

    assert code == 1
    positions_adapter.assert_not_called()
    err = capsys.readouterr().err
    assert "linia 2: quantity -1" in err
    assert "linia 2: currency USD1" in err


def test_import_positions_reports_a_missing_file(
    positions_adapter: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main_tools.main(["import-positions", "--csv", str(tmp_path / "missing.csv")])

    assert code == 1
    assert "missing.csv" in capsys.readouterr().err


def test_import_positions_apply_says_nothing_was_written_without_db_writes(
    positions_adapter: MagicMock,
    mocker: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # BEZ-08: lokalny run jest domyślnie tylko do odczytu — `--apply` nie może
    # wtedy drukować „✅ zapisano”, skoro tabela positions została nietknięta.
    settings = mocker.patch("src.config.Settings.from_env").return_value
    settings.db_writes_enabled = False

    code = main_tools.main(["import-positions", "--csv", _csv_file(tmp_path), "--apply"])

    assert code == 1
    positions_adapter.return_value.replace_positions.assert_not_called()
    captured = capsys.readouterr()
    assert "✅" not in captured.out
    assert "nic nie zapisano" in captured.err
    assert "DB_WRITES_ENABLED=true" in captured.err
    # Diff z tabelą i tak wychodzi — to ten sam odczyt co w dry-runie.
    assert "+ NVDA       10 × 118.5 USD, zakup 2025-03-01" in captured.out


def test_import_positions_dry_run_is_unchanged_without_db_writes(
    positions_adapter: MagicMock,
    mocker: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = mocker.patch("src.config.Settings.from_env").return_value
    settings.db_writes_enabled = False

    code = main_tools.main(["import-positions", "--csv", _csv_file(tmp_path)])

    assert code == 0
    assert "dry-run" in capsys.readouterr().out


# --- ARCH-39: evaluate jako podkomenda ----------------------------------------


def test_evaluate_is_a_main_tools_subcommand(
    mocker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    mocker.patch("src.config.Settings.from_env")
    repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
    repo_cls.return_value.get_resolved_predictions_for_eval.return_value = []

    code = main_tools.main(["evaluate", "--days", "7"])

    assert code == 0
    repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(days=7)
    assert "ostatnie 7 dni" in capsys.readouterr().out


def test_src_tools_evaluate_has_no_entry_point_of_its_own() -> None:
    from src.tools import evaluate

    assert not hasattr(evaluate, "main")


# --- FIN-06: cost-envelope ----------------------------------------------------


def test_cost_envelope_prints_the_bound_for_the_configured_watchlist(
    mocker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = mocker.patch("src.config.Settings.from_env").return_value
    settings.symbols = ["AAPL", "MSFT", "CSPX.L"]
    settings.crypto_symbols = ["BTC"]
    settings.symbols_unsupported_price = ["CSPX.L"]
    settings.council_personas_dir = "data/council_personas"
    settings.debate_mode_enabled = False
    settings.tool_use_enabled = True
    settings.tool_use_max_rounds = 3
    settings.frontier_escalations_per_cycle.return_value = 0
    settings.shadow_calls_per_cycle.return_value = 0
    from src.domain.finops import WeeklyCostBudget

    settings.fast_loop_weekly_budget.return_value = WeeklyCostBudget(
        cycle_worst_usd=0.0, cycles_per_week=5, budget_usd=None
    )

    code = main_tools.main(["cost-envelope"])

    out = capsys.readouterr().out
    assert code == 0
    # 3 symbole za bramką (CSPX.L bez ceny), rada z 7 plików person + chairman.
    assert "| `council_llm` | 0 | 24 |" in out
    # Diagnoza + pętla tool-use (3 rundy + 1) na symbol.
    assert "| `llm` | 0 | 15 |" in out
    # Feed AV: AAPL, MSFT, BTC — CSPX.L odpada na filtrze tickerów AV.
    assert "0–3" in out


def test_cost_envelope_prints_the_weekly_worst_case_against_the_budget(
    mocker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    # WIA-11: wiersz tygodniowy liczy `Settings.fast_loop_weekly_budget` z personami
    # wczytanymi przez narzędzie (7 plików person w repo).
    from src.config import Settings

    real = Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://t.supabase.co",
        supabase_key="anon",
        symbols=[f"S{i:02d}" for i in range(42)],
        max_paid_llm_calls_per_cycle=100,
        symbol_concurrency=4,
        fast_loop_weekly_budget_usd=7.0,
    )
    mocker.patch("src.config.Settings.from_env", return_value=real)

    code = main_tools.main(["cost-envelope"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Weekly worst case at the FIN-12 ceiling: 5 cycles" in out
    assert "1.2112 USD = 6.0559 USD" in out
    assert "`fast_loop_weekly_budget_usd` = 7.00 USD, within the budget." in out


def test_cost_envelope_counts_frontier_escalations_and_shadow_calls(
    mocker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    # WIA-11: routing modeli (FIN-21) i wariant w cieniu (EXP-26) z config.toml trafiają
    # do tabeli cyklu i do wiersza tygodniowego.
    from src.config import Settings, ShadowPromptVariantSettings

    real = Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://t.supabase.co",
        supabase_key="anon",
        symbols=[f"S{i:02d}" for i in range(42)],
        crypto_symbols=[],
        model_routing_enabled=True,
        routing_max_escalations=5,
        shadow_prompt_variant_enabled=True,
        shadow_prompt_variant=ShadowPromptVariantSettings(
            name="mini_rounded", max_calls_per_cycle=3
        ),
        max_paid_llm_calls_per_cycle=100,
        symbol_concurrency=4,
        fast_loop_weekly_budget_usd=7.0,
    )
    mocker.patch("src.config.Settings.from_env", return_value=real)

    code = main_tools.main(["cost-envelope"])

    out = capsys.readouterr().out
    assert code == 0
    assert "| `llm` | 0 | 89 |" in out
    assert "| `shadow_llm` | 0 | 3 |" in out
    # Cykl przy suficie: 1.2112 + 5 eskalacji i 3 wywołania cienia po 0.01 USD.
    assert "1.2912 USD = 6.4559 USD" in out


# --- DX-01: render-sample-report ----------------------------------------------


def test_render_sample_report_writes_both_bodies_without_settings(
    mocker: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from_env = mocker.patch("src.config.Settings.from_env", side_effect=AssertionError)

    code = main_tools.main(["render-sample-report", "--out-dir", str(tmp_path / "out")])

    assert code == 0
    from_env.assert_not_called()
    html = (tmp_path / "out" / "report-email.html").read_text(encoding="utf-8")
    text = (tmp_path / "out" / "report-email.txt").read_text(encoding="utf-8")
    assert html.startswith("<") and "Krypto" in html
    assert "przebieg #sample" in text
    assert "report-email.html" in capsys.readouterr().out


# --- WIA-56: explain ----------------------------------------------------------


def test_explain_by_prediction_id_reads_the_receipt(
    mocker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    mocker.patch("src.config.Settings.from_env")
    repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
    repo_cls.return_value.get_receipt.return_value = None

    code = main_tools.main(["explain", "pred-404"])

    assert code == 0
    repo_cls.return_value.get_receipt.assert_called_once_with("pred-404")
    assert "nie znaleziono predykcji pred-404" in capsys.readouterr().out


def test_explain_by_symbol_and_date_reads_that_utc_day(
    mocker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    mocker.patch("src.config.Settings.from_env")
    repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
    repo_cls.return_value.find_receipts.return_value = []

    code = main_tools.main(["explain", "--symbol", "nvda", "--date", "2026-08-14"])

    assert code == 0
    repo_cls.return_value.find_receipts.assert_called_once_with("NVDA", date(2026, 8, 14))
    assert "brak predykcji NVDA z 2026-08-14 (UTC)" in capsys.readouterr().out


def test_explain_reads_the_council_votes_of_the_first_round(mocker: Any) -> None:
    # ML-40: kwit z głosami rady pokazuje niezależną rundę 1 — po 082 także wtedy,
    # gdy debata (rewizja w rundzie 2) jest już wyłączona.
    mocker.patch("src.config.Settings.from_env")
    repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
    repo_cls.return_value.get_receipt.return_value = None

    main_tools.main(["explain", "pred-404"])

    assert repo_cls.call_args.kwargs["council_vote_rounds_from_schema"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["explain"],
        ["explain", "--symbol", "NVDA"],
        ["explain", "pred-1", "--symbol", "NVDA", "--date", "2026-08-14"],
        ["explain", "pred-1", "--date", "2026-08-14"],
        ["explain", "--symbol", "NVDA", "--date", "14.08.2026"],
    ],
)
def test_explain_needs_an_id_or_a_symbol_with_a_date(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main_tools.main(argv)

    assert exit_info.value.code == 2


# --- WIA-46: persona-ablation -------------------------------------------------


def test_persona_ablation_reads_the_history_behind_the_scored_window(
    mocker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = mocker.patch("src.config.Settings.from_env").return_value
    settings.crypto_symbols = ["BTC"]
    repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
    repo_cls.return_value.get_persona_vote_outcomes.return_value = []

    code = main_tools.main(["persona-ablation", "--days", "30", "--window-days", "60"])

    assert code == 0
    # Wagi pierwszej ocenianej predykcji potrzebują pełnego okna historii przed nią.
    repo_cls.return_value.get_persona_vote_outcomes.assert_called_once_with(window_days=90)
    assert "persona_weighting_enabled" in capsys.readouterr().out


def test_persona_ablation_without_migration_045_says_so(
    mocker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    mocker.patch("src.config.Settings.from_env")
    repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
    repo_cls.return_value.get_persona_vote_outcomes.side_effect = RuntimeError("PGRST202")

    code = main_tools.main(["persona-ablation"])

    assert code == 1
    assert "045" in capsys.readouterr().err


# --- DOC-17: render-graph-diagram -----------------------------------------------


def test_render_graph_diagram_writes_both_variants_without_settings(
    mocker: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from_env = mocker.patch("src.config.Settings.from_env", side_effect=AssertionError)

    code = main_tools.main(["render-graph-diagram", "--out-dir", str(tmp_path)])

    assert code == 0
    from_env.assert_not_called()
    production = (tmp_path / "graph.mmd").read_text(encoding="utf-8")
    assert "fetch_fundamentals" in production
    assert "fetch_fundamentals" not in (tmp_path / "graph-without-fundamentals.mmd").read_text(
        encoding="utf-8"
    )
    assert "graph.mmd" in capsys.readouterr().out


# --- DOC-18: repo-stats -----------------------------------------------------------


def test_repo_stats_writes_the_counts_without_settings(
    mocker: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from_env = mocker.patch("src.config.Settings.from_env", side_effect=AssertionError)
    out = tmp_path / "stats.json"

    code = main_tools.main(["repo-stats", "--out", str(out)])

    assert code == 0
    from_env.assert_not_called()
    stats = json.loads(out.read_text(encoding="utf-8"))
    assert stats["domain_stdlib_only"] is True
    assert stats["test_functions"] > 0 and stats["migrations"] > 0
    assert '"ports"' in capsys.readouterr().out


# --- WIA-26: verify-attestation ------------------------------------------------


def _write_ledger(tmp_path: Path, *, tamper: bool = False) -> tuple[str, str]:
    import json

    from src.domain import attestation

    payload = attestation.prediction_commitment_payload(
        symbol="AAPL",
        predicted_trend="BULLISH",
        predicted_target_price=Decimal("101.5"),
        price_at_prediction=Decimal("100"),
    )
    plaintext = attestation.canonicalize(payload, schema_version=attestation.SCHEMA_VERSION)
    salt = "0123456789abcdef0123456789abcdef"
    commitment = attestation.commit_canonical(plaintext, salt)
    reveal = {
        "symbol": "AAPL",
        "commitment": commitment,
        "salt": "f" * 32 if tamper else salt,
        "payload": plaintext,
    }
    commitments_path = tmp_path / "commitments.jsonl"
    reveals_path = tmp_path / "reveals.jsonl"
    commitments_path.write_text(
        json.dumps({"symbol": "AAPL", "commitment": commitment}) + "\n", encoding="utf-8"
    )
    reveals_path.write_text(json.dumps(reveal) + "\n", encoding="utf-8")
    return str(commitments_path), str(reveals_path)


def test_verify_attestation_prints_ok_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    commitments, reveals = _write_ledger(tmp_path)

    code = main_tools.main(
        ["verify-attestation", "--commitments", commitments, "--reveals", reveals]
    )

    assert code == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("OK")


def test_verify_attestation_prints_fail_and_exits_one_for_a_tampered_reveal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    commitments, reveals = _write_ledger(tmp_path, tamper=True)

    code = main_tools.main(
        ["verify-attestation", "--commitments", commitments, "--reveals", reveals]
    )

    assert code == 1
    assert capsys.readouterr().out.splitlines()[-1].startswith("FAIL")


def test_verify_attestation_exits_two_for_a_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main_tools.main(
        [
            "verify-attestation",
            "--commitments",
            str(tmp_path / "missing.jsonl"),
            "--reveals",
            str(tmp_path / "missing-too.jsonl"),
        ]
    )

    assert code == 2
    assert "missing.jsonl" in capsys.readouterr().err


def test_verify_attestation_needs_no_settings_and_no_network(tmp_path: Path, mocker: Any) -> None:
    # Weryfikator ma działać u sceptyka, który nie ma żadnych kluczy.
    from_env = mocker.patch("src.config.Settings.from_env")
    commitments, reveals = _write_ledger(tmp_path)

    main_tools.main(["verify-attestation", "--commitments", commitments, "--reveals", reveals])

    from_env.assert_not_called()


def test_verify_attestation_defaults_to_the_published_ledger_paths() -> None:
    args = main_tools.build_parser().parse_args(["verify-attestation"])

    assert str(args.commitments) == "public/attestation/commitments.jsonl"
    assert str(args.reveals) == "public/attestation/reveals.jsonl"


def test_verify_attestation_checks_rekor_entries_against_the_fast_loop_by_default() -> None:
    # WIA-73: tożsamość workflowu, którą musi nazywać certyfikat wpisu Rekora.
    args = main_tools.build_parser().parse_args(["verify-attestation"])

    assert (getattr(args, "rekor_repository", None), getattr(args, "rekor_workflow", None)) == (
        "DominikSienkiewicz/StockAgent",
        ".github/workflows/fast_loop_12h.yml",
    )


def test_verify_attestation_takes_the_rekor_identity_of_a_fork(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    from datetime import UTC, datetime, timedelta

    from src.domain import attestation
    from tests.sigstore_support import rekor_entry_for

    committed_at = datetime(2026, 10, 26, 5, 31, 7, tzinfo=UTC)
    line = attestation.cycle_root_record(
        cycle_id="1.1.20261026T053000Z", committed_at=committed_at, commitments=["aa" * 32]
    )
    fork = "https://github.com/someone/fork/.github/workflows/agent.yml@refs/heads/main"
    line["rekor"] = rekor_entry_for(
        str(line["root"]), integrated_at=committed_at + timedelta(seconds=3), uri=fork
    )
    commitments = tmp_path / "commitments.jsonl"
    commitments.write_text(json.dumps(line) + "\n", encoding="utf-8")
    (tmp_path / "reveals.jsonl").write_text("", encoding="utf-8")
    base = [
        "verify-attestation",
        "--commitments",
        str(commitments),
        "--reveals",
        str(tmp_path / "reveals.jsonl"),
    ]

    default_code = main_tools.main(base)
    fork_code = main_tools.main(
        [
            *base,
            "--rekor-repository",
            "someone/fork",
            "--rekor-workflow",
            ".github/workflows/agent.yml",
        ]
    )

    assert (default_code, fork_code) == (1, 0)
    assert "Cycle roots logged in Rekor: 1" in capsys.readouterr().out


# --- ML-22: reproduce-training ---------------------------------------------------


def _reproduction_case(mocker: Any, tmp_path: Path, *, train: bool = True) -> tuple[Any, Any]:
    """Prawdziwy trening XGBoost na małej próbce: plik wag + sidecar w `tmp_path`,
    a use case Slow Loopa (podmieniony) oddaje tę samą próbkę co przy treningu."""
    import numpy as np

    from src.application.use_cases.train_model import TrainingSet
    from src.domain.training_provenance import DatasetDescription
    from src.infrastructure.adapters.xgboost_local import XGBoostAdapter

    rng = np.random.default_rng(3)
    stamps = [datetime(2026, 10, 1, tzinfo=UTC) + timedelta(hours=12 * i) for i in range(200)]
    features: list[dict[str, float | None]] = [
        {"price_delta": float(delta), "llm_trend_signal": float(signal)}
        for delta, signal in zip(rng.normal(0, 0.05, 200), rng.choice([-1, 0, 1], 200), strict=True)
    ]
    target = [
        0.4 * float(row["price_delta"] or 0.0) + 0.02 * float(row["llm_trend_signal"] or 0.0)
        for row in features
    ]
    training_set = TrainingSet(
        features, target, stamps, DatasetDescription(n_per_symbol={"AAPL": 200})
    )
    model_path = tmp_path / "price_predictor.ubj"
    if train:
        report = XGBoostAdapter(model_path=str(model_path)).train(
            features, target, timestamps=stamps
        )
        assert report["status"] == "trained_successfully"
    settings = mocker.patch("src.config.Settings.from_env").return_value
    settings.ml_model_path = str(model_path)
    settings.symbols = ["AAPL"]
    settings.crypto_symbols = []
    use_case = mocker.patch("main_trainer.build_use_case").return_value
    use_case.pooled_training_set.return_value = training_set
    return use_case, training_set


def test_reproduce_training_rebuilds_the_saved_booster(
    mocker: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_case, _ = _reproduction_case(mocker, tmp_path)

    code = main_tools.main(["reproduce-training"])

    assert code == 0
    assert "REPRODUCED" in capsys.readouterr().out
    use_case.pooled_training_set.assert_called_once_with(["AAPL"])
    use_case.run_pooled.assert_not_called()


def test_reproduce_training_with_a_changed_dataset_exits_one(
    mocker: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import dataclasses

    use_case, training_set = _reproduction_case(mocker, tmp_path)
    use_case.pooled_training_set.return_value = dataclasses.replace(
        training_set, target=[*training_set.target[:-1], 1.0]
    )

    code = main_tools.main(["reproduce-training"])

    assert code == 1
    assert "dataset differs" in capsys.readouterr().out


def test_reproduce_training_without_a_sidecar_exits_two(
    mocker: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _reproduction_case(mocker, tmp_path, train=False)

    code = main_tools.main(["reproduce-training"])

    assert code == 2
    assert "price_predictor.json" in capsys.readouterr().err
