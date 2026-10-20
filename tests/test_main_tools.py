"""`main_tools.py` — composition root narzędzi jednorazowych (ARCH-39).

Narzędzia CLI dostają `Settings` i adaptery wyłącznie tutaj; `src/tools/` trzyma
czystą logikę. Każde narzędzie to podkomenda argparse.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_tools
from src.domain.portfolio import Position


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
    settings = mocker.patch("src.config.Settings.from_env").return_value
    settings.positions_currency_enabled = True
    adapter_cls = mocker.patch(
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
    assert positions_adapter.call_args.kwargs["currency_columns"] is True


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


def test_import_positions_apply_refuses_what_the_schema_cannot_store(
    positions_adapter: MagicMock, mocker: Any, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mocker.patch("src.config.Settings.from_env").return_value.positions_currency_enabled = False

    code = main_tools.main(["import-positions", "--csv", _csv_file(tmp_path), "--apply"])

    assert code == 1
    positions_adapter.return_value.replace_positions.assert_not_called()
    assert "migracji 035" in capsys.readouterr().err


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
    positions_adapter: MagicMock, mocker: Any, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # BEZ-08: lokalny run jest domyślnie tylko do odczytu — `--apply` nie może
    # wtedy drukować „✅ zapisano”, skoro tabela positions została nietknięta.
    settings = mocker.patch("src.config.Settings.from_env").return_value
    settings.positions_currency_enabled = True
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
    positions_adapter: MagicMock, mocker: Any, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = mocker.patch("src.config.Settings.from_env").return_value
    settings.positions_currency_enabled = True
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
        _env_file=None,
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
