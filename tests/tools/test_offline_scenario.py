"""DX-04: scenariusz trybu offline i podsumowanie przebiegu (czyste funkcje)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from src.application.report_models import ResolvedPrediction
from src.tools.offline_scenario import (
    OfflineCycleSummary,
    format_offline_summary,
    parse_offline_scenario,
)

SCENARIO_FILE = Path(__file__).parents[2] / "data" / "offline" / "scenario.json"


def _raw(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "started_at": "2026-10-15T10:30:00+00:00",
        "cycle_interval_hours": 12,
        "symbols": ["NVDA"],
        "crypto_symbols": ["BTC"],
        "volatility_threshold": "0.02",
        "crypto_volatility_threshold": "0.05",
        "reflection_min_age_hours": 6,
        "track_record_days": 30,
        "seed_prices": {"NVDA": "100", "BTC": "60000"},
        "cycles": [{"NVDA": "103", "BTC": "61000"}],
    }
    raw.update(overrides)
    return raw


def test_parses_the_cycles_and_their_clock() -> None:
    scenario = parse_offline_scenario(_raw())

    assert scenario.all_symbols == ["NVDA", "BTC"]
    assert scenario.seeded_at == datetime(2026, 10, 14, 22, 30, tzinfo=UTC)
    assert scenario.cycle_at(1) == scenario.started_at + timedelta(hours=12)
    assert scenario.cycles[0]["NVDA"] == Decimal("103")
    assert scenario.council_volatility_threshold is None
    assert scenario.display_timezone == "Europe/Warsaw"


def test_the_shipped_scenario_parses() -> None:
    scenario = parse_offline_scenario(json.loads(SCENARIO_FILE.read_text(encoding="utf-8")))

    assert len(scenario.cycles) == 2
    assert set(scenario.seed_prices) == set(scenario.all_symbols)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"started_at": "2026-10-15T10:30:00"}, "time zone"),
        ({"cycles": []}, "cycles"),
        ({"symbols": [], "crypto_symbols": []}, "at least one symbol"),
        ({"volatility_threshold": "-0.02"}, "volatility_threshold"),
        ({"seed_prices": {"NVDA": "abc"}}, "seed_prices"),
        ({"cycles": [{"NVDA": "0"}]}, "cycles"),
        ({"reflection_min_age_hours": "6"}, "reflection_min_age_hours"),
        ({"paper_cost_bps": -1}, "paper_cost_bps"),
    ],
)
def test_malformed_scenario_names_the_field(override: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        parse_offline_scenario(_raw(**override))


def test_summary_lists_statuses_settlements_and_the_mail(tmp_path: Path) -> None:
    cycle = OfflineCycleSummary(
        index=2,
        started_at=datetime(2026, 10, 15, 22, 30, tzinfo=UTC),
        subject="StockAgent — NVDA +3,1%",
        statuses={"NVDA": "saved", "AAPL": "ignored", "MSFT": "saved"},
        resolved=(
            ResolvedPrediction(symbol="NVDA", predicted_trend="BULLISH", is_correct=True),
            ResolvedPrediction(symbol="MSFT", predicted_trend="BEARISH", is_correct=False),
        ),
        html_path=tmp_path / "report-email.html",
        text_path=tmp_path / "report-email.txt",
    )

    summary = format_offline_summary([cycle])

    assert summary.splitlines() == [
        "Cykl 2 (2026-10-15 22:30 UTC): zapisane: NVDA, MSFT · pominięte: AAPL · "
        "błędy: — · rozliczone: NVDA ✓, MSFT ✗",
        "  temat: StockAgent — NVDA +3,1%",
        f"  mail: {tmp_path / 'report-email.html'} i report-email.txt",
    ]
