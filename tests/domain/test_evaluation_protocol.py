"""WIA-42: parser protokołu ewaluacji (czysta domena, bez plików)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from src.domain.evaluation_protocol import parse_evaluation_protocol


def _data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "version": 1,
        "adopted_on": date(2026, 10, 20),
        "horizon": {"hours": 12},
        "hit_rule": {
            "directional_dead_zone": "0",
            "sideways_band": "0.005",
            "crypto_sideways_band": "0.025",
            "sideways_in_hit_rate": False,
        },
        "baselines": {"naive": ["always_bullish"], "benchmarks": ["sp500_buy_and_hold"]},
        "metrics": {"primary": ["directional_hit_rate"], "secondary": ["paper_pnl_net"]},
        "significance": {"alpha": 0.05, "fdr_q": 0.1},
        "window": {"track_record_days": 30},
        "universe": {"source": "config.toml", "selection": "manual", "selected": "2026-06"},
    }
    data.update(overrides)
    return data


def test_parses_bands_as_exact_decimals_and_keeps_the_hash() -> None:
    protocol = parse_evaluation_protocol(_data(), "c" * 64)

    assert protocol.sideways_band == Decimal("0.005")
    assert protocol.crypto_sideways_band == Decimal("0.025")
    assert protocol.directional_dead_zone == Decimal("0")
    assert protocol.naive_baselines == ("always_bullish",)
    assert protocol.adopted_on == date(2026, 10, 20)
    assert protocol.sha256 == "c" * 64


def test_missing_section_stops_the_run() -> None:
    data = _data()
    del data["hit_rule"]

    with pytest.raises(KeyError, match="hit_rule"):
        parse_evaluation_protocol(data, "c" * 64)


def test_adopted_on_must_be_a_date() -> None:
    with pytest.raises(ValueError, match="adopted_on"):
        parse_evaluation_protocol(_data(adopted_on="2026-10-20"), "c" * 64)
