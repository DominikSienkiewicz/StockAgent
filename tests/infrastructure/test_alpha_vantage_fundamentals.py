from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.adapters.alpha_vantage_fundamentals import (
    AlphaVantageFundamentalsAdapter,
)

OVERVIEW_JSON: dict[str, Any] = {
    "Symbol": "AAPL",
    "PERatio": "28.50",
    "ForwardPE": "25.10",
    "PEGRatio": "2.30",
    "EPS": "6.10",
}

EARNINGS_JSON: dict[str, Any] = {
    "symbol": "AAPL",
    "quarterlyEarnings": [
        {"fiscalDateEnding": "2026-03-31", "reportedEPS": "1.60"},
        {"fiscalDateEnding": "2025-12-31", "reportedEPS": "1.55"},
        {"fiscalDateEnding": "2025-09-30", "reportedEPS": "1.50"},
        {"fiscalDateEnding": "2025-06-30", "reportedEPS": "1.45"},
        {"fiscalDateEnding": "2025-03-31", "reportedEPS": "1.40"},
    ],
}


def _mock_response(json_data: dict[str, Any], status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def test_get_fundamentals_parses_overview_and_earnings() -> None:
    adapter = AlphaVantageFundamentalsAdapter(api_keys=["KEY1"])
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.side_effect = [
            _mock_response(OVERVIEW_JSON),
            _mock_response(EARNINGS_JSON),
        ]
        f = adapter.get_fundamentals("AAPL")

    assert f is not None
    assert f.trailing_pe == pytest.approx(28.50)
    assert f.forward_pe == pytest.approx(25.10)
    assert f.peg_ratio == pytest.approx(2.30)
    assert f.eps_growth_yoy == pytest.approx(0.1428, abs=1e-3)


def test_get_fundamentals_returns_none_on_empty_overview() -> None:
    adapter = AlphaVantageFundamentalsAdapter(api_keys=["KEY1"])
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.side_effect = [_mock_response({}), _mock_response({})]
        assert adapter.get_fundamentals("VOO") is None


def test_get_fundamentals_handles_string_none_values() -> None:
    bad_overview: dict[str, Any] = {
        "Symbol": "XYZ",
        "PERatio": "None",
        "ForwardPE": "-",
        "PEGRatio": "None",
        "EPS": "None",
    }
    adapter = AlphaVantageFundamentalsAdapter(api_keys=["KEY1"])
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.side_effect = [_mock_response(bad_overview), _mock_response({})]
        f = adapter.get_fundamentals("XYZ")
    assert f is None


def test_get_fundamentals_returns_none_on_http_error() -> None:
    adapter = AlphaVantageFundamentalsAdapter(api_keys=["KEY1"])
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.side_effect = Exception("network down")
        assert adapter.get_fundamentals("AAPL") is None
