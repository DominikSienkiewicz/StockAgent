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


# Body sygnalizujący wyczerpanie dziennego limitu (25/dobę free tier):
# HTTP 200, ale zamiast danych pole `Information` (czasem `Note`).
RATE_LIMITED_JSON: dict[str, Any] = {
    "Information": (
        "Thank you for using Alpha Vantage! Our standard API rate limit "
        "is 25 requests per day."
    )
}


def test_get_fundamentals_emits_critical_alert_on_daily_quota() -> None:
    """Soft-limit 200 (pole `Information`) musi wystawić CRITICAL alert,
    a nie po cichu zwrócić None jak przy zwykłym braku danych."""
    from src.application.quota_monitor import QuotaMonitor
    from src.domain.quota import QuotaSeverity

    monitor = QuotaMonitor()
    adapter = AlphaVantageFundamentalsAdapter(
        api_keys=["KEY1"], quota_monitor=monitor
    )
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.return_value = _mock_response(RATE_LIMITED_JSON)
        result = adapter.get_fundamentals("AAPL")

    assert result is None
    assert len(monitor.alerts) == 1
    alert = monitor.alerts[0]
    assert alert.severity is QuotaSeverity.CRITICAL
    assert "fundamentals" in alert.source.lower()
    assert alert.message
    assert alert.action


def test_get_fundamentals_emits_critical_alert_on_note_key() -> None:
    """AV bywa niespójne: czasem soft-limit ma pole `Note` zamiast
    `Information`. Oba muszą być wykryte."""
    from src.application.quota_monitor import QuotaMonitor
    from src.domain.quota import QuotaSeverity

    monitor = QuotaMonitor()
    adapter = AlphaVantageFundamentalsAdapter(
        api_keys=["KEY1"], quota_monitor=monitor
    )
    note_json: dict[str, Any] = {
        "Note": "Thank you for using Alpha Vantage! ... 25 requests per day ..."
    }
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.return_value = _mock_response(note_json)
        result = adapter.get_fundamentals("AAPL")

    assert result is None
    assert len(monitor.alerts) == 1
    assert monitor.alerts[0].severity is QuotaSeverity.CRITICAL


def test_genuine_empty_response_does_not_emit_alert() -> None:
    """Zwykły pusty 200 (np. ETF bez fundamentów) to NIE jest quota —
    None bez alertu, distinguishable od soft-limitu."""
    from src.application.quota_monitor import QuotaMonitor

    monitor = QuotaMonitor()
    adapter = AlphaVantageFundamentalsAdapter(
        api_keys=["KEY1"], quota_monitor=monitor
    )
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.side_effect = [_mock_response({}), _mock_response({})]
        result = adapter.get_fundamentals("VOO")

    assert result is None
    assert monitor.alerts == []


def test_quota_monitor_is_optional_no_crash_on_quota() -> None:
    """Backward-compat: bez wstrzykniętego monitora soft-limit nie wywala
    adaptera, tylko zwraca None (z logiem)."""
    adapter = AlphaVantageFundamentalsAdapter(api_keys=["KEY1"])
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.return_value = _mock_response(RATE_LIMITED_JSON)
        assert adapter.get_fundamentals("AAPL") is None


class TestKeyRotation:
    """Rotacja kluczy wyłącznie na wykrytym wyczerpaniu (soft-limit),
    nie co request — inaczej 2 req/symbol smarują load po wszystkich
    kluczach i biją w dzienny cap równocześnie, bez rezerwy."""

    def test_non_exhausted_key_reused_across_calls(self) -> None:
        # 2 klucze, 3 symbole × 2 requesty = 6 wywołań na KEY1 (bez rotacji).
        adapter = AlphaVantageFundamentalsAdapter(api_keys=["KEY1", "KEY2"])
        with patch.object(adapter._session, "get") as mock_get:
            mock_get.side_effect = [
                _mock_response(OVERVIEW_JSON),
                _mock_response(EARNINGS_JSON),
            ] * 3
            adapter.get_fundamentals("AAPL")
            adapter.get_fundamentals("MSFT")
            adapter.get_fundamentals("GOOG")

        keys_used = [c.kwargs["params"]["apikey"] for c in mock_get.call_args_list]
        assert keys_used == ["KEY1"] * 6  # ani razu nie sięgnięto po KEY2

    def test_rotates_to_next_key_on_exhaustion(self) -> None:
        # KEY1 wyczerpany na OVERVIEW → rotacja → KEY2 dowozi OVERVIEW+EARNINGS.
        adapter = AlphaVantageFundamentalsAdapter(api_keys=["KEY1", "KEY2"])
        with patch.object(adapter._session, "get") as mock_get:
            mock_get.side_effect = [
                _mock_response(RATE_LIMITED_JSON),   # KEY1 / OVERVIEW — exhausted
                _mock_response(OVERVIEW_JSON),       # KEY2 / OVERVIEW — OK
                _mock_response(EARNINGS_JSON),       # KEY2 / EARNINGS — OK
            ]
            result = adapter.get_fundamentals("AAPL")

        assert result is not None
        keys_used = [c.kwargs["params"]["apikey"] for c in mock_get.call_args_list]
        assert keys_used == ["KEY1", "KEY2", "KEY2"]

    def test_returns_none_and_alerts_when_all_keys_exhausted(self) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        monitor = QuotaMonitor()
        adapter = AlphaVantageFundamentalsAdapter(
            api_keys=["KEY1", "KEY2"], quota_monitor=monitor
        )
        with patch.object(adapter._session, "get") as mock_get:
            mock_get.return_value = _mock_response(RATE_LIMITED_JSON)
            result = adapter.get_fundamentals("AAPL")

        assert result is None
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].severity is QuotaSeverity.CRITICAL

    def test_does_not_retry_already_exhausted_keys(self) -> None:
        # Po wyczerpaniu wszystkich kluczy kolejny symbol nie spala
        # już budżetu na martwych kluczach — short-circuit, zero requestów.
        adapter = AlphaVantageFundamentalsAdapter(api_keys=["KEY1", "KEY2"])
        with patch.object(adapter._session, "get") as mock_get:
            mock_get.return_value = _mock_response(RATE_LIMITED_JSON)
            adapter.get_fundamentals("AAPL")   # wyczerpuje oba klucze
            calls_after_first = mock_get.call_count
            result = adapter.get_fundamentals("MSFT")  # nie powinien wołać API

        assert result is None
        assert mock_get.call_count == calls_after_first
