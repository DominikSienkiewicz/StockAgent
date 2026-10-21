"""Wiring `main_agent` dla uczciwego track recordu (W2-2a).

- WIA-35: chip trafności per symbol z darmowego odczytu zamkniętych predykcji
  okna track recordu, za tą samą flagą co krzywa kapitału.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import main_agent
from src.application.ports import RepositoryPort


def _settings(**overrides: object) -> MagicMock:
    settings = MagicMock()
    settings.track_record_days = 30
    settings.equity_curve_enabled = True
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


def _row(symbol: str, trend: str, correct: bool) -> dict[str, object]:
    return {"symbol": symbol, "predicted_trend": trend, "is_trend_correct": correct}


class TestSymbolTrackRecordWiring:
    def test_records_come_from_the_track_record_window(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.return_value = [
            _row("NVDA", "BULLISH", True),
            _row("NVDA", "BEARISH", False),
            _row("NVDA", "SIDEWAYS", True),
        ]

        view = main_agent._build_symbol_track_record(_settings(), repo)

        assert view is not None
        assert view.window_days == 30
        assert (view.records["NVDA"].hits, view.records["NVDA"].directional) == (1, 2)
        repo.get_resolved_predictions.assert_called_once_with(30)

    def test_flag_off_reads_nothing(self) -> None:
        repo = MagicMock(spec=RepositoryPort)

        view = main_agent._build_symbol_track_record(_settings(equity_curve_enabled=False), repo)

        assert view is None
        repo.get_resolved_predictions.assert_not_called()

    def test_read_error_leaves_the_mail_without_chips(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.side_effect = RuntimeError("supabase down")

        assert main_agent._build_symbol_track_record(_settings(), repo) is None
