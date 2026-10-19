"""ML-34 — mapowanie rozliczonych predykcji na okna kalibracyjne per klasa
aktywa i na pokrycie przedziałów. Zero I/O: wejście to wiersze z
`RepositoryPort.get_settled_moves`, wyjście to obiekty domeny."""

from __future__ import annotations

import pytest

from src.application.move_interval import calibration_windows, interval_coverage
from src.domain.conformal import CalibrationPoint, IntervalCoverage
from src.domain.value_objects import AssetType


def _asset_type(symbol: str) -> AssetType:
    return {"BTC": AssetType.CRYPTO, "VOO": AssetType.ETF}.get(symbol, AssetType.STOCK)


def _row(symbol: str, price: float, target: float | None, actual: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "price_at_prediction": price,
        "predicted_target_price": target,
        "actual_price_after_12h": actual,
    }


def test_windows_are_split_by_asset_class_in_input_order() -> None:
    rows = [
        _row("AAPL", 100.0, 100.0, 102.0),
        _row("BTC", 50_000.0, 50_000.0, 47_500.0),
        _row("MSFT", 200.0, 202.0, 199.0),
        _row("VOO", 400.0, 400.0, 404.0),
    ]

    windows = calibration_windows(rows, _asset_type)

    assert set(windows) == {AssetType.STOCK, AssetType.CRYPTO, AssetType.ETF}
    stock = windows[AssetType.STOCK]
    assert stock[0] == CalibrationPoint(predicted_return=0.0, realized_return=pytest.approx(0.02))
    assert stock[1].predicted_return == pytest.approx(0.01)
    assert stock[1].realized_return == pytest.approx(-0.005)
    assert windows[AssetType.CRYPTO][0].realized_return == pytest.approx(-0.05)


def test_missing_target_means_the_zero_predictor() -> None:
    windows = calibration_windows([_row("AAPL", 100.0, None, 101.0)], _asset_type)

    assert windows[AssetType.STOCK][0].predicted_return == 0.0


def test_unusable_rows_are_skipped() -> None:
    rows = [
        _row("AAPL", 0.0, 0.0, 1.0),
        {"symbol": "AAPL", "price_at_prediction": "x", "actual_price_after_12h": 1.0},
        {"price_at_prediction": 100.0, "actual_price_after_12h": 101.0},
        "not a row",
        _row("AAPL", 100.0, 100.0, 103.0),
    ]

    windows = calibration_windows(rows, _asset_type)

    assert len(windows[AssetType.STOCK]) == 1


def test_coverage_counts_only_rows_with_a_stored_interval() -> None:
    rows = [
        {"interval_low": 98.0, "interval_high": 102.0, "actual_price_after_12h": 101.0},
        {"interval_low": 98.0, "interval_high": 102.0, "actual_price_after_12h": 103.0},
        {"interval_low": None, "interval_high": None, "actual_price_after_12h": 100.0},
        {"interval_low": "98.0", "interval_high": "102.0", "actual_price_after_12h": "98.0"},
    ]

    assert interval_coverage(rows) == IntervalCoverage(hits=2, n=3)
