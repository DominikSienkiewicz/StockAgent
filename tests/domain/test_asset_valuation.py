from datetime import UTC, datetime

import pytest

from src.domain.asset import Asset
from src.domain.value_objects import AssetType, Fundamentals, ValuationVerdict

_NOW = datetime(2026, 5, 17, tzinfo=UTC)


def _f(
    trailing: float | None = 25.0,
    forward: float | None = 20.0,
    peg: float | None = 1.2,
    growth: float | None = 0.15,
) -> Fundamentals:
    return Fundamentals(
        trailing_pe=trailing,
        forward_pe=forward,
        peg_ratio=peg,
        eps_growth_yoy=growth,
        fetched_at=_NOW,
    )


def test_etf_returns_unknown() -> None:
    asset = Asset(symbol="VOO", asset_type=AssetType.ETF)
    assert asset.evaluate_valuation(_f()) is ValuationVerdict.UNKNOWN


def test_crypto_returns_unknown() -> None:
    # Krypto nie ma EPS/P/E — wycena fundamentalna nie ma sensu, jak dla ETF.
    asset = Asset(symbol="BTC", asset_type=AssetType.CRYPTO)
    assert asset.evaluate_valuation(_f()) is ValuationVerdict.UNKNOWN


def test_none_fundamentals_returns_unknown() -> None:
    asset = Asset(symbol="AAPL")
    assert asset.evaluate_valuation(None) is ValuationVerdict.UNKNOWN


def test_missing_peg_returns_unknown() -> None:
    asset = Asset(symbol="AAPL")
    assert asset.evaluate_valuation(_f(peg=None)) is ValuationVerdict.UNKNOWN


@pytest.mark.parametrize(
    "peg,trailing,forward,growth,expected",
    [
        (0.8, 25.0, 20.0, 0.20, ValuationVerdict.UNDERVALUED),
        (0.8, 20.0, 25.0, 0.20, ValuationVerdict.FAIR),
        (2.5, 30.0, 28.0, 0.10, ValuationVerdict.OVERVALUED),
        (1.5, 35.0, 32.0, 0.05, ValuationVerdict.OVERVALUED),
        (1.5, 22.0, 20.0, 0.12, ValuationVerdict.FAIR),
    ],
)
def test_verdict_heuristic(
    peg: float,
    trailing: float,
    forward: float,
    growth: float,
    expected: ValuationVerdict,
) -> None:
    asset = Asset(symbol="AAPL")
    verdict = asset.evaluate_valuation(
        _f(trailing=trailing, forward=forward, peg=peg, growth=growth)
    )
    assert verdict is expected


def test_default_asset_type_is_stock() -> None:
    assert Asset(symbol="AAPL").asset_type is AssetType.STOCK
