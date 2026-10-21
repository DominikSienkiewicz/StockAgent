"""TEST-12 (G65): bramka volatility jest symetryczna i monotoniczna.

Bramka decyduje, czy cykl w ogóle płaci za LLM, Alpha Vantage i embeddingi.
Spadek o x% kosztuje tyle samo co wzrost o x%, większy ruch nie może zamknąć
bramki otwartej dla mniejszego, a ostrzejszy próg nie może otworzyć bramki
zamkniętej przez łagodniejszy.
"""

from __future__ import annotations

from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from src.domain.asset import Asset, PriceDelta
from src.domain.value_objects import AssetType, Threshold

_fraction = st.decimals(min_value=Decimal("-5"), max_value=Decimal("5"), places=6)
_threshold = st.decimals(min_value=Decimal("0"), max_value=Decimal("1"), places=6)
_asset = st.builds(Asset, symbol=st.just("AAPL"), asset_type=st.sampled_from(AssetType))


def _opens(asset: Asset, fraction: Decimal, threshold: Decimal) -> bool:
    return asset.evaluate_volatility(PriceDelta(fraction), Threshold(threshold))


@given(asset=_asset, fraction=_fraction, threshold=_threshold, stale=st.booleans())
def test_gate_is_symmetric_in_the_sign_of_the_move(
    asset: Asset, fraction: Decimal, threshold: Decimal, stale: bool
) -> None:
    gate = Threshold(threshold)

    assert asset.evaluate_volatility(
        PriceDelta(fraction), gate, reference_stale=stale
    ) == asset.evaluate_volatility(PriceDelta(-fraction), gate, reference_stale=stale)


@given(asset=_asset, first=_fraction, second=_fraction, threshold=_threshold)
def test_gate_is_monotonic_in_the_size_of_the_move(
    asset: Asset, first: Decimal, second: Decimal, threshold: Decimal
) -> None:
    smaller, larger = sorted((abs(first), abs(second)))

    if _opens(asset, smaller, threshold):
        assert _opens(asset, larger, threshold)


@given(asset=_asset, fraction=_fraction, first=_threshold, second=_threshold)
def test_gate_is_monotonic_in_the_threshold(
    asset: Asset, fraction: Decimal, first: Decimal, second: Decimal
) -> None:
    looser, stricter = sorted((first, second))

    if _opens(asset, fraction, stricter):
        assert _opens(asset, fraction, looser)


@given(asset=_asset, fraction=_fraction, threshold=_threshold)
def test_stale_reference_keeps_the_gate_closed(
    asset: Asset, fraction: Decimal, threshold: Decimal
) -> None:
    assert not asset.evaluate_volatility(
        PriceDelta(fraction), Threshold(threshold), reference_stale=True
    )
