from datetime import UTC, datetime, timedelta

import pytest

from src.domain.training_labels import (
    MAX_LABEL_HORIZON,
    FeatureSnapshot,
    label_snapshots,
)

T0 = datetime(2026, 10, 1, 7, 30, tzinfo=UTC)


def _snapshot(
    symbol: str = "AAPL",
    as_of: datetime = T0,
    price: object = 100.0,
    *,
    gated: bool = False,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol=symbol,
        as_of=as_of,
        gated=gated,
        features={"price": price, "snapshot_delta": 0.01},
        feature_set_version=4,
    )


def _hours(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


class TestLabelSnapshots:
    """ML-52 (1): etykieta 12 h z `price_snapshots`, point-in-time, bez podglądu."""

    def test_label_is_the_first_price_at_or_after_the_horizon(self) -> None:
        series = {
            "AAPL": [(T0, 100.0), (_hours(6), 150.0), (_hours(24), 110.0), (_hours(48), 130.0)]
        }

        [labeled] = label_snapshots([_snapshot()], series)

        # Cena sprzed horyzontu (6 h) i późniejsze niż pierwsza dojrzała (48 h) nie wchodzą.
        assert labeled.target_return == pytest.approx(0.10)
        assert labeled.settled_at == _hours(24)
        assert labeled.snapshot.features["price"] == 100.0

    def test_snapshot_without_a_matured_price_has_no_label(self) -> None:
        series = {"AAPL": [(T0, 100.0), (_hours(11), 120.0)]}

        assert label_snapshots([_snapshot()], series) == []

    def test_label_after_a_pause_longer_than_the_max_horizon_is_dropped(self) -> None:
        # Pauza harmonogramu: „zwrot do następnego cyklu” z 10 dni to nie obserwacja cyklu.
        late = T0 + MAX_LABEL_HORIZON + timedelta(hours=1)
        series = {"AAPL": [(T0, 100.0), (late, 120.0)]}

        assert label_snapshots([_snapshot()], series) == []

    def test_label_uses_only_the_snapshot_symbol(self) -> None:
        series = {"BTC": [(_hours(24), 50_000.0)], "AAPL": [(_hours(30), 90.0)]}

        [labeled] = label_snapshots([_snapshot()], series)

        assert labeled.target_return == pytest.approx(-0.10)

    @pytest.mark.parametrize("price", [None, 0.0, -1.0, float("nan"), "100"])
    def test_snapshot_without_a_valid_price_has_no_label(self, price: object) -> None:
        series = {"AAPL": [(_hours(24), 110.0)]}

        assert label_snapshots([_snapshot(price=price)], series) == []

    def test_non_positive_label_prices_are_skipped(self) -> None:
        series = {"AAPL": [(_hours(13), 0.0), (_hours(14), 105.0)]}

        [labeled] = label_snapshots([_snapshot()], series)

        assert labeled.target_return == pytest.approx(0.05)

    def test_labels_follow_global_time_then_symbol(self) -> None:
        series = {
            "AAPL": [(_hours(30), 101.0), (_hours(60), 102.0)],
            "BTC": [(_hours(30), 101.0)],
        }
        snapshots = [
            _snapshot("BTC", T0),
            _snapshot("AAPL", _hours(24)),
            _snapshot("AAPL", T0),
        ]

        labeled = label_snapshots(snapshots, series)

        assert [(item.snapshot.symbol, item.snapshot.as_of) for item in labeled] == [
            ("AAPL", T0),
            ("BTC", T0),
            ("AAPL", _hours(24)),
        ]
