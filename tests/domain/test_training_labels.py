from datetime import UTC, datetime, timedelta

import pytest

from src.domain.training_labels import (
    MAX_LABEL_HORIZON,
    FeatureSnapshot,
    LabeledSnapshot,
    ReturnDistribution,
    label_snapshots,
    selection_report,
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


def _labeled(target_return: float, *, gated: bool) -> LabeledSnapshot:
    return LabeledSnapshot(
        snapshot=_snapshot(gated=gated), target_return=target_return, settled_at=_hours(24)
    )


class TestSelectionReport:
    """ML-52 (3): bramka wybiera dni z ruchem — raport porównuje rozkład zwrotów
    cykli za bramką (gated) i pod nią, żeby widać było skalę selekcji próbki."""

    def test_splits_returns_into_gated_and_non_gated(self) -> None:
        labeled = [
            _labeled(0.04, gated=True),
            _labeled(-0.02, gated=True),
            _labeled(0.01, gated=False),
            _labeled(-0.01, gated=False),
            _labeled(0.0, gated=False),
        ]

        report = selection_report(labeled)

        assert report.gated.n == 2
        assert report.gated.mean == pytest.approx(0.01)
        assert report.gated.mean_abs == pytest.approx(0.03)
        assert report.non_gated.n == 3
        assert report.non_gated.mean_abs == pytest.approx(0.02 / 3)
        assert report.non_gated.median == pytest.approx(0.0)
        assert report.abs_return_ratio == pytest.approx(0.03 / (0.02 / 3))

    def test_dispersion_and_tails_of_one_group(self) -> None:
        returns = [i / 100 for i in range(-5, 6)]

        report = selection_report([_labeled(r, gated=True) for r in returns])

        assert report.gated.std == pytest.approx(0.033166, abs=1e-6)
        assert report.gated.p10 == pytest.approx(-0.04)
        assert report.gated.p90 == pytest.approx(0.04)

    def test_empty_group_has_no_statistics_and_no_ratio(self) -> None:
        report = selection_report([_labeled(0.02, gated=True)])

        assert report.non_gated == ReturnDistribution(n=0)
        assert report.gated.std is None
        assert report.abs_return_ratio is None
