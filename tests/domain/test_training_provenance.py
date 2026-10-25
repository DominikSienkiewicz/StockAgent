"""ML-22: odtwarzalny trening — fingerprint zbioru i proweniencja w karcie modelu."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone

from src.domain.backtest import LabelOverlap
from src.domain.training_provenance import (
    PROVENANCE_RECORD_KEYS,
    DatasetDescription,
    TrainingProvenance,
    dataset_fingerprint,
)

T0 = datetime(2026, 10, 1, 7, 30, tzinfo=UTC)


def _rows() -> list[tuple[datetime | None, float, dict[str, float | None]]]:
    return [
        (T0, 0.01, {"price_delta": 0.002, "llm_trend_signal": 1.0}),
        (T0 + timedelta(hours=12), -0.02, {"price_delta": -0.004, "llm_trend_signal": -1.0}),
        (T0 + timedelta(days=1), 0.0, {"price_delta": None, "llm_trend_signal": 0.0}),
    ]


class TestDatasetFingerprint:
    def test_is_a_sha256_hex_digest(self) -> None:
        digest = dataset_fingerprint(_rows())

        assert len(digest) == 64
        assert int(digest, 16) >= 0

    def test_does_not_depend_on_the_row_order(self) -> None:
        rows = _rows()

        assert dataset_fingerprint(rows) == dataset_fingerprint(rows[::-1])

    def test_does_not_depend_on_the_feature_key_order(self) -> None:
        rows = _rows()
        reordered = [
            (stamp, target, dict(reversed(features.items()))) for stamp, target, features in rows
        ]

        assert dataset_fingerprint(rows) == dataset_fingerprint(reordered)

    def test_changes_with_a_target(self) -> None:
        rows = _rows()
        changed = [*rows[:2], (rows[2][0], 1e-12, rows[2][2])]

        assert dataset_fingerprint(rows) != dataset_fingerprint(changed)

    def test_changes_with_a_timestamp(self) -> None:
        rows = _rows()
        shifted = [(T0 + timedelta(seconds=1), *rows[0][1:]), *rows[1:]]

        assert dataset_fingerprint(rows) != dataset_fingerprint(shifted)

    def test_changes_with_a_duplicated_row(self) -> None:
        rows = _rows()

        assert dataset_fingerprint(rows) != dataset_fingerprint([*rows, rows[0]])

    def test_same_moment_in_another_timezone_is_the_same_row(self) -> None:
        warsaw = timezone(timedelta(hours=2))
        rows = _rows()
        local = [(T0.astimezone(warsaw), *rows[0][1:]), *rows[1:]]

        assert dataset_fingerprint(rows) == dataset_fingerprint(local)

    def test_missing_and_nan_values_are_the_same_gap(self) -> None:
        with_none = [(None, 0.5, {"a": None})]
        with_nan = [(None, 0.5, {"a": math.nan})]

        assert dataset_fingerprint(with_none) == dataset_fingerprint(with_nan)

    def test_integer_and_float_of_the_same_value_are_the_same_row(self) -> None:
        assert dataset_fingerprint([(None, 1, {"a": 2})]) == dataset_fingerprint(
            [(None, 1.0, {"a": 2.0})]
        )


class TestTrainingProvenance:
    def _provenance(self, dataset: DatasetDescription) -> TrainingProvenance:
        return TrainingProvenance(
            dataset_sha256="ab" * 32,
            data_start=T0,
            data_end=T0 + timedelta(days=3),
            params={"max_depth": 4, "random_state": 0},
            library_version="xgboost 3.2.0",
            dataset=dataset,
        )

    def test_record_carries_every_column_of_the_model_card(self) -> None:
        record = self._provenance(
            DatasetDescription(
                n_per_symbol={"MSFT": 2, "AAPL": 3}, feature_set_version=4, git_sha="abc1234"
            )
        ).as_record()

        assert record == {
            "dataset_sha256": "ab" * 32,
            "data_start": "2026-10-01T07:30:00+00:00",
            "data_end": "2026-10-04T07:30:00+00:00",
            "n_per_symbol": {"AAPL": 3, "MSFT": 2},
            "training_params": {"max_depth": 4, "random_state": 0},
            "library_version": "xgboost 3.2.0",
            "feature_set_version": 4,
            "git_sha": "abc1234",
            "label_overlap_count": None,
            "label_overlap_measured": None,
        }
        assert tuple(record) == PROVENANCE_RECORD_KEYS

    def test_record_carries_the_label_overlap_of_the_sample(self) -> None:
        # ML-25: nakładanie etykiet zostaje w karcie, nie tylko w logu GHA.
        record = self._provenance(
            DatasetDescription(label_overlap=LabelOverlap(measured=49, overlapping=10))
        ).as_record()

        assert record["label_overlap_count"] == 10
        assert record["label_overlap_measured"] == 49

    def test_record_sorts_symbols_so_the_sidecar_is_stable(self) -> None:
        record = self._provenance(DatasetDescription(n_per_symbol={"MSFT": 2, "AAPL": 3}))

        n_per_symbol = record.as_record()["n_per_symbol"]
        assert isinstance(n_per_symbol, dict)
        assert list(n_per_symbol) == ["AAPL", "MSFT"]

    def test_unknown_parts_stay_none(self) -> None:
        provenance = TrainingProvenance(
            dataset_sha256="cd" * 32,
            data_start=None,
            data_end=None,
            params={},
            library_version="xgboost 3.2.0",
        )

        record = provenance.as_record()

        assert record["data_start"] is None
        assert record["data_end"] is None
        assert record["n_per_symbol"] is None
        assert record["feature_set_version"] is None
        assert record["git_sha"] is None

    def test_naive_moments_are_read_as_utc(self) -> None:
        provenance = TrainingProvenance(
            dataset_sha256="cd" * 32,
            data_start=datetime(2026, 10, 1, 7, 30),
            data_end=datetime(2026, 10, 2, 7, 30),
            params={},
            library_version="xgboost 3.2.0",
        )

        assert provenance.as_record()["data_start"] == "2026-10-01T07:30:00+00:00"
