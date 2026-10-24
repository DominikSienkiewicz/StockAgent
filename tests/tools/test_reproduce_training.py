"""ML-22: `reproduce_training` — czy odcisk zbioru i ziarno z sidecara dają zapisany booster.

Czyste funkcje: parsowanie sidecara, wybór wierszy z jego zakresu dat, odcisk próbki
tą samą funkcją domeny, co adapter, i werdykt. Odczyt bazy, refit i kod wyjścia
należą do `main_tools.py reproduce-training` (ARCH-39).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.application.use_cases.train_model import TrainingSet
from src.domain.training_provenance import DatasetDescription, dataset_fingerprint
from src.tools import reproduce_training as tool

T0 = datetime(2026, 10, 1, 7, 30, tzinfo=UTC)


def _training_set(n: int = 5) -> TrainingSet:
    return TrainingSet(
        features=[
            {"price_delta": 0.001 * i, "llm_trend_signal": float(i % 3 - 1)} for i in range(n)
        ],
        target=[0.01 * i for i in range(n)],
        timestamps=[T0 + timedelta(hours=12 * i) for i in range(n)],
        dataset=DatasetDescription(n_per_symbol={"AAPL": n}),
    )


def _sidecar(**overrides: object) -> str:
    record: dict[str, object] = {
        "model_file": "price_predictor.ubj",
        "weights_sha256": "ef" * 32,
        "target_unit": "return",
        "n_samples": 3,
        "dataset_sha256": "ab" * 32,
        "data_start": T0.isoformat(),
        "data_end": (T0 + timedelta(hours=24)).isoformat(),
        "n_per_symbol": {"AAPL": 3},
        "training_params": {"max_depth": 4, "random_state": 0},
        "library_version": "xgboost 3.2.0",
        "feature_set_version": 4,
        "git_sha": "abc1234",
    }
    record.update(overrides)
    return json.dumps(record)


class TestParseSidecar:
    def test_reads_the_fields_the_reproduction_needs(self) -> None:
        sidecar = tool.parse_sidecar(_sidecar())

        assert sidecar.dataset_sha256 == "ab" * 32
        assert sidecar.data_start == T0
        assert sidecar.data_end == T0 + timedelta(hours=24)
        assert sidecar.params == {"max_depth": 4, "random_state": 0}
        assert sidecar.library_version == "xgboost 3.2.0"
        assert sidecar.git_sha == "abc1234"

    def test_open_date_range_stays_none(self) -> None:
        sidecar = tool.parse_sidecar(_sidecar(data_start=None, data_end=None))

        assert sidecar.data_start is None
        assert sidecar.data_end is None

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            "[]",
            _sidecar(dataset_sha256=None),
            _sidecar(training_params="max_depth=4"),
            _sidecar(data_start="yesterday"),
        ],
    )
    def test_rejects_a_sidecar_it_cannot_reproduce_from(self, text: str) -> None:
        with pytest.raises(tool.SidecarError):
            tool.parse_sidecar(text)


class TestRowsInRange:
    def test_keeps_rows_inside_the_sidecar_range_inclusive(self) -> None:
        sidecar = tool.parse_sidecar(_sidecar())

        kept = tool.rows_in_range(_training_set(), sidecar)

        assert kept.timestamps == [T0, T0 + timedelta(hours=12), T0 + timedelta(hours=24)]
        assert kept.target == [0.0, 0.01, 0.02]
        assert len(kept.features) == 3

    def test_open_range_keeps_every_row(self) -> None:
        sidecar = tool.parse_sidecar(_sidecar(data_start=None, data_end=None))

        assert tool.rows_in_range(_training_set(), sidecar) == _training_set()


def test_fingerprint_is_the_domain_fingerprint_of_the_rows() -> None:
    training_set = _training_set()

    assert tool.fingerprint_of(training_set) == dataset_fingerprint(
        zip(training_set.timestamps, training_set.target, training_set.features, strict=True)
    )


class TestAssessReproduction:
    def _assess(
        self, *, fingerprint: str, refit: str | None, library: str = "xgboost 3.2.0"
    ) -> tool.ReproductionReport:
        return tool.assess_reproduction(
            tool.parse_sidecar(_sidecar()),
            fingerprint=fingerprint,
            n_rows=3,
            refit_dump=refit,
            saved_dump="trees",
            library_version=library,
        )

    def test_same_fingerprint_and_same_dump_is_reproduced(self) -> None:
        report = self._assess(fingerprint="ab" * 32, refit="trees")

        assert report.ok is True
        assert "REPRODUCED" in tool.format_reproduction(report)

    def test_another_dataset_is_not_reproduced(self) -> None:
        report = self._assess(fingerprint="cd" * 32, refit=None)

        assert report.ok is False
        text = tool.format_reproduction(report)
        assert "dataset differs" in text
        assert "3 rows" in text

    def test_same_dataset_with_another_booster_is_not_reproduced(self) -> None:
        report = self._assess(fingerprint="ab" * 32, refit="other trees")

        assert report.ok is False
        assert "booster differs" in tool.format_reproduction(report)

    def test_another_library_version_is_named(self) -> None:
        report = self._assess(fingerprint="ab" * 32, refit="other trees", library="xgboost 3.3.0")

        assert "xgboost 3.2.0" in tool.format_reproduction(report)
        assert "xgboost 3.3.0" in tool.format_reproduction(report)
