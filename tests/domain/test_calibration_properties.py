"""TEST-12 (G65): kalibracja pewności zostaje w [0, 1] dla każdego wejścia.

Pewność z bazy nie zawsze jest w skali jednostkowej — LLM dryfujący na skalę
0–100 zapisywał np. 85 przed `clamp_unit`. Wynik kalibracji ląduje w kolumnie
`calibration_score`, więc kontrakt [0, 1] z docstringu musi trzymać i wtedy.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from src.domain.calibration_curve import calibrated_confidence, reliability_curve
from src.domain.prediction import calibration_error, calibration_score

_any_confidence = st.floats(allow_nan=False)
_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@given(confidence=_any_confidence, correct=st.booleans())
def test_calibration_score_within_unit_interval(confidence: float, correct: bool) -> None:
    assert 0.0 <= calibration_score(confidence, correct) <= 1.0


@given(samples=st.lists(st.tuples(_any_confidence, st.booleans()), max_size=30))
def test_calibration_error_within_unit_interval(samples: list[tuple[float, bool]]) -> None:
    assert 0.0 <= calibration_error(samples) <= 1.0


@given(
    raw=_unit,
    samples=st.lists(st.tuples(_unit, st.booleans()), max_size=60),
    min_count=st.integers(min_value=-5, max_value=50),
)
def test_calibrated_confidence_stays_in_unit_interval(
    raw: float, samples: list[tuple[float, bool]], min_count: int
) -> None:
    buckets = reliability_curve(samples)

    assert 0.0 <= calibrated_confidence(raw, buckets, min_count) <= 1.0
