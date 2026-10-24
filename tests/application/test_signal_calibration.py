"""WIA-39: kalibracja kierunkowa osobno od SIDEWAYS.

„SIDEWAYS 0.8” trafia za darmo w spokojne dni; wrzucone do jednego kubełka
z „BULLISH 0.8” zawyża hit-rate, którym ranking kalibruje predykcje kierunkowe.
"""

from __future__ import annotations

from src.application.signal_calibration import (
    directional_calibration_curve,
    split_calibration_samples,
)


def _row(trend: str | None, confidence: float, correct: bool, stamp: str = "") -> dict[str, object]:
    row: dict[str, object] = {"confidence_score": confidence, "is_trend_correct": correct}
    if trend is not None:
        row["predicted_trend"] = trend
    if stamp:
        row["timestamp"] = stamp
    return row


def test_sideways_samples_do_not_enter_directional_calibration() -> None:
    buckets = directional_calibration_curve(
        [_row("SIDEWAYS", 0.9, True), _row("BULLISH", 0.9, False)]
    )

    assert buckets is not None
    (bucket,) = buckets
    assert bucket.count == 1
    assert bucket.hit_rate == 0.0


def test_split_keeps_sideways_in_their_own_sample() -> None:
    samples = split_calibration_samples(
        [
            _row("SIDEWAYS", 0.7, True),
            _row("BEARISH", 0.6, True),
            _row("BULLISH", 0.8, False),
        ]
    )

    assert samples.sideways == ((0.7, True),)
    assert samples.directional == [(0.6, True), (0.8, False)]


def test_row_without_trend_stays_directional() -> None:
    # Wiersz bez `predicted_trend` (odczyt sprzed WIA-39) — zachowanie jak dotąd.
    samples = split_calibration_samples([_row(None, 0.85, True)])

    assert samples.directional == [(0.85, True)]
    assert samples.sideways == ()


def test_incomplete_rows_are_skipped() -> None:
    samples = split_calibration_samples(
        [
            {"confidence_score": None, "is_trend_correct": True, "predicted_trend": "BULLISH"},
            {"confidence_score": 0.7, "is_trend_correct": None, "predicted_trend": "SIDEWAYS"},
        ]
    )

    assert samples.directional == []
    assert samples.sideways == ()


def test_directional_samples_are_grouped_by_utc_day() -> None:
    samples = split_calibration_samples(
        [
            _row("BULLISH", 0.6, True, "2026-10-19T05:30:00+00:00"),
            _row("BEARISH", 0.7, False, "2026-10-18T05:30:00+00:00"),
            _row("BULLISH", 0.8, True, "2026-10-19T05:31:00+00:00"),
            _row("SIDEWAYS", 0.9, True, "2026-10-19T05:32:00+00:00"),
        ]
    )

    assert samples.directional_by_cycle == (
        ((0.7, False),),
        ((0.6, True), (0.8, True)),
    )


def test_rows_without_timestamp_share_one_cycle() -> None:
    # Brak znacznika zaniża n_eff (ostrożnie), zamiast udawać niezależne próby.
    samples = split_calibration_samples([_row("BULLISH", 0.6, True), _row("BEARISH", 0.7, True)])

    assert samples.directional_by_cycle == (((0.6, True), (0.7, True)),)


def test_no_directional_samples_give_no_curve() -> None:
    assert directional_calibration_curve([_row("SIDEWAYS", 0.9, True)]) is None
