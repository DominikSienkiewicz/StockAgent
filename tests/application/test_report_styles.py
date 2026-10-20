"""Arkusz stylów maila (MAIL-06 c): ton kierunku w klasach ma te same kolory co
`delta_color` / `trend_color` z palety (ARCH-28) — klasa nie może rozjechać się
z kolorem, który reszta maila wstawia inline."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.application.report_formatting import PALETTE, delta_color, trend_color
from src.application.report_styles import (
    COUNCIL_TONE,
    EMAIL_STYLESHEET,
    delta_tone,
    fill_tone,
    trend_tone,
)


@pytest.mark.parametrize("delta", [Decimal("0.01"), Decimal("-0.01"), Decimal("0"), None])
def test_delta_tone_class_carries_the_delta_color(delta: Decimal | None) -> None:
    assert f".{delta_tone(delta)}{{color:{delta_color(delta)}}}" in EMAIL_STYLESHEET
    assert f".{fill_tone(delta)}{{background:{delta_color(delta)}}}" in EMAIL_STYLESHEET


@pytest.mark.parametrize("trend", ["BULLISH", "BEARISH", "SIDEWAYS", None])
def test_trend_tone_class_carries_the_trend_color(trend: str | None) -> None:
    assert f".{trend_tone(trend)}{{color:{trend_color(trend)}}}" in EMAIL_STYLESHEET


def test_council_hold_reads_as_caution() -> None:
    assert f".{COUNCIL_TONE['HOLD']}{{color:{PALETTE.caution}}}" in EMAIL_STYLESHEET
