"""WIA-35: trafność kierunkowa per symbol — „czy model rozumie TĘ spółkę”.

Globalny hit-rate nie mówi nic o konkretnej spółce, a decyzja dotyczy konkretnej
spółki. Definicja trafienia jest ta sama co w mailu i scorecardzie (G24,
`directional_hit_rate`): SIDEWAYS nie stawia zakładu, więc nie wchodzi do
mianownika. Poniżej minimalnej próby wynik jest oznaczony jako niewystarczający
— mała próba daje złudzenie umiejętności.
"""

from __future__ import annotations

import pytest

from src.domain.statistics import wilson_interval
from src.domain.symbol_track_record import MIN_SYMBOL_SAMPLE, symbol_track_record


def test_per_symbol_hits_exclude_sideways_and_respect_min_sample() -> None:
    calls = [
        ("NVDA", "BULLISH", True),
        ("NVDA", "BEARISH", True),
        ("NVDA", "BULLISH", True),
        ("NVDA", "BULLISH", False),
        ("NVDA", "SIDEWAYS", True),
        ("NVDA", "SIDEWAYS", False),
        ("AAPL", "BULLISH", True),
    ]

    records = symbol_track_record(calls)

    nvda = records["NVDA"]
    assert (nvda.hits, nvda.directional) == (3, 4)
    assert nvda.sideways == 2
    assert nvda.insufficient
    assert records["AAPL"].directional == 1


def test_min_sample_reached_gives_rate_and_wilson_interval() -> None:
    calls = [("MSFT", "BULLISH", index < 4) for index in range(MIN_SYMBOL_SAMPLE + 2)]

    record = symbol_track_record(calls)["MSFT"]

    assert MIN_SYMBOL_SAMPLE == 5
    assert not record.insufficient
    assert record.rate == pytest.approx(4 / 7)
    assert record.interval == pytest.approx(wilson_interval(4, 7))


def test_only_sideways_calls_are_insufficient_without_a_rate() -> None:
    record = symbol_track_record([("BTC", "SIDEWAYS", True)] * 6)["BTC"]

    assert record.directional == 0
    assert record.rate is None
    assert record.insufficient


def test_no_calls_no_records() -> None:
    assert symbol_track_record([]) == {}
