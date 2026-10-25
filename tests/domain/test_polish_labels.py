"""Wspólny słownik polskich etykiet sygnałów (MAIL-27, MAIL-15).

Mail, lead i temat mówią jednym językiem: trend modelu i werdykt rady mają po
jednej polskiej etykiecie, trzymanej w domenie, żeby `digest_lead` (domena)
i renderery (application) nie rozjechały się słownikami.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import src.domain.polish_labels as polish_labels
from src.domain.polish_labels import (
    MODEL_SIGNAL_TITLE,
    council_recommendation_label,
    model_signal_label,
)


@pytest.mark.parametrize(
    ("trend", "label"),
    [("BULLISH", "KUP"), ("BEARISH", "SPRZEDAJ"), ("SIDEWAYS", "TRZYMAJ")],
)
def test_model_signal_maps_the_trend_to_a_polish_action(trend: str, label: str) -> None:
    assert model_signal_label(trend) == label


@pytest.mark.parametrize("trend", [None, "", "UNKNOWN"])
def test_unknown_trend_is_a_hold_signal(trend: str | None) -> None:
    assert model_signal_label(trend) == "TRZYMAJ"


def test_sideways_signal_and_council_hold_share_one_label() -> None:
    # Decyzja 2026-10-19: jedna etykieta ruchu bocznego w mailu, leadzie i temacie.
    assert model_signal_label("SIDEWAYS") == council_recommendation_label("HOLD")


@pytest.mark.parametrize(
    ("recommendation", "label"),
    [("BUY", "KUP"), ("SELL", "SPRZEDAJ"), ("HOLD", "TRZYMAJ")],
)
def test_council_recommendation_is_polish(recommendation: str, label: str) -> None:
    assert council_recommendation_label(recommendation) == label


def test_unknown_council_recommendation_is_passed_through() -> None:
    assert council_recommendation_label("ABSTAIN") == "ABSTAIN"


def test_model_signal_title_is_not_a_recommendation() -> None:
    assert MODEL_SIGNAL_TITLE == "Sygnał modelu"


def test_dictionary_imports_nothing_outside_the_domain() -> None:
    tree = ast.parse(Path(polish_labels.__file__).read_text(encoding="utf-8"))
    imported = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]

    outside_domain = [
        name for name in imported if name.startswith("src.") and not name.startswith("src.domain")
    ]
    assert outside_domain == []


@pytest.mark.parametrize(
    ("trend", "label"),
    [
        ("BULLISH", "prognoza wzrostu"),
        ("BEARISH", "prognoza spadku"),
        ("SIDEWAYS", "prognoza ruchu bocznego"),
    ],
)
def test_trend_forecast_reads_as_polish_noun_phrase(trend: str, label: str) -> None:
    # MAIL-15: „trafiona prognoza wzrostu” zamiast „trafiona predykcja BULLISH”.
    assert polish_labels.trend_forecast_label(trend) == label


def test_unknown_trend_forecast_keeps_the_raw_value_visible() -> None:
    assert polish_labels.trend_forecast_label("FLAT") == "prognoza FLAT"


@pytest.mark.parametrize(
    ("trend", "label"),
    [("BULLISH", "Wzrostowy"), ("BEARISH", "Spadkowy"), ("SIDEWAYS", "Boczny"), (None, "—")],
)
def test_trend_label_is_the_shared_polish_trend_name(trend: str | None, label: str) -> None:
    assert polish_labels.trend_label(trend) == label
