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
    [("BULLISH", "KUP"), ("BEARISH", "SPRZEDAJ"), ("SIDEWAYS", "WSTRZYMAJ")],
)
def test_model_signal_maps_the_trend_to_a_polish_action(trend: str, label: str) -> None:
    assert model_signal_label(trend) == label


@pytest.mark.parametrize("trend", [None, "", "UNKNOWN"])
def test_unknown_trend_is_a_hold_signal(trend: str | None) -> None:
    assert model_signal_label(trend) == "WSTRZYMAJ"


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
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
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
