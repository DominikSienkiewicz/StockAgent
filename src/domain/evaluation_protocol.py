"""Protokół ewaluacji (WIA-42) — maszynowa wersja reguł track recordu.

`evaluation_protocol.toml` w roocie repo zamraża PRZED wynikami: horyzont, regułę
trafności (pasma SIDEWAYS, martwa strefa kierunku), baseline'y, metryki główne,
α, okno, datę przyjęcia i uniwersum. sha256 bajtów pliku wchodzi do przepisu
predykcji i do commitmentu v2, więc metryk nie da się wybrać po zobaczeniu
wyników (HARKing).

Domena nie czyta plików: `parse_evaluation_protocol` dostaje gotowe dane TOML
i hash policzony przez DI (`src/config.py`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class EvaluationProtocol:
    """Jedna wersja protokołu ewaluacji razem z hashem pliku, z którego pochodzi.

    Pola `sideways_band` / `crypto_sideways_band` / `directional_dead_zone` są
    ułamkami ceny przy predykcji; `naive_baselines`, `benchmarks` i metryki to
    stabilne identyfikatory (etykiety dla strony publicznej nadaje aplikacja).
    """

    version: int
    adopted_on: date
    horizon_hours: int
    directional_dead_zone: Decimal
    sideways_band: Decimal
    crypto_sideways_band: Decimal
    sideways_in_hit_rate: bool
    naive_baselines: tuple[str, ...]
    benchmarks: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    alpha: float
    fdr_q: float
    window_days: int
    universe_source: str
    universe_selection: str
    universe_selected: str
    sha256: str


def parse_evaluation_protocol(data: Mapping[str, Any], sha256: str) -> EvaluationProtocol:
    """Protokół z danych TOML; brak sekcji albo klucza → `KeyError` (plik jest
    częścią repo, więc błąd ma zatrzymać przebieg, a nie dać pusty protokół)."""
    adopted_on = data["adopted_on"]
    if not isinstance(adopted_on, date):
        raise ValueError(f"adopted_on must be a TOML date, got {adopted_on!r}")
    hit_rule = data["hit_rule"]
    return EvaluationProtocol(
        version=int(data["version"]),
        adopted_on=adopted_on,
        horizon_hours=int(data["horizon"]["hours"]),
        directional_dead_zone=Decimal(str(hit_rule["directional_dead_zone"])),
        sideways_band=Decimal(str(hit_rule["sideways_band"])),
        crypto_sideways_band=Decimal(str(hit_rule["crypto_sideways_band"])),
        sideways_in_hit_rate=bool(hit_rule["sideways_in_hit_rate"]),
        naive_baselines=tuple(data["baselines"]["naive"]),
        benchmarks=tuple(data["baselines"]["benchmarks"]),
        primary_metrics=tuple(data["metrics"]["primary"]),
        secondary_metrics=tuple(data["metrics"]["secondary"]),
        alpha=float(data["significance"]["alpha"]),
        fdr_q=float(data["significance"]["fdr_q"]),
        window_days=int(data["window"]["track_record_days"]),
        universe_source=str(data["universe"]["source"]),
        universe_selection=str(data["universe"]["selection"]),
        universe_selected=str(data["universe"]["selected"]),
        sha256=sha256,
    )
