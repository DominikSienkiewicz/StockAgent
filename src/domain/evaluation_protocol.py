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


def _section(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Tabela TOML pod `key`; brak klucza → `KeyError`, zły typ → `ValueError`."""
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"[{key}] must be a TOML table, got {value!r}")
    return value


def _number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{key} must be a number, got {value!r}")
    return float(value)


def _names(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{key} must be a TOML array, got {value!r}")
    return tuple(str(item) for item in value)


def parse_evaluation_protocol(data: Mapping[str, object], sha256: str) -> EvaluationProtocol:
    """Protokół z danych TOML; brak sekcji albo klucza → `KeyError` (plik jest
    częścią repo, więc błąd ma zatrzymać przebieg, a nie dać pusty protokół)."""
    adopted_on = data["adopted_on"]
    if not isinstance(adopted_on, date):
        raise ValueError(f"adopted_on must be a TOML date, got {adopted_on!r}")
    hit_rule = _section(data, "hit_rule")
    baselines = _section(data, "baselines")
    metrics = _section(data, "metrics")
    significance = _section(data, "significance")
    universe = _section(data, "universe")
    return EvaluationProtocol(
        version=int(_number(data["version"], "version")),
        adopted_on=adopted_on,
        horizon_hours=int(_number(_section(data, "horizon")["hours"], "horizon.hours")),
        directional_dead_zone=Decimal(str(hit_rule["directional_dead_zone"])),
        sideways_band=Decimal(str(hit_rule["sideways_band"])),
        crypto_sideways_band=Decimal(str(hit_rule["crypto_sideways_band"])),
        sideways_in_hit_rate=bool(hit_rule["sideways_in_hit_rate"]),
        naive_baselines=_names(baselines["naive"], "baselines.naive"),
        benchmarks=_names(baselines["benchmarks"], "baselines.benchmarks"),
        primary_metrics=_names(metrics["primary"], "metrics.primary"),
        secondary_metrics=_names(metrics["secondary"], "metrics.secondary"),
        alpha=_number(significance["alpha"], "significance.alpha"),
        fdr_q=_number(significance["fdr_q"], "significance.fdr_q"),
        window_days=int(
            _number(_section(data, "window")["track_record_days"], "window.track_record_days")
        ),
        universe_source=str(universe["source"]),
        universe_selection=str(universe["selection"]),
        universe_selected=str(universe["selected"]),
        sha256=sha256,
    )
