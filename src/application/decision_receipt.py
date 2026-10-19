"""Kwit decyzyjny predykcji (#13, WIA-20) — wersje schematu i czytnik.

Kwit żyje w JSONB `prediction_logs.decision_receipts` miesiącami po zapisie,
a JSONB nie ma migracji kolumn, więc wersję niesie pole `schema_version`:

- v1 — analogi RAG, atrybucje cech, próg efektywny;
- v2 — v1 plus płaskie pola przepisu (`PredictionRecipe`) i `recipe_version`.
  WIA-66 dołoży w v2 opcjonalny klucz `provenance` bez kolejnego bumpu.

Każda wersja jest nadzbiorem poprzedniej, a czytnik toleruje brak każdego
opcjonalnego klucza. Nieznanej wersji nie zgaduje — zwraca None.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.domain.recipe import RECIPE_FIELDS

RECEIPT_SCHEMA_V1 = 1
RECEIPT_SCHEMA_V2 = 2
_SUPPORTED_SCHEMAS = frozenset({RECEIPT_SCHEMA_V1, RECEIPT_SCHEMA_V2})


@dataclass(frozen=True)
class DecisionReceipt:
    """Odczytany kwit. `recipe` jest pusty dla v1 i niepełny, gdy kwit v2 nie
    niesie któregoś pola przepisu."""

    schema_version: int
    effective_threshold: str | None = None
    precedents: tuple[Mapping[str, Any], ...] = ()
    feature_attribution: tuple[Mapping[str, Any], ...] = ()
    recipe: Mapping[str, str] = field(default_factory=dict)

    @property
    def recipe_version(self) -> str | None:
        return self.recipe.get("recipe_version")


def read_decision_receipt(payload: object) -> DecisionReceipt | None:
    """Kwit z JSONB; None, gdy to nie kwit albo wersja schematu jest nieznana."""
    if not isinstance(payload, Mapping):
        return None
    schema_version = payload.get("schema_version")
    # `type is int`, bo `True in {1, 2}` — bool nie jest wersją schematu.
    if type(schema_version) is not int or schema_version not in _SUPPORTED_SCHEMAS:
        return None
    threshold = payload.get("effective_threshold")
    return DecisionReceipt(
        schema_version=schema_version,
        effective_threshold=None if threshold is None else str(threshold),
        precedents=_mappings(payload.get("precedents")),
        feature_attribution=_mappings(payload.get("feature_attribution")),
        recipe=_recipe_fields(payload) if schema_version >= RECEIPT_SCHEMA_V2 else {},
    )


def _mappings(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _recipe_fields(payload: Mapping[str, Any]) -> dict[str, str]:
    keys = (*RECIPE_FIELDS, "recipe_version")
    return {key: str(payload[key]) for key in keys if payload.get(key) is not None}
