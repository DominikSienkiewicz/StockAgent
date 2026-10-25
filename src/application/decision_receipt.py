"""Kwit decyzyjny predykcji (#13, WIA-20) — wersje schematu i czytnik.

Kwit żyje w JSONB `prediction_logs.decision_receipts` miesiącami po zapisie,
a JSONB nie ma migracji kolumn, więc wersję niesie pole `schema_version`:

- v1 — analogi RAG, atrybucje cech, próg efektywny;
- v2 — v1 plus płaskie pola przepisu (`PredictionRecipe`) i `recipe_version`.
  WIA-66 dołoży w v2 opcjonalny klucz `provenance` bez kolejnego bumpu.
  EXP-05 dokłada opcjonalny klucz `experiment_arms` (`{id: ramię}`) — tylko gdy
  w cyklu działał jakiś eksperyment. FIN-21 dokłada opcjonalny klucz `model_used`
  (`dostawca:model`) — tylko gdy odpowiedź przyszła z kaskady routingu modeli.

Każda wersja jest nadzbiorem poprzedniej, a czytnik toleruje brak każdego
opcjonalnego klucza. Nieznanej wersji nie zgaduje — zwraca None.

WIA-56: kwit przestaje być write-only — `receipt_line_pl` skraca go do jednej
linii pod chybieniem w mailu, a `src/tools/explain.py` pokazuje go w całości.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from src.application.report_formatting import number, pct
from src.domain.recipe import RECIPE_FIELDS

RECEIPT_SCHEMA_V1 = 1
RECEIPT_SCHEMA_V2 = 2
SUPPORTED_RECEIPT_SCHEMAS = frozenset({RECEIPT_SCHEMA_V1, RECEIPT_SCHEMA_V2})
# Analogi w linii maila: tyle najbliższych, krótkie id jak skrót commita.
_LINE_ANALOGS = 3
_SHORT_ID = 8


@dataclass(frozen=True)
class DecisionReceipt:
    """Odczytany kwit. `recipe` jest pusty dla v1 i niepełny, gdy kwit v2 nie
    niesie któregoś pola przepisu."""

    schema_version: int
    effective_threshold: str | None = None
    precedents: tuple[Mapping[str, Any], ...] = ()
    feature_attribution: tuple[Mapping[str, Any], ...] = ()
    recipe: Mapping[str, str] = field(default_factory=dict)
    experiment_arms: Mapping[str, str] = field(default_factory=dict)
    model_used: str | None = None

    @property
    def recipe_version(self) -> str | None:
        return self.recipe.get("recipe_version")


def read_decision_receipt(payload: object) -> DecisionReceipt | None:
    """Kwit z JSONB; None, gdy to nie kwit albo wersja schematu jest nieznana."""
    if not isinstance(payload, Mapping):
        return None
    schema_version = payload.get("schema_version")
    # `type is int`, bo `True in {1, 2}` — bool nie jest wersją schematu.
    if type(schema_version) is not int or schema_version not in SUPPORTED_RECEIPT_SCHEMAS:
        return None
    threshold = payload.get("effective_threshold")
    return DecisionReceipt(
        schema_version=schema_version,
        effective_threshold=None if threshold is None else str(threshold),
        precedents=_mappings(payload.get("precedents")),
        feature_attribution=_mappings(payload.get("feature_attribution")),
        recipe=_recipe_fields(payload) if schema_version >= RECEIPT_SCHEMA_V2 else {},
        experiment_arms=(
            _experiment_arms(payload.get("experiment_arms"))
            if schema_version >= RECEIPT_SCHEMA_V2
            else {}
        ),
        model_used=(
            _model_used(payload.get("model_used"))
            if schema_version >= RECEIPT_SCHEMA_V2
            else None
        ),
    )


def effective_threshold_pct(receipt: DecisionReceipt) -> str | None:
    """Próg efektywny jako procent po polsku („3,0%”); brak albo śmieć → None."""
    if receipt.effective_threshold is None:
        return None
    try:
        return pct(Decimal(receipt.effective_threshold), decimals=1)
    except InvalidOperation:
        return None


def similarity_label(precedent: Mapping[str, Any]) -> str | None:
    """Podobieństwo analogu („0,91”); brak albo nieliczbowe → None."""
    value = precedent.get("similarity")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return number(value, decimals=2)


def receipt_line_pl(receipt: DecisionReceipt) -> str:
    """WIA-56: jedna linia kwitu pod chybieniem, np. „stała na analogach
    #3f2a1b9c (0,91), #77aa0e12 (0,87) przy progu 3,0%”.

    Analog bez `id` (zapis sprzed #13) nie da się wskazać, więc go pomijamy."""
    analogs: list[str] = []
    for precedent in receipt.precedents:
        analog_id = precedent.get("id")
        if not analog_id:
            continue
        similarity = similarity_label(precedent)
        suffix = f" ({similarity})" if similarity is not None else ""
        analogs.append(f"#{str(analog_id)[:_SHORT_ID]}{suffix}")
    threshold = effective_threshold_pct(receipt)
    if not analogs:
        return "bez analogów RAG" + (f", przy progu {threshold}" if threshold else "")
    line = "stała na analogach " + ", ".join(analogs[:_LINE_ANALOGS])
    return line + (f" przy progu {threshold}" if threshold else "")


def _mappings(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _recipe_fields(payload: Mapping[str, Any]) -> dict[str, str]:
    keys = (*RECIPE_FIELDS, "recipe_version")
    return {key: str(payload[key]) for key in keys if payload.get(key) is not None}


def _experiment_arms(value: object) -> dict[str, str]:
    """Ramiona `{id: ramię}`; wpis o nietekstowej wartości pomijamy."""
    if not isinstance(value, Mapping):
        return {}
    return {
        str(experiment_id): arm
        for experiment_id, arm in value.items()
        if isinstance(arm, str)
    }


def _model_used(value: object) -> str | None:
    """Model z routingu (FIN-21); brak albo nietekstowa wartość → None."""
    return value if isinstance(value, str) and value else None
