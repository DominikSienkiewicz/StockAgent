"""Payloady dostawców z `tests/fixtures/providers/` i porównanie ich kształtu (TEST-16).

Adaptery testowały ręcznie pisane payloady, więc zmiana formatu u dostawcy (prefiks
`xsl…/` w `primaryDocument` EDGAR, pole `Note` w Alpha Vantage) przechodziła przez
zielone testy. Pliki w katalogu fixture'ów mają kształt prawdziwych odpowiedzi
(klucze, typy, zagnieżdżenie), przycięty do kilku elementów, bez kluczy API i bez
treści newsów. Testy jednostkowe adapterów parsują je, a cotygodniowy test na żywym
API porównuje ich kształt ze świeżą odpowiedzią.

Kształt JSON-a to zbiór par (ścieżka, typ JSON). Ścieżka listy kończy się `[]`
i zbiera wszystkie elementy, więc pole `null` w jednym artykule i tekst w drugim
dają dwie pary pod tą samą ścieżką. Liczby całkowite i zmiennoprzecinkowe to jeden
typ `number` — dostawca, który zwróci `100` zamiast `100.0`, nie zmienia kształtu.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

PROVIDERS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "providers"

# Kowariantne kontenery: literał `dict[str, list[...]]` w teście też jest JsonValue.
type JsonValue = None | bool | int | float | str | Sequence[JsonValue] | Mapping[str, JsonValue]
type ShapeEntry = tuple[str, str]


def fixture_path(name: str) -> Path:
    return PROVIDERS_DIR / name


def load_json(name: str) -> JsonValue:
    loaded: JsonValue = json.loads(fixture_path(name).read_text(encoding="utf-8"))
    return loaded


def load_text(name: str) -> str:
    return fixture_path(name).read_text(encoding="utf-8")


def json_type(value: JsonValue) -> str:
    """Typ JSON wartości; `bool` przed liczbą, bo w Pythonie `True` jest intem."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    return "array"


def json_shape(value: JsonValue, path: str = "$") -> frozenset[ShapeEntry]:
    """Pary (ścieżka, typ) wartości i wszystkich jej potomków."""
    entries = {(path, json_type(value))}
    if isinstance(value, Mapping):
        for key, child in value.items():
            entries |= json_shape(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for child in value:
            entries |= json_shape(child, f"{path}[]")
    return frozenset(entries)


def shape_drift(expected: JsonValue, actual: JsonValue) -> list[str]:
    """Pary z kształtu `expected`, których brak w `actual`, z typami, jakie `actual`
    ma pod tą ścieżką. Dodatkowe pola po stronie `actual` nie są dryfem — dostawca
    może dopisywać pola, adapter czyta tylko te z fixture'a."""
    actual_shape = json_shape(actual)
    actual_types: dict[str, set[str]] = {}
    for path, kind in actual_shape:
        actual_types.setdefault(path, set()).add(kind)
    return sorted(
        f"{path}: {kind} (jest: {', '.join(sorted(actual_types.get(path, set()))) or 'brak'})"
        for path, kind in json_shape(expected) - actual_shape
    )


def xml_element_paths(document: str) -> frozenset[str]:
    """Ścieżki elementów XML z lokalnych nazw (bez namespace'u), od korzenia."""
    root = ET.fromstring(document)
    paths: set[str] = set()

    def walk(element: ET.Element, prefix: str) -> None:
        path = f"{prefix}/{element.tag.rsplit('}', 1)[-1]}"
        paths.add(path)
        for child in element:
            walk(child, path)

    walk(root, "")
    return frozenset(paths)
