"""Loader plików JSON z personami członków rady doradczej.

Każdy persona = jeden plik JSON `{"name": str, "style": str}` w katalogu
`data/council_personas/` (konfigurowalne przez `COUNCIL_PERSONAS_DIR` env).

Dodanie / usunięcie członka rady = dodanie / usunięcie pliku JSON. Nie wymaga
zmian w kodzie. Walidator CLI `python -m src.tools.validate_personas` łapie
literówki przed runtime'em.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.domain.council import InvestorPersona

# Domyślna lokalizacja katalogu z plikami JSON. Override przez Settings
# (`council_personas_dir`) → main_agent.py.
DEFAULT_PERSONAS_DIR = Path("data/council_personas")

# Minimalna długość pola `style` — niżej to prawie na pewno literówka, nie
# persona. Powyżej 20 znaków = sensowne zdanie po polsku/angielsku.
_MIN_STYLE_LENGTH = 20

# Górne granice długości. `name` i `style` wpadają wprost do promptu LLM rady
# (finding #37) — operator-controlled, więc ryzyko ograniczone, ale oversized
# plik to wektor kosztu/prompt-injection. Fail-fast przy ładowaniu.
_MAX_NAME_LENGTH = 120
_MAX_STYLE_LENGTH = 2000


class PersonaSchemaError(ValueError):
    """Plik persony nie spełnia kontraktu: brak pól, zła struktura, duplikat."""


def load_council_personas(directory: Path) -> tuple[InvestorPersona, ...]:
    """Czyta wszystkie *.json z `directory`, zwraca personas posortowane po name.

    Walidacja fail-fast: błąd JSON, brak pola, pusty name/style, duplikat
    name między plikami — wszystko rzuca natychmiast przy starcie aplikacji.
    """
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(
            f"Persona directory does not exist: {directory}"
        )

    files = sorted(directory.glob("*.json"))
    if not files:
        raise ValueError(
            f"Found no persona JSON files in {directory} — council cannot operate."
        )

    seen_names: set[str] = set()
    personas: list[InvestorPersona] = []
    for path in files:
        persona = _load_persona_file(path)
        if persona.name in seen_names:
            raise PersonaSchemaError(
                f"duplicate persona name {persona.name!r} (file: {path.name})"
            )
        seen_names.add(persona.name)
        personas.append(persona)

    return tuple(sorted(personas, key=lambda p: p.name))


def _load_persona_file(path: Path) -> InvestorPersona:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PersonaSchemaError(
            f"malformed JSON in {path.name}: {exc.msg}"
        ) from exc

    if not isinstance(raw, dict):
        raise PersonaSchemaError(
            f"{path.name}: expected JSON object, got {type(raw).__name__}"
        )

    name = raw.get("name")
    style = raw.get("style")

    if not isinstance(name, str) or not name.strip():
        raise PersonaSchemaError(
            f"{path.name}: missing or empty 'name' field"
        )
    name = name.strip()
    if len(name) > _MAX_NAME_LENGTH:
        raise PersonaSchemaError(
            f"{path.name}: 'name' too long (>{_MAX_NAME_LENGTH} chars)"
        )
    # Znaki kontrolne / nowe linie w name = wektor prompt-injection (name
    # wpada do promptu LLM). Dopuszczamy tylko drukowalne znaki w jednej linii.
    if any(ch == "\n" or ch == "\r" or ord(ch) < 0x20 for ch in name):
        raise PersonaSchemaError(
            f"{path.name}: 'name' contains control characters or newlines"
        )
    if not isinstance(style, str):
        raise PersonaSchemaError(
            f"{path.name}: missing 'style' field"
        )
    style = style.strip()
    if len(style) < _MIN_STYLE_LENGTH:
        raise PersonaSchemaError(
            f"{path.name}: 'style' too short (<{_MIN_STYLE_LENGTH} chars)"
        )
    if len(style) > _MAX_STYLE_LENGTH:
        raise PersonaSchemaError(
            f"{path.name}: 'style' too long (>{_MAX_STYLE_LENGTH} chars)"
        )

    return InvestorPersona(name=name, style=style)
