"""Odciski składników przepisu predykcji (WIA-20), które wymagają czytania plików.

Domena (`src/domain/recipe.py`) dostaje gotowe hashe; tutaj jest jedyne I/O:
kod modułów z szablonami promptów i plik wag XGBoost. Liczone raz na cykl w DI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

from src.application import _prompt_safety, council_prompts, prompts

# Moduły, z których składa się prompt predykcji i prompty rady. Hash obejmuje
# CAŁY kod modułu (szablon, a nie wyrenderowany prompt), więc zmienia się też
# przy kosmetyce: komentarzu, docstringu, formatowaniu.
PROMPT_TEMPLATE_MODULES: tuple[ModuleType, ...] = (prompts, council_prompts, _prompt_safety)

# Wersja modelu ML, gdy nie ma pliku wag (Fast Loop leci na baseline).
COLD_START_MODEL_VERSION = "cold_start"


def prompt_templates_sha256() -> str:
    """sha256 kodu modułów promptów (nazwa modułu + bajty pliku, w stałej kolejności)."""
    digest = hashlib.sha256()
    for module in PROMPT_TEMPLATE_MODULES:
        if module.__file__ is None:
            raise RuntimeError(f"prompt module {module.__name__} has no source file")
        digest.update(module.__name__.encode("utf-8") + b"\0")
        digest.update(Path(module.__file__).read_bytes() + b"\0")
    return digest.hexdigest()


def ml_model_version(model_path: str) -> str:
    """"cold_start", gdy brak pliku wag; inaczej sha256 jego bajtów."""
    path = Path(model_path)
    if not path.is_file():
        return COLD_START_MODEL_VERSION
    return hashlib.sha256(path.read_bytes()).hexdigest()
