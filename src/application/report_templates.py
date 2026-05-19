"""Jinja2 setup dla report_builder.

Etap migracji 1: dwie izolowane sekcje (council + valuation) renderowane
przez templates/ zamiast inline'owych f-string. Reszta raportu pozostaje
w report_builder.py — migracja przyrostowa, sekcja po sekcji.

Autoescape włączony dla .html / .j2 — input z LLM-a (reasoning, summary,
dissenting_views) jest niezaufany i mógłby zawierać HTML/script.
"""
from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader, Template, select_autoescape

_env = Environment(
    loader=PackageLoader("src.application", "templates"),
    autoescape=select_autoescape(default_for_string=True, default=True),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_template(name: str) -> Template:
    """Zwraca Jinja2 Template po nazwie pliku w `src/application/templates/`."""
    return _env.get_template(name)


def render_template(name: str, context: dict[str, Any]) -> str:
    """Renderuje template z dict-em kontekstu. Pusty string gdy verdict=None
    (template ma guard) — wygodne do podstawienia w miejscu f-string."""
    return get_template(name).render(**context)
