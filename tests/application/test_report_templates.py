# tests/application/test_report_templates.py
"""Testy infrastruktury Jinja2 dla report_builder.

Migracja inline'owych f-string HTML do templates/ — etap pierwszy:
council_section + valuation_section. Pozostałe sekcje raportu (główny layout,
trade signals, mood box) zostają w f-string render do następnej iteracji.

Tutaj testujemy *infrastrukturę* (loader działa, autoescape włączone,
templates istnieją na dysku), a nie ich treść — content jest pokryty
przez test_report_council.py / test_report_valuation_section.py.
"""
from __future__ import annotations

from src.application.report_templates import get_template, render_template


def test_get_template_loads_council_section():
    template = get_template("council_section.html.j2")
    assert template is not None


def test_get_template_loads_valuation_section():
    template = get_template("valuation_section.html.j2")
    assert template is not None


def test_render_template_returns_str():
    result = render_template(
        "council_section.html.j2",
        {
            "verdict": None,  # template ma guard: pusty string gdy None
        },
    )
    assert isinstance(result, str)


def test_autoescape_html_attacks():
    # Jinja2 environment musi mieć włączony autoescape dla .html.j2.
    # Mały smoke test: złośliwy input z <script> ma być zamieniony na encje.
    from src.application.report_templates import _env

    template_str = "{{ user_input }}"
    template = _env.from_string(template_str)
    rendered = template.render(user_input="<script>alert('xss')</script>")
    # Z autoescape powinien być &lt;script&gt;...
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
