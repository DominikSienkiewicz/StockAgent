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
from src.domain.council import CouncilVerdict, InvestorOpinion

_SCRIPT = "<script>alert('xss')</script>"


def test_get_template_loads_council_section() -> None:
    template = get_template("council_section.html.j2")
    assert template is not None


def test_get_template_loads_valuation_section() -> None:
    template = get_template("valuation_section.html.j2")
    assert template is not None


def test_render_template_returns_str() -> None:
    result = render_template(
        "council_section.html.j2",
        {
            "verdict": None,  # template ma guard: pusty string gdy None
        },
    )
    assert isinstance(result, str)


def test_autoescape_html_attacks() -> None:
    # Jinja2 environment musi mieć włączony autoescape dla .html.j2.
    # Mały smoke test: złośliwy input z <script> ma być zamieniony na encje.
    from src.application.report_templates import _env

    template_str = "{{ user_input }}"
    template = _env.from_string(template_str)
    rendered = template.render(user_input="<script>alert('xss')</script>")
    # Z autoescape powinien być &lt;script&gt;...
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_real_council_template_escapes_llm_text() -> None:
    """TEST-22 (G80): autoescape na PRAWDZIWYM pliku `council_section.html.j2`.

    Rozszerzenie `.j2` nie należy do `enabled_extensions` `select_autoescape`,
    więc o escapowaniu pliku decyduje `default=` — test `from_string` wyżej
    sprawdza tylko `default_for_string` i tej gałęzi nie widzi. Podsumowanie,
    czynniki i głosy niezgodne pochodzą z LLM-a."""
    verdict = CouncilVerdict(
        final_recommendation="BUY",
        consensus_strength=0.73,
        summary=f"Rada radzi kupować {_SCRIPT}",
        dissenting_views=(f"Soros: {_SCRIPT}",),
        investor_opinions=(
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",
                confidence=0.9,
                reasoning=_SCRIPT,
                key_factors=(_SCRIPT,),
            ),
        ),
    )

    html = render_template(
        "council_section.html.j2",
        {
            "verdict": verdict,
            "label": "KUP",
            "fill": "fu",
            "row_tone": "gu",
            "strength_pct": 73,
            "opinion_rows": [
                {
                    "name": "Warren Buffett",
                    "label": "KUP",
                    "emoji": "🟢",
                    "tone": "up",
                    "row_tone": "gu",
                    "confidence_pct": 90,
                    "confidence_label": "wysoka",
                    "factors": _SCRIPT,
                }
            ],
            "vote_dist": verdict.vote_distribution(),
            "is_split": False,
            "strong_consensus": False,
            "dissenting_views": verdict.dissenting_views,
        },
    )

    assert "RADA DORADCZA" in html
    assert "Głosy niezgodne" in html
    assert "<script>" not in html
    assert html.count("&lt;script&gt;") == 3
