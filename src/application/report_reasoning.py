"""Blok 💡 Uzasadnienia jednej predykcji i jego części.

Sekcja 🔮 predykcji (`report_predictions.py`) składa z nich pełny blok dla top N
sygnałów i symboli z leadu, a zwiniętą linię rady dla reszty (MAIL-06). Rada
i wycena idą przez szablony Jinja (`templates/`), blok „Top newsy” i linia rady
przez autoescapowany `_env`; reszta to f-stringi z ręcznym `_html(...)` (patrz
`report_formatting.escape`).
"""

from __future__ import annotations

from collections.abc import Mapping

from src.application.report_formatting import (
    COLD_START_FORECAST as _COLD_START_FORECAST,
)
from src.application.report_formatting import (
    PALETTE,
    direction_pct,
    safe_http_url,
)
from src.application.report_formatting import (
    company_label as _company_label,
)
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    money as _money,
)
from src.application.report_formatting import (
    number as _number,
)
from src.application.report_formatting import (
    pct as _pct,
)
from src.application.report_formatting import (
    provenance_badges_html as _provenance_badges_html,
)
from src.application.report_formatting import (
    sector_label as _sector_label,
)
from src.application.report_formatting import (
    trend_color as _trend_color,
)
from src.application.report_formatting import (
    trend_label as _trend_label,
)
from src.application.report_models import (
    SimilarPrecedent,
    SymbolResult,
    TopNewsItem,
    ValuationSection,
)
from src.application.report_styles import (
    COUNCIL_FILL,
    COUNCIL_ROW_TONE,
    COUNCIL_TONE,
    EDGE_TONE,
    SIGNAL_TONE,
    delta_tone,
    trend_tone,
)
from src.application.report_templates import _env, render_template
from src.domain.conformal import MoveInterval
from src.domain.council import CouncilVerdict
from src.domain.feature_attribution import FeatureContribution
from src.domain.polish_labels import (
    MODEL_SIGNAL_TITLE,
    council_recommendation_label,
    model_signal_label,
)

# MAIL-06 (c): tło bloku sygnału modelu jako klasa tonu (report_styles).
_SIGNAL_FILL = {"up": "gu", "dn": "gd", "nt": "gn"}

_COUNCIL_REC_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

_VERDICT_COLOR = {
    "UNDERVALUED": "#2e7d32",  # zielony
    "FAIR": "#616161",         # szary
    "OVERVALUED": "#c62828",   # czerwony
    "UNKNOWN": "#9e9e9e",      # jasnoszary
}


def _render_valuation(section: ValuationSection | None) -> str:
    """Renderuje sekcję wyceny fundamentalnej do HTML (Jinja2 template)."""
    if section is None:
        return ""

    def fmt(v: float | None) -> str:
        return _number(v, decimals=2) if isinstance(v, (int, float)) else "—"

    growth = (
        _pct(section.eps_growth_yoy, decimals=1)
        if section.eps_growth_yoy is not None
        else "—"
    )
    return render_template(
        "valuation_section.html.j2",
        {
            "section": section,
            "verdict_value": section.verdict.value,
            "verdict_color": _VERDICT_COLOR.get(section.verdict.value, "#616161"),
            "trailing_pe_fmt": fmt(section.trailing_pe),
            "forward_pe_fmt": fmt(section.forward_pe),
            "peg_ratio_fmt": fmt(section.peg_ratio),
            "growth_fmt": growth,
            "fetched_at_iso": section.fetched_at.isoformat(),
        },
    )


def render_valuation_text(section: ValuationSection | None) -> str:
    """Renderuje sekcję wyceny fundamentalnej do plain text."""
    if section is None:
        return ""

    def fmt(v: float | None) -> str:
        return _number(v, decimals=2) if isinstance(v, (int, float)) else "—"

    growth = (
        _pct(section.eps_growth_yoy, decimals=1)
        if section.eps_growth_yoy is not None
        else "—"
    )

    lines = [
        "  Wycena fundamentalna:",
        f"    Trailing P/E: {fmt(section.trailing_pe)}  · Forward P/E: {fmt(section.forward_pe)}",
        f"    PEG ratio: {fmt(section.peg_ratio)}  · EPS growth YoY: {growth}",
        f"    Werdykt: {section.verdict.value}",
    ]
    return "\n".join(lines)


def model_signal(r: SymbolResult) -> str:
    """MAIL-27: sygnał modelu (KUP/SPRZEDAJ/TRZYMAJ) — etykieta ze wspólnego
    słownika domeny, nie „rekomendacja”."""
    return model_signal_label(r.trend)


def _signed_move(value: float) -> str:
    """Zwrot jako „+2,4%” / „-2,1%” (minus ASCII, jedno miejsce, MAIL-58)."""
    return _pct(value, signed=True, decimals=1)


def move_interval_label(interval: MoveInterval) -> str:
    """ML-34: „80% przedział: -2,1% … +2,4% (n=57)” albo jawne „za mało
    historii (n/20)" — krótkie okno nie udaje liczby."""
    level = f"{_pct(interval.nominal_coverage, decimals=0)} przedział"
    if interval.low is None or interval.high is None:
        return (
            f"{level}: za mało historii "
            f"({interval.calibration_size}/{interval.required_size})"
        )
    return (
        f"{level}: {_signed_move(interval.low)} … {_signed_move(interval.high)} "
        f"(n={interval.calibration_size})"
    )


def _model_signal_reason_html(r: SymbolResult) -> str:
    """Zwraca krótki HTML-owy powód sygnału modelu (1–2 zdania)."""
    direction = model_signal(r)
    tone = SIGNAL_TONE.get(direction, "nt")

    conf_part = (
        f"Pewność: <strong>{_pct(r.confidence_score, decimals=0)}</strong>."
        if r.confidence_score is not None
        else ""
    )
    change_part = ""
    if r.expected_change is not None and not r.ml_cold_start:
        change_part = (
            f" Prognozowana zmiana: "
            f"<strong class='{delta_tone(r.expected_change)}'>"
            f"{direction_pct(r.expected_change)}</strong>."
        )
    sentiment_part = ""
    if r.sentiment_score is not None:
        if r.sentiment_score > 0.2:
            sentiment_part = " Sentyment rynkowy pozytywny."
        elif r.sentiment_score < -0.2:
            sentiment_part = " Sentyment rynkowy negatywny."
        else:
            sentiment_part = " Sentyment neutralny."

    return (
        f'<div class="rs {_SIGNAL_FILL[tone]}">'
        f'<span class="b7 {tone}">➤ {MODEL_SIGNAL_TITLE}: {direction}</span> — '
        f"{conf_part}{change_part}{sentiment_part}"
        f"</div>"
    )


def model_signal_reason_text(r: SymbolResult) -> str:
    direction = model_signal(r)
    conf_part = (
        f"pewność {_pct(r.confidence_score, decimals=0)}"
        if r.confidence_score is not None
        else ""
    )
    change_part = (
        f"prognoza {_pct(r.expected_change, signed=True)}"
        if r.expected_change is not None and not r.ml_cold_start
        else ""
    )
    parts = [p for p in [conf_part, change_part] if p]
    detail = ", ".join(parts)
    return f"{MODEL_SIGNAL_TITLE}: {direction}" + (f" ({detail})" if detail else "")


def _render_council_section(verdict: CouncilVerdict) -> str:
    """Renderuje sekcję rady doradczej (Jinja2 template).

    Logika decyzyjna (split decision, strong consensus, vote distribution)
    pochodzi z metod domenowych `CouncilVerdict` — template tylko renderuje.
    """
    final = verdict.final_recommendation
    opinion_rows = [
        {
            "name": op.investor_name,
            "label": council_recommendation_label(op.recommendation),
            "emoji": _COUNCIL_REC_EMOJI.get(op.recommendation, "⬜"),
            "tone": COUNCIL_TONE.get(op.recommendation, "nt"),
            "row_tone": COUNCIL_ROW_TONE.get(op.recommendation, "gn"),
            "confidence_pct": int(op.confidence * 100),
            "confidence_label": op.confidence_label(),
            "factors": ", ".join(op.key_factors[:3]),
        }
        for op in verdict.investor_opinions
    ]
    return render_template(
        "council_section.html.j2",
        {
            "verdict": verdict,
            "label": council_recommendation_label(final),
            "fill": COUNCIL_FILL.get(final, "fn"),
            "row_tone": COUNCIL_ROW_TONE.get(final, "gn"),
            "strength_pct": int(verdict.consensus_strength * 100),
            "opinion_rows": opinion_rows,
            "vote_dist": verdict.vote_distribution(),
            "is_split": verdict.is_split_decision(),
            "strong_consensus": verdict.has_strong_consensus(),
            "dissenting_views": verdict.dissenting_views,
        },
    )


# MAIL-06 (b): rada spoza top N sygnałów i spoza leadu — jedna linia (werdykt,
# rozkład głosów, podsumowanie) i zawsze widoczne głosy niezgodne. Pełna tabela
# person to ~4 KB na symbol, a Gmail przycina mail powyżej ~102 KB. Summary i
# dysydenci pochodzą z LLM-a, więc linia idzie przez autoescapowany `_env`.
_COUNCIL_LINE_TEMPLATE = _env.from_string(
    "<span class='b7 {{ tone }}'>{{ label }}</span>"
    " · zgodność {{ strength_pct }}%"
    "{% if failed %} · ⚠️ odpowiedziało {{ answered }}/{{ size }}{% endif %}"
    "{% if is_split %} · ⚠️ podzielona{% endif %} · "
    "🟢 {{ dist['BUY'] }} KUP · 🔴 {{ dist['SELL'] }} SPRZEDAJ"
    " · 🟡 {{ dist['HOLD'] }} TRZYMAJ"
    " — „{{ summary }}”"
    "{% if dissent %}<div class='dv'>"
    "⚠️ Głosy niezgodne: {{ dissent | join('; ') }}</div>{% endif %}"
)


def render_council_line(verdict: CouncilVerdict) -> str:
    """MAIL-06 (b): zwinięta rada — jedna linia zamiast tabeli person (komórka
    „Rada” w skrócie pozostałych predykcji)."""
    final = verdict.final_recommendation
    return _COUNCIL_LINE_TEMPLATE.render(
        label=council_recommendation_label(final),
        tone=COUNCIL_TONE.get(final, "nt"),
        strength_pct=int(verdict.consensus_strength * 100),
        failed=verdict.failed_personas,
        answered=len(verdict.investor_opinions),
        size=verdict.council_size(),
        is_split=verdict.is_split_decision(),
        dist=verdict.vote_distribution(),
        summary=verdict.summary,
        dissent=verdict.dissenting_views,
    )


# Inline Jinja template bloku "Top newsy" — renderowany przez WSPÓLNY,
# autoescapowany `_env` (select_autoescape(default_for_string=True)).
# Escapowanie `title`/`source` jest tu DOMYŚLNE; `href` to zwalidowany surowy
# URL (sam autoescape zadba o atrybut, bez podwójnego escapowania).
_NEWS_BLOCK_TEMPLATE = _env.from_string(
    "{%- if items -%}"
    "<div class='nw'>"
    "📰 <strong>Top newsy:</strong>"
    "<ul class='nl'>"
    "{%- for n in items -%}"
    "<li class='ni'>"
    "<strong>[{{ n.source_label }}]</strong> "
    "{% if n.href %}"
    "<a href='{{ n.href }}' target='_blank' class='na'>{{ n.title }}</a>"
    "{% else %}{{ n.title }}{% endif %}"
    " <span class='nm'>"
    "(relevance {{ n.relevance_fmt }}, sentyment {{ n.sentiment_fmt }})"
    "</span></li>"
    "{%- endfor -%}"
    "</ul></div>"
    "{%- endif -%}"
)


def _render_news_block_html(top_news: list[TopNewsItem]) -> str:
    """Renderuje blok 'Top newsy' przez autoescapowany Jinja env.

    Migracja przyrostowa f-string → Jinja (finding #35): escapowanie tytułów
    i źródeł newsów jest tu domyślne, nie opt-in. Output równoważny dawnemu
    f-stringowi (te same style inline, kolejność i format relevance/sentyment)."""
    items = [
        {
            "source_label": n.source or "unknown",
            "title": n.title,
            "href": safe_http_url(n.url),
            "relevance_fmt": _number(n.relevance, decimals=2),
            "sentiment_fmt": _number(n.sentiment, decimals=2, signed=True),
        }
        for n in top_news
    ]
    return _NEWS_BLOCK_TEMPLATE.render(items=items)


# Q5: etykieta wyniku analogu (kierunkowo trafiony / chybiony / nieoceniony).
_PRECEDENT_OUTCOME = {
    True: ("✅ trafił", PALETTE.positive),
    False: ("❌ chybił", PALETTE.negative),
}


def _render_precedents_block_html(precedents: list[SimilarPrecedent]) -> str:
    """Renderuje blok 'Na podstawie analogów' (Q5) — audytowalny ślad RAG.

    Pokazuje historyczne sytuacje, które agent pobrał w tym cyklu, z ich
    rzeczywistym wynikiem kierunkowym. Pusta lista → pusty string (brak bloku,
    brak błędu). `summary` pochodzi z LLM-a (dane zewnętrzne) → escapujemy
    przez `_html`."""
    if not precedents:
        return ""
    items = []
    for p in precedents:
        outcome_txt, outcome_color = (
            _PRECEDENT_OUTCOME[p.is_trend_correct]
            if p.is_trend_correct is not None
            else ("• wynik nieznany", "#6b7280")
        )
        items.append(
            f"<li style='margin: 2px 0;'>"
            f"<span style='color: {_trend_color(p.predicted_trend)}; font-weight: 600;'>"
            f"{_html(_trend_label(p.predicted_trend))}</span> "
            f"<span style='color: {outcome_color};'>({outcome_txt})</span> — "
            f"{_html(p.summary)}</li>"
        )
    return (
        "<div style='font-size: 11px; color: #4b5563; margin-top: 8px;'>"
        "🧭 <strong>Na podstawie analogów:</strong>"
        "<ul style='margin: 4px 0 0 20px; padding: 0;'>"
        + "".join(items)
        + "</ul></div>"
    )


def _render_attribution_block_html(
    contribs: tuple[FeatureContribution, ...],
) -> str:
    """Renderuje blok 'Wkład cech' (Q3 SHAP-lite) — które cechy ML pchnęły
    predykcję w górę/dół i o ile. Pusta krotka → pusty string."""
    if not contribs:
        return ""
    chips = []
    for c in contribs:
        positive = c.contribution >= 0
        color = PALETTE.positive if positive else PALETTE.negative
        chips.append(
            f"<span style='display:inline-block; margin-right:10px;'>"
            f"{_html(c.feature)} "
            f"<strong style='color:{color};'>"
            f"{_number(c.contribution, decimals=4, signed=True)}</strong></span>"
        )
    return (
        "<div style='font-size: 11px; color: #475569; margin-top: 6px; "
        "padding: 4px 8px; background: #f8fafc; border-radius: 3px;'>"
        "🔍 <strong>Wkład cech:</strong> " + "".join(chips) + "</div>"
    )


def render_reasoning_block_html(
    r: SymbolResult, sector_map: Mapping[str, str], chip: str | None = None
) -> str:
    """Blok 'Uzasadnienia' dla jednej predykcji. Pusty string, gdy nie ma
    czego pokazać (brak reasoning / rady / odznak / analogów / atrybucji).
    `chip` (WIA-35) — trafność kierunkowa na symbolu, pod oczekiwanym ruchem."""
    # Q5/Q6: blok renderujemy też, gdy są analogi (precedent receipts)
    # lub odznaki proweniencji — nawet bez reasoning/rady.
    badges_html = _provenance_badges_html(r.provenance_badges)
    precedents_block = _render_precedents_block_html(r.similar_precedents)
    attribution_block = _render_attribution_block_html(r.feature_attribution)
    if (
        not r.reasoning
        and r.council_verdict is None
        and not badges_html
        and not precedents_block
        and not attribution_block
        and r.move_interval is None
    ):
        return ""
    move_line = ""
    if r.current_price is not None and r.target_price is not None:
        conf_part = (
            f" · Pewność {_pct(r.confidence_score, decimals=0)}"
            if r.confidence_score is not None
            else ""
        )
        move_text = (
            _COLD_START_FORECAST
            if r.ml_cold_start
            else (
                f"{_money(r.current_price)} → {_money(r.target_price)} "
                f"<span class='{delta_tone(r.expected_change)}'>"
                f"({direction_pct(r.expected_change)})</span>"
            )
        )
        move_line = f'<div class="rm">Oczekiwany ruch: {move_text}{conf_part}</div>'
    if r.move_interval is not None:
        move_line += f'<div class="rm">{move_interval_label(r.move_interval)}</div>'
    if chip:
        move_line += f'<div class="rm">{_html(chip)}</div>'
    # Blok 'Top newsy' — zmigrowany na autoescapowany Jinja env
    # (finding #35). Escapowanie title/source jest tu domyślne.
    news_block = _render_news_block_html(r.top_news)
    rec_block = _model_signal_reason_html(r)
    council_block = (
        _render_council_section(r.council_verdict)
        if r.council_verdict is not None
        else ""
    )
    valuation_block = _render_valuation(r.valuation)
    _sector = _sector_label(r.symbol, sector_map)
    sector_tag = (
        f' <span style="color: #6b7280; font-weight: 500;">· {_html(_sector)}</span>'
        if _sector
        else ""
    )
    company = _html(_company_label(r.symbol))
    trend = f'<span class="{trend_tone(r.trend)}">{_html(_trend_label(r.trend))}</span>'
    return f"""
              <div class="rb {EDGE_TONE[trend_tone(r.trend)]}">
                <div class="rh">{company}{sector_tag} {trend}{badges_html}</div>
                {move_line}
                {rec_block}
                {f'<div class="rt">{_html(r.reasoning)}</div>' if r.reasoning else ''}
                {attribution_block}
                {precedents_block}
                {news_block}
                {council_block}
                {valuation_block}
              </div>
            """
