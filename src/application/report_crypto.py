"""Sekcja 🪙 Krypto — każdy ticker krypto z ceną, Δ i statusem, także pominięty.

Krypto ma osobny próg (5%) i rynek 24/7, więc często ląduje jako 'ignored'
i ginęło w chipach „Pominięte”. Status pominiętego mówi, czy to próg, nowy
punkt odniesienia (FIN-14), czy budżet cyklu (FIN-12).
"""

from __future__ import annotations

from src.application.report_formatting import (
    DEFAULT_DISPLAY_TIMEZONE,
    direction_pct,
    format_local_date,
)
from src.application.report_formatting import (
    company_label as _company_label,
)
from src.application.report_formatting import (
    delta_color as _delta_color,
)
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    money as _money,
)
from src.application.report_formatting import (
    pct as _pct,
)
from src.application.report_formatting import (
    section_h2 as _section_h2,
)
from src.application.report_formatting import (
    trend_color as _trend_color,
)
from src.application.report_formatting import (
    trend_label as _trend_label,
)
from src.application.report_models import SymbolResult
from src.application.report_skipped import (
    BUDGET_CUT_LABEL,
    BUDGET_CUT_SHORT,
    is_budget_cut,
    is_rebaselined,
    rebaseline_label,
)
from src.application.reporting.section import ReportSection, SectionInput, text_block


def _is_crypto(r: SymbolResult) -> bool:
    return r.asset_class == "CRYPTO"


def _crypto_skip_label(r: SymbolResult, display_timezone: str, *, short: bool) -> str:
    """Status pominiętego krypto: „poniżej progu…”, (FIN-14) nowy punkt odniesienia
    albo (FIN-12) odcięcie budżetem płatnych wywołań cyklu."""
    if is_rebaselined(r) and r.reference_at is not None:
        return rebaseline_label(format_local_date(r.reference_at, display_timezone))
    if is_budget_cut(r):
        return BUDGET_CUT_SHORT if short else BUDGET_CUT_LABEL
    return "poniżej progu" if short else "poniżej progu zmienności"


def _render_crypto_section_html(
    results: list[SymbolResult], display_timezone: str = DEFAULT_DISPLAY_TIMEZONE
) -> str:
    """Dedykowana sekcja 🪙 Krypto — widoczna NIEZALEŻNIE od bramki volatility.

    Krypto ma osobny próg (5%) i 24/7 rynek, więc często ląduje jako 'ignored'
    i ginęło w chipach 'Pominięte'. Tu pokazujemy każdy ticker krypto z ceną,
    Δ i statusem (prognoza dla 'saved', 'poniżej progu' dla 'ignored')."""
    cryptos = [r for r in results if _is_crypto(r) and r.status != "error"]
    if not cryptos:
        return ""
    rows = []
    for r in cryptos:
        if r.status == "saved":
            trend_part = (
                f"<span style='color: {_trend_color(r.trend)}; font-weight: 600;'>"
                f"{_html(_trend_label(r.trend))}</span>"
            )
            forecast_part = (
                f" · prognoza <span style='color: {_delta_color(r.expected_change)};'>"
                f"{direction_pct(r.expected_change)}</span>"
                if r.expected_change is not None and not r.ml_cold_start
                else ""
            )
            conf_part = (
                f" · pewność {_pct(r.confidence_score, decimals=0)}"
                if r.confidence_score is not None
                else ""
            )
            status_html = f"{trend_part}{forecast_part}{conf_part}"
        else:
            status_html = (
                "<span style='color: #6b7280;'>"
                f"{_html(_crypto_skip_label(r, display_timezone, short=False))}</span>"
            )
        rows.append(f"""
          <div style="margin-bottom: 6px; padding: 8px 12px; background: #fdf4ff;
                      border-left: 3px solid #a855f7; border-radius: 4px; font-size: 13px;">
            <strong>{_html(_company_label(r.symbol))}</strong>
            <span style="color: #4b5563;">· {_money(r.current_price)}
              · Δ <span style="color: {_delta_color(r.delta)};">{direction_pct(r.delta)}</span>
              · {status_html}</span>
          </div>
        """)
    return (
        _section_h2("🪙 Krypto")
        + "".join(rows)
    )


def _render_crypto_section_text(
    results: list[SymbolResult], display_timezone: str = DEFAULT_DISPLAY_TIMEZONE
) -> str:
    cryptos = [r for r in results if _is_crypto(r) and r.status != "error"]
    if not cryptos:
        return ""
    lines = ["KRYPTO", "-" * 64]
    for r in cryptos:
        if r.status == "saved":
            status = (
                f"{_trend_label(r.trend)} "
                f"prognoza {_pct(r.expected_change, signed=True)}"
                if r.expected_change is not None and not r.ml_cold_start
                else _trend_label(r.trend)
            )
        else:
            status = _crypto_skip_label(r, display_timezone, short=True)
        lines.append(
            f"  {_company_label(r.symbol):40s} {_money(r.current_price):>12s}  "
            f"Δ {_pct(r.delta, signed=True):>8s}  {status}"
        )
    return "\n".join(lines)


def _section_html(data: SectionInput) -> str:
    return _render_crypto_section_html(data.results, data.context.display_timezone)


def _section_text(data: SectionInput) -> list[str]:
    return text_block(
        _render_crypto_section_text(data.results, data.context.display_timezone), spaced=True
    )


CRYPTO_SECTION = ReportSection(key="crypto", render_html=_section_html, render_text=_section_text)
