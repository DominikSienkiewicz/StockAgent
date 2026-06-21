"""Render sekcji 🛰️ Alpha Signals (nowe źródła danych) w raporcie e-mail.

Czysta prezentacja: makro-blok krzywej rentowności (FRED) + tabela sygnałów
alpha per symbol (insider / analitycy / opcje / social / earnings). Sekcja
chowa się w całości, gdy nie ma ani krzywej, ani żadnego symbolu z sygnałem."""

from __future__ import annotations

import html as _html
from decimal import Decimal

from src.application.alpha_signals import AlphaSignals
from src.domain.analyst_consensus import AnalystRating
from src.domain.earnings import EarningsProximity
from src.domain.insider_flow import InsiderSignal
from src.domain.macro_rates import YieldCurveSnapshot, YieldCurveState
from src.domain.options_flow import OptionsSentiment
from src.domain.social_velocity import SocialTrend

_DASH = "—"

_INSIDER_LABEL = {
    InsiderSignal.NET_BUYING: "kupno insiderów",
    InsiderSignal.NET_SELLING: "sprzedaż insiderów",
    InsiderSignal.NEUTRAL: _DASH,
}
_OPTIONS_LABEL = {
    OptionsSentiment.BULLISH: "byczo (calls)",
    OptionsSentiment.BEARISH: "niedźwiedzio (puts)",
    OptionsSentiment.NEUTRAL: _DASH,
}
_SOCIAL_LABEL = {
    SocialTrend.SURGING: "skok wzmianek",
    SocialTrend.NORMAL: "normalnie",
    SocialTrend.QUIET: "cicho",
}
_RATING_LABEL = {
    AnalystRating.STRONG_BUY: "Strong Buy",
    AnalystRating.BUY: "Buy",
    AnalystRating.HOLD: "Hold",
    AnalystRating.SELL: "Sell",
    AnalystRating.STRONG_SELL: "Strong Sell",
}
_PROXIMITY_LABEL = {
    EarningsProximity.IMMINENT: "tuż przed",
    EarningsProximity.NEAR: "w tym tygodniu",
    EarningsProximity.FAR: _DASH,
}
_YIELD_STATE_LABEL = {
    YieldCurveState.INVERTED: "INWERSJA (sygnał recesyjny)",
    YieldCurveState.FLAT: "płaska",
    YieldCurveState.NORMAL: "normalna",
}
_YIELD_STATE_COLOR = {
    YieldCurveState.INVERTED: "#dc2626",
    YieldCurveState.FLAT: "#ca8a04",
    YieldCurveState.NORMAL: "#16a34a",
}


def render_alpha_html(
    signals_by_symbol: dict[str, AlphaSignals],
    yield_curve: YieldCurveSnapshot | None,
    current_prices: dict[str, Decimal] | None = None,
) -> str:
    """Fragment HTML sekcji Alpha Signals. Pusty string, gdy brak danych."""
    prices = current_prices or {}
    rows = [
        sig for sig in signals_by_symbol.values() if sig.has_any()
    ]
    has_yield = yield_curve is not None and (
        yield_curve.ten_year is not None or yield_curve.two_year is not None
    )
    if not rows and not has_yield:
        return ""

    parts: list[str] = [
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>"
        "🛰️ Alpha Signals — dodatkowe źródła danych</h2>"
    ]
    if has_yield and yield_curve is not None:
        parts.append(_render_yield_curve_html(yield_curve))
    if rows:
        parts.append(_render_signals_table_html(rows, prices))
    return "".join(parts)


def _render_yield_curve_html(curve: YieldCurveSnapshot) -> str:
    """Makro-blok krzywej rentowności (FRED) — stan + kluczowe stopy."""
    state = curve.state()
    color = _YIELD_STATE_COLOR[state]
    spread = curve.spread_10y_2y
    spread_txt = f"{spread:+.2f} pp" if spread is not None else _DASH

    def fmt(v: float | None) -> str:
        return f"{v:.2f}%" if v is not None else _DASH

    return (
        "<div style='padding: 10px 14px; background: #f8fafc; "
        f"border-left: 3px solid {color}; border-radius: 4px; "
        "margin-bottom: 12px; font-size: 13px;'>"
        "<strong>📉 Krzywa rentowności (FRED):</strong> "
        f"<span style='color: {color}; font-weight: 600;'>"
        f"{_YIELD_STATE_LABEL[state]}</span> · "
        f"10Y {fmt(curve.ten_year)} · 2Y {fmt(curve.two_year)} · "
        f"spread 10Y-2Y {spread_txt}"
        + (
            f" · Fed funds {fmt(curve.fed_funds)}"
            if curve.fed_funds is not None
            else ""
        )
        + "</div>"
    )


def _render_signals_table_html(
    rows: list[AlphaSignals], prices: dict[str, Decimal]
) -> str:
    """Tabela sygnałów alpha per symbol (tylko symbole z jakimkolwiek sygnałem)."""
    body: list[str] = []
    for sig in rows:
        body.append(_render_signal_row_html(sig, prices.get(sig.symbol)))
    return (
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th style='padding: 6px 8px; text-align: left;'>Symbol</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Insiderzy</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Analitycy</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Opcje</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Social</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Earnings</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def _render_signal_row_html(sig: AlphaSignals, price: Decimal | None) -> str:
    insider_txt = (
        _INSIDER_LABEL[sig.insider.evaluate_signal()] if sig.insider else _DASH
    )
    analyst_txt = _DASH
    if sig.analyst is not None:
        analyst_txt = _RATING_LABEL[sig.analyst.rating()]
        upside = sig.analyst.upside(price) if price is not None else None
        if upside is not None:
            analyst_txt += f" ({upside * 100:+.0f}%)"
    options_txt = (
        _OPTIONS_LABEL[sig.options.sentiment()] if sig.options else _DASH
    )
    social_txt = _DASH
    if sig.social is not None:
        social_txt = (
            f"{_SOCIAL_LABEL[sig.social.trend()]} "
            f"(×{sig.social.velocity_ratio():.1f})"
        )
    earnings_txt = _DASH
    if sig.earnings is not None:
        prox = _PROXIMITY_LABEL[sig.earnings.proximity()]
        if sig.earnings.days_until is not None and prox != _DASH:
            earnings_txt = f"{prox} ({sig.earnings.days_until}d)"
        else:
            earnings_txt = prox

    def cell(text: str) -> str:
        return f"<td style='padding: 6px 8px;'>{_html.escape(text)}</td>"

    return (
        "<tr>"
        f"<td style='padding: 6px 8px;'><strong>{_html.escape(sig.symbol)}</strong></td>"
        + cell(insider_txt)
        + cell(analyst_txt)
        + cell(options_txt)
        + cell(social_txt)
        + cell(earnings_txt)
        + "</tr>"
    )
