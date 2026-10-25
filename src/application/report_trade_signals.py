"""Sekcja 🎯 Najsilniejsze sygnały — czołówka rankingu `build_cycle_signals`.

Siła to |ruch| × pewność (skalibrowana, gdy jest historia — WIA-12), a przy
cold-starcie ML sama pewność (WIA-13). Przy znanym portfelu kierunek idzie
w kontekście pozycji (PORT-08), a pasmo wielkości pozycji (Q7) tylko tam, gdzie
nie przeczy kierunkowi. Pod wierszem trafność kierunkowa symbolu (WIA-35).
"""

from __future__ import annotations

from collections.abc import Mapping

from src.application.report_formatting import (
    COLD_START_FORECAST as _COLD_START_FORECAST,
)
from src.application.report_formatting import (
    PALETTE,
    direction_pct,
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
    pct as _pct,
)
from src.application.report_formatting import (
    section_h2 as _section_h2,
)
from src.application.report_models import TradeSignal
from src.application.reporting.section import ReportSection, SectionInput
from src.domain.sizing import PositionAction, SizeBand

# WIA-13: przy cold-starcie siła sygnału to sama pewność — sekcja mówi to wprost.
_SIGNALS_COLD_START_NOTE = (
    "Model ML bez wag (cold-start): ranking wg pewności LLM, skalibrowanej historią, "
    "gdy jest próbka — bez prognozy ruchu."
)


# PORT-08: język ostrożny (MAR/KNF) — kierunek w kontekście portfela to nadal sygnał modelu.
_SIGNALS_POSITION_NOTE = (
    "Kierunek w kontekście Twojego portfela (tylko pozycje długie): spadek na "
    "trzymanej spółce to „ZREDUKUJ”, poza portfelem „UNIKAJ”, wzrost na już dużej "
    "pozycji „TRZYMAJ, nie dokładaj”. To sygnał modelu, nie rekomendacja."
)


# PORT-08: kolor kierunku po akcji pozycji — spadek czerwony, wzrost zielony.
_POSITION_ACTION_TONE = {
    PositionAction.BUY: "KUP",
    PositionAction.HOLD: "OBSERWUJ",
    PositionAction.REDUCE: "SPRZEDAJ",
    PositionAction.AVOID: "SPRZEDAJ",
    PositionAction.WATCH: "OBSERWUJ",
}


def _calibration_suffix(sig: TradeSignal) -> str:
    """„ po kalibracji (n kubełka: 12)” — n pomijamy, gdy sygnał go nie niesie."""
    if sig.calibration_bucket_count is None:
        return " po kalibracji"
    return f" po kalibracji (n kubełka: {sig.calibration_bucket_count})"


def _confidence_html(sig: TradeSignal) -> str:
    """WIA-12: „pewność 85% → 58% po kalibracji (n kubełka: 12)” albo sama surowa."""
    raw = f"pewność <strong>{_pct(sig.confidence, decimals=0)}</strong>"
    if sig.calibrated_confidence is None:
        return raw
    return (
        f"{raw} → <strong>{_pct(sig.calibrated_confidence, decimals=0)}</strong>"
        f"{_html(_calibration_suffix(sig))}"
    )


def _confidence_text(sig: TradeSignal) -> str:
    raw = _pct(sig.confidence, decimals=0)
    if sig.calibrated_confidence is None:
        return raw
    return f"{raw} → {_pct(sig.calibrated_confidence, decimals=0)}{_calibration_suffix(sig)}"


def _strength_label(rank: int, sig: TradeSignal) -> str:
    """WIA-12: pozycja w rankingu i składowe siły zamiast gołej liczby.

    Siła to |ruch| × pewność (skalibrowana, gdy jest historia), a przy
    cold-starcie ML sama pewność — etykieta mówi, z czego wynika kolejność.
    """
    calibrated = sig.calibrated_confidence is not None
    effective = sig.confidence if sig.calibrated_confidence is None else sig.calibrated_confidence
    confidence_pct = _pct(effective, decimals=0)
    if sig.ml_cold_start:
        basis = "pewności skalibrowanej" if calibrated else "pewności"
        return f"#{rank} · wg {basis} {confidence_pct}"
    basis = "pewność skalibrowana" if calibrated else "pewność"
    move_pct = _pct(abs(sig.expected_change), decimals=1)
    return f"#{rank} · ruch {move_pct} × {basis} {confidence_pct}"


def _signal_label(sig: TradeSignal) -> str:
    """PORT-08: kierunek sygnału — przy znanym portfelu w kontekście pozycji
    („ZREDUKUJ (12% portfela)”), inaczej KUP / SPRZEDAJ / OBSERWUJ jak dotąd."""
    action = sig.position_action
    if action is None:
        return sig.direction
    held = action in (PositionAction.REDUCE, PositionAction.HOLD)
    if held and sig.position_weight is not None:
        return f"{action.label} ({_pct(sig.position_weight, decimals=0)} portfela)"
    return action.label


def _signal_tone(sig: TradeSignal) -> str:
    action = sig.position_action
    return sig.direction if action is None else _POSITION_ACTION_TONE[action]


def _signal_size_band(sig: TradeSignal) -> SizeBand | None:
    """Banda wielkości to sugestia wejścia — przy „UNIKAJ”, „ZREDUKUJ” i „TRZYMAJ,
    nie dokładaj” sprzeczna z kierunkiem, więc jej nie pokazujemy (PORT-08)."""
    if sig.position_action in (None, PositionAction.BUY):
        return sig.size_band
    return None


def _render_trade_signals_html(
    trade_signals: list[TradeSignal], symbol_chips: Mapping[str, str] | None = None
) -> str:
    """🎯 Najsilniejsze sygnały transakcyjne. Pusty string, gdy brak sygnałów.

    `symbol_chips` (WIA-35): trafność kierunkowa na danym symbolu pod wierszem."""
    if not trade_signals:
        return ""
    chips = symbol_chips or {}
    parts = [_section_h2("🎯 Najsilniejsze sygnały")]
    if any(sig.ml_cold_start for sig in trade_signals):
        parts.append(
            "<div style='font-size: 12px; color: #6b7280; margin-bottom: 8px;'>"
            f"{_html(_SIGNALS_COLD_START_NOTE)}</div>"
        )
    if any(sig.position_action is not None for sig in trade_signals):
        parts.append(
            "<div style='font-size: 12px; color: #6b7280; margin-bottom: 8px;'>"
            f"{_html(_SIGNALS_POSITION_NOTE)}</div>"
        )
    for rank, sig in enumerate(trade_signals, start=1):
        dir_color = {
            "KUP": PALETTE.positive,
            "SPRZEDAJ": PALETTE.negative,
            "OBSERWUJ": PALETTE.neutral,
        }[_signal_tone(sig)]
        dir_bg = {
            "KUP": "#f0fdf4",
            "SPRZEDAJ": PALETTE.tint_negative,
            "OBSERWUJ": "#f9fafb",
        }[_signal_tone(sig)]
        forecast_html = (
            '<span style="color: #6b7280;">'
            f"prognoza <strong>{_COLD_START_FORECAST}</strong></span>"
            if sig.ml_cold_start
            else (
                f'<span style="color: {_delta_color(sig.expected_change)};">'
                f"prognoza <strong>{direction_pct(sig.expected_change)}</strong></span>"
            )
        )
        # Q7: pasmo wielkości pozycji (Kelly-lite z konsensusu rady + hit-rate).
        size_band = _signal_size_band(sig)
        size_html = (
            f'<span style="color: #6b7280; font-size: 12px;">'
            f"{_html(size_band.label)}</span><br>"
            if size_band
            else ""
        )
        company = _html(_company_label(sig.symbol))
        # MAIL-23: karta jako tabela trzech komórek `width: 33%` zamiast flexa
        # (Outlook). Bez klas — karty muszą wyglądać tak samo bez arkusza.
        parts.append(f"""
              <table role="presentation" style="width: 100%; margin-bottom: 8px;
                          background: {dir_bg}; border-left: 4px solid {dir_color};
                          border-radius: 4px; border-collapse: collapse; font-size: 13px;">
                <tr>
                  <td style="width: 33%; padding: 10px 14px; vertical-align: top;">
                    <span style="font-weight: 700; color: {dir_color};">
                      {_html(_signal_label(sig))}
                    </span><br>
                    <span style="font-weight: 600;">{company}</span>
                  </td>
                  <td style="width: 33%; padding: 10px 14px; vertical-align: top;">
                    <span style="color: #4b5563;">{_confidence_html(sig)}</span><br>
                    {forecast_html}
                  </td>
                  <td style="width: 33%; padding: 10px 14px; vertical-align: top;">
                    {size_html}
                    <span style="color: #6b7280; font-size: 12px;">
                      {_html(_strength_label(rank, sig))}
                    </span>
                  </td>
                </tr>
              </table>
            """)
        chip = chips.get(sig.symbol)
        if chip:
            parts.append(
                "<div style='font-size: 12px; color: #6b7280; margin: -4px 0 8px 18px;'>"
                f"{_html(chip)}</div>"
            )
    return "".join(parts)


def _render_trade_signals_text(
    trade_signals: list[TradeSignal], symbol_chips: Mapping[str, str] | None = None
) -> list[str]:
    """Sekcja 'Najsilniejsze sygnały' (plain). Pusta lista, gdy brak sygnałów."""
    if not trade_signals:
        return []
    chips = symbol_chips or {}
    lines = ["NAJSILNIEJSZE SYGNAŁY", "-" * 64]
    if any(sig.ml_cold_start for sig in trade_signals):
        lines.append(f"  {_SIGNALS_COLD_START_NOTE}")
    if any(sig.position_action is not None for sig in trade_signals):
        lines.append(f"  {_SIGNALS_POSITION_NOTE}")
    for rank, sig in enumerate(trade_signals, start=1):
        size_band = _signal_size_band(sig)
        size_txt = f"  [{size_band.label}]" if size_band else ""
        forecast = (
            _COLD_START_FORECAST if sig.ml_cold_start else _pct(sig.expected_change, signed=True)
        )
        lines.append(
            f"  {_signal_label(sig):9s} {_company_label(sig.symbol):40s} "
            f"pewność {_confidence_text(sig)}  "
            f"prognoza {forecast:>8s}  "
            f"{_strength_label(rank, sig)}{size_txt}"
        )
        chip = chips.get(sig.symbol)
        if chip:
            lines.append(f"            {chip}")
    lines.append("")
    return lines


def _section_html(data: SectionInput) -> str:
    return _render_trade_signals_html(data.trade_signals, data.symbol_chips)


def _section_text(data: SectionInput) -> list[str]:
    return _render_trade_signals_text(data.trade_signals, data.symbol_chips)


TRADE_SIGNALS_SECTION = ReportSection(
    key="trade_signals", render_html=_section_html, render_text=_section_text
)
