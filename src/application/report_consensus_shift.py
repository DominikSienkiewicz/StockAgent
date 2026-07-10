"""Render sekcji "🔄 Zmiany nastawienia" w raporcie e-mail (roadmap #8).

Day-over-day narracja rady: pokazuje, jak zmieniło się nastawienie między
poprzednim a bieżącym cyklem — pełne odwrócenie (FLIP), osłabienie przekonania
(SOFTENING) albo jawnie oznaczone stęchłe porównanie (STALE_COMPARISON).

Czysta logika prezentacji — bierze już-sklasyfikowane obiekty domenowe
`ConsensusShift` (produkowane przez `domain.consensus_shift.detect_shift`) i
produkuje fragment HTML oraz wariant plain text. Render-only, zero I/O i zero
płatnych wywołań.

Kształt wejścia — `list[tuple[str, ConsensusShift]]` (para `(symbol, shift)`):
świadomie NIE bierzemy DTO `SymbolResult` z warstwy application, bo renderer
potrzebuje wyłącznie symbolu (do nagłówka wiersza) i już-policzonej zmiany.
Krotka `(symbol, shift)` jest najlżejszym możliwym kontraktem — orkiestrator
składa ją z tego, co ma pod ręką, a sekcja nie ciągnie zależności do cięższego
DTO. Kolejność listy = kolejność renderu (deterministyczna, bez sortowania).

Sekcja samosupresująca: pokazuje TYLKO realny ruch narracji. Wiersze o
`kind is ShiftKind.STABLE` są odfiltrowywane (rada trzyma kurs — nie ma o czym
pisać); gdy po filtrze nic nie zostaje, `render_*` zwracają "".

Twarde wymaganie: `STALE_COMPARISON` renderuje się z JAWNYM wiekiem porównania
("porównanie z cyklu sprzed N dni", `N = round(gap_days)`). Nigdy nie robimy
dramaturgii ze stęchłych danych — cron Fast Loopa bywa zakomentowany, więc
"wczoraj" potrafi być sprzed tygodnia.
"""

from __future__ import annotations

import html as _html

from src.domain.consensus_shift import ConsensusShift, ShiftKind

# Kolory rekomendacji spójne z sekcją RADA DORADCZA / historią głosów.
_REC_COLOR: dict[str, str] = {
    "BUY": "#16a34a",
    "SELL": "#dc2626",
    "HOLD": "#ca8a04",
}

# Etykieta i kolor akcentu per rodzaj zmiany. FLIP jest najbardziej actionable,
# więc dostaje najmocniejszy (czerwony) akcent; STALE jest wyszarzone, bo celowo
# odbieramy mu dramaturgię.
_KIND_LABEL: dict[ShiftKind, str] = {
    ShiftKind.FLIP: "🔄 Odwrócenie",
    ShiftKind.SOFTENING: "🌗 Osłabienie",
    ShiftKind.STALE_COMPARISON: "🕰️ Stęchłe porównanie",
}

_KIND_COLOR: dict[ShiftKind, str] = {
    ShiftKind.FLIP: "#dc2626",
    ShiftKind.SOFTENING: "#ca8a04",
    ShiftKind.STALE_COMPARISON: "#9ca3af",
}


def _visible(shifts: list[tuple[str, ConsensusShift]]) -> list[tuple[str, ConsensusShift]]:
    """Odfiltrowuje STABLE — pokazujemy tylko realny ruch narracji."""
    return [(sym, s) for sym, s in shifts if s.kind is not ShiftKind.STABLE]


def _stale_days(shift: ConsensusShift) -> int:
    """Wiek porównania w pełnych dniach — do zdania 'sprzed N dni'.

    Zaokrąglamy do najbliższego dnia; bramka STALE działa dopiero powyżej 48h,
    więc wynik jest zawsze >= 2 i nigdy nie wprowadza w błąd zerem.
    """
    return round(shift.gap_days)


def _direction_html(shift: ConsensusShift) -> str:
    """Kolorowa strzałka kierunku 'BUY→SELL' z escapowanymi rekomendacjami."""
    prev = shift.previous_recommendation
    curr = shift.current_recommendation
    prev_color = _REC_COLOR.get(prev, "#374151")
    curr_color = _REC_COLOR.get(curr, "#374151")
    return (
        f"<span style='color:{prev_color};font-weight:600;'>{_html.escape(prev)}</span>"
        "<span style='color:#9ca3af;'>→</span>"
        f"<span style='color:{curr_color};font-weight:600;'>{_html.escape(curr)}</span>"
    )


def _narrative_html(shift: ConsensusShift) -> str:
    """Jednozdaniowa narracja per rodzaj zmiany (fragment HTML)."""
    direction = _direction_html(shift)
    if shift.kind is ShiftKind.FLIP:
        return (
            f"Rada odwróciła się {direction} — "
            f"<strong>{shift.votes_changed}</strong> zmienionych głosów."
        )
    if shift.kind is ShiftKind.SOFTENING:
        return f"Osłabienie przekonania {direction} — {shift.votes_changed} zmienionych głosów."
    # STALE_COMPARISON — jawnie oznaczony wiek, bez dramaturgii.
    days = _stale_days(shift)
    return (
        f"Porównanie z cyklu sprzed <strong>{days}</strong> dni {direction} — "
        "dane zbyt stare, żeby ufać zmianie."
    )


def _render_row_html(symbol: str, shift: ConsensusShift) -> str:
    """Renderuje pojedynczy wiersz tabeli zmian nastawienia."""
    label = _KIND_LABEL.get(shift.kind, shift.kind.value)
    color = _KIND_COLOR.get(shift.kind, "#374151")
    return (
        "<tr>"
        f"<td style='padding: 6px 8px;'><strong>{_html.escape(symbol)}</strong></td>"
        f"<td style='padding: 6px 8px; color: {color}; font-weight: 600;'>{label}</td>"
        f"<td style='padding: 6px 8px;'>{_narrative_html(shift)}</td>"
        "</tr>"
    )


def render_consensus_shift_html(shifts: list[tuple[str, ConsensusShift]]) -> str:
    """Renderuje sekcję HTML "🔄 Zmiany nastawienia" lub "" gdy brak ruchu.

    Wejście: lista par `(symbol, shift)`. Wiersze STABLE są odfiltrowywane;
    gdy po filtrze nic nie zostaje, zwraca "" (samosupresja).
    """
    rows = _visible(shifts)
    if not rows:
        return ""

    body = "".join(_render_row_html(sym, s) for sym, s in rows)
    return (
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>"
        "🔄 Zmiany nastawienia</h2>"
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th style='padding: 6px 8px; text-align: left;'>Symbol</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Rodzaj</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Co się zmieniło</th>"
        "</tr></thead><tbody>"
        f"{body}"
        "</tbody></table>"
    )


def _narrative_text(shift: ConsensusShift) -> str:
    """Jednozdaniowa narracja per rodzaj zmiany (plain text)."""
    direction = f"{shift.previous_recommendation}→{shift.current_recommendation}"
    if shift.kind is ShiftKind.FLIP:
        return f"Odwrócenie {direction} — {shift.votes_changed} zmienionych głosów."
    if shift.kind is ShiftKind.SOFTENING:
        return f"Osłabienie {direction} — {shift.votes_changed} zmienionych głosów."
    days = _stale_days(shift)
    return (
        f"Stęchłe porównanie z cyklu sprzed {days} dni {direction} — "
        "dane zbyt stare, żeby ufać zmianie."
    )


def render_consensus_shift_text(shifts: list[tuple[str, ConsensusShift]]) -> str:
    """Wariant plain-text sekcji "🔄 Zmiany nastawienia" lub "" gdy brak ruchu."""
    rows = _visible(shifts)
    if not rows:
        return ""

    lines: list[str] = ["=== 🔄 Zmiany nastawienia ===", ""]
    for sym, s in rows:
        lines.append(f"  {sym}: {_narrative_text(s)}")
    lines.append("")
    return "\n".join(lines)
