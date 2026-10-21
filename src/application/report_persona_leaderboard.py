"""Render sekcji "Rada — ranking wiarygodności" w raporcie e-mail (#3).

Persona leaderboard: sekcja historii głosów pokazuje, KTO jak głosował — ta
pokazuje, KTO MIAŁ RACJĘ. "Buffett — 68% trafności (22 głosy)" zamienia radę
7 person z teatru LLM w audytowalne track-recordy.

Czysta logika prezentacji — bierze gotowy, już posortowany i odszumiony ranking
z domeny (`rank_personas`) i produkuje fragment HTML. Render-only, zero płatnych
wywołań: wszystko liczone z już-zapisanych, rozliczonych głosów w bazie.

Sekcja samosupresująca: brak rekordów (bo brak danych albo żadna persona nie
przekroczyła progu `min_votes`) → `render_*` zwraca "".
"""

from __future__ import annotations

import html as _html

from src.application.report_evidence import interval_label
from src.domain.number_format import DecimalMark
from src.domain.persona_track_record import PersonaTrackRecord

# Medal dla lidera — tylko dla pierwszego miejsca. Przy 2-3 personach podium
# z brązem wyglądałoby jak nagroda pocieszenia, nie jak ranking.
_LEADER_MEDAL = "🥇"

# Progi kolorowania hit-rate'u. Rada głosuje BUY/SELL/HOLD, więc losowy strzał
# to ~33% — 55% to realnie dobry wynik, poniżej 45% persona szkodzi.
_HIT_RATE_GOOD = 0.55
_HIT_RATE_POOR = 0.45

# WIA-22: jedno zdanie o przybliżeniu — RPC person nie zwraca cykli ani ruchów
# cen, więc baseline i n_eff pochodzą z predykcji okna scorecardu.
_VERDICT_NOTE_PL = (
    " Werdykt: test dwumianowy względem najlepszej naiwnej strategii („zawsze "
    "KUP / SPRZEDAJ / TRZYMAJ”) z korektą BH-FDR po personach, a kolejność po "
    "dolnej granicy przedziału — to przybliżenie: baseline i n_eff liczymy "
    "z predykcji okna track recordu, nie z samych głosów person."
)

_COLOR_GOOD = "#16a34a"
_COLOR_NEUTRAL = "#ca8a04"
_COLOR_POOR = "#dc2626"


def _hit_rate_color(hit_rate: float) -> str:
    if hit_rate >= _HIT_RATE_GOOD:
        return _COLOR_GOOD
    if hit_rate < _HIT_RATE_POOR:
        return _COLOR_POOR
    return _COLOR_NEUTRAL


def _votes_label(votes: int) -> str:
    """Polska odmiana rzeczownika "głos" przez liczebnik.

    1 głos / 2-4 głosy / 5+ głosów, z wyjątkiem nastek (12-14 → "głosów").
    Odmiana idzie po KOŃCÓWCE liczby, więc 22 → "głosy", a 25 → "głosów".
    """
    if votes == 1:
        return "1 głos"
    last_two = votes % 100
    last_one = votes % 10
    if 2 <= last_one <= 4 and not 12 <= last_two <= 14:
        return f"{votes} głosy"
    return f"{votes} głosów"


def render_persona_leaderboard_html(records: list[PersonaTrackRecord]) -> str:
    """Renderuje sekcję HTML rankingu person lub "" gdy brak rekordów."""
    if not records:
        return ""

    rows = [
        _render_row_html(record, position=position)
        for position, record in enumerate(records, start=1)
    ]
    note = _VERDICT_NOTE_PL if any(r.p_value is not None for r in records) else ""

    return (
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>"
        "🏆 Rada — ranking wiarygodności</h2>"
        "<div style='font-size: 11px; color: #6b7280; margin-bottom: 8px;'>"
        "Trafność rekomendacji vs realny ruch ceny na rozliczonych predykcjach. "
        "W nawiasie 95% przedział Wilsona: przy kilkunastu głosach różnice "
        "między personami zwykle mieszczą się w szumie."
        f"{_html.escape(note)}</div>"
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>#</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Inwestor</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Trafność</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Próbka</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_row_html(record: PersonaTrackRecord, *, position: int) -> str:
    name = _html.escape(record.investor_name)
    color = _hit_rate_color(record.hit_rate)
    medal = f"{_LEADER_MEDAL} " if position == 1 else ""
    # WIA-22: przedział na n_eff, gdy znane — ten sam, po którym idzie ranking.
    low, high = record.wilson_interval
    verdict = record.verdict_in(DecimalMark.COMMA)
    verdict_html = (
        ""
        if verdict is None
        else "<div style='font-size: 11px; color: #6b7280; font-weight: 400;'>"
        f"{_html.escape(verdict)}</div>"
    )
    n_eff = "" if record.n_eff is None else f" (n_eff≈{record.n_eff:.0f})"
    return (
        "<tr>"
        f"<td style='padding: 6px 8px; color: #9ca3af;'>{position}</td>"
        f"<td style='padding: 6px 8px;'>{medal}<strong>{name}</strong></td>"
        f"<td style='padding: 6px 8px; color: {color}; font-weight: 600;'>"
        f"{record.hit_rate_pct}% <span style='color: #9ca3af; font-weight: 400;'>"
        f"({interval_label(low, high)})</span>{verdict_html}</td>"
        f"<td style='padding: 6px 8px; color: #6b7280;'>"
        f"{_votes_label(record.votes)}{n_eff}</td>"
        "</tr>"
    )
