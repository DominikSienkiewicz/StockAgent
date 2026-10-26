"""Render sekcji "Rada — historia głosów" w raporcie e-mail (Feature U5).

Council-vote drill-down: per inwestor pokazuje jego ostatnie rekomendacje na
obserwowanym symbolu ("Soros — ostatnie 30 dni na NVDA: SELL, SELL, HOLD —
najbardziej bearish głos") + zwięzły tally BUY/SELL/HOLD.

Czysta logika prezentacji — bierze już-zmaterializowany audit trail z tabeli
`council_votes` (jako lista `InvestorHistory`, którą orkiestrator produkuje z
`RepositoryPort.get_council_vote_history(...)`) i produkuje fragment HTML oraz
wariant plain text. Render-only, zero płatnych wywołań — wszystko liczone z
już-zapisanych głosów.

Sekcja samosupresująca: gdy brak historii (lub wszystkie historie puste),
`render_*` zwracają "".
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass

from src.application.report_formatting import PALETTE
from src.application.reporting.section import ReportSection, SectionInput, text_block
from src.domain.council import Recommendation
from src.domain.polish_labels import council_recommendation_label

_RECOMMENDATIONS: tuple[Recommendation, ...] = ("BUY", "SELL", "HOLD")

# Kolory rekomendacji spójne z sekcją RADA DORADCZA (council_section.html.j2).
_REC_COLOR: dict[str, str] = {
    "BUY": PALETTE.positive,
    "SELL": PALETTE.negative,
    "HOLD": PALETTE.caution,
}

_REC_EMOJI: dict[str, str] = {
    "BUY": "🟢",
    "SELL": "🔴",
    "HOLD": "🟡",
}

# Punktacja "bearish-ości": SELL ciągnie w dół, BUY w górę, HOLD neutralnie.
# Najbardziej bearish = najniższy wynik (suma po głosach w oknie).
_BEARISH_SCORE: dict[str, int] = {"SELL": -1, "HOLD": 0, "BUY": 1}


@dataclass(frozen=True)
class InvestorHistory:
    """Historia głosów jednego inwestora na jednym symbolu w oknie czasowym.

    `recommendations` — krotka od NAJNOWSZEGO do najstarszego głosu (taka
    kolejność, w jakiej zwraca je SQL z `ORDER BY timestamp DESC`). Pusta
    krotka = inwestor nie głosował na ten symbol w oknie → wiersz pomijany.
    `window_days` — szerokość okna (np. 30 dla "ostatnie 30 dni").
    """

    investor_name: str
    symbol: str
    recommendations: tuple[Recommendation, ...]
    window_days: int = 30

    def __post_init__(self) -> None:
        # Krotka zamiast listy — realna niezmienność frozen value objectu
        # (i przyjmuje listę od callera dla wstecznej kompatybilności).
        object.__setattr__(self, "recommendations", tuple(self.recommendations))

    def tally(self) -> dict[str, int]:
        """Liczba głosów na każdą rekomendację (zawsze 3 deterministyczne klucze)."""
        dist: dict[str, int] = dict.fromkeys(_RECOMMENDATIONS, 0)
        for rec in self.recommendations:
            if rec in dist:
                dist[rec] += 1
        return dist

    def bearish_score(self) -> int:
        """Wynik nastawienia: ujemny = bearish, dodatni = bullish, 0 = neutralny."""
        return sum(_BEARISH_SCORE.get(rec, 0) for rec in self.recommendations)


def _non_empty(histories: list[InvestorHistory]) -> list[InvestorHistory]:
    """Odfiltrowuje historie bez głosów w oknie — pusty wiersz nic nie wnosi."""
    return [h for h in histories if h.recommendations]


def _bearish_tag(histories: list[InvestorHistory]) -> str | None:
    """Inwestor z najniższym (najbardziej bearish) wynikiem — tylko gdy jest
    jednoznacznie 'najbardziej bearish' (przegłosował SELL) i nie remisuje."""
    if not histories:
        return None
    scores = [h.bearish_score() for h in histories]
    min_score = min(scores)
    # Bez sensu oznaczać "najbardziej bearish", gdy nikt nie jest faktycznie
    # bearish (min_score >= 0) albo gdy remis (≥2 inwestorów z tym wynikiem).
    if min_score >= 0:
        return None
    candidates = [h for h, s in zip(histories, scores, strict=True) if s == min_score]
    if len(candidates) != 1:
        return None
    return candidates[0].investor_name


def render_council_history_html(histories: list[InvestorHistory]) -> str:
    """Renderuje sekcję HTML "Rada — historia głosów" lub "" gdy brak danych."""
    rows = _non_empty(histories)
    if not rows:
        return ""

    most_bearish = _bearish_tag(rows)

    parts: list[str] = []
    parts.append("<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>🗳️ Rada — historia głosów</h2>")

    body: list[str] = []
    for h in rows:
        body.append(_render_history_row_html(h, is_most_bearish=h.investor_name == most_bearish))

    parts.append(
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Inwestor</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Symbol</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Ostatnie głosy</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Bilans</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )
    return "".join(parts)


def _render_history_row_html(h: InvestorHistory, *, is_most_bearish: bool) -> str:
    name = _html.escape(h.investor_name)
    symbol = _html.escape(h.symbol)
    seq = _render_sequence_html(h.recommendations)
    tally = h.tally()
    tally_str = f"🟢 {tally['BUY']} · 🔴 {tally['SELL']} · 🟡 {tally['HOLD']}"
    bearish_badge = ""
    if is_most_bearish:
        bearish_badge = (
            " <span style='background:#fee2e2;color:#991b1b;padding:1px 6px;"
            "border-radius:10px;font-size:0.78em;font-weight:600;'>"
            "najbardziej pesymistyczny</span>"
        )
    return (
        "<tr>"
        f"<td style='padding: 6px 8px;'><strong>{name}</strong>{bearish_badge}</td>"
        f"<td style='padding: 6px 8px; color: #6b7280;'>{symbol}</td>"
        f"<td style='padding: 6px 8px;'>"
        f"<span style='color:{PALETTE.subtle};'>{h.window_days}d:</span> {seq}</td>"
        f"<td style='padding: 6px 8px; color: #6b7280;'>{tally_str}</td>"
        "</tr>"
    )


def _render_sequence_html(recommendations: tuple[Recommendation, ...]) -> str:
    """Kolorowa sekwencja "SPRZEDAJ, SPRZEDAJ, TRZYMAJ" od najnowszego głosu (MAIL-61)."""
    chips: list[str] = []
    for rec in recommendations:
        color = _REC_COLOR.get(rec, "#374151")
        chips.append(
            f"<span style='color:{color};font-weight:600;'>"
            f"{_html.escape(council_recommendation_label(rec))}</span>"
        )
    return ", ".join(chips)


def render_council_history_text(histories: list[InvestorHistory]) -> str:
    """Wariant plain-text sekcji historii głosów lub "" gdy brak danych."""
    rows = _non_empty(histories)
    if not rows:
        return ""

    most_bearish = _bearish_tag(rows)

    lines: list[str] = []
    lines.append("=== 🗳️ Rada — historia głosów ===")
    lines.append("")
    for h in rows:
        seq = ", ".join(council_recommendation_label(rec) for rec in h.recommendations)
        tally = h.tally()
        suffix = " — najbardziej pesymistyczny głos" if h.investor_name == most_bearish else ""
        lines.append(
            f"  {h.investor_name} — ostatnie {h.window_days} dni na {h.symbol}: {seq}{suffix}"
        )
        lines.append(
            f"    Bilans: {council_recommendation_label('BUY')} {tally['BUY']} · "
            f"{council_recommendation_label('SELL')} {tally['SELL']} · "
            f"{council_recommendation_label('HOLD')} {tally['HOLD']}"
        )
    lines.append("")
    return "\n".join(lines)


def _section_html(data: SectionInput) -> str:
    return render_council_history_html(data.context.council_history or [])


def _section_text(data: SectionInput) -> list[str]:
    return text_block(render_council_history_text(data.context.council_history or []))


# U5: historia głosów rady per inwestor.
COUNCIL_HISTORY_SECTION = ReportSection(
    key="council_history",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"council_history"}),
)
