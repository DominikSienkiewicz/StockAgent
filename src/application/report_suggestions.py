"""Sekcja 💡 WARTE UWAGI — sugestie peerów z 'gorących' sektorów.

Czysta logika prezentacji, BEZ I/O i bez dodatkowych calli. 'Gorący sektor'
wyznaczamy wyłącznie z danych BIEŻĄCEGO cyklu (Δ12h + sentyment już
policzone dla monitorowanych spółek), a peerów bierzemy z kuratorowanej mapy
`sector_peers` z config.toml (PORT-05: klucze to te same sektory co `sectors`,
pilnuje tego walidator `Settings`). Sekcja pomijana w całości, gdy nie ma
gorących sektorów z wolnymi kandydatami (`render_*` → "").
"""

from __future__ import annotations

import html as _html_mod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from src.application.report_formatting import PALETTE, pct, sector_label
from src.application.report_models import SymbolResult
from src.application.reporting.section import ReportSection, SectionInput, text_block

# Progi 'gorącości' — wyższe niż główna bramka volatility (0.02), by sugestie
# pojawiały się tylko przy realnym ruchu sektora, nie przy szumie.
MOVE_THRESHOLD = Decimal("0.03")
SENTIMENT_THRESHOLD = 0.3
MAX_SECTORS = 3
MAX_CANDIDATES = 3
MAX_DRIVERS = 2


@dataclass(frozen=True)
class SectorSuggestion:
    """Jedna sugestia: gorący sektor + co go rozgrzało + kandydaci-peerzy."""

    sector: str
    drivers: list[tuple[str, Decimal]]  # (symbol, Δ12h) malejąco wg |Δ|
    candidates: list[str]


def build_sector_suggestions(
    results: list[SymbolResult],
    sector_map: Mapping[str, str],
    sector_peers: Mapping[str, Sequence[str]],
) -> list[SectorSuggestion]:
    """Z wyników cyklu buduje listę sugestii dla gorących sektorów akcji.

    Sektor kwalifikuje się, gdy jego najsilniejszy monitorowany ruch
    |Δ12h| ≥ MOVE_THRESHOLD LUB średni |sentyment| ≥ SENTIMENT_THRESHOLD,
    i ma wolnych peerów (spoza symboli z bieżącego cyklu). `sector_map`
    (symbol → sektor) i `sector_peers` (sektor → peerzy) idą z tego samego
    configu, więc nazwy sektorów się zgadzają.
    """
    monitored = {r.symbol for r in results}

    # Grupujemy monitorowane saved-wyniki po sektorze (tylko sektory z peerami,
    # co automatycznie wyklucza ETF/Krypto i symbole nieznane).
    by_sector: dict[str, list[SymbolResult]] = {}
    for r in results:
        if r.status != "saved":
            continue
        sector = sector_label(r.symbol, sector_map)
        if sector is None or sector not in sector_peers:
            continue
        by_sector.setdefault(sector, []).append(r)

    suggestions: list[tuple[Decimal, SectorSuggestion]] = []
    for sector, members in by_sector.items():
        deltas = [abs(m.delta) for m in members if m.delta is not None]
        sentiments = [abs(m.sentiment_score) for m in members if m.sentiment_score is not None]
        max_move = max(deltas, default=Decimal("0"))
        mean_sent = sum(sentiments) / len(sentiments) if sentiments else 0.0

        is_hot = max_move >= MOVE_THRESHOLD or mean_sent >= SENTIMENT_THRESHOLD
        if not is_hot:
            continue

        candidates = [p for p in sector_peers[sector] if p not in monitored][:MAX_CANDIDATES]
        if not candidates:
            continue

        drivers_sorted = sorted(
            (m for m in members if m.delta is not None),
            key=lambda m: abs(m.delta),  # type: ignore[arg-type]
            reverse=True,
        )
        drivers = [(m.symbol, m.delta) for m in drivers_sorted[:MAX_DRIVERS] if m.delta is not None]
        suggestions.append((max_move, SectorSuggestion(sector, drivers, candidates)))

    # Ranking: najmocniejszy ruch pierwszy, remis → alfabetycznie wg sektora.
    suggestions.sort(key=lambda pair: (-pair[0], pair[1].sector))
    return [s for _, s in suggestions[:MAX_SECTORS]]


def _fmt_drivers(drivers: list[tuple[str, Decimal]]) -> str:
    return ", ".join(f"{sym} {pct(delta, signed=True, decimals=1)}" for sym, delta in drivers)


def render_suggestions_html(suggestions: list[SectorSuggestion]) -> str:
    """Render sekcji do HTML. Pusta lista → "" (sekcja znika z maila)."""
    if not suggestions:
        return ""

    def esc(v: str) -> str:
        return _html_mod.escape(v, quote=True)

    rows = []
    for s in suggestions:
        why = esc(_fmt_drivers(s.drivers))
        cands = esc(", ".join(s.candidates))
        rows.append(
            f"""
              <div style="margin-bottom: 6px; padding: 8px 12px; background: {PALETTE.tint_info};
                          border-left: 3px solid #2563eb; border-radius: 4px; font-size: 12px;">
                <strong>{esc(s.sector)}</strong>
                <span style="color: #6b7280;">· gorący ({why})</span><br/>
                <span style="color: #4b5563;">rozważ: <strong>{cands}</strong></span>
              </div>"""
        )
    return (
        "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
        "💡 Warte uwagi (sektory w ruchu)</h2>" + "".join(rows)
    )


def render_suggestions_text(suggestions: list[SectorSuggestion]) -> str:
    """Render sekcji do plain text. Pusta lista → "" (sekcja znika)."""
    if not suggestions:
        return ""
    lines = ["WARTE UWAGI (sektory w ruchu)", "-" * 64]
    for s in suggestions:
        lines.append(f"  {s.sector} — gorący ({_fmt_drivers(s.drivers)})")
        lines.append(f"      rozważ: {', '.join(s.candidates)}")
    return "\n".join(lines)


def _suggestions(data: SectionInput) -> list[SectorSuggestion]:
    return build_sector_suggestions(data.results, data.sector_map, data.context.sector_peers or {})


def _section_html(data: SectionInput) -> str:
    return render_suggestions_html(_suggestions(data))


def _section_text(data: SectionInput) -> list[str]:
    return text_block(render_suggestions_text(_suggestions(data)), spaced=True)


# 💡 Warte uwagi: peerzy sektorów z config.toml (PORT-05) dla ruchów cyklu.
SUGGESTIONS_SECTION = ReportSection(
    key="suggestions",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"sector_peers"}),
)
