"""Sekcja 💡 WARTE UWAGI — sugestie peerów z 'gorących' sektorów.

Czysta logika prezentacji, BEZ I/O i bez dodatkowych calli. 'Gorący sektor'
wyznaczamy wyłącznie z danych BIEŻĄCEGO cyklu (Δ12h + sentyment już
policzone dla monitorowanych spółek), a peerów bierzemy z kuratorowanej mapy
`PEERS`. Sekcja pomijana w całości, gdy nie ma gorących sektorów z wolnymi
kandydatami (`render_*` → "").
"""

from __future__ import annotations

import html as _html_mod
from dataclasses import dataclass
from decimal import Decimal

from src.application.report_formatting import PEERS, sector_label
from src.application.report_models import SymbolResult

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
) -> list[SectorSuggestion]:
    """Z wyników cyklu buduje listę sugestii dla gorących sektorów akcji.

    Sektor kwalifikuje się, gdy jego najsilniejszy monitorowany ruch
    |Δ12h| ≥ MOVE_THRESHOLD LUB średni |sentyment| ≥ SENTIMENT_THRESHOLD,
    i ma wolnych peerów (spoza symboli z bieżącego cyklu).
    """
    monitored = {r.symbol for r in results}

    # Grupujemy monitorowane saved-wyniki po sektorze (tylko sektory z PEERS,
    # co automatycznie wyklucza ETF/Krypto i symbole nieznane).
    by_sector: dict[str, list[SymbolResult]] = {}
    for r in results:
        if r.status != "saved":
            continue
        sector = sector_label(r.symbol)
        if sector is None or sector not in PEERS:
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

        candidates = [p for p in PEERS[sector] if p not in monitored][:MAX_CANDIDATES]
        if not candidates:
            continue

        drivers_sorted = sorted(
            (m for m in members if m.delta is not None),
            key=lambda m: abs(m.delta),  # type: ignore[arg-type]
            reverse=True,
        )
        drivers = [
            (m.symbol, m.delta) for m in drivers_sorted[:MAX_DRIVERS] if m.delta is not None
        ]
        suggestions.append(
            (max_move, SectorSuggestion(sector, drivers, candidates))
        )

    # Ranking: najmocniejszy ruch pierwszy, remis → alfabetycznie wg sektora.
    suggestions.sort(key=lambda pair: (-pair[0], pair[1].sector))
    return [s for _, s in suggestions[:MAX_SECTORS]]


def _fmt_drivers(drivers: list[tuple[str, Decimal]]) -> str:
    return ", ".join(f"{sym} {delta * 100:+.1f}%" for sym, delta in drivers)


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
              <div style="margin-bottom: 6px; padding: 8px 12px; background: #eff6ff;
                          border-left: 3px solid #2563eb; border-radius: 4px; font-size: 12px;">
                <strong>{esc(s.sector)}</strong>
                <span style="color: #6b7280;">· gorący ({why})</span><br/>
                <span style="color: #4b5563;">rozważ: <strong>{cands}</strong></span>
              </div>"""
        )
    return (
        "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
        "💡 Warte uwagi (sektory w ruchu)</h2>"
        + "".join(rows)
    )


def render_suggestions_text(suggestions: list[SectorSuggestion]) -> str:
    """Render sekcji do plain text. Pusta lista → "" (sekcja znika)."""
    if not suggestions:
        return ""
    lines = ["WARTE UWAGI (sektory w ruchu)", "-" * 64]
    for s in suggestions:
        lines.append(
            f"  {s.sector} — gorący ({_fmt_drivers(s.drivers)})"
        )
        lines.append(f"      rozważ: {', '.join(s.candidates)}")
    return "\n".join(lines)
