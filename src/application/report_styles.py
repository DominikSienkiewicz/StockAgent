"""Arkusz stylów maila (MAIL-06 c): klasy zamiast powtarzanych stylów inline.

Gmail przycina wiadomość powyżej ~102 KB, a atrybuty `style` to ponad połowa
bajtów maila. Style elementów, które powtarzają się per symbol (wiersze tabeli
predykcji, słupki wykresów, chipy kwadrantów, rada doradcza, newsy, skrót
pozostałych predykcji), żyją tu jako klasy — razem z tonem kierunku (wzrost /
spadek / bez zmian), liczonym z tej samej palety co `delta_color`.

Inline zostaje to, co musi wyglądać tak samo bez arkusza — Gmail w aplikacji
dla kont spoza Google pomija `<style>`, a Outlook desktop obsługuje tylko proste
selektory: nagłówek, lead, pierwszy ekran, pasek wiarygodności, karty sygnałów,
nagłówki sekcji i stopka z zastrzeżeniem. Wymiar liczony z danych (szerokość
słupka) też zostaje inline.
"""

from __future__ import annotations

from decimal import Decimal

from src.application.report_formatting import PALETTE

# Ton kierunku: klasa koloru tekstu (`up`/`dn`/`nt`), wypełnienia słupka
# (`fu`/`fd`/`fn`) i tła wiersza (`gu`/`gd`/`gn`). `cu` = ostrożność (HOLD rady).
_TONE_RULES = (
    f".up{{color:{PALETTE.positive}}}"
    f".dn{{color:{PALETTE.negative}}}"
    f".nt{{color:{PALETTE.neutral}}}"
    f".cu{{color:{PALETTE.caution}}}"
    f".fu{{background:{PALETTE.positive}}}"
    f".fd{{background:{PALETTE.negative}}}"
    f".fn{{background:{PALETTE.neutral}}}"
    f".fc{{background:{PALETTE.caution}}}"
    ".gu{background:#f0fdf4}"
    ".gd{background:#fef2f2}"
    ".gc{background:#fefce8}"
    f".gn{{background:{PALETTE.surface}}}"
    f".lu{{border-left:3px solid {PALETTE.positive}}}"
    f".ld{{border-left:3px solid {PALETTE.negative}}}"
    f".ln{{border-left:3px solid {PALETTE.neutral}}}"
)

# Selektory wyłącznie klasowe: obsługują je i Gmail, i Outlook desktop.
EMAIL_STYLESHEET = (
    "<style>"
    + _TONE_RULES
    # Tabela predykcji (🔮) i skrót pozostałych predykcji (📋).
    + ".pt{width:100%;border-collapse:collapse;font-size:13px}"
    ".kt{width:100%;border-collapse:collapse;font-size:12px}"
    ".pr{background:#f3f4f6;text-align:left}"
    ".ph{padding:6px 8px;border-bottom:1px solid #e5e7eb}"
    ".kn{font-size:12px;color:#6b7280;margin-bottom:6px}"
    ".pc{padding:6px 8px;border-bottom:1px solid #f3f4f6}"
    ".pk{padding:4px 8px;border-bottom:1px solid #f3f4f6;vertical-align:top}"
    ".b6{font-weight:600}"
    ".b7{font-weight:700}"
    # Blok uzasadnień (💡).
    ".rb{margin-bottom:10px;padding:10px 12px;background:#fafafa;border-radius:4px}"
    ".rh{font-weight:600;font-size:13px}"
    ".rm{font-size:11px;color:#6b7280;margin-top:2px}"
    ".rs{margin-top:6px;padding:6px 10px;border-radius:3px;font-size:12px}"
    ".rt{font-size:12px;color:#4b5563;margin-top:6px}"
    # Top newsy.
    ".nw{font-size:11px;color:#4b5563;margin-top:8px}"
    ".nl{margin:4px 0 0 20px;padding:0}"
    ".ni{margin:2px 0}"
    ".na{color:#2563eb;text-decoration:none}"
    ".nm{color:#6b7280}"
    # Rada doradcza: nagłówek z werdyktem i tabela person.
    ".cx{margin-top:16px;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden}"
    ".cy{padding:12px 16px;border-bottom:1px solid #e5e7eb}"
    ".ck{font-weight:700;font-size:1.05em}"
    ".cv{color:#fff;padding:2px 10px;border-radius:12px;font-weight:700}"
    ".cs{color:#555;font-size:0.9em}"
    ".cg{background:#dcfce7;color:#166534;padding:2px 8px;border-radius:10px;"
    "font-size:0.8em;font-weight:600}"
    ".cp{background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:10px;"
    "font-size:0.8em;font-weight:600}"
    ".cz{margin-top:6px;font-size:0.85em;color:#4b5563}"
    ".cq{margin-top:6px;font-size:0.9em;color:#374151;font-style:italic}"
    ".ct{width:100%;border-collapse:collapse;font-size:0.9em}"
    ".gh{background:#f3f4f6}"
    ".tc{text-align:center}"
    ".ch{padding:6px 10px;font-weight:600;text-align:left}"
    ".cn{padding:6px 10px;font-weight:500}"
    ".co{padding:6px 10px;font-weight:700}"
    ".cc{padding:6px 10px;text-align:center}"
    f".cl{{font-size:0.7em;color:{PALETTE.muted}}}"
    ".cf{padding:6px 10px;font-size:0.85em;color:#555}"
    ".cd{padding:8px 10px;font-size:0.85em;color:#92400e;background:#fffbeb}"
    ".dv{margin-top:4px;color:#92400e}"
    # Wykresy słupkowe (Δ, prognoza) i tabela kwadrantów.
    ".gw{margin:16px 0}"
    ".gt{font-size:14px;font-weight:600;margin-bottom:8px;color:#1f2937}"
    ".bt{border-collapse:collapse;width:100%}"
    ".qt{border-collapse:collapse;width:100%;table-layout:fixed}"
    ".qh{padding:6px 8px;font-size:11px;color:#6b7280;text-transform:uppercase;"
    "border:1px solid #e5e7eb;background:#f9fafb}"
    ".ql{padding:8px;font-size:12px;font-weight:600;color:#374151;"
    "border:1px solid #e5e7eb;background:#f9fafb;white-space:nowrap}"
    ".qd{padding:8px;border:1px solid #e5e7eb;vertical-align:top;width:40%}"
    ".bl{padding:2px 8px 2px 0;font-size:12px;white-space:nowrap;color:#374151}"
    ".bv{padding:2px 0}"
    ".br{display:inline-block;height:12px;border-radius:2px;vertical-align:middle}"
    ".bn{font-size:12px;color:#374151;margin-left:6px}"
    ".qc{display:inline-block;margin:2px 4px 2px 0;padding:1px 6px;border-radius:3px;"
    "font-size:12px;background:#f3f4f6}"
    # MAIL-23: telefon (~360 px) — wiersz tabeli predykcji to blok z komórkami
    # w jednej zawijanej linii, bez wiersza nagłówka.
    "@media (max-width:480px){"
    ".pt tr{display:block;border-bottom:1px solid #e5e7eb;padding:4px 0}"
    ".pt .pr{display:none}"
    ".pt td{display:inline-block;border:0;padding:2px 6px}"
    "}"
    "</style>"
)


def delta_tone(delta: Decimal | None) -> str:
    """Klasa tonu zmiany — ta sama reguła co `delta_color` (0 i brak = bez zmian)."""
    if delta is None or delta == 0:
        return "nt"
    return "up" if delta > 0 else "dn"


def trend_tone(trend: str | None) -> str:
    """Klasa tonu trendu — ta sama reguła co `trend_color`."""
    return {"BULLISH": "up", "BEARISH": "dn"}.get(trend or "", "nt")


def fill_tone(delta: Decimal | None) -> str:
    """Klasa wypełnienia słupka wykresu dla zmiany `delta`."""
    return {"up": "fu", "dn": "fd"}.get(delta_tone(delta), "fn")


# Sygnał modelu (KUP/SPRZEDAJ/TRZYMAJ) i werdykt rady (BUY/SELL/HOLD).
SIGNAL_TONE = {"KUP": "up", "SPRZEDAJ": "dn", "TRZYMAJ": "nt", "OBSERWUJ": "nt"}
COUNCIL_TONE = {"BUY": "up", "SELL": "dn", "HOLD": "cu"}
COUNCIL_ROW_TONE = {"BUY": "gu", "SELL": "gd", "HOLD": "gc"}
COUNCIL_FILL = {"BUY": "fu", "SELL": "fd", "HOLD": "fc"}
# Lewa krawędź bloku uzasadnień w kolorze trendu.
EDGE_TONE = {"up": "lu", "dn": "ld", "nt": "ln"}
