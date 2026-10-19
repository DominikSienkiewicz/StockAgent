"""Renderer zwięzłego digestu push (Telegram/Slack).

Kanały messenger (`TelegramNotifier`/`SlackNotifier`) współdzielą dziś pełny
plain-text maila. Pełny raport regularnie przekracza limit 4096 znaków
Telegrama → API zwraca 400 i push po cichu ginie. Tu budujemy osobny, ~5-liniowy
format z TWARDYM limitem 1500 znaków ZAWSZE (margines bezpieczeństwa pod limit
Telegrama) i stałym zestawem sekcji.

Czysta funkcja render-only: zero I/O, zero portów, zero płatnych wywołań.
Wejście to warstwowe DTO (`SymbolResult`, `ResolvedPrediction`) z
`report_models`. Treść po polsku — to komunikat wysyłany użytkownikowi.
"""

from __future__ import annotations

from datetime import datetime

from src.application.report_formatting import (
    DEFAULT_DISPLAY_TIMEZONE,
    MESSENGER_DISCLAIMER,
    format_local,
)
from src.application.report_models import ResolvedPrediction, SymbolResult

# Twardy limit długości digestu. Ostrzejszy niż 4096 Telegrama — zostawiamy
# margines na ewentualny prefiks/formatowanie kanału i emoji liczone jako
# wiele bajtów.
MAX_DIGEST_CHARS = 1500

# MAIL-02: linia pod nagłówkiem, gdy mail z raportem nie doszedł.
UNDELIVERED_EMAIL_LINE = "⚠️ Mail z raportem nie został dostarczony"

# Sufit liczby sygnałów renderowanych imiennie — reszta trafia do overflow.
# Chroni czytelność i (razem z przycięciem symbolu) gwarantuje mieszczenie się
# w limicie nawet przy 45+ symbolach.
_MAX_SIGNALS = 12

# Maksymalna długość pojedynczego tickera w linii sygnałów. Patologicznie długie
# nazwy (np. skażony symbol) są ucinane, żeby jeden wpis nie zjadł budżetu.
_MAX_SYMBOL_LEN = 12

# Znak wielokropka (jeden znak zamiast trzech kropek — oszczędza budżet).
_ELLIPSIS = "…"

# MAIL-08: ogon z zastrzeżeniem doklejany PO clampie, więc jego długość
# schodzi z budżetu treści — clamp nie może go uciąć.
_DISCLAIMER_TAIL = f"\n{MESSENGER_DISCLAIMER}"

_TREND_ARROWS = {"BULLISH": "📈", "BEARISH": "📉"}
_TREND_NEUTRAL = "➡️"


def _shorten(symbol: str) -> str:
    """Przycina zbyt długi ticker do limitu z wielokropkiem."""
    if len(symbol) <= _MAX_SYMBOL_LEN:
        return symbol
    return symbol[: _MAX_SYMBOL_LEN - 1] + _ELLIPSIS


def _arrow(trend: str | None) -> str:
    """Mapuje trend na strzałkę; nieznany/None → neutralna."""
    if trend is None:
        return _TREND_NEUTRAL
    return _TREND_ARROWS.get(trend.strip().upper(), _TREND_NEUTRAL)


def _header_line(started_at: datetime, display_timezone: str) -> str:
    """Nagłówek: marka + czas startu cyklu w strefie odbiorcy (MAIL-14)."""
    return f"📊 StockAgent — {format_local(started_at, display_timezone)}"


def _cycle_line(results: list[SymbolResult]) -> str:
    """Podsumowanie cyklu: zapisane / pominięte / błędy."""
    saved = sum(1 for r in results if r.status == "saved")
    ignored = sum(1 for r in results if r.status == "ignored")
    errors = sum(1 for r in results if r.status == "error")
    return f"Cykl: {saved} zapisane · {ignored} pominięte · {errors} błędów"


def _scorecard_line(resolved: list[ResolvedPrediction]) -> str:
    """Scorecard ✅/❌ zamkniętych predykcji."""
    if not resolved:
        return "Scorecard: brak zamkniętych predykcji"
    correct = sum(1 for p in resolved if p.is_correct)
    incorrect = len(resolved) - correct
    return f"Scorecard: ✅ {correct} / ❌ {incorrect} ({len(resolved)} zamkniętych)"


def _signals_line(results: list[SymbolResult]) -> str:
    """Linia sygnałów: strzałka + ticker, z ucięciem i podsumowaniem reszty."""
    saved = [r for r in results if r.status == "saved"]
    if not saved:
        return "Sygnały: brak"
    shown = saved[:_MAX_SIGNALS]
    entries = [f"{_arrow(r.trend)} {_shorten(r.symbol)}" for r in shown]
    line = "Sygnały: " + " · ".join(entries)
    remaining = len(saved) - len(shown)
    if remaining > 0:
        line += f" {_ELLIPSIS}+{remaining} więcej"
    return line


def build_messenger_digest(
    results: list[SymbolResult],
    started_at: datetime,
    resolved_predictions: list[ResolvedPrediction] | None = None,
    *,
    email_delivered: bool = True,
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE,
) -> str:
    """Buduje zwięzły digest push (~5 linii) dla Telegrama/Slacka.

    Zestaw sekcji jest stały (nagłówek, cykl, scorecard, sygnały, zastrzeżenie
    MAIL-08) — to zamraża kontrakt formatu przeciw dryfowi. Wynik jest ZAWSZE ≤ `MAX_DIGEST_CHARS`,
    także przy patologicznym wejściu (60 symboli, bardzo długie nazwy): sekcje
    zmienne są ucinane, a na końcu działa twardy clamp jako siatka
    bezpieczeństwa. Pusty cykl zwraca krótki, ustrukturyzowany komunikat z
    zerowymi licznikami (nigdy "").

    Args:
        results: wyniki analizy symboli w cyklu.
        started_at: moment startu cyklu (do nagłówka).
        resolved_predictions: predykcje zamknięte w tym oknie (do scorecardu).
        email_delivered: False, gdy wysyłka maila padła — push mówi o tym zaraz
            pod nagłówkiem, bo przy 401/403 kolejny mail też by nie doszedł.
        display_timezone: strefa (IANA) godziny w nagłówku; `started_at` jest w UTC.

    Returns:
        Gotowy tekst push, ≤ `MAX_DIGEST_CHARS` znaków.
    """
    resolved = resolved_predictions or []
    lines = [
        _header_line(started_at, display_timezone),
        *([] if email_delivered else [UNDELIVERED_EMAIL_LINE]),
        _cycle_line(results),
        _scorecard_line(resolved),
        _signals_line(results),
    ]
    digest = "\n".join(lines)
    # Twardy clamp — ostateczna gwarancja kontraktu długości niezależnie od
    # danych. Sekcje wymagane są na górze, więc przetrwają ewentualne ucięcie;
    # zastrzeżenie ma zarezerwowane miejsce w budżecie.
    budget = MAX_DIGEST_CHARS - len(_DISCLAIMER_TAIL)
    if len(digest) > budget:
        digest = digest[: budget - 1] + _ELLIPSIS
    return digest + _DISCLAIMER_TAIL
