"""Testy kontraktowe dla `build_messenger_digest`.

Największe ryzyko produktowe (roadmap #5): dryf dwóch formatów tekstowych —
pełny raport plain-text vs. zwięzły push do Telegrama/Slacka. Push MUSI zmieścić
się w twardym limicie znaków (Telegram odrzuca > 4096 znaków statusem 400 i push
po cichu ginie), więc trzymamy własny, ostrzejszy limit 1500 znaków ZAWSZE oraz
stały zestaw sekcji. Te testy zamrażają oba niezmienniki.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from src.application.report_messenger import MAX_DIGEST_CHARS, build_messenger_digest
from src.application.report_models import ResolvedPrediction, SymbolResult

_STARTED_AT = datetime(2026, 7, 10, 14, 30)


def _saved(symbol: str, trend: str = "BULLISH") -> SymbolResult:
    """Pojedynczy zapisany wynik z zadanym trendem."""
    return SymbolResult(
        symbol=symbol,
        status="saved",
        trend=trend,
        current_price=Decimal("100"),
        target_price=Decimal("110"),
    )


def _resolved(symbol: str, is_correct: bool) -> ResolvedPrediction:
    """Zamknięta predykcja o znanej trafności."""
    return ResolvedPrediction(
        symbol=symbol,
        predicted_trend="BULLISH",
        is_correct=is_correct,
    )


# --- (a) twardy limit znaków przy patologicznym wejściu -------------------


def test_digest_under_limit_with_pathological_input() -> None:
    """60 symboli, bardzo długie nazwy, dużo zamkniętych predykcji → wynik
    nadal mieści się w twardym limicie."""
    long_name = "X" * 200
    results = [_saved(f"{long_name}-{i}") for i in range(60)]
    resolved = [_resolved(f"{long_name}-{i}", is_correct=i % 2 == 0) for i in range(60)]

    digest = build_messenger_digest(results, _STARTED_AT, resolved)

    assert len(digest) <= MAX_DIGEST_CHARS
    assert len(digest) <= 1500


def test_digest_under_limit_with_single_gigantic_symbol() -> None:
    """Nawet jeden absurdalnie długi symbol nie może przebić limitu."""
    results = [_saved("A" * 5000)]
    digest = build_messenger_digest(results, _STARTED_AT, [])
    assert len(digest) <= 1500


# --- (b) obecność wymaganego zestawu sekcji -------------------------------


def test_digest_contains_required_sections() -> None:
    """Zestaw sekcji jest stały — nagłówek, podsumowanie cyklu, scorecard,
    sygnały. To zamraża kontrakt formatu."""
    results = [_saved("AAPL"), _saved("TSLA", trend="BEARISH")]
    resolved = [_resolved("MSFT", is_correct=True)]

    digest = build_messenger_digest(results, _STARTED_AT, resolved)

    assert "StockAgent" in digest
    assert "Cykl:" in digest
    assert "Scorecard:" in digest
    assert "Sygnały:" in digest


def test_required_sections_survive_pathological_truncation() -> None:
    """Sekcje wymagane są na górze i muszą przetrwać nawet twarde przycięcie."""
    results = [_saved("Z" * 100 + f"-{i}") for i in range(60)]
    digest = build_messenger_digest(results, _STARTED_AT, [])
    for marker in ("StockAgent", "Cykl:", "Scorecard:", "Sygnały:"):
        assert marker in digest


# --- (c) pusty cykl → krótka, sensowna treść (nie "") ----------------------


def test_empty_cycle_returns_short_nonempty_message() -> None:
    """Pusty cykl NIE zwraca "" — zwraca krótki, ustrukturyzowany komunikat.

    Decyzja: pusty push wciąż pełni rolę codziennego 'tickera zaufania'
    (potwierdza, że agent przeszedł cykl) i unika wysłania pustej wiadomości,
    którą Telegram i tak odrzuca. Dlatego zwracamy stałe sekcje z zerowymi
    licznikami zamiast pustego stringa."""
    digest = build_messenger_digest([], _STARTED_AT, [])

    assert digest != ""
    assert len(digest) <= 1500
    assert "StockAgent" in digest
    assert "Cykl:" in digest


# --- Scorecard ✅/❌ ------------------------------------------------------


def test_scorecard_counts_closed_predictions() -> None:
    """Scorecard liczy trafne/nietrafne zamknięte predykcje."""
    resolved = [
        _resolved("AAPL", is_correct=True),
        _resolved("MSFT", is_correct=True),
        _resolved("TSLA", is_correct=False),
    ]
    digest = build_messenger_digest([], _STARTED_AT, resolved)

    assert "✅ 2" in digest
    assert "❌ 1" in digest
    assert "3" in digest  # łączna liczba zamkniętych


def test_scorecard_without_resolved_is_explicit() -> None:
    """Brak zamkniętych predykcji → jawny, krótki komunikat, nie mylący 0/0."""
    digest = build_messenger_digest([_saved("AAPL")], _STARTED_AT, [])
    assert "Scorecard:" in digest
    assert "brak" in digest.lower()


# --- Sygnały: strzałki trendu i overflow ----------------------------------


def test_signals_render_trend_arrows() -> None:
    """BULLISH → 📈, BEARISH → 📉."""
    results = [_saved("AAPL", trend="BULLISH"), _saved("TSLA", trend="BEARISH")]
    digest = build_messenger_digest(results, _STARTED_AT, [])
    assert "📈" in digest
    assert "📉" in digest


def test_signals_overflow_is_summarised() -> None:
    """Przy wielu sygnałach lista jest ucinana z podsumowaniem reszty."""
    results = [_saved(f"SYM{i}") for i in range(40)]
    digest = build_messenger_digest(results, _STARTED_AT, [])
    assert "+" in digest  # marker overflow, np. "…+N"
    assert len(digest) <= 1500


def test_no_saved_signals_is_explicit() -> None:
    """Brak zapisanych sygnałów → sekcja Sygnały nadal obecna, z 'brak'."""
    results = [SymbolResult(symbol="AAPL", status="ignored")]
    digest = build_messenger_digest(results, _STARTED_AT, [])
    assert "Sygnały:" in digest
