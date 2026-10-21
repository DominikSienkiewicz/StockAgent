"""Testy kontraktowe dla `build_messenger_digest`.

Największe ryzyko produktowe (roadmap #5): dryf dwóch formatów tekstowych —
pełny raport plain-text vs. zwięzły push do Telegrama/Slacka. Push MUSI zmieścić
się w twardym limicie znaków (Telegram odrzuca > 4096 znaków statusem 400 i push
po cichu ginie), więc trzymamy własny, ostrzejszy limit 1500 znaków ZAWSZE oraz
stały zestaw sekcji. Te testy zamrażają oba niezmienniki.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_messenger import MAX_DIGEST_CHARS, build_messenger_digest
from src.application.report_models import ResolvedPrediction, SymbolResult
from src.application.report_signals import build_trade_signals
from src.domain.digest_lead import LeadItem

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


def test_digest_warns_when_email_undelivered() -> None:
    # MAIL-02: przy 401/403 kolejny mail też padnie, więc o niedostarczonym
    # raporcie mówi kanał niezależny od Resend.
    started = datetime(2026, 10, 18, 5, 30)
    undelivered = build_messenger_digest([], started, email_delivered=False)
    delivered = build_messenger_digest([], started)

    assert "⚠️ Mail z raportem nie został dostarczony" in undelivered
    assert "nie został dostarczony" not in delivered
    assert len(undelivered) <= MAX_DIGEST_CHARS


def test_header_uses_display_timezone() -> None:
    # MAIL-14: „16.10 05:30” bez strefy wyglądało na godzinę lokalną, a było UTC
    # (2 h różnicy latem, 1 h zimą).
    summer = build_messenger_digest(
        [], datetime(2026, 10, 16, 5, 30, tzinfo=UTC), display_timezone="Europe/Warsaw"
    )
    winter = build_messenger_digest(
        [], datetime(2026, 11, 2, 5, 30, tzinfo=UTC), display_timezone="Europe/Warsaw"
    )

    assert summer.splitlines()[0] == "📊 StockAgent — pt 16.10 07:30"
    assert winter.splitlines()[0] == "📊 StockAgent — pon 02.11 06:30"


# --- MAIL-08: zastrzeżenie inwestycyjne -----------------------------------

_PUSH_DISCLAIMER = "ℹ️ Sygnały modelu nie stanowią rekomendacji inwestycyjnej."


def test_digest_ends_with_short_disclaimer() -> None:
    digest = build_messenger_digest([_saved("AAPL")], _STARTED_AT, [])

    assert digest.splitlines()[-1] == _PUSH_DISCLAIMER


def test_hard_clamp_never_cuts_the_disclaimer() -> None:
    # Zastrzeżenie jest liczone do budżetu PRZED clampem — patologiczny cykl
    # ucina sekcje zmienne, ale ogon z zastrzeżeniem zostaje w całości.
    results = [_saved("Z" * 100 + f"-{i}") for i in range(60)]
    resolved = [_resolved("X" * 200 + f"-{i}", is_correct=True) for i in range(60)]

    digest = build_messenger_digest(results, _STARTED_AT, resolved)

    assert len(digest) <= MAX_DIGEST_CHARS
    assert digest.endswith(_PUSH_DISCLAIMER)



# --- MAIL-25: awaria odczytu rozliczeń ------------------------------------


def test_scorecard_on_read_failure_does_not_claim_nothing_closed() -> None:
    digest = build_messenger_digest([], _STARTED_AT, [], resolved_unavailable=True)

    assert "Scorecard: rozliczenia niedostępne (błąd odczytu)" in digest
    assert "brak zamkniętych predykcji" not in digest


# --- MAIL-20: lead i ranking sygnałów zamiast listy strzałek -----------------


def _ranked_input() -> list[SymbolResult]:
    """Watchlista A, B, C; siła = pewność × |ruch| × 100 → A 1,0 · B 5,0 · C 3,0."""
    return [
        SymbolResult(symbol="AAA", status="saved", trend="BULLISH", confidence_score=0.5,
                     current_price=Decimal("100"), target_price=Decimal("102")),
        SymbolResult(symbol="BBB", status="saved", trend="BEARISH", confidence_score=0.5,
                     current_price=Decimal("100"), target_price=Decimal("90")),
        SymbolResult(symbol="CCC", status="saved", trend="BULLISH", confidence_score=0.5,
                     current_price=Decimal("100"), target_price=Decimal("106")),
    ]


def _signals_line(digest: str) -> str:
    return next(line for line in digest.splitlines() if line.startswith("Sygnały:"))


def test_digest_lists_signals_in_ranking_order() -> None:
    results = _ranked_input()
    ranked = build_trade_signals(results)

    line = _signals_line(
        build_messenger_digest(results, _STARTED_AT, [], ranked_signals=ranked)
    )

    assert [s.strength for s in ranked] == [5.0, 3.0, 1.0]
    assert line.index("BBB") < line.index("CCC") < line.index("AAA")


def test_digest_shows_at_most_three_signals_with_calibrated_confidence() -> None:
    results = [*_ranked_input(), _saved("DDD")]
    ranked = [
        dataclasses.replace(sig, calibrated_confidence=0.41) if sig.symbol == "BBB" else sig
        for sig in build_trade_signals(results)
    ]

    line = _signals_line(
        build_messenger_digest(results, _STARTED_AT, [], ranked_signals=ranked)
    )

    assert line.startswith("Sygnały: 1. 📉 BBB 41%")
    assert "2. 📈 CCC 50%" in line
    assert "3. 📈 AAA 50%" in line
    assert "DDD" not in line
    assert line.endswith("…+1 więcej")


def test_digest_opens_with_up_to_three_lead_items() -> None:
    lead = [
        LeadItem(icon="⚖️", headline=f"S{i} -4,2%, rada PODZIELONA", detail="x")
        for i in range(4)
    ]

    lines = build_messenger_digest([], _STARTED_AT, [], lead_items=lead).splitlines()

    assert lines[1:4] == [f"⚖️ S{i} -4,2%, rada PODZIELONA" for i in range(3)]
    assert "S3" not in "\n".join(lines)
    assert lines[4].startswith("Cykl:")


def test_long_lead_headline_is_clipped_and_keeps_the_limit() -> None:
    lead = [LeadItem(icon="🔥", headline="N" * 2000, detail="")]
    results = [_saved("Z" * 100 + f"-{i}") for i in range(60)]

    digest = build_messenger_digest(results, _STARTED_AT, [], lead_items=lead)

    assert len(digest) <= MAX_DIGEST_CHARS
    for marker in ("StockAgent", "Cykl:", "Scorecard:", "Sygnały:"):
        assert marker in digest
