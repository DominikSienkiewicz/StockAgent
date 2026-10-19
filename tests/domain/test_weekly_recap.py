"""Testy domeny niedzielnej retrospektywy „Tydzień StockAgenta" (roadmap #17).

Deterministyczny wybór bohaterów tygodnia (strzał / wtopa po `accuracy_score`),
reużycie `vindicated_dissenters` z rady oraz delta ECE z krzywej kalibracji.
Reguła domenowa: poniżej progu N zamkniętych predykcji recap NIE POWSTAJE
(mail NIE WYCHODZI) — zamrożona osobnym testem. Zero LLM, zero I/O.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

from src.domain.council import InvestorOpinion
from src.domain.prediction import Prediction, TrendDirection
from src.domain.weekly_recap import (
    MIN_CLOSED_PREDICTIONS_FOR_RECAP,
    ClosedPrediction,
    RecapHighlight,
    WeeklyRecap,
    build_weekly_recap,
)


def _opinion(
    name: str, rec: str, confidence: float = 0.8
) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name=name,
        recommendation=rec,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning="test",
        key_factors=(),
    )


def _closed(
    symbol: str,
    *,
    trend: TrendDirection = TrendDirection.BULLISH,
    base: str = "100",
    target: str = "110",
    actual: str = "110",
    confidence: float = 0.8,
    insight: str = "",
    opinions: tuple[InvestorOpinion, ...] = (),
    council_rec: str = "HOLD",
) -> ClosedPrediction:
    return ClosedPrediction(
        prediction=Prediction(
            symbol=symbol,
            predicted_trend=trend,
            price_at_prediction=Decimal(base),
            predicted_target_price=Decimal(target),
        ),
        actual_price=Decimal(actual),
        confidence=confidence,
        correction_insight=insight,
        council_opinions=opinions,
        council_recommendation=council_rec,  # type: ignore[arg-type]
    )


def _n_closed(n: int) -> list[ClosedPrediction]:
    # n różnych predykcji-wypełniaczy o accuracy 0.95 (błąd 5%) — celowo NIE
    # 1.0/0.5, by nie remisowały z bohaterami testów.
    return [_closed(f"S{i}", target="105", actual="100") for i in range(n)]


class TestSampleThresholdGate:
    def test_below_threshold_no_recap_email_goes_out(self) -> None:
        # REGUŁA DOMENOWA (roadmap #17, „wymaganie, nie opcja"): przy < N
        # zamkniętych predykcji recap NIE POWSTAJE → mail NIE WYCHODZI.
        # Chudy tydzień nie może robić dramatu z szumu.
        thin_week = _n_closed(MIN_CLOSED_PREDICTIONS_FOR_RECAP - 1)
        assert build_weekly_recap(thin_week) is None

    def test_at_threshold_recap_is_built(self) -> None:
        # Dokładnie na progu recap powstaje (próg to N, nie N+1).
        recap = build_weekly_recap(_n_closed(MIN_CLOSED_PREDICTIONS_FOR_RECAP))
        assert isinstance(recap, WeeklyRecap)

    def test_empty_week_no_recap(self) -> None:
        assert build_weekly_recap([]) is None

    def test_threshold_constant_is_five(self) -> None:
        # Próg jest jawną stałą modułu = 5 (roadmap #17).
        assert MIN_CLOSED_PREDICTIONS_FOR_RECAP == 5


class TestHeroSelection:
    def test_best_and_worst_by_accuracy_score(self) -> None:
        # Strzał tygodnia = najwyższy accuracy_score; wtopa = najniższy.
        predictions = [
            _closed("HIT", base="100", target="100", actual="100"),  # accuracy 1.0
            _closed("MISS", base="100", target="150", actual="100"),  # accuracy 0.5
            *_n_closed(3),  # dopełnienie progu (accuracy 0.95)
        ]
        recap = build_weekly_recap(predictions)
        assert recap is not None
        assert recap.best.symbol == "HIT"
        assert recap.worst.symbol == "MISS"
        assert recap.best.accuracy == Decimal("1.0")
        assert recap.worst.accuracy == Decimal("0.5")

    def test_worst_carries_correction_insight_lesson(self) -> None:
        predictions = [
            _closed(
                "MISS",
                base="100",
                target="150",
                actual="100",
                insight="Zignorowano wynik kwartalny — lekcja: czytaj earningsy.",
            ),
            *_n_closed(4),
        ]
        recap = build_weekly_recap(predictions)
        assert recap is not None
        assert recap.worst.symbol == "MISS"
        assert "earningsy" in recap.worst.correction_insight

    def test_tie_break_is_deterministic_by_symbol(self) -> None:
        # Dwie predykcje o identycznym accuracy — remis rozstrzygany
        # alfabetycznie po symbolu, więc wynik jest deterministyczny.
        predictions = [
            _closed("BBB", base="100", target="100", actual="100"),
            _closed("AAA", base="100", target="100", actual="100"),
            *_n_closed(3),
        ]
        recap = build_weekly_recap(predictions)
        assert recap is not None
        # Najwyższe accuracy (1.0) mają AAA i BBB; deterministycznie AAA.
        assert recap.best.symbol == "AAA"

    def test_best_of_week_is_never_a_wrong_direction_call(self) -> None:
        # WIA-07 (G40): pudło kierunkowe bliskie celu (A, accuracy 0.993) nie
        # może zostać strzałem tygodnia, gdy jest trafienie kierunkowe (B, 0.985).
        predictions = [
            _closed(
                "A",
                trend=TrendDirection.BEARISH,
                base="100",
                target="99.5",
                actual="100.2",
            ),
            _closed("B", base="100", target="101.5", actual="103"),
            *_n_closed(3),
        ]
        recap = build_weekly_recap(predictions)
        assert recap is not None
        assert recap.best.symbol == "B"
        assert recap.best.direction_correct is True

    def test_week_without_direction_hits_falls_back_to_closest_miss(self) -> None:
        # Bez trafień kierunkowych bohater to pudło najbliższe celu — oznaczone
        # jako chybione, żeby render nie nazwał go strzałem.
        predictions = [
            _closed("NEAR", base="100", target="100.5", actual="100"),
            *_n_closed(4),
        ]
        recap = build_weekly_recap(predictions)
        assert recap is not None
        assert recap.best.symbol == "NEAR"
        assert recap.best.direction_correct is False

    def test_highlights_carry_direction_verdict(self) -> None:
        predictions = [
            _closed("HIT", base="100", target="105", actual="105"),
            _closed("MISS", base="100", target="150", actual="90"),
            *_n_closed(3),
        ]
        recap = build_weekly_recap(predictions)
        assert recap is not None
        assert recap.best.direction_correct is True
        assert recap.worst.symbol == "MISS"
        assert recap.worst.direction_correct is False

    def test_lone_hit_with_lowest_accuracy_is_not_also_the_miss_of_week(self) -> None:
        # Decyzja 2026-10-19 (a): wtopa = lustro klucza strzału — najpierw pudła
        # kierunkowe, potem najniższe accuracy. Jedyne trafienie (accuracy 0.51)
        # było strzałem i zarazem wtopą, bo miało najniższe accuracy.
        predictions = [
            _closed("HIT", base="100", target="150", actual="101"),
            *_n_closed(4),
        ]
        recap = build_weekly_recap(predictions)
        assert recap is not None
        assert recap.best.symbol == "HIT"
        assert recap.worst.symbol != recap.best.symbol
        assert recap.worst.direction_correct is False


def _recorded(closed: ClosedPrediction, verdict: bool) -> ClosedPrediction:
    return dataclasses.replace(closed, recorded_trend_correct=verdict)


def _btc_sideways_up_1_5pct() -> ClosedPrediction:
    # SIDEWAYS, ruch +1.5%: pasmo akcyjne ±0.5% mówi „pudło”, krypto ±2.5% „trafienie”.
    return _closed(
        "BTC", trend=TrendDirection.SIDEWAYS, base="100", target="100", actual="101.5"
    )


class TestRecordedDirectionVerdict:
    """Decyzja 2026-10-19 (a): werdykt kierunku recapu to zapisany
    `is_trend_correct` z reflect_node (krypto: pasmo SIDEWAYS ±2.5%), a nie
    ponowne liczenie z cen pasmem akcyjnym ±0.5%. Bez zapisu → ceny."""

    def test_recorded_crypto_sideways_hit_counts_as_hit(self) -> None:
        assert _recorded(_btc_sideways_up_1_5pct(), True).is_correct() is True

    def test_recorded_miss_wins_over_the_price_rule(self) -> None:
        call = _closed("AAPL", base="100", target="105", actual="105")
        assert _recorded(call, False).is_correct() is False

    def test_without_recorded_verdict_prices_decide(self) -> None:
        assert _btc_sideways_up_1_5pct().is_correct() is False

    def test_hero_verdict_uses_the_recorded_value(self) -> None:
        predictions = [_recorded(_btc_sideways_up_1_5pct(), True), *_n_closed(4)]
        recap = build_weekly_recap(predictions)
        assert recap is not None
        assert recap.best.symbol == "BTC"
        assert recap.best.direction_correct is True


class TestVindicatedDissenters:
    def test_reuses_council_vindicated_dissenters(self) -> None:
        # „Soros miał rację, gdy wszyscy mówili KUP": rada zaleciła BUY,
        # cena spadła (DOWN), dysydent głosujący SELL zostaje wybroniony.
        opinions = (
            _opinion("Buffett", "BUY"),
            _opinion("Soros", "SELL"),
        )
        predictions = [
            _closed(
                "DROP",
                trend=TrendDirection.BEARISH,
                base="100",
                target="90",
                actual="90",  # accuracy 1.0 → najlepsza
                opinions=opinions,
                council_rec="BUY",
            ),
            *_n_closed(4),
        ]
        recap = build_weekly_recap(predictions)
        assert recap is not None
        names = [op.investor_name for op in recap.best.vindicated_dissenters]
        assert names == ["Soros"]

    def test_no_dissenters_when_council_was_right(self) -> None:
        # Rada zaleciła SELL, cena spadła — nikogo nie trzeba bronić.
        opinions = (_opinion("Soros", "SELL"),)
        predictions = [
            _closed(
                "DROP",
                trend=TrendDirection.BEARISH,
                base="100",
                target="90",
                actual="90",
                opinions=opinions,
                council_rec="SELL",
            ),
            *_n_closed(4),
        ]
        recap = build_weekly_recap(predictions)
        assert recap is not None
        assert recap.best.vindicated_dissenters == ()


class TestEceDelta:
    def test_current_ece_computed_from_confidences(self) -> None:
        recap = build_weekly_recap(_n_closed(5))
        assert recap is not None
        assert isinstance(recap.current_ece, float)
        assert recap.current_ece >= 0.0

    def test_ece_delta_against_previous(self) -> None:
        recap = build_weekly_recap(_n_closed(5), previous_ece=0.2)
        assert recap is not None
        assert recap.previous_ece == 0.2
        assert recap.ece_delta is not None
        # Delta = bieżące ECE − poprzednie ECE.
        assert abs(recap.ece_delta - (recap.current_ece - 0.2)) < 1e-9

    def test_ece_delta_none_when_no_previous(self) -> None:
        recap = build_weekly_recap(_n_closed(5))
        assert recap is not None
        assert recap.previous_ece is None
        assert recap.ece_delta is None


class TestRecapShape:
    def test_recap_exposes_sample_size(self) -> None:
        recap = build_weekly_recap(_n_closed(7))
        assert recap is not None
        assert recap.sample_size == 7

    def test_highlights_are_recap_highlight_instances(self) -> None:
        recap = build_weekly_recap(_n_closed(5))
        assert recap is not None
        assert isinstance(recap.best, RecapHighlight)
        assert isinstance(recap.worst, RecapHighlight)
