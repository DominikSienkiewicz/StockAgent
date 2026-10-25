"""Testy renderu sekcji „Tydzień StockAgenta" (#17) — HTML i plain-text.

Zamrażają: samosupresję (recap None → ""), obecność czterech sekcji (strzał,
wtopa z lekcją, wybronieni dysydenci, delta ECE), escaping HTML i polską treść.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

from src.application.report_weekly_recap import (
    render_weekly_recap_html,
    render_weekly_recap_text,
)
from src.domain.calibration_curve import ProbabilityScores
from src.domain.council import InvestorOpinion
from src.domain.weekly_recap import RecapHighlight, WeeklyRecap


def _recap(
    *,
    best_insight: str = "",
    worst_insight: str = "Zignorowano ostrzeżenie",
    vindicated: tuple[InvestorOpinion, ...] = (),
    previous_ece: float | None = 0.20,
    ece_delta: float | None = -0.05,
) -> WeeklyRecap:
    best = RecapHighlight(
        symbol="NVDA",
        accuracy=Decimal("0.95"),
        correction_insight=best_insight,
    )
    worst = RecapHighlight(
        symbol="TSLA",
        accuracy=Decimal("0.10"),
        correction_insight=worst_insight,
        vindicated_dissenters=vindicated,
    )
    return WeeklyRecap(
        sample_size=7,
        best=best,
        worst=worst,
        current_ece=0.15,
        previous_ece=previous_ece,
        ece_delta=ece_delta,
    )


class TestWeeklyRecapHtml:
    def test_none_recap_is_self_suppressing(self) -> None:
        assert render_weekly_recap_html(None) == ""

    def test_contains_all_sections(self) -> None:
        soros = InvestorOpinion(
            investor_name="Soros",
            recommendation="SELL",
            confidence=0.9,
            reasoning="rynek przegrzany",
            key_factors=(),
        )
        html = render_weekly_recap_html(
            _recap(vindicated=(soros,)),
            narrative="Tydzień pełen zwrotów.",
        )
        assert "NVDA" in html
        assert "TSLA" in html
        assert "Zignorowano ostrzeżenie" in html
        assert "Soros" in html
        assert "Tydzień pełen zwrotów." in html
        # Delta ECE obecna (kalibracja).
        assert "ECE" in html

    def test_escapes_html_in_insight_and_narrative(self) -> None:
        html = render_weekly_recap_html(
            _recap(worst_insight="<script>alert('x')</script>"),
            narrative="<b>hej</b>",
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "<b>hej</b>" not in html

    def test_no_previous_ece_still_renders(self) -> None:
        html = render_weekly_recap_html(
            _recap(previous_ece=None, ece_delta=None)
        )
        assert html != ""
        assert "TSLA" in html


class TestDissenterLabelsArePolish:
    """MAIL-61: wybroniony dysydent niesie głos w słowniku `polish_labels`."""

    @staticmethod
    def _soros() -> InvestorOpinion:
        return InvestorOpinion(
            investor_name="Soros",
            recommendation="SELL",
            confidence=0.9,
            reasoning="rynek przegrzany",
            key_factors=(),
        )

    def test_html_shows_the_vote_as_sprzedaj(self) -> None:
        html = render_weekly_recap_html(_recap(vindicated=(self._soros(),)))

        assert "Soros</strong> (SPRZEDAJ)" in html
        assert "(SELL)" not in html

    def test_text_shows_the_vote_as_sprzedaj(self) -> None:
        text = render_weekly_recap_text(_recap(vindicated=(self._soros(),)))

        assert "Wybroniony: Soros (SPRZEDAJ) — rynek przegrzany" in text
        assert "(SELL)" not in text


class TestWeeklyRecapText:
    def test_none_recap_is_self_suppressing(self) -> None:
        assert render_weekly_recap_text(None) == ""

    def test_contains_heroes_and_narrative(self) -> None:
        text = render_weekly_recap_text(
            _recap(), narrative="Podsumowanie tygodnia."
        )
        assert "NVDA" in text
        assert "TSLA" in text
        assert "Podsumowanie tygodnia." in text


def _hero_recap(*, best_correct: bool | None, worst_correct: bool | None = False) -> WeeklyRecap:
    best = RecapHighlight(
        symbol="NVDA",
        accuracy=Decimal("0.95"),
        correction_insight="",
        direction_correct=best_correct,
    )
    worst = RecapHighlight(
        symbol="TSLA",
        accuracy=Decimal("0.10"),
        correction_insight="",
        direction_correct=worst_correct,
    )
    return WeeklyRecap(
        sample_size=7,
        best=best,
        worst=worst,
        current_ece=0.15,
        previous_ece=None,
        ece_delta=None,
    )


class TestHeroDirectionVerdict:
    """WIA-07 (G40): `accuracy` to bliskość ceny do celu, nie trafność. Bohater
    pokazuje werdykt kierunku i błąd ceny docelowej, a tydzień bez trafień
    kierunkowych nie ma „strzału”, tylko „najbliżej celu”."""

    def test_html_shows_direction_verdict_and_target_error(self) -> None:
        html = render_weekly_recap_html(_hero_recap(best_correct=True))

        assert "Strzał tygodnia" in html
        assert "✅ kierunek trafiony" in html
        assert "❌ kierunek chybiony" in html
        assert "błąd ceny docelowej 5,0%" in html
        assert "trafność 95%" not in html

    def test_html_week_without_direction_hits_has_no_best_shot(self) -> None:
        html = render_weekly_recap_html(_hero_recap(best_correct=False))

        assert "Strzał tygodnia" not in html
        assert "Najbliżej celu (kierunek chybiony)" in html

    def test_text_shows_direction_verdict_and_target_error(self) -> None:
        text = render_weekly_recap_text(_hero_recap(best_correct=True))

        assert "Strzał tygodnia: NVDA" in text
        assert "✅ kierunek trafiony" in text
        assert "błąd ceny docelowej 5,0%" in text
        assert "trafność 95%" not in text

    def test_text_week_without_direction_hits_has_no_best_shot(self) -> None:
        text = render_weekly_recap_text(_hero_recap(best_correct=False))

        assert "Strzał tygodnia" not in text
        assert "Najbliżej celu (kierunek chybiony): NVDA" in text


class TestWorstHeroMirrorTitle:
    """Decyzja 2026-10-19 (a): tydzień bez pudeł kierunkowych nie ma „wtopy”,
    tylko „najdalej od celu” — lustro „Najbliżej celu (kierunek chybiony)”."""

    def test_html_week_without_direction_misses_has_no_miss_of_week(self) -> None:
        html = render_weekly_recap_html(_hero_recap(best_correct=True, worst_correct=True))

        assert "Wtopa tygodnia" not in html
        assert "Najdalej od celu (kierunek trafiony)" in html

    def test_text_week_without_direction_misses_has_no_miss_of_week(self) -> None:
        text = render_weekly_recap_text(_hero_recap(best_correct=True, worst_correct=True))

        assert "Wtopa tygodnia" not in text
        assert "Najdalej od celu (kierunek trafiony): TSLA" in text

    def test_a_directional_miss_keeps_the_miss_of_week_title(self) -> None:
        text = render_weekly_recap_text(_hero_recap(best_correct=True))

        assert "Wtopa tygodnia: TSLA" in text


class TestWeeklyRecapDisclaimer:
    """MAIL-08: recap wychodzi jako osobny mail, więc niesie własne zastrzeżenie
    na końcu — HTML i tekst. Recap None nadal jest pusty (samosupresja)."""

    _DISCLAIMER = "nie stanowi rekomendacji inwestycyjnej"

    def test_html_recap_ends_with_disclaimer(self) -> None:
        html = render_weekly_recap_html(_recap(), narrative="Tydzień pełen zwrotów.")

        assert self._DISCLAIMER in html
        assert html.index("Kalibracja") < html.index(self._DISCLAIMER)

    def test_text_recap_ends_with_disclaimer(self) -> None:
        text = render_weekly_recap_text(_recap())

        assert self._DISCLAIMER in text.splitlines()[-1]


def test_ece_line_uses_a_decimal_comma() -> None:
    # MAIL-58: mail niedzielny pisze liczby po polsku — „ECE 0,150”, nie „ECE 0.150”.
    html = render_weekly_recap_html(_recap())
    text = render_weekly_recap_text(_recap())

    assert "ECE 0,150 (poprawa o 0,050 tydzień do tygodnia)" in html
    assert "ECE 0,150 (poprawa o 0,050 tydzień do tygodnia)" in text


class TestWeeklyRecapProbabilityScores:
    """WIA-36: Brier i BSS obok ECE — samo ECE nie odróżnia umiejętności od zera."""

    @staticmethod
    def _scored_recap() -> WeeklyRecap:
        scores = ProbabilityScores(
            n=5,
            brier=0.168,
            log_loss=0.5,
            climatology_brier=0.24,
            skill=0.3,
            reliability=0.01,
            resolution=0.08,
            uncertainty=0.24,
        )
        return dataclasses.replace(_recap(), scores=scores)

    def test_html_shows_brier_climatology_and_skill(self) -> None:
        html = render_weekly_recap_html(self._scored_recap())

        assert "Brier = 0,168 (klimatologia 0,24, BSS = +30,0%)" in html

    def test_text_shows_brier_climatology_and_skill(self) -> None:
        text = render_weekly_recap_text(self._scored_recap())

        assert "Brier = 0,168 (klimatologia 0,24, BSS = +30,0%)" in text

    def test_recap_without_scores_has_no_brier_line(self) -> None:
        assert "Brier" not in render_weekly_recap_html(_recap())
        assert "Brier" not in render_weekly_recap_text(_recap())


class TestCouncilRedundancyLine:
    """WIA-45: „7 person ≈ 2,1 niezależnego głosu” w recapie — tylko gdy policzone."""

    @staticmethod
    def _with_redundancy(n_eff: float = 2.1, personas: int = 7) -> WeeklyRecap:
        from src.domain.council_redundancy import CouncilRedundancy, PairAgreement

        pair = PairAgreement("Graham", "Buffett", n=30, agreement=0.95, kappa=0.91)
        return dataclasses.replace(
            _recap(),
            redundancy=CouncilRedundancy(
                personas=personas,
                predictions=40,
                pairs=(pair,),
                llm_pairs=(),
                mean_kappa=0.52,
                mean_agreement=0.81,
                n_eff=n_eff,
            ),
        )

    def test_html_states_the_effective_number_of_voters(self) -> None:
        html = render_weekly_recap_html(self._with_redundancy())

        assert "7 person ≈ 2,1 niezależnego głosu" in html
        assert "Graham" in html and "Buffett" in html
        assert "kappa 0,91" in html

    def test_text_states_the_effective_number_of_voters(self) -> None:
        text = render_weekly_recap_text(self._with_redundancy())

        assert "7 person ≈ 2,1 niezależnego głosu" in text

    def test_a_council_that_speaks_with_one_voice_is_called_out(self) -> None:
        html = render_weekly_recap_html(self._with_redundancy(n_eff=1.0, personas=3))

        assert "w praktyce jeden głos" in html

    def test_recap_without_redundancy_has_no_such_line(self) -> None:
        assert "niezależnego głosu" not in render_weekly_recap_html(_recap())
        assert "niezależnego głosu" not in render_weekly_recap_text(_recap())
