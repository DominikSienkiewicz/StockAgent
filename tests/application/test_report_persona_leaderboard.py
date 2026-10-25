"""Testy sekcji "Rada — ranking wiarygodności" w raporcie e-mail (#3)."""

from __future__ import annotations

import pytest

from src.application.report_formatting import PALETTE
from src.application.report_persona_leaderboard import (
    render_persona_leaderboard_html,
)
from src.domain.persona_track_record import PersonaTrackRecord


def _record(name: str, hit_rate: float, votes: int) -> PersonaTrackRecord:
    return PersonaTrackRecord(investor_name=name, hit_rate=hit_rate, votes=votes)


class TestRenderPersonaLeaderboardHtml:
    def test_empty_ranking_suppresses_section(self) -> None:
        # Sekcja samosupresująca: brak danych = brak sekcji, nie pusty nagłówek.
        assert render_persona_leaderboard_html([]) == ""

    def test_renders_section_heading(self) -> None:
        html = render_persona_leaderboard_html([_record("Buffett", 0.68, 22)])

        assert "ranking wiarygodności" in html.lower()

    def test_shows_hit_rate_as_percent_next_to_vote_count(self) -> None:
        # Rdzeń pozycji: procent BEZ liczby głosów to szum udający dowód.
        html = render_persona_leaderboard_html([_record("Buffett", 0.68, 22)])

        assert "Buffett" in html
        assert "68%" in html
        assert "22" in html

    def test_preserves_domain_ranking_order(self) -> None:
        html = render_persona_leaderboard_html(
            [
                _record("Buffett", 0.68, 22),
                _record("Lynch", 0.55, 10),
                _record("Wood", 0.41, 18),
            ]
        )

        assert html.index("Buffett") < html.index("Lynch") < html.index("Wood")

    @pytest.mark.parametrize(
        ("votes", "expected"),
        [
            (1, "1 głos"),
            (2, "2 głosy"),
            (4, "4 głosy"),
            (5, "5 głosów"),
            (12, "12 głosów"),
            (14, "14 głosów"),
            (18, "18 głosów"),
            (22, "22 głosy"),
            (25, "25 głosów"),
            (102, "102 głosy"),
        ],
    )
    def test_vote_count_uses_polish_plural(self, votes: int, expected: str) -> None:
        html = render_persona_leaderboard_html([_record("Buffett", 0.6, votes)])

        assert expected in html

    def test_leader_is_marked(self) -> None:
        html = render_persona_leaderboard_html(
            [_record("Buffett", 0.68, 22), _record("Wood", 0.41, 18)]
        )

        assert "🥇" in html

    @pytest.mark.parametrize(
        ("hit_rate", "color"),
        [
            (0.68, PALETTE.positive),  # >= 55% — persona realnie dodaje wartość
            (0.55, PALETTE.positive),  # próg włącznie
            (0.50, PALETTE.caution),  # okolice losowego strzału (~33% na 3 opcjach)
            (0.45, PALETTE.caution),  # próg włącznie
            (0.41, PALETTE.negative),  # < 45% — persona szkodzi
        ],
    )
    def test_hit_rate_is_color_coded(self, hit_rate: float, color: str) -> None:
        html = render_persona_leaderboard_html([_record("Buffett", hit_rate, 20)])

        assert color in html

    def test_escapes_investor_name(self) -> None:
        html = render_persona_leaderboard_html(
            [_record("<script>alert(1)</script>", 0.6, 10)]
        )

        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestPersonaWilsonInterval:
    """WIA-32: „68% (22 głosy)” bez przedziału nie mówi, czy 68% różni się od 50%."""

    def test_hit_rate_carries_a_95_percent_wilson_interval(self) -> None:
        # 15/22 ≈ 68% → Wilson 95% ≈ 47–84%.
        html = render_persona_leaderboard_html([_record("Buffett", 15 / 22, 22)])

        assert "68%" in html
        assert "95% CI 47–84%" in html


class TestPersonaVerdict:
    """WIA-22: werdykt istotności persony (test dwumianowy na n_eff vs najlepszy
    naiwny baseline, BH-FDR po personach) i jedno zdanie o przybliżeniu."""

    @staticmethod
    def _assessed(significant: bool) -> PersonaTrackRecord:
        return PersonaTrackRecord(
            investor_name="Buffett",
            hit_rate=15 / 22,
            votes=22,
            n_eff=11.0,
            p_value=0.034,
            significant=significant,
        )

    def test_assessed_persona_shows_its_verdict_with_a_decimal_comma(self) -> None:
        html = render_persona_leaderboard_html([self._assessed(False)])

        assert "nieodróżnialne od rzutu monetą (p=0,03, n=22, n_eff≈11)" in html

    def test_significant_persona_reads_significant(self) -> None:
        html = render_persona_leaderboard_html([self._assessed(True)])

        assert "istotnie lepsze od naiwnego baseline" in html

    def test_interval_and_sample_use_n_eff(self) -> None:
        html = render_persona_leaderboard_html([self._assessed(False)])

        assert "95% CI 47–84%" not in html
        assert "22 głosy (n_eff≈11)" in html

    def test_approximation_note_only_with_verdicts(self) -> None:
        assessed = render_persona_leaderboard_html([self._assessed(False)])
        plain = render_persona_leaderboard_html([_record("Buffett", 0.68, 22)])

        assert "przybliżenie" in assessed
        assert "BH-FDR" in assessed
        assert "przybliżenie" not in plain


class TestPersonaBrierScore:
    """WIA-44: pod hit-rate'em stoi Brier pewności i jego umiejętność (BSS) —
    tylko dla rekordów z oceną Briera (flaga), reszta maila bez zmian."""

    @staticmethod
    def _scored(name: str, brier: float) -> PersonaTrackRecord:
        return PersonaTrackRecord(
            investor_name=name, hit_rate=0.6, votes=20, brier=brier, log_loss=0.7
        )

    def test_row_shows_brier_and_skill_with_decimal_commas(self) -> None:
        html = render_persona_leaderboard_html([self._scored("Alfa", 0.40)])

        assert "Brier 0,40" in html
        assert "BSS -0,67" in html

    def test_note_explains_the_brier_order_only_when_scores_are_present(self) -> None:
        with_scores = render_persona_leaderboard_html([self._scored("Alfa", 0.40)])
        without = render_persona_leaderboard_html([_record("Alfa", 0.6, 20)])

        assert "Brier" in with_scores
        assert "przepewna" in with_scores
        assert "Brier" not in without

    def test_records_without_a_score_render_exactly_as_before(self) -> None:
        html = render_persona_leaderboard_html([_record("Buffett", 0.68, 22)])

        assert "BSS" not in html
        assert "Brier" not in html

    def test_undefined_skill_shows_only_the_brier(self) -> None:
        record = PersonaTrackRecord(
            investor_name="Perfect", hit_rate=1.0, votes=10, brier=0.01, log_loss=0.05
        )

        html = render_persona_leaderboard_html([record])

        assert "Brier 0,01" in html
        assert "· BSS" not in html
