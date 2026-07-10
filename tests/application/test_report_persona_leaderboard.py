"""Testy sekcji "Rada — ranking wiarygodności" w raporcie e-mail (#3)."""

from __future__ import annotations

import pytest

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
            (0.68, "#16a34a"),  # >= 55% — persona realnie dodaje wartość
            (0.55, "#16a34a"),  # próg włącznie
            (0.50, "#ca8a04"),  # okolice losowego strzału (~33% na 3 opcjach)
            (0.45, "#ca8a04"),  # próg włącznie
            (0.41, "#dc2626"),  # < 45% — persona szkodzi
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
