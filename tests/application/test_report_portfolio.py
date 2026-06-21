"""Testy renderu sekcji Portfolio Risk (korelacja/koncentracja) w raporcie.

Czysta prezentacja: bierze PortfolioRiskReport i produkuje fragment HTML
(lista klastrów + werdykt). Pusty raport → pusty string (sekcja pomijana).
Wszystkie nazwy symboli i tekst werdyktu MUSZĄ być escapowane (HTML-safe).
"""

from __future__ import annotations

from src.application.report_portfolio import render_correlation_html
from src.application.use_cases.portfolio_risk import PortfolioRiskReport
from src.domain.scenarios import ScenarioImpact


class TestRenderCorrelationHtml:
    def test_empty_report_renders_nothing(self) -> None:
        assert render_correlation_html(PortfolioRiskReport()) == ""

    def test_clusters_and_verdict_appear(self) -> None:
        report = PortfolioRiskReport(
            matrix={("MSFT", "NVDA"): 0.82, ("BTC", "NVDA"): 0.75},
            clusters=[("BTC", "MSFT", "NVDA")],
            verdict="BTC/MSFT/NVDA poruszają się razem 0.79 — 60% watchlisty to jeden zakład",
            watchlist_size=5,
            threshold=0.7,
        )
        html = render_correlation_html(report)

        assert "NVDA" in html
        assert "MSFT" in html
        assert "BTC" in html
        # Werdykt widoczny w treści.
        assert "jeden zakład" in html

    def test_symbols_are_html_escaped(self) -> None:
        # Symbol z metaznakami HTML — nie wolno wstrzyknąć surowego <script>.
        evil = "<script>x</script>"
        report = PortfolioRiskReport(
            matrix={(evil, "NVDA"): 0.9},
            clusters=[(evil, "NVDA")],
            verdict=f"{evil}/NVDA poruszają się razem 0.90 — 50% watchlisty to jeden zakład",
            watchlist_size=2,
            threshold=0.7,
        )
        html = render_correlation_html(report)

        assert "<script>x</script>" not in html
        assert "&lt;script&gt;" in html

    def test_verdict_only_no_clusters_still_safe(self) -> None:
        # Obronny przypadek: werdykt pusty, brak klastrów, ale matrix niepusty.
        report = PortfolioRiskReport(
            matrix={("A", "B"): 0.1},
            clusters=[],
            verdict="",
            watchlist_size=2,
            threshold=0.7,
        )
        # Brak klastrów i brak werdyktu → nic ciekawego do pokazania.
        html = render_correlation_html(report)
        assert html == "" or "Portfolio" in html

    def test_heading_present_when_clusters_exist(self) -> None:
        report = PortfolioRiskReport(
            matrix={("A", "B"): 0.95},
            clusters=[("A", "B")],
            verdict="A/B poruszają się razem 0.95 — 100% watchlisty to jeden zakład",
            watchlist_size=2,
            threshold=0.7,
        )
        html = render_correlation_html(report)
        # Sekcja ma nagłówek h2 (spójność z innymi sekcjami raportu).
        assert "<h2" in html


class TestRenderVarAndStress:
    def test_var_block_rendered_when_present(self) -> None:
        report = PortfolioRiskReport(
            var=0.08,
            cvar=0.12,
            var_confidence=0.95,
        )
        html = render_correlation_html(report)
        # Sekcja renderuje się też bez klastrów, gdy jest VaR.
        assert "<h2" in html
        assert "Value-at-Risk" in html
        # VaR/CVaR jako procenty + poziom ufności.
        assert "8.0%" in html
        assert "12.0%" in html
        assert "95%" in html

    def test_stress_table_rendered_when_present(self) -> None:
        report = PortfolioRiskReport(
            scenarios=(
                ScenarioImpact(
                    scenario_name="Krach -20%",
                    market_shock=-0.20,
                    portfolio_pnl=-0.18,
                    per_symbol={"NVDA": -0.30, "GLD": 0.06},
                    worst_symbol="NVDA",
                    worst_pnl=-0.30,
                ),
            ),
        )
        html = render_correlation_html(report)
        assert "<h2" in html
        assert "Krach -20%" in html
        # Najsłabszy symbol scenariusza widoczny.
        assert "NVDA" in html

    def test_stress_scenario_name_html_escaped(self) -> None:
        report = PortfolioRiskReport(
            scenarios=(
                ScenarioImpact(
                    scenario_name="<b>x</b>",
                    market_shock=-0.10,
                    portfolio_pnl=-0.09,
                    per_symbol={"<i>S</i>": -0.09},
                    worst_symbol="<i>S</i>",
                    worst_pnl=-0.09,
                ),
            ),
        )
        html = render_correlation_html(report)
        assert "<b>x</b>" not in html
        assert "&lt;b&gt;" in html

    def test_no_var_no_stress_no_clusters_renders_nothing(self) -> None:
        assert render_correlation_html(PortfolioRiskReport()) == ""
