"""Testy renderu sekcji 🛰️ Alpha Signals (nowe źródła danych) w raporcie.

Render-only: bierze sygnały alpha per symbol + snapshot krzywej rentowności
i produkuje fragment HTML. Brak danych → pusty string (sekcja się chowa).
Symbole i wartości escapowane (HTML-safe)."""

from __future__ import annotations

from decimal import Decimal

from src.application.alpha_signals import AlphaSignals
from src.application.report_alpha import render_alpha_html
from src.domain.analyst_consensus import AnalystConsensus
from src.domain.earnings import EarningsEvent
from src.domain.insider_flow import InsiderFlowSnapshot
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.options_flow import OptionsFlowSnapshot
from src.domain.social_velocity import SocialVelocitySnapshot


class TestEmpty:
    def test_no_data_renders_nothing(self) -> None:
        assert render_alpha_html({}, None) == ""

    def test_only_empty_signals_render_nothing(self) -> None:
        empty = {"AAPL": AlphaSignals(symbol="AAPL")}
        assert render_alpha_html(empty, None) == ""


class TestYieldCurve:
    def test_inverted_curve_block(self) -> None:
        curve = YieldCurveSnapshot(ten_year=4.25, two_year=4.80, fed_funds=5.33)
        html = render_alpha_html({}, curve)
        assert "<h2" in html
        assert "rentown" in html.lower()
        # Inwersja (spread ujemny) zaznaczona.
        assert "-0.55" in html or "INWERSJA" in html.upper() or "INVERTED" in html


class TestPerSymbol:
    def test_renders_all_signal_kinds(self) -> None:
        signals = {
            "AAPL": AlphaSignals(
                symbol="AAPL",
                insider=InsiderFlowSnapshot("AAPL", 5000.0, 4, 1, 90),
                earnings=EarningsEvent("AAPL", days_until=2, report_date="2026-06-23"),
                options=OptionsFlowSnapshot("AAPL", put_call_ratio=0.5, implied_vol=0.7),
                social=SocialVelocitySnapshot("AAPL", mentions_24h=60, baseline_mentions=20.0),
                analyst=AnalystConsensus("AAPL", strong_buy=12, buy=6, price_target=240.0),
            ),
        }
        html = render_alpha_html(signals, None)
        assert "<h2" in html
        assert "AAPL" in html
        # Sygnał insiderów (netto kupno) widoczny.
        assert "kupno" in html.lower() or "BUY" in html.upper()

    def test_symbol_html_escaped(self) -> None:
        signals = {
            "<x>": AlphaSignals(
                symbol="<x>",
                analyst=AnalystConsensus("<x>", buy=5),
            ),
        }
        html = render_alpha_html(signals, None)
        assert "<x>" not in html.replace("&lt;x&gt;", "")
        assert "&lt;x&gt;" in html

    def test_skips_symbols_without_signals(self) -> None:
        signals = {
            "AAPL": AlphaSignals(symbol="AAPL"),  # pusty → pominięty
            "MSFT": AlphaSignals(
                symbol="MSFT", analyst=AnalystConsensus("MSFT", strong_buy=8)
            ),
        }
        html = render_alpha_html(signals, None)
        assert "MSFT" in html
        # AAPL bez sygnałów nie powinno mieć własnego wiersza danych.
        assert html.count("AAPL") == 0


class TestComposition:
    def test_analyst_upside_uses_current_price_when_available(self) -> None:
        signals = {
            "AAPL": AlphaSignals(
                symbol="AAPL",
                analyst=AnalystConsensus("AAPL", strong_buy=10, price_target=220.0),
            ),
        }
        # current price 200 → upside +10%
        html = render_alpha_html(signals, None, current_prices={"AAPL": Decimal("200")})
        assert "+10" in html
