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
        assert "-0,55" in html or "INWERSJA" in html.upper() or "INVERTED" in html


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



class TestInsiderWithoutEvidence:
    """DAT-03: przy niesparsowanych Form 4 raport mówi „brak danych EDGAR”, a nie
    pokazuje kreski jak przy spokojnym oknie bez transakcji."""

    def test_unparsed_filings_render_as_missing_edgar_data(self) -> None:
        signals = {
            "AAPL": AlphaSignals(
                symbol="AAPL",
                insider=InsiderFlowSnapshot(
                    "AAPL", 0.0, 0, 0, 90, filings_seen=3, filings_parsed=0
                ),
            ),
        }

        html = render_alpha_html(signals, None)

        assert "brak danych EDGAR" in html

    def test_quiet_window_keeps_the_dash(self) -> None:
        signals = {
            "AAPL": AlphaSignals(
                symbol="AAPL",
                insider=InsiderFlowSnapshot(
                    "AAPL", 0.0, 0, 0, 90, filings_seen=0, filings_parsed=0
                ),
            ),
        }

        html = render_alpha_html(signals, None)

        assert "brak danych EDGAR" not in html


class TestSignificantInsiderBuy:
    """DAT-21: znaczący zakup officera dostaje własną linię z kwotą i linkiem do
    Form 4 — sama etykieta „kupno insiderów” nie mówi o skali."""

    _URL = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000001/xslF345X05/form4.xml"

    def _signals(self, value_usd: float, *, url: str | None = None) -> dict[str, AlphaSignals]:
        from src.domain.insider_flow import InsiderBuy

        insider = InsiderFlowSnapshot(
            "NVDA",
            10_000.0,
            1,
            0,
            90,
            net_value_usd=value_usd,
            largest_buy=InsiderBuy(
                role="CEO", value_usd=value_usd, filing_url=self._URL if url is None else url
            ),
        )
        return {"NVDA": AlphaSignals(symbol="NVDA", insider=insider)}

    def test_large_officer_buy_gets_a_line_with_amount_and_form4_link(self) -> None:
        html = render_alpha_html(self._signals(1_200_000.0), None)

        assert "NVDA — CEO kupił akcje za 1,2 mln USD" in html
        assert f"href='{self._URL}'" in html
        assert "Form 4 →" in html

    def test_amount_below_a_million_is_in_thousands(self) -> None:
        html = render_alpha_html(self._signals(250_000.0), None)

        assert "za 250 tys. USD" in html

    def test_buy_below_the_threshold_has_no_line(self) -> None:
        html = render_alpha_html(self._signals(50_000.0), None)

        assert "kupił akcje" not in html

    def test_link_outside_sec_gov_is_dropped(self) -> None:
        html = render_alpha_html(self._signals(1_200_000.0, url="https://evil.example/x"), None)

        assert "kupił akcje" in html
        assert "evil.example" not in html


class TestInsiderThresholdFromMarketCap:
    """DAT-21: próg linii znaczącego zakupu skaluje się kapitalizacją z cache."""

    @staticmethod
    def _signals(market_cap: float | None) -> dict[str, AlphaSignals]:
        from src.domain.insider_flow import InsiderBuy

        insider = InsiderFlowSnapshot(
            "NVDA",
            10_000.0,
            1,
            0,
            90,
            largest_buy=InsiderBuy(role="CEO", value_usd=1_200_000.0),
        )
        return {"NVDA": AlphaSignals(symbol="NVDA", insider=insider, market_cap=market_cap)}

    def test_buy_below_the_cap_scaled_threshold_has_no_line(self) -> None:
        assert "kupił akcje" not in render_alpha_html(self._signals(4e12), None)

    def test_buy_above_the_cap_scaled_threshold_keeps_its_line(self) -> None:
        html = render_alpha_html(self._signals(100e9), None)

        assert "NVDA — CEO kupił akcje za 1,2 mln USD" in html
