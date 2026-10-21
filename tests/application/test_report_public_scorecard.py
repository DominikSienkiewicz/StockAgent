"""Testy publicznego scorecardu kalibracji (roadmap #10).

Kluczowy test to TWARDA asercja anty-PII: renderer dostaje dane niosące symbol
watchlisty, adres e-mail i kwoty (ceny), a strona publiczna NIE MOŻE ich
ujawnić. Publikujemy wyłącznie agregaty: N, hit-rate, ECE, krzywą kalibracji.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from decimal import Decimal

from src.application.report_models import ResolvedPrediction
from src.application.report_public_scorecard import render_public_scorecard_html
from src.domain.equity_curve import equity_curve


def _resolved(
    symbol: str,
    trend: str,
    correct: bool,
    *,
    reasoning: str | None = None,
    price_at: Decimal | None = None,
    actual: Decimal | None = None,
) -> ResolvedPrediction:
    """Skrót do budowy zamkniętej predykcji w testach."""
    return ResolvedPrediction(
        symbol=symbol,
        predicted_trend=trend,
        is_correct=correct,
        reasoning=reasoning,
        price_at_prediction=price_at,
        actual_price=actual,
    )


def test_empty_inputs_self_suppress() -> None:
    """Brak jakichkolwiek danych → pusty string (nie publikujemy pustej strony)."""
    assert render_public_scorecard_html(None, None, None) == ""
    assert render_public_scorecard_html([], None, []) == ""


def test_renders_standalone_html_page() -> None:
    """Kompletna, samodzielna strona HTML (doctype + head + body)."""
    resolved = [_resolved("AAA", "BULLISH", True)]
    curve = equity_curve([("BULLISH", 0.03)])
    html = render_public_scorecard_html(resolved, curve, [(0.8, True)])
    assert html != ""
    lower = html.lower()
    assert "<!doctype html>" in lower
    assert "<html" in lower
    assert "</html>" in lower
    assert "<head" in lower
    assert "<body" in lower


def test_shows_aggregate_metrics_n_and_hit_rate() -> None:
    """Publikujemy liczbę predykcji N i hit-rate jako agregaty."""
    resolved = [
        _resolved("AAA", "BULLISH", True),
        _resolved("BBB", "BEARISH", True),
        _resolved("CCC", "BULLISH", False),
        _resolved("DDD", "BEARISH", True),
    ]
    curve = equity_curve(
        [("BULLISH", 0.02), ("BEARISH", -0.01), ("BULLISH", -0.02), ("BEARISH", -0.03)]
    )
    html = render_public_scorecard_html(resolved, curve, None)
    # N = 4 predykcje.
    assert "4" in html
    # hit-rate = 3/4 = 75%.
    assert "75" in html


def test_calibration_curve_rendered_as_div_sparkline_with_ece() -> None:
    """Krzywa kalibracji jako div-sparkline + wartość ECE na stronie."""
    samples = [
        (0.55, True),
        (0.55, False),
        (0.95, True),
        (0.95, True),
        (0.15, False),
    ]
    resolved = [_resolved("AAA", "BULLISH", True)]
    html = render_public_scorecard_html(resolved, None, samples)
    # Technika div-sparkline: słupki jako <div> o zadanej wysokości w px.
    assert "<div" in html
    assert "height:" in html.replace(" ", "")
    # ECE jest nazwane wprost na publicznej stronie.
    assert "ECE" in html


def test_equity_metrics_read_from_dataclass_fields() -> None:
    """Metryki equity pochodzą z PÓL EquityCurve (total_return, max_drawdown)."""
    curve = equity_curve([("BULLISH", 0.10), ("BEARISH", 0.05)])
    resolved = [
        _resolved("AAA", "BULLISH", True),
        _resolved("BBB", "BEARISH", False),
    ]
    html = render_public_scorecard_html(resolved, curve, None)
    # total_return dla [(+10%),(sygnał -1 * +5% = -5%)]: 1.1*0.95-1 = +4.5%.
    assert "+4.5%" in html


def test_methodology_section_is_honest_polish_copy() -> None:
    """Sekcja „Metodologia" po polsku — uczciwie tłumaczy okresy słabości."""
    resolved = [_resolved("AAA", "BULLISH", True)]
    html = render_public_scorecard_html(resolved, None, [(0.8, True)])
    assert "Metodologia" in html
    # Uczciwość: pokazujemy też własne pomyłki / słabe okresy.
    assert "pomyłk" in html.lower() or "słab" in html.lower()


def test_hard_anti_pii_no_symbols_emails_or_amounts_leak() -> None:
    """TWARDY test anty-PII — najważniejszy w tym zadaniu.

    Renderer dostaje dane z symbolem „NVDA", adresem e-mail i kwotami (ceny).
    Publiczna strona nie może zawierać ŻADNEGO z tych wycieków: ani symbolu,
    ani znaku „@" (e-mail), ani znaczników kwot („$", „zł", „PLN").
    """
    leaky_email = "insider@example.com"
    resolved = [
        _resolved(
            "NVDA",
            "BULLISH",
            True,
            reasoning=f"Kup wg cynku od {leaky_email} — cena 1234 USD",
            price_at=Decimal("1234.56"),
            actual=Decimal("1300.00"),
        ),
        _resolved(
            "TSLA",
            "BEARISH",
            False,
            reasoning="Short 999 PLN",
            price_at=Decimal("250.00"),
            actual=Decimal("260.00"),
        ),
    ]
    curve = equity_curve([("BULLISH", 0.05), ("BEARISH", 0.04)])
    samples = [(0.8, True), (0.6, False)]
    html = render_public_scorecard_html(resolved, curve, samples)

    assert html != ""
    # Zero symboli watchlisty.
    assert "NVDA" not in html
    assert "TSLA" not in html
    # Zero e-maili (żaden znak „@" nie może wyciec).
    assert leaky_email not in html
    assert "@" not in html
    # Zero kwot / walut.
    assert "$" not in html
    assert "zł" not in html
    assert "PLN" not in html
    assert "1234" not in html
    assert "1300" not in html
    assert "250" not in html


def _dated(
    trend: str, correct: bool, day: int, *, price_at: str = "100", actual: str = "101"
) -> ResolvedPrediction:
    """Zamknięta predykcja z datą (klucz cyklu = dzień UTC) i ruchem ceny."""
    return ResolvedPrediction(
        symbol="AAA",
        predicted_trend=trend,
        is_correct=correct,
        price_at_prediction=Decimal(price_at),
        actual_price=Decimal(actual),
        predicted_at=datetime(2026, 10, day, 14, 0, tzinfo=UTC),
    )


def _card(value: str, label: str) -> str:
    """Dokładny markup karty — asercje na podciągach przechodziłyby dzięki CSS."""
    return (
        f"<div class='card-value'>{value}</div>"
        f"<div class='card-label'>{label}</div>"
    )


class TestDirectionalHitRateHeadline:
    """WIA-05: publiczna „trafność kierunkowa” wliczała SIDEWAYS wbrew własnej
    metodologii („prognozy bez ruchu nie stawiają zakładu”)."""

    def test_directional_hit_rate_excludes_sideways(self) -> None:
        resolved = [
            _resolved("AAA", "BULLISH", True),
            _resolved("BBB", "BEARISH", False),
            _resolved("CCC", "SIDEWAYS", True),
            _resolved("DDD", "SIDEWAYS", True),
        ]

        html = render_public_scorecard_html(resolved, None, None)

        assert _card("50.0%", "Hit-rate kierunkowy") in html

    def test_sideways_predictions_get_their_own_card(self) -> None:
        resolved = [
            _resolved("AAA", "BULLISH", True),
            _resolved("CCC", "SIDEWAYS", True),
            _resolved("DDD", "SIDEWAYS", False),
        ]

        html = render_public_scorecard_html(resolved, None, None)

        assert _card("2", "Prognozy „bez ruchu” (poza hit-rate)") in html

    def test_missing_directional_data_shows_a_dash_not_the_equity_win_rate(self) -> None:
        # Fallback na `EquityCurve.win_rate` mieszał definicje — brak danych = „—”.
        curve = equity_curve([("BULLISH", 0.05)])

        html = render_public_scorecard_html([], curve, None)

        assert _card("—", "Hit-rate kierunkowy") in html
        assert _card("100.0%", "Hit-rate kierunkowy") not in html


class TestHonestHeadline:
    """WIA-31/32/22/23: trafność z przedziałem, n, n_eff, baseline'em i werdyktem."""

    def test_hit_rate_card_carries_wilson_interval_and_n(self) -> None:
        resolved = [_dated("BULLISH", i % 2 == 0, 1 + i) for i in range(10)]

        html = render_public_scorecard_html(resolved, None, None)

        # 5/10 na 10 niezależnych cyklach → Wilson 95% ≈ 24–76%.
        assert "95% CI 24–76%, n=10" in html

    def test_effective_n_is_disclosed_when_cycles_are_correlated(self) -> None:
        # 2 cykle × 5 identycznych wyników → n_eff = 2.
        resolved = [_dated("BULLISH", day == 1, day) for day in (1, 2) for _ in range(5)]

        html = render_public_scorecard_html(resolved, None, None)

        assert "n=10, n_eff≈2" in html

    def test_naive_baselines_are_shown_next_to_hit_rate(self) -> None:
        resolved = [
            _dated("BULLISH", True, 1, actual="102"),
            _dated("BULLISH", True, 2, actual="101"),
            _dated("BEARISH", True, 3, actual="99"),
        ]

        html = render_public_scorecard_html(resolved, None, None)

        assert "zawsze-BULLISH: 67%" in html
        assert "zawsze-BEARISH: 33%" in html

    def test_significance_verdict_is_stated(self) -> None:
        resolved = [_dated("BULLISH", i % 2 == 0, 1 + i) for i in range(6)]

        html = render_public_scorecard_html(resolved, None, None)

        assert "nieodróżnialne od rzutu monetą" in html

    def test_significance_verdict_keeps_the_decimal_point(self) -> None:
        # MAIL-58: polski zapis dotyczy tylko maila — strona publiczna zostaje przy kropce.
        resolved = [_dated("BULLISH", i % 2 == 0, 1 + i) for i in range(6)]

        html = render_public_scorecard_html(resolved, None, None)

        assert re.search(r"p=\d\.\d", html)
        assert not re.search(r"p=\d,\d", html)


class TestProbabilityScoresOnScorecard:
    """WIA-36: Brier i Brier Skill Score jako główne metryki pewności; ECE pomocniczo."""

    def test_calibration_section_leads_with_brier_and_skill(self) -> None:
        samples = [(0.6, True)] * 6 + [(0.6, False)] * 4

        html = render_public_scorecard_html([_resolved("AAA", "BULLISH", True)], None, samples)

        assert "Brier = 0.24 " in html
        assert "klimatologia 0.24," in html
        assert "BSS = +0.0%" in html
        assert "log loss" in html
        assert "ECE" in html


class TestWindowDisclosure:
    """WIA-33: strona deklaruje brak cherry-pickingu, więc musi mówić, jakie okno pokazuje."""

    def test_page_discloses_window_and_date(self) -> None:
        html = render_public_scorecard_html(
            [_resolved("AAA", "BULLISH", True)],
            None,
            None,
            generated_on=date(2026, 10, 18),
            window_days=30,
        )

        assert "Okno: ostatnie 30 dni (2026-09-18 – 2026-10-18)" in html
        assert "Wygenerowano: 2026-10-18" in html
        assert "@" not in html

    def test_calibration_sparkline_shows_bucket_counts(self) -> None:
        samples = [(0.55, True), (0.55, False), (0.95, True)]

        html = render_public_scorecard_html([_resolved("AAA", "BULLISH", True)], None, samples)

        assert "<div class='spark-count'>n=2</div>" in html
        assert "<div class='spark-count'>n=1</div>" in html


class TestPortfolioCurveDisclosure:
    """WIA-17: krzywa kapitału to portfel o równej wadze per cykl — strona mówi to wprost."""

    def test_equity_section_states_its_assumptions(self) -> None:
        curve = equity_curve([("BULLISH", 0.03)])

        html = render_public_scorecard_html([_resolved("AAA", "BULLISH", True)], curve, None)

        assert "równa waga" in html
        assert "gotówka" in html


def test_scorecard_states_universe_selection_caveat() -> None:
    """WIA-48: watchlista to dzisiejsi zwycięzcy wybrani ręcznie — strona mówi
    wprost, że long-bias na takim uniwersum nie jest umiejętnością."""
    resolved = [_resolved("NVDA", "BULLISH", True)]

    html = render_public_scorecard_html(resolved, None, [(0.8, True)])

    assert "Uniwersum wybrane ręcznie" in html
    # Decyzja właściciela 2026-10-19: lista z gita powstała w czerwcu 2026.
    assert "2026-06" in html
    assert "2026-05" not in html
    assert "long-bias" in html
    assert "nie jest umiejętnością" in html
    # Zastrzeżenie nie może łamać inwariantu „zero symboli watchlisty”.
    assert "NVDA" not in html


class TestPaperCostDisclosure:
    """WIA-50: strona publiczna podaje koszt paper-PnL z kropką dziesiętną."""

    def test_equity_section_names_the_cost(self) -> None:
        from src.application.report_evidence import PaperCosts

        curve = equity_curve([("BULLISH", 0.03)], cost_bps=12.5)
        costs = PaperCosts(
            cost_bps=12.5, crypto_cost_bps=25, crypto_symbols=frozenset({"ZZCOIN"})
        )

        html = render_public_scorecard_html(
            [_resolved("AAA", "BULLISH", True)], curve, None, costs=costs
        )

        assert "paper-PnL (koszt 12.5 bps)" in html
        assert "krypto 25 bps" in html
        assert "12,5" not in html
        assert "bez kosztów transakcyjnych" not in html
        # Symbole z założeń kosztowych nie trafiają na stronę.
        assert "ZZCOIN" not in html

    def test_without_costs_the_page_says_the_curve_is_gross(self) -> None:
        curve = equity_curve([("BULLISH", 0.03)])

        html = render_public_scorecard_html([_resolved("AAA", "BULLISH", True)], curve, None)

        assert "bez kosztów transakcyjnych" in html


class TestBenchmarkDisclosure:
    """WIA-21: strona publiczna stawia krzywą agenta obok koszyka i indeksu —
    z kropką dziesiętną i bez symboli watchlisty. Decyzja 2026-10-19: indeks
    nazywa się „S&P 500”, nie tickerem IVV (który też jest w watchliście)."""

    @staticmethod
    def _comparison(*, with_index: bool = True):
        from src.domain.benchmark import compare_with_benchmarks
        from src.domain.equity_curve import PaperTrade

        cycles = [
            [PaperTrade("NVDA", "BEARISH", c, benchmark_change=0.008 if with_index else None)]
            for c in (0.01, -0.005, 0.02)
        ]
        return compare_with_benchmarks(cycles, index_symbol="IVV", index_cost_bps=0)

    def test_equity_section_compares_with_basket_and_ivv(self) -> None:
        comparison = self._comparison()

        html = render_public_scorecard_html(
            [_resolved("NVDA", "BEARISH", False)], comparison.agent, None, benchmark=comparison
        )

        assert "agent -2.5% · koszyk +2.5% · S&amp;P 500 +2.4%" in html
        assert "Zwrot nadwyżkowy vs koszyk: -5.0 p.p." in html
        assert "hit-rate względem rynku (S&amp;P 500): 33.3% (n=3)" in html
        assert "spark-bar bench" in html
        assert ".spark-bar.bench{" in html
        assert "NVDA" not in html

    def test_missing_ivv_data_is_stated_explicitly(self) -> None:
        comparison = self._comparison(with_index=False)

        html = render_public_scorecard_html(
            [_resolved("NVDA", "BEARISH", False)], comparison.agent, None, benchmark=comparison
        )

        assert "S&amp;P 500: brak danych benchmarku" in html

    def test_without_a_comparison_the_page_says_there_is_no_benchmark(self) -> None:
        curve = equity_curve([("BULLISH", 0.03)])

        html = render_public_scorecard_html([_resolved("AAA", "BULLISH", True)], curve, None)

        assert "Na tle rynku: brak danych benchmarku" in html
        assert "spark-bar bench" not in html


def test_benchmark_never_leaks_the_index_ticker_or_basket_symbols() -> None:
    """Inwariant „zero symboli watchlisty” bez wyjątków — także z benchmarkiem:
    ticker indeksu (IVV jest w watchliście) i symbole koszyka nie trafiają na stronę."""
    from src.domain.benchmark import compare_with_benchmarks
    from src.domain.equity_curve import PaperTrade

    cycles = [
        [
            PaperTrade("NVDA", "BULLISH", 0.02, 10, benchmark_change=0.01),
            PaperTrade("IVV", "BEARISH", 0.01, 10, benchmark_change=0.01),
        ]
    ]
    comparison = compare_with_benchmarks(cycles, index_symbol="IVV", index_cost_bps=10)
    resolved = [_resolved("NVDA", "BULLISH", True), _resolved("IVV", "BEARISH", False)]

    html = render_public_scorecard_html(resolved, comparison.agent, None, benchmark=comparison)

    assert "S&amp;P 500" in html
    assert "IVV" not in html
    assert "NVDA" not in html


class TestEvaluationProtocolSection:
    """WIA-42: sekcja metodologii pokazuje protokół ewaluacji z jego hashem —
    ten sam sha256, który niesie każdy commitment v2."""

    @staticmethod
    def _protocol() -> object:
        from src.config import load_evaluation_protocol

        return load_evaluation_protocol()

    def test_methodology_renders_the_protocol_and_its_hash(self) -> None:
        protocol = self._protocol()

        html = render_public_scorecard_html(
            [_resolved("AAA", "BULLISH", True)], None, [(0.8, True)], protocol=protocol
        )

        assert "Protokół ewaluacji" in html
        assert protocol.sha256 in html
        assert "12 h" in html
        # Strona publiczna pisze liczby z kropką (DOT_NUMBERS, MAIL-58).
        assert "±0.5%" in html and "±2.5%" in html
        assert "α = 0.05" in html

    def test_without_a_protocol_the_methodology_is_unchanged(self) -> None:
        resolved = [_resolved("AAA", "BULLISH", True)]

        html = render_public_scorecard_html(resolved, None, [(0.8, True)])

        assert "Protokół ewaluacji" not in html

    def test_protocol_section_names_no_watchlist_symbol(self) -> None:
        import tomllib
        from pathlib import Path

        config = tomllib.loads(
            (Path(__file__).parents[2] / "config.toml").read_text(encoding="utf-8")
        )
        protocol = self._protocol()

        html = render_public_scorecard_html(
            [_resolved("AAA", "BULLISH", True)], None, [(0.8, True)], protocol=protocol
        )
        section = html[html.index("Protokół ewaluacji"):]

        # Jednoliterowy ticker („S”) koliduje z nazwą indeksu „S&P 500”, którą strona
        # podaje celowo zamiast tickera IVV.
        watchlist = {
            symbol
            for symbol in (*config["symbols"], *config["crypto_symbols"], *config["symbols_etf"])
            if len(symbol) > 1
        }
        leaked = {s for s in watchlist if re.search(rf"\b{re.escape(s)}\b", section)}
        assert leaked == set()
        assert "@" not in section

class TestDirectionalCalibrationDisclosure:
    """WIA-39: krzywa kalibracji strony jest kierunkowa — strona mówi to wprost."""

    def test_calibration_section_says_sideways_are_excluded(self) -> None:
        html = render_public_scorecard_html(
            [_resolved("AAA", "BULLISH", True)], None, [(0.6, True), (0.7, False)]
        )

        assert "bez prognoz „bez ruchu”" in html


class TestCalibrationErrorInterval:
    """WIA-58: „ECE 7.0% [3.0%; 12.0%], n_eff≈…” — kubełki równoliczne, korekta
    obciążenia i przedział z bootstrapu po cyklach, gdy znamy cykle próbek."""

    @staticmethod
    def _cycles(n_cycles: int) -> list[list[tuple[float, bool]]]:
        return [
            [(0.6 + 0.02 * (i % 10), (day + i) % 3 != 0) for i in range(8)]
            for day in range(n_cycles)
        ]

    def test_ece_line_carries_the_interval_n_eff_and_debiased_value(self) -> None:
        cycles = self._cycles(25)
        samples = [sample for cycle in cycles for sample in cycle]

        html = render_public_scorecard_html(
            [_resolved("AAA", "BULLISH", True)], None, samples, calibration_by_cycle=cycles
        )

        assert re.search(r"ECE = \d+\.\d% \[\d+\.\d%; \d+\.\d%\], n=200, n_eff≈\d+", html)
        assert "kubełki równoliczne" in html
        assert "po korekcie obciążenia" in html

    def test_too_few_cycles_say_so_instead_of_an_interval(self) -> None:
        cycles = self._cycles(3)
        samples = [sample for cycle in cycles for sample in cycle]

        html = render_public_scorecard_html(
            [_resolved("AAA", "BULLISH", True)], None, samples, calibration_by_cycle=cycles
        )

        assert "za mało cykli na bootstrap" in html
        assert not re.search(r"ECE = \d+\.\d% \[", html)

    def test_methodology_describes_the_equal_mass_buckets(self) -> None:
        html = render_public_scorecard_html(
            [_resolved("AAA", "BULLISH", True)], None, [(0.6, True)]
        )

        assert "równoliczn" in html

def test_coverage_line_counts_unsettled_symbols_without_naming_them() -> None:
    # DAT-16: pokrycie i luka są na stronie, ale symbole bez rozliczeń tylko jako liczba.
    from src.domain.coverage import CoverageGap, CoverageReport

    coverage = CoverageReport(
        expected=22,
        covered=11,
        longest_gap=CoverageGap(first=date(2026, 8, 17), last=date(2026, 8, 31), sessions=11),
        unsettled_symbols=("SSNLF",),
    )

    html = render_public_scorecard_html(
        [_resolved("AAA", "BULLISH", True)], None, [(0.8, True)], coverage=coverage
    )

    assert "Pokrycie danych: 50% sesji ze snapshotem ceny (11 z 22)" in html
    assert "najdłuższa luka 2026-08-17 – 2026-08-31 (11 sesji)" in html
    assert "1 symbol bez rozliczeń w oknie" in html
    assert "SSNLF" not in html
