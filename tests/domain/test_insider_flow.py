"""Testy domeny InsiderFlowSnapshot.evaluate_signal.

Czysta logika klasyfikacji przepływu insiderów — bez I/O, bez adaptera.
Sprawdzamy progi NET_BUYING / NET_SELLING / NEUTRAL oraz przypadek braku
transakcji (zero buy_count + sell_count → NEUTRAL niezależnie od net_shares).
"""

from __future__ import annotations

from datetime import date

import pytest

from src.domain.insider_flow import InsiderBuy, InsiderFlowSnapshot, InsiderSignal


def _snap(
    net_shares: float,
    buy_count: int = 0,
    sell_count: int = 0,
    window_days: int = 90,
) -> InsiderFlowSnapshot:
    return InsiderFlowSnapshot(
        symbol="AAPL",
        net_shares=net_shares,
        buy_count=buy_count,
        sell_count=sell_count,
        window_days=window_days,
    )


class TestEvaluateSignal:
    def test_zero_transactions_is_neutral(self) -> None:
        # Brak jakichkolwiek transakcji → NEUTRAL nawet gdy net_shares != 0.
        snap = _snap(net_shares=0.0, buy_count=0, sell_count=0)
        assert snap.evaluate_signal() is InsiderSignal.NEUTRAL

    def test_zero_transactions_neutral_even_with_nonzero_net(self) -> None:
        # net_shares "zabłąkany" przy zerze transakcji nie zmienia werdyktu.
        snap = _snap(net_shares=1000.0, buy_count=0, sell_count=0)
        assert snap.evaluate_signal() is InsiderSignal.NEUTRAL

    def test_positive_net_is_buying(self) -> None:
        snap = _snap(net_shares=500.0, buy_count=2, sell_count=1)
        assert snap.evaluate_signal() is InsiderSignal.NET_BUYING

    def test_negative_net_is_selling(self) -> None:
        snap = _snap(net_shares=-500.0, buy_count=1, sell_count=3)
        assert snap.evaluate_signal() is InsiderSignal.NET_SELLING

    def test_net_zero_with_transactions_is_neutral(self) -> None:
        # Kupno i sprzedaż znoszą się do zera → martwa strefa → NEUTRAL.
        snap = _snap(net_shares=0.0, buy_count=1, sell_count=1)
        assert snap.evaluate_signal() is InsiderSignal.NEUTRAL

    def test_threshold_dead_zone_is_neutral(self) -> None:
        # net_shares w zakresie [-min, +min] → NEUTRAL.
        snap = _snap(net_shares=50.0, buy_count=1, sell_count=0)
        assert snap.evaluate_signal(min_net_shares=100.0) is InsiderSignal.NEUTRAL

    def test_above_threshold_is_buying(self) -> None:
        snap = _snap(net_shares=150.0, buy_count=1, sell_count=0)
        assert snap.evaluate_signal(min_net_shares=100.0) is InsiderSignal.NET_BUYING

    def test_below_negative_threshold_is_selling(self) -> None:
        snap = _snap(net_shares=-150.0, buy_count=0, sell_count=1)
        assert snap.evaluate_signal(min_net_shares=100.0) is InsiderSignal.NET_SELLING

    def test_exactly_at_threshold_is_neutral(self) -> None:
        # Próg jest ścisły (>), więc równość = NEUTRAL.
        snap = _snap(net_shares=100.0, buy_count=1, sell_count=0)
        assert snap.evaluate_signal(min_net_shares=100.0) is InsiderSignal.NEUTRAL


class TestHasEvidence:
    """DAT-03: snapshot mówi, czy EDGAR dał dowód o insiderach. Niesparsowane Form 4
    są nieodróżnialne od braku transakcji, jeśli liczniki filingów tego nie powiedzą."""

    def test_defaults_keep_the_snapshot_as_evidence(self) -> None:
        snap = _snap(net_shares=0.0)
        assert (snap.filings_seen, snap.filings_parsed) == (0, 0)
        assert snap.has_evidence()

    def test_seen_but_unparsed_filings_are_no_evidence(self) -> None:
        snap = InsiderFlowSnapshot("AAPL", 0.0, 0, 0, 90, filings_seen=3, filings_parsed=0)
        assert not snap.has_evidence()

    def test_one_parsed_filing_is_evidence(self) -> None:
        snap = InsiderFlowSnapshot("AAPL", -500.0, 0, 1, 90, filings_seen=3, filings_parsed=1)
        assert snap.has_evidence()

    def test_no_filings_in_the_window_is_evidence_of_no_trades(self) -> None:
        snap = InsiderFlowSnapshot("AAPL", 0.0, 0, 0, 90, filings_seen=0, filings_parsed=0)
        assert snap.has_evidence()


class TestSignificantBuy:
    """DAT-21: liczba akcji nie mówi o skali — 10 000 akcji po 5 USD i po 900 USD
    to różne sygnały. Znaczący zakup = kod P od officera za co najmniej 100k USD."""

    @staticmethod
    def _with_buy(value_usd: float, *, is_officer: bool = True) -> InsiderFlowSnapshot:
        from src.domain.insider_flow import InsiderBuy

        return InsiderFlowSnapshot(
            symbol="AAPL",
            net_shares=10_000.0,
            buy_count=1,
            sell_count=0,
            window_days=90,
            largest_buy=InsiderBuy(role="CEO", value_usd=value_usd, is_officer=is_officer),
        )

    def test_signal_uses_dollar_value_threshold(self) -> None:
        cheap = self._with_buy(10_000 * 5.0)
        large = self._with_buy(1_000 * 900.0)

        assert cheap.significant_buy() is None
        assert large.significant_buy() is not None
        assert large.significant_buy() == large.largest_buy

    def test_director_buy_is_not_significant(self) -> None:
        assert self._with_buy(500_000.0, is_officer=False).significant_buy() is None

    def test_threshold_is_inclusive_and_configurable(self) -> None:
        exact = self._with_buy(100_000.0)

        assert exact.significant_buy() is not None
        assert exact.significant_buy(min_value_usd=250_000.0) is None

    def test_snapshot_without_value_data_has_no_significant_buy(self) -> None:
        snapshot = _snap(5000.0, buy_count=3)

        assert snapshot.significant_buy() is None
        assert snapshot.net_value_usd is None
        assert snapshot.latest_filing_url is None


class TestSignificantBuyThreshold:
    """DAT-21: próg znaczącego zakupu jako ułamek kapitalizacji — 1,2 mln USD to
    sygnał w spółce za 100 mld USD, a szum w spółce za 4 bln USD. Bez kapitalizacji
    próg zostaje 100k USD."""

    def test_threshold_scales_with_the_market_cap(self) -> None:
        from src.domain.insider_flow import significant_buy_threshold

        assert significant_buy_threshold(100e9) == pytest.approx(100_000.0)
        assert significant_buy_threshold(4e12) == pytest.approx(4_000_000.0)

    def test_threshold_never_drops_below_100k_usd(self) -> None:
        # Decyzja właściciela: max(1e-6 × kapitalizacja, 100k USD) — zakup za
        # 10 tys. USD w spółce za 10 mld USD to nie „znaczący zakup”.
        from src.domain.insider_flow import SIGNIFICANT_BUY_MIN_USD, significant_buy_threshold

        assert significant_buy_threshold(10e9) == SIGNIFICANT_BUY_MIN_USD
        assert significant_buy_threshold(250e9) == pytest.approx(250_000.0)

    @pytest.mark.parametrize("market_cap", [None, 0.0, -5.0, float("nan"), float("inf")])
    def test_without_a_usable_market_cap_the_threshold_is_100k_usd(
        self, market_cap: float | None
    ) -> None:
        from src.domain.insider_flow import SIGNIFICANT_BUY_MIN_USD, significant_buy_threshold

        assert significant_buy_threshold(market_cap) == SIGNIFICANT_BUY_MIN_USD

    def test_the_same_purchase_is_significant_only_in_the_smaller_company(self) -> None:
        from src.domain.insider_flow import InsiderBuy, significant_buy_threshold

        snapshot = InsiderFlowSnapshot(
            symbol="NVDA",
            net_shares=10_000.0,
            buy_count=1,
            sell_count=0,
            window_days=90,
            largest_buy=InsiderBuy(role="CEO", value_usd=1_200_000.0),
        )

        assert snapshot.significant_buy(significant_buy_threshold(100e9)) is not None
        assert snapshot.significant_buy(significant_buy_threshold(4e12)) is None


class TestInsiderLeadWindow:
    """DAT-21 (decyzja właściciela): zakup insidera prowadzi maila tylko przez
    N dni od daty filingu — inaczej jeden Form 4 wracałby do leadu w każdym cyklu,
    dopóki jest w 90-dniowym oknie insider flow."""

    @staticmethod
    def _buy(filed_on: date | None) -> InsiderBuy:
        return InsiderBuy(role="CEO", value_usd=1_200_000.0, filed_on=filed_on)

    def test_buy_leads_up_to_n_days_after_the_filing(self) -> None:
        from src.domain.insider_flow import InsiderLeadWindow

        window = InsiderLeadWindow(as_of=date(2026, 10, 26), max_age_days=3)

        # Piątkowy filing widzi dopiero poniedziałkowy cykl (3 dni) — wchodzi.
        assert window.admits(self._buy(date(2026, 10, 23)))
        assert window.admits(self._buy(date(2026, 10, 26)))

    def test_older_filing_does_not_lead(self) -> None:
        from src.domain.insider_flow import InsiderLeadWindow

        window = InsiderLeadWindow(as_of=date(2026, 10, 26), max_age_days=3)

        assert not window.admits(self._buy(date(2026, 10, 22)))

    def test_buy_without_a_filing_date_keeps_leading(self) -> None:
        # Adapter bez dat (np. starszy) — bez dowodu wieku zachowanie sprzed okna.
        from src.domain.insider_flow import InsiderLeadWindow

        window = InsiderLeadWindow(as_of=date(2026, 10, 26), max_age_days=3)

        assert window.admits(self._buy(None))

    def test_default_window_is_three_days(self) -> None:
        from src.domain.insider_flow import INSIDER_LEAD_MAX_AGE_DAYS, InsiderLeadWindow

        assert INSIDER_LEAD_MAX_AGE_DAYS == 3
        assert InsiderLeadWindow(as_of=date(2026, 10, 26)).max_age_days == 3
