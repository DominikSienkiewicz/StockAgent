"""Testy domeny przepływów opcji — progi sentymentu i flaga wysokiej IV."""

from src.domain.options_flow import OptionsFlowSnapshot, OptionsSentiment


class TestSentiment:
    def test_low_put_call_ratio_is_bullish(self) -> None:
        snapshot = OptionsFlowSnapshot(
            symbol="AAPL", put_call_ratio=0.5, implied_vol=0.3
        )

        assert snapshot.sentiment() is OptionsSentiment.BULLISH

    def test_high_put_call_ratio_is_bearish(self) -> None:
        snapshot = OptionsFlowSnapshot(
            symbol="AAPL", put_call_ratio=1.4, implied_vol=0.3
        )

        assert snapshot.sentiment() is OptionsSentiment.BEARISH

    def test_mid_put_call_ratio_is_neutral(self) -> None:
        # Pomiędzy bullish_below (0.7) a bearish_above (1.0).
        snapshot = OptionsFlowSnapshot(
            symbol="AAPL", put_call_ratio=0.85, implied_vol=0.3
        )

        assert snapshot.sentiment() is OptionsSentiment.NEUTRAL

    def test_ratio_exactly_at_bullish_threshold_is_neutral(self) -> None:
        # < bullish_below jest bycze; równe progowi już nie.
        snapshot = OptionsFlowSnapshot(symbol="AAPL", put_call_ratio=0.7)

        assert snapshot.sentiment() is OptionsSentiment.NEUTRAL

    def test_ratio_exactly_at_bearish_threshold_is_neutral(self) -> None:
        # > bearish_above jest niedźwiedzie; równe progowi już nie.
        snapshot = OptionsFlowSnapshot(symbol="AAPL", put_call_ratio=1.0)

        assert snapshot.sentiment() is OptionsSentiment.NEUTRAL

    def test_none_ratio_is_neutral(self) -> None:
        snapshot = OptionsFlowSnapshot(symbol="AAPL", put_call_ratio=None)

        assert snapshot.sentiment() is OptionsSentiment.NEUTRAL


class TestIsHighIv:
    def test_iv_above_threshold_is_high(self) -> None:
        snapshot = OptionsFlowSnapshot(
            symbol="AAPL", put_call_ratio=0.8, implied_vol=0.75
        )

        assert snapshot.is_high_iv() is True

    def test_iv_at_threshold_is_high(self) -> None:
        snapshot = OptionsFlowSnapshot(
            symbol="AAPL", put_call_ratio=0.8, implied_vol=0.6
        )

        assert snapshot.is_high_iv() is True

    def test_iv_below_threshold_is_not_high(self) -> None:
        snapshot = OptionsFlowSnapshot(
            symbol="AAPL", put_call_ratio=0.8, implied_vol=0.4
        )

        assert snapshot.is_high_iv() is False

    def test_none_iv_is_not_high(self) -> None:
        snapshot = OptionsFlowSnapshot(symbol="AAPL", put_call_ratio=0.8)

        assert snapshot.is_high_iv() is False
