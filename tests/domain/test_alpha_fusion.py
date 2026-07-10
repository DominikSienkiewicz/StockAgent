"""Testy domeny Alpha Fusion Score — deterministyczna fuzja sygnałów alfa.

Czysta logika: ważona suma ISTNIEJĄCYCH klasyfikacji domenowych (insider,
options, analyst, social) z renormalizacją wag przy brakujących źródłach i
dampeningiem earnings. Bez I/O, bez adapterów.

Testy zamrażają największe ryzyko ze spec ("ręcznie dobrane wagi okażą się
szumem — walk-forward gate decyduje, nie intuicja"):
  (a) wagi to jawne stałe modułu (`ALPHA_WEIGHTS`), nie magic numbers,
  (b) brak wszystkich źródeł → score 0.0,
  (c) renormalizacja: pojedyncze źródło o maksymalnym sygnale → score bliski
      jego znaku, a NIE ułamek przeskalowany brakującymi źródłami,
  (d) rozbicie wkładów (`contributions`) sumuje się DOKŁADNIE do `score`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.alpha_fusion import (
    ALPHA_WEIGHTS,
    AlphaFusionScore,
    fuse_alpha_signals,
)
from src.domain.analyst_consensus import AnalystConsensus
from src.domain.earnings import EarningsEvent
from src.domain.insider_flow import InsiderFlowSnapshot
from src.domain.options_flow import OptionsFlowSnapshot
from src.domain.social_velocity import SocialVelocitySnapshot


def _insider(net_shares: float) -> InsiderFlowSnapshot:
    """Insider z jednoznacznym kierunkiem (jedna transakcja, by ominąć NEUTRAL)."""
    return InsiderFlowSnapshot(
        symbol="AAPL",
        net_shares=net_shares,
        buy_count=1 if net_shares > 0 else 0,
        sell_count=1 if net_shares < 0 else 0,
        window_days=90,
    )


def _options(put_call: float | None, implied_vol: float | None = None) -> OptionsFlowSnapshot:
    return OptionsFlowSnapshot(symbol="AAPL", put_call_ratio=put_call, implied_vol=implied_vol)


def _social(mentions: int, baseline: float, sentiment: float | None) -> SocialVelocitySnapshot:
    return SocialVelocitySnapshot(
        symbol="AAPL",
        mentions_24h=mentions,
        baseline_mentions=baseline,
        avg_sentiment=sentiment,
    )


def _analyst(**counts: int) -> AnalystConsensus:
    return AnalystConsensus(symbol="AAPL", **counts)


# ---------------------------------------------------------------------------
# (a) Wagi jako jawne stałe modułu — nie magic numbers.
# ---------------------------------------------------------------------------
class TestWeightsAreExplicitConstants:
    def test_weights_cover_the_four_directional_sources(self) -> None:
        assert set(ALPHA_WEIGHTS) == {"insider", "options", "analyst", "social"}

    def test_weights_sum_to_one(self) -> None:
        # Pełny zestaw wag sumuje się do 1.0 (renormalizacja liczy od tego bazuje).
        assert sum(ALPHA_WEIGHTS.values()) == pytest.approx(1.0)

    def test_weights_are_positive(self) -> None:
        assert all(w > 0.0 for w in ALPHA_WEIGHTS.values())


# ---------------------------------------------------------------------------
# (b) Brak wszystkich źródeł → 0.0 (zachowanie identyczne jak dziś).
# ---------------------------------------------------------------------------
class TestAllSourcesMissing:
    def test_no_sources_returns_zero(self) -> None:
        result = fuse_alpha_signals()
        assert isinstance(result, AlphaFusionScore)
        assert result.score == 0.0
        assert result.contributions == {}
        assert result.available_sources == ()

    def test_only_earnings_present_still_zero(self) -> None:
        # Earnings to modyfikator zaufania, nie kierunkowy głos — sam nie tworzy score.
        result = fuse_alpha_signals(earnings=EarningsEvent(symbol="AAPL", days_until=1))
        assert result.score == 0.0
        assert result.contributions == {}


# ---------------------------------------------------------------------------
# (c) Renormalizacja: pojedyncze źródło o maksymalnym sygnale → score ~ znak.
# ---------------------------------------------------------------------------
class TestRenormalizationSingleSource:
    def test_single_insider_buy_gives_full_positive(self) -> None:
        result = fuse_alpha_signals(insider=_insider(1_000_000))
        assert result.score == pytest.approx(1.0)
        assert result.available_sources == ("insider",)

    def test_single_insider_sell_gives_full_negative(self) -> None:
        result = fuse_alpha_signals(insider=_insider(-1_000_000))
        assert result.score == pytest.approx(-1.0)

    def test_single_options_bearish_gives_full_negative(self) -> None:
        result = fuse_alpha_signals(options=_options(1.5))
        assert result.score == pytest.approx(-1.0)

    def test_single_source_is_not_scaled_by_missing_sources(self) -> None:
        # Sedno ryzyka: 1 dostępne źródło o max sygnale NIE może dostać
        # sztucznie niskiego score przez brakujące źródła. Bez renormalizacji
        # dostałoby raw wagę (np. 0.35) zamiast ~1.0.
        result = fuse_alpha_signals(insider=_insider(1_000_000))
        assert result.score > 0.99
        assert result.score != pytest.approx(ALPHA_WEIGHTS["insider"])


# ---------------------------------------------------------------------------
# (d) Rozbicie wkładów sumuje się do score — audytowalność.
# ---------------------------------------------------------------------------
class TestContributionsSumToScore:
    def test_contributions_sum_equals_score_multi_source(self) -> None:
        result = fuse_alpha_signals(
            insider=_insider(1_000_000),
            options=_options(0.5),
            social=_social(300, 100, -0.8),
            analyst=_analyst(buy=3, hold=1),
        )
        assert sum(result.contributions.values()) == pytest.approx(result.score)

    def test_contributions_keyed_by_available_sources(self) -> None:
        result = fuse_alpha_signals(insider=_insider(1_000_000), options=_options(0.5))
        assert set(result.contributions) == {"insider", "options"}
        assert set(result.available_sources) == {"insider", "options"}


# ---------------------------------------------------------------------------
# Mapowanie kierunkowe per źródło (reużycie klasyfikacji domenowych).
# ---------------------------------------------------------------------------
class TestPerSourceDirection:
    def test_options_bullish_is_positive(self) -> None:
        assert fuse_alpha_signals(options=_options(0.5)).score > 0

    def test_options_neutral_present_but_zero(self) -> None:
        # P/C w martwej strefie → NEUTRAL → wkład 0, ale źródło jest dostępne.
        result = fuse_alpha_signals(options=_options(0.85))
        assert result.score == pytest.approx(0.0)
        assert result.available_sources == ("options",)

    def test_analyst_strong_buy_is_full_positive(self) -> None:
        result = fuse_alpha_signals(analyst=_analyst(strong_buy=5))
        assert result.score == pytest.approx(1.0)

    def test_analyst_buy_is_half_positive(self) -> None:
        # BUY → sygnał +0.5; jedyne źródło → renorm waga 1.0 → score 0.5.
        result = fuse_alpha_signals(analyst=_analyst(buy=5))
        assert result.score == pytest.approx(0.5)

    def test_analyst_strong_sell_is_full_negative(self) -> None:
        result = fuse_alpha_signals(analyst=_analyst(strong_sell=5))
        assert result.score == pytest.approx(-1.0)

    def test_social_surge_positive_sentiment_is_positive(self) -> None:
        result = fuse_alpha_signals(social=_social(300, 100, 0.9))
        assert result.score == pytest.approx(1.0)

    def test_social_surge_negative_sentiment_is_negative(self) -> None:
        result = fuse_alpha_signals(social=_social(300, 100, -0.9))
        assert result.score == pytest.approx(-1.0)

    def test_social_surge_without_sentiment_is_zero(self) -> None:
        # Skok wzmianek bez kierunku sentymentu → brak kierunku → wkład 0.
        result = fuse_alpha_signals(social=_social(300, 100, None))
        assert result.score == pytest.approx(0.0)

    def test_social_normal_is_zero(self) -> None:
        result = fuse_alpha_signals(social=_social(100, 100, 0.9))
        assert result.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Renormalizacja przy częściowym zestawie źródeł.
# ---------------------------------------------------------------------------
class TestRenormalizationPartial:
    def test_two_agreeing_sources_reach_extreme(self) -> None:
        # Dwa zgodne max sygnały → renorm wagi sumują się do 1 → score ~1.0.
        result = fuse_alpha_signals(insider=_insider(1_000_000), options=_options(0.4))
        assert result.score == pytest.approx(1.0)

    def test_opposing_equal_sources_cancel(self) -> None:
        result = fuse_alpha_signals(
            insider=_insider(1_000_000),
            analyst=_analyst(strong_sell=5),
        )
        # insider +1 * w_i/(w_i+w_a) + analyst -1 * w_a/(w_i+w_a).
        w_i, w_a = ALPHA_WEIGHTS["insider"], ALPHA_WEIGHTS["analyst"]
        expected = (w_i - w_a) / (w_i + w_a)
        assert result.score == pytest.approx(expected)

    def test_present_neutral_source_dilutes_composite(self) -> None:
        # Obecne-neutralne źródło ZOSTAJE w mianowniku (to realna informacja
        # "brak sygnału"), więc rozcieńcza — inaczej niż BRAKUJĄCE źródło.
        with_neutral = fuse_alpha_signals(
            insider=_insider(1_000_000),
            options=_options(0.85),  # NEUTRAL, obecne
        )
        alone = fuse_alpha_signals(insider=_insider(1_000_000))
        assert with_neutral.score < alone.score


# ---------------------------------------------------------------------------
# Earnings jako dampener zaufania (reużycie EarningsProximity).
# ---------------------------------------------------------------------------
class TestEarningsDampening:
    def test_imminent_earnings_shrinks_magnitude_toward_zero(self) -> None:
        base = fuse_alpha_signals(insider=_insider(1_000_000))
        damped = fuse_alpha_signals(
            insider=_insider(1_000_000),
            earnings=EarningsEvent(symbol="AAPL", days_until=1),
        )
        assert abs(damped.score) < abs(base.score)
        assert damped.earnings_confidence < 1.0

    def test_far_earnings_no_dampening(self) -> None:
        result = fuse_alpha_signals(
            insider=_insider(1_000_000),
            earnings=EarningsEvent(symbol="AAPL", days_until=90),
        )
        assert result.score == pytest.approx(1.0)
        assert result.earnings_confidence == pytest.approx(1.0)

    def test_dampened_contributions_still_sum_to_score(self) -> None:
        # Dampening jest wtopiony we wkłady → niezmiennik (d) trzyma się nadal.
        result = fuse_alpha_signals(
            insider=_insider(1_000_000),
            options=_options(0.4),
            earnings=EarningsEvent(symbol="AAPL", days_until=1),
        )
        assert sum(result.contributions.values()) == pytest.approx(result.score)


# ---------------------------------------------------------------------------
# Niezmienniki wyniku.
# ---------------------------------------------------------------------------
class TestResultInvariants:
    def test_score_always_within_unit_interval(self) -> None:
        result = fuse_alpha_signals(
            insider=_insider(1_000_000),
            options=_options(0.3),
            social=_social(400, 100, 0.95),
            analyst=_analyst(strong_buy=10),
        )
        assert -1.0 <= result.score <= 1.0

    def test_result_is_frozen(self) -> None:
        result = fuse_alpha_signals(insider=_insider(1_000_000))
        with pytest.raises((AttributeError, TypeError)):
            result.score = 0.0  # type: ignore[misc]

    def test_analyst_upside_unused_no_decimal_leak(self) -> None:
        # Sanity: fuzja korzysta z rating(), nie upside() — brak wycieku Decimal.
        result = fuse_alpha_signals(analyst=_analyst(hold=5))
        assert isinstance(result.score, float)
        # Kontrola, że nasz helper nie potrzebuje ceny (Decimal) do klasyfikacji.
        assert AnalystConsensus(symbol="AAPL", hold=5).upside(Decimal("100")) is None
