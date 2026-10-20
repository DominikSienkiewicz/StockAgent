import pytest

from src.domain.sizing import SizeBand, suggest_band


class TestSuggestBand:
    """Kelly-lite mapowanie konsensus rady + track record → wielkość pozycji.

    Reguła jest deterministyczna i czysta: mocny konsensus + niski dissent +
    przyzwoity hit-rate windują pozycję ku górnej bandzie; słabość któregokolwiek
    czynnika (albo brak hit-rate) ściąga sugestię do bandy startowej.
    """

    def test_strong_consensus_low_dissent_good_hitrate_gives_full_band(self) -> None:
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.0, hit_rate=0.7)
        assert band.tier == "full"
        # Górna banda — realna, agresywniejsza alokacja.
        assert band.min_pct == pytest.approx(4.0)
        assert band.max_pct == pytest.approx(5.0)
        assert "%" in band.label

    def test_weak_consensus_gives_starter_band(self) -> None:
        band = suggest_band(consensus_strength=0.4, dissent_ratio=0.0, hit_rate=0.7)
        assert band.tier == "starter"
        assert band.min_pct == pytest.approx(1.0)
        assert band.max_pct == pytest.approx(2.0)

    def test_high_dissent_pulls_down_to_starter(self) -> None:
        # Konsensus i hit-rate mocne, ale rada się sypie pod werdyktem.
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.6, hit_rate=0.7)
        assert band.tier == "starter"

    def test_unknown_hitrate_is_conservative(self) -> None:
        # Brak track recordu (cold-start) → nigdy nie sięgamy po pełną bandę.
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.0, hit_rate=None)
        assert band.tier != "full"

    def test_poor_hitrate_is_conservative(self) -> None:
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.0, hit_rate=0.3)
        assert band.tier != "full"

    def test_moderate_factors_give_standard_band(self) -> None:
        # Mocny konsensus + niski dissent + nieznany hit-rate → środkowa banda.
        band = suggest_band(consensus_strength=0.85, dissent_ratio=0.1, hit_rate=None)
        assert band.tier == "standard"
        assert band.min_pct == pytest.approx(2.0)
        assert band.max_pct == pytest.approx(3.0)

    def test_clamps_consensus_above_one(self) -> None:
        # Wejście poza [0,1] nie może wysadzić reguły — clamp do sensownego zakresu.
        band = suggest_band(consensus_strength=5.0, dissent_ratio=-2.0, hit_rate=2.0)
        assert band.tier == "full"

    def test_clamps_negative_inputs_to_conservative(self) -> None:
        band = suggest_band(consensus_strength=-1.0, dissent_ratio=5.0, hit_rate=-1.0)
        assert band.tier == "starter"

    def test_band_is_frozen_value_object(self) -> None:
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.0, hit_rate=0.7)
        import dataclasses

        assert dataclasses.is_dataclass(band)
        try:
            band.tier = "starter"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("SizeBand powinien być frozen")

    def test_label_is_human_readable_polish(self) -> None:
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.0, hit_rate=0.7)
        # Etykieta dla maila — czytelna, polska, zawiera zakres procentowy.
        assert "4" in band.label
        assert "5" in band.label
        assert isinstance(band.label, str)
        assert band.label.strip()

    def test_deterministic(self) -> None:
        a = suggest_band(0.9, 0.0, 0.7)
        b = suggest_band(0.9, 0.0, 0.7)
        assert a == b


class TestSizeBand:
    def test_equality_by_value(self) -> None:
        # Dwie ODRĘBNE instancje o tych samych polach — równość ma iść po
        # wartości, nie po tożsamości obiektu.
        band = SizeBand("full", 4.0, 5.0, "x")
        same_values = SizeBand("full", 4.0, 5.0, "x")
        assert band == same_values


class TestSizingKnowsExistingExposure:
    """#15 faza 2 — `suggest_band` nie znała istniejącej ekspozycji. Sugerowała
    „pełną bandę 4-5%" komuś, kto ma już 40% kapitału w tym symbolu."""

    def test_existing_exposure_can_only_shrink_the_band(self) -> None:
        # Warunki idealne (mocny konsensus, brak dissentu, dobry hit-rate).
        full = suggest_band(0.9, 0.0, 0.7)
        capped = suggest_band(0.9, 0.0, 0.7, current_weight=0.40)

        assert full.tier == "full"
        assert capped.tier != "full"

    def test_exposure_never_upgrades_a_weak_signal(self) -> None:
        # Brak ekspozycji nie ma prawa podbić słabego sygnału.
        weak = suggest_band(0.2, 0.9, None)
        weak_no_exposure = suggest_band(0.2, 0.9, None, current_weight=0.0)

        assert weak.tier == weak_no_exposure.tier == "starter"

    def test_cluster_cap_shrinks_the_band(self) -> None:
        # Trzy pozycje w jednym klastrze korelacji to jedna pozycja przebrana
        # za trzy — banda musi to widzieć.
        full = suggest_band(0.9, 0.0, 0.7)
        clustered = suggest_band(0.9, 0.0, 0.7, cluster_exposure=0.65)

        assert full.tier == "full"
        assert clustered.tier != "full"

    def test_defaults_preserve_the_pre_phase_two_behaviour(self) -> None:
        # Wsteczna kompatybilność: bez nowych argumentów wynik jak dotąd.
        assert suggest_band(0.9, 0.0, 0.7).tier == "full"
        assert suggest_band(0.9, 0.0, None).tier == "standard"
        assert suggest_band(0.2, 0.9, 0.7).tier == "starter"

    def test_full_exposure_forces_the_most_conservative_band(self) -> None:
        assert suggest_band(0.9, 0.0, 0.7, current_weight=0.95).tier == "starter"


# PORT-01 (G31): „Pozycja pełna 4–5%” po jednym trafieniu. Pełna banda wymaga
# efektywnej próby n_eff ≥ MIN_HIT_SAMPLE; mniejsza próba = hit-rate nieznany.


def test_hit_rate_below_min_sample_is_treated_as_unknown() -> None:
    from src.domain.sizing import MIN_HIT_SAMPLE

    tiny = suggest_band(0.9, 0.0, 1.0, hit_sample=MIN_HIT_SAMPLE - 1)

    assert tiny.tier != "full"
    assert tiny == suggest_band(0.9, 0.0, None)


def test_hit_rate_on_min_sample_can_reach_the_full_band() -> None:
    from src.domain.sizing import MIN_HIT_SAMPLE

    assert suggest_band(0.9, 0.0, 0.7, hit_sample=MIN_HIT_SAMPLE).tier == "full"


def test_min_hit_sample_is_twenty_effective_bets() -> None:
    from src.domain.sizing import MIN_HIT_SAMPLE

    assert MIN_HIT_SAMPLE == 20


# PORT-04: VaR, drawdown i skuteczność hedge'a portfela działają jak nadmierna
# ekspozycja — mogą bandę wyłącznie zwęzić.

_TIER_RANK = {"starter": 0, "standard": 1, "full": 2}
# Przypadki sprzed PORT-04: (consensus, dissent, hit_rate, current_weight, cluster).
_PRE_RISK_CASES = [
    (0.9, 0.0, 0.7, None, None),
    (0.9, 0.0, None, None, None),
    (0.85, 0.1, None, None, None),
    (0.9, 0.6, 0.7, None, None),
    (0.4, 0.0, 0.7, None, None),
    (0.9, 0.0, 0.3, None, None),
    (0.9, 0.0, 0.7, 0.40, None),
    (0.9, 0.0, 0.7, None, 0.65),
    (0.9, 0.0, 0.7, 0.95, None),
    (0.2, 0.9, None, 0.0, None),
]


class TestSizingKnowsPortfolioRisk:
    def test_high_one_day_var_lowers_the_band_one_step(self) -> None:
        assert suggest_band(0.9, 0.1, 0.7, portfolio_var=0.05).tier == "standard"

    def test_var_threshold_is_inclusive(self) -> None:
        from src.domain.sizing import VAR_TIGHTEN_THRESHOLD

        assert VAR_TIGHTEN_THRESHOLD == 0.03
        assert suggest_band(0.9, 0.1, 0.7, portfolio_var=0.03).tier == "standard"
        assert suggest_band(0.9, 0.1, 0.7, portfolio_var=0.0299).tier == "full"

    def test_deep_drawdown_forces_the_starter_band(self) -> None:
        assert suggest_band(0.9, 0.1, 0.7, drawdown=0.2).tier == "starter"

    def test_drawdown_threshold_is_inclusive(self) -> None:
        from src.domain.sizing import DEEP_DRAWDOWN_THRESHOLD

        assert DEEP_DRAWDOWN_THRESHOLD == 0.15
        assert suggest_band(0.9, 0.1, 0.7, drawdown=0.15).tier == "starter"
        assert suggest_band(0.9, 0.1, 0.7, drawdown=0.1499).tier == "full"

    def test_ineffective_hedge_lowers_the_band_one_step(self) -> None:
        assert suggest_band(0.9, 0.1, 0.7, hedge_effective=False).tier == "standard"

    def test_effective_hedge_does_not_widen_the_band(self) -> None:
        assert suggest_band(0.9, 0.1, None, hedge_effective=True).tier == "standard"

    def test_risk_steps_add_up(self) -> None:
        # Każdy sygnał ryzyka to osobny stopień; starter jest dnem.
        assert (
            suggest_band(0.9, 0.1, 0.7, portfolio_var=0.05, hedge_effective=False).tier
            == "starter"
        )

    def test_risk_applies_after_the_existing_exposure(self) -> None:
        # Ekspozycja zdejmuje stopień, VaR kolejny.
        assert suggest_band(0.9, 0.1, 0.7, current_weight=0.2).tier == "standard"
        assert (
            suggest_band(0.9, 0.1, 0.7, current_weight=0.2, portfolio_var=0.05).tier
            == "starter"
        )

    @pytest.mark.parametrize(
        ("consensus", "dissent", "hit_rate", "weight", "cluster"), _PRE_RISK_CASES
    )
    def test_no_risk_inputs_keep_the_phase_two_band(
        self,
        consensus: float,
        dissent: float,
        hit_rate: float | None,
        weight: float | None,
        cluster: float | None,
    ) -> None:
        phase_two = suggest_band(
            consensus, dissent, hit_rate, current_weight=weight, cluster_exposure=cluster
        )

        assert suggest_band(
            consensus,
            dissent,
            hit_rate,
            current_weight=weight,
            cluster_exposure=cluster,
            portfolio_var=None,
            drawdown=None,
            hedge_effective=None,
        ) == phase_two

    def test_risk_never_widens_the_band(self) -> None:
        import itertools

        grid = itertools.product(
            (0.2, 0.75, 0.9),  # konsensus
            (0.0, 0.5),  # dissent
            (None, 0.4, 0.7),  # hit-rate
            (None, 0.0, 0.2, 0.35),  # waga symbolu
            (None, 0.45, 0.7),  # ekspozycja klastra
            (None, 0.0, 0.02, 0.03, 0.2, -1.0),  # VaR
            (None, 0.0, 0.1, 0.15, 0.5, -1.0),  # drawdown
            (None, True, False),  # hedge skuteczny
        )
        for consensus, dissent, hit, weight, cluster, var, dd, hedge in grid:
            base = suggest_band(
                consensus, dissent, hit, current_weight=weight, cluster_exposure=cluster
            )
            risky = suggest_band(
                consensus,
                dissent,
                hit,
                current_weight=weight,
                cluster_exposure=cluster,
                portfolio_var=var,
                drawdown=dd,
                hedge_effective=hedge,
            )
            assert _TIER_RANK[risky.tier] <= _TIER_RANK[base.tier], (
                consensus, dissent, hit, weight, cluster, var, dd, hedge
            )
