from src.domain.sizing import SizeBand, suggest_band


class TestSuggestBand:
    """Kelly-lite mapowanie konsensus rady + track record → wielkość pozycji.

    Reguła jest deterministyczna i czysta: mocny konsensus + niski dissent +
    przyzwoity hit-rate windują pozycję ku górnej bandzie; słabość któregokolwiek
    czynnika (albo brak hit-rate) ściąga sugestię do bandy startowej.
    """

    def test_strong_consensus_low_dissent_good_hitrate_gives_full_band(self):
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.0, hit_rate=0.7)
        assert band.tier == "full"
        # Górna banda — realna, agresywniejsza alokacja.
        assert band.min_pct == 4.0
        assert band.max_pct == 5.0
        assert "%" in band.label

    def test_weak_consensus_gives_starter_band(self):
        band = suggest_band(consensus_strength=0.4, dissent_ratio=0.0, hit_rate=0.7)
        assert band.tier == "starter"
        assert band.min_pct == 1.0
        assert band.max_pct == 2.0

    def test_high_dissent_pulls_down_to_starter(self):
        # Konsensus i hit-rate mocne, ale rada się sypie pod werdyktem.
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.6, hit_rate=0.7)
        assert band.tier == "starter"

    def test_unknown_hitrate_is_conservative(self):
        # Brak track recordu (cold-start) → nigdy nie sięgamy po pełną bandę.
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.0, hit_rate=None)
        assert band.tier != "full"

    def test_poor_hitrate_is_conservative(self):
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.0, hit_rate=0.3)
        assert band.tier != "full"

    def test_moderate_factors_give_standard_band(self):
        # Mocny konsensus + niski dissent + nieznany hit-rate → środkowa banda.
        band = suggest_band(consensus_strength=0.85, dissent_ratio=0.1, hit_rate=None)
        assert band.tier == "standard"
        assert band.min_pct == 2.0
        assert band.max_pct == 3.0

    def test_clamps_consensus_above_one(self):
        # Wejście poza [0,1] nie może wysadzić reguły — clamp do sensownego zakresu.
        band = suggest_band(consensus_strength=5.0, dissent_ratio=-2.0, hit_rate=2.0)
        assert band.tier == "full"

    def test_clamps_negative_inputs_to_conservative(self):
        band = suggest_band(consensus_strength=-1.0, dissent_ratio=5.0, hit_rate=-1.0)
        assert band.tier == "starter"

    def test_band_is_frozen_value_object(self):
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.0, hit_rate=0.7)
        import dataclasses

        assert dataclasses.is_dataclass(band)
        try:
            band.tier = "starter"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("SizeBand powinien być frozen")

    def test_label_is_human_readable_polish(self):
        band = suggest_band(consensus_strength=0.9, dissent_ratio=0.0, hit_rate=0.7)
        # Etykieta dla maila — czytelna, polska, zawiera zakres procentowy.
        assert "4" in band.label and "5" in band.label
        assert isinstance(band.label, str) and band.label.strip()

    def test_deterministic(self):
        a = suggest_band(0.9, 0.0, 0.7)
        b = suggest_band(0.9, 0.0, 0.7)
        assert a == b


class TestSizeBand:
    def test_equality_by_value(self):
        assert SizeBand("full", 4.0, 5.0, "x") == SizeBand("full", 4.0, 5.0, "x")
