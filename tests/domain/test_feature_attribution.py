from src.domain.feature_attribution import (
    FeatureContribution,
    top_contributions,
)


class TestFeatureContribution:
    """Czysty value object: nazwa cechy + jej znakowany wkład w predykcję."""

    def test_is_frozen(self):
        contribution = FeatureContribution(feature="price_delta", contribution=0.5)
        try:
            contribution.contribution = 1.0  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 — interesuje nas tylko, że rzuca
            raised = exc
        else:
            raised = None
        assert raised is not None

    def test_holds_name_and_signed_value(self):
        contribution = FeatureContribution(feature="av_sentiment_score", contribution=-0.3)
        assert contribution.feature == "av_sentiment_score"
        assert contribution.contribution == -0.3


class TestTopContributions:
    """`top_contributions` rankuje wkłady po WARTOŚCI BEZWZGLĘDNEJ (siła pchnięcia
    predykcji w górę/dół), tnie do top_n i jest stabilne przy remisach."""

    def test_ranks_by_absolute_value_descending(self):
        contribs = {
            "a": 0.1,
            "b": -0.9,
            "c": 0.5,
            "d": -0.2,
        }
        ranked = top_contributions(contribs)
        names = [c.feature for c in ranked]
        # |b|=0.9 > |c|=0.5 > |d|=0.2 > |a|=0.1 — silny ujemny wkład na czele.
        assert names == ["b", "c", "d", "a"]

    def test_preserves_signed_contribution(self):
        contribs = {"up": 0.4, "down": -0.7}
        ranked = top_contributions(contribs)
        as_dict = {c.feature: c.contribution for c in ranked}
        assert as_dict["up"] == 0.4
        assert as_dict["down"] == -0.7

    def test_caps_at_top_n(self):
        contribs = {name: float(idx) for idx, name in enumerate("abcdefgh")}
        ranked = top_contributions(contribs, top_n=3)
        assert len(ranked) == 3
        # Trzy najsilniejsze co do modułu: h=7, g=6, f=5.
        assert [c.feature for c in ranked] == ["h", "g", "f"]

    def test_default_top_n_is_five(self):
        contribs = {name: float(idx) for idx, name in enumerate("abcdefgh")}
        ranked = top_contributions(contribs)
        assert len(ranked) == 5

    def test_fewer_than_top_n_returns_all(self):
        contribs = {"a": 0.2, "b": -0.4}
        ranked = top_contributions(contribs, top_n=5)
        assert len(ranked) == 2

    def test_empty_input_returns_empty(self):
        assert top_contributions({}) == ()

    def test_ties_are_stable_by_insertion_order(self):
        # Trzy cechy o identycznym module wkładu — sort musi zachować kolejność
        # wstawienia (stabilny), żeby raport był deterministyczny między cyklami.
        contribs = {"first": 0.5, "second": -0.5, "third": 0.5}
        ranked = top_contributions(contribs)
        assert [c.feature for c in ranked] == ["first", "second", "third"]

    def test_returns_tuple(self):
        ranked = top_contributions({"a": 1.0})
        assert isinstance(ranked, tuple)
