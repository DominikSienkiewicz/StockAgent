"""WIA-20: czytnik kwitów decyzyjnych — v1 (sprzed przepisu) i v2 (z przepisem).

Kwity w JSONB żyją miesiącami, a schemat rośnie: v2 dokłada pola przepisu,
WIA-66 dołoży w v2 opcjonalny klucz `provenance` bez kolejnego bumpu. Czytnik
nie może się wywrócić na braku opcjonalnego klucza ani zgadywać nieznanej wersji.
"""

from __future__ import annotations

from src.application.decision_receipt import (
    RECEIPT_SCHEMA_V1,
    RECEIPT_SCHEMA_V2,
    read_decision_receipt,
)
from src.domain.recipe import PredictionRecipe

_V1 = {
    "schema_version": 1,
    "effective_threshold": "0.02",
    "precedents": [{"id": "pred-1", "similarity": 0.9}],
    "feature_attribution": [{"feature": "price_delta", "contribution": 0.01}],
}


def _recipe() -> PredictionRecipe:
    return PredictionRecipe(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-6",
        council_provider="openai",
        council_model="gpt-5-mini",
        prompt_sha256="a" * 64,
        personas_sha256="b" * 64,
        config_hash="c" * 64,
        model_version="cold_start",
        universe_version="d" * 64,
        git_sha="abc1234",
        run_id="77",
    )


def test_schema_versions_are_one_and_two() -> None:
    assert (RECEIPT_SCHEMA_V1, RECEIPT_SCHEMA_V2) == (1, 2)


def test_v1_receipt_reads_without_a_recipe() -> None:
    receipt = read_decision_receipt(_V1)

    assert receipt is not None
    assert receipt.schema_version == 1
    assert receipt.effective_threshold == "0.02"
    assert receipt.precedents == ({"id": "pred-1", "similarity": 0.9},)
    assert receipt.feature_attribution == (
        {"feature": "price_delta", "contribution": 0.01},
    )
    assert receipt.recipe == {}
    assert receipt.recipe_version is None


def test_v2_receipt_reads_the_recipe() -> None:
    recipe = _recipe()
    payload = {**_V1, "schema_version": 2, **recipe.receipt_fields()}

    receipt = read_decision_receipt(payload)

    assert receipt is not None
    assert receipt.schema_version == 2
    assert receipt.recipe_version == recipe.version
    assert receipt.recipe["llm_model"] == "claude-sonnet-4-6"
    assert receipt.recipe["git_sha"] == "abc1234"


def test_v2_tolerates_missing_optional_keys_and_an_unknown_provenance_key() -> None:
    # WIA-66 doda `provenance` w v2; starsze kwity v2 go nie mają i odwrotnie.
    payload = {"schema_version": 2, "llm_model": "gpt-4o", "provenance": [{"source": "AV"}]}

    receipt = read_decision_receipt(payload)

    assert receipt is not None
    assert receipt.recipe == {"llm_model": "gpt-4o"}
    assert receipt.recipe_version is None
    assert receipt.effective_threshold is None
    assert receipt.precedents == ()


def test_unknown_or_missing_schema_version_is_not_guessed() -> None:
    assert read_decision_receipt({**_V1, "schema_version": 3}) is None
    assert read_decision_receipt({"effective_threshold": "0.02"}) is None
    assert read_decision_receipt(None) is None
    assert read_decision_receipt("not a receipt") is None


def test_v2_reads_the_experiment_arms_and_tolerates_their_absence() -> None:
    # EXP-05: opcjonalny klucz v2 — bez bumpu wersji, brak = brak eksperymentu.
    with_arms = read_decision_receipt(
        {"schema_version": 2, "experiment_arms": {"EXP-02": "blind", "bad": 3}}
    )
    without_arms = read_decision_receipt({"schema_version": 2})

    assert with_arms is not None and without_arms is not None
    assert with_arms.experiment_arms == {"EXP-02": "blind"}
    assert without_arms.experiment_arms == {}


def test_v1_receipt_has_no_experiment_arms() -> None:
    receipt = read_decision_receipt({**_V1, "experiment_arms": {"EXP-02": "blind"}})

    assert receipt is not None
    assert receipt.experiment_arms == {}


def test_v2_reads_the_model_used_and_tolerates_its_absence() -> None:
    # FIN-21: opcjonalny klucz v2 — model, który odpowiedział przy routingu modeli.
    routed = read_decision_receipt({"schema_version": 2, "model_used": "openai:gpt-5-mini"})
    plain = read_decision_receipt({"schema_version": 2})
    garbage = read_decision_receipt({"schema_version": 2, "model_used": 7})

    assert routed is not None and plain is not None and garbage is not None
    assert routed.model_used == "openai:gpt-5-mini"
    assert plain.model_used is None
    assert garbage.model_used is None


def test_v1_receipt_has_no_model_used() -> None:
    receipt = read_decision_receipt({**_V1, "model_used": "openai:gpt-5-mini"})

    assert receipt is not None
    assert receipt.model_used is None
