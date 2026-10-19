"""WIA-20: przepis predykcji i jego wersja (`recipe_version`)."""

from __future__ import annotations

import dataclasses

from src.domain.equity_curve import DirectionalHitRate
from src.domain.recipe import (
    RECIPE_FIELDS,
    UNVERSIONED,
    PredictionRecipe,
    hit_rate_by_version,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _recipe(**overrides: str) -> PredictionRecipe:
    fields: dict[str, str] = {
        "llm_provider": "anthropic",
        "llm_model": "claude-sonnet-4-6",
        "council_provider": "openai",
        "council_model": "gpt-5-mini",
        "prompt_sha256": _SHA_A,
        "personas_sha256": _SHA_A,
        "config_hash": _SHA_A,
        "model_version": "cold_start",
        "git_sha": "abc1234",
        "run_id": "77",
    }
    fields.update(overrides)
    return PredictionRecipe(**fields)


def test_recipe_hash_stable_for_same_inputs() -> None:
    assert _recipe().version == _recipe().version
    assert len(_recipe().version) == 64


def test_recipe_hash_changes_when_prompt_template_changes() -> None:
    assert _recipe(prompt_sha256=_SHA_A).version != _recipe(prompt_sha256=_SHA_B).version


def test_recipe_hash_changes_with_model_personas_config_and_ml_weights() -> None:
    base = _recipe().version

    assert _recipe(llm_model="gpt-4o").version != base
    assert _recipe(council_model="claude-haiku-4-5").version != base
    assert _recipe(personas_sha256=_SHA_B).version != base
    assert _recipe(config_hash=_SHA_B).version != base
    assert _recipe(model_version=_SHA_B).version != base


def test_run_identity_does_not_change_the_recipe_version() -> None:
    """Ten sam przepis w dwóch przebiegach to wciąż jeden przepis — inaczej każdy
    run byłby osobnym segmentem track recordu."""
    assert _recipe(git_sha="abc1234", run_id="77").version == _recipe(
        git_sha="def5678", run_id="78"
    ).version


def test_receipt_fields_carry_every_recipe_field_and_the_version() -> None:
    recipe = _recipe()

    fields = recipe.receipt_fields()

    assert set(fields) == {*RECIPE_FIELDS, "recipe_version"}
    assert fields["recipe_version"] == recipe.version
    assert fields["llm_model"] == "claude-sonnet-4-6"


def test_recipe_fields_follow_the_dataclass() -> None:
    assert tuple(f.name for f in dataclasses.fields(PredictionRecipe)) == RECIPE_FIELDS


def test_hit_rate_by_version_segments_directional_calls() -> None:
    calls = [
        ("v1", "BULLISH", True),
        ("v1", "BEARISH", False),
        ("v1", "SIDEWAYS", True),
        ("v2", "BULLISH", True),
        (None, "BEARISH", True),
    ]

    segments = hit_rate_by_version(calls)

    assert segments == {
        "v1": DirectionalHitRate(hits=1, directional=2, sideways=1),
        "v2": DirectionalHitRate(hits=1, directional=1, sideways=0),
        UNVERSIONED: DirectionalHitRate(hits=1, directional=1, sideways=0),
    }


def test_hit_rate_by_version_keeps_first_seen_order() -> None:
    calls = [("v2", "BULLISH", True), ("v1", "BULLISH", True), ("v2", "BEARISH", True)]

    assert list(hit_rate_by_version(calls)) == ["v2", "v1"]
