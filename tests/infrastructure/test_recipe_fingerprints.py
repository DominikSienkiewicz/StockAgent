"""WIA-20: odciski składników przepisu, które wymagają czytania plików —
szablony promptów (kod modułów, nie wyrenderowany prompt) i wagi modelu ML."""

from __future__ import annotations

from pathlib import Path

from src.infrastructure.recipe_fingerprints import (
    COLD_START_MODEL_VERSION,
    PROMPT_TEMPLATE_MODULES,
    ml_model_version,
    prompt_templates_sha256,
)


def test_prompt_hash_is_stable_sha256() -> None:
    assert prompt_templates_sha256() == prompt_templates_sha256()
    assert len(prompt_templates_sha256()) == 64


def test_prompt_hash_covers_the_prediction_and_council_templates() -> None:
    names = {module.__name__ for module in PROMPT_TEMPLATE_MODULES}

    assert {
        "src.application.prompts",
        "src.application.council_prompts",
        "src.application._prompt_safety",
    } <= names


def test_missing_weights_are_the_cold_start_version(tmp_path: Path) -> None:
    assert COLD_START_MODEL_VERSION == "cold_start"
    assert ml_model_version(str(tmp_path / "missing.ubj")) == "cold_start"


def test_weights_version_is_the_file_sha256(tmp_path: Path) -> None:
    weights = tmp_path / "model.ubj"
    weights.write_bytes(b"weights-v1")
    first = ml_model_version(str(weights))

    weights.write_bytes(b"weights-v2")

    assert len(first) == 64
    assert ml_model_version(str(weights)) != first
