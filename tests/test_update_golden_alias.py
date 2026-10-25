"""TEST-10: `UPDATE_GOLDEN=1` jako alias opcji `--update-golden`.

Zmienna działa tam, gdzie nie da się dopisać opcji do wywołania pytest (konfiguracja
IDE). W CI (`CI` ustawione przez GitHub Actions) jest ignorowana: snapshot przepisany
po cichu na runnerze zazieleniłby zmianę, której nikt nie obejrzał.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests import conftest as suite_conftest


def _update_golden_default() -> object:
    parser = MagicMock()
    suite_conftest.pytest_addoption(parser)
    (call,) = [
        c for c in parser.addoption.call_args_list if c.args and c.args[0] == "--update-golden"
    ]
    return call.kwargs["default"]


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UPDATE_GOLDEN", raising=False)
    monkeypatch.delenv("CI", raising=False)


def test_env_alias_turns_the_update_option_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPDATE_GOLDEN", "1")

    assert _update_golden_default() is True


def test_without_the_variable_golden_files_are_compared() -> None:
    assert _update_golden_default() is False


@pytest.mark.parametrize("value", ["0", "", "true", "yes"])
def test_only_the_value_one_counts(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("UPDATE_GOLDEN", value)

    assert _update_golden_default() is False


def test_the_alias_never_rewrites_golden_files_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPDATE_GOLDEN", "1")
    monkeypatch.setenv("CI", "true")

    assert _update_golden_default() is False


def test_the_explicit_option_stays_a_store_true_flag() -> None:
    parser = MagicMock()
    suite_conftest.pytest_addoption(parser)
    (call,) = [
        c for c in parser.addoption.call_args_list if c.args and c.args[0] == "--update-golden"
    ]

    assert call.kwargs["action"] == "store_true"
