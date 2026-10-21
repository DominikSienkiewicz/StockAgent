"""Globalna konfiguracja testów.

Izolacja od `config.toml`: testy jednostkowe asertują czyste defaulty pól
`Settings` oraz przekazują config przez jawne kwargs / fixture'y. Gdyby źródło
TOML było aktywne, `config.toml` z repo nadpisywałby defaulty (np.
`nbp_enabled=true`, `crypto_symbols=[BTC,ETH]`) i psuł te asercje.

Flagę ustawiamy na poziomie importu — ZANIM jakikolwiek test zbuduje `Settings`.
Testy, które celowo weryfikują ładowanie z TOML (`test_config_toml.py`),
zdejmują ją lokalnie przez `monkeypatch.delenv`.
"""

import os

import pytest

os.environ["STOCKAGENT_DISABLE_TOML"] = "1"
# NIEZ-22: w CI (GitHub Actions) zmienna wskazuje summary jobu testów — testy
# entry pointów dopisywałyby do niej podsumowania cykli zbudowanych z mocków.
os.environ.pop("GITHUB_STEP_SUMMARY", None)


def pytest_addoption(parser: pytest.Parser) -> None:
    # TEST-02: jawna (i jedyna) droga do przepisania golden snapshotów maila;
    # TEST-10: ta sama dla golden promptów.
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help=(
            "przepisz snapshoty tests/application/snapshots/ i tests/golden/prompts/ "
            "zamiast je porównywać"
        ),
    )
