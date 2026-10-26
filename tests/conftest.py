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
from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import settings as hypothesis_settings

from src.config import Settings
from tests.entrypoint_support import build_settings

os.environ["STOCKAGENT_DISABLE_TOML"] = "1"
# NIEZ-22: w CI (GitHub Actions) zmienna wskazuje summary jobu testów — testy
# entry pointów dopisywałyby do niej podsumowania cykli zbudowanych z mocków.
os.environ.pop("GITHUB_STEP_SUMMARY", None)

# TEST-12: profil testów własności. 200 przykładów na własność i bez deadline'u —
# maszyna i runner CI pod obciążeniem robiłyby z limitu czasu flaky test.
# `HYPOTHESIS_PROFILE=default` przywraca ustawienia biblioteki.
hypothesis_settings.register_profile("ci", max_examples=200, deadline=None)
hypothesis_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


def _update_golden_from_env() -> bool:
    """TEST-10: `UPDATE_GOLDEN=1` jako alias `--update-golden` (np. dla konfiguracji IDE,
    gdzie nie da się dopisać opcji). W CI (`CI` ustawione) ignorowany: snapshot przepisany
    po cichu na runnerze zazieleniłby zmianę, której nikt nie obejrzał."""
    return os.environ.get("UPDATE_GOLDEN") == "1" and not os.environ.get("CI")


def pytest_addoption(parser: pytest.Parser) -> None:
    # TEST-02: jawna droga do przepisania golden snapshotów maila (TEST-10: i promptów);
    # sama zmienna `UPDATE_GOLDEN=1` jest aliasem tej opcji poza CI.
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=_update_golden_from_env(),
        help=(
            "przepisz snapshoty tests/application/snapshots/ i tests/golden/prompts/ "
            "zamiast je porównywać"
        ),
    )


# TEST-11: markery testów, które z definicji potrzebują sieci (prawdziwe API,
# kontener Postgresa na localhost). Reszta biegnie z `--disable-socket`.
NETWORK_MARKERS = ("integration", "containers")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Testom sieciowym oddaje gniazda markerem `enable_socket` z pytest-socket.

    Marker, a nie fixture: pytest-socket blokuje gniazda w swoim
    `pytest_runtest_setup`, PRZED fixture'ami testu, więc fixture funkcyjny
    odblokowałby sieć dopiero po starcie fixture'ów modułowych (`pg_conn`
    stawia kontener, zanim ruszy pierwszy test)."""
    for item in items:
        if any(item.get_closest_marker(marker) for marker in NETWORK_MARKERS):
            item.add_marker(pytest.mark.enable_socket)


@pytest.fixture(autouse=True)
def _closed_http_circuits() -> Iterator[None]:
    """NIEZ-31: obwody HTTP żyją przez cały proces — test, który otworzył obwód
    hosta, nie może odciąć tego hosta kolejnym testom."""
    from src.infrastructure.adapters._http import process_circuits

    process_circuits().reset()
    yield
    process_circuits().reset()


@pytest.fixture
def settings() -> Settings:
    return build_settings()


@pytest.fixture
def mock_external_clients(mocker: Any) -> None:
    """Mockuje wszystkie I/O na poziomie bibliotek zewnętrznych — żaden test
    entry pointa nie strzela do sieci ani na dysk po model XGBoost."""
    mocker.patch("src.infrastructure.adapters.supabase_repo.build_supabase_client")
    mocker.patch("src.infrastructure.llm.openai_client.OpenAI")
