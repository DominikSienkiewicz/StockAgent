"""Testy źródła config.toml — single source of truth dla configu niewrażliwego.

Te testy ZDEJMUJĄ globalną flagę `STOCKAGENT_DISABLE_TOML` (ustawioną w conftest),
bo celowo weryfikują, że źródło TOML działa i wpina wartości do `Settings`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import get_args

import pytest

from src.config import Settings

CONFIG_TOML = Path(__file__).parent.parent / "config.toml"
# TEST-21 (G79): dozwolony zbiór providerów bierzemy z kontraktu `Settings`
# (Literal), nie z bieżącej konfiguracji produkcji.
_LLM_PROVIDERS = frozenset(get_args(Settings.model_fields["llm_provider"].annotation))

_REQUIRED_SECRETS = {
    "openai_api_key": "sk-test",
    "finnhub_api_key": "fh",
    "alpha_vantage_api_keys": ["av1"],
    "supabase_url": "https://t.supabase.co",
    "supabase_key": "anon",
}


def _model_provider(model: str) -> str:
    """Rodzina modelu po nazwie — tylko na potrzeby niezmiennika spójności."""
    return "anthropic" if model.startswith("claude") else "openai"


def _assert_unique_tickers(name: str, values: list[str]) -> None:
    assert all(isinstance(v, str) and v.strip() for v in values), f"pusty ticker w {name}"
    assert len(values) == len(set(values)), f"duplikat w {name}"


def test_real_config_toml_parses_and_has_expected_invariants() -> None:
    # TEST-21 (G79): niezmienniki struktury i spójności, nie dane produkcji —
    # dopisanie tickera albo zmiana modelu nie może czerwienić tego testu.
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["llm_provider"] in _LLM_PROVIDERS
    assert data["council_llm_provider"] in _LLM_PROVIDERS
    # Model rady musi należeć do providera rady — inaczej rada dziedziczy
    # providera z innym modelem i cykl pada (komentarz w config.toml).
    council_model = data["council_llm_model"]
    assert isinstance(council_model, str) and council_model.strip()
    assert _model_provider(council_model) == data["council_llm_provider"]
    assert len(data["symbols"]) > 0
    for name in ("symbols", "symbols_etf", "symbols_unsupported_price", "crypto_symbols"):
        _assert_unique_tickers(name, data[name])
    # Krypto idzie osobnym routingiem (CoinGecko) i jest doklejane do `symbols`
    # w cyklu — symbol w obu listach byłby analizowany dwa razy.
    crypto = set(data["crypto_symbols"])
    assert not crypto & set(data["symbols"]), "krypto także w symbols"
    assert not crypto & set(data["symbols_etf"]), "krypto także w symbols_etf"
    assert isinstance(data["symbol_concurrency"], int)
    assert data["symbol_concurrency"] >= 1
    # Sekrety NIE mogą wyciec do commitowanego pliku.
    for secret in ("openai_api_key", "anthropic_api_key", "supabase_key",
                   "finnhub_api_key", "digest_to_email"):
        assert secret not in data, f"sekret {secret} w config.toml!"


def test_real_config_toml_bounds_reflection_settlement_age() -> None:
    """WIA-04: produkcja rozlicza predykcje w oknie 6 h – 96 h."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["reflection_min_age_hours"] == 6
    assert data["reflection_max_age_hours"] == 96


def test_settings_reads_non_secret_from_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    toml = tmp_path / "config.toml"
    toml.write_text(
        'symbol_throttle_seconds = 7.5\nllm_provider = "anthropic"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("STOCKAGENT_TOML_FILE", str(toml))
    # Sekrety wstrzykujemy jako kwargs (wygrywają nad TOML); env odcięte.
    s = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]
    assert s.symbol_throttle_seconds == pytest.approx(7.5)
    assert s.llm_provider == "anthropic"


def test_config_toml_found_regardless_of_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Config.toml musi być znajdowany niezależnie od CWD — inaczej agent
    odpalony spoza roota repo dostaje DEFAULTY (np. notifications_enabled=False,
    symbols=[AAPL,MSFT,NVDA]) i po cichu nie wysyła maila / gubi portfolio."""
    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    monkeypatch.delenv("STOCKAGENT_TOML_FILE", raising=False)
    monkeypatch.chdir(tmp_path)  # CWD != root repo
    s = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert s.symbols == data["symbols"], "config.toml nie został znaleziony spoza roota"
    assert s.notifications_enabled is data["notifications_enabled"]


def test_env_overrides_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    toml = tmp_path / "config.toml"
    toml.write_text("symbol_throttle_seconds = 7.5\n", encoding="utf-8")
    monkeypatch.setenv("STOCKAGENT_TOML_FILE", str(toml))
    monkeypatch.setenv("SYMBOL_THROTTLE_SECONDS", "3.0")
    s = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]
    # env > toml
    assert s.symbol_throttle_seconds == pytest.approx(3.0)


def _comment_above(lines: list[str], index: int) -> str:
    """Ostatnia linia bloku komentarza nad kluczem `lines[index]`.

    Przeskakuje sąsiednie przypisania tej samej grupy (np. trzy flagi pod
    jednym komentarzem); pusta linia kończy grupę.
    """
    for line in reversed(lines[:index]):
        stripped = line.strip()
        if not stripped:
            return ""
        if stripped.startswith("#"):
            return stripped
    return ""


def test_no_flag_set_to_true_sits_under_an_off_comment() -> None:
    # „Off." nad `= true` to config, który kłamie o sobie — a ten plik jest
    # deklarowanym single source of truth.
    lines = CONFIG_TOML.read_text(encoding="utf-8").splitlines()
    contradicted = [
        line.split("=", 1)[0].strip()
        for index, line in enumerate(lines)
        if line.replace(" ", "").endswith("=true")
        and _comment_above(lines, index).endswith("Off.")
    ]
    assert not contradicted, f"flagi = true pod komentarzem „Off.”: {contradicted}"


def test_real_config_toml_shows_times_in_warsaw() -> None:
    """MAIL-14: temat, nagłówek i push w czasie lokalnym odbiorcy."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["display_timezone"] == "Europe/Warsaw"


def test_real_config_toml_tags_predictions_with_the_regime() -> None:
    """ML-38: produkcja nie nadpisuje `regime_tagging_enabled`, więc tag reżimu
    dziedziczy włączony `regime_aware_enabled` (migracja 017 jest na produkcji)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["regime_aware_enabled"] is True
    assert "regime_tagging_enabled" not in data


def test_real_config_toml_ships_regime_hysteresis_off_until_migration() -> None:
    """ALR-03: bez migracji 033 odczyt/zapis `regime_history` by padał."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["regime_hysteresis_enabled"] is False
    assert data["regime_exit_confirmations"] == 2


def test_real_config_toml_sectors_hold_only_watchlist_stocks() -> None:
    """PORT-05: sektor to grupa AKCJI z watchlisty — ETF i krypto nie mają
    sektora, a symbol spoza `symbols` nie dostaje ceny w cyklu. Symbol w dwóch
    sektorach dawałby niejednoznaczny klaster w sizingu."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    sectors: dict[str, list[str]] = data["sectors"]
    assert sectors
    members = [symbol for symbols in sectors.values() for symbol in symbols]
    assert all(symbols for symbols in sectors.values()), "pusty sektor"
    assert len(members) == len(set(members)), "symbol w dwóch sektorach"
    assert set(members) <= set(data["symbols"])
    assert not set(members) & set(data["symbols_etf"])
    assert not set(members) & set(data["crypto_symbols"])


def test_real_config_toml_keeps_contagion_off() -> None:
    """PORT-05: `peer_groups` z sektorów zasila też contagion (werdykty peerów
    do promptu). Ten zostaje wyłączony — sektory mają działać tylko w sizingu
    i ostrzeżeniu o koncentracji."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["contagion_enabled"] is False


def test_settings_derive_peer_groups_from_config_sectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    monkeypatch.delenv("STOCKAGENT_TOML_FILE", raising=False)
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))

    s = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]

    assert s.sectors == data["sectors"]
    assert s.peer_groups == data["sectors"]
    assert s.contagion_enabled is False


def test_real_config_toml_peers_are_keyed_by_its_sectors() -> None:
    """PORT-05: peerzy sekcji 💡 Warte uwagi są kluczowani nazwą sektora z
    `sectors` — zmiana nazwy sektora bez peerów nie może ich wyłączyć po cichu."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    peers: dict[str, list[str]] = data["sector_peers"]
    assert peers
    assert set(peers) <= set(data["sectors"])
    assert all(candidates for candidates in peers.values()), "sektor bez peerów"
