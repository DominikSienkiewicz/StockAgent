"""Testy źródła config.toml — single source of truth dla configu niewrażliwego.

Te testy ZDEJMUJĄ globalną flagę `STOCKAGENT_DISABLE_TOML` (ustawioną w conftest),
bo celowo weryfikują, że źródło TOML działa i wpina wartości do `Settings`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from src.config import Settings

CONFIG_TOML = Path(__file__).parent.parent / "config.toml"

_REQUIRED_SECRETS = {
    "openai_api_key": "sk-test",
    "finnhub_api_key": "fh",
    "alpha_vantage_api_keys": ["av1"],
    "supabase_url": "https://t.supabase.co",
    "supabase_key": "anon",
}


def test_real_config_toml_parses_and_has_expected_invariants() -> None:
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["llm_provider"] == "anthropic"
    assert data["council_llm_provider"] == "openai"
    assert data["council_llm_model"] == "gpt-5-mini"
    assert len(data["symbols"]) == 43
    assert "SSNLF" in data["symbols_unsupported_price"]
    assert data["crypto_symbols"] == ["BTC", "ETH"]
    # Sekrety NIE mogą wyciec do commitowanego pliku.
    for secret in ("openai_api_key", "anthropic_api_key", "supabase_key",
                   "finnhub_api_key", "digest_to_email"):
        assert secret not in data, f"sekret {secret} w config.toml!"


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
    assert s.symbol_throttle_seconds == 7.5
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
    assert len(s.symbols) == 43, "config.toml nie został znaleziony spoza roota"
    assert s.notifications_enabled is True


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
    assert s.symbol_throttle_seconds == 3.0
