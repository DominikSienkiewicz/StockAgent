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
    assert data["symbol_concurrency"] == 4
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
