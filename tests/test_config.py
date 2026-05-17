from decimal import Decimal

import pytest

from src.config import Settings

REQUIRED_ENV = {
    "OPENAI_API_KEY": "sk-test",
    "FINNHUB_API_KEY": "fh-test",
    "ALPHA_VANTAGE_API_KEYS": "av-test",
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_KEY": "supa-key",
}


@pytest.fixture
def env(monkeypatch):
    """Czysty zestaw zmiennych — wymagane ustawione, opcjonalne wykasowane,
    by nie wyciekały z shella developera (np. ANTHROPIC_API_KEY='')."""
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    for optional in (
        "VOLATILITY_THRESHOLD",
        "SYMBOLS",
        "LLM_PROVIDER",
        "ML_MODEL_PATH",
        "ANTHROPIC_API_KEY",
        "ALPHA_VANTAGE_API_KEY",  # legacy fallback — chcemy czysty stan
    ):
        monkeypatch.delenv(optional, raising=False)
    return monkeypatch


class TestSettings:
    def test_loads_required_keys_from_env(self, env):
        settings = Settings(_env_file=None)
        assert settings.openai_api_key == "sk-test"
        assert settings.finnhub_api_key == "fh-test"
        assert settings.supabase_url == "https://test.supabase.co"

    def test_raises_when_required_key_missing(self, env):
        env.delenv("FINNHUB_API_KEY")
        with pytest.raises(ValueError, match="finnhub_api_key"):
            Settings(_env_file=None)

    def test_alpha_vantage_is_required(self, env):
        env.delenv("ALPHA_VANTAGE_API_KEYS")
        env.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="alpha_vantage"):
            Settings(_env_file=None)

    def test_parses_alpha_vantage_keys_csv(self, env):
        env.setenv("ALPHA_VANTAGE_API_KEYS", "key1, key2 , key3")
        settings = Settings(_env_file=None)
        assert settings.alpha_vantage_api_keys == ["key1", "key2", "key3"]

    def test_deduplicates_alpha_vantage_keys(self, env):
        env.setenv("ALPHA_VANTAGE_API_KEYS", "k1,k2,k1,k3,k2")
        settings = Settings(_env_file=None)
        assert settings.alpha_vantage_api_keys == ["k1", "k2", "k3"]

    def test_legacy_singular_alpha_vantage_key_still_works(self, env):
        # Backward-compat: user może mieć w .env stary ALPHA_VANTAGE_API_KEY.
        env.delenv("ALPHA_VANTAGE_API_KEYS")
        env.setenv("ALPHA_VANTAGE_API_KEY", "legacy-single-key")
        settings = Settings(_env_file=None)
        assert settings.alpha_vantage_api_keys == ["legacy-single-key"]

    def test_default_volatility_threshold_is_two_percent(self, env):
        settings = Settings(_env_file=None)
        assert settings.volatility_threshold == Decimal("0.02")

    def test_custom_threshold_from_env(self, env):
        env.setenv("VOLATILITY_THRESHOLD", "0.05")
        settings = Settings(_env_file=None)
        assert settings.volatility_threshold == Decimal("0.05")

    def test_default_symbols_list(self, env):
        settings = Settings(_env_file=None)
        assert isinstance(settings.symbols, list)
        assert len(settings.symbols) > 0
        assert "AAPL" in settings.symbols

    def test_parses_symbols_csv_from_env(self, env):
        env.setenv("SYMBOLS", "AAPL,MSFT,GOOGL")
        settings = Settings(_env_file=None)
        assert settings.symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_default_ml_model_path(self, env):
        settings = Settings(_env_file=None)
        assert settings.ml_model_path == "data/models/price_predictor.ubj"

    def test_llm_provider_defaults_to_openai(self, env):
        settings = Settings(_env_file=None)
        assert settings.llm_provider == "openai"

    def test_anthropic_provider_requires_anthropic_key(self, env):
        env.setenv("LLM_PROVIDER", "anthropic")
        # Bez ANTHROPIC_API_KEY config nadal się buduje — walidacja providera odbywa się w DI.
        settings = Settings(_env_file=None)
        assert settings.llm_provider == "anthropic"
        assert settings.anthropic_api_key is None

    def test_symbols_etf_defaults_to_empty(self, env: pytest.MonkeyPatch) -> None:
        settings = Settings(_env_file=None)
        assert settings.symbols_etf == []

    def test_symbols_etf_parses_csv(self, env: pytest.MonkeyPatch) -> None:
        env.setenv("SYMBOLS_ETF", "VOO, CSPX.L , SPY")
        settings = Settings(_env_file=None)
        assert settings.symbols_etf == ["VOO", "CSPX.L", "SPY"]
