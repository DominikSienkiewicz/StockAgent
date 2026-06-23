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

    def test_negative_volatility_threshold_rejected(self, env):
        # Ujemny próg sprawia, że abs(delta) >= threshold ZAWSZE prawdziwe →
        # płatne porty (LLM/AV/embeddingi) odpalają co cykl, bez błędu startu.
        with pytest.raises(ValueError, match="volatility_threshold"):
            Settings(_env_file=None, volatility_threshold=Decimal("-0.01"))

    def test_negative_council_volatility_threshold_rejected(self, env):
        with pytest.raises(ValueError, match="council_volatility_threshold"):
            Settings(
                _env_file=None, council_volatility_threshold=Decimal("-0.01")
            )

    def test_negative_crypto_volatility_threshold_rejected(self, env):
        with pytest.raises(ValueError, match="crypto_volatility_threshold"):
            Settings(
                _env_file=None, crypto_volatility_threshold=Decimal("-0.01")
            )

    def test_council_volatility_threshold_zero_is_accepted(self, env):
        # 0.0 to udokumentowany "always run" disable switch — nie footgun.
        settings = Settings(
            _env_file=None, council_volatility_threshold=Decimal("0")
        )
        assert settings.council_volatility_threshold == Decimal("0")

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

    def test_risk_symbols_default_empty(self, env: pytest.MonkeyPatch) -> None:
        settings = Settings(_env_file=None)
        assert settings.risk_symbols == []
        assert settings.risk_symbol_types == {}

    def test_risk_symbols_parses_csv(self, env: pytest.MonkeyPatch) -> None:
        env.setenv("RISK_SYMBOLS", "EPOL, SH , VIXY")
        settings = Settings(_env_file=None)
        assert settings.risk_symbols == ["EPOL", "SH", "VIXY"]

    def test_risk_symbol_types_parses_mapping(
        self, env: pytest.MonkeyPatch
    ) -> None:
        from src.domain.macro_risk import MacroRiskInstrumentType

        env.setenv(
            "RISK_SYMBOL_TYPES",
            "EPOL:SOVEREIGN_PROXY,SH:INVERSE_EQUITY,VIXY:VOLATILITY",
        )
        settings = Settings(_env_file=None)
        assert settings.risk_symbol_types == {
            "EPOL": MacroRiskInstrumentType.SOVEREIGN_PROXY,
            "SH": MacroRiskInstrumentType.INVERSE_EQUITY,
            "VIXY": MacroRiskInstrumentType.VOLATILITY,
        }

    def test_risk_symbol_types_rejects_unknown_enum(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("RISK_SYMBOL_TYPES", "EPOL:NOT_A_REAL_TYPE")
        with pytest.raises(ValueError, match="NOT_A_REAL_TYPE"):
            Settings(_env_file=None)

    def test_nbp_enabled_defaults_false(self, env: pytest.MonkeyPatch) -> None:
        settings = Settings(_env_file=None)
        assert settings.nbp_enabled is False

    def test_nbp_enabled_reads_env(self, env: pytest.MonkeyPatch) -> None:
        env.setenv("NBP_ENABLED", "true")
        settings = Settings(_env_file=None)
        assert settings.nbp_enabled is True


class TestResilienceSettings:
    def test_symbols_unsupported_price_defaults_empty(
        self, env: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(_env_file=None)
        assert settings.symbols_unsupported_price == []

    def test_symbols_unsupported_price_parses_csv(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("SYMBOLS_UNSUPPORTED_PRICE", "CSPX.L, XDWD.DE ,IUSN.DE")
        settings = Settings(_env_file=None)
        assert settings.symbols_unsupported_price == [
            "CSPX.L", "XDWD.DE", "IUSN.DE",
        ]

    def test_symbol_throttle_defaults_to_zero(
        self, env: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(_env_file=None)
        assert settings.symbol_throttle_seconds == pytest.approx(0.0)

    def test_symbol_throttle_parses_float(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("SYMBOL_THROTTLE_SECONDS", "2.5")
        settings = Settings(_env_file=None)
        assert settings.symbol_throttle_seconds == pytest.approx(2.5)

    def test_symbol_throttle_negative_rejected(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("SYMBOL_THROTTLE_SECONDS", "-1")
        with pytest.raises(ValueError, match="symbol_throttle_seconds"):
            Settings(_env_file=None)

    def test_reflection_min_age_hours_default_is_safe_nonzero(
        self, env: pytest.MonkeyPatch
    ) -> None:
        # Shipped code default MUSI być bezpieczny (≠0). Default 0 wyłączał
        # bramkę wieku w reflect_node → przedwczesna ocena świeżej predykcji
        # zatruwała accuracy_score (XGBoost) i hit-rate raportu. conftest
        # ustawia STOCKAGENT_DISABLE_TOML=1, więc to czysty default z kodu,
        # nie wartość z config.toml.
        settings = Settings(_env_file=None)
        assert settings.reflection_min_age_hours == 6

    def test_reflection_min_age_hours_reads_env(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("REFLECTION_MIN_AGE_HOURS", "12")
        settings = Settings(_env_file=None)
        assert settings.reflection_min_age_hours == 12


class TestCryptoSettings:
    def test_crypto_symbols_defaults_empty(
        self, env: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(_env_file=None)
        assert settings.crypto_symbols == []

    def test_crypto_symbols_parses_csv(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("CRYPTO_SYMBOLS", "BTC, ETH ,SOL")
        settings = Settings(_env_file=None)
        assert settings.crypto_symbols == ["BTC", "ETH", "SOL"]

    def test_crypto_volatility_threshold_defaults_to_five_percent(
        self, env: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(_env_file=None)
        assert settings.crypto_volatility_threshold == Decimal("0.05")

    def test_crypto_volatility_threshold_reads_env(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("CRYPTO_VOLATILITY_THRESHOLD", "0.08")
        settings = Settings(_env_file=None)
        assert settings.crypto_volatility_threshold == Decimal("0.08")
