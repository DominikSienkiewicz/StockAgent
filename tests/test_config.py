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

# Te same wartości jako kwargi `Settings` (bez czytania env).
REQUIRED_ENV_KWARGS = {
    "openai_api_key": "sk-test",
    "finnhub_api_key": "fh-test",
    "alpha_vantage_api_keys": ["av-test"],
    "supabase_url": "https://test.supabase.co",
    "supabase_key": "supa-key",
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
        negative = Decimal("-0.01")
        with pytest.raises(ValueError, match="volatility_threshold"):
            Settings(_env_file=None, volatility_threshold=negative)

    def test_negative_council_volatility_threshold_rejected(self, env):
        negative = Decimal("-0.01")
        with pytest.raises(ValueError, match="council_volatility_threshold"):
            Settings(_env_file=None, council_volatility_threshold=negative)

    def test_negative_crypto_volatility_threshold_rejected(self, env):
        negative = Decimal("-0.01")
        with pytest.raises(ValueError, match="crypto_volatility_threshold"):
            Settings(_env_file=None, crypto_volatility_threshold=negative)

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

    def test_reflection_max_age_hours_default_bounds_settlement_age(
        self, env: pytest.MonkeyPatch
    ) -> None:
        # WIA-04: 96 h obejmuje weekend (piątek → poniedziałek ≈ 72 h) z zapasem,
        # a predykcja sprzed tygodni nie jest już rozliczana bieżącą ceną.
        settings = Settings(_env_file=None)
        assert settings.reflection_max_age_hours == 96

    def test_reflection_max_age_hours_reads_env(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("REFLECTION_MAX_AGE_HOURS", "120")
        settings = Settings(_env_file=None)
        assert settings.reflection_max_age_hours == 120

    def test_reflection_max_age_zero_disables_the_bound(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("REFLECTION_MAX_AGE_HOURS", "0")
        settings = Settings(_env_file=None)
        assert settings.reflection_max_age_hours == 0

    def test_reflection_max_age_must_exceed_min_age(
        self, env: pytest.MonkeyPatch
    ) -> None:
        # Okno (min, max] puste → reflect nigdy niczego by nie rozliczył.
        env.setenv("REFLECTION_MIN_AGE_HOURS", "6")
        env.setenv("REFLECTION_MAX_AGE_HOURS", "6")
        with pytest.raises(ValueError, match="reflection_max_age_hours"):
            Settings(_env_file=None)

    def test_reflection_max_age_negative_rejected(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("REFLECTION_MAX_AGE_HOURS", "-1")
        with pytest.raises(ValueError, match="reflection_max_age_hours"):
            Settings(_env_file=None)


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


class TestSymbolConcurrency:
    def _base(self, **overrides):
        base = {
            "_env_file": None,
            "openai_api_key": "sk-test",
            "finnhub_api_key": "fh",
            "alpha_vantage_api_keys": ["av1"],
            "supabase_url": "https://t.supabase.co",
            "supabase_key": "anon",
            "symbols": ["AAPL"],
            "volatility_threshold": Decimal("0.02"),
        }
        base.update(overrides)
        return base

    def test_defaults_to_one(self):
        s = Settings(**self._base())
        assert s.symbol_concurrency == 1

    def test_accepts_positive(self):
        s = Settings(**self._base(symbol_concurrency=4))
        assert s.symbol_concurrency == 4

    def test_rejects_below_one(self):
        kwargs = self._base(symbol_concurrency=0)
        with pytest.raises(ValueError, match="symbol_concurrency must be >= 1"):
            Settings(**kwargs)


class TestDisplayTimezone:
    """MAIL-14: strefa czasu tematu, nagłówka maila i pushu."""

    def test_defaults_to_warsaw(self, env):
        env.delenv("DISPLAY_TIMEZONE", raising=False)

        assert Settings(_env_file=None).display_timezone == "Europe/Warsaw"

    def test_accepts_any_iana_zone(self, env):
        env.setenv("DISPLAY_TIMEZONE", "America/New_York")

        assert Settings(_env_file=None).display_timezone == "America/New_York"

    @pytest.mark.parametrize("bad", ["Europe/Warszawa", "CEST+2", ""])
    def test_rejects_an_unknown_zone(self, env, bad):
        # Literówka ma zatrzymać start, a nie wysłać godzinę UTC podpisaną jako lokalna.
        with pytest.raises(ValueError, match="display_timezone"):
            Settings(_env_file=None, display_timezone=bad)


class TestConfigHash:
    """WIA-20: `config_hash` identyfikuje konfigurację przepisu predykcji.

    Liczony WYŁĄCZNIE z niesekretnych pól: sekret (klucz API, poświadczenia
    bazy, adres e-mail, adres kanału push) nie może zmienić hasha ani trafić
    do kwitu, który żyje w bazie."""

    _SECRET_VALUES = {
        "openai_api_key": "sk-SECRET-openai",
        "anthropic_api_key": "sk-ant-SECRET",
        "fred_api_key": "fred-SECRET",
        "finnhub_api_key": "fh-SECRET",
        "alpha_vantage_api_keys": ["av-SECRET-1", "av-SECRET-2"],
        "supabase_url": "https://SECRET.supabase.co",
        "supabase_key": "supa-SECRET",
        "resend_api_key": "re_SECRET",
        "digest_from_email": "secret-sender@example.com",
        "digest_to_email": "secret-owner@example.com",
        "telegram_bot_token": "123:SECRET",
        "telegram_chat_id": "SECRET-chat",
        "slack_webhook_url": "https://hooks.slack.com/services/SECRET",
        "edgar_user_agent": "StockAgent/1.0 (secret-contact@example.com)",
    }

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_hash_is_a_stable_sha256(self, env):
        assert self._settings().config_hash() == self._settings().config_hash()
        assert len(self._settings().config_hash()) == 64

    def test_threshold_change_changes_the_hash(self, env):
        assert (
            self._settings(volatility_threshold=Decimal("0.02")).config_hash()
            != self._settings(volatility_threshold=Decimal("0.03")).config_hash()
        )

    def test_no_secret_moves_the_hash(self, env):
        baseline = self._settings().config_hash()

        for field, value in self._SECRET_VALUES.items():
            assert self._settings(**{field: value}).config_hash() == baseline, field

    def test_every_secret_looking_field_is_excluded(self, env):
        # Nowe pole z kluczem/tokenem/adresem w nazwie musi trafić na listę
        # wykluczeń — inaczej po cichu wejdzie do hasha (i do kwitu w bazie).
        import re

        from src.config import SECRET_SETTINGS_FIELDS

        secret_like = re.compile(
            r"key|token|secret|password|email|webhook|chat_id|user_agent|supabase"
        )
        named_like_secrets = {name for name in Settings.model_fields if secret_like.search(name)}

        assert named_like_secrets <= SECRET_SETTINGS_FIELDS
        assert set(self._SECRET_VALUES) <= SECRET_SETTINGS_FIELDS
        assert set(Settings.model_fields) >= SECRET_SETTINGS_FIELDS


class TestConfigHashScope:
    """WIA-20 (decyzja 2026-10-19): `config_hash` obejmuje tylko pola przepisu.

    Każde pole `Settings` jest sklasyfikowane DOKŁADNIE raz: sekret, poza
    przepisem (prezentacja, dostarczanie, flagi samego zapisu, inne joby) albo
    przepis. Nowe pole bez klasyfikacji = czerwony test, zanim po cichu wejdzie
    do hasha albo z niego wypadnie."""

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_every_field_is_classified_exactly_once(self):
        from src.config import (
            NON_RECIPE_SETTINGS_FIELDS,
            RECIPE_SETTINGS_FIELDS,
            SECRET_SETTINGS_FIELDS,
        )

        groups = (SECRET_SETTINGS_FIELDS, NON_RECIPE_SETTINGS_FIELDS, RECIPE_SETTINGS_FIELDS)
        classified = [name for group in groups for name in group]

        assert len(classified) == len(set(classified)), "pole w dwóch grupach naraz"
        assert set(classified) == set(Settings.model_fields)

    def test_presentation_and_write_only_fields_do_not_move_the_hash(self, env):
        baseline = self._settings().config_hash()

        for field, value in {
            "display_timezone": "America/New_York",
            "notifications_enabled": True,
            "web_digest_path": "elsewhere/index.html",
            "receipts_enabled": True,
            "recipe_version_enabled": True,
            "track_record_days": 90,
        }.items():
            assert self._settings(**{field: value}).config_hash() == baseline, field

    def test_flipping_any_non_recipe_flag_keeps_the_hash(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        baseline = self._settings().config_hash()
        flags = [
            name for name in NON_RECIPE_SETTINGS_FIELDS
            if Settings.model_fields[name].annotation is bool
        ]

        assert flags
        for flag in flags:
            flipped = not Settings.model_fields[flag].default
            assert self._settings(**{flag: flipped}).config_hash() == baseline, flag

    def test_flipping_any_recipe_flag_moves_the_hash(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        baseline = self._settings().config_hash()
        flags = [
            name for name in RECIPE_SETTINGS_FIELDS
            if Settings.model_fields[name].annotation is bool
        ]

        assert flags
        for flag in flags:
            flipped = not Settings.model_fields[flag].default
            assert self._settings(**{flag: flipped}).config_hash() != baseline, flag

    def test_prediction_inputs_move_the_hash(self, env):
        baseline = self._settings().config_hash()

        for field, value in {
            "tool_use_max_rounds": 5,
            "symbols": ["AAPL", "PLTR"],
            "llm_provider": "anthropic",
            "council_llm_model": "gpt-5-mini",
            "rag_outcome_weight": 0.3,
        }.items():
            assert self._settings(**{field: value}).config_hash() != baseline, field


class TestFeatureSnapshotsFlag:
    """ML-14: zapis wektora cech w każdym cyklu (migracja 031) — flaga samego
    zapisu, więc domyślnie off i poza przepisem (`config_hash` się nie rusza)."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).feature_snapshots_enabled is False

    def test_is_a_write_only_flag_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "feature_snapshots_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, feature_snapshots_enabled=True
        ).config_hash()
        assert flipped == baseline
