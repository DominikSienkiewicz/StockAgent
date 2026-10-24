import base64
import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

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
        # ARCH-05: brak klucza zatrzymuje fabrykę, która go woła, nie konstrukcję.
        env.delenv("FINNHUB_API_KEY")
        with pytest.raises(ValueError, match="FINNHUB_API_KEY"):
            Settings(_env_file=None).require_finnhub_api_key()

    def test_alpha_vantage_is_required(self, env):
        env.delenv("ALPHA_VANTAGE_API_KEYS")
        env.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="alpha_vantage"):
            Settings(_env_file=None).require_alpha_vantage_api_keys()

    def test_settings_constructs_without_paid_keys(self, env):
        # ARCH-05: darmowe joby (watch, deadman) nie potrzebują atrap płatnych kluczy.
        for key in ("OPENAI_API_KEY", "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEYS"):
            env.delenv(key)

        settings = Settings(_env_file=None)

        assert settings.openai_api_key is None
        assert settings.finnhub_api_key is None
        assert settings.alpha_vantage_api_keys == []

    def test_require_openai_api_key_names_the_variable(self, env):
        env.delenv("OPENAI_API_KEY")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            Settings(_env_file=None).require_openai_api_key()

    def test_require_helpers_return_the_configured_keys(self, env):
        settings = Settings(_env_file=None)

        assert settings.require_openai_api_key() == "sk-test"
        assert settings.require_finnhub_api_key() == "fh-test"
        assert settings.require_alpha_vantage_api_keys() == ["av-test"]

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

    def test_crypto_tool_use_threshold_defaults_to_the_shared_one(
        self, env: pytest.MonkeyPatch
    ) -> None:
        # FIN-08: brak wartości = krypto bierze `tool_use_volatility_threshold`.
        settings = Settings(_env_file=None)
        assert settings.crypto_tool_use_volatility_threshold is None

    def test_crypto_tool_use_threshold_reads_env(
        self, env: pytest.MonkeyPatch
    ) -> None:
        env.setenv("CRYPTO_TOOL_USE_VOLATILITY_THRESHOLD", "0.10")
        settings = Settings(_env_file=None)
        assert settings.crypto_tool_use_volatility_threshold == Decimal("0.10")

    def test_zero_crypto_tool_use_threshold_rejected(
        self, env: pytest.MonkeyPatch
    ) -> None:
        # FIN-07: próg 0 bramki płatnej ścieżki przepuszcza każdy ruch.
        with pytest.raises(
            ValueError, match="crypto_tool_use_volatility_threshold must be positive"
        ):
            Settings(_env_file=None, crypto_tool_use_volatility_threshold=Decimal("0"))

    def test_crypto_tool_use_threshold_is_part_of_the_recipe(
        self, env: pytest.MonkeyPatch
    ) -> None:
        # FIN-08: pole steruje płatną pętlą tool-use, więc należy do przepisu.
        from src.config import RECIPE_SETTINGS_FIELDS

        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        changed = Settings(
            _env_file=None,
            **REQUIRED_ENV_KWARGS,
            crypto_tool_use_volatility_threshold=Decimal("0.10"),
        ).config_hash()

        assert "crypto_tool_use_volatility_threshold" in RECIPE_SETTINGS_FIELDS
        assert changed != baseline


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
            "public_scorecard_path": "elsewhere/index.html",
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


class TestMarketFeaturesFlag:
    """ML-31: cechy rynkowe w `feature_snapshots` i wersja zestawu cech 4 —
    domyślnie off (zestaw i wersja 3 bez zmian) i poza przepisem: prompt, wektor
    modelu i progi się nie zmieniają, a wpływ na wagi łapie `model_version`."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).market_features_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "market_features_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, market_features_enabled=True
        ).config_hash()
        assert flipped == baseline


class TestSnapshotTrainingFlag:
    """ML-52 (4): model wspólny z wektorów `feature_snapshots` zamiast widoku SQL.
    Zmienia to, co Fast Loop serwuje (wagi), więc domyślnie off; wpływ na prognozę
    idzie wyłącznie przez wagi, które łapie `model_version` — poza przepisem."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).snapshot_training_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "snapshot_training_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestSigmaTargetFlag:
    """ML-11: target modelu wspólnego w jednostkach σ, a serving mnoży prognozę
    przez σ — zmiana tego, co Fast Loop serwuje, więc off i w przepisie."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).ml_sigma_target_enabled is False

    def test_is_part_of_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        assert "ml_sigma_target_enabled" in RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, ml_sigma_target_enabled=True
        ).config_hash()
        assert flipped != baseline


class TestIntervalCoverageFlag:
    """ML-34: zapis kolumn przedziału (migracja 032) i pokrycie w Track Record.
    Przedział to prezentacja niepewności, nie prognoza — flaga jest poza
    przepisem, a domyślnie off, bo przed migracją zapis dałby PGRST204."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).interval_coverage_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "interval_coverage_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, interval_coverage_enabled=True
        ).config_hash()
        assert flipped == baseline


class TestRegimeTaggingFlag:
    """ML-38: tag reżimu przy każdej predykcji (kolumna z migracji 017). Brak
    wartości = jak `regime_aware_enabled` (rozstrzyga DI); flaga samego zapisu,
    więc poza przepisem (`config_hash` się nie rusza)."""

    def test_defaults_to_unset(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).regime_tagging_enabled is None

    def test_is_a_write_only_flag_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "regime_tagging_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        for value in (True, False):
            flipped = Settings(
                _env_file=None, **REQUIRED_ENV_KWARGS, regime_tagging_enabled=value
            ).config_hash()
            assert flipped == baseline, value


class TestRegimeHysteresisSettings:
    """ALR-03: histereza reżimu (tabela `regime_history`, migracja 033). Flaga
    domyślnie off — przed migracją tabeli nie ma. Flaga i liczba potwierdzeń
    zmieniają efektywny próg bramki, więc należą do przepisu (`config_hash`)."""

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_defaults_off_with_two_exit_confirmations(self, env):
        settings = self._settings()

        assert settings.regime_hysteresis_enabled is False
        assert settings.regime_exit_confirmations == 2

    def test_exit_confirmations_below_one_rejected(self, env):
        with pytest.raises(ValueError, match="regime_exit_confirmations"):
            self._settings(regime_exit_confirmations=0)

    def test_both_fields_belong_to_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        assert {"regime_hysteresis_enabled", "regime_exit_confirmations"} <= (
            RECIPE_SETTINGS_FIELDS
        )
        baseline = self._settings().config_hash()
        assert self._settings(regime_hysteresis_enabled=True).config_hash() != baseline
        assert self._settings(regime_exit_confirmations=3).config_hash() != baseline


class TestMacroSeriesFlag:
    """DAT-22: zmienne stanu z FRED — domyślnie off; głosy VIX i spreadu HY mogą
    zacieśnić próg bramki, więc flaga należy do przepisu."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).macro_series_enabled is False

    def test_belongs_to_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        assert "macro_series_enabled" in RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, macro_series_enabled=True
        ).config_hash()
        assert flipped != baseline


class TestTrendProbabilitiesFlag:
    """ML-29: rozkład trzech klas w prompcie i kolumna z migracji 051 — domyślnie
    off, a że zmienia prompt i rozkład pewności, należy do przepisu."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).trend_probabilities_enabled is False

    def test_belongs_to_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        assert "trend_probabilities_enabled" in RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, trend_probabilities_enabled=True
        ).config_hash()
        assert flipped != baseline


class TestGateDeltaFlag:
    """EXP-01: zapis delty bramki (migracja 034) — flaga samego zapisu, więc
    domyślnie off i poza przepisem (`config_hash` się nie rusza)."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).gate_delta_enabled is False

    def test_is_a_write_only_flag_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "gate_delta_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, gate_delta_enabled=True
        ).config_hash()
        assert flipped == baseline


class TestSectorsDrivePeerGroups:
    """PORT-05: `sectors` (sektor → symbole akcji) to jedno źródło grup.

    `peer_groups` (klastry w sizingu, ostrzeżenie o koncentracji, contagion)
    wyprowadzamy z sektorów — bez ETF-ów i krypto, które sektora akcji nie mają."""

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_defaults_are_empty(self, env):
        settings = self._settings()

        assert settings.sectors == {}
        assert settings.peer_groups == {}

    def test_peer_groups_follow_sectors(self, env):
        settings = self._settings(sectors={"Półprzewodniki/AI": ["NVDA", "AMD"]})

        assert settings.peer_groups == {"Półprzewodniki/AI": ["NVDA", "AMD"]}

    def test_derived_peer_groups_skip_etfs_and_crypto(self, env):
        settings = self._settings(
            sectors={"Mix": ["NVDA", "VOO", "BTC"], "Tylko ETF": ["VT"]},
            symbols_etf=["VOO", "VT"],
            crypto_symbols=["BTC"],
        )

        assert settings.peer_groups == {"Mix": ["NVDA"]}

    def test_explicit_peer_groups_win_over_sectors(self, env):
        settings = self._settings(
            sectors={"Półprzewodniki/AI": ["NVDA", "AMD"]},
            peer_groups={"Własny": ["NVDA", "TSM"]},
        )

        assert settings.peer_groups == {"Własny": ["NVDA", "TSM"]}

    def test_sectors_belong_to_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        assert "sectors" in RECIPE_SETTINGS_FIELDS
        baseline = self._settings().config_hash()
        assert self._settings(sectors={"A": ["NVDA"]}).config_hash() != baseline


class TestSectorPeers:
    """PORT-05: peerzy 💡 Warte uwagi kluczowani sektorami z `sectors`."""

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_defaults_are_empty(self, env):
        assert self._settings().sector_peers == {}

    def test_peers_of_a_known_sector_are_accepted(self, env):
        settings = self._settings(
            sectors={"Big Tech": ["AAPL"]}, sector_peers={"Big Tech": ["NFLX"]}
        )

        assert settings.sector_peers == {"Big Tech": ["NFLX"]}

    def test_renamed_sector_without_its_peers_stops_the_start(self, env):
        # Sektor przemianowany w `sectors`, peerzy zostali pod starą nazwą.
        with pytest.raises(ValueError, match="Big Tech"):
            self._settings(
                sectors={"Wielka technologia": ["AAPL"]},
                sector_peers={"Big Tech": ["NFLX"]},
            )

    def test_peers_are_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "sector_peers" in NON_RECIPE_SETTINGS_FIELDS
        sectors = {"Big Tech": ["AAPL"]}
        assert (
            self._settings(sectors=sectors, sector_peers={"Big Tech": ["NFLX"]}).config_hash()
            == self._settings(sectors=sectors).config_hash()
        )


class TestFxRatesCacheFlag:
    """PORT-11: cache kursów NBP z dnia w `fx_rates` (migracja 036) — przed migracją
    tabeli nie ma, więc domyślnie off; kursy nie zmieniają prognozy."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).fx_rates_cache_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "fx_rates_cache_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, fx_rates_cache_enabled=True
        ).config_hash()
        assert flipped == baseline


class TestPaperCostSettings:
    """WIA-50: koszt jednostronny paper-PnL w bps (akcje/ETF i krypto). To
    parametry oceny i prezentacji, nie prognozy — poza przepisem."""

    def test_defaults_match_the_decision(self, env):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.paper_cost_bps == 10
        assert settings.crypto_paper_cost_bps == 25

    @pytest.mark.parametrize("field", ["paper_cost_bps", "crypto_paper_cost_bps"])
    def test_negative_cost_is_rejected(self, env, field):
        with pytest.raises(ValueError, match=field):
            Settings(_env_file=None, **REQUIRED_ENV_KWARGS, **{field: -1})

    @pytest.mark.parametrize("field", ["paper_cost_bps", "crypto_paper_cost_bps"])
    def test_zero_cost_is_allowed(self, env, field):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS, **{field: 0})

        assert getattr(settings, field) == 0

    def test_costs_are_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert {"paper_cost_bps", "crypto_paper_cost_bps"} <= NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        changed = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, paper_cost_bps=30, crypto_paper_cost_bps=60
        ).config_hash()
        assert changed == baseline


class TestExperimentSettings:
    """EXP-05: definicje eksperymentów (`[experiments]` w config.toml) zmieniają
    prompt, więc są w przepisie; flaga samego zapisu kolumny `experiment_arms`
    (migracja 037) jest poza nim i domyślnie off."""

    _DEFINITION = {
        "hypothesis": "Rada bez werdyktu LLM jest trafniejsza.",
        "arms": ["sighted", "blind"],
        "start": "2026-11-02",
        "end": "2027-01-29",
        "primary_metric": "council_hit_rate",
        "enabled": True,
    }

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_defaults_to_no_experiments_and_the_column_off(self, env):
        settings = self._settings()

        assert settings.experiments == {}
        assert settings.experiment_definitions() == ()
        assert settings.experiment_arms_enabled is False

    def test_definitions_become_domain_experiments(self, env):
        from datetime import date

        from src.domain.experiment import ExperimentMetric, PromptPart

        settings = self._settings(experiments={"EXP-02": self._DEFINITION})

        (experiment,) = settings.experiment_definitions()
        assert experiment.id == "EXP-02"
        assert experiment.arms == ("sighted", "blind")
        assert experiment.start == date(2026, 11, 2)
        assert experiment.primary_metric is ExperimentMetric.COUNCIL_HIT_RATE
        assert experiment.part is PromptPart.COUNCIL
        assert experiment.enabled is True

    def test_unknown_treatment_arm_stops_the_start(self, env):
        broken = {**self._DEFINITION, "arms": ["sighted", "blnd"]}

        with pytest.raises(ValueError, match="unknown treatment arms"):
            self._settings(experiments={"EXP-02": broken})

    def test_unknown_metric_stops_the_start(self, env):
        broken = {**self._DEFINITION, "primary_metric": "sharpe"}

        with pytest.raises(ValueError):
            self._settings(experiments={"EXP-02": broken})

    def test_unknown_key_stops_the_start(self, env):
        with pytest.raises(ValueError):
            self._settings(experiments={"EXP-02": {**self._DEFINITION, "enabeld": True}})

    def test_two_enabled_experiments_on_one_prompt_part_stop_the_start(self, env):
        with pytest.raises(ValueError, match="overlapping"):
            self._settings(
                experiments={"EXP-02": self._DEFINITION, "EXP-09": self._DEFINITION}
            )

    def test_definitions_are_in_the_recipe_and_the_column_flag_is_not(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS, RECIPE_SETTINGS_FIELDS

        assert "experiments" in RECIPE_SETTINGS_FIELDS
        assert "experiment_arms_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = self._settings().config_hash()
        assert self._settings(experiments={"EXP-02": self._DEFINITION}).config_hash() != baseline
        assert self._settings(experiment_arms_enabled=True).config_hash() == baseline


class TestSecretValues:
    """BEZ-02: wartości sekretów, które formatter logów redaguje dosłownie."""

    _VALUES = {
        "openai_api_key": "sk-openai-0001",
        "anthropic_api_key": "sk-ant-0002",
        "fred_api_key": "fred-key-0003",
        "finnhub_api_key": "finnhub-0004",
        "alpha_vantage_api_keys": ["av-key-0005", "av-key-0006"],
        "supabase_url": "https://ref0007.supabase.co",
        "supabase_key": "supa-key-0008",
        "resend_api_key": "re_key_0009",
        "digest_from_email": "sender-0010@example.com",
        "digest_to_email": "owner-0011@example.com",
        "telegram_bot_token": "123:bot-0012",
        "telegram_chat_id": "chat-0013",
        "slack_webhook_url": "https://hooks.slack.com/services/0014",
        "edgar_user_agent": "StockAgent/1.0 (contact-0015@example.com)",
        "heartbeat_url": "https://hc-ping.com/check-0016",
        "backup_age_recipient": "age1recipient-0017",
    }

    def test_fixture_covers_every_secret_field(self):
        from src.config import SECRET_SETTINGS_FIELDS

        assert set(self._VALUES) == SECRET_SETTINGS_FIELDS

    def test_lists_every_secret_with_each_alpha_vantage_key_separately(self, env):
        settings = Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **self._VALUES})

        values = settings.secret_values()

        expected = {
            item
            for value in self._VALUES.values()
            for item in (value if isinstance(value, list) else [value])
        }
        assert set(values) == expected
        assert "av-key-0005,av-key-0006" not in values

    def test_skips_unset_optional_secrets(self, env):
        values = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).secret_values()

        assert all(values)
        assert {"sk-test", "av-test", "supa-key"} <= set(values)


class TestDbWritesFlag:
    """BEZ-08: zapis do wspólnej bazy tylko jawnie. Lokalny run i GHA mają ten
    sam `config.toml` i tę samą bazę, więc przełącznik żyje w środowisku
    (`DB_WRITES_ENABLED`): domyślnie false, workflowy ustawiają "true"."""

    def test_defaults_off(self, env):
        env.delenv("DB_WRITES_ENABLED", raising=False)

        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).db_writes_enabled is False

    def test_reads_the_environment_variable(self, env):
        env.setenv("DB_WRITES_ENABLED", "true")

        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).db_writes_enabled is True

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "db_writes_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, db_writes_enabled=True
        ).config_hash()
        assert flipped == baseline


def _unsigned_jwt(claims: dict[str, object]) -> str:
    """JWT w formacie kluczy legacy Supabase; podpis jest atrapą, bo walidator go
    nie sprawdza (rolę czyta z payloadu)."""

    def segment(data: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{segment({'alg': 'HS256', 'typ': 'JWT'})}.{segment(claims)}.signature"


class TestSupabaseKeyRole:
    """BEZ-10: po migracji 038 (RLS) klucz publiczny Supabase nie odczyta ani nie
    zapisze niczego, więc start z nim ma się zatrzymać z jasnym komunikatem, a nie
    produkować cykl pustych odczytów. Klucz nie może wyciec do komunikatu błędu."""

    @pytest.mark.parametrize(
        "key",
        [
            "sb_publishable_0123456789abcdef",
            _unsigned_jwt({"iss": "supabase", "ref": "project", "role": "anon"}),
            _unsigned_jwt({"iss": "supabase", "ref": "project", "role": "authenticated"}),
        ],
        ids=["publishable", "legacy-anon-jwt", "authenticated-jwt"],
    )
    def test_rejects_public_supabase_key(self, env, key):
        env.setenv("SUPABASE_KEY", key)

        with pytest.raises(ValidationError, match="sb_secret_") as raised:
            Settings(_env_file=None)

        assert "service_role" in str(raised.value)
        assert key not in str(raised.value)

    @pytest.mark.parametrize(
        "key",
        [
            "sb_secret_0123456789abcdef",
            _unsigned_jwt({"iss": "supabase", "ref": "project", "role": "service_role"}),
            # Fikstury testów i nieprzezroczyste stringi: bez prefiksu i bez JWT.
            "supa-key",
            "anon",
            "not.a-jwt.payload",
            _unsigned_jwt({"iss": "supabase"}),
        ],
        ids=[
            "secret", "service-role-jwt", "fixture", "bare-anon-word", "dotted", "jwt-without-role",
        ],
    )
    def test_accepts_secret_and_opaque_keys(self, env, key):
        env.setenv("SUPABASE_KEY", key)

        assert Settings(_env_file=None).supabase_key == key

    def test_other_validation_errors_do_not_echo_the_key(self, env):
        """`hide_input_in_errors`: błąd innego pola nie wypisuje wejścia, więc
        traceback startu nie niesie wartości sekretów."""
        with pytest.raises(ValidationError) as raised:
            Settings(_env_file=None, supabase_key=["sb_secret_list-instead-of-str"])

        assert "sb_secret_list-instead-of-str" not in str(raised.value)


@pytest.mark.parametrize(
    "field",
    ["volatility_threshold", "crypto_volatility_threshold", "tool_use_volatility_threshold"],
)
def test_zero_volatility_threshold_rejected(env, field: str) -> None:
    # FIN-07: próg 0 przepuszcza każdy ruch (|Δ| >= 0), więc bramka FinOps
    # cicho znika i płatne porty odpalają dla wszystkich symboli w każdym cyklu.
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        Settings(_env_file=None, **{field: Decimal("0")})


def test_zero_council_volatility_threshold_stays_the_always_run_switch(env) -> None:
    settings = Settings(_env_file=None, council_volatility_threshold=Decimal("0"))

    assert settings.council_volatility_threshold == Decimal("0")


class TestAlphaVantageBudget:
    """FIN-32: najgorszy przypadek doby (cała watchlista za bramką) ≤ 25 × klucze.

    Przekroczenie wychodziło dopiero jako `av_keys_exhausted` w środku cyklu.
    Domyślnie WARNING i fail-open (liczba kluczy w sekretach jest nieznana),
    twardy błąd startu tylko przy `av_budget_strict = true`."""

    _WATCHLIST = [f"S{i:02d}" for i in range(43)]

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_settings_warns_when_av_worst_case_exceeds_capacity(self, env, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="src.config"):
            settings = self._settings(symbols=self._WATCHLIST)  # jeden klucz

        assert settings.av_budget().ok is False
        assert "43" in caplog.text
        assert "25" in caplog.text
        assert "ALPHA_VANTAGE_API_KEYS" in caplog.text

    def test_enough_keys_keep_the_log_quiet(self, env, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="src.config"):
            settings = self._settings(
                symbols=self._WATCHLIST, alpha_vantage_api_keys=["av-1", "av-2"]
            )

        assert settings.av_budget().ok is True
        assert "Alpha Vantage" not in caplog.text

    def test_worst_case_counts_the_watchlist_without_unpriced_symbols(self, env):
        # WIA-11: symbole + krypto − symbols_unsupported_price (te nie wchodzą do grafu).
        settings = self._settings(
            symbols=["AAPL", "MSFT", "CSPX.L"],
            crypto_symbols=["BTC"],
            symbols_unsupported_price=["CSPX.L"],
        )

        assert settings.av_budget().worst_case == 3

    def test_earnings_calendar_adds_one_request_per_cycle(self, env):
        # FIN-09: jeden EARNINGS_CALENDAR bez symbolu obsługuje całą watchlistę.
        settings = self._settings(symbols=["AAPL", "MSFT"], earnings_calendar_enabled=True)

        assert settings.av_budget().worst_case == 3

    def test_earnings_from_the_slow_loop_cache_cost_the_fast_loop_nothing(self, env):
        # FIN-11: kalendarz z `earnings_cache` — Fast Loop pyta AV tylko o newsy.
        settings = self._settings(
            symbols=["AAPL", "MSFT"],
            earnings_calendar_enabled=True,
            earnings_cache_enabled=True,
        )

        assert settings.av_budget().worst_case == 2

    def test_earnings_cache_is_off_by_default_and_outside_the_recipe(self, env):
        # FIN-11 (decyzja 2026-10-21…23): cache zmienia źródło tej samej daty wyników,
        # nie przepis — przełączenie flagi nie może ciąć track recordu na segmenty.
        from src.config import NON_RECIPE_SETTINGS_FIELDS, RECIPE_SETTINGS_FIELDS

        assert self._settings().earnings_cache_enabled is False
        assert "earnings_cache_enabled" in NON_RECIPE_SETTINGS_FIELDS
        assert "earnings_cache_enabled" not in RECIPE_SETTINGS_FIELDS

    def test_strict_flag_stops_startup(self, env):
        with pytest.raises(ValueError, match="Alpha Vantage"):
            self._settings(symbols=self._WATCHLIST, av_budget_strict=True)

    def test_strict_flag_defaults_off_and_stays_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert self._settings().av_budget_strict is False
        assert "av_budget_strict" in NON_RECIPE_SETTINGS_FIELDS


class TestFastLoopCyclesPerDay:
    """FIN-32: kadencja Fast Loopa w budżecie AV to pole konfiguracji (config.toml),
    a nie stała w kodzie — zgodność z cronem pilnuje test w test_config_toml.py."""

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_defaults_to_one_cycle_a_day(self, env):
        assert self._settings().fast_loop_cycles_per_day == 1

    def test_budget_multiplies_by_the_cycles_per_day(self, env):
        symbols = [f"S{i:02d}" for i in range(13)]

        assert self._settings(symbols=symbols).av_budget().ok is True
        twice = self._settings(symbols=symbols, fast_loop_cycles_per_day=2).av_budget()
        assert twice.worst_case == 26
        assert twice.ok is False

    def test_zero_cycles_are_rejected(self, env):
        # Zero wyzerowałoby worst case i po cichu wyłączyło kontrolę budżetu.
        with pytest.raises(ValueError, match="fast_loop_cycles_per_day"):
            self._settings(fast_loop_cycles_per_day=0)

    def test_stays_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "fast_loop_cycles_per_day" in NON_RECIPE_SETTINGS_FIELDS
        baseline = self._settings().config_hash()
        assert self._settings(fast_loop_cycles_per_day=2).config_hash() == baseline


class TestMaxPaidLlmCallsPerCycle:
    """FIN-12: sufit płatnych wywołań LLM na cykl (predykcja + diagnoza + rada).
    None = bez limitu; produkcyjny `config.toml` ustawia 100."""

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_defaults_to_no_cap(self, env):
        assert self._settings().max_paid_llm_calls_per_cycle is None

    def test_accepts_a_positive_cap(self, env):
        assert self._settings(max_paid_llm_calls_per_cycle=100).max_paid_llm_calls_per_cycle == 100

    def test_zero_cap_is_rejected(self, env):
        # Zero zamknęłoby każdą analizę po cichu; lokalny wyłącznik to PAID_CALLS_ENABLED.
        with pytest.raises(ValueError, match="max_paid_llm_calls_per_cycle"):
            self._settings(max_paid_llm_calls_per_cycle=0)

    def test_stays_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "max_paid_llm_calls_per_cycle" in NON_RECIPE_SETTINGS_FIELDS
        baseline = self._settings().config_hash()
        assert self._settings(max_paid_llm_calls_per_cycle=50).config_hash() == baseline


class TestSlowLoopWeeklyLlmBudget:
    """FIN-30: tygodniowy sufit płatnych wywołań LLM Slow Loopa (recap, sędzia
    kalibracji i przyszłe joby). None = bez limitu; `config.toml` ustawia 15."""

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_defaults_to_no_cap(self, env):
        assert self._settings().slow_loop_max_paid_llm_calls_per_week is None

    def test_zero_is_a_valid_llm_free_week(self, env):
        # Zero = recap bez narracji i kalibracja bez wniosku (deterministyczne fallbacki).
        assert self._settings(
            slow_loop_max_paid_llm_calls_per_week=0
        ).slow_loop_max_paid_llm_calls_per_week == 0

    def test_negative_cap_is_rejected(self, env):
        with pytest.raises(ValueError, match="slow_loop_max_paid_llm_calls_per_week"):
            self._settings(slow_loop_max_paid_llm_calls_per_week=-1)

    def test_stays_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "slow_loop_max_paid_llm_calls_per_week" in NON_RECIPE_SETTINGS_FIELDS
        baseline = self._settings().config_hash()
        assert self._settings(slow_loop_max_paid_llm_calls_per_week=3).config_hash() == baseline


class TestModelScorecardRunIdFlag:
    """ML-24: `run_id` w karcie modelu dopiero po migracji 049 — domyślnie off."""

    def test_defaults_off(self, env):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.model_scorecard_run_id_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "model_scorecard_run_id_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestModelScorecardFeatureStatsFlag:
    """ML-21: rozkład cech treningu w karcie dopiero po migracji 052 — domyślnie off,
    poza przepisem (sam zapis w Slow Loopie)."""

    def test_defaults_off(self, env):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.model_scorecard_feature_stats_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "model_scorecard_feature_stats_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, model_scorecard_feature_stats_enabled=True
        ).config_hash()
        assert flipped == baseline


class TestModelScorecardWeightsSha256Flag:
    """CI-13: sha256 pliku wag w karcie dopiero po migracji 060 — domyślnie off,
    poza przepisem (sam zapis w Slow Loopie)."""

    def test_defaults_off(self, env):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.model_scorecard_weights_sha256_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "model_scorecard_weights_sha256_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, model_scorecard_weights_sha256_enabled=True
        ).config_hash()
        assert flipped == baseline


class TestModelScorecardProvenanceFlag:
    """ML-22: proweniencja treningu w karcie dopiero po migracji 064 — domyślnie off,
    poza przepisem (sam zapis w Slow Loopie)."""

    def test_defaults_off(self, env):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.model_scorecard_provenance_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "model_scorecard_provenance_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, model_scorecard_provenance_enabled=True
        ).config_hash()
        assert flipped == baseline


class TestSettledPurgeFlag:
    """ML-26: purge foldów po chwili rozliczenia etykiety zmienia werdykt bramki,
    więc i wagi — domyślnie off; wpływ na prognozę idzie wyłącznie przez wagi
    (`model_version`), więc poza `config_hash` jak `snapshot_training_enabled`."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).ml_settled_purge_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "ml_settled_purge_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestNyseClosureSkipFlag:
    """FIN-34: w dzień zamknięcia NYSE Fast Loop pomija akcje. Flaga w kodzie
    domyślnie off (testy entry pointów nie zależą od daty), `config.toml` ją włącza."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).nyse_closure_skip_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "nyse_closure_skip_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestPaidCallsFlag:
    """FIN-35: płatne porty (LLM, rada, Alpha Vantage, embeddingi) tylko jawnie.
    Jak `DB_WRITES_ENABLED`: przełącznik żyje w środowisku, domyślnie false,
    workflowy, które płacą, ustawiają "true"."""

    def test_defaults_off(self, env):
        env.delenv("PAID_CALLS_ENABLED", raising=False)

        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).paid_calls_enabled is False

    def test_reads_the_environment_variable(self, env):
        env.setenv("PAID_CALLS_ENABLED", "true")

        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).paid_calls_enabled is True

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "paid_calls_enabled" in NON_RECIPE_SETTINGS_FIELDS
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        flipped = Settings(
            _env_file=None, **REQUIRED_ENV_KWARGS, paid_calls_enabled=True
        ).config_hash()
        assert flipped == baseline


class TestCostLedgerSettings:
    """FIN-13: cennik LLM per model (USD za 1M tokenów, z datą) i zapis ledgera
    `llm_calls` za flagą — produkcja nie ma migracji 040, więc w kodzie off."""

    def test_ledger_write_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).cost_ledger_enabled is False

    def test_no_price_list_in_code(self, env):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.llm_pricing == {}
        assert settings.llm_pricing_as_of is None
        assert settings.llm_price_table() == {}

    def test_price_list_becomes_the_domain_price_table(self, env):
        from datetime import date

        from src.domain.finops import ModelPrice

        settings = Settings(
            _env_file=None,
            **REQUIRED_ENV_KWARGS,
            llm_pricing_as_of="2026-10-20",
            llm_pricing={
                "claude-sonnet-4-6": {
                    "input": 3.0,
                    "output": 15.0,
                    "cache_read": 0.3,
                    "cache_write": 3.75,
                },
                "text-embedding-3-small": {"input": 0.02},
            },
        )

        assert settings.llm_pricing_as_of == date(2026, 10, 20)
        assert settings.llm_price_table() == {
            "claude-sonnet-4-6": ModelPrice(
                input_usd_per_mtok=3.0,
                output_usd_per_mtok=15.0,
                cache_read_usd_per_mtok=0.3,
                cache_write_usd_per_mtok=3.75,
            ),
            "text-embedding-3-small": ModelPrice(input_usd_per_mtok=0.02),
        }

    def test_negative_rate_is_rejected(self, env):
        with pytest.raises(ValidationError, match="input"):
            Settings(
                _env_file=None,
                **REQUIRED_ENV_KWARGS,
                llm_pricing={"gpt-5-mini": {"input": -0.25, "output": 2.0}},
            )

    def test_unknown_rate_key_is_rejected(self, env):
        # Literówka („ouput”) nie może po cichu zostawić modelu bez stawki wyjścia.
        with pytest.raises(ValidationError, match="ouput"):
            Settings(
                _env_file=None,
                **REQUIRED_ENV_KWARGS,
                llm_pricing={"gpt-5-mini": {"input": 0.25, "ouput": 2.0}},
            )

    def test_pricing_and_ledger_stay_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert {"cost_ledger_enabled", "llm_pricing", "llm_pricing_as_of"} <= (
            NON_RECIPE_SETTINGS_FIELDS
        )
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS).config_hash()
        priced = Settings(
            _env_file=None,
            **REQUIRED_ENV_KWARGS,
            llm_pricing={"gpt-5-mini": {"input": 0.25, "output": 2.0}},
            llm_pricing_as_of="2026-10-20",
        ).config_hash()
        assert priced == baseline


class TestTwoPhaseCycleFlag:
    """EXP-27: cykl dwufazowy to eksperyment za flagą, domyślnie wyłączony."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).two_phase_cycle_enabled is False

    def test_is_outside_the_recipe(self, env):
        # Faza 2 to ten sam przepis płatnej analizy co cykl jednofazowy; flaga
        # zmienia kolejność i to, kogo tnie sufit — jak `max_paid_llm_calls_per_cycle`.
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "two_phase_cycle_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestCycleRunsFlag:
    """NIEZ-29: zapis historii przebiegów (migracja 041) — flaga samego zapisu,
    domyślnie off i poza przepisem (`config_hash` się nie rusza)."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).cycle_runs_enabled is False

    def test_is_a_write_only_flag_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "cycle_runs_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestHeartbeatUrl:
    """NIEZ-13: `HEARTBEAT_URL` to sekret (UUID checku w ścieżce) — poza hashem
    przepisu i redagowany w logach jak każdy klucz."""

    def test_defaults_to_none(self, env):
        env.delenv("HEARTBEAT_URL", raising=False)

        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).heartbeat_url is None

    def test_is_a_secret_outside_the_hash_and_redacted(self, env):
        from src.config import SECRET_SETTINGS_FIELDS

        url = "https://hc-ping.com/5f3c0b9e-7a1d-4c2e-9b8f-0d6a2e4c1b7f"
        baseline = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)
        with_url = Settings(_env_file=None, **REQUIRED_ENV_KWARGS, heartbeat_url=url)

        assert "heartbeat_url" in SECRET_SETTINGS_FIELDS
        assert with_url.config_hash() == baseline.config_hash()
        assert url in with_url.secret_values()

    def test_is_read_from_the_environment(self, env, monkeypatch):
        monkeypatch.setenv("HEARTBEAT_URL", "https://hc-ping.com/abc-from-env")

        assert (
            Settings(_env_file=None, **REQUIRED_ENV_KWARGS).heartbeat_url
            == "https://hc-ping.com/abc-from-env"
        )


class TestReferenceSignalFlag:
    """WIA-24: sygnał referencyjny w Slow Loopie (migracja 042) — flaga osobnej tabeli,
    domyślnie off i poza przepisem (`config_hash` się nie rusza)."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).reference_signal_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "reference_signal_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestFastLoopWeeklyBudget:
    """WIA-11: jawny budżet tygodniowy Fast Loopa (`fast_loop_weekly_budget_usd`).
    Najgorszy przypadek tygodnia = koszt cyklu przy suficie FIN-12 (cała watchlista
    za bramką) × cykle na dobę × 5 dni handlowych. None = brak kontroli."""

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_defaults_to_no_budget(self, env):
        assert self._settings().fast_loop_weekly_budget_usd is None

    def test_negative_budget_is_rejected(self, env):
        with pytest.raises(ValueError, match="fast_loop_weekly_budget_usd"):
            self._settings(fast_loop_weekly_budget_usd=-1.0)

    def test_stays_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "fast_loop_weekly_budget_usd" in NON_RECIPE_SETTINGS_FIELDS
        baseline = self._settings().config_hash()
        assert self._settings(fast_loop_weekly_budget_usd=7.0).config_hash() == baseline

    def test_worst_case_uses_the_eligible_watchlist_and_the_ceiling(self, env):
        from src.domain.finops import capped_cycle_call_upper_bound, estimate_cycle_cost

        settings = self._settings(
            symbols=["AAPL", "MSFT", "CSPX.L"],
            crypto_symbols=["BTC"],
            symbols_unsupported_price=["CSPX.L"],
            max_paid_llm_calls_per_cycle=5,
            symbol_concurrency=2,
            fast_loop_cycles_per_day=2,
            fast_loop_weekly_budget_usd=1.0,
        )

        weekly = settings.fast_loop_weekly_budget(personas=7)

        # 3 symbole za bramką (CSPX.L bez ceny), sufit 5 i dwa symbole naraz.
        bound = capped_cycle_call_upper_bound(
            n_gated=3, n_av_symbols=3, personas=7, debate=False, tool_rounds=0,
            cap=5, concurrency=2,
        )
        assert bound.calls["council_llm"] == 4 + 2 * 8
        assert weekly.cycles_per_week == 10
        assert weekly.cycle_worst_usd == pytest.approx(estimate_cycle_cost(bound.calls).total_usd)
        assert weekly.budget_usd == 1.0

    def test_contagion_runs_symbols_one_at_a_time(self, env):
        # Contagion wymusza przetwarzanie sekwencyjne, więc w locie jest jeden symbol.
        parallel = self._settings(
            symbols=[f"S{i:02d}" for i in range(20)],
            max_paid_llm_calls_per_cycle=20,
            symbol_concurrency=4,
        )
        sequential = parallel.model_copy(update={"contagion_enabled": True})

        assert (
            sequential.fast_loop_weekly_budget(personas=7).cycle_worst_usd
            < parallel.fast_loop_weekly_budget(personas=7).cycle_worst_usd
        )

    def test_tool_use_rounds_raise_the_worst_case(self, env):
        base = self._settings(symbols=["AAPL"])
        tool_use = base.model_copy(update={"tool_use_enabled": True, "tool_use_max_rounds": 3})

        assert (
            tool_use.fast_loop_weekly_budget(personas=7).cycle_worst_usd
            > base.fast_loop_weekly_budget(personas=7).cycle_worst_usd
        )


class TestPersonaVoteOutcomesFlag:
    """WIA-51: wagi i ranking person z surowych głosów RPC 045 regułą domeny —
    flaga odczytu wymagającego migracji, domyślnie off. Zmienia wagi konsensusu
    (przy `persona_weighting_enabled`), więc należy do przepisu."""

    def test_defaults_off(self, env):
        assert (
            Settings(_env_file=None, **REQUIRED_ENV_KWARGS).persona_vote_outcomes_enabled
            is False
        )

    def test_is_part_of_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        assert "persona_vote_outcomes_enabled" in RECIPE_SETTINGS_FIELDS


class TestPersonaBrierRankingFlag:
    """WIA-44: kolejność leaderboardu person po Brier Skill (RPC 065) — flaga
    odczytu wymagającego migracji, domyślnie off, tylko treść maila."""

    def test_defaults_off(self, env):
        assert (
            Settings(_env_file=None, **REQUIRED_ENV_KWARGS).persona_brier_ranking_enabled
            is False
        )

    def test_is_not_part_of_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS, RECIPE_SETTINGS_FIELDS

        assert "persona_brier_ranking_enabled" in NON_RECIPE_SETTINGS_FIELDS
        assert "persona_brier_ranking_enabled" not in RECIPE_SETTINGS_FIELDS


class TestCouncilRedundancyFlag:
    """WIA-45: linia redundancji w recapie i brak odznaki „silny konsensus” przy
    jednym efektywnym głosie — sama treść maili, domyślnie off, poza przepisem."""

    def test_defaults_off(self, env):
        assert (
            Settings(_env_file=None, **REQUIRED_ENV_KWARGS).council_redundancy_enabled
            is False
        )

    def test_is_not_part_of_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS, RECIPE_SETTINGS_FIELDS

        assert "council_redundancy_enabled" in NON_RECIPE_SETTINGS_FIELDS
        assert "council_redundancy_enabled" not in RECIPE_SETTINGS_FIELDS


class TestCorrectionCategoryFlag:
    """WIA-57: kolumna `correction_category` (migracja 046) — sam zapis kategorii
    diagnozy i jej odczyt w panelu lessons; prognoza ta sama, poza przepisem."""

    def test_defaults_off(self, env):
        assert (
            Settings(_env_file=None, **REQUIRED_ENV_KWARGS).correction_category_enabled
            is False
        )

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "correction_category_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestDiagnosisBackfillSettings:
    """WIA-53: job Slow Loopa diagnozujący błędne predykcje bez diagnozy — off,
    z capem płatnych wywołań na przebieg; inny job niż Fast Loop, poza przepisem."""

    def test_defaults_off_with_a_cap_of_five(self, env):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.diagnosis_backfill_enabled is False
        assert settings.diagnosis_backfill_max_calls == 5

    def test_negative_cap_is_rejected(self, env):
        with pytest.raises(ValueError):
            Settings(_env_file=None, **REQUIRED_ENV_KWARGS, diagnosis_backfill_max_calls=-1)

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert {"diagnosis_backfill_enabled", "diagnosis_backfill_max_calls"} <= (
            NON_RECIPE_SETTINGS_FIELDS
        )

class TestPriceAlerts:
    """ALR-11: lista startowa `price_alerts` z config.toml i flaga watcha (off do
    migracji 057). Wpis jest walidowany przy starcie — literówka w kierunku albo
    poziom ≤ 0 zatrzymuje start, zamiast po cichu nie alarmować."""

    def _settings(self, **overrides) -> Settings:
        return Settings(_env_file=None, **{**REQUIRED_ENV_KWARGS, **overrides})

    def test_entries_become_domain_levels(self, env):
        from src.domain.price_levels import LevelDirection, PriceLevel

        settings = self._settings(
            price_alerts=[
                {"symbol": " nvda ", "level": "150", "direction": "ABOVE", "note": "zysk"},
                {"symbol": "BTC", "level": 60000, "direction": "BELOW"},
            ]
        )

        assert settings.price_level_seeds() == (
            PriceLevel("NVDA", Decimal("150"), LevelDirection.ABOVE, "zysk"),
            PriceLevel("BTC", Decimal("60000"), LevelDirection.BELOW),
        )

    @pytest.mark.parametrize(
        "entry",
        [
            {"symbol": "NVDA", "level": "150", "direction": "UP"},
            {"symbol": "NVDA", "level": "0", "direction": "ABOVE"},
            {"symbol": "NVDA", "level": "150", "direction": "ABOVE", "notes": "literówka"},
        ],
    )
    def test_bad_entry_stops_the_start(self, env, entry):
        with pytest.raises(ValueError):
            self._settings(price_alerts=[entry])

    def test_defaults_are_off_and_empty_and_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        settings = self._settings()

        assert settings.price_level_alerts_enabled is False
        assert settings.price_level_seeds() == ()
        assert {"price_level_alerts_enabled", "price_alerts"} <= NON_RECIPE_SETTINGS_FIELDS


class TestInsiderLeadMaxAgeDays:
    """DAT-21: zakup insidera prowadzi maila tylko przez N dni od filingu —
    prezentacja maila, poza przepisem predykcji."""

    def test_defaults_to_three_days(self, env):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.insider_lead_max_age_days == 3

    def test_negative_age_is_rejected(self, env):
        with pytest.raises(ValueError):
            Settings(_env_file=None, **REQUIRED_ENV_KWARGS, insider_lead_max_age_days=-1)

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "insider_lead_max_age_days" in NON_RECIPE_SETTINGS_FIELDS


class TestSessionCloseSettlementFlag:
    """WIA-61: zapis zamknięcia sesji obok ceny /quote (migracja 059) — off do
    migracji; pomiar rozliczenia, nie prognoza, więc poza przepisem."""

    def test_defaults_off_and_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.session_close_settlement_enabled is False
        assert "session_close_settlement_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestExecutableEntryFlag:
    """WIA-54: cena wykonalna od otwarcia sesji (migracja 072) — off do migracji;
    pomiar rozliczenia i druga krzywa w mailu, nie prognoza, więc poza przepisem."""

    def test_defaults_off_and_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.executable_entry_enabled is False
        assert "executable_entry_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestEdgeCalendarHorizonFlag:
    """WIA-28: horyzont edge z kalendarza NYSE zamiast stałych 24 h — zmienia zapisany
    `edge_sigma` (sygnał oceniany w `evaluate`), więc należy do przepisu; off na czas
    EXP-02/03."""

    def test_defaults_off_and_inside_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.edge_calendar_horizon_enabled is False
        assert "edge_calendar_horizon_enabled" in RECIPE_SETTINGS_FIELDS


class TestSplitDetectionFlag:
    """DAT-23: podejrzenie splitu z `pc` koryguje bramkę, rozliczenie i `price_delta`
    (przepis) i zapisuje zdarzenie w `corporate_actions` (migracja 062). Off."""

    def test_defaults_off_and_inside_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.split_detection_enabled is False
        assert "split_detection_enabled" in RECIPE_SETTINGS_FIELDS


class TestPriceOutlierGuardFlag:
    """DAT-15: anomalny odczyt ceny zamyka bramkę i odkłada rozliczenie — zmiana
    przepisu, bez migracji. Off na czas EXP-02/03."""

    def test_defaults_off_and_inside_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.price_outlier_guard_enabled is False
        assert "price_outlier_guard_enabled" in RECIPE_SETTINGS_FIELDS


class TestRagExactSearchFlag:
    """ML-23: dokładne KNN analogów RAG (migracja 063) zmienia analogi w prompcie,
    więc należy do przepisu; off na czas EXP-03."""

    def test_defaults_off_and_inside_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.rag_exact_search_enabled is False
        assert "rag_exact_search_enabled" in RECIPE_SETTINGS_FIELDS


class TestPromptTaintFenceFlag:
    """BEZ-14: fence lekcji z reflect i uzasadnień person zmienia tekst promptów,
    więc należy do przepisu; off na czas EXP-02/03."""

    def test_defaults_off_and_inside_the_recipe(self, env):
        from src.config import RECIPE_SETTINGS_FIELDS

        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.prompt_taint_fence_enabled is False
        assert "prompt_taint_fence_enabled" in RECIPE_SETTINGS_FIELDS


class TestDirectionClassifierFlag:
    """ML-30: wariant binarny to sama ewaluacja w Slow Loopie — domyślnie off,
    poza przepisem (champion i prognoza bez zmian)."""

    def test_defaults_off(self, env):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.ml_direction_classifier_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "ml_direction_classifier_enabled" in NON_RECIPE_SETTINGS_FIELDS


class TestShadowPredictionsFlag:
    """ML-33: prognozy cienia championa i challengerów — nowy zapis do bazy (migracja
    075), domyślnie off i poza przepisem (wynik cyklu ten sam); do dwóch challengerów
    o unikalnych nazwach innych niż champion."""

    def test_defaults_off_without_challengers(self, env):
        settings = Settings(_env_file=None, **REQUIRED_ENV_KWARGS)

        assert settings.shadow_predictions_enabled is False
        assert settings.ml_challengers == []

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert {"shadow_predictions_enabled", "ml_challengers"} <= NON_RECIPE_SETTINGS_FIELDS

    def test_reads_challengers_with_their_kind(self, env):
        settings = Settings(
            _env_file=None,
            **REQUIRED_ENV_KWARGS,
            ml_challengers=[
                {"name": "direction", "path": "data/models/d.ubj", "kind": "direction"},
                {"name": "alt_features", "path": "data/models/alt.ubj", "kind": "return"},
            ],
        )

        assert [(c.name, c.kind) for c in settings.ml_challengers] == [
            ("direction", "direction"),
            ("alt_features", "return"),
        ]

    @pytest.mark.parametrize(
        "challengers",
        [
            [
                {"name": f"c{i}", "path": f"data/models/c{i}.ubj", "kind": "return"}
                for i in range(3)
            ],
            [
                {"name": "alt", "path": "data/models/a.ubj", "kind": "return"},
                {"name": "alt", "path": "data/models/b.ubj", "kind": "return"},
            ],
            [{"name": "champion", "path": "data/models/c.ubj", "kind": "return"}],
            [{"name": "alt", "path": "data/models/price_predictor.ubj", "kind": "return"}],
            [{"name": "Alt", "path": "data/models/a.ubj", "kind": "return"}],
            [{"name": "alt", "path": "data/models/a.ubj", "kind": "price"}],
        ],
    )
    def test_rejects_challengers_the_shadow_table_cannot_tell_apart(self, env, challengers):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **REQUIRED_ENV_KWARGS, ml_challengers=challengers)


class TestIsotonicCalibrationFlag:
    """ML-28: mapa izotoniczna w rankingu sygnałów — domyślnie off, poza przepisem
    (ranking liczy się po predykcji, jak `calibrated_confidence_enabled`)."""

    def test_defaults_off(self, env):
        assert Settings(_env_file=None, **REQUIRED_ENV_KWARGS).isotonic_calibration_enabled is False

    def test_is_outside_the_recipe(self, env):
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        assert "isotonic_calibration_enabled" in NON_RECIPE_SETTINGS_FIELDS


def test_env_file_is_anchored_at_the_repo_root() -> None:
    """ARCH-23: `.env` jak `config.toml` leży względem roota repo, nie CWD — job
    odpalony z innego katalogu nie traci po cichu kluczy i sekretów."""
    repo_root = Path(__file__).resolve().parent.parent

    assert Path(str(Settings.model_config["env_file"])) == repo_root / ".env"
