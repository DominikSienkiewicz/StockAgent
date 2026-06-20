from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from src.domain.macro_risk import MacroRiskInstrumentType

# Ścieżka do config.toml zakotwiczona w ROOCIE repo (src/config.py → ../config.toml),
# NIE względem CWD. Inaczej agent odpalony spoza roota nie znajduje pliku i cały
# config niewrażliwy po cichu spada do defaultów (np. notifications_enabled=False).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TOML_FILE = str(_REPO_ROOT / "config.toml")


class Settings(BaseSettings):
    """Konfiguracja runtime — dwa rozłączne źródła:

    - **Config niewrażliwy** (symbole, progi, modele, providerzy, throttle…)
      żyje w commitowanym `config.toml` — single source of truth, wersjonowany,
      review-owalny. Bez duplikacji w `.env`/`.env.example`/workflow.
    - **Sekrety** (klucze API, DB credentials, odbiorca maila) NIGDY nie trafiają
      do `config.toml` — czytane z env / `.env` (lokalnie) / GitHub Secrets (CI).

    Precedencja (od najwyższej): init kwargs → env var → `.env` → `config.toml`
    → defaulty w kodzie. Dzięki temu env/.env może nadpisać dowolną wartość
    (np. szybki eksperyment lokalny), a sekrety wstrzykuje się przez env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Dokłada `config.toml` jako źródło PONIŻEJ env/.env (sekrety i
        eksperymenty z env nadpisują), a POWYŻEJ defaultów w kodzie.

        `STOCKAGENT_DISABLE_TOML=1` pomija źródło TOML — używają tego testy
        jednostkowe, by asertować czyste defaulty pól bez wpływu config.toml.
        """
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if os.environ.get("STOCKAGENT_DISABLE_TOML") != "1":
            # Ścieżkę podajemy wprost (nie przez model_config), by w trybie
            # wyłączonym nie było ostrzeżenia o nieużytym kluczu `toml_file`.
            # `STOCKAGENT_TOML_FILE` pozwala testom wskazać tymczasowy plik.
            toml_file = os.environ.get("STOCKAGENT_TOML_FILE", _DEFAULT_TOML_FILE)
            sources.append(
                TomlConfigSettingsSource(settings_cls, toml_file=toml_file)
            )
        sources.append(file_secret_settings)
        return tuple(sources)

    # ----- LLM -----
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str
    anthropic_api_key: str | None = None

    # Heterogeniczna strategia: rada doradcza (N person + chairman × M symboli) może
    # działać na tańszym modelu niż główna analiza, bo persona-acting + JSON
    # nie wymaga frontier reasoningu. Gdy obie None → rada używa głównego
    # LLM (zachowanie domyślne, wstecznie kompatybilne). Override przykładowo:
    #   council_llm_provider=anthropic, council_llm_model=claude-haiku-4-5
    # albo (ten sam provider co main, inny model):
    #   council_llm_model=gpt-4o-mini
    council_llm_provider: Literal["openai", "anthropic"] | None = None
    council_llm_model: str | None = None

    # Katalog z plikami JSON definiującymi członków rady doradczej. Dodanie /
    # usunięcie persony = dodanie / usunięcie pliku, bez zmian w kodzie.
    # Walidator: `uv run python -m src.tools.validate_personas`.
    council_personas_dir: str = "data/council_personas"

    # ----- Market / sentiment / news -----
    finnhub_api_key: str
    # Alpha Vantage akceptuje CSV — wiele kluczy rotujemy, gdy jeden wyczerpie
    # dzienny limit (25 req/dobę free). Wpisz w .env: ALPHA_VANTAGE_API_KEYS=k1,k2,k3
    alpha_vantage_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # ----- Database -----
    supabase_url: str
    supabase_key: str

    # ----- Agent config -----
    volatility_threshold: Decimal = Field(default=Decimal("0.02"))
    # Osobny, wyższy próg dla rady doradczej. Rada (N wywołań LLM person + chairman)
    # jest droga — przy małych Δ jej werdykt to prawie zawsze HOLD. Domyślnie
    # 3% — można podnieść do 5% przy ciasnym budżecie albo wyłączyć (0.0) dla
    # backtestów. Ustaw 0.0, by rada chodziła zawsze gdy główna bramka przepuści.
    council_volatility_threshold: Decimal = Field(default=Decimal("0.03"))
    # NoDecode wyłącza próbę JSON-parse'owania env var — używamy własnego validatora CSV.
    # Default to małe smoke-test portfolio; w produkcji nadpisz przez SYMBOLS env var.
    symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA"]
    )
    # Symbole klasyfikowane jako ETF — pomijają fundamentale (brak EPS/P/E).
    symbols_etf: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # ----- Krypto -----
    # Tickery klasyfikowane jako AssetType.CRYPTO. Routing: CoinGecko zamiast
    # Finnhuba dla ceny, AV NEWS_SENTIMENT z prefiksem CRYPTO:. Krypto nie
    # ma EPS/P/E, więc fundamentale pomijane jak dla ETF.
    crypto_symbols: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Osobny, wyższy próg volatility — natywna zmienność BTC/ETH to 3-5%
    # dziennie, więc 2% (akcyjny) odpalałby pełną analizę KAŻDEGO cyklu.
    # 5% sprawia, że pełen pipeline (LLM + council) idzie tylko gdy ruch
    # jest faktycznie sygnałem, nie szumem.
    crypto_volatility_threshold: Decimal = Field(default=Decimal("0.05"))

    # ----- Risk Watch -----
    # Lista instrumentów proxy ryzyka makro (inverse ETFs, gold, VIX, EPOL).
    # Monitorowane przez osobny use case (MonitorMacroRisk), nie wchodzą do
    # głównej pętli predykcyjnej.
    risk_symbols: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Mapowanie symbol → typ instrumentu (np. SH:INVERSE_EQUITY).
    # Format env: "SYM:TYPE,SYM:TYPE,..." gdzie TYPE ∈ MacroRiskInstrumentType.
    risk_symbol_types: Annotated[
        dict[str, MacroRiskInstrumentType], NoDecode
    ] = Field(default_factory=dict)
    # Włącza adapter NBP do raportu (kursy EUR/PLN, USD/PLN).
    nbp_enabled: bool = False

    # ----- Resilience -----
    # Tickery, które wiadomo z góry, że nie zadziałają z obecnymi adapterami
    # (Finnhub free → 403 dla EU-listed). Pre-filtrujemy je w głównej pętli
    # i mailujemy jako "ignored", nie "error" — żeby nie zaśmiecać sekcji
    # błędów oczywistymi ograniczeniami planu.
    symbols_unsupported_price: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )
    # Sleep w sekundach między symbolami w głównej pętli — chroni OpenAI TPM
    # przy 30+ tickerach z radą doradczą. 0 = bez throttle.
    symbol_throttle_seconds: float = Field(default=0.0)

    # Minimalny wiek (godziny) predykcji, zanim reflect_node ją oceni. Chroni
    # przed przedwczesną oceną przy nakładających się cyklach (ręczny
    # workflow_dispatch tuż po scheduled run oceniłby świeżą predykcję po
    # cenie sprzed minut → zatruty accuracy_score i zawyżony hit-rate).
    # Default = 6h: bezpieczna bramka zgodna z produkcyjną kadencją dzienną
    # (config.toml też ustawia 6). Deployment na samych defaultach kodu — albo
    # konstrukcja Settings() bez TOML — dostaje działającą bramkę, nie footgun.
    # Ustaw 0, by świadomie wyłączyć filtr (np. backtest / zachowanie wsteczne).
    reflection_min_age_hours: int = Field(default=6)

    # ----- ML -----
    ml_model_path: str = "data/models/price_predictor.ubj"

    # ----- Notifications (Resend.com — opcjonalne) -----
    notifications_enabled: bool = False
    resend_api_key: str | None = None
    # sandbox sender (działa od razu, bez weryfikacji domeny)
    digest_from_email: str = "onboarding@resend.dev"
    digest_to_email: str | None = None

    @field_validator("symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, value: str | list[str]) -> list[str]:
        """Pozwala podać SYMBOLS=AAPL,MSFT,GOOGL w .env (CSV → list[str])."""
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return value

    @field_validator("symbols_etf", mode="before")
    @classmethod
    def _parse_symbols_etf(cls, value: str | list[str]) -> list[str]:
        """Pozwala podać SYMBOLS_ETF=VOO,CSPX.L w .env (CSV → list[str])."""
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return value

    @field_validator("risk_symbols", mode="before")
    @classmethod
    def _parse_risk_symbols(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return value

    @field_validator("symbols_unsupported_price", mode="before")
    @classmethod
    def _parse_symbols_unsupported_price(
        cls, value: str | list[str]
    ) -> list[str]:
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return value

    @field_validator("crypto_symbols", mode="before")
    @classmethod
    def _parse_crypto_symbols(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return value

    @field_validator("symbol_throttle_seconds")
    @classmethod
    def _validate_symbol_throttle(cls, value: float) -> float:
        if value < 0:
            raise ValueError(
                f"symbol_throttle_seconds must be non-negative (got {value})"
            )
        return value

    @field_validator(
        "volatility_threshold",
        "council_volatility_threshold",
        "crypto_volatility_threshold",
    )
    @classmethod
    def _validate_volatility_threshold(
        cls, value: Decimal, info: ValidationInfo
    ) -> Decimal:
        """Odrzuca ujemne progi volatility.

        Ujemny próg sprawia, że `abs(delta) >= threshold` jest ZAWSZE
        prawdziwe → bramka FinOps cicho znika i każdy płatny port (LLM,
        Alpha Vantage, embeddingi) odpala co cykl, bez błędu startu.
        `0` jest dozwolone — `council_volatility_threshold = 0.0` to
        udokumentowany "always run" disable switch.
        """
        if value < 0:
            raise ValueError(
                f"{info.field_name} must be non-negative (got {value})"
            )
        return value

    @field_validator("risk_symbol_types", mode="before")
    @classmethod
    def _parse_risk_symbol_types(
        cls,
        value: str | dict[str, MacroRiskInstrumentType] | dict[str, str],
    ) -> dict[str, MacroRiskInstrumentType]:
        """Pozwala podać RISK_SYMBOL_TYPES=SH:INVERSE_EQUITY,GLD:SAFE_HAVEN
        (CSV `SYM:TYPE` → dict[str, MacroRiskInstrumentType]).
        """
        if isinstance(value, dict):
            return {
                k: (v if isinstance(v, MacroRiskInstrumentType)
                    else MacroRiskInstrumentType(v))
                for k, v in value.items()
            }
        if not value:
            return {}
        out: dict[str, MacroRiskInstrumentType] = {}
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(
                    f"RISK_SYMBOL_TYPES entry '{chunk}' must be SYMBOL:TYPE"
                )
            sym, raw_type = chunk.split(":", 1)
            sym = sym.strip()
            raw_type = raw_type.strip()
            try:
                out[sym] = MacroRiskInstrumentType(raw_type)
            except ValueError as exc:
                raise ValueError(
                    f"RISK_SYMBOL_TYPES: '{raw_type}' is not a valid "
                    f"MacroRiskInstrumentType (allowed: "
                    f"{[t.value for t in MacroRiskInstrumentType]})"
                ) from exc
        return out

    @field_validator("alpha_vantage_api_keys", mode="before")
    @classmethod
    def _parse_av_keys(cls, value: str | list[str]) -> list[str]:
        """CSV → list[str], usuwa duplikaty zachowując kolejność."""
        if isinstance(value, str):
            raw = [k.strip() for k in value.split(",") if k.strip()]
        else:
            raw = list(value)
        # Dedup zachowując kolejność (Python 3.7+ dict)
        return list(dict.fromkeys(raw))

    @model_validator(mode="after")
    def _resolve_alpha_vantage_keys(self) -> Settings:
        """Backward-compat: jeśli ALPHA_VANTAGE_API_KEYS puste, próbujemy
        starego `ALPHA_VANTAGE_API_KEY` (single). Po resolucji wymagamy ≥1 klucza."""
        if not self.alpha_vantage_api_keys:
            legacy = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
            if legacy:
                object.__setattr__(self, "alpha_vantage_api_keys", [legacy])
        if not self.alpha_vantage_api_keys:
            raise ValueError(
                "alpha_vantage_api_keys is required — set ALPHA_VANTAGE_API_KEYS "
                "(comma-separated) or legacy ALPHA_VANTAGE_API_KEY in env."
            )
        return self

    @classmethod
    def from_env(cls) -> Settings:
        """Mypy-friendly entry point — pydantic-settings ładuje pola z env/.env.

        Bezpośrednie `Settings()` daje fałszywy alarm `Missing named argument`,
        bo mypy nie zna mechanizmu source loading w BaseSettings.
        """
        return cls()  # type: ignore[call-arg]
