from __future__ import annotations

import os
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Konfiguracja runtime — wczytywana z `.env` i zmiennych środowiskowych.

    Wszystkie tajemnice (klucze API, DB credentials) trzymane są tutaj.
    Klucze nieobowiązkowe mają wartości domyślne. Wymagane (bez default)
    rzucają `ValidationError` przy starcie, jeśli brakuje ich w env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- LLM -----
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str
    anthropic_api_key: str | None = None

    # ----- Market / sentiment / news -----
    finnhub_api_key: str
    # Alpha Vantage akceptuje CSV — wiele kluczy rotujemy, gdy jeden wyczerpie
    # dzienny limit (25 req/dobę free). Wpisz w .env: ALPHA_VANTAGE_API_KEYS=k1,k2,k3
    alpha_vantage_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Stare klucze pozostawione jako opcjonalne — kod ich już nie używa,
    # ale można je trzymać w .env na potrzeby fallback / eksperymentów.
    lunarcrush_api_key: str | None = None
    news_api_key: str | None = None

    # ----- Database -----
    supabase_url: str
    supabase_key: str

    # ----- Agent config -----
    volatility_threshold: Decimal = Field(default=Decimal("0.02"))
    # NoDecode wyłącza próbę JSON-parse'owania env var — używamy własnego validatora CSV.
    symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["AAPL", "VOO", "CSPX.L"]
    )

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
