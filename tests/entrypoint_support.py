"""Pomocniki wspólne dla testów entry pointów: test_main_agent, test_main_trainer,
test_main_backtest."""

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src.config import Settings
from src.domain.finops import LLMUsage
from src.domain.quota import QuotaAlert, QuotaSeverity

# Konstruktor przez `Callable` z czasów, gdy mypy nie znał kwargs pydantic-settings
# (`_env_file`); od ARCH-22 plugin `pydantic.mypy` je zna, a fabryka zostaje bez zmian.
_SETTINGS_FACTORY: Callable[..., Settings] = Settings

_BASE_SETTINGS: dict[str, Any] = {
    "openai_api_key": "sk-test",
    "finnhub_api_key": "fh",
    "alpha_vantage_api_keys": ["av1", "av2"],
    "supabase_url": "https://test.supabase.co",
    "supabase_key": "anon",
    "symbols": ["AAPL", "VOO"],
    "volatility_threshold": Decimal("0.02"),
    "ml_model_path": "data/models/missing.ubj",
    "db_writes_enabled": True,  # BEZ-08: smoke testy odwzorowują workflowy GHA
    "paid_calls_enabled": True,  # FIN-35: jak wyżej — workflowy GHA płacą jawnie
}


def isolated_settings(**values: Any) -> Settings:
    """`Settings(_env_file=None, **values)`: bez `.env`, ale z env i TOML jak w produkcji.

    `_env_file=None` odcina pydantic-settings od `.env` (developerzy mają w nim
    klucze, risk_symbols itp. — testy muszą być deterministyczne)."""
    return _SETTINGS_FACTORY(_env_file=None, **values)


def build_settings(**overrides: Any) -> Settings:
    """Prawdziwe `Settings` testów entry pointów: workflowy GHA plus nadpisania.

    Nieznana nazwa pola jest błędem, bo `Settings` ignoruje nadmiarowe kwargs
    (`extra="ignore"`) i literówka w fladze po cichu zostawiałaby domyślną wartość."""
    unknown = overrides.keys() - Settings.model_fields.keys()
    if unknown:
        raise AttributeError(f"Settings has no field(s): {sorted(unknown)}")
    return isolated_settings(**{**_BASE_SETTINGS, **overrides})


def build_quota_alert(source: str = "Resend") -> QuotaAlert:
    return QuotaAlert(
        source=source,
        severity=QuotaSeverity.CRITICAL,
        message=f"{source} returned 503",
        action="sprawdź dostawcę",
        occurred_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
    )


def build_usage(model: str = "claude-sonnet-4-6") -> LLMUsage:
    return LLMUsage(provider="anthropic", model=model, input_tokens=1000, output_tokens=200)


UNDELIVERED_ANNOTATION = "::error title=StockAgent::Raport nie został dostarczony"
