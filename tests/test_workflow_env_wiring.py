"""Regression guard — po centralizacji configu do `config.toml` workflow
wstrzykuje WYŁĄCZNIE sekrety; config niewrażliwy czytany jest wprost z pliku.

Historia: GitHub Actions nie eksportuje `vars.*`/`secrets.*` automatycznie —
trzeba je jawnie wymienić w `env:` kroku. Wcześniej trzymaliśmy tam też config
niewrażliwy (SYMBOLS, CRYPTO_SYMBOLS, COUNCIL_*…), co rodziło drift
`.env` ↔ `.env.example` ↔ workflow. Teraz ten config żyje w `config.toml`
(czytany przez `Settings`), więc workflow ma zostać czysty od niego.

Ten test pilnuje obu stron:
1. wszystkie SEKRETY są zmapowane (inaczej brak klucza w procesie → crash),
2. żaden config niewrażliwy nie wrócił do workflow (inaczej znów drift).
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "fast_loop_12h.yml"

# Sekrety, które MUSZĄ trafić do env kroku agenta (config.toml ich nie zawiera).
REQUIRED_SECRET_KEYS = {
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEYS",
    "RESEND_API_KEY",
    "DIGEST_TO_EMAIL",
}

# Config niewrażliwy — należy do config.toml, NIE do workflow. Gdyby któryś tu
# wrócił, drift wraca razem z nim.
FORBIDDEN_NONSECRET_KEYS = {
    "SYMBOLS",
    "SYMBOLS_ETF",
    "CRYPTO_SYMBOLS",
    "CRYPTO_VOLATILITY_THRESHOLD",
    "VOLATILITY_THRESHOLD",
    "COUNCIL_VOLATILITY_THRESHOLD",
    "COUNCIL_LLM_PROVIDER",
    "COUNCIL_LLM_MODEL",
    "LLM_PROVIDER",
    "SYMBOL_THROTTLE_SECONDS",
    "SYMBOLS_UNSUPPORTED_PRICE",
    "RISK_SYMBOLS",
    "RISK_SYMBOL_TYPES",
    "NBP_ENABLED",
    "NOTIFICATIONS_ENABLED",
    "DIGEST_FROM_EMAIL",
    "ML_MODEL_PATH",
}


def _steps() -> list[dict[str, object]]:
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return spec["jobs"]["run-market-agent"]["steps"]


def _run_agent_env() -> dict[str, object]:
    run_step = next(s for s in _steps() if str(s.get("run", "")).endswith("main_agent.py"))
    return run_step.get("env", {})


def test_all_required_secrets_are_mapped() -> None:
    env_keys = set(_run_agent_env().keys())
    missing = REQUIRED_SECRET_KEYS - env_keys
    assert not missing, (
        f"fast_loop_12h.yml nie mapuje sekretów {sorted(missing)} do env "
        f"kroku agenta — bez nich proces dostanie pusty klucz i padnie."
    )


def test_no_non_secret_config_leaks_into_workflow() -> None:
    env_keys = set(_run_agent_env().keys())
    leaked = FORBIDDEN_NONSECRET_KEYS & env_keys
    assert not leaked, (
        f"Config niewrażliwy {sorted(leaked)} wrócił do workflow — miejsce na "
        f"to jest w config.toml (single source of truth), inaczej drift wraca."
    )


def test_config_toml_exists() -> None:
    assert (ROOT / "config.toml").is_file(), "brak config.toml w repo"


def test_dependency_sync_installs_anthropic_extra() -> None:
    sync_cmds = [
        str(s.get("run", "")) for s in _steps() if "uv sync" in str(s.get("run", ""))
    ]
    assert sync_cmds, "brak kroku `uv sync` w workflow"
    assert any("--extra anthropic" in cmd for cmd in sync_cmds), (
        "krok `uv sync` musi instalować `--extra anthropic` (LLM_PROVIDER=anthropic)."
    )
