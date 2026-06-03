"""Regression guard — workflow `env:` musi mapować config potrzebny w produkcji.

Bug (2026-06-03): raport cyklu nie zawierał BTC/ETH, mimo że `CRYPTO_SYMBOLS`
było ustawione w `.env`. Przyczyna: GitHub Actions NIE eksportuje automatycznie
`vars.*` / `secrets.*` do procesu — zmienna trafia do agenta tylko, jeśli jest
jawnie wymieniona w bloku `env:` kroku. `fast_loop_12h.yml` mapował `SYMBOLS`,
ale pomijał `CRYPTO_SYMBOLS`, więc `Settings.crypto_symbols` spadało do pustego
defaultu i krypto było odsiewane PRZED pętlą.

Ten test pilnuje, by config sterujący doborem symboli był podpięty w workflow —
inaczej dryf `.env` ↔ workflow po cichu gubi całe klasy aktywów.
"""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "fast_loop_12h.yml"

# Config niewrażliwy, który steruje TYM, jakie symbole i progi widzi Fast Loop.
# Brak któregokolwiek w workflow = produkcja leci na cichym defaultcie Settings.
REQUIRED_ENV_KEYS = {
    "SYMBOLS",
    "CRYPTO_SYMBOLS",
    "CRYPTO_VOLATILITY_THRESHOLD",
    # FinOps: bez nich rada leci na drogim gpt-4o bez throttle → alerty TPM.
    # GHA nie eksportuje vars.* automatycznie; musi być jawne w env: kroku.
    "COUNCIL_LLM_MODEL",
    "SYMBOL_THROTTLE_SECONDS",
    "COUNCIL_VOLATILITY_THRESHOLD",
    # Bez tego tickery niewspierane przez Finnhub free (EU dot-notation, OTC)
    # lecą jako BŁĘDY zamiast jako "pominięte/ignored".
    "SYMBOLS_UNSUPPORTED_PRICE",
    # Analiza główna na Anthropic (Claude). Bez tych kluczy w prod Settings
    # spada do LLM_PROVIDER=openai / brak ANTHROPIC_API_KEY, a rada bez jawnego
    # COUNCIL_LLM_PROVIDER=openai poszłaby na Anthropic z modelem OpenAI → crash.
    "LLM_PROVIDER",
    "ANTHROPIC_API_KEY",
    "COUNCIL_LLM_PROVIDER",
}


def _steps() -> list[dict[str, object]]:
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return spec["jobs"]["run-market-agent"]["steps"]


def _run_agent_env() -> dict[str, object]:
    run_step = next(s for s in _steps() if str(s.get("run", "")).endswith("main_agent.py"))
    return run_step.get("env", {})


def test_fast_loop_maps_crypto_config_into_env() -> None:
    env_keys = set(_run_agent_env().keys())
    missing = REQUIRED_ENV_KEYS - env_keys
    assert not missing, (
        f"fast_loop_12h.yml nie mapuje {sorted(missing)} do env kroku agenta — "
        f"te zmienne nie dotrą do procesu i Settings spadnie do defaultu "
        f"(np. brak CRYPTO_SYMBOLS = BTC/ETH gubione przed pętlą)."
    )


def test_dependency_sync_installs_anthropic_extra() -> None:
    # Analiza główna na Claude wymaga SDK anthropic, które jest optional-dependency.
    # Bez `--extra anthropic` w `uv sync` import w prod się wywali.
    sync_cmds = [
        str(s.get("run", "")) for s in _steps() if "uv sync" in str(s.get("run", ""))
    ]
    assert sync_cmds, "brak kroku `uv sync` w workflow"
    assert any("--extra anthropic" in cmd for cmd in sync_cmds), (
        "krok `uv sync` musi instalować `--extra anthropic` (LLM_PROVIDER=anthropic)."
    )
