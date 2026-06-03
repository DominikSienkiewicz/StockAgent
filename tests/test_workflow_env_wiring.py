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
}


def _run_agent_env() -> dict[str, object]:
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = spec["jobs"]["run-market-agent"]["steps"]
    run_step = next(s for s in steps if str(s.get("run", "")).endswith("main_agent.py"))
    return run_step.get("env", {})


def test_fast_loop_maps_crypto_config_into_env() -> None:
    env_keys = set(_run_agent_env().keys())
    missing = REQUIRED_ENV_KEYS - env_keys
    assert not missing, (
        f"fast_loop_12h.yml nie mapuje {sorted(missing)} do env kroku agenta — "
        f"te zmienne nie dotrą do procesu i Settings spadnie do defaultu "
        f"(np. brak CRYPTO_SYMBOLS = BTC/ETH gubione przed pętlą)."
    )
