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

import tomllib
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


# Flagi, które obiecują PUBLICZNY artefakt (commitmenty/revealy, scorecard).
# Adaptery piszą wyłącznie na dysk runnera, który znika po jobie — bez kroku
# publikacji włączona flaga to twierdzenie bez pokrycia.
PUBLIC_ARTIFACT_FLAGS = ("attestation_enabled", "public_scorecard_enabled")


def _agent_workflows() -> dict[str, list[dict[str, object]]]:
    """Kroki wszystkich jobów, które uruchamiają `main_agent.py` (per workflow)."""
    found: dict[str, list[dict[str, object]]] = {}
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in spec.get("jobs", {}).values():
            steps = job.get("steps", [])
            if any("main_agent.py" in str(s.get("run", "")) for s in steps):
                found.setdefault(path.name, []).extend(steps)
    return found


def _has_durable_publish_step(steps: list[dict[str, object]]) -> bool:
    return any(
        str(s.get("uses", "")).startswith("actions/deploy-pages")
        or "git push" in str(s.get("run", ""))
        for s in steps
    )


def test_enabled_public_artifacts_have_a_durable_publish_step() -> None:
    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    enabled = [flag for flag in PUBLIC_ARTIFACT_FLAGS if config.get(flag) is True]
    workflows = _agent_workflows()
    assert workflows, "żaden workflow nie uruchamia main_agent.py"

    unpublished = [
        name for name, steps in workflows.items() if not _has_durable_publish_step(steps)
    ]
    assert not (enabled and unpublished), (
        f"config.toml włącza {enabled}, ale {unpublished} nie ma kroku "
        "`actions/deploy-pages` ani `git push` — artefakt ginie z dyskiem runnera."
    )


def _slow_loop_trainer_env() -> dict[str, object]:
    workflow = ROOT / ".github" / "workflows" / "slow_loop_weekly.yml"
    spec = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    steps = [s for job in spec["jobs"].values() for s in job.get("steps", [])]
    run_step = next(s for s in steps if str(s.get("run", "")).endswith("main_trainer.py"))
    return run_step.get("env", {})


def test_slow_loop_maps_real_alpha_vantage_secret() -> None:
    # DAT-04: Slow Loop odświeża fundamenty z Alpha Vantage. Z atrapą klucza AV
    # odrzucał każde żądanie jako wyczerpany limit i `fundamentals_cache` zostawał pusty.
    expected = "${{ secrets.ALPHA_VANTAGE_API_KEYS }}"
    assert _slow_loop_trainer_env().get("ALPHA_VANTAGE_API_KEYS") == expected
