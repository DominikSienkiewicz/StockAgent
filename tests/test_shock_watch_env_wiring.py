"""Regression guard — `shock_watch_hourly.yml` musi wstrzyknąć KOMPLET sekretów,
których `Settings()` wymaga do samej KONSTRUKCJI.

Historia: watch (#11) nie woła żadnego płatnego portu, ale `Settings` jest jednym
wspólnym obiektem configu z twardo wymaganymi sekretami — `openai_api_key` (pole
`str`) ORAZ `alpha_vantage_api_keys` (walidator `_resolve_alpha_vantage_keys`).
Workflow wstrzykiwał `OPENAI_API_KEY`, ale gubił `ALPHA_VANTAGE_API_KEYS`, więc
`Settings()` rzucał `ValueError` zanim pętla cokolwiek zrobiła — cały watch padał
na starcie w CI.

Ten test pilnuje, że run-step watcha mapuje wszystkie sekrety konieczne do
zbudowania `Settings`, żeby regres nie wrócił.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "shock_watch_hourly.yml"

# Sekrety, bez których `Settings()` nie da się skonstruować (pola `str` bez
# defaultu + walidator wymagający ≥1 klucza Alpha Vantage). Watch ich nie WOŁA,
# ale musi je mieć w env, inaczej konstrukcja Settings rzuca.
REQUIRED_SECRET_KEYS = {
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "FINNHUB_API_KEY",
    "OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEYS",
}


def _run_watch_env() -> dict[str, object]:
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = spec["jobs"]["watch-market-shock"]["steps"]
    run_step = next(
        s for s in steps if str(s.get("run", "")).endswith("main_watch.py")
    )
    return run_step.get("env", {})


def test_all_settings_required_secrets_are_mapped() -> None:
    env_keys = set(_run_watch_env().keys())
    missing = REQUIRED_SECRET_KEYS - env_keys
    assert not missing, (
        f"shock_watch_hourly.yml nie mapuje sekretów {sorted(missing)} do env "
        f"kroku watcha — bez nich Settings() rzuca ValueError na starcie i cały "
        f"watch pada, choć nie woła żadnego płatnego portu."
    )
