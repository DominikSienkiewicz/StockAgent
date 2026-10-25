"""Strażnik `deadman_daily.yml` (NIEZ-11, NIEZ-28): dzienny deadman i monitor bazy
Supabase są darmowe z konstrukcji.

Workflow czyta bazę (ślady życia jobów i jeden wiersz `price_snapshots` jako dowód,
że projekt Free nie został wstrzymany) i mailuje przez Resend. Płatne klucze nie
mają tu czego szukać: `Settings` wymaga ich pól, więc dostają jawne zaślepki,
nigdy `${{ secrets.* }}`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "deadman_daily.yml"

PAID_SECRET_KEYS = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEYS",
}
ALLOWED_SECRET_KEYS = {"SUPABASE_URL", "SUPABASE_KEY", "RESEND_API_KEY", "DIGEST_TO_EMAIL"}


def _steps() -> list[dict[str, object]]:
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps: list[dict[str, object]] = spec["jobs"]["check-liveness"]["steps"]
    return steps


def _deadman_env() -> dict[str, str]:
    step = next(s for s in _steps() if str(s.get("run", "")).endswith("main_deadman.py"))
    env = step.get("env", {})
    assert isinstance(env, dict)
    return {str(key): str(value) for key, value in env.items()}


def test_deadman_maps_no_paid_secret() -> None:
    env = _deadman_env()
    leaked = sorted(key for key in PAID_SECRET_KEYS if "secrets." in env.get(key, ""))
    assert not leaked, f"deadman_daily.yml mapuje płatne sekrety {leaked} — deadman jest darmowy"


def test_deadman_reads_only_the_free_secrets() -> None:
    from_secrets = {key for key, value in _deadman_env().items() if "secrets." in value}
    assert from_secrets == ALLOWED_SECRET_KEYS


def test_deadman_still_builds_settings_with_placeholder_paid_keys() -> None:
    # Settings wymaga pól płatnych kluczy do samej konstrukcji — bez nich deadman
    # padłby na starcie, zanim cokolwiek sprawdzi.
    env = _deadman_env()
    for key in ("OPENAI_API_KEY", "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEYS"):
        assert env.get(key), f"brak zaślepki {key} w kroku deadmana"
