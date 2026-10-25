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
from typing import Any

import yaml

from tests.settings_kwargs import NO_ENV_FILE

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
    env: dict[str, object] = run_step.get("env", {})
    return env


def test_all_settings_required_secrets_are_mapped() -> None:
    env_keys = set(_run_watch_env().keys())
    missing = REQUIRED_SECRET_KEYS - env_keys
    assert not missing, (
        f"shock_watch_hourly.yml nie mapuje sekretów {sorted(missing)} do env "
        f"kroku watcha — bez nich Settings() rzuca ValueError na starcie i cały "
        f"watch pada, choć nie woła żadnego płatnego portu."
    )


# BEZ-12: płatni dostawcy, których watch nigdy nie woła. Settings wymaga dwóch
# pierwszych kluczy do konstrukcji, więc w env stoją niepuste atrapy, nie sekrety.
PAID_PROVIDER_KEYS = ("OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEYS", "ANTHROPIC_API_KEY")


def test_paid_provider_keys_are_dummies_in_free_watch() -> None:
    env = _run_watch_env()

    real = [key for key in PAID_PROVIDER_KEYS if "secrets." in str(env.get(key, ""))]
    assert not real, (
        f"shock_watch_hourly.yml podaje darmowemu watchowi prawdziwe płatne klucze {real} "
        f"— wystarczy atrapa, watch ich nie woła."
    )
    assert all(str(env.get(key, "")).strip() for key in PAID_PROVIDER_KEYS[:2])


def _schedule(workflow: str) -> object:
    spec = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True.
    triggers = spec.get("on", spec.get(True)) or {}
    return triggers.get("schedule")


def _wired_daily_change_source(mocker: Any) -> object:
    """Źródło dziennej zmiany, które `main_watch.main()` składa z PRAWDZIWEJ fabryki
    ceny (`main_agent.build_market_port`); baza, push i sam skan są atrapami."""
    import main_watch
    from src.config import Settings

    settings = Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL"],
        crypto_symbols=["BTC"],
        shock_alerts_enabled=True,
        telegram_bot_token="tg-token",
        telegram_chat_id="42",
    )
    mocker.patch("main_agent.build_repository")
    mocker.patch("main_agent.build_push_notifier")
    scan_watch = mocker.patch(
        "main_watch.scan_watch", return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1)
    )

    main_watch.main(settings)

    return scan_watch.call_args.kwargs.get("daily_change")


def test_watch_cron_without_the_fast_loop_cron_needs_its_daily_change_source(mocker: Any) -> None:
    # ALR-02: watch ma własne źródło ruchu — dzienną zmianę z darmowego /quote —
    # więc cron watcha może chodzić bez Fast Loopa. Bez tego źródła jedynymi
    # referencjami są snapshoty Fast Loopa i cron dawałby co godzinę „0 alertów".
    from src.application.ports import DailyChangePort

    if _schedule("shock_watch_hourly.yml") and not _schedule("fast_loop_12h.yml"):
        assert isinstance(_wired_daily_change_source(mocker), DailyChangePort), (
            "shock_watch_hourly.yml ma aktywny cron bez fast_loop_12h.yml, a main_watch "
            "nie składa źródła dziennej zmiany — watch byłby ślepy."
        )


def test_watch_wires_the_free_price_router_as_its_daily_change_source(mocker: Any) -> None:
    # Strażnik wyżej sprawdza okablowanie tylko przy aktywnym cronie — ten test pilnuje,
    # że to okablowanie istnieje już teraz, zanim właściciel odkomentuje cron.
    from src.application.ports import DailyChangePort
    from src.infrastructure.adapters.routing_market_data import RoutingMarketDataPort

    source = _wired_daily_change_source(mocker)

    assert isinstance(source, DailyChangePort)
    assert isinstance(source, RoutingMarketDataPort)


def test_watch_starts_without_paid_provider_keys(mocker: Any) -> None:
    # ARCH-05: Settings nie żąda kluczy płatnych dostawców, więc watch startuje bez
    # OPENAI_API_KEY i ALPHA_VANTAGE_API_KEYS — atrapy w workflow są zbędne.
    import main_watch
    from src.config import Settings

    settings = Settings(
        **NO_ENV_FILE,
        finnhub_api_key="fh",
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL"],
        shock_alerts_enabled=True,
        telegram_bot_token="tg-token",
        telegram_chat_id="42",
    )
    mocker.patch("main_agent.build_repository")
    mocker.patch("main_agent.build_push_notifier")
    scan_watch = mocker.patch(
        "main_watch.scan_watch", return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1)
    )

    main_watch.main(settings)

    scan_watch.assert_called_once()
