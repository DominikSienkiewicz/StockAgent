"""Testy źródła config.toml — single source of truth dla configu niewrażliwego.

Te testy ZDEJMUJĄ globalną flagę `STOCKAGENT_DISABLE_TOML` (ustawioną w conftest),
bo celowo weryfikują, że źródło TOML działa i wpina wartości do `Settings`.
"""

from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path
from typing import get_args

import pytest

from src.config import Settings

CONFIG_TOML = Path(__file__).parent.parent / "config.toml"
# Pierwszy cykl Fast Loopa po wznowieniu crona 2026-10-23 (poniedziałek).
RESUME_DAY = date(2026, 10, 26)
EXPERIMENTS_END = date(2027, 1, 29)
# TEST-21 (G79): dozwolony zbiór providerów bierzemy z kontraktu `Settings`
# (Literal), nie z bieżącej konfiguracji produkcji.
_LLM_PROVIDERS = frozenset(get_args(Settings.model_fields["llm_provider"].annotation))

_REQUIRED_SECRETS = {
    "openai_api_key": "sk-test",
    "finnhub_api_key": "fh",
    "alpha_vantage_api_keys": ["av1"],
    "supabase_url": "https://t.supabase.co",
    "supabase_key": "anon",
}


def _model_provider(model: str) -> str:
    """Rodzina modelu po nazwie — tylko na potrzeby niezmiennika spójności."""
    return "anthropic" if model.startswith("claude") else "openai"


def _assert_unique_tickers(name: str, values: list[str]) -> None:
    assert all(isinstance(v, str) and v.strip() for v in values), f"pusty ticker w {name}"
    assert len(values) == len(set(values)), f"duplikat w {name}"


def test_real_config_toml_parses_and_has_expected_invariants() -> None:
    # TEST-21 (G79): niezmienniki struktury i spójności, nie dane produkcji —
    # dopisanie tickera albo zmiana modelu nie może czerwienić tego testu.
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["llm_provider"] in _LLM_PROVIDERS
    assert data["council_llm_provider"] in _LLM_PROVIDERS
    # Model rady musi należeć do providera rady — inaczej rada dziedziczy
    # providera z innym modelem i cykl pada (komentarz w config.toml).
    council_model = data["council_llm_model"]
    assert isinstance(council_model, str) and council_model.strip()
    assert _model_provider(council_model) == data["council_llm_provider"]
    assert len(data["symbols"]) > 0
    for name in ("symbols", "symbols_etf", "symbols_unsupported_price", "crypto_symbols"):
        _assert_unique_tickers(name, data[name])
    # Krypto idzie osobnym routingiem (CoinGecko) i jest doklejane do `symbols`
    # w cyklu — symbol w obu listach byłby analizowany dwa razy.
    crypto = set(data["crypto_symbols"])
    assert not crypto & set(data["symbols"]), "krypto także w symbols"
    assert not crypto & set(data["symbols_etf"]), "krypto także w symbols_etf"
    assert isinstance(data["symbol_concurrency"], int)
    assert data["symbol_concurrency"] >= 1
    # Sekrety NIE mogą wyciec do commitowanego pliku.
    for secret in ("openai_api_key", "anthropic_api_key", "supabase_key",
                   "finnhub_api_key", "digest_to_email"):
        assert secret not in data, f"sekret {secret} w config.toml!"


def test_toml_cross_list_invariants() -> None:
    """DAT-27: listy klasyfikujące symbole (ETF, brak ceny) opisują watchlistę.
    Ticker spoza `symbols` to martwa konfiguracja — tak zniknęła historia VOO."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    symbols = set(data["symbols"])

    assert sorted(set(data["symbols_etf"]) - symbols) == []
    assert sorted(set(data["symbols_unsupported_price"]) - symbols) == []


def test_real_config_toml_bounds_reflection_settlement_age() -> None:
    """WIA-04: produkcja rozlicza predykcje w oknie 6 h – 96 h."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["reflection_min_age_hours"] == 6
    assert data["reflection_max_age_hours"] == 96


def test_settings_reads_non_secret_from_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    toml = tmp_path / "config.toml"
    toml.write_text(
        'symbol_throttle_seconds = 7.5\nllm_provider = "anthropic"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("STOCKAGENT_TOML_FILE", str(toml))
    # Sekrety wstrzykujemy jako kwargs (wygrywają nad TOML); env odcięte.
    s = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]
    assert s.symbol_throttle_seconds == pytest.approx(7.5)
    assert s.llm_provider == "anthropic"


def test_config_toml_found_regardless_of_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Config.toml musi być znajdowany niezależnie od CWD — inaczej agent
    odpalony spoza roota repo dostaje DEFAULTY (np. notifications_enabled=False,
    symbols=[AAPL,MSFT,NVDA]) i po cichu nie wysyła maila / gubi portfolio."""
    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    monkeypatch.delenv("STOCKAGENT_TOML_FILE", raising=False)
    monkeypatch.chdir(tmp_path)  # CWD != root repo
    s = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert s.symbols == data["symbols"], "config.toml nie został znaleziony spoza roota"
    assert s.notifications_enabled is data["notifications_enabled"]


def test_unknown_toml_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TEST-03 (G64): literówka w kluczu `config.toml` przerywa start, zamiast po
    cichu zostawić default (tu: `weekly_recap_enabled` zostałby wyłączony)."""
    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    toml = tmp_path / "config.toml"
    toml.write_text("weekly_recap_enable = true\n", encoding="utf-8")
    monkeypatch.setenv("STOCKAGENT_TOML_FILE", str(toml))

    with pytest.raises(ValueError, match="weekly_recap_enable"):
        Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]


def test_unknown_env_variable_is_still_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """TEST-03: ścisłe jest tylko źródło TOML — env i `.env` niosą klucze innych
    narzędzi (SONAR_TOKEN, GH_TOKEN), więc `extra = "ignore"` dla nich zostaje."""
    monkeypatch.setenv("SONAR_TOKEN", "not-a-setting")

    settings = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]

    assert not hasattr(settings, "sonar_token")


def test_every_real_config_toml_key_is_a_settings_field() -> None:
    """TEST-03: produkcyjny `config.toml` nie ma klucza spoza `Settings`."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))

    assert set(data) - set(Settings.model_fields) == set()


def test_env_overrides_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    toml = tmp_path / "config.toml"
    toml.write_text("symbol_throttle_seconds = 7.5\n", encoding="utf-8")
    monkeypatch.setenv("STOCKAGENT_TOML_FILE", str(toml))
    monkeypatch.setenv("SYMBOL_THROTTLE_SECONDS", "3.0")
    s = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]
    # env > toml
    assert s.symbol_throttle_seconds == pytest.approx(3.0)


def _comment_above(lines: list[str], index: int) -> str:
    """Ostatnia linia bloku komentarza nad kluczem `lines[index]`.

    Przeskakuje sąsiednie przypisania tej samej grupy (np. trzy flagi pod
    jednym komentarzem); pusta linia kończy grupę.
    """
    for line in reversed(lines[:index]):
        stripped = line.strip()
        if not stripped:
            return ""
        if stripped.startswith("#"):
            return stripped
    return ""


def test_no_flag_set_to_true_sits_under_an_off_comment() -> None:
    # „Off." nad `= true` to config, który kłamie o sobie — a ten plik jest
    # deklarowanym single source of truth.
    lines = CONFIG_TOML.read_text(encoding="utf-8").splitlines()
    contradicted = [
        line.split("=", 1)[0].strip()
        for index, line in enumerate(lines)
        if line.replace(" ", "").endswith("=true")
        and _comment_above(lines, index).endswith("Off.")
    ]
    assert not contradicted, f"flagi = true pod komentarzem „Off.”: {contradicted}"


def test_real_config_toml_shows_times_in_warsaw() -> None:
    """MAIL-14: temat, nagłówek i push w czasie lokalnym odbiorcy."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["display_timezone"] == "Europe/Warsaw"


def test_real_config_toml_tags_predictions_with_the_regime() -> None:
    """ML-38: produkcja nie nadpisuje `regime_tagging_enabled`, więc tag reżimu
    dziedziczy włączony `regime_aware_enabled` (migracja 017 jest na produkcji)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["regime_aware_enabled"] is True
    assert "regime_tagging_enabled" not in data


def test_real_config_toml_ships_regime_hysteresis_on_after_migration() -> None:
    """ALR-03: migracja 033 (`regime_history`) jest na produkcji od 2026-10-23."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["regime_hysteresis_enabled"] is True
    assert data["regime_exit_confirmations"] == 2


def test_real_config_toml_sectors_hold_only_watchlist_stocks() -> None:
    """PORT-05: sektor to grupa AKCJI z watchlisty — ETF i krypto nie mają
    sektora, a symbol spoza `symbols` nie dostaje ceny w cyklu. Symbol w dwóch
    sektorach dawałby niejednoznaczny klaster w sizingu."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    sectors: dict[str, list[str]] = data["sectors"]
    assert sectors
    members = [symbol for symbols in sectors.values() for symbol in symbols]
    assert all(symbols for symbols in sectors.values()), "pusty sektor"
    assert len(members) == len(set(members)), "symbol w dwóch sektorach"
    assert set(members) <= set(data["symbols"])
    assert not set(members) & set(data["symbols_etf"])
    assert not set(members) & set(data["crypto_symbols"])


def test_real_config_toml_keeps_contagion_off() -> None:
    """PORT-05: `peer_groups` z sektorów zasila też contagion (werdykty peerów
    do promptu). Ten zostaje wyłączony — sektory mają działać tylko w sizingu
    i ostrzeżeniu o koncentracji."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["contagion_enabled"] is False


def test_settings_derive_peer_groups_from_config_sectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    monkeypatch.delenv("STOCKAGENT_TOML_FILE", raising=False)
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))

    s = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]

    assert s.sectors == data["sectors"]
    assert s.peer_groups == data["sectors"]
    assert s.contagion_enabled is False


def test_real_config_toml_peers_are_keyed_by_its_sectors() -> None:
    """PORT-05: peerzy sekcji 💡 Warte uwagi są kluczowani nazwą sektora z
    `sectors` — zmiana nazwy sektora bez peerów nie może ich wyłączyć po cichu."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    peers: dict[str, list[str]] = data["sector_peers"]
    assert peers
    assert set(peers) <= set(data["sectors"])
    assert all(candidates for candidates in peers.values()), "sektor bez peerów"


def test_real_config_toml_sets_the_paper_pnl_costs() -> None:
    """WIA-50 (decyzja 2026-10-19): 10 bps dla akcji/ETF, 25 bps dla krypto."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["paper_cost_bps"] == 10
    assert data["crypto_paper_cost_bps"] == 25


def test_real_config_toml_ships_the_experiment_arms_column_on_after_migration() -> None:
    """EXP-05: kolumna `experiment_arms` (migracja 037) jest na produkcji od 2026-10-23."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["experiment_arms_enabled"] is True


def test_real_config_toml_runs_the_blind_council_experiment_from_the_resume() -> None:
    """EXP-02: aktywny od pierwszego cyklu po wznowieniu Fast Loopa."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    experiment = data["experiments"]["EXP-02"]
    assert experiment["arms"] == ["sighted", "blind"]
    assert experiment["primary_metric"] == "council_hit_rate"
    assert (experiment["start"], experiment["end"]) == (RESUME_DAY, EXPERIMENTS_END)
    assert experiment["enabled"] is True


def test_real_config_toml_runs_the_memory_experiment_from_the_resume() -> None:
    """EXP-03: aktywny od pierwszego cyklu po wznowieniu Fast Loopa."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    experiment = data["experiments"]["EXP-03"]
    assert experiment["arms"] == ["memory", "no_memory"]
    assert experiment["primary_metric"] == "hit_rate"
    assert (experiment["start"], experiment["end"]) == (RESUME_DAY, EXPERIMENTS_END)
    assert experiment["enabled"] is True


# --- FIN-32: kadencja Fast Loopa w budżecie AV = harmonogram workflow ----------

FAST_LOOP_WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "fast_loop_12h.yml"


def _cron_field_values(field: str, lowest: int, highest: int) -> set[int]:
    """Wartości jednego pola crona (`*`, listy, zakresy, kroki) w [lowest, highest]."""
    values: set[int] = set()
    for part in field.split(","):
        base, _, step = part.partition("/")
        if base == "*":
            start, end = lowest, highest
        elif "-" in base:
            start, end = (int(bound) for bound in base.split("-"))
        else:
            start = int(base)
            end = highest if step else start
        values.update(range(start, end + 1, int(step) if step else 1))
    return values


def _cron_runs_per_day(cron: str) -> int:
    """Uruchomienia w dniu, w którym cron w ogóle startuje (dzień tygodnia nie
    zmienia kadencji doby, tylko to, które doby są aktywne)."""
    minute, hour, *_ = cron.split()
    return len(_cron_field_values(minute, 0, 59)) * len(_cron_field_values(hour, 0, 23))


def _scheduled_fast_loop_runs_per_day() -> int:
    import yaml

    spec = yaml.safe_load(FAST_LOOP_WORKFLOW.read_text(encoding="utf-8"))
    # YAML 1.1: klucz `on` parsuje się jako True.
    triggers = spec.get("on", spec.get(True)) or {}
    schedule = triggers.get("schedule") if isinstance(triggers, dict) else None
    if not schedule:
        # Cron zakomentowany, start tylko ręczny: budżet liczy jeden cykl na dobę.
        return 1
    return sum(_cron_runs_per_day(entry["cron"]) for entry in schedule)


@pytest.mark.parametrize(
    ("cron", "runs"),
    [("30 5 * * 1-5", 1), ("30 5,17 * * 1-5", 2), ("0 */6 * * *", 4), ("*/30 9-10 * * *", 4)],
)
def test_cron_runs_per_day_counts_minutes_times_hours(cron: str, runs: int) -> None:
    assert _cron_runs_per_day(cron) == runs


def test_fast_loop_cycles_per_day_matches_the_workflow_schedule() -> None:
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))

    assert data["fast_loop_cycles_per_day"] == _scheduled_fast_loop_runs_per_day()


def test_real_config_toml_caps_paid_llm_calls_per_cycle() -> None:
    """FIN-12 (decyzja 2026-10-20): sufit 100 płatnych wywołań LLM na cykl."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["max_paid_llm_calls_per_cycle"] == 100


def test_real_config_toml_caps_the_slow_loop_llm_week() -> None:
    """FIN-30 (decyzja 2026-10-20): Slow Loop ma 15 płatnych wywołań LLM na tydzień."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["slow_loop_max_paid_llm_calls_per_week"] == 15


def test_real_config_toml_skips_stocks_on_closed_nyse_days() -> None:
    """FIN-34: produkcja nie analizuje akcji w święto NYSE (cron `1-5` bez zmian)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["nyse_closure_skip_enabled"] is True


def test_real_config_toml_ships_the_cost_ledger_on_after_migration() -> None:
    """FIN-13: migracja 040 (`llm_calls`) jest na produkcji od 2026-10-23."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["cost_ledger_enabled"] is True


def test_real_config_toml_prices_only_configured_models() -> None:
    """FIN-13 (decyzja 2026-10-20): cennik standard tier z oficjalnych stron
    Anthropic i OpenAI, USD za 1M tokenów, z datą `llm_pricing_as_of`. Wpisane są
    tylko modele, które repo faktycznie konfiguruje (config.toml + domyślne
    adapterów); model bez stawki zostaje na stałej `UNIT_COST_USD`, bez zgadywania."""
    from src.infrastructure.llm.anthropic_client import DEFAULT_MODEL as ANTHROPIC_DEFAULT
    from src.infrastructure.llm.openai_client import DEFAULT_MODEL as OPENAI_DEFAULT
    from src.infrastructure.llm.openai_embeddings import DEFAULT_MODEL as EMBEDDING_DEFAULT
    from src.infrastructure.llm.tool_calling_anthropic import (
        DEFAULT_MODEL as TOOL_USE_ANTHROPIC_DEFAULT,
    )
    from src.infrastructure.llm.tool_calling_openai import (
        DEFAULT_MODEL as TOOL_USE_OPENAI_DEFAULT,
    )

    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    configured = {
        data["council_llm_model"],
        data["router_cheap_model"],
        data["router_frontier_model"],
        ANTHROPIC_DEFAULT,
        OPENAI_DEFAULT,
        EMBEDDING_DEFAULT,
        TOOL_USE_ANTHROPIC_DEFAULT,
        TOOL_USE_OPENAI_DEFAULT,
    }

    assert data["llm_pricing_as_of"] == "2026-10-20"
    assert set(data["llm_pricing"]) <= configured
    assert data["llm_pricing"] == {
        "claude-sonnet-4-6": {
            "input": 3.00,
            "cache_read": 0.30,
            "cache_write": 3.75,
            "output": 15.00,
        },
        "gpt-5-mini": {"input": 0.25, "cache_read": 0.025, "output": 2.00},
        "gpt-4o": {"input": 2.50, "cache_read": 1.25, "output": 10.00},
        "text-embedding-3-small": {"input": 0.02},
    }


def test_settings_read_the_price_list_from_config_toml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    monkeypatch.delenv("STOCKAGENT_TOML_FILE", raising=False)

    settings = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]

    table = settings.llm_price_table()
    assert table["claude-sonnet-4-6"].cache_write_usd_per_mtok == pytest.approx(3.75)
    assert table["gpt-5-mini"].cache_write_usd_per_mtok is None
    assert table["text-embedding-3-small"].output_usd_per_mtok is None


def test_real_config_toml_ships_the_two_phase_cycle_off() -> None:
    """EXP-27 (decyzja 2026-10-20): eksperyment za flagą, domyślnie wyłączony."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["two_phase_cycle_enabled"] is False


def test_real_config_toml_ships_the_cycle_runs_history_on_after_migration() -> None:
    """NIEZ-29: migracja 041 (`cycle_runs`) jest na produkcji od 2026-10-23."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["cycle_runs_enabled"] is True


def test_real_config_toml_ships_the_scorecard_run_id_off_until_migration() -> None:
    """ML-24: kolumna `model_scorecards.run_id` dopiero po migracji 049 (ręczny DB Migrate)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["model_scorecard_run_id_enabled"] is False


def test_real_config_toml_ships_the_scorecard_feature_stats_off_until_migration() -> None:
    """ML-21: kolumna `model_scorecards.feature_stats` dopiero po migracji 052."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["model_scorecard_feature_stats_enabled"] is False


def test_real_config_toml_ships_the_trend_probabilities_off_until_migration() -> None:
    """ML-29: kolumna `prediction_logs.trend_probabilities` dopiero po migracji 051."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["trend_probabilities_enabled"] is False


def test_real_config_toml_ships_the_market_features_on() -> None:
    """ML-31: cechy rynkowe i wersja zestawu cech 4 — włączone świadomie przez
    właściciela 2026-10-25 (wektory v4 dla treningu ML-52)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["market_features_enabled"] is True


def test_real_config_toml_ships_the_snapshot_training_off() -> None:
    """ML-52: wagi z `feature_snapshots` dopiero po zebraniu wektorów wersji 4."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["snapshot_training_enabled"] is False


def test_real_config_toml_ships_the_sigma_target_off() -> None:
    """ML-11: target zwrot / σ zmienia serwowaną prognozę — świadome włączenie."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["ml_sigma_target_enabled"] is False


def test_real_config_toml_ships_the_macro_series_off() -> None:
    """DAT-22: głosy VIX/HY zmieniają reżim (przepis) — świadome włączenie właściciela."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["macro_series_enabled"] is False


def test_real_config_toml_ships_the_llm_failover_off() -> None:
    """NIEZ-44: zapasowy dostawca LLM gotowy w kodzie, ale wyłączony — w pliku jest
    tylko zakomentowany przykład, dopóki właściciel nie doda klucza drugiego dostawcy."""
    text = CONFIG_TOML.read_text(encoding="utf-8")
    data = tomllib.loads(text)

    assert "llm_fallback_provider" not in data
    assert "llm_fallback_model" not in data
    assert "# llm_fallback_provider = " in text


def test_real_config_toml_ships_the_reference_signal_on_after_migration() -> None:
    """WIA-24: migracja 042 (`reference_predictions`) jest na produkcji od 2026-10-23."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["reference_signal_enabled"] is True


def test_real_config_toml_weekly_budget_covers_its_worst_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WIA-11: produkcyjny budżet tygodniowy pokrywa najgorszy przypadek tej samej
    konfiguracji — inaczej każdy start Fast Loopa dawałby ostrzeżenie w bannerze."""
    from src.infrastructure.persona_loader import load_council_personas

    monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
    monkeypatch.delenv("STOCKAGENT_TOML_FILE", raising=False)
    settings = Settings(_env_file=None, **_REQUIRED_SECRETS)  # type: ignore[arg-type]
    personas = load_council_personas(CONFIG_TOML.parent / settings.council_personas_dir)

    weekly = settings.fast_loop_weekly_budget(personas=len(personas))

    assert settings.fast_loop_weekly_budget_usd is not None
    assert weekly.ok, (
        f"tygodniowy worst case {weekly.worst_case_usd:.2f} USD > budżet "
        f"{settings.fast_loop_weekly_budget_usd:.2f} USD w config.toml"
    )


def test_real_config_toml_ships_the_earnings_cache_off_until_migration() -> None:
    """FIN-11: `earnings_cache` powstaje dopiero w migracji 053 (ręczny DB Migrate)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["earnings_cache_enabled"] is False


def test_real_config_toml_ships_the_council_votes_upsert_off_until_migration() -> None:
    """NIEZ-14: upsert głosów wymaga indeksu UNIQUE z migracji 054 (ręczny DB Migrate)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["council_votes_upsert_enabled"] is False


def test_real_config_toml_ships_the_persona_vote_outcomes_off_until_migration() -> None:
    """WIA-51: RPC `persona_vote_outcomes` powstaje dopiero w migracji 045 (DB Migrate)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["persona_vote_outcomes_enabled"] is False


def test_real_config_toml_ships_the_correction_category_off_until_migration() -> None:
    """WIA-57: kolumna `correction_category` powstaje dopiero w migracji 046 (DB Migrate)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["correction_category_enabled"] is False


def test_real_config_toml_ships_the_diagnosis_backfill_off_with_its_cap() -> None:
    """WIA-53: job spłaty diagnoz off; cap 5 mieści się w tygodniu Slow Loopa (15)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["diagnosis_backfill_enabled"] is False
    assert data["diagnosis_backfill_max_calls"] == 5

def test_real_config_toml_ships_the_stale_quote_flag_off_until_migration() -> None:
    """DAT-14: kolumna `prediction_logs.stale_quote` powstaje w migracji 055."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["stale_quote_enabled"] is False


def test_real_config_toml_ships_the_market_cap_write_off_until_migration() -> None:
    """DAT-21: kolumna `fundamentals_cache.market_cap` powstaje w migracji 056."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["fundamentals_market_cap_enabled"] is False


def test_real_config_toml_ships_the_price_level_alerts_off_until_migration() -> None:
    """ALR-11: tabela `price_level_alerts` powstaje w migracji 057; lista startowa
    jest pusta — poziomy dopisuje właściciel."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["price_level_alerts_enabled"] is False
    assert data["price_alerts"] == []


def test_real_config_toml_keeps_crypto_on_the_shared_tool_use_threshold() -> None:
    """FIN-08: bez własnego progu krypto bierze `tool_use_volatility_threshold`."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert "crypto_tool_use_volatility_threshold" not in data


def test_real_config_toml_ships_the_calendar_edge_horizon_off() -> None:
    """WIA-28: horyzont edge z kalendarza NYSE czeka do końca EXP-02/03."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["edge_calendar_horizon_enabled"] is False


def test_real_config_toml_ships_the_split_detection_off_until_migration() -> None:
    """DAT-23: tabela `corporate_actions` powstaje w migracji 062, a korekta zmienia
    rozliczenie, więc czeka też do końca EXP-02/03."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["split_detection_enabled"] is False


def test_real_config_toml_ships_the_price_outlier_guard_off() -> None:
    """DAT-15: strażnik zmienia bramkę i rozliczenie, więc czeka do końca EXP-02/03."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["price_outlier_guard_enabled"] is False


def test_real_config_toml_ships_the_exact_rag_search_off_until_migration() -> None:
    """ML-23: funkcja `match_news_embeddings_exact` powstaje w migracji 063, a analogi
    w prompcie to wejście EXP-03 — czeka do końca eksperymentu."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["rag_exact_search_enabled"] is False


def test_real_config_toml_ships_the_prompt_taint_fence_off() -> None:
    """BEZ-14: fence zmienia tekst promptów predykcji i przewodniczącego (EXP-02/03)."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    assert data["prompt_taint_fence_enabled"] is False
