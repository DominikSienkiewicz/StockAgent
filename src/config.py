from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from src.domain.experiment import Experiment, ExperimentMetric, validate_experiments
from src.domain.finops import (
    AV_DAILY_REQUESTS_PER_KEY,
    AvBudget,
    ModelPrice,
    WeeklyCostBudget,
    av_daily_budget,
    capped_cycle_call_upper_bound,
    weekly_cost_budget,
)
from src.domain.macro_risk import MacroRiskInstrumentType
from src.domain.regime import DEFAULT_EXIT_CONFIRMATIONS

# Ścieżka do config.toml zakotwiczona w ROOCIE repo (src/config.py → ../config.toml),
# NIE względem CWD. Inaczej agent odpalony spoza roota nie znajduje pliku i cały
# config niewrażliwy po cichu spada do defaultów (np. notifications_enabled=False).
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TOML_FILE = str(_REPO_ROOT / "config.toml")

# WIA-20: pola, które NIE wchodzą do `Settings.config_hash()` — sekrety (klucze
# API, tokeny, poświadczenia bazy), adresy e-mail i adresy docelowe kanałów
# push. Hash ląduje w kwicie predykcji w bazie, więc sekret nie może go zmieniać.
# Test pilnuje, że każde pole o „sekretnej” nazwie jest na tej liście.
SECRET_SETTINGS_FIELDS: frozenset[str] = frozenset({
    "openai_api_key",
    "anthropic_api_key",
    "fred_api_key",
    "finnhub_api_key",
    "alpha_vantage_api_keys",
    "supabase_url",
    "supabase_key",
    "resend_api_key",
    "digest_from_email",
    "digest_to_email",
    "telegram_bot_token",
    "telegram_chat_id",
    "slack_webhook_url",
    # Nagłówek SEC niesie adres kontaktowy.
    "edgar_user_agent",
    # NIEZ-13: URL pingu heartbeatu niesie UUID checku — kto go zna, ten pinguje.
    "heartbeat_url",
})

# WIA-20: pola niesekretne, które NIE zmieniają prognozy, więc nie wchodzą do
# `config_hash` — inaczej zmiana strefy czasu maila albo włączenie zapisu
# kolumny cięłyby track record na sztuczne segmenty. Wpływ na prognozę, który
# idzie wyłącznie przez wagi ML (filtr wersji cech, bramka karty), łapie
# `model_version`. Pole niejednoznaczne zostaje w `RECIPE_SETTINGS_FIELDS`:
# lepiej uciąć segment niż skleić dwa różne przepisy.
NON_RECIPE_SETTINGS_FIELDS: frozenset[str] = frozenset({
    # Prezentacja i dostarczanie raportu.
    "display_timezone",
    "notifications_enabled",
    "messenger_digest_enabled",
    "realtime_alerts_enabled",
    "subscriptions_enabled",
    "web_digest_enabled",
    "web_digest_path",
    "public_scorecard_enabled",
    "public_scorecard_path",
    "weekly_recap_enabled",
    # Sekcje maila liczone po predykcji (render-only, bez wpływu na prognozę).
    "portfolio_var_enabled",
    "var_confidence",
    "stress_test_enabled",
    "hedge_effectiveness_enabled",
    "equity_curve_enabled",
    "calibration_curve_enabled",
    "lessons_enabled",
    "calibrated_confidence_enabled",
    # PORT-05: peerzy sekcji 💡 Warte uwagi — sugestie liczone po predykcji.
    "sector_peers",
    "model_scorecard_enabled",
    "cycle_diff_enabled",
    "settled_at_windows_enabled",
    "track_record_days",
    # WIA-50: koszt paper-PnL / backtestu — parametr oceny, nie prognozy.
    "paper_cost_bps",
    "crypto_paper_cost_bps",
    "portfolio_positions_enabled",
    # PORT-18: waluta i data zakupu pozycji (odczyt + zapis importu) — tylko raport.
    "positions_currency_enabled",
    # PORT-11: cache kursów NBP z dnia (P/L w zł) — tylko raport.
    "fx_rates_cache_enabled",
    # Flagi samego zapisu i publikacji.
    # BEZ-08: czy przebieg w ogóle zapisuje do bazy; prognoza cyklu ta sama.
    "db_writes_enabled",
    # FIN-35: atrapy płatnych portów tylko lokalnie (produkcja zawsze true), a lokalny
    # zapis do bazy i tak jest off (`db_writes_enabled`) — przepisów produkcji nie dzieli.
    "paid_calls_enabled",
    "receipts_enabled",
    "feature_set_version_enabled",
    "recipe_version_enabled",
    # ML-38: sam tag reżimu w rekordzie; filtr RAG to `vector_memory_enabled`.
    "regime_tagging_enabled",
    # ML-14: sam zapis wektora cech; prognoza cyklu jest identyczna z flagą i bez.
    "feature_snapshots_enabled",
    # ML-34: zapis kolumn przedziału + pokrycie w Track Record. Przedział opisuje
    # niepewność prognozy, nie zmienia trendu, celu ani rekomendacji.
    "interval_coverage_enabled",
    # EXP-01: sam zapis delty bramki; test echa liczy ją offline w `evaluate`.
    "gate_delta_enabled",
    # EXP-05: sam zapis ramion w kolumnie; definicje eksperymentów są w przepisie.
    "experiment_arms_enabled",
    "model_scorecard_baselines_enabled",
    "attestation_enabled",
    "attestation_commitments_path",
    "attestation_reveals_path",
    # Inne joby i trening (wagi → `model_version`).
    "shock_alerts_enabled",
    "liveness_fast_loop_max_gap_h",
    "liveness_slow_loop_max_gap_h",
    "calibration_judge_enabled",
    "feature_set_filter_enabled",
    # FIN-32: sama kontrola startu (WARNING vs błąd) i jej kadencja; prognoza ta sama.
    "av_budget_strict",
    "fast_loop_cycles_per_day",
    # FIN-12: sufit kosztu cyklu decyduje, KTÓRE symbole dostają analizę, nie jaką —
    # prognoza przeanalizowanego symbolu jest identyczna z sufitem i bez.
    "max_paid_llm_calls_per_cycle",
    # WIA-11: budżet tygodniowy to sama kontrola startu (WARNING i banner); prognoza ta sama.
    "fast_loop_weekly_budget_usd",
    # FIN-30: tygodniowy sufit LLM Slow Loopa (recap, sędzia) — inny job, nie Fast Loop.
    "slow_loop_max_paid_llm_calls_per_week",
    # FIN-34: dzień zamknięcia NYSE decyduje, CZY akcje idą do grafu, nie jak.
    "nyse_closure_skip_enabled",
    # EXP-27: cykl dwufazowy zmienia kolejność analizy (malejące |Δ|) i to, kogo
    # tnie sufit FIN-12 — płatna analiza symbolu jest ta sama co w cyklu jednofazowym.
    "two_phase_cycle_enabled",
    # FIN-13: cennik wycenia tokeny w mailu i ledgerze, a flaga włącza sam zapis
    # `llm_calls` — żadne z nich nie zmienia prognozy.
    "llm_pricing",
    "llm_pricing_as_of",
    "cost_ledger_enabled",
    # NIEZ-29: sam zapis historii przebiegów `cycle_runs` — prognoza ta sama.
    "cycle_runs_enabled",
    # WIA-24: sygnał referencyjny w Slow Loopie, osobna tabela — predykcja LLM ta sama.
    "reference_signal_enabled",
    # NIEZ-44 (decyzja 2026-10-20): zapasowy LLM działa tylko po awarii głównego,
    # a predykcja od niego niesie własny przepis — flagę `llm_failover` i faktyczny
    # dostawcę/model w kwicie i `recipe_version` — więc track record ją odsieje.
    "llm_fallback_provider",
    "llm_fallback_model",
})

# WIA-20: pola przepisu — progi, modele, providerzy, symbole, źródła danych
# i flagi zmieniające ścieżkę predykcji. Razem z dwiema listami wyżej dzielą
# `Settings.model_fields` dokładnie na trzy części (test strażnik).
RECIPE_SETTINGS_FIELDS: frozenset[str] = frozenset({
    "llm_provider",
    "council_llm_provider",
    "council_llm_model",
    "persona_weighting_enabled",
    "debate_mode_enabled",
    "model_routing_enabled",
    "routing_floor",
    "router_cheap_model",
    "router_frontier_provider",
    "router_frontier_model",
    "regime_aware_enabled",
    "regime_threshold_max_multiplier",
    # ALR-03: histereza przedłuża RISK_OFF, czyli wyższy efektywny próg bramki.
    "regime_hysteresis_enabled",
    "regime_exit_confirmations",
    "contagion_enabled",
    # PORT-05: sektory → `peer_groups` (contagion w prompcie, ekspozycja w sizingu).
    "sectors",
    "peer_groups",
    "alpha_fusion_enabled",
    "insider_flow_enabled",
    "earnings_calendar_enabled",
    "options_flow_enabled",
    "social_velocity_enabled",
    "yield_curve_enabled",
    "analyst_consensus_enabled",
    "vector_memory_enabled",
    "tool_use_enabled",
    "tool_use_volatility_threshold",
    "tool_use_max_rounds",
    "council_personas_dir",
    "volatility_threshold",
    "council_volatility_threshold",
    "symbols",
    "symbols_etf",
    "crypto_symbols",
    "crypto_volatility_threshold",
    # Risk Watch zasila reżim rynku (#7), który zacieśnia próg.
    "risk_symbols",
    "risk_symbol_types",
    "nbp_enabled",
    "symbols_unsupported_price",
    # Tempo i równoległość: przy > 1 wyłączają contagion, a throttle chroni
    # radę przed 429 (rada bez głosów to inna prognoza).
    "symbol_throttle_seconds",
    "symbol_concurrency",
    # Okno rozliczeń decyduje, które wnioski z pomyłek trafiają do promptu.
    "reflection_min_age_hours",
    "reflection_max_age_hours",
    "rag_outcome_weight",
    "ml_model_path",
    # EXP-05: ramiona zmieniają prompt (rada bez trendu LLM, prompt bez pamięci).
    "experiments",
})


# FIN-07: jedyny próg volatility, dla którego 0 ma znaczenie ("rada zawsze leci,
# gdy główna bramka przepuści"); pozostałe progi bramek muszą być > 0.
_ALWAYS_RUN_THRESHOLD_FIELD = "council_volatility_threshold"


# BEZ-10: klucze Supabase działające jako role API, które po migracji 038 (RLS bez
# polityk i REVOKE) nie widzą żadnego wiersza: publishable (nowy format) oraz
# legacy JWT z rolą anon / authenticated. Aplikacja potrzebuje klucza secret
# (`sb_secret_…`) albo legacy JWT service_role, które omijają RLS.
_PUBLIC_SUPABASE_KEY_PREFIX = "sb_publishable_"
_PUBLIC_SUPABASE_JWT_ROLES = frozenset({"anon", "authenticated"})


def _supabase_jwt_role(key: str) -> str | None:
    """Claim `role` z payloadu JWT, bez weryfikacji podpisu (to nie uwierzytelnienie,
    tylko rozpoznanie rodzaju klucza). None, gdy klucz nie jest JWT albo nie ma roli."""
    parts = key.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except ValueError:
        # binascii.Error, UnicodeDecodeError i JSONDecodeError dziedziczą po ValueError.
        return None
    role = claims.get("role") if isinstance(claims, dict) else None
    return role if isinstance(role, str) else None


class ExperimentSettings(BaseModel):
    """EXP-05: jeden wpis `[experiments.<id>]` z `config.toml`. Domenową
    walidację (ramiona, metryka, okno) robi `Experiment` przy starcie."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: str
    arms: list[str]
    start: date
    end: date
    primary_metric: ExperimentMetric
    enabled: bool = False


class ModelPriceSettings(BaseModel):
    """FIN-13: stawki jednego modelu z `config.toml`, USD za 1M tokenów (standard
    tier). Brak stawki = dostawca jej nie nalicza; request z takimi tokenami nie
    jest wyceniany (fallback na `UNIT_COST_USD`), więc literówka w kluczu to błąd."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: float = Field(ge=0)
    output: float | None = Field(default=None, ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)


class Settings(BaseSettings):
    """Konfiguracja runtime — dwa rozłączne źródła:

    - **Config niewrażliwy** (symbole, progi, modele, providerzy, throttle…)
      żyje w commitowanym `config.toml` — single source of truth, wersjonowany,
      review-owalny. Bez duplikacji w `.env`/`.env.example`/workflow.
    - **Sekrety** (klucze API, DB credentials, odbiorca maila) NIGDY nie trafiają
      do `config.toml` — czytane z env / `.env` (lokalnie) / GitHub Secrets (CI).

    Precedencja (od najwyższej): init kwargs → env var → `.env` → `config.toml`
    → defaulty w kodzie. Dzięki temu env/.env może nadpisać dowolną wartość
    (np. szybki eksperyment lokalny), a sekrety wstrzykuje się przez env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # BEZ-10: błąd walidacji nie wypisuje wejścia — traceback startu nie może
        # nieść wartości sekretu (klucza Supabase, tokenu), nawet gdy pole ma zły typ.
        hide_input_in_errors=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Dokłada `config.toml` jako źródło PONIŻEJ env/.env (sekrety i
        eksperymenty z env nadpisują), a POWYŻEJ defaultów w kodzie.

        `STOCKAGENT_DISABLE_TOML=1` pomija źródło TOML — używają tego testy
        jednostkowe, by asertować czyste defaulty pól bez wpływu config.toml.
        """
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if os.environ.get("STOCKAGENT_DISABLE_TOML") != "1":
            # Ścieżkę podajemy wprost (nie przez model_config), by w trybie
            # wyłączonym nie było ostrzeżenia o nieużytym kluczu `toml_file`.
            # `STOCKAGENT_TOML_FILE` pozwala testom wskazać tymczasowy plik.
            toml_file = os.environ.get("STOCKAGENT_TOML_FILE", _DEFAULT_TOML_FILE)
            sources.append(
                TomlConfigSettingsSource(settings_cls, toml_file=toml_file)
            )
        sources.append(file_secret_settings)
        return tuple(sources)

    # ----- LLM -----
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str
    anthropic_api_key: str | None = None
    # NIEZ-44: zapasowy dostawca głównej analizy — dostaje request tylko, gdy główny
    # rzuci po swoich retry (4xx poza 429 nie przełącza). Wymaga klucza drugiego
    # dostawcy; bez niego WARNING i sam główny. None = bez failovera (domyślnie;
    # config.toml ma tylko zakomentowany przykład). Model None = default providera.
    llm_fallback_provider: Literal["openai", "anthropic"] | None = None
    llm_fallback_model: str | None = None

    # Heterogeniczna strategia: rada doradcza (N person + chairman × M symboli) może
    # działać na tańszym modelu niż główna analiza, bo persona-acting + JSON
    # nie wymaga frontier reasoningu. Gdy obie None → rada używa głównego
    # LLM (zachowanie domyślne, wstecznie kompatybilne). Override przykładowo:
    #   council_llm_provider=anthropic, council_llm_model=claude-haiku-4-5
    # albo (ten sam provider co main, inny model):
    #   council_llm_model=gpt-4o-mini
    council_llm_provider: Literal["openai", "anthropic"] | None = None
    council_llm_model: str | None = None

    # ----- Rada: ważenie person (#3) i tryb debaty (#1) -----
    persona_weighting_enabled: bool = False
    debate_mode_enabled: bool = False

    # ----- Model routing (#2 — tani model + eskalacja do frontier) -----
    model_routing_enabled: bool = False
    routing_floor: float = 0.6
    router_cheap_model: str = "gpt-5-mini"
    router_frontier_provider: Literal["openai", "anthropic"] = "anthropic"
    router_frontier_model: str = "claude-sonnet-4-6"

    # ----- Reżim rynku (#7), contagion (#5), kalibracja (#4) -----
    # Regime może tylko ZACIEŚNIĆ próg volatility (mnożnik ≥ 1.0).
    regime_aware_enabled: bool = False
    regime_threshold_max_multiplier: float = 2.0
    # ALR-03: histereza reżimu — wejście w RISK_OFF natychmiast, wyjście dopiero po
    # `regime_exit_confirmations` KOLEJNYCH surowych odczytach poza RISK_OFF. Stan
    # w tabeli `regime_history` (migracja 033). Off: przed migracją tabeli nie ma,
    # więc odczyt i zapis by padały (cykl i tak bierze wtedy surowy reżim).
    regime_hysteresis_enabled: bool = False
    regime_exit_confirmations: int = DEFAULT_EXIT_CONFIRMATIONS
    # Contagion: werdykty spółek z tej samej grupy z tego cyklu do promptu.
    contagion_enabled: bool = False
    # PORT-05: sektor → akcje (jedno źródło grup). Puste `peer_groups` wyprowadzamy
    # z sektorów bez ETF-ów i krypto; jawnie podane `peer_groups` wygrywają.
    sectors: dict[str, list[str]] = Field(default_factory=dict)
    peer_groups: dict[str, list[str]] = Field(default_factory=dict)
    # PORT-05: sektor → kuratorowani peerzy spoza watchlisty (sekcja 💡 Warte
    # uwagi). Klucze MUSZĄ być sektorami z `sectors` — inaczej start pada, żeby
    # zmiana nazwy sektora nie wyłączyła sugestii po cichu.
    sector_peers: dict[str, list[str]] = Field(default_factory=dict)
    # LLM-as-Judge kalibracji pewności (Slow Loop, migracja 016).
    calibration_judge_enabled: bool = False

    # ----- Risk: VaR / stress-test / hedge-effectiveness -----
    # Wszystkie liczone z DARMOWEJ historii snapshotów (zero płatnych portów),
    # render-only sekcje raportu. Domyślnie off (additywne, bezpieczne).
    portfolio_var_enabled: bool = False
    # Poziom ufności VaR/CVaR (empiryczny percentyl ogona strat). 0.95 = strata
    # przekraczana w ~5% najgorszych okresów.
    var_confidence: float = 0.95
    stress_test_enabled: bool = False
    hedge_effectiveness_enabled: bool = False

    # ----- Trust / track-record (equity curve, calibration, lessons) -----
    # Render-only, zero płatnych. Czytają zamknięte predykcje z okna
    # `track_record_days`. Domyślnie off.
    equity_curve_enabled: bool = False
    calibration_curve_enabled: bool = False
    lessons_enabled: bool = False
    # #9: surowa pewność LLM korygowana historycznym hit-rate'em swojego kubełka
    # kalibracji. Steruje rankingiem "🎯 Najsilniejsze sygnały" (strength =
    # pewność × |Δ|). Niezależna od `calibration_curve_enabled` — tamta flaga
    # rysuje sekcję Track Record, ta zmienia ranking. Render-only, bez zapisu.
    calibrated_confidence_enabled: bool = False
    # #12: karta kondycji modelu — jawny scorecard walk-forward (migracja 019).
    # Pokazuje bramkę "nie shipuj modelu gorszego od baseline'u", łącznie
    # z odrzutami. Render-only + insert w Slow Loopie. Off.
    model_scorecard_enabled: bool = False
    # ML-10: karta zapisuje też baseline'y „dryf średniej" i „sam LLM", które
    # bramka walk-forward musi pobić obok „zwrot = 0" (kolumny z migracji 028).
    # Off: nowy klucz przed migracją → PGRST204 i przepadły wiersz karty.
    model_scorecard_baselines_enabled: bool = False
    # #8: sekcja "🔄 Zmiany nastawienia" — flipy rady i skoki sentymentu vs
    # poprzedni cykl. Odczyt z Supabase, zero płatnych. Bez migracji. Off.
    cycle_diff_enabled: bool = False
    # #13: kwity decyzyjne — JSONB `decision_receipts` na `prediction_logs`
    # (migracja 020). Flaga to WARUNEK graceful degradation, nie opcja: nowy
    # klucz przed migracją → PGRST204 i śmierć zapisu całej predykcji. Off.
    receipts_enabled: bool = False
    # DAT-09: wersja zestawu cech (migracja 027). `feature_set_version_enabled`
    # stempluje `feature_set_version` na każdej predykcji, a
    # `feature_set_filter_enabled` każe Slow Loopowi trenować tylko na wierszach
    # bieżącej wersji (`FEATURE_SET_VERSION`). Obie off: nowy klucz albo filtr po
    # kolumnie przed migracją 027 → PGRST204/400 (zapis predykcji albo trening).
    feature_set_version_enabled: bool = False
    feature_set_filter_enabled: bool = False
    # WIA-20: kolumna `prediction_logs.recipe_version` (migracja 030) — wersja
    # przepisu predykcji (model, szablon promptu, persony, config, wagi ML). Pola
    # przepisu w kwicie v2 idą za `receipts_enabled` (JSONB, bez migracji). Off:
    # nowy klucz przed migracją 030 → PGRST204 i śmierć zapisu całej predykcji.
    recipe_version_enabled: bool = False
    # ML-14: tabela `feature_snapshots` (migracja 031) — wektor cech w KAŻDYM
    # cyklu, także ignored: cechy darmowe ze stanu cyklu zawsze, cechy LLM tylko
    # za bramką. Zero nowych wywołań zewnętrznych. Off: przed migracją tabeli nie
    # ma, więc każdy cykl kończyłby się błędem zapisu w logu.
    feature_snapshots_enabled: bool = False
    # ML-34: przedział 80% ruchu (conformal + ACI) jest w mailu zawsze, liczony
    # w cyklu bez migracji. Ta flaga dokłada zapis `interval_low/high` (migracja
    # 032) i pokrycie zrealizowane vs nominalne w Track Record. Off: nowy klucz
    # przed migracją → PGRST204 i śmierć zapisu całej predykcji.
    interval_coverage_enabled: bool = False
    # EXP-01: delta ceny, która otworzyła bramkę, w `prediction_logs.gate_delta`
    # (migracja 034) — test echa momentum w `evaluate --echo`. Zero nowych
    # wywołań, prognoza bez zmian. Off: nowy klucz przed migracją → PGRST204
    # i śmierć zapisu całej predykcji.
    gate_delta_enabled: bool = False
    # EXP-05: eksperymenty w pętli — id → definicja (hipoteza, ramiona, okno,
    # metryka główna, `enabled`). Ramię przydziela sha256(id|symbol|dzień UTC)
    # mod liczba ramion; pierwsze ramię to status quo. Walidacja przy starcie:
    # nieznany zabieg, metryka spoza części promptu albo dwa włączone eksperymenty
    # na tej samej części w nakładających się oknach zatrzymują start.
    experiments: dict[str, ExperimentSettings] = Field(default_factory=dict)
    # EXP-05: kolumna `prediction_logs.experiment_arms` (migracja 037). Ramię
    # trafia też do kwitu v2 (za `receipts_enabled`, bez migracji). Off: nowy
    # klucz przed migracją → PGRST204 i śmierć zapisu całej predykcji.
    experiment_arms_enabled: bool = False
    # WIA-19: okna „zamknięte predykcje” po chwili ROZLICZENIA (`settled_at`,
    # migracja 029), nie utworzenia — sekcja maila bierze predykcje rozliczone
    # w tym cyklu, recap tygodniowy rozliczone w ostatnich 7 dniach. Off: filtr
    # po kolumnie przed migracją 029 → 400 i mail bez sekcji, tydzień bez recapu.
    settled_at_windows_enabled: bool = False
    # #11: darmowy watch szoku poza cyklem (main_watch.py, cron co godzinę 24/7,
    # migracja 021). Zero płatnych wywołań. Off.
    shock_alerts_enabled: bool = False
    # NIEZ-11: deadman (main_deadman.py, cron raz dziennie) — maksymalna przerwa
    # między śladami życia jobu w godzinach; 0 = job świadomie wstrzymany. Off.
    # Fast Loop: zapas od północy ET każdego dnia handlowego NYSE (weekend i święto
    # nie alarmują).
    liveness_fast_loop_max_gap_h: float = 0.0
    liveness_slow_loop_max_gap_h: float = 0.0
    # #14: Alpha Fusion Score — ważona fuzja 5 źródeł alfa (migracja 022).
    # Wchodzi do promptu LLM ORAZ jako ósma cecha XGBoost. Wdrożenie cechy
    # rozstrzyga walk-forward gate „pobij baseline", nie intuicja wag. Off.
    alpha_fusion_enabled: bool = False
    # #16: commit-reveal (migracja 024). W chwili predykcji publikujemy SHA-256
    # commitment; po rozliczeniu sweep ujawnia sól i plaintext. UWAGA: daty
    # commitów Gita są fałszowalne — to tamper-evidence, nie niepodrabialny
    # timestamp. Off (nowe kolumny przed migracją 024 → PGRST204).
    attestation_enabled: bool = False
    # #15: realne pozycje użytkownika (tabela `positions`, migracja 023).
    # Nazwa CELOWO nie `portfolio_enabled` — kolidowałaby z `portfolio_var_enabled`
    # i z sekcją watchlisty. Off → Null adapter → dotychczasowe równe wagi.
    portfolio_positions_enabled: bool = False
    # PORT-18: kolumny `positions.currency` i `purchase_date` (migracja 035) —
    # adapter czyta z nich walutę i PRAWDZIWĄ datę zakupu (zamiast `as_of`),
    # a import CSV (`main_tools.py import-positions --apply`) je zapisuje. Off:
    # nowy klucz w zapisie przed migracją → PGRST204; odczyt zostaje przy USD
    # i dotychczasowym mapowaniu.
    positions_currency_enabled: bool = False
    # PORT-11: cache kursów NBP z minionych dni w tabeli `fx_rates` (migracja 036)
    # pod P/L w zł po kursie z dnia zakupu. Same kursy idą z NBP za `nbp_enabled`
    # (darmowe) także bez cache'u. Off: przed migracją tabeli nie ma, więc każdy
    # odczyt i zapis kończyłby się błędem w logu.
    fx_rates_cache_enabled: bool = False
    # #17: niedzielna retrospektywa narracyjna (Slow Loop). JEDYNA pozycja
    # z niezerowym kosztem FinOps: DOKŁADNIE 1 wywołanie LLM tygodniowo
    # (nie per symbol, nie per cykl). Poniżej 5 zamkniętych predykcji mail
    # nie wychodzi — reguła domenowa, nie opcja. Off.
    weekly_recap_enabled: bool = False
    # #10: publiczny scorecard kalibracji (landing page). WYŁĄCZNIE agregaty —
    # zero PII, zero symboli watchlisty, zero kwot. Osobny publisher i osobna
    # ścieżka niż prywatny `web_digest`. Render-only, zero płatnych. Off.
    public_scorecard_enabled: bool = False
    public_scorecard_path: str = "public/scorecard/index.html"
    attestation_commitments_path: str = "public/attestation/commitments.jsonl"
    attestation_reveals_path: str = "public/attestation/reveals.jsonl"
    # Okno (dni) dla krzywej kapitału / panelu lekcji / krzywej kalibracji.
    track_record_days: int = 30
    # WIA-50: jednostronny koszt transakcyjny (bps od obrotu) w paper-PnL,
    # benchmarkach i backteście. Naliczany tylko przy zmianie pozycji symbolu:
    # wejście i wyjście do gotówki 1×, odwrócenie 2×. Krypto ma własny koszt.
    paper_cost_bps: float = 10.0
    crypto_paper_cost_bps: float = 25.0

    # ----- Dane: nowe źródła alpha (default off → Null adapter, zero płatnych) -----
    # Każde źródło ma adapter (wolne/istniejące API) i Null-fallback. W Fast Loop
    # domyślnie Null; realny adapter włącza dopiero flaga (+ ewentualny klucz).
    insider_flow_enabled: bool = False
    earnings_calendar_enabled: bool = False
    options_flow_enabled: bool = False
    social_velocity_enabled: bool = False
    yield_curve_enabled: bool = False
    analyst_consensus_enabled: bool = False
    # Cross-Asset Vector Memory — tagowanie embeddingów reżimem + filtr w RAG
    # (migracja 017). Bez flagi RAG działa jak dotąd (bez filtra reżimu).
    vector_memory_enabled: bool = False
    # ML-38: tag reżimu przy KAŻDEJ predykcji (`prediction_logs.regime`, migracja
    # 017), niezależnie od filtra RAG. None = jak `regime_aware_enabled` (rozstrzyga
    # DI); jawne false to furtka dla bazy bez migracji 017 (inaczej PGRST204).
    regime_tagging_enabled: bool | None = None
    # SEC EDGAR wymaga nagłówka User-Agent z kontaktem (polityka SEC). Niewrażliwe
    # — w config.toml realna wartość, nadpisywalna przez env.
    edgar_user_agent: str = "StockAgent/1.0 (contact@example.com)"
    # FRED API key (darmowy, ale wymagany do pobrań krzywej rentowności) — SEKRET.
    fred_api_key: str | None = None

    # ----- Tool-use research agent (#6) -----
    # Na NAJWIĘKSZYCH ruchach (osobny, wyższy próg) główna analiza idzie pętlą
    # tool-use: model może dociągnąć read-only toole (fundamenty/makro) przed
    # werdyktem. Domyślnie OFF — to płatna, wielorundowa ścieżka.
    tool_use_enabled: bool = False
    # Osobny próg volatility — wyższy niż główny (2%), bo każda runda tool-use to
    # dodatkowe wywołanie LLM. Tylko ruchy ≥ tego progu uruchamiają pętlę.
    tool_use_volatility_threshold: Decimal = Field(default=Decimal("0.05"))
    # Twardy cap rund tool-call (FinOps). Worst-case wywołań płatnych na analizę
    # = tool_use_max_rounds + 1 (ostatnie wymuszone bez toolów).
    tool_use_max_rounds: int = Field(default=3)

    # Katalog z plikami JSON definiującymi członków rady doradczej. Dodanie /
    # usunięcie persony = dodanie / usunięcie pliku, bez zmian w kodzie.
    # Walidator: `uv run python -m src.tools.validate_personas`.
    council_personas_dir: str = "data/council_personas"

    # ----- Market / sentiment / news -----
    finnhub_api_key: str
    # Alpha Vantage akceptuje CSV — wiele kluczy rotujemy, gdy jeden wyczerpie
    # dzienny limit (25 req/dobę free). Wpisz w .env: ALPHA_VANTAGE_API_KEYS=k1,k2,k3
    alpha_vantage_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # FIN-32: najgorszy przypadek doby AV ponad 25 × klucze → WARNING w logu
    # i w bannerze maila (fail-open). True = błąd startu zamiast ostrzeżenia.
    av_budget_strict: bool = False
    # FIN-32: cykle Fast Loopa na dobę w budżecie AV — tyle, ile uruchomień na dobę
    # daje harmonogram `.github/workflows/fast_loop_12h.yml` (zakomentowany = 1,
    # start ręczny; test strażnik pilnuje zgodności). Ręczny dispatch tego samego
    # dnia to kolejny cykl ponad ten rachunek.
    fast_loop_cycles_per_day: int = 1
    # FIN-12: sufit płatnych wywołań LLM jednego cyklu Fast Loopa (predykcja +
    # diagnoza pomyłki + każdy request rady). Miękki: symbol za bramką wchodzi,
    # dopóki wydane < sufit. None = bez limitu (config.toml: 100).
    max_paid_llm_calls_per_cycle: int | None = None
    # WIA-11: jawny budżet tygodniowy Fast Loopa w USD. Najgorszy tydzień (cała
    # watchlista za bramką przy suficie FIN-12 × cykle na dobę × 5 dni handlowych)
    # ponad budżet → WARNING i alert w bannerze maila przy starcie (fail-open).
    # None = bez kontroli.
    fast_loop_weekly_budget_usd: float | None = None
    # FIN-30: sufit płatnych wywołań LLM Slow Loopa na tydzień (= przebieg cronu
    # niedzielnego): recap, sędzia kalibracji i przyszłe joby diagnoz. Przydział
    # liczy domena (`slow_loop_paid_budget`), recap ma pierwszeństwo. None = bez
    # limitu, 0 = tydzień bez LLM (deterministyczne fallbacki). config.toml: 15.
    slow_loop_max_paid_llm_calls_per_week: int | None = None
    # FIN-34: w dzień zamknięcia NYSE (święto, weekend; `market_calendar`) Fast Loop
    # pomija akcje, krypto idzie normalnie; bez krypto przebieg kończy się bez maila.
    # Off w kodzie, żeby testy entry pointów nie zależały od daty; config.toml: true.
    nyse_closure_skip_enabled: bool = False
    # FIN-13: cennik per model (id modelu → stawki USD za 1M tokenów) i jego data.
    # Wycenia tokeny zgłoszone przez adaptery w sekcji „Koszt cyklu” i w ledgerze;
    # model bez wpisu zostaje na szacunku `UNIT_COST_USD`. Pusty w kodzie — stawki
    # z oficjalnych cenników żyją w config.toml razem z `llm_pricing_as_of`.
    llm_pricing: dict[str, ModelPriceSettings] = Field(default_factory=dict)
    llm_pricing_as_of: date | None = None
    # FIN-13: zapis ledgera `llm_calls` (migracja 040) batchem po przebiegu Fast
    # i Slow Loopa. Off w kodzie i w config.toml, dopóki produkcja nie ma migracji.
    cost_ledger_enabled: bool = False
    # NIEZ-29: wiersz `cycle_runs` (migracja 041) na końcu cyklu Fast Loopa: run_id
    # (ten sam co w `llm_calls`), próba runu, liczby cyklu, płatne wywołania i stan
    # maila. Off w kodzie i w config.toml, dopóki produkcja nie ma migracji.
    cycle_runs_enabled: bool = False
    # WIA-24: darmowy sygnał referencyjny `reference-v1` raz na przebieg Slow Loopa
    # (odczyt /quote, rozliczenie, wywołanie) do tabeli `reference_predictions`
    # (migracja 042). Off w kodzie i w config.toml, dopóki produkcja nie ma migracji.
    reference_signal_enabled: bool = False
    # EXP-27: cykl dwufazowy — faza 1 (cena, reflect, bramka) dla wszystkich symboli,
    # faza 2 (klient AV tylko dla symboli za bramką, płatna analiza malejąco po |Δ|).
    # Eksperyment do porównania z FIN-04; off w kodzie i w config.toml.
    two_phase_cycle_enabled: bool = False
    # ----- Database -----
    supabase_url: str
    supabase_key: str
    # BEZ-08: lokalny run i GHA piszą do TEJ SAMEJ bazy. Off = run tylko do odczytu
    # (`write_guard`): odczyty idą do bazy, zapisy są pomijane, predykcja dostaje
    # id `dry-run-<uuid>`. Ustawiane z env / `.env` (`DB_WRITES_ENABLED`) — workflowy,
    # które zapisują, ustawiają "true". Nie w `config.toml` (test strażnik): plik jest
    # wspólny dla lokalnego runu i produkcji, więc zapis włączony w nim byłby wszędzie.
    db_writes_enabled: bool = False
    # FIN-35: lokalny run nie płaci — LLM, rada, Alpha Vantage (newsy, sentyment,
    # fundamenty, kalendarz wyników) i embeddingi idą przez atrapy z jednym WARNING-iem.
    # Jak `db_writes_enabled`: z env / `.env` (`PAID_CALLS_ENABLED`), nie w config.toml
    # (test strażnik); workflowy, które płacą (Fast Loop, Slow Loop), ustawiają "true".
    paid_calls_enabled: bool = False

    # ----- Agent config -----
    volatility_threshold: Decimal = Field(default=Decimal("0.02"))
    # Osobny, wyższy próg dla rady doradczej. Rada (N wywołań LLM person + chairman)
    # jest droga — przy małych Δ jej werdykt to prawie zawsze HOLD. Domyślnie
    # 3% — można podnieść do 5% przy ciasnym budżecie albo wyłączyć (0.0) dla
    # backtestów. Ustaw 0.0, by rada chodziła zawsze gdy główna bramka przepuści.
    council_volatility_threshold: Decimal = Field(default=Decimal("0.03"))
    # NoDecode wyłącza próbę JSON-parse'owania env var — używamy własnego validatora CSV.
    # Default to małe smoke-test portfolio; w produkcji nadpisz przez SYMBOLS env var.
    symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA"]
    )
    # Symbole klasyfikowane jako ETF — pomijają fundamentale (brak EPS/P/E).
    symbols_etf: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # ----- Krypto -----
    # Tickery klasyfikowane jako AssetType.CRYPTO. Routing: CoinGecko zamiast
    # Finnhuba dla ceny, AV NEWS_SENTIMENT z prefiksem CRYPTO:. Krypto nie
    # ma EPS/P/E, więc fundamentale pomijane jak dla ETF.
    crypto_symbols: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Osobny, wyższy próg volatility — natywna zmienność BTC/ETH to 3-5%
    # dziennie, więc 2% (akcyjny) odpalałby pełną analizę KAŻDEGO cyklu.
    # 5% sprawia, że pełen pipeline (LLM + council) idzie tylko gdy ruch
    # jest faktycznie sygnałem, nie szumem.
    crypto_volatility_threshold: Decimal = Field(default=Decimal("0.05"))

    # ----- Risk Watch -----
    # Lista instrumentów proxy ryzyka makro (inverse ETFs, gold, VIX, EPOL).
    # Monitorowane przez osobny use case (MonitorMacroRisk), nie wchodzą do
    # głównej pętli predykcyjnej.
    risk_symbols: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Mapowanie symbol → typ instrumentu (np. SH:INVERSE_EQUITY).
    # Format env: "SYM:TYPE,SYM:TYPE,..." gdzie TYPE ∈ MacroRiskInstrumentType.
    risk_symbol_types: Annotated[
        dict[str, MacroRiskInstrumentType], NoDecode
    ] = Field(default_factory=dict)
    # Włącza adapter NBP do raportu (kursy EUR/PLN, USD/PLN).
    nbp_enabled: bool = False

    # ----- Resilience -----
    # Tickery, które wiadomo z góry, że nie zadziałają z obecnymi adapterami
    # (Finnhub free → 403 dla EU-listed). Pre-filtrujemy je w głównej pętli
    # i mailujemy jako "ignored", nie "error" — żeby nie zaśmiecać sekcji
    # błędów oczywistymi ograniczeniami planu.
    symbols_unsupported_price: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )
    # Sleep w sekundach między symbolami w głównej pętli — chroni OpenAI TPM
    # przy 30+ tickerach z radą doradczą. 0 = bez throttle.
    symbol_throttle_seconds: float = Field(default=0.0)

    # Liczba symboli przetwarzanych równolegle w głównej pętli (Fast Loop).
    # 1 = sekwencyjnie (wstecznie zgodne). > 1 = ThreadPoolExecutor; uwaga: każdy
    # symbol wewnętrznie rozwija radę do ~7 równoległych callsów OpenAI, więc
    # realny fan-out to ~N×7 — trzymaj zachowawczo (config.toml: 4). Throttle i
    # contagion (#5) działają tylko w trybie sekwencyjnym (patrz _analyze_symbols).
    symbol_concurrency: int = Field(default=1)

    # Minimalny wiek (godziny) predykcji, zanim reflect_node ją oceni. Chroni
    # przed przedwczesną oceną przy nakładających się cyklach (ręczny
    # workflow_dispatch tuż po scheduled run oceniłby świeżą predykcję po
    # cenie sprzed minut → zatruty accuracy_score i zawyżony hit-rate).
    # Default = 6h: bezpieczna bramka zgodna z produkcyjną kadencją dzienną
    # (config.toml też ustawia 6). Deployment na samych defaultach kodu — albo
    # konstrukcja Settings() bez TOML — dostaje działającą bramkę, nie footgun.
    # Ustaw 0, by świadomie wyłączyć filtr (np. backtest / zachowanie wsteczne).
    reflection_min_age_hours: int = Field(default=6)

    # WIA-04: maksymalny wiek (godziny) predykcji, którą reflect_node jeszcze
    # rozlicza. Po przerwie w cronie predykcja sprzed tygodni byłaby oceniona
    # bieżącą ceną jak „12h” — taka zostaje nierozliczona i wypada z hit-rate
    # i ml_feature_store. 96 h = weekend (piątek → poniedziałek ≈ 72 h) z
    # zapasem. 0 = bez granicy. Musi być większe niż reflection_min_age_hours.
    reflection_max_age_hours: int = Field(default=96)

    # Waga wyniku w outcome-aware reranku analogów RAG (#9). 0.0 = sama
    # kolejność similarity (zachowanie wsteczne); >0 promuje analogi, których
    # prognoza realnie się sprawdziła. Produkcja: 0.3.
    rag_outcome_weight: float = Field(default=0.0)

    # ----- ML -----
    ml_model_path: str = "data/models/price_predictor.ubj"

    # ----- Notifications (Resend.com — opcjonalne) -----
    notifications_enabled: bool = False
    resend_api_key: str | None = None
    # sandbox sender (działa od razu, bez weryfikacji domeny)
    digest_from_email: str = "onboarding@resend.dev"
    digest_to_email: str | None = None
    # MAIL-14: strefa (nazwa IANA) godzin w temacie i nagłówku maila, pushu oraz
    # alercie szoku. Logi zostają w UTC. Niewrażliwe → config.toml.
    display_timezone: str = "Europe/Warsaw"

    # ----- Kanały push (Telegram / Slack — opcjonalne, U1/U4) -----
    # SEKRETY (placeholdery w .env.example): kanał jest budowany, gdy jego
    # sekrety są obecne (analogicznie do Resend). chat_id/webhook to adresy
    # docelowe → traktujemy jak sekret (jak digest_to_email).
    # #5: push (Telegram/Slack) dostaje 5-linijkowy skrót zamiast pełnego
    # plain-textu raportu. Pełny raport przekracza limit 4096 znaków Telegrama →
    # API zwraca 400 i push ginie po cichu. Off = zachowanie sprzed #5.
    messenger_digest_enabled: bool = False
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    slack_webhook_url: str | None = None
    # NIEZ-13: SEKRET — URL pingu zewnętrznego heartbeatu (np. healthchecks.io).
    # Fast Loop, Slow Loop i shock watch pingują go na koniec przebiegu (porażka →
    # `<URL>/fail`). Brak / pusty = NullHeartbeat, zero requestów.
    heartbeat_url: str | None = None
    # Real-time alerty CRITICAL (U4): pushuj alert kwoty w chwili zdarzenia,
    # out-of-band od dziennego maila. NIEWRAŻLIWY flag → config.toml.
    realtime_alerts_enabled: bool = False

    # ----- Dystrybucja (U2/U3 — opcjonalne, niewrażliwe → config.toml) -----
    # U2: personalizowany fan-out — każdy subskrybent (tabela `subscribers`)
    # dostaje raport tnięty do swojej watchlisty. Brak/false → pojedyncza wysyłka.
    subscriptions_enabled: bool = False
    # U3: publikacja statycznego digestu web (read-only dashboard).
    web_digest_enabled: bool = False
    web_digest_path: str = "public/digest/index.html"

    @field_validator(
        "symbols",
        "symbols_etf",
        "risk_symbols",
        "symbols_unsupported_price",
        "crypto_symbols",
        mode="before",
    )
    @classmethod
    def _parse_csv_symbol_list(cls, value: str | list[str]) -> list[str]:
        """Pozwala podać listy symboli jako CSV w .env (np. ``SYMBOLS=AAPL,MSFT``,
        ``SYMBOLS_ETF=VOO,CSPX.L``). String dzielimy po przecinku i trymujemy;
        gotową listę przepuszczamy bez zmian."""
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return value

    @field_validator("supabase_key")
    @classmethod
    def _reject_public_supabase_key(cls, value: str) -> str:
        """BEZ-10: klucz publiczny zatrzymuje start (fail-fast). Po migracji 038 każdy
        odczyt nim zwraca pusto albo 401, a zapis pada — cykl wyglądałby na zdrowy,
        ale pracowałby na pustej bazie. Nieprzezroczyste stringi (fikstury testów)
        przechodzą: odrzucamy tylko to, co da się rozpoznać jako klucz publiczny."""
        is_public = (
            value.startswith(_PUBLIC_SUPABASE_KEY_PREFIX)
            or _supabase_jwt_role(value) in _PUBLIC_SUPABASE_JWT_ROLES
        )
        if is_public:
            raise ValueError(
                "SUPABASE_KEY is a public Supabase key (publishable or anon / authenticated "
                "JWT). With Row Level Security on (migration 038) it can read and write "
                "nothing. Set the secret key (sb_secret_...) or the legacy service_role key "
                "in .env and in the GitHub SUPABASE_KEY secret "
                "(Supabase dashboard -> Project Settings -> API Keys)."
            )
        return value

    @field_validator("display_timezone")
    @classmethod
    def _validate_display_timezone(cls, value: str) -> str:
        """Literówka w nazwie strefy ma zatrzymać start, a nie wywrócić wysyłkę
        raportu dopiero przy formatowaniu tematu."""
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"display_timezone must be an IANA time zone name (got {value!r})"
            ) from exc
        return value

    @field_validator("symbol_throttle_seconds")
    @classmethod
    def _validate_symbol_throttle(cls, value: float) -> float:
        if value < 0:
            raise ValueError(
                f"symbol_throttle_seconds must be non-negative (got {value})"
            )
        return value

    @field_validator("symbol_concurrency")
    @classmethod
    def _validate_symbol_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                f"symbol_concurrency must be >= 1 (got {value})"
            )
        return value

    @field_validator("var_confidence")
    @classmethod
    def _validate_var_confidence(cls, value: float) -> float:
        """Poziom ufności VaR musi leżeć w (0,1) — inaczej percentyl ogona jest
        bez sensu (0 lub 1 → pusty/pełny ogon)."""
        if not 0.0 < value < 1.0:
            raise ValueError(f"var_confidence must be in (0,1) (got {value})")
        return value

    @field_validator("reflection_max_age_hours")
    @classmethod
    def _validate_reflection_max_age(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"reflection_max_age_hours must be >= 0, got {value}")
        return value

    @model_validator(mode="after")
    def _validate_sector_peers(self) -> Settings:
        """PORT-05: peerzy są kluczowani nazwą sektora z `sectors`. Nieznana
        nazwa (np. po zmianie nazwy sektora) zatrzymuje start zamiast po cichu
        wyłączyć sugestie tego sektora."""
        unknown = sorted(set(self.sector_peers) - set(self.sectors))
        if unknown:
            raise ValueError(
                f"sector_peers has sectors missing from sectors: {unknown}"
            )
        return self

    @model_validator(mode="after")
    def _derive_peer_groups_from_sectors(self) -> Settings:
        """PORT-05: grupy sizingu i contagion to sektory akcji. ETF i krypto nie
        mają sektora, więc wypadają; sektor bez akcji znika."""
        if self.peer_groups:
            return self
        non_stock = set(self.symbols_etf) | set(self.crypto_symbols)
        derived = {
            sector: [symbol for symbol in symbols if symbol not in non_stock]
            for sector, symbols in self.sectors.items()
        }
        object.__setattr__(
            self, "peer_groups", {sector: symbols for sector, symbols in derived.items() if symbols}
        )
        return self

    @model_validator(mode="after")
    def _validate_experiments(self) -> Settings:
        """EXP-05: definicja, która nie zadziała zgodnie z zamiarem, zatrzymuje
        start, zamiast po cichu zbierać dane z testu A/A."""
        validate_experiments(self.experiment_definitions())
        return self

    def llm_price_table(self) -> dict[str, ModelPrice]:
        """FIN-13: domenowy cennik per model z `llm_pricing`."""
        return {
            model: ModelPrice(
                input_usd_per_mtok=price.input,
                output_usd_per_mtok=price.output,
                cache_read_usd_per_mtok=price.cache_read,
                cache_write_usd_per_mtok=price.cache_write,
            )
            for model, price in self.llm_pricing.items()
        }

    def experiment_definitions(self) -> tuple[Experiment, ...]:
        """EXP-05: domenowe definicje eksperymentów, posortowane po id."""
        return tuple(
            Experiment(
                id=experiment_id,
                hypothesis=definition.hypothesis,
                arms=tuple(definition.arms),
                start=definition.start,
                end=definition.end,
                primary_metric=definition.primary_metric,
                enabled=definition.enabled,
            )
            for experiment_id, definition in sorted(self.experiments.items())
        )

    @model_validator(mode="after")
    def _validate_reflection_age_window(self) -> Settings:
        """Okno (min, max] puste → reflect nigdy niczego by nie rozliczył."""
        max_age = self.reflection_max_age_hours
        if max_age > 0 and max_age <= self.reflection_min_age_hours:
            raise ValueError(
                "reflection_max_age_hours must exceed reflection_min_age_hours "
                f"({max_age} <= {self.reflection_min_age_hours}); 0 disables the bound"
            )
        return self

    @field_validator("paper_cost_bps", "crypto_paper_cost_bps")
    @classmethod
    def _validate_paper_cost_bps(cls, value: float, info: ValidationInfo) -> float:
        """Ujemny koszt byłby rabatem za obrót — paper-PnL rósłby od samego handlu."""
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0 (got {value})")
        return value

    @field_validator("track_record_days")
    @classmethod
    def _validate_track_record_days(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"track_record_days must be >= 1 (got {value})")
        return value

    @field_validator("regime_exit_confirmations")
    @classmethod
    def _validate_regime_exit_confirmations(cls, value: int) -> int:
        """1 = wyjście z RISK_OFF od razu (jak bez histerezy); mniej nie ma sensu."""
        if value < 1:
            raise ValueError(f"regime_exit_confirmations must be >= 1 (got {value})")
        return value

    @field_validator("fast_loop_cycles_per_day")
    @classmethod
    def _validate_fast_loop_cycles_per_day(cls, value: int) -> int:
        """Zero cykli wyzerowałoby worst case i wyłączyło kontrolę budżetu AV."""
        if value < 1:
            raise ValueError(f"fast_loop_cycles_per_day must be >= 1 (got {value})")
        return value

    @field_validator("max_paid_llm_calls_per_cycle")
    @classmethod
    def _validate_max_paid_llm_calls_per_cycle(cls, value: int | None) -> int | None:
        """Zero zamknęłoby po cichu każdą analizę; brak limitu to None."""
        if value is not None and value < 1:
            raise ValueError(
                f"max_paid_llm_calls_per_cycle must be >= 1 or unset (got {value})"
            )
        return value

    @field_validator("fast_loop_weekly_budget_usd")
    @classmethod
    def _validate_fast_loop_weekly_budget_usd(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError(
                f"fast_loop_weekly_budget_usd must be >= 0 or unset (got {value})"
            )
        return value

    @field_validator("slow_loop_max_paid_llm_calls_per_week")
    @classmethod
    def _validate_slow_loop_weekly_llm_cap(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError(
                f"slow_loop_max_paid_llm_calls_per_week must be >= 0 or unset (got {value})"
            )
        return value

    @field_validator("tool_use_max_rounds")
    @classmethod
    def _validate_tool_use_max_rounds(cls, value: int) -> int:
        """Cap rund tool-use musi być ≥ 1 — inaczej pętla nie zdąży wywołać
        żadnego toola (research agent zdegradowałby się do zwykłego analyze)."""
        if value < 1:
            raise ValueError(f"tool_use_max_rounds must be >= 1 (got {value})")
        return value

    @field_validator(
        "volatility_threshold",
        "council_volatility_threshold",
        "crypto_volatility_threshold",
        "tool_use_volatility_threshold",
    )
    @classmethod
    def _validate_volatility_threshold(
        cls, value: Decimal, info: ValidationInfo
    ) -> Decimal:
        """Odrzuca progi volatility, które wyłączają bramkę FinOps.

        Ujemny próg sprawia, że `abs(delta) >= threshold` jest ZAWSZE
        prawdziwe → bramka FinOps cicho znika i każdy płatny port (LLM,
        Alpha Vantage, embeddingi) odpala co cykl, bez błędu startu. Próg 0
        robi to samo (FIN-07), więc progi bramek płatnych portów muszą być > 0.
        Wyjątek: `council_volatility_threshold = 0.0` to udokumentowany
        "always run" disable switch drugiego sita rady — ten próg zostaje ≥ 0.
        """
        if info.field_name == _ALWAYS_RUN_THRESHOLD_FIELD:
            if value < 0:
                raise ValueError(f"{info.field_name} must be non-negative (got {value})")
        elif value <= 0:
            raise ValueError(
                f"{info.field_name} must be positive (got {value}) — "
                "a threshold of 0 or less lets every move through the volatility gate"
            )
        return value

    @field_validator("risk_symbol_types", mode="before")
    @classmethod
    def _parse_risk_symbol_types(
        cls,
        value: str | dict[str, MacroRiskInstrumentType] | dict[str, str],
    ) -> dict[str, MacroRiskInstrumentType]:
        """Pozwala podać RISK_SYMBOL_TYPES=SH:INVERSE_EQUITY,GLD:SAFE_HAVEN
        (CSV `SYM:TYPE` → dict[str, MacroRiskInstrumentType]).
        """
        if isinstance(value, dict):
            return {
                k: (v if isinstance(v, MacroRiskInstrumentType)
                    else MacroRiskInstrumentType(v))
                for k, v in value.items()
            }
        if not value:
            return {}
        out: dict[str, MacroRiskInstrumentType] = {}
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(
                    f"RISK_SYMBOL_TYPES entry '{chunk}' must be SYMBOL:TYPE"
                )
            sym, raw_type = chunk.split(":", 1)
            sym = sym.strip()
            raw_type = raw_type.strip()
            try:
                out[sym] = MacroRiskInstrumentType(raw_type)
            except ValueError as exc:
                raise ValueError(
                    f"RISK_SYMBOL_TYPES: '{raw_type}' is not a valid "
                    f"MacroRiskInstrumentType (allowed: "
                    f"{[t.value for t in MacroRiskInstrumentType]})"
                ) from exc
        return out

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

    @model_validator(mode="after")
    def _check_alpha_vantage_budget(self) -> Settings:
        """FIN-32: budżet sprawdzony przy starcie, a nie dopiero jako
        `av_keys_exhausted` w środku cyklu. Domyślnie WARNING i start dalej —
        liczba kluczy w sekretach produkcji nie jest znana z repo, więc błąd
        mógłby zablokować start już dziś; `av_budget_strict` daje twardy błąd.
        Walidator działa w każdym entry poincie; banner maila dokłada Fast Loop."""
        budget = self.av_budget()
        if budget.ok:
            return self
        message = self._av_budget_message(budget)
        if self.av_budget_strict:
            raise ValueError(message)
        logger.warning(message)
        return self

    def av_budget(self) -> AvBudget:
        """FIN-32: worst case doby AV dla pełnej watchlisty Fast Loopa (WIA-11:
        `symbols` + `crypto_symbols` − `symbols_unsupported_price`). Na symbol:
        jedno NEWS_SENTIMENT i jedno EARNINGS_CALENDAR przy włączonym kalendarzu."""
        return av_daily_budget(
            n_av_symbols=self._eligible_symbol_count(),
            requests_per_symbol=2 if self.earnings_calendar_enabled else 1,
            cycles_per_day=self.fast_loop_cycles_per_day,
            n_keys=len(self.alpha_vantage_api_keys),
        )

    def _eligible_symbol_count(self) -> int:
        """Symbole, które Fast Loop wpuszcza do grafu: `symbols` + `crypto_symbols`
        − `symbols_unsupported_price` (bez ceny symbol kończy jako ignored)."""
        unsupported = set(self.symbols_unsupported_price)
        watchlist = [*self.symbols, *self.crypto_symbols]
        return sum(1 for symbol in watchlist if symbol not in unsupported)

    def fast_loop_weekly_budget(self, personas: int) -> WeeklyCostBudget:
        """WIA-11: najgorszy tydzień Fast Loopa vs `fast_loop_weekly_budget_usd`.

        Cała watchlista za bramką przy suficie FIN-12; równolegle idzie
        `symbol_concurrency` symboli, contagion wymusza jeden. `personas` to skład
        rady z plików person — czyta je entry point, nie konfiguracja. Requesty AV
        nie mają ceny, więc liczba tickerów AV nie zmienia kosztu."""
        n_gated = self._eligible_symbol_count()
        bound = capped_cycle_call_upper_bound(
            n_gated=n_gated,
            n_av_symbols=n_gated,
            personas=personas,
            debate=self.debate_mode_enabled,
            tool_rounds=self.tool_use_max_rounds if self.tool_use_enabled else 0,
            cap=self.max_paid_llm_calls_per_cycle,
            concurrency=1 if self.contagion_enabled else self.symbol_concurrency,
        )
        return weekly_cost_budget(
            bound,
            cycles_per_day=self.fast_loop_cycles_per_day,
            budget_usd=self.fast_loop_weekly_budget_usd,
        )

    def _av_budget_message(self, budget: AvBudget) -> str:
        """Diagnoza przekroczenia po angielsku (log, błąd startu)."""
        return (
            f"Alpha Vantage daily budget exceeded: the worst case is {budget.worst_case} "
            f"requests per day (every eligible symbol past the volatility gate in "
            f"{self.fast_loop_cycles_per_day} cycle(s)), the capacity is {budget.capacity} "
            f"({len(self.alpha_vantage_api_keys)} key(s) × {AV_DAILY_REQUESTS_PER_KEY}). "
            "Add keys to ALPHA_VANTAGE_API_KEYS or shrink the watchlist; "
            "av_budget_strict = true turns this warning into a startup error."
        )

    def config_hash(self) -> str:
        """WIA-20: sha256 posortowanych pól przepisu (progi, modele, providerzy,
        symbole, flagi ścieżki predykcji) — składnik wersji przepisu predykcji.

        Sekrety (`SECRET_SETTINGS_FIELDS`) i pola poza przepisem
        (`NON_RECIPE_SETTINGS_FIELDS`: prezentacja, dostarczanie, flagi samego
        zapisu) są wycięte, zanim cokolwiek zostanie zserializowane. Pole spoza
        obu list zostaje w hashu."""
        excluded = SECRET_SETTINGS_FIELDS | NON_RECIPE_SETTINGS_FIELDS
        payload = self.model_dump(mode="json", exclude=set(excluded))
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def secret_values(self) -> tuple[str, ...]:
        """BEZ-02: wartości pól `SECRET_SETTINGS_FIELDS` dla formattera logów —
        każdy klucz z `alpha_vantage_api_keys` osobno, nieustawione pominięte."""
        values: list[str] = []
        for name in sorted(SECRET_SETTINGS_FIELDS):
            value = getattr(self, name)
            items = value if isinstance(value, list) else [value]
            values.extend(str(item) for item in items if item)
        return tuple(values)

    @classmethod
    def from_env(cls) -> Settings:
        """Mypy-friendly entry point — pydantic-settings ładuje pola z env/.env.

        Bezpośrednie `Settings()` daje fałszywy alarm `Missing named argument`,
        bo mypy nie zna mechanizmu source loading w BaseSettings.
        """
        return cls()  # type: ignore[call-arg]
