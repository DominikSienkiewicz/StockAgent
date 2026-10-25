from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import tomllib
from datetime import date, time
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal
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

from src.domain.evaluation_protocol import EvaluationProtocol, parse_evaluation_protocol
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
from src.domain.insider_flow import INSIDER_LEAD_MAX_AGE_DAYS
from src.domain.macro_risk import MacroRiskInstrumentType
from src.domain.price_levels import LevelDirection, PriceLevel
from src.domain.promotion import CHAMPION_MODEL_NAME
from src.domain.regime import DEFAULT_EXIT_CONFIRMATIONS
from src.domain.shadow_prompt import SHADOW_PROMPT_MAX_PER_CYCLE

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
    # NIEZ-32: klucz odbiorcy kopii — wartość per deployment, jak sekret (placeholder
    # w `.env.example`); w logach redagowany.
    "backup_age_recipient",
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
    # ALR-09: cisza nocna pushy watcha — sposób dostarczenia, nie prognoza.
    "quiet_hours_enabled",
    "quiet_hours_start",
    "quiet_hours_end",
    # WIA-44: sama kolejność i opis leaderboardu person w mailu; wagi konsensusu
    # (`persona_weighting_enabled`) liczy osobna ścieżka.
    "persona_brier_ranking_enabled",
    # ALR-07: szok portfela w watchu — push poza cyklem, nie prognoza.
    "portfolio_shock_enabled",
    "portfolio_shock_threshold",
    # WIA-45: linia redundancji w recapie i brak odznaki „silny konsensus” w mailu —
    # sama treść maili, ocena po predykcji.
    "council_redundancy_enabled",
    # ALR-11: alerty poziomów cenowych watcha — push poza cyklem, nie prognoza.
    "price_level_alerts_enabled",
    "price_alerts",
    "subscriptions_enabled",
    # MAIL-24: prywatne archiwum digestu i link w mailu — dostarczanie, nie prognoza.
    "digest_archive_enabled",
    # MAIL-35: postać maila w cichy dzień — prezentacja, nie prognoza.
    "digest_quiet_policy",
    # MAIL-48: bot właściciela w Telegramie — osobny job z odczytami bazy, nie prognoza.
    "telegram_bot_enabled",
    # MAIL-45: wyciszenie decyduje, CZY symbol idzie do grafu, nie jak (jak FIN-34).
    "symbol_mutes_enabled",
    "public_scorecard_enabled",
    "public_scorecard_path",
    # NIEZ-33: publiczna strona statusu — prezentacja historii przebiegów, nie prognoza.
    "public_status_page_enabled",
    "public_status_path",
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
    # ML-28: kalibrator rankingu (mapa izotoniczna zamiast kubełków) — render-only.
    "isotonic_calibration_enabled",
    # ML-54: ta sama mapa osobno dla krypto i dla reszty — wciąż tylko ranking.
    "isotonic_class_calibration_enabled",
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
    # ML-31: cechy rynkowe w wektorze i wersja zestawu cech 4 — prompt, wektor modelu
    # i progi bez zmian; kurs cyklu pobierany tylko wcześniej (`CycleQuoteCache`).
    # Wpływ na wagi (filtr wersji w treningu) łapie `model_version`.
    "market_features_enabled",
    # ML-34: zapis kolumn przedziału + pokrycie w Track Record. Przedział opisuje
    # niepewność prognozy, nie zmienia trendu, celu ani rekomendacji.
    "interval_coverage_enabled",
    # EXP-01: sam zapis delty bramki; test echa liczy ją offline w `evaluate`.
    "gate_delta_enabled",
    # EXP-05: sam zapis ramion w kolumnie; definicje eksperymentów są w przepisie.
    "experiment_arms_enabled",
    "model_scorecard_baselines_enabled",
    # ML-24: sam identyfikator przebiegu treningu w karcie; prognoza ta sama.
    "model_scorecard_run_id_enabled",

    # NIEZ-14: upsert zamiast insertu głosów rady — te same głosy, bez duplikatów.
    "council_votes_upsert_enabled",
    # ML-21: sam zapis rozkładu cech treningu (biny + PSI) w karcie; prognoza ta sama.
    "model_scorecard_feature_stats_enabled",
    # CI-13: sam zapis sha256 pliku wag w karcie; prognoza ta sama.
    "model_scorecard_weights_sha256_enabled",
    # ML-22: sam zapis proweniencji treningu (odcisk zbioru, parametry, wersje) w karcie.
    "model_scorecard_provenance_enabled",
    # WIA-57: sam zapis kategorii diagnozy (046) i jej odczyt w panelu lessons.
    "correction_category_enabled",
    # DAT-14: sam zapis znacznika `stale_quote` i odsiew trafności; prognoza ta sama.
    "stale_quote_enabled",
    # DAT-21: sam zapis kapitalizacji w cache fundamentów (próg zakupu insidera w mailu).
    "fundamentals_market_cap_enabled",
    # DAT-21: okno, w którym zakup insidera może otwierać maila — sam lead maila.
    "insider_lead_max_age_days",
    # WIA-61: druga cena rozliczenia (zamknięcie sesji) obok /quote — pomiar, nie prognoza.
    "session_close_settlement_enabled",
    # WIA-54: cena wykonalna (otwarcie sesji) obok /quote i druga krzywa w mailu —
    # pomiar rozliczenia, prognoza i jej ocena bez zmian.
    "executable_entry_enabled",
    # WIA-59: append-only paper ledger i krzywa kapitału z niego — zapis i pomiar
    # tych samych predykcji, prognoza bez zmian.
    "paper_ledger_enabled",
    "attestation_enabled",
    "attestation_commitments_path",
    "attestation_reveals_path",
    # WIA-73: kotwica rootu cyklu w Sigstore Rekor — publikacja po predykcjach, nie prognoza.
    "attestation_rekor_enabled",
    # WIA-59: plik głów paper ledgera obok ledgera atestacji — publikacja, nie prognoza.
    "paper_ledger_heads_path",
    # Inne joby i trening (wagi → `model_version`).
    "shock_alerts_enabled",
    "liveness_fast_loop_max_gap_h",
    "liveness_slow_loop_max_gap_h",
    "calibration_judge_enabled",
    # WIA-37: wersja reguły oceny kalibracji (Slow Loop) — prognoza ta sama.
    "calibration_score_version_enabled",
    # NIEZ-25: sonda Models API w Slow Loopie — sam odczyt katalogu dostawcy.
    "model_probe_enabled",
    "feature_set_filter_enabled",
    # ML-52: źródło próbki modelu wspólnego (wektory zamiast widoku) — wpływ na
    # prognozę wyłącznie przez wagi, które łapie `model_version`.
    "snapshot_training_enabled",
    # ML-26: purge foldów po `settled_at` zmienia werdykt bramki — wpływ na prognozę
    # wyłącznie przez wagi, które łapie `model_version`.
    "ml_settled_purge_enabled",
    # ML-30: wariant binarny to sama ewaluacja w Slow Loopie z osobnym plikiem wag —
    # nie zmienia championa ani prognozy.
    "ml_direction_classifier_enabled",
    # ML-33: prognozy cienia challengerów idą wyłącznie do `shadow_predictions` —
    # wynik cyklu, prompt i mail są te same z flagą i bez.
    "shadow_predictions_enabled",
    "ml_challengers",
    # EXP-26: trend wariantu promptu albo modelu idzie wyłącznie do `shadow_predictions`
    # — prompt, rekord, rada i mail produkcji są te same z flagą i bez.
    "shadow_prompt_variant_enabled",
    "shadow_prompt_variant",
    # FIN-32: sama kontrola startu (WARNING vs błąd) i jej kadencja; prognoza ta sama.
    "av_budget_strict",
    "fast_loop_cycles_per_day",
    # FIN-11 (decyzja 2026-10-21…23): cache Slow Loopa zmienia źródło tej samej daty
    # wyników, nie przepis — włączenie nie tnie track recordu na segmenty.
    "earnings_cache_enabled",
    # DAT-20: sekcja maila „📅 Ten tydzień” — prezentacja; bramka earnings bez zmian.
    "finnhub_earnings_calendar_enabled",
    # DAT-18: DVOL zasila wyłącznie implied edge, sygnał informacyjny bez roli
    # decyzyjnej (R-18) — sam zapis `edge_sigma` dla krypto; prompt i fuzja bez zmian.
    "deribit_dvol_enabled",
    # FIN-12: sufit kosztu cyklu decyduje, KTÓRE symbole dostają analizę, nie jaką —
    # prognoza przeanalizowanego symbolu jest identyczna z sufitem i bez.
    "max_paid_llm_calls_per_cycle",
    # WIA-11: budżet tygodniowy to sama kontrola startu (WARNING i banner); prognoza ta sama.
    "fast_loop_weekly_budget_usd",
    # FIN-30: tygodniowy sufit LLM Slow Loopa (recap, diagnozy) — inny job, nie Fast Loop.
    "slow_loop_max_paid_llm_calls_per_week",
    # FIN-25: limit USD doby i miesiąca z ledgera decyduje, CZY cykl płaci, nie jak —
    # przeanalizowany symbol dostaje tę samą prognozę z limitem i bez.
    "daily_llm_budget_usd",
    "monthly_llm_budget_usd",
    # NIEZ-37: próg ostrzeżenia o rozmiarze bazy — operacje, nie prognoza.
    "db_size_budget_mb",
    # WIA-53: job Slow Loopa diagnozujący błędne predykcje bez diagnozy — przepisuje
    # dane (wnioski w lessons i RAG), nie przepis predykcji.
    "diagnosis_backfill_enabled",
    "diagnosis_backfill_max_calls",
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
    # WIA-51: reguła trafności z domeny zmienia wagi konsensusu (przy ważeniu person).
    "persona_vote_outcomes_enabled",
    "debate_mode_enabled",
    "model_routing_enabled",
    "routing_floor",
    # NIEZ-17: sufit eskalacji decyduje, który model odpowiada niepewnym symbolom.
    "routing_max_escalations",
    "router_cheap_model",
    "router_frontier_provider",
    "router_frontier_model",
    "regime_aware_enabled",
    "regime_threshold_max_multiplier",
    # ALR-03: histereza przedłuża RISK_OFF, czyli wyższy efektywny próg bramki.
    "regime_hysteresis_enabled",
    "regime_exit_confirmations",
    # ALR-10: dryf od ostatniej predykcji otwiera bramkę w cyklach pod progiem.
    "drift_gate_enabled",
    "contagion_enabled",
    # PORT-05: sektory → `peer_groups` (contagion w prompcie, ekspozycja w sizingu).
    "sectors",
    "peer_groups",
    "alpha_fusion_enabled",
    "insider_flow_enabled",
    # DAT-17: świeży 8-K w prompcie predykcji i w wektorze cech.
    "corporate_events_enabled",
    "earnings_calendar_enabled",
    "options_flow_enabled",
    # WIA-28: horyzont edge z kalendarza NYSE — inny zapisany `edge_sigma`.
    "edge_calendar_horizon_enabled",
    # DAT-23: korekta splitu zmienia deltę bramki, rozliczenie i `price_delta`.
    "split_detection_enabled",
    # DAT-15: anomalny odczyt zamyka bramkę i odkłada rozliczenie.
    "price_outlier_guard_enabled",
    "social_velocity_enabled",
    "yield_curve_enabled",
    # DAT-22: głosy VIX i spreadu HY w reżimie zacieśniają próg bramki.
    "macro_series_enabled",
    "analyst_consensus_enabled",
    "vector_memory_enabled",
    "tool_use_enabled",
    "tool_use_volatility_threshold",
    # FIN-08: osobny próg pętli tool-use dla krypto — steruje płatną ścieżką.
    "crypto_tool_use_volatility_threshold",
    "tool_use_max_rounds",
    "council_personas_dir",
    "volatility_threshold",
    "council_volatility_threshold",
    "symbols",
    "symbols_etf",
    "crypto_symbols",
    "crypto_volatility_threshold",
    # Risk Watch zasila reżim rynku (R-25), który zacieśnia próg.
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
    # ML-23: dokładne KNN zmienia analogi RAG w prompcie predykcji.
    "rag_exact_search_enabled",
    # ML-41: analogi RAG tylko rozliczone — inne analogi w prompcie predykcji.
    "rag_point_in_time_enabled",
    # EXP-17: reflect bez diagnozy per błąd — lekcja w prompcie dopiero po tygodniowej
    # diagnozie zbiorczej Slow Loopa.
    "batch_diagnosis_enabled",
    # EXP-10: odrzucone diagnozy wypadają z analogów RAG w prompcie predykcji.
    "diagnosis_judge_enabled",
    "ml_model_path",
    # EXP-05: ramiona zmieniają prompt (rada bez trendu LLM, prompt bez pamięci).
    "experiments",
    # ML-29: prompt prosi o rozkład trzech klas, a trend i pewność idą z argmaxu —
    # inny format odpowiedzi to inny rozkład pewności, więc nowy przepis.
    "trend_probabilities_enabled",
    # BEZ-14: fence lekcji z reflect i uzasadnień person zmienia tekst promptów.
    "prompt_taint_fence_enabled",
    # ML-27: poziom unieważnienia tezy w output_schema promptu predykcji — inny prompt
    # i nowe pole odpowiedzi, więc nowy przepis.
    "invalidation_level_enabled",
    # ML-36: ruch ceny w prompcie predykcji z dwoma miejscami — inny tekst promptu.
    "prompt_delta_rounding_enabled",
    # ML-11: prognoza ML = wyjście modelu × σ symbolu (model w jednostkach σ) — inny
    # przepis serwowania liczby, nawet przy tych samych wagach.
    "ml_sigma_target_enabled",
})


# ML-33: Fast Loop ładuje championa i najwyżej tylu challengerów nogi ML.
MAX_ML_CHALLENGERS = 2


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


class PriceAlertSettings(BaseModel):
    """ALR-11: jeden wpis listy startowej `price_alerts` z `config.toml` — poziom
    cenowy, który watch dopisuje do tabeli `price_level_alerts` (migracja 057)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(min_length=1)
    level: Decimal = Field(gt=0, allow_inf_nan=False)
    direction: Literal["ABOVE", "BELOW"]
    note: str | None = None


class MLChallengerSettings(BaseModel):
    """ML-33: jeden wpis listy `ml_challengers` z `config.toml` — challenger nogi ML:
    nazwa (`model_name` w `shadow_predictions`), plik wag (z Release `model-latest`)
    i rodzaj prognozy: `return` (regresor zwrotu) albo `direction` (klasyfikator
    P(wzrost) z ML-30). Wejście to zawsze wektor cech championa."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    path: str = Field(min_length=1)
    kind: Literal["return", "direction"]


class ShadowPromptVariantSettings(BaseModel):
    """EXP-26: wariant promptu albo modelu LLM w cieniu z `config.toml`.

    `name` — `model_name` wiersza w `shadow_predictions`; `llm_provider` / `llm_model`
    — model wariantu (None = dostawca `llm_provider` i jego domyślny model); opcje
    promptu None = jak produkcja, `without_memory` = bez lekcji i analogów RAG.
    `sample_share` — udział stałej podpróbki symboli, `max_calls_per_cycle` — limit
    wywołań wariantu na cykl (twardo najwyżej `SHADOW_PROMPT_MAX_PER_CYCLE`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    llm_provider: Literal["openai", "anthropic"] | None = None
    llm_model: str | None = Field(default=None, min_length=1)
    prompt_delta_rounding: bool | None = None
    prompt_taint_fence: bool | None = None
    trend_probabilities: bool | None = None
    without_memory: bool = False
    sample_share: float = Field(default=0.5, gt=0, le=1)
    max_calls_per_cycle: int = Field(default=3, ge=1, le=SHADOW_PROMPT_MAX_PER_CYCLE)


class ModelPriceSettings(BaseModel):
    """FIN-13: stawki jednego modelu z `config.toml`, USD za 1M tokenów (standard
    tier). Brak stawki = dostawca jej nie nalicza; request z takimi tokenami nie
    jest wyceniany (fallback na `UNIT_COST_USD`), więc literówka w kluczu to błąd."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: float = Field(ge=0)
    output: float | None = Field(default=None, ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)


def _required_secret(value: str | None, env_name: str) -> str:
    """ARCH-05: sekret wymagany przez fabrykę adaptera; pusty albo brak → ValueError."""
    if not value:
        raise ValueError(f"{env_name} is required by this entry point but is not set.")
    return value


class _StrictTomlSource(TomlConfigSettingsSource):
    """TEST-03 (G64): źródło `config.toml`, które odrzuca klucze spoza `Settings`.

    `extra = "ignore"` zostaje dla env i `.env` (niosą klucze innych narzędzi),
    ale w commitowanym pliku konfiguracji literówka (`weekly_recap_enable`) po
    cichu zostawiałaby default — także płatnej ścieżki."""

    def __call__(self) -> dict[str, Any]:
        unknown = sorted(set(self.toml_data) - set(self.settings_cls.model_fields))
        if unknown:
            raise ValueError(
                f"Nieznane klucze w {self.toml_file_path}: {', '.join(unknown)}"
            )
        return super().__call__()


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
        # ARCH-23: `.env` jak `config.toml` — względem roota repo, nie CWD.
        env_file=str(_REPO_ROOT / ".env"),
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
                _StrictTomlSource(settings_cls, toml_file=toml_file)
            )
        sources.append(file_secret_settings)
        return tuple(sources)

    # ----- LLM -----
    llm_provider: Literal["openai", "anthropic"] = "openai"
    # ARCH-05: klucze płatnych dostawców nie są wymagane do KONSTRUKCJI Settings —
    # darmowe joby (watch, deadman) startują bez nich. Klucza żąda fabryka adaptera
    # (`require_*`), więc jego brak zatrzymuje start entry pointu, który go woła.
    openai_api_key: str | None = None
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

    # ----- Rada: ważenie person (R-21) i tryb debaty (R-19) -----
    persona_weighting_enabled: bool = False
    # WIA-51: wagi person i ich ranking liczy domena z surowych głosów RPC
    # `persona_vote_outcomes` (migracja 045): HOLD na krypto w paśmie ±2.5%, nie
    # akcyjnym ±0.5%. Off: RPC 015/018 — przed migracją 045 funkcji nie ma.
    persona_vote_outcomes_enabled: bool = False
    # WIA-44: leaderboard person w mailu układa się po Brier Skill pewności (RPC
    # `persona_scoring_stats`, migracja 065) zamiast po samym hit-rate'cie. Tylko
    # kolejność i opis w mailu — prognoza ta sama. Off: ranking jak dotąd.
    persona_brier_ranking_enabled: bool = False
    # WIA-45: redundancja rady z zapisanych głosów (kappa Cohena między personami,
    # N_eff). Recap tygodniowy dopisuje linię „7 person ≈ 2,1 niezależnego głosu”,
    # a mail Fast Loopa nie nadaje „SILNY KONSENSUS”, gdy rada mówi jednym głosem
    # (N_eff < 1,5). Tylko treść maili — prognoza ta sama. Off: maile bez zmian.
    council_redundancy_enabled: bool = False
    debate_mode_enabled: bool = False

    # ----- Model routing (R-20 — tani model + eskalacja do frontier) -----
    model_routing_enabled: bool = False
    routing_floor: float = 0.6
    # NIEZ-17: najwyżej tyle eskalacji do frontiera na cykl (jedna kaskada na cykl).
    routing_max_escalations: int = Field(default=5, ge=0)
    router_cheap_model: str = "gpt-5-mini"
    router_frontier_provider: Literal["openai", "anthropic"] = "anthropic"
    router_frontier_model: str = "claude-sonnet-4-6"

    # ----- Reżim rynku (R-25), contagion (R-23), kalibracja (R-22) -----
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
    # Arytmetyczna ocena kalibracji pewności (Slow Loop, R-22) jednym zapisem zbiorczym
    # (migracja 077). WIA-65: bez sędziego LLM — zero płatnych wywołań.
    calibration_judge_enabled: bool = False
    # WIA-37: ocena kalibracji to właściwa reguła Briera (v2) z kolumną
    # `calibration_score_version` (migracja 044). Off: reguła liniowa v1.
    calibration_score_version_enabled: bool = False
    # NIEZ-25: tygodniowa sonda Models API (Slow Loop, bez tokenów, bez migracji).
    model_probe_enabled: bool = False
    # WIA-53: job Slow Loopa diagnozuje błędne predykcje, którym bramka volatility
    # pominęła płatną diagnozę (placeholder w `correction_insights`), najwyżej
    # `diagnosis_backfill_max_calls` na przebieg — w tygodniowym budżecie FIN-30,
    # po recapie i diagnozie zbiorczej. Bez migracji (kolumny 001/009, kwit 020 opcjonalnie).
    diagnosis_backfill_enabled: bool = False
    diagnosis_backfill_max_calls: int = Field(default=5, ge=0)
    # EXP-17: reflect nie płaci za diagnozę błędu; Slow Loop diagnozuje pudła tygodnia
    # JEDNYM wywołaniem LLM w tygodniowym budżecie FIN-30. Zmienia lekcje w prompcie
    # (przepis); włączenie po EXP-02/03 (po 2027-01-29). Bez migracji. Off.
    batch_diagnosis_enabled: bool = False
    # EXP-10: sędzia diagnoz z rubryką — co tydzień ≤ 10 diagnoz (Slow Loop, budżet FIN-30);
    # diagnoza poniżej progu wypada z lessons i RAG (przepis). Wymaga migracji 078.
    # Włączenie po EXP-02/03 (po 2027-01-29) i po kalibracji na ocenach właściciela. Off.
    diagnosis_judge_enabled: bool = False

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
    # WIA-57: kategoria przyczyny błędu z diagnozy LLM w `correction_category`
    # (migracja 046) i jej histogram w panelu lessons. Off: nowy klucz przed
    # migracją → PGRST204 i przepadłe rozliczenie predykcji.
    correction_category_enabled: bool = False
    # R-9: surowa pewność LLM korygowana historycznym hit-rate'em swojego kubełka
    # kalibracji. Steruje rankingiem "🎯 Najsilniejsze sygnały" (strength =
    # pewność × |Δ|). Niezależna od `calibration_curve_enabled` — tamta flaga
    # rysuje sekcję Track Record, ta zmienia ranking. Render-only, bez zapisu.
    calibrated_confidence_enabled: bool = False
    # ML-28: ranking sygnałów kalibruje pewność mapą izotoniczną (PAV) zamiast
    # shrinkage kubełkowego, gdy mapa dopasowana na [t−60 d, t−30 d) ma niższy Brier
    # na [t−30 d, t) niż surowa pewność i kubełki (reguła w domenie). Działa tylko
    # z `calibrated_confidence_enabled`; jeden darmowy odczyt 60 dni rozliczeń.
    # Render-only, bez migracji. Off.
    isotonic_calibration_enabled: bool = False
    # ML-54: osobna mapa izotoniczna dla krypto (`crypto_symbols`) i dla reszty, gdy
    # klasa ma co najmniej MIN_CLASS_ISOTONIC_SAMPLES próbek w każdym oknie i własną
    # ocenę poza próbą; mniejsza klasa zostaje przy mapie globalnej. Działa tylko
    # z `isotonic_calibration_enabled`; ten sam odczyt 60 dni. Render-only. Off.
    isotonic_class_calibration_enabled: bool = False
    # R-12: karta kondycji modelu — jawny scorecard walk-forward (migracja 019).
    # Pokazuje bramkę "nie shipuj modelu gorszego od baseline'u", łącznie
    # z odrzutami. Render-only + insert w Slow Loopie. Off.
    model_scorecard_enabled: bool = False
    # ML-10: karta zapisuje też baseline'y „dryf średniej" i „sam LLM", które
    # bramka walk-forward musi pobić obok „zwrot = 0" (kolumny z migracji 028).
    # Off: nowy klucz przed migracją → PGRST204 i przepadły wiersz karty.
    model_scorecard_baselines_enabled: bool = False
    # ML-24: wiersze karty niosą `run_id` przebiegu Slow Loopa (kolumna z migracji
    # 049), więc re-run tego samego dnia nie skleja się z cronem. Off: nowy klucz
    # przed migracją → PGRST204 i przepadły wiersz karty.
    model_scorecard_run_id_enabled: bool = False
    # ML-21: wiersze karty niosą `feature_stats` treningu — biny kwantylowe cech
    # i PSI ostatnich 30 dni próbki (kolumna JSONB z migracji 052); karta ostrzega
    # przy PSI > 0.25. Off: nowy klucz przed migracją → PGRST204 i przepadły wiersz.
    model_scorecard_feature_stats_enabled: bool = False
    # CI-13: wiersze karty niosą sha256 pliku wag zapisanego w przebiegu (kolumna
    # z migracji 060) — wiąże metryki przebiegu z wagami z Release i z
    # `model_version` predykcji. Off: nowy klucz przed migracją → PGRST204.
    model_scorecard_weights_sha256_enabled: bool = False
    # ML-22: wiersze karty niosą proweniencję treningu — odcisk zbioru (sha256 po
    # wierszach), zakres dat, liczności per symbol, parametry z jawnym ziarnem,
    # wersję XGBoost, wersję zestawu cech i SHA kodu (kolumny z migracji 064).
    # Sidecar `.json` obok pliku wag powstaje bez flagi. Off: PGRST204 przed 064.
    model_scorecard_provenance_enabled: bool = False
    # R-8: sekcja "🔄 Zmiany nastawienia" — flipy rady i skoki sentymentu vs
    # poprzedni cykl. Odczyt z Supabase, zero płatnych. Bez migracji. Off.
    cycle_diff_enabled: bool = False
    # R-13: kwity decyzyjne — JSONB `decision_receipts` na `prediction_logs`
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
    # ML-31: `feature_snapshots` dostaje cechy rynkowe cyklu (zmienność i momentum
    # z historii `price_snapshots`), a stemple wersji zestawu cech idą jako 4 (trening
    # i backtest filtrują wtedy po 4). Działa tylko z `feature_snapshots_enabled`
    # (migracja 031); nowej migracji nie wymaga. Zero płatnych wywołań. Off.
    market_features_enabled: bool = False
    # ML-52 (4): model wspólny (jedyne wagi, które serwuje Fast Loop) trenowany
    # z wektorów `feature_snapshots` w wersji 4 (cykle gated, wektor dokładnie
    # z servingu) z etykietą 12 h z `price_snapshots`, zamiast z widoku
    # `ml_feature_store`. Zmienia serwowane wagi, więc off; potrzebuje wektorów
    # wersji 4 (`market_features_enabled`) — bez nich model wspólny jest pomijany.
    snapshot_training_enabled: bool = False
    # ML-11: target modelu wspólnego = zwrot / σ (σ zwrotów symbolu z 20 poprzednich
    # cykli, `return_sigma`, z podłogą); karty i baseline'y w jednostkach σ, a Fast
    # Loop mnoży prognozę modelu w jednostkach σ przez σ z tej samej funkcji. Model
    # niesie jednostkę w pliku wag. Zmienia serwowaną prognozę, więc off.
    ml_sigma_target_enabled: bool = False
    # ML-26: foldy walk-forward purgują wiersze treningu po chwili rozliczenia
    # etykiety (`settled_at` z widoku po migracji 050, chwila etykiety wektora
    # `feature_snapshots`), nie tylko po stałym horyzoncie 12 h — etykieta piątku
    # rozliczona w poniedziałek nie trenuje foldu walidowanego w weekend. Zmienia
    # werdykt bramki, a więc i serwowane wagi; bez kolumny purge zostaje 12 h. Off.
    ml_settled_purge_enabled: bool = False
    # ML-30: Slow Loop ewaluuje wariant binarny modelu — klasyfikator P(zwrot 12 h
    # > +martwa strefa) z protokołu ewaluacji (`directional_dead_zone`, EXP-06),
    # wiersze w martwej strefie wypadają; Brier i log loss vs stopa bazowa w summary.
    # Wagi wariantu (po przejściu bramki) idą do osobnego pliku `*_direction.ubj`,
    # którego Fast Loop nie czyta — champion bez zmian. Off.
    ml_direction_classifier_enabled: bool = False
    # ML-33: champion/challenger nogi ML. Fast Loop liczy prognozy cienia challengerów
    # (`ml_challengers`) z tego samego wektora cech co champion i zapisuje je razem
    # z prognozą championa do `shadow_predictions` (migracja 075) — wynik cyklu, prompt,
    # mail i rekord predykcji bez zmian. Slow Loop raportuje w summary werdykt promocji
    # (domena `should_promote`); podmianę serwowanych wag robi właściciel. Off: przed
    # migracją tabeli nie ma.
    shadow_predictions_enabled: bool = False
    # ML-33: najwyżej `MAX_ML_CHALLENGERS` challengerów; brak pliku wag = pominięty.
    ml_challengers: list[MLChallengerSettings] = Field(default_factory=list)
    # EXP-26: wariant promptu albo modelu LLM w cieniu. Dla stałej podpróbki symboli za
    # bramką volatility Fast Loop pyta dodatkowo wariant (`shadow_prompt_variant`)
    # i zapisuje jego trend do `shadow_predictions` (kolumny z migracji 083) — mail,
    # rada i rekord predykcji bez zmian. Wywołanie liczy się do sufitu cyklu tylko w jego
    # luzie, najwyżej `max_calls_per_cycle` razy. Off: przed migracją kolumn nie ma.
    shadow_prompt_variant_enabled: bool = False
    shadow_prompt_variant: ShadowPromptVariantSettings | None = None
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
    # ML-29: prompt prosi o rozkład P(BULLISH/BEARISH/SIDEWAYS); trend = argmax,
    # pewność = max p (domena `ProbabilityForecast`, fallback na stare pola), a rozkład
    # trafia do `prediction_logs.trend_probabilities` (migracja 051). Zmienia prompt,
    # więc należy do przepisu. Off: nowy klucz przed migracją → PGRST204 i śmierć
    # zapisu całej predykcji.
    trend_probabilities_enabled: bool = False
    # ML-27: prompt prosi o poziom unieważnienia tezy {price, reason}; domena go waliduje
    # (strona ceny, ±15%), zapis idzie do `prediction_logs.invalidation_price` (migracja
    # 074), mail pokazuje go w bloku uzasadnienia, a watch alarmuje po przekroczeniu.
    # Zmienia prompt, więc należy do przepisu. Off.
    invalidation_level_enabled: bool = False
    # ML-36: ruch ceny w `<current_market_data>` promptu predykcji jako „+3.46” zamiast
    # pełnej precyzji `Decimal` (kilkanaście tokenów mniej). Zmienia tekst promptu, więc
    # należy do przepisu. Bez migracji. Off.
    prompt_delta_rounding_enabled: bool = False
    # BEZ-14: pola półzaufane (lekcja z reflect w prompcie predykcji, uzasadnienia
    # person w prompcie przewodniczącego) w fence'u DATA-ONLY. Zmienia tekst promptów,
    # więc należy do przepisu. Bez migracji. Off.
    prompt_taint_fence_enabled: bool = False
    # WIA-19: okna „zamknięte predykcje” po chwili ROZLICZENIA (`settled_at`,
    # migracja 029), nie utworzenia — sekcja maila bierze predykcje rozliczone
    # w tym cyklu, recap tygodniowy rozliczone w ostatnich 7 dniach. Off: filtr
    # po kolumnie przed migracją 029 → 400 i mail bez sekcji, tydzień bez recapu.
    settled_at_windows_enabled: bool = False
    # R-11: darmowy watch szoku poza cyklem (main_watch.py, cron co godzinę 24/7,
    # migracja 021). Zero płatnych wywołań. Off.
    shock_alerts_enabled: bool = False
    # ALR-07: szok PORTFELA w watchu — Σ waga × Δ po pozycjach (tabela `positions`,
    # `portfolio_positions_enabled`) przy |Δ| ≥ `portfolio_shock_threshold`; jeden push
    # „💼 Portfel -3,4% (NVDA -6,0%, AMD -5,0%)” dziennie, debounce w `shock_alerts`
    # pod symbolem „__PORTFOLIO__” (nowy zapis, więc osobna flaga). Off: watch bez zmian.
    portfolio_shock_enabled: bool = False
    portfolio_shock_threshold: Decimal = Field(default=Decimal("0.03"))
    # NIEZ-11: deadman (main_deadman.py, cron raz dziennie) — maksymalna przerwa
    # między śladami życia jobu w godzinach; 0 = job świadomie wstrzymany. Off.
    # Fast Loop: zapas od północy ET każdego dnia handlowego NYSE (weekend i święto
    # nie alarmują).
    liveness_fast_loop_max_gap_h: float = 0.0
    liveness_slow_loop_max_gap_h: float = 0.0
    # R-14: Alpha Fusion Score — ważona fuzja 5 źródeł alfa (migracja 022).
    # Wchodzi do promptu LLM i do ml_feature_store jako KANDYDAT na cechę —
    # XGBoost go nie używa (poza `ML_FEATURE_COLUMNS`). Promocję do cech modelu
    # rozstrzyga porównanie A/B (EXP-19), nie intuicja wag. Off.
    alpha_fusion_enabled: bool = False
    # R-16: commit-reveal (migracja 024). W chwili predykcji publikujemy SHA-256
    # commitment; po rozliczeniu sweep ujawnia sól i plaintext. UWAGA: daty
    # commitów Gita są fałszowalne — to tamper-evidence, nie niepodrabialny
    # timestamp. Off (nowe kolumny przed migracją 024 → PGRST204).
    attestation_enabled: bool = False
    # R-15: realne pozycje użytkownika (tabela `positions`, migracja 023).
    # Nazwa CELOWO nie `portfolio_enabled` — kolidowałaby z `portfolio_var_enabled`
    # i z sekcją watchlisty. Off → Null adapter → dotychczasowe równe wagi.
    portfolio_positions_enabled: bool = False
    # PORT-11: cache kursów NBP z minionych dni w tabeli `fx_rates` (migracja 036)
    # pod P/L w zł po kursie z dnia zakupu. Same kursy idą z NBP za `nbp_enabled`
    # (darmowe) także bez cache'u. Off: przed migracją tabeli nie ma, więc każdy
    # odczyt i zapis kończyłby się błędem w logu.
    fx_rates_cache_enabled: bool = False
    # R-17: niedzielna retrospektywa narracyjna (Slow Loop). JEDYNA pozycja
    # z niezerowym kosztem FinOps: DOKŁADNIE 1 wywołanie LLM tygodniowo
    # (nie per symbol, nie per cykl). Poniżej 5 zamkniętych predykcji mail
    # nie wychodzi — reguła domenowa, nie opcja. Off.
    weekly_recap_enabled: bool = False
    # R-10: publiczny scorecard kalibracji (landing page). WYŁĄCZNIE agregaty —
    # zero PII, zero symboli watchlisty, zero kwot. Osobny publisher i osobna
    # ścieżka, zero danych z prywatnego maila. Render-only, zero płatnych. Off.
    public_scorecard_enabled: bool = False
    public_scorecard_path: str = "public/scorecard/index.html"
    # NIEZ-33: publiczna strona statusu (`cycle_runs`, migracja 041) i JSON odznaki shields
    # „last digest” obok niej. Ścieżka w katalogu scorecardu — ten sam artefakt Pages,
    # więc strona wychodzi razem ze scorecardem. Same agregaty. Off.
    public_status_page_enabled: bool = False
    public_status_path: str = "public/scorecard/status/index.html"
    attestation_commitments_path: str = "public/attestation/commitments.jsonl"
    attestation_reveals_path: str = "public/attestation/reveals.jsonl"
    # WIA-73: root cyklu z predykcjami logowany w publicznym Sigstore Rekor (podpis keyless
    # tokenem OIDC joba GitHub Actions), wpis Rekora w polu `rekor` linii ledgera. Działa
    # tylko z `attestation_enabled` i zapisem do bazy; wymaga `id-token: write` na jobie
    # agenta. Wpisy są publiczne na zawsze. Off.
    attestation_rekor_enabled: bool = False
    # WIA-59: append-only JSONL z głową paper ledgera, raz dziennie (`paper_ledger_enabled`
    # i `attestation_enabled`) — w katalogu checkoutu gałęzi `attestation-log`.
    paper_ledger_heads_path: str = "public/attestation/paper_ledger_heads.jsonl"
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
    # DAT-17: świeży raport 8-K (ta sama odpowiedź submissions EDGAR co insider flow,
    # wspólny adapter) w prompcie predykcji i jako `fresh_8k` w `feature_snapshots`.
    # Zmiana przepisu. Off do naprawy dostępu do EDGAR (403) i do końca EXP-02/03.
    corporate_events_enabled: bool = False
    earnings_calendar_enabled: bool = False
    # FIN-11 (wymaga migracji 053): Fast Loop czyta datę raportu z `earnings_cache`
    # (odświeżanego w Slow Loopie) zamiast pytać AV o każdy symbol w każdym cyklu.
    earnings_cache_enabled: bool = False
    # DAT-20: poniedziałkowa sekcja „📅 Ten tydzień” z kalendarza Finnhub (jedno
    # zapytanie na watchlistę, darmowe). Bramka earnings bez zmian. Off do przeglądu
    # ToS Finnhub (BEZ-28) i sondy darmowego planu (docs/configuration.md).
    finnhub_earnings_calendar_enabled: bool = False
    options_flow_enabled: bool = False
    # WIA-28: horyzont edge (R-18) akcji i ETF z kalendarza NYSE — godziny do
    # następnego dnia handlowego (~72 h w piątek) zamiast stałych 24 h. Off.
    edge_calendar_horizon_enabled: bool = False
    # DAT-18: port opcji grafu kieruje BTC/ETH do Deribit DVOL (publiczne API, bez
    # klucza) — implied edge (R-18) i kolumna `edge_sigma` dla krypto. Off; włączenie
    # dopiero po falsyfikacji hipotezy edge_sigma (WIA-75), docs/configuration.md.
    deribit_dvol_enabled: bool = False
    social_velocity_enabled: bool = False
    yield_curve_enabled: bool = False
    # DAT-22: zmienne stanu rynku z FRED (VIXCLS, VXNCLS, BAMLH0A0HYM2, T10Y2Y, NFCI),
    # raz na cykl, darmowe (ten sam `fred_api_key`). Z `regime_aware_enabled` percentyl
    # VIX z roku i 5-sesyjna zmiana spreadu HY to głosy reżimu (mogą tylko zacieśnić
    # próg); z `market_features_enabled` cechy trafiają do `feature_snapshots`. Off.
    macro_series_enabled: bool = False
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

    # ----- Tool-use research agent (R-24) -----
    # Na NAJWIĘKSZYCH ruchach (osobny, wyższy próg) główna analiza idzie pętlą
    # tool-use: model może dociągnąć read-only toole (fundamenty/makro) przed
    # werdyktem. Domyślnie OFF — to płatna, wielorundowa ścieżka.
    tool_use_enabled: bool = False
    # Osobny próg volatility — wyższy niż główny (2%), bo każda runda tool-use to
    # dodatkowe wywołanie LLM. Tylko ruchy ≥ tego progu uruchamiają pętlę.
    tool_use_volatility_threshold: Decimal = Field(default=Decimal("0.05"))
    # FIN-08: osobny próg tool-use dla krypto (bramka główna rozróżnia je własnym
    # `crypto_volatility_threshold`). None = krypto bierze `tool_use_volatility_threshold`.
    crypto_tool_use_volatility_threshold: Decimal | None = None
    # Twardy cap rund tool-call (FinOps). Worst-case wywołań płatnych na analizę
    # = tool_use_max_rounds + 1 (ostatnie wymuszone bez toolów).
    tool_use_max_rounds: int = Field(default=3)

    # Katalog z plikami JSON definiującymi członków rady doradczej. Dodanie /
    # usunięcie persony = dodanie / usunięcie pliku, bez zmian w kodzie.
    # Walidator: `uv run python -m src.tools.validate_personas`.
    council_personas_dir: str = "data/council_personas"

    # ----- Market / sentiment / news -----
    finnhub_api_key: str | None = None
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
    # niedzielnego): recap i joby diagnoz (WIA-53, EXP-17, EXP-10). Przydział
    # liczy domena (`slow_loop_paid_budget`), recap ma pierwszeństwo. None = bez
    # limitu, 0 = tydzień bez LLM (deterministyczne fallbacki). config.toml: 15.
    slow_loop_max_paid_llm_calls_per_week: int | None = None
    # FIN-25: limit USD doby i miesiąca kalendarzowego UTC z ledgera `llm_calls`
    # (wymaga `cost_ledger_enabled`, bez zapisu ledgera suma stoi w miejscu). Czytany
    # raz przed cyklem Fast Loopa; wyczerpany → cykl bez płatnych wywołań (same bramki
    # i snapshoty pod progiem), symbole ponad progiem czekają bez snapshotu jak przy
    # suficie FIN-12, mail dostaje banner. Nieudany odczyt → cykl płaci dalej
    # i WARNING (fail-open, decyzja 2026-10-24). None = bez limitu, 0 = bez płatnych.
    daily_llm_budget_usd: float | None = None
    monthly_llm_budget_usd: float | None = None

    # NIEZ-37: budżet rozmiaru bazy w MB (darmowy plan Supabase: 500). Slow Loop raportuje
    # rozmiar relacji `public` (RPC z migracji 077) i ostrzega od 70% budżetu; bez
    # retencji. None = sam raport bez progu. config.toml: 500.
    db_size_budget_mb: int | None = Field(default=None, gt=0)
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
    # NIEZ-14: głosy rady upsertem po (prediction_id, investor_name) — re-run w tej
    # samej godzinie nie dubluje ich wagi. Wymaga UNIQUE z migracji 054; do tego
    # czasu goły insert (upsert bez indeksu PostgREST odrzuca).
    council_votes_upsert_enabled: bool = False
    # DAT-14: kolumna `prediction_logs.stale_quote` (migracja 055) — zapis znacznika
    # nieaktualnego notowania i odsiew takich wierszy z trafności. Off w kodzie
    # i w config.toml, dopóki produkcja nie ma migracji.
    stale_quote_enabled: bool = False
    # DAT-21: kolumna `fundamentals_cache.market_cap` (migracja 056) — Slow Loop zapisuje
    # kapitalizację z OVERVIEW. Off w kodzie i w config.toml, dopóki produkcja nie ma
    # migracji; odczyt (`select("*")`) działa i bez niej.
    fundamentals_market_cap_enabled: bool = False
    # DAT-21: znaczący zakup insidera prowadzi maila (lead i temat) tylko przez tyle
    # dni od daty filingu; sekcja alfy pokazuje go dalej. 3 dni obejmują piątkowy
    # filing, który pierwszy raz widzi poniedziałkowy cykl.
    insider_lead_max_age_days: int = Field(default=INSIDER_LEAD_MAX_AGE_DAYS, ge=0)
    # WIA-61: rozliczenie zapisuje obok ceny /quote oficjalne zamknięcie sesji NYSE
    # z datą i źródłem, a mail pokazuje obie trafności (okres przejściowy). Kolumny
    # z migracji 059; off w kodzie i w config.toml, dopóki produkcja jej nie ma.
    session_close_settlement_enabled: bool = False
    # WIA-54: rozliczenie zapisuje obok ceny /quote cenę wykonalną — otwarcie pierwszej
    # sesji po predykcji z tego samego notowania — a mail rysuje z niej drugą krzywą
    # „od otwarcia”. Kolumna z migracji 072; off w kodzie i w config.toml.
    executable_entry_enabled: bool = False
    # WIA-59: append-only paper ledger z łańcuchem hashy (`paper_trades`, migracja 073):
    # save otwiera pozycję, reflect ją zamyka, krzywa kapitału w mailu czyta ledger,
    # a głowa łańcucha idzie raz dziennie obok atestacji. Off w kodzie i w config.toml.
    paper_ledger_enabled: bool = False
    # DAT-23: `pc` z /quote vs ostatni snapshot — stosunek bliski stosunkowi splitu
    # koryguje referencję bramki, rozliczenie (w jednostkach predykcji) i `price_delta`,
    # a zdarzenie idzie do `corporate_actions` (migracja 062). Off.
    split_detection_enabled: bool = False
    # DAT-15: odczyt ceny dalej niż 8 MAD od mediany zwrotów historii snapshotów (bez
    # potwierdzenia w `pc`) nie otwiera bramki, nie rozlicza predykcji i nie nadpisuje
    # snapshotu-referencji. Bez migracji. Off.
    price_outlier_guard_enabled: bool = False
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
    # ALR-10: bramka widzi dryf — ruch od ceny ostatniej predykcji z okna 10 dni
    # (albo od pierwszej ceny okna) z tym samym progiem i mnożnikami co ruch cyklu.
    # Koszt: najwyżej +1 pełny pipeline na symbol na epizod dryfu. Off.
    drift_gate_enabled: bool = False
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
    # contagion (R-23) działają tylko w trybie sekwencyjnym (patrz _analyze_symbols).
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

    # Waga wyniku w outcome-aware reranku analogów RAG (R-27). 0.0 = sama
    # kolejność similarity (zachowanie wsteczne); >0 promuje analogi, których
    # prognoza realnie się sprawdziła. Produkcja: 0.3.
    rag_outcome_weight: float = Field(default=0.0)
    # ML-23: analogi RAG z dokładnego KNN (`match_news_embeddings_exact`, migracja
    # 063) zamiast ivfflat z probes = 1. Inne analogi w prompcie (EXP-03). Off.
    rag_exact_search_enabled: bool = False
    # ML-41: analogi RAG wyłącznie rozliczone (z wynikiem), z 4× większej puli kandydatów
    # tej samej RPC — nierozliczony analog (także z TEGO cyklu) nie trafia do promptu ani
    # do reranku. Z migracją 076 RPC `match_news_embeddings_pit` (rozliczone przed chwilą
    # predykcji, bez tego samego symbolu z okna horyzontu). Inne analogi w prompcie
    # (EXP-03), więc w przepisie. Off.
    rag_point_in_time_enabled: bool = False

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
    # R-5: push (Telegram/Slack) dostaje 5-linijkowy skrót zamiast pełnego
    # plain-textu raportu. Pełny raport przekracza limit 4096 znaków Telegrama →
    # API zwraca 400 i push ginie po cichu. Off = zachowanie sprzed R-5.
    messenger_digest_enabled: bool = False
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    slack_webhook_url: str | None = None
    # NIEZ-13: SEKRET — URL pingu zewnętrznego heartbeatu (np. healthchecks.io).
    # Fast Loop, Slow Loop i shock watch pingują go na koniec przebiegu (porażka →
    # `<URL>/fail`). Brak / pusty = NullHeartbeat, zero requestów.
    heartbeat_url: str | None = None
    # NIEZ-32: klucz publiczny age odbiorcy tygodniowej kopii tabel dowodowych (Slow
    # Loop), z sekretu repo / `.env`. Brak / pusty = kopia pominięta z WARNING-iem,
    # nigdy bez szyfrowania.
    backup_age_recipient: str | None = None
    # Real-time alerty CRITICAL (U4): pushuj alert kwoty w chwili zdarzenia,
    # out-of-band od dziennego maila. NIEWRAŻLIWY flag → config.toml.
    realtime_alerts_enabled: bool = False
    # ALR-09: cisza nocna pushy watcha szoku w strefie `display_timezone` — w oknie
    # [start, end) push dociera bez dźwięku (Telegram `disable_notification`), nie
    # znika. Okno może przechodzić przez północ. Alerty CRITICAL ciszy nie znają. Off.
    quiet_hours_enabled: bool = False
    quiet_hours_start: time = time(22, 0)
    quiet_hours_end: time = time(7, 0)
    # ALR-11: alerty poziomów cenowych w watchu (tabela `price_level_alerts`, migracja
    # 057): push przy przekroczeniu poziomu, alert wyłącza się po trafieniu. Off w kodzie
    # i w config.toml, dopóki produkcja nie ma migracji — przy off zero odczytów i zapisów.
    price_level_alerts_enabled: bool = False
    # ALR-11: lista startowa poziomów; watch dopisuje brakujące do tabeli (zapisanych,
    # także wyłączonych po trafieniu, nie rusza).
    price_alerts: list[PriceAlertSettings] = Field(default_factory=list)

    # ----- Dystrybucja (U2, MAIL-24 — opcjonalne, niewrażliwe → config.toml) -----
    # U2: personalizowany fan-out — każdy subskrybent (tabela `subscribers`)
    # dostaje raport tnięty do swojej watchlisty. Brak/false → pojedyncza wysyłka.
    subscriptions_enabled: bool = False
    # MAIL-24: pełny HTML cyklu w PRYWATNYM buckecie Supabase Storage `digests`
    # i link „Otwórz w przeglądarce” (podpisany URL, TTL 7 dni) w mailu właściciela
    # i w pushu; indeks w tabeli `digests` (migracja 061). Zapis do bazy i Storage,
    # więc działa tylko z `db_writes_enabled`. Off.
    digest_archive_enabled: bool = False
    # MAIL-35: polityka cichego dnia (pusty lead, zero rozliczeń, szoków, symboli za
    # bramką i błędów): „always” = zawsze pełny mail (jak przed MAIL-35),
    # „compact_when_quiet” = zwarty mail z pełną stopką źródeł. Domena zna też
    # „skip_when_quiet”, ale Settings go nie przyjmuje: bez tygodniowego heartbeatu
    # (MAIL-28) brak maila wygląda jak awaria.
    digest_quiet_policy: Literal["always", "compact_when_quiet"] = "always"

    # ----- Bot właściciela (MAIL-48, MAIL-45 — niewrażliwe → config.toml) -----
    # MAIL-48: `main_bot.py` (cron `bot_poll.yml` co 15 min) odpowiada na komendy
    # z czatu TELEGRAM_CHAT_ID (allowlista); offset `getUpdates` w `bot_state`
    # (migracja 079). Zero płatnych wywołań. Off = przebieg kończy się od razu.
    telegram_bot_enabled: bool = False
    # MAIL-45: wyciszenia z `/mute` (tabela `symbol_mutes`, migracja 080): Fast Loop
    # pomija wyciszony symbol w płatnej analizie i w sekcjach maila, ale zapisuje jego
    # snapshot ceny; symbole dodaje wyłącznie config.toml (decyzja 2026-10-23, opcja b).
    # Off = zero odczytów i zapisów tabeli, mail bajt w bajt jak przed MAIL-45.
    symbol_mutes_enabled: bool = False

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

    @model_validator(mode="after")
    def _validate_ml_challengers(self) -> Settings:
        """ML-33: wiersz cienia rozpoznaje model po nazwie — dwie takie same nazwy
        albo nazwa championa skleiłyby różne modele w jedną serię, a plik championa
        jako challenger porównywałby model z samym sobą."""
        names = [challenger.name for challenger in self.ml_challengers]
        if len(names) > MAX_ML_CHALLENGERS:
            raise ValueError(
                f"ml_challengers holds at most {MAX_ML_CHALLENGERS} entries, got {len(names)}"
            )
        if len(set(names)) != len(names) or CHAMPION_MODEL_NAME in names:
            raise ValueError(
                f"ml_challengers names must be unique and not '{CHAMPION_MODEL_NAME}': {names}"
            )
        if any(challenger.path == self.ml_model_path for challenger in self.ml_challengers):
            raise ValueError("ml_challengers must not point at the champion's ml_model_path")
        return self

    @model_validator(mode="after")
    def _validate_shadow_prompt_variant(self) -> Settings:
        """EXP-26: wiersz wariantu dzieli klucz (prediction_id, model_name) z nogą ML —
        nazwa championa albo challengera nadpisałaby jego prognozę upsertem."""
        variant = self.shadow_prompt_variant
        taken = {CHAMPION_MODEL_NAME, *(challenger.name for challenger in self.ml_challengers)}
        if variant is not None and variant.name in taken:
            raise ValueError(
                f"shadow_prompt_variant name must differ from the ML leg's names: {variant.name}"
            )
        return self

    def price_level_seeds(self) -> tuple[PriceLevel, ...]:
        """ALR-11: lista startowa `price_alerts` jako poziomy domeny."""
        return tuple(
            PriceLevel(
                symbol=alert.symbol.strip().upper(),
                level=alert.level,
                direction=LevelDirection(alert.direction),
                note=alert.note,
            )
            for alert in self.price_alerts
        )

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

    @field_validator("daily_llm_budget_usd", "monthly_llm_budget_usd")
    @classmethod
    def _validate_llm_ledger_budget(
        cls, value: float | None, info: ValidationInfo
    ) -> float | None:
        if value is not None and value < 0:
            raise ValueError(f"{info.field_name} must be >= 0 or unset (got {value})")
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
        "crypto_tool_use_volatility_threshold",
    )
    @classmethod
    def _validate_volatility_threshold(
        cls, value: Decimal | None, info: ValidationInfo
    ) -> Decimal | None:
        """Odrzuca progi volatility, które wyłączają bramkę FinOps.

        Ujemny próg sprawia, że `abs(delta) >= threshold` jest ZAWSZE
        prawdziwe → bramka FinOps cicho znika i każdy płatny port (LLM,
        Alpha Vantage, embeddingi) odpala co cykl, bez błędu startu. Próg 0
        robi to samo (FIN-07), więc progi bramek płatnych portów muszą być > 0.
        Wyjątek: `council_volatility_threshold = 0.0` to udokumentowany
        "always run" disable switch drugiego sita rady — ten próg zostaje ≥ 0.
        None (tylko opcjonalny próg tool-use krypto, FIN-08) = brak własnego progu.
        """
        if value is None:
            return None
        if info.field_name == _ALWAYS_RUN_THRESHOLD_FIELD:
            if value < 0:
                raise ValueError(f"{info.field_name} must be non-negative (got {value})")
        elif value <= 0:
            raise ValueError(
                f"{info.field_name} must be positive (got {value}) — "
                "a threshold of 0 or less lets every move through the volatility gate"
            )
        return value

    @field_validator("portfolio_shock_threshold")
    @classmethod
    def _validate_portfolio_shock_threshold(cls, value: Decimal) -> Decimal:
        """Próg szoku portfela jest ułamkiem w (0, 1): zero albo ujemny alarmowałby
        przy każdym przebiegu, 1 i więcej nigdy."""
        if not Decimal(0) < value < Decimal(1):
            raise ValueError(f"portfolio_shock_threshold must be in (0, 1) (got {value})")
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
        starego `ALPHA_VANTAGE_API_KEY` (single). Brak kluczy nie jest błędem
        konstrukcji (ARCH-05) — wymaga ich `require_alpha_vantage_api_keys`."""
        if not self.alpha_vantage_api_keys:
            legacy = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
            if legacy:
                object.__setattr__(self, "alpha_vantage_api_keys", [legacy])
        return self

    @model_validator(mode="after")
    def _check_alpha_vantage_budget(self) -> Settings:
        """FIN-32: budżet sprawdzony przy starcie, a nie dopiero jako
        `av_keys_exhausted` w środku cyklu. Domyślnie WARNING i start dalej —
        liczba kluczy w sekretach produkcji nie jest znana z repo, więc błąd
        mógłby zablokować start już dziś; `av_budget_strict` daje twardy błąd.
        Walidator działa w każdym entry poincie; banner maila dokłada Fast Loop.
        Bez kluczy (ARCH-05: job bez Alpha Vantage) nie ma czego liczyć — entry
        point, który woła AV, zatrzyma `require_alpha_vantage_api_keys`."""
        if not self.alpha_vantage_api_keys:
            return self
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
        jedno NEWS_SENTIMENT; przy włączonym kalendarzu jedno EARNINGS_CALENDAR na
        cykl (FIN-09: jedna lista dla wszystkich symboli) — chyba że kalendarz idzie
        z cache Slow Loopa (FIN-11)."""
        live_calendar = self.earnings_calendar_enabled and not self.earnings_cache_enabled
        return av_daily_budget(
            n_av_symbols=self._eligible_symbol_count(),
            requests_per_symbol=1,
            cycles_per_day=self.fast_loop_cycles_per_day,
            n_keys=len(self.alpha_vantage_api_keys),
            requests_per_cycle=1 if live_calendar else 0,
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

    def require_openai_api_key(self) -> str:
        """ARCH-05: klucz OpenAI dla fabryki adaptera; brak → ValueError przy starcie."""
        return _required_secret(self.openai_api_key, "OPENAI_API_KEY")

    def require_finnhub_api_key(self) -> str:
        """ARCH-05: klucz Finnhub dla fabryki adaptera; brak → ValueError przy starcie."""
        return _required_secret(self.finnhub_api_key, "FINNHUB_API_KEY")

    def require_alpha_vantage_api_keys(self) -> list[str]:
        """ARCH-05: ≥1 klucz Alpha Vantage dla fabryki adaptera; brak → ValueError."""
        if not self.alpha_vantage_api_keys:
            raise ValueError(
                "alpha_vantage_api_keys is required — set ALPHA_VANTAGE_API_KEYS "
                "(comma-separated) or legacy ALPHA_VANTAGE_API_KEY in env."
            )
        return self.alpha_vantage_api_keys

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


# WIA-42: protokół ewaluacji w roocie repo. Hash liczony z BAJTÓW pliku (razem
# z komentarzami) — każda zmiana pliku to nowy protokół, bez dyskusji o tym,
# czy zmiana „była merytoryczna”.
EVALUATION_PROTOCOL_PATH = _REPO_ROOT / "evaluation_protocol.toml"


def load_evaluation_protocol(path: Path = EVALUATION_PROTOCOL_PATH) -> EvaluationProtocol:
    """Protokół ewaluacji z pliku razem z sha256 jego bajtów (przepis, commitment v2)."""
    raw = path.read_bytes()
    return parse_evaluation_protocol(
        tomllib.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()
    )
