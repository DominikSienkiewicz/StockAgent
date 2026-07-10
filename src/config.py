from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from src.domain.macro_risk import MacroRiskInstrumentType

# Ścieżka do config.toml zakotwiczona w ROOCIE repo (src/config.py → ../config.toml),
# NIE względem CWD. Inaczej agent odpalony spoza roota nie znajduje pliku i cały
# config niewrażliwy po cichu spada do defaultów (np. notifications_enabled=False).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TOML_FILE = str(_REPO_ROOT / "config.toml")


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
    # Contagion: werdykty skorelowanych spółek z tego cyklu do promptu.
    contagion_enabled: bool = False
    peer_groups: dict[str, list[str]] = Field(default_factory=dict)
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
    # #8: sekcja "🔄 Zmiany nastawienia" — flipy rady i skoki sentymentu vs
    # poprzedni cykl. Odczyt z Supabase, zero płatnych. Bez migracji. Off.
    cycle_diff_enabled: bool = False
    # #13: kwity decyzyjne — JSONB `decision_receipts` na `prediction_logs`
    # (migracja 020). Flaga to WARUNEK graceful degradation, nie opcja: nowy
    # klucz przed migracją → PGRST204 i śmierć zapisu całej predykcji. Off.
    receipts_enabled: bool = False
    # Okno (dni) dla krzywej kapitału / panelu lekcji / krzywej kalibracji.
    track_record_days: int = 30

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
    # ----- Database -----
    supabase_url: str
    supabase_key: str

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

    @field_validator("track_record_days")
    @classmethod
    def _validate_track_record_days(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"track_record_days must be >= 1 (got {value})")
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
        """Odrzuca ujemne progi volatility.

        Ujemny próg sprawia, że `abs(delta) >= threshold` jest ZAWSZE
        prawdziwe → bramka FinOps cicho znika i każdy płatny port (LLM,
        Alpha Vantage, embeddingi) odpala co cykl, bez błędu startu.
        `0` jest dozwolone — `council_volatility_threshold = 0.0` to
        udokumentowany "always run" disable switch.
        """
        if value < 0:
            raise ValueError(
                f"{info.field_name} must be non-negative (got {value})"
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

    @classmethod
    def from_env(cls) -> Settings:
        """Mypy-friendly entry point — pydantic-settings ładuje pola z env/.env.

        Bezpośrednie `Settings()` daje fałszywy alarm `Missing named argument`,
        bo mypy nie zna mechanizmu source loading w BaseSettings.
        """
        return cls()  # type: ignore[call-arg]
