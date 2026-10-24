from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.application.report_council_history import InvestorHistory
from src.domain.analyst_consensus import AnalystConsensus
from src.domain.attestation import StoredCommitment
from src.domain.council import CouncilInput, CouncilVerdict, InvestorOpinion
from src.domain.earnings import CachedEarningsDate, EarningsEvent
from src.domain.finops import LLMCallEntry, LLMUsage
from src.domain.fx import FxRate
from src.domain.insider_flow import InsiderFlowSnapshot
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.macro_series import MacroSeriesSnapshot
from src.domain.mistake_taxonomy import MistakeCategory
from src.domain.model_scorecard import TrainingReport
from src.domain.options_flow import OptionsFlowSnapshot
from src.domain.persona_track_record import PersonaVoteOutcome
from src.domain.polish_macro import PolishMacroSnapshot
from src.domain.portfolio import Position
from src.domain.prediction import Prediction
from src.domain.price_levels import PriceLevel, PriceLevelAlert
from src.domain.quota import QuotaAlert
from src.domain.reference_signal import ReferenceCall, ReferenceOutcome, ReferenceRead
from src.domain.regime import MarketRegime, RegimeState
from src.domain.session_close import SessionCloseSettlement
from src.domain.shock import DailyChange
from src.domain.social_velocity import SocialVelocitySnapshot
from src.domain.subscriber import Subscriber
from src.domain.value_objects import AssetType, Fundamentals, Money, Quote


class MarketDataPort(ABC):
    """Źródło aktualnych cen rynkowych (np. Finnhub, Alpha Vantage)."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Money: ...


class QuotePort(ABC):
    """DAT-14: notowanie z chwilą, z której pochodzi (Finnhub /quote).

    Obok `MarketDataPort`, nie w nim: `get_current_price` ma wielu konsumentów
    i źródła bez czasu notowania, więc migracja na `Quote` idzie stopniowo.
    Brak ceny albo czasu w odpowiedzi → `ValueError`."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...


class DailyChangePort(ABC):
    """Dzienna zmiana ceny z darmowego notowania (ALR-02, Finnhub /quote, CoinGecko).

    Watch szoku liczy z niej alert bez snapshotów Fast Loopa. Brak danych w odpowiedzi
    → `ValueError`, jak w `MarketDataPort`.
    """

    @abstractmethod
    def get_daily_change(self, symbol: str) -> DailyChange: ...


class SentimentPort(ABC):
    """Źródło metryk sentymentu finansowego (Alpha Vantage NEWS_SENTIMENT)."""

    @abstractmethod
    def get_social_score(self, symbol: str) -> dict[str, Any]: ...


class NewsPort(ABC):
    """Źródło artykułów newsowych (Alpha Vantage NEWS_SENTIMENT feed)."""

    @abstractmethod
    def get_news_context(self, symbol: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class PreviousVerdict:
    """#8 — snapshot poprzedniego cyklu potrzebny detektorowi zmian nastawienia."""

    verdict: CouncilVerdict
    sentiment_score: float
    timestamp: datetime


@dataclass(frozen=True)
class LatestPrediction:
    """ALR-08/ALR-10 — ostatnia zalogowana predykcja symbolu: cena wejścia, chwila
    zapisu i uzasadnienie LLM (`reasoning_text`, None = brak)."""

    price: Money
    timestamp: datetime
    reasoning: str | None


@dataclass(frozen=True)
class CycleRun:
    """NIEZ-29 — jeden przebieg pętli (wiersz `cycle_runs`, migracja 041).

    Same liczby przebiegu, bez symboli: `run_id` to `RunContext.run_id` (ten sam
    co w `llm_calls`), `run_attempt` — `GITHUB_RUN_ATTEMPT` (None poza GitHub
    Actions), `paid_calls` — płatne wywołania per klucz cennika,
    `report_delivered` — czy mail wyszedł (None = przebieg bez kanału e-mail)."""

    loop: str
    run_id: str
    run_attempt: int | None
    started_at: datetime
    duration_s: float
    eligible: int
    saved: int
    failures: int
    paid_calls: Mapping[str, int]
    report_delivered: bool | None


@dataclass(frozen=True)
class StoredReceipt:
    """WIA-56 — kwit decyzyjny predykcji z kontekstem jej wiersza `prediction_logs`.

    `payload` to surowy JSONB `decision_receipts` (migracja 020) albo None, gdy
    predykcja kwitu nie ma (zapis sprzed migracji albo bez `receipts_enabled`).
    Wersję schematu rozpoznaje czytający (`read_decision_receipt`), więc nieznana
    wersja dociera do niego jako dane, nie jako wyjątek adaptera.
    `is_trend_correct` None = predykcja jeszcze nierozliczona."""

    prediction_id: str
    symbol: str
    predicted_at: datetime | None
    predicted_trend: str | None
    is_trend_correct: bool | None
    payload: Mapping[str, Any] | None


@dataclass(frozen=True)
class DeferredDiagnosis:
    """WIA-53 — błędna predykcja bez płatnej diagnozy, z jej persystowanym wejściem.

    `prediction` niesie nagłówki (`news_summary`), uzasadnienie i ceny wejścia
    i celu; `actual_price` — cenę rozliczenia; `placeholder` — tekst, który dziś
    stoi w `correction_insights` (zapis go podmienia tylko, gdy wciąż tam jest);
    `receipt` — surowy JSONB `decision_receipts` (020) albo None.
    `is_trend_correct` None = wiersz bez oceny kierunku."""

    prediction: Prediction
    actual_price: Decimal
    is_trend_correct: bool | None
    placeholder: str
    predicted_at: datetime | None = None
    receipt: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class OpenReferenceCall:
    """WIA-24 — nierozliczone wywołanie sygnału referencyjnego z id wiersza
    `reference_predictions` (migracja 042), którym rozlicza je kolejny przebieg."""

    id: str
    call: ReferenceCall


class RepositoryPort(ABC):
    """Persystencja: zapis predykcji, odczyt historii, dostęp do feature store."""

    @abstractmethod
    def get_last_price(self, symbol: str) -> Money | None:
        """Najnowszy snapshot ceny z `price_snapshots` (nie z prediction_logs)."""

    @abstractmethod
    def get_price_history(
        self, symbol: str, days: int
    ) -> list[tuple[datetime, Money]]:
        """Chronologiczna historia cen z `price_snapshots` (rosnąco po czasie),
        za ostatnie `days` dni. Zasila analizy portfelowe (drawdown, korelacje) —
        wszystko liczone z DARMOWYCH snapshotów, bez płatnych portów. Pusta lista
        gdy brak historii."""

    @abstractmethod
    def get_last_prediction_price(
        self, symbol: str, *, before: datetime | None = None
    ) -> Money | None:
        """Cena OSTATNIEJ zalogowanej predykcji (`prediction_logs.price_at_prediction`).

        Referencja dla cechy `price_delta` w inference. Widok `ml_feature_store`
        liczy `price_delta` jako `LAG(price_at_prediction)` — czyli zmianę względem
        POPRZEDNIEJ predykcji. Fast Loop musi użyć tej samej referencji (nie
        ostatniego snapshotu ceny), inaczej cecha ma inny rozkład w treningu i
        w predykcji (train/serve skew). Zwraca None gdy brak wcześniejszej
        predykcji dla symbolu (pierwszy cykl — widok i tak takie wiersze odsiewa).

        `before` (ML-18) to chwila bieżącej predykcji: odczyt pomija wiersze z jej
        godziny UTC i późniejsze. Zapis predykcji to upsert po
        `(symbol, timestamp_hour)`, więc re-run w tej samej godzinie nadpisze
        wiersz z tej godziny — w widoku LAG go nie zobaczy. None = bez filtra."""

    @abstractmethod
    def get_latest_prediction(self, symbol: str) -> LatestPrediction | None:
        """ALR-08/ALR-10 — ostatnia predykcja symbolu (`prediction_logs`) z chwilą
        i uzasadnieniem: kontekst alertu szoku i referencja dryfu bramki.

        Adapter jest SUROWY (bez try/except) — degradację robi caller. None = brak
        predykcji albo wiersz bez ceny lub chwili."""

    @abstractmethod
    def save_price_snapshot(self, symbol: str, price: Money) -> None:
        """Zapisuje bieżącą cenę — wywoływane w KAŻDYM cyklu (rozwiązuje
        cold-start deadlock: następny cykl ma punkt odniesienia do delty)."""

    @abstractmethod
    def save_feature_snapshot(
        self,
        symbol: str,
        features: Mapping[str, Any],
        *,
        gated: bool,
        feature_set_version: int,
    ) -> None:
        """ML-14 — wektor cech point-in-time z tego cyklu (tabela
        `feature_snapshots`, migracja 031), idempotentnie per symbol i godzina.

        `features` musi być JSON-serializowalne. `gated` mówi, czy cykl przeszedł
        bramkę volatility (wtedy wektor niesie też cechy LLM). Etykiety tu nie ma —
        zwrot dołącza się później z `price_snapshots`. Wołane z węzłów terminalnych
        tylko za flagą `feature_snapshots_enabled`."""

    @abstractmethod
    def get_latest_regime_state(self) -> RegimeState | None:
        """ALR-03 — stan histerezy reżimu z najnowszego wiersza `regime_history`
        (migracja 033, kolejność po `cycle_at`). None = brak historii (pierwszy
        cykl → surowy reżim). Wołane raz na cykl tylko za flagą
        `regime_hysteresis_enabled`."""

    @abstractmethod
    def save_regime_state(
        self, cycle_at: datetime, raw: MarketRegime, state: RegimeState
    ) -> None:
        """ALR-03 — jeden wiersz `regime_history` na cykl: surowy reżim
        z `classify()`, reżim efektywny po histerezie i seria potwierdzeń
        wyjścia z RISK_OFF."""

    @abstractmethod
    def save_prediction(self, prediction: dict[str, Any]) -> str:
        """Zwraca ID zapisanej predykcji."""

    @abstractmethod
    def get_unverified_prediction(
        self, symbol: str, min_age_hours: int = 0, max_age_hours: int | None = None
    ) -> Prediction | None:
        """NAJSTARSZA nierozliczona predykcja (bez rzeczywistej ceny), starsza
        niż `min_age_hours`. Oldest-first, by drenować zaległości — inaczej
        starsze nierozliczone predykcje nigdy nie wejdą do feature store.

        `min_age_hours` > 0 → bierze tylko predykcje starsze niż próg
        (timestamp ≤ now − min_age_hours). Chroni przed przedwczesną oceną:
        gdy ręczny `workflow_dispatch` nałoży się na scheduled run, świeżo
        zapisana predykcja nie zostanie oceniona po cenie sprzed kilku minut
        (co zatruwałoby accuracy_score i zawyżało hit-rate raportu).

        `max_age_hours` > 0 → pomija predykcje starsze niż próg (timestamp ≥
        now − max_age_hours; WIA-04). Po przerwie predykcja sprzed tygodni nie
        jest oceniana bieżącą ceną jak „12h”; zostaje nierozliczona (NULL) i
        nie wchodzi do hit-rate ani ml_feature_store. None = bez granicy."""

    @abstractmethod
    def expire_unverified_predictions(self, symbol: str, older_than_hours: int) -> int:
        """WIA-49: oznacza jako wygasłe nierozliczone predykcje symbolu starsze niż
        `older_than_hours` — te, których `get_unverified_prediction` z tym samym
        `max_age_hours` już nigdy nie poda do oceny. Zwraca liczbę oznaczonych.

        Wygaśnięcie to jawny ślad w danych zamiast ciszy, która wygląda jak brak
        predykcji. NIE wpisuje `actual_price_after_12h` (wiersz trafiłby do
        ml_feature_store z fałszywą etykietą) ani oceny kierunku. Idempotentne:
        wiersz już oznaczony nie liczy się drugi raz."""

    @abstractmethod
    def update_prediction_accuracy(
        self,
        prediction_id: str,
        actual_price: Decimal,
        accuracy_score: float,
        is_trend_correct: bool,
        insight: str,
        correction_category: MistakeCategory | None = None,
        *,
        stale_quote: bool | None = None,
        session_close: SessionCloseSettlement | None = None,
    ) -> None:
        """Zamyka predykcję feedbackiem. `accuracy_score` (bliskość ceny do
        celu) napędza trening; `is_trend_correct` (zgodność kierunku) napędza
        trafność raportu — to dwie różne miary, nie wolno ich mylić.

        `correction_category` (WIA-57) — kategoria przyczyny błędu z diagnozy;
        kolumnę z migracji 046 zapisuje adapter tylko za swoją flagą.
        `stale_quote` (DAT-14, migracja 055): True = rozliczenie z notowania sprzed
        sesji po predykcji, wiersz wypada z trafności. None = kolumny nie dotykamy
        (graf podaje ją wyłącznie za `stale_quote_enabled`).
        `session_close` (WIA-61, migracja 059): oficjalne zamknięcie sesji z datą,
        źródłem i trafnością na nim — obok ceny /quote, która zostaje rozliczeniem.
        None = kolumn nie dotykamy (graf podaje je wyłącznie za
        `session_close_settlement_enabled`)."""

    @abstractmethod
    def get_feature_store_data(
        self, symbol: str, feature_set_version: int | None = None
    ) -> list[dict[str, Any]]:
        """Zwraca rekordy ze zmaterializowanego widoku ml_feature_store.

        `feature_set_version` (DAT-09, migracja 027) zawęża odczyt do wierszy tej
        wersji zestawu cech — wiersze innej i nieznanej (NULL) wersji odpadają.
        None = bez filtra (zachowanie sprzed DAT-09; działa bez migracji)."""

    @abstractmethod
    def refresh_feature_store(self) -> None:
        """Odświeża zmaterializowany widok ml_feature_store (Slow Loop, przed treningiem)."""

    @abstractmethod
    def get_accuracy_stats(self, days: int) -> dict[str, Any]:
        """Statystyki trafności ostatnich `days` dni z prediction_logs.

        `mean_accuracy` / `sample_count` / `correct_count` liczą wszystkie
        rozliczone predykcje, także SIDEWAYS („bez ruchu”) — z nich korzysta
        sizing. `directional_count` / `directional_correct_count` (WIA-29) to
        ta sama próba zawężona do prognoz z zakładem na kierunek. `n_eff`
        (PORT-01) to efektywna liczność całej próby po dniach UTC
        (`significance.effective_n`) — sizing wymaga jej do pełnej bandy.

        Ten i pozostałe odczyty track-recordu (`get_recently_resolved_predictions`,
        `get_resolved_predictions`, `get_resolved_for_calibration`) pomijają
        atrapy po awarii głównego LLM (flaga `llm_analysis_failed`, WIA-09).

        WIA-61 (za `session_close_settlement_enabled`, migracja 059): wiersze
        z zapisanym zamknięciem sesji dają `session_close_count`,
        `session_close_correct_count` (trafne na zamknięciu) i
        `session_close_quote_correct_count` (te same predykcje trafne na /quote)."""

    @abstractmethod
    def get_recently_resolved_predictions(
        self, hours: int, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Predykcje z ocenionym kierunkiem (`is_trend_correct`) UTWORZONE
        w ostatnich `hours` godzinach.

        `since` (WIA-19, migracja 029): okno to predykcje ROZLICZONE
        (`settled_at`) od `since` — Fast Loop podaje start cyklu, czyli „rozliczone
        w tym cyklu” — od najświeżej rozliczonej; `hours` nie gra wtedy roli.
        Przed migracją kolumny nie ma, więc wołający podaje `since` tylko za
        flagą `settled_at_windows_enabled`. None = okno po dacie utworzenia."""

    @abstractmethod
    def get_resolved_predictions(self, days: int) -> list[dict[str, Any]]:
        """Predykcje z ocenionym kierunkiem z ostatnich `days` dni — szersze okno
        niż `get_recently_resolved_predictions` (godziny), do analiz track-recordu.

        Ten sam kształt wierszy co `get_recently_resolved_predictions`
        (symbol, predicted_trend, is_trend_correct, timestamp, reasoning_text,
        correction_insights, price_at_prediction, actual_price_after_12h), więc
        `parse_resolved_predictions` parsuje oba (wiersz niesie też
        `data_quality_flags`). JEDEN odczyt zasila krzywą
        kapitału (paper-PnL), panel „lessons learned" (correction_insights) i
        inne sekcje track-recordu — bez dublowania zapytań. Pusta lista gdy brak
        danych / błąd po stronie wywołującego."""

    @abstractmethod
    def get_cached_fundamentals(self, symbol: str) -> Fundamentals | None:
        """Zwraca fundamentale z cache jeśli `fetched_at` mieści się w
        FUNDAMENTALS_CACHE_TTL_HOURS. Filtr TTL po stronie repo —
        adapter nie zna polityki świeżości."""

    @abstractmethod
    def save_fundamentals(self, symbol: str, fundamentals: Fundamentals) -> None:
        """Upsert jednego wiersza per symbol (nadpisuje poprzedni snapshot)."""

    @abstractmethod
    def get_fundamentals_fetched_at(self, symbols: Sequence[str]) -> dict[str, datetime]:
        """DAT-19: `fetched_at` wpisów cache fundamentów dla `symbols`, BEZ filtra
        TTL — rotacja refreshu wybiera po nim najdawniej odświeżone spółki.
        Symbol bez wpisu nie trafia do wyniku."""

    @abstractmethod
    def get_cached_earnings(self, symbol: str) -> CachedEarningsDate | None:
        """FIN-11 — data najbliższego raportu z `earnings_cache` (migracja 053), gdy
        wpis jest młodszy niż EARNINGS_CACHE_TTL_HOURS; filtr TTL po stronie repo."""

    @abstractmethod
    def save_earnings(self, entry: CachedEarningsDate) -> None:
        """FIN-11 — upsert daty raportu per symbol (Slow Loop odświeża cache)."""

    @abstractmethod
    def save_quota_alert(self, alert: QuotaAlert) -> None:
        """Persystuje pojedynczy alert wyczerpania limitu / subskrypcji.

        main_agent woła po cyklu dla każdego alertu zebranego przez
        `QuotaMonitor`. Tabela `quota_alerts` (migracja 010) jest
        audit trail; banner w mailu używa `get_recent_quota_alerts`.
        """

    @abstractmethod
    def get_recent_quota_alerts(self, hours: int) -> list[QuotaAlert]:
        """Zwraca alerty z ostatnich `hours` godzin, posortowane malejąco
        po `occurred_at`. Banner w mailu używa tej listy, by pokazać też
        alerty z poprzednich cykli, których jeszcze nie naprawiono."""

    @abstractmethod
    def find_similar_predictions(
        self, embedding: list[float], limit: int = 3, regime: str | None = None
    ) -> list[dict[str, Any]]:
        """Wyszukuje historyczne predykcje o najbardziej podobnym kontekście
        newsowym (similarity search nad `prediction_logs.embedding`, pgvector).

        Zasila RAG w `predict_node`: wstrzykuje "jak podobne sytuacje skończyły
        się w przeszłości" do promptu. Implementacja MUSI być graceful — gdy
        pgvector / RPC niedostępne, zwraca `[]` (predykcja działa bez RAG).
        Zwracane rekordy zawierają m.in. `news_summary`, `predicted_trend`,
        `is_trend_correct`, `correction_insights`.

        `regime` (Cross-Asset Vector Memory): gdy podany, retrieval preferuje
        analogi z tego samego reżimu rynku (tag zapisany przy embeddingu,
        migracja 017). None = bez filtra (zachowanie wsteczne)."""

    @abstractmethod
    def get_council_votes_for_prediction(
        self, prediction_id: str
    ) -> list[InvestorOpinion]:
        """Głosy rady zapisane dla danej predykcji (`council_votes`).

        Napędza counterfactual dissent-replay: gdy predykcja okazała się błędna,
        self-reflection sprawdza, który dysydent rady miał rację. Pusta lista =
        brak zapisanych głosów (rada padła / pominięta progiem)."""

    @abstractmethod
    def get_receipt(self, prediction_id: str) -> StoredReceipt | None:
        """WIA-56 — kwit decyzyjny jednej predykcji; None, gdy nie ma takiej
        predykcji. Odczyt darmowy, tylko do audytu („dlaczego?”)."""

    @abstractmethod
    def find_receipts(self, symbol: str, day: date) -> list[StoredReceipt]:
        """WIA-56 — kwity wszystkich predykcji symbolu z jednego dnia UTC,
        od najstarszej. Pusta lista = brak predykcji tego dnia."""

    @abstractmethod
    def get_resolved_for_calibration(self, days: int) -> list[dict[str, Any]]:
        """#4 — rozliczone predykcje z `id`, `confidence_score`,
        `is_trend_correct`, `predicted_trend`, `timestamp`, `reasoning_text` do
        oceny kalibracji (Slow Loop) i krzywych kalibracji (WIA-39).
        Pomija wiersze bez `confidence_score`/`is_trend_correct`."""

    @abstractmethod
    def save_calibration(
        self,
        prediction_id: str,
        score: float,
        insight: str,
        *,
        score_version: int | None = None,
    ) -> None:
        """#4 — zapisuje ocenę kalibracji pewności (sędzia-LLM, Slow Loop):
        `confidence_calibration` (0-1, czy deklarowana pewność była uzasadniona)
        + `calibration_insight` na `prediction_logs` (migracja 016).

        `score_version` (WIA-37) trafia do `calibration_score_version`
        (migracja 044); None = bez tego klucza w zapisie (wiersz v1)."""

    @abstractmethod
    def get_persona_accuracy(
        self, asset_type: AssetType, window_days: int = 90
    ) -> dict[str, float]:
        """#3 — waga każdej persony rady wg jej historycznej trafności
        (rekomendacja vs realny ruch ceny), zmapowana do ~[0.5, 1.5].
        `{investor_name: waga}`. Pusta mapa = brak danych / RPC niedostępne
        → ważenie wyłączone (każda persona waży 1.0)."""

    @abstractmethod
    def get_persona_track_record(
        self, window_days: int = 90
    ) -> dict[str, tuple[float, int]]:
        """#3 — surowy track record person rady pod leaderboard w raporcie:
        `{investor_name: (hit_rate, liczba_rozliczonych_głosów)}` (migracja 018).

        W odróżnieniu od `get_persona_accuracy` (zwraca gotową WAGĘ do
        konsensusu) tu wychodzi nieprzetworzony hit-rate ∈ [0, 1] RAZEM
        z wielkością próbki — bez niej nie da się odciąć szumu ani pokazać
        "68% (22 głosy)". Pusta mapa = brak danych / RPC niedostępne
        → sekcja leaderboardu sama się chowa."""

    @abstractmethod
    def get_deferred_diagnoses(self, limit: int) -> list[DeferredDiagnosis]:
        """WIA-53 — do `limit` najświeższych BŁĘDNYCH predykcji (`is_trend_correct`
        = false), których `correction_insights` to placeholder bez diagnozy
        (`UNDIAGNOSED_MISTAKE_PLACEHOLDERS`). Atrapy po awarii LLM odpadają."""

    @abstractmethod
    def save_backfilled_diagnosis(
        self,
        prediction_id: str,
        insight: str,
        *,
        replaces: str,
        correction_category: MistakeCategory | None = None,
    ) -> None:
        """WIA-53 — podmienia `correction_insights` na diagnozę, ale tylko gdy
        kolumna wciąż niesie `replaces` (inaczej nic nie zapisuje). Kategoria
        (WIA-57, migracja 046) jak w `update_prediction_accuracy`."""

    @abstractmethod
    def get_persona_vote_outcomes(self, window_days: int = 90) -> list[PersonaVoteOutcome]:
        """WIA-51 — rozliczone głosy person z okna `window_days` (po chwili
        predykcji), bez oceny: RPC `persona_vote_outcomes` (migracja 045).

        Trafność liczy domena (`persona_vote_correct`), a oś czasu
        `predicted_at` / `settled_at` zasila wagi prequential (WIA-46). Błąd
        RPC (np. brak migracji) leci do wołającego — tryb graceful mają
        `get_persona_accuracy` i `get_persona_track_record`."""

    @abstractmethod
    def get_resolved_predictions_detailed(
        self, days: int, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """#17 — jak `get_resolved_predictions`, ale z polami, których tamta NIE
        zwraca: `id`, `price_at_prediction`, `predicted_target_price`,
        `confidence_score`, `actual_price_after_12h` i `council_verdict` (JSONB).

        Recap tygodniowy potrzebuje ich, żeby policzyć accuracy, ECE i
        wybronionych dysydentów rady. Pusta lista = brak danych.

        `since` (WIA-19, migracja 029): jak w `get_recently_resolved_predictions`
        — predykcje ROZLICZONE od `since` zamiast utworzonych w ostatnich `days`
        dniach. None = okno po dacie utworzenia (działa bez migracji)."""

    @abstractmethod
    def get_settled_moves(
        self, days: int, *, with_intervals: bool = False
    ) -> list[dict[str, Any]]:
        """ML-34 — rozliczone predykcje z ostatnich `days` dni, ROSNĄCO po czasie:
        `symbol`, `timestamp`, `price_at_prediction`, `predicted_target_price`,
        `actual_price_after_12h`. Zasila okno kalibracyjne przedziału ruchu.

        `with_intervals` dokłada `interval_low` / `interval_high` do pomiaru
        pokrycia — WYMAGA migracji 032 (inaczej PostgREST zwraca 400).

        DAT-14: za `stale_quote_enabled` (migracja 055) bez rozliczeń z notowania
        sprzed sesji po predykcji — cena równa cenie wejścia to nie ruch."""

    @abstractmethod
    def get_unrevealed_commitments(self, days: int | None = None) -> list[dict[str, Any]]:
        """#16 — ROZLICZONE predykcje z commitmentem, jeszcze nieujawnione.

        Napędza sweep: bez niego predykcje symboli usuniętych z `config.toml`
        nigdy nie zostałyby ujawnione, a dla sceptyka wygląda to jak ukrywanie
        nietrafień. Pusta lista = nic do ujawnienia / brak migracji 024.
        WIA-03: domyślnie bez okna czasowego i bez filtra symboli; `days`
        zawęża do ostatnich dni (sweep go nie podaje).
        WIA-70: wiersz niesie `commitment_payload` (dosłowny plaintext, migracja
        043); przed migracją klucza nie ma, a stare wiersze mają None."""

    @abstractmethod
    def get_current_hour_commitment_salt(self, symbol: str) -> str | None:
        """WIA-01 — sól commitmentu wiersza `symbol` z bieżącej godziny UTC (klucz
        upsertu `save_prediction`). None = brak takiego wiersza albo błąd odczytu.

        Ponowny zapis w tej godzinie (retry LangGraph, re-run GHA) nadpisuje wiersz:
        z tą samą solą ten sam payload daje ten sam, już opublikowany hash."""

    @abstractmethod
    def get_current_hour_commitment(self, symbol: str) -> StoredCommitment | None:
        """WIA-01 — hash, sól i plaintext commitmentu wiersza `symbol` z bieżącej
        godziny UTC. None = brak wiersza z hashem i solą albo błąd odczytu.

        Inny payload nadpisze ten wiersz, więc graf ujawnia stary commitment jako
        „superseded", póki jego sól i plaintext są jeszcze w bazie."""

    @abstractmethod
    def attestation_columns_available(self) -> bool:
        """WIA-70 — czy baza ma kolumny commitmentu (`commitment_hash`,
        `commitment_salt` z 024 i `commitment_payload` z 043). False → cykl bez
        atestacji: zapis tych kluczy przed migracją dałby PGRST204 i zabił zapis
        całej predykcji."""

    @abstractmethod
    def mark_commitment_revealed(self, prediction_id: str) -> None:
        """#16 — stempluje `revealed_at`, żeby sweep nie ujawnił tego dwa razy."""

    @abstractmethod
    def get_latest_snapshot_at(self) -> datetime | None:
        """NIEZ-11 — znacznik najnowszego snapshotu ceny dowolnego symbolu: ślad
        życia Fast Loopa dla deadmana. None = brak snapshotów."""

    @abstractmethod
    def get_last_scorecard_at(self) -> datetime | None:
        """NIEZ-11 — znacznik najnowszego wiersza `model_scorecards`: ślad życia
        Slow Loopa (zapisuje wiersz per symbol także przy pominięciu). None = brak."""

    @abstractmethod
    def probe_database(self) -> None:
        """NIEZ-28 — odczyt jednego wiersza (tylko do odczytu): dowód, że baza
        odpowiada. Projekt Supabase na planie Free jest wstrzymywany po 7 dniach bez
        aktywności, a wtedy każdy job pada na połączeniu. Wyjątek = baza nie odpowiada."""

    @abstractmethod
    def missing_prediction_columns(self, columns: frozenset[str]) -> frozenset[str]:
        """FIN-05 — które z `columns` nie istnieją w `prediction_logs` (select bez
        wierszy). Brak kolumny to wyłącznie kod 42703; każdy inny błąd (sieć, timeout,
        wstrzymany projekt) leci wyjątkiem, żeby wołający nie wziął go za brak migracji."""

    @abstractmethod
    def get_last_price_snapshot(self, symbol: str) -> tuple[Money, datetime] | None:
        """#11 — najnowszy snapshot ceny WRAZ ze znacznikiem czasu.

        Alert szoku potrzebuje wieku punktu odniesienia: stęchły snapshot
        (>30h) daje fałszywy alarm, więc `detect_shock` go tłumi.
        None = brak historii."""

    @abstractmethod
    def save_shock_alert(
        self, symbol: str, alert_date: date, delta: float, direction: str
    ) -> None:
        """#11 — persystuje wysłany alert szoku (migracja 021).
        `UNIQUE(symbol, alert_date)` egzekwuje debounce po stronie bazy."""

    @abstractmethod
    def get_sent_shock_alerts(self, since: date) -> set[tuple[str, date]]:
        """#11 — pary (symbol, dzień) już zaalarmowane od `since`.
        Pusty zbiór = brak danych / brak migracji 021 → debounce w domenie
        polega wtedy wyłącznie na pamięci bieżącego przebiegu."""

    @abstractmethod
    def get_shock_alert_times(self, since: datetime) -> dict[str, datetime]:
        """ALR-02 — chwila NAJNOWSZEGO alertu szoku per symbol od `since`
        (`shock_alerts.created_at`, migracja 021). Watch tłumi nią alert krypto
        ze zmiany 24 h, póki okno zawiera ruch już zaalarmowany. Pusty słownik =
        brak alertów / brak migracji 021."""

    @abstractmethod
    def get_active_price_levels(self) -> list[PriceLevelAlert]:
        """ALR-11 — aktywne alerty poziomów cenowych z ostatnim odczytem watcha
        (tabela `price_level_alerts`, migracja 057). Adapter SUROWY — watch czyta
        tylko za flagą `price_level_alerts_enabled` i sam łapie błąd."""

    @abstractmethod
    def add_price_levels(self, levels: Sequence[PriceLevel]) -> None:
        """ALR-11 — dopisuje poziomy (lista startowa z config.toml). Poziom już
        zapisany (`UNIQUE(symbol, level, direction)`) zostaje bez zmian — także
        wyłączony po trafieniu, więc ponowne dopisanie go nie wskrzesza."""

    @abstractmethod
    def record_price_level_reading(self, alert_id: str, price: Decimal) -> None:
        """ALR-11 — zapisuje odczyt ceny jako `last_seen_price`: poprzedni odczyt
        następnego przebiegu (watch nie zapisuje `price_snapshots`)."""

    @abstractmethod
    def mark_price_level_triggered(
        self, alert_id: str, price: Decimal, triggered_at: datetime
    ) -> None:
        """ALR-11 — wyłącza alert po trafieniu (`active = false`, `triggered_at`,
        cena trafienia w `last_seen_price`)."""

    @abstractmethod
    def get_previous_verdict(self, symbol: str) -> PreviousVerdict | None:
        """#8 — ostatni ZAPISANY werdykt rady dla symbolu (JSONB `council_verdict`
        z migracji 005) wraz z sentymentem i znacznikiem czasu tamtego cyklu.

        Adapter jest SUROWY (bez try/except) — graceful degradation robi caller
        (`_fetch_report_context`), zgodnie ze stylem repo. None = brak historii."""

    @abstractmethod
    def save_model_scorecard(
        self, symbol: str, result: Mapping[str, Any], *, run_id: str | None = None
    ) -> None:
        """#12 — zapisuje jeden przebieg treningu walk-forward (migracja 019).

        Slow Loop zapisuje wiersz per symbol (ewaluacja bez zapisu wag) i jeden
        wiersz modelu wspólnego z `symbol = "ALL"` (ML-13), więc `symbol` jest
        wymagany — bez niego scorecardy są nierozróżnialne.
        `result` to surowe wyjście `MLPredictionPort.train()` albo wynik `skipped`.
        `run_id` (ML-24, migracja 049) — identyfikator przebiegu wspólny dla
        wszystkich jego wierszy; None = bez identyfikatora (grupuje heurystyka)."""

    @abstractmethod
    def get_recent_model_scorecards(self, days: int = 30) -> list[dict[str, Any]]:
        """#12 — scorecardy z ostatnich `days` dni, od najnowszego.
        Pusta lista = brak danych / brak migracji 019 → sekcja się chowa."""

    @abstractmethod
    def get_council_vote_history(
        self,
        symbols: list[str],
        *,
        window_days: int = 30,
        investors: list[str] | None = None,
    ) -> list[InvestorHistory]:
        """U5 — historia głosów rady per inwestor na obserwowanych symbolach
        z ostatnich `window_days` dni (jeden `InvestorHistory` na parę
        inwestor×symbol, rekomendacje od najnowszej). Napędza panel drill-down
        w raporcie. `investors=None` = wszyscy. Pusta lista gdy brak danych."""

    @abstractmethod
    def save_council_votes(
        self,
        prediction_id: str,
        symbol: str,
        votes: list[InvestorOpinion],
    ) -> None:
        """Strukturalny audit trail rady — jedna linia per inwestor.

        Umożliwia odpytywanie "jak Burry głosował na NVDA w ostatnim miesiącu"
        bez parsowania JSONB blob `prediction_logs.council_verdict`. Pusta lista
        głosów to no-op (rada padła całkowicie / została pominięta progiem).
        """

    @abstractmethod
    def save_llm_calls(self, calls: Sequence[LLMCallEntry]) -> None:
        """FIN-13 — ledger kosztów: jeden wiersz `llm_calls` (migracja 040) na
        request do dostawcy LLM / embeddingów, zapisany JEDNYM batchem po cyklu.
        Same tokeny, koszt i identyfikatory — bez treści promptu. Pusta lista =
        no-op. Wołający pilnuje flagi `cost_ledger_enabled` (bez migracji 040
        zapis kończyłby się błędem PostgREST)."""

    @abstractmethod
    def save_cycle_run(self, run: CycleRun) -> None:
        """NIEZ-29 — jeden wiersz `cycle_runs` (migracja 041) na przebieg pętli,
        jednym insertem na końcu przebiegu. Wołający pilnuje flagi
        `cycle_runs_enabled` (bez migracji 041 zapis kończyłby się błędem PostgREST)."""

    # WIA-24 — sygnał referencyjny w osobnej tabeli `reference_predictions` (migracja
    # 042), nigdy w `prediction_logs`. Wołający pilnuje flagi `reference_signal_enabled`.

    @abstractmethod
    def get_reference_reads(self, since: datetime) -> list[ReferenceRead]:
        """Odczyty sygnału referencyjnego wszystkich symboli od `since`, rosnąco po czasie."""

    @abstractmethod
    def get_open_reference_calls(self) -> list[OpenReferenceCall]:
        """Wywołania (BULLISH / BEARISH) jeszcze bez rozliczenia, bez limitu wieku —
        stare wygasają w kolejnym przebiegu."""

    @abstractmethod
    def save_reference_read(self, read: ReferenceRead, call: ReferenceCall | None) -> None:
        """Jeden wiersz: odczyt przebiegu i wywołanie z niego (None = abstynencja)."""

    @abstractmethod
    def settle_reference_call(
        self,
        record_id: str,
        outcome: ReferenceOutcome,
        settled_price: Decimal | None,
        settled_at: datetime,
    ) -> None:
        """Rozliczenie wywołania: HIT / MISS z ceną odczytu albo EXPIRED bez ceny."""


class MLPredictionPort(ABC):
    """Lokalny model predykcyjny (np. XGBoost)."""

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Czy model jest gotów do predykcji (wagi załadowane / wytrenowane).

        Fast Loop sprawdza to przed wywołaniem `predict()` — przy cold-starcie
        (brak pliku modelu) używa baseline zamiast crashować."""

    @abstractmethod
    def predict(
        self, current_features: dict[str, float], current_price: Decimal
    ) -> Money:
        """Predykcja ceny docelowej 12h. Model przewiduje ZWROT — `current_price`
        jest potrzebna do rekonstrukcji ceny bezwzględnej (`cena*(1+zwrot)`)."""

    @abstractmethod
    def explain(self, current_features: dict[str, float]) -> dict[str, float]:
        """Q3 — znakowane wkłady per-cecha (SHAP-lite) w surową predykcję.

        Zwraca `{nazwa_cechy: wkład, ..., "_bias": base}`. Lokalny CPU, zero
        płatnych wywołań. Rzuca RuntimeError gdy model nietrenowany (jak predict)."""

    @abstractmethod
    def train(
        self,
        features: Any,
        target: Any,
        *,
        persist: bool = True,
        timestamps: Any = None,
    ) -> TrainingReport:
        """`features`/`target` to `pandas.DataFrame`/`Series` — typujemy luźno,
        by nie wprowadzać zależności od pandas w warstwie application.

        Zwraca domenowy `TrainingReport` (ARCH-09): `ACCEPTED` / `REJECTED` przy
        zapisie wag, a przy `persist=False` — samej ewaluacji walk-forward, bez
        zapisu ani ładowania wag (ML-13, per-symbol pętla Slow Loopa) —
        `EVALUATION_PASSED` / `EVALUATION_FAILED`. `timestamps` (wyrównane
        z wierszami, rosnąco) włączają foldy po czasie GLOBALNYM z purgem horyzontu
        etykiety — wymagane dla modelu wspólnego na wielu symbolach."""

    @abstractmethod
    def walk_forward_report(
        self, features: Any, target: Any, *, timestamps: Any = None
    ) -> list[dict[str, float]]:
        """Per-fold metryki out-of-sample (walk-forward, expanding window) BEZ
        persystencji — zasila offline backtest harness (`main_backtest.py`).

        Reużywa tę samą logikę foldów co `train()` (finding #23), ale zwraca
        SZCZEGÓŁ każdego foldu zamiast średniej: lista dictów z kluczami
        `train_end`, `valid_end`, `n_train`, `n_valid`, `candidate_rmse`,
        `baseline_rmse` (zwrot=0), `hit_rate` (trafność kierunku) oraz (ML-20)
        `ic` (korelacja rang Spearmana), `sign_strategy_return` i `turnover`,
        a (EXP-04) `sum_loss_diff` / `sum_loss_diff_sq` względem „zwrot 0” i te
        same sumy z prefiksem `drift_` / `llm_` dla testu sparowanego w domenie.
        `timestamps` działają jak w `train()`: purge horyzontu etykiety w foldach.
        Czysto ewaluacyjna — nie zapisuje modelu, nie zmienia stanu adaptera."""


# NIEZ-44: klucze, którymi adapter failover znakuje wynik `LLMPort.analyze` od
# dostawcy ZAPASOWEGO (dostawca i model, który faktycznie odpowiedział). Graf zdejmuje
# je przed normalizacją analizy i dopisuje flagę jakości `llm_failover`.
LLM_FAILOVER_PROVIDER_KEY = "_llm_provider"
LLM_FAILOVER_MODEL_KEY = "_llm_model"


class LLMPort(ABC):
    """Model językowy do rozumowania i samokorekty (np. OpenAI, Anthropic)."""

    @abstractmethod
    def analyze(self, prompt: str) -> dict[str, Any]:
        """Główna analiza — zwraca ustrukturyzowany JSON wg output_schema z promptu.

        Wynik od dostawcy zapasowego (NIEZ-44) niesie dodatkowo
        `LLM_FAILOVER_PROVIDER_KEY` i `LLM_FAILOVER_MODEL_KEY`."""

    @abstractmethod
    def analyze_mistake(self, prompt: str) -> str:
        """Diagnoza błędnej predykcji — zwraca tekstowy correction_insight."""


class UsageSink(ABC):
    """FIN-13: odbiorca zużycia tokenów zgłaszanego przez adaptery LLM i embeddingów.

    Adapter oddaje samo `LLMUsage` z odpowiedzi dostawcy — nie zna Supabase,
    cennika ani grafu. Źródło cennika, węzeł i symbol dopisuje implementacja
    w application (`UsageLedger`) z zakresu wołającego."""

    @abstractmethod
    def record(self, usage: LLMUsage) -> None: ...


class EmbeddingPort(ABC):
    """Generowanie wektorów embeddingu dla tekstu (np. OpenAI embeddings).

    Wektor podsumowania newsów trafia do `prediction_logs.embedding` (pgvector),
    co umożliwia przyszły similarity search nad historycznymi sytuacjami."""

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...


class ReportNotifierPort(ABC):
    """Wysyłka raportów z cyklu (np. email przez Resend).

    `silent` (ALR-09): wiadomość ma dotrzeć bez dźwięku — cisza nocna pushy watcha.
    Kanał bez takiej opcji (mail) wysyła to samo co zwykle. Alerty CRITICAL
    operacyjne wołają bez `silent`, więc cisza ich nie dotyczy."""

    @abstractmethod
    def send_report(
        self, subject: str, html_body: str, plain_text: str, *, silent: bool = False
    ) -> None: ...


class HeartbeatPort(ABC):
    """NIEZ-13: zewnętrzny watchdog żywotności cronów (np. healthchecks.io).

    Watchdog na tym samym GitHub Actions umiera razem z nim, a GitHub wyłącza
    crony po 60 dniach bez aktywności repo — o ciszy musi powiedzieć ktoś z
    zewnątrz. Entry point pinguje RAZ, na końcu przebiegu: `ok=True` = przebieg
    zakończony kodem 0, `ok=False` = porażka. Brak pingu w oknie checku to alarm."""

    @abstractmethod
    def ping(self, ok: bool) -> None: ...


class AlertNotifierPort(ABC):
    """Real-time push pojedynczych alertów CRITICAL — odsprzężony od dziennego maila.

    Pozwala wystawić alert (np. wyczerpanie limitu, email nie poszedł) natychmiast
    w cyklu, nie czekając na następny raport dobowy. Adapter formatuje alerty i
    deleguje do dowolnego transportu (Telegram / Slack / Resend)."""

    @abstractmethod
    def send_alert(self, alerts: list[QuotaAlert]) -> None: ...


class AdvisoryCouncilPort(ABC):
    """Rada doradcza inwestorów — N równoległych analiz person + konsensus.

    Liczba person jest data-driven (pliki JSON w `council_personas_dir`),
    więc nie podajemy jej tu na sztywno."""

    @abstractmethod
    def analyze(self, symbol: str, data: CouncilInput) -> CouncilVerdict: ...


class FundamentalsPort(ABC):
    """Źródło danych fundamentalnych spółek (P/E, PEG, EPS growth).

    Zwraca None dla aktywów bez sensownych fundamentów (ETF-y, błąd API,
    brak danych w źródle)."""

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Fundamentals | None: ...


class MacroIndicatorsPort(ABC):
    """Wskaźniki makroekonomiczne polskiego rynku (kurs PLN, ew. rentowności).

    Zwraca None, gdy źródło niedostępne — wywołujący traktuje brak danych
    jako "neutralne tło" i nie blokuje raportu."""

    @abstractmethod
    def fetch_polish_macro(self) -> PolishMacroSnapshot | None: ...


class FxRatePort(ABC):
    """PORT-11 — kurs średni waluty do PLN z danego dnia (NBP, tabela A).

    Dzień bez tabeli (weekend, święto, przed publikacją) → kurs z poprzedniego
    dnia roboczego (`FxRate.effective_date`). None = źródło niedostępne albo
    waluta spoza tabeli; wołający pokazuje wtedy brak kursu, nigdy kurs innej
    waluty. Darmowe (NBP bez klucza), poza bramką volatility."""

    @abstractmethod
    def get_rate(self, currency: str, on_date: date) -> FxRate | None: ...


class SubscriberRepositoryPort(ABC):
    """Źródło listy subskrybentów raportu (U2 — personalizowany fan-out).

    Każdy subskrybent ma własny podzbiór symboli; analiza leci RAZ nad unią,
    a raport jest tylko tnięty per odbiorca. Zwraca [] gdy brak / błąd."""

    @abstractmethod
    def list_subscribers(self) -> list[Subscriber]: ...


class PortfolioPositionsPort(ABC):
    """#15 — realne pozycje użytkownika (tabela `positions`, migracja 023).

    Bez nich cała warstwa ryzyka liczy fikcję na RÓWNYCH wagach. Pusta lista =
    brak pozycji / port wyłączony → fallback na dotychczasowe równe wagi."""

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_as_of(self) -> datetime | None:
        """Znacznik czasu najświeższej pozycji. None = brak danych.
        Napędza badge STALE — stęchły portfel daje FAŁSZYWY P/L, gorszy niż
        brak sekcji, więc świeżość jest wymaganiem, nie ozdobnikiem."""


class PositionsImportPort(ABC):
    """PORT-18 — zapis tabeli `positions` z importu CSV (`main_tools.py
    import-positions`). W przeciwieństwie do odczytu dla raportu jest ŚCISŁY:
    błąd to wyjątek, bo dry-run na pustej liście pokazałby „wszystko nowe”."""

    @abstractmethod
    def current_positions(self) -> list[Position]:
        """Bieżące loty z tabeli (te same, które czyta raport)."""

    @abstractmethod
    def replace_positions(self, positions: Sequence[Position], as_of: datetime) -> None:
        """Zastępuje całą zawartość tabeli podanymi lotami z jednym `as_of`."""


class AttestationPublisherPort(ABC):
    """#16 — publikacja commitmentów i revealów (append-only JSONL).

    Lustro `WebPublisherPort`: zwraca ścieżkę/URL albo "" gdy wyłączone lub
    błąd. Awaria publikacji NIGDY nie wywala cyklu."""

    @abstractmethod
    def publish_commitment(self, record: Mapping[str, Any]) -> str: ...

    @abstractmethod
    def publish_reveal(self, record: Mapping[str, Any]) -> str: ...


class WebPublisherPort(ABC):
    """Publikacja raportu HTML jako statyczny artefakt web (U3 — read-only
    dashboard, alternatywa dla maila). Zwraca URL/ścieżkę albo "" gdy
    wyłączone / błąd (publikacja nigdy nie wywala cyklu)."""

    @abstractmethod
    def publish(self, html: str) -> str: ...


@dataclass(frozen=True)
class Tool:
    """#6 — read-only narzędzie udostępniane modelowi w pętli tool-use.

    `func` przyjmuje argumenty wywołania (dict) i zwraca JSON-safe wynik (dict).
    `parameters` to JSON Schema argumentów (dla SDK function-calling)."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[[dict[str, Any]], dict[str, Any]]


class ToolUseLLMPort(ABC):
    """#6 — LLM z pętlą tool-use: model może na żądanie wywołać read-only toole
    (fundamenty, RAG, makro) zanim wyda werdykt. Implementacja MUSI mieć twardy
    cap rund (FinOps). Zwraca ustrukturyzowany JSON jak `LLMPort.analyze`."""

    @abstractmethod
    def analyze_with_tools(
        self, prompt: str, tools: Sequence[Tool]
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Nowe źródła alpha (kategoria „Dane"). Każdy port ma adapter (wolne źródło)
# i Null-adapter (default Fast Loop / brak klucza → zero płatnych wywołań).
# ---------------------------------------------------------------------------


class InsiderFlowPort(ABC):
    """Przepływy insiderów (SEC Form 4 / 13F) per symbol.

    Zwraca None gdy brak danych / źródło niedostępne. Null-adapter (Fast Loop /
    brak konfiguracji) zwraca None bez żadnego wywołania sieciowego."""

    @abstractmethod
    def get_insider_flow(self, symbol: str) -> InsiderFlowSnapshot | None: ...


class EarningsCalendarPort(ABC):
    """Najbliższe zdarzenie wyników (earnings) per symbol.

    Zwraca None gdy brak nadchodzącego raportu / brak danych."""

    @abstractmethod
    def get_next_earnings(self, symbol: str) -> EarningsEvent | None: ...


class OptionsFlowPort(ABC):
    """Przepływy opcji i IV per symbol (put/call ratio, implied vol).

    Brak darmowego źródła → domyślnie Null-adapter. Zwraca None gdy brak danych."""

    @abstractmethod
    def get_options_flow(self, symbol: str) -> OptionsFlowSnapshot | None: ...


class SocialVelocityPort(ABC):
    """Prędkość społeczna (Reddit/X) per symbol — wzmianki 24h vs baseline.

    Domyślnie Null-adapter (free-tier/OAuth opcjonalne). Zwraca None gdy brak."""

    @abstractmethod
    def get_social_velocity(self, symbol: str) -> SocialVelocitySnapshot | None: ...


class MacroRatesPort(ABC):
    """Stopy i krzywa rentowności (FRED) — raz na cykl, nie per symbol.

    Zwraca None gdy źródło niedostępne / brak klucza."""

    @abstractmethod
    def fetch_yield_curve(self) -> YieldCurveSnapshot | None: ...


class MacroSeriesPort(ABC):
    """DAT-22: zmienne stanu z FRED (VIXCLS, VXNCLS, BAMLH0A0HYM2, T10Y2Y, NFCI) —
    raz na cykl, nie per symbol.

    Obserwacje z dni <= `as_of` (point-in-time), każda z dniem obserwacji, nie
    pobrania. None gdy źródło niedostępne / brak klucza."""

    @abstractmethod
    def fetch_macro_series(self, as_of: date) -> MacroSeriesSnapshot | None: ...


class AnalystConsensusPort(ABC):
    """Konsensus analityków i cel cenowy per symbol.

    Zwraca None gdy brak pokrycia analitycznego / źródło niedostępne."""

    @abstractmethod
    def get_consensus(self, symbol: str) -> AnalystConsensus | None: ...


class ModelCatalogPort(ABC):
    """NIEZ-25: katalog modeli dostawcy LLM (Models API — nie zużywa tokenów).

    `validate_model_id` sprawdza tylko prefiks id, więc wycofany model kończy się
    cichym total outage dopiero przy pierwszym wywołaniu. Sonda Slow Loopa pyta
    katalog o każdy skonfigurowany model raz w tygodniu."""

    @abstractmethod
    def is_available(self, model_id: str) -> bool:
        """True = dostawca zna model, False = nie zna (404). Błąd sieci albo
        uprawnień klucza to wyjątek, nie False — to nie jest brak modelu."""


class EvidenceSourcePort(ABC):
    """NIEZ-32: odczyt tabel dowodowych (track record) do kopii zapasowej."""

    @abstractmethod
    def read_rows(self, table: str, columns: Sequence[str]) -> list[dict[str, Any]]:
        """Wszystkie wiersze `table` (paginacja), wyłącznie kolumny `columns`.
        Kolumny, której baza jeszcze nie ma (brak migracji), w wierszach nie ma."""


@dataclass(frozen=True)
class ArchiveReceipt:
    """NIEZ-32: zapisany (zaszyfrowany) artefakt kopii: ścieżka, sha256, rozmiar."""

    path: str
    sha256: str
    size_bytes: int


class EvidenceArchivePort(ABC):
    """NIEZ-32: magazyn kopii — szyfruje PRZED zapisem; jawny tekst nie trafia na dysk."""

    @abstractmethod
    def store(self, name: str, files: Mapping[str, bytes]) -> ArchiveReceipt:
        """Pakuje `files` (nazwa pliku → treść) w jeden zaszyfrowany artefakt `name`."""


@dataclass(frozen=True)
class CircuitState:
    """NIEZ-31: stan circuit breakera jednego hosta HTTP w tym procesie.

    `source` — nazwa dostawcy dla człowieka („Finnhub”), dla nieznanego hosta sam
    host. `skipped_symbols` — symbole, których żądania otwarty obwód pominął bez
    ruchu sieciowego; `skipped_calls` liczy też żądania spoza kontekstu symbolu."""

    source: str
    host: str
    is_open: bool
    consecutive_failures: int
    skipped_calls: int
    skipped_symbols: frozenset[str]


class CircuitStatePort(ABC):
    """NIEZ-31: odczyt stanu obwodów HTTP — banner kwot na koniec cyklu i linijka
    stanu każdego źródła w mailu (MAIL-21)."""

    @abstractmethod
    def circuit_states(self) -> tuple[CircuitState, ...]:
        """Obwody hostów, z którymi proces rozmawiał (otwarte i zamknięte)."""

