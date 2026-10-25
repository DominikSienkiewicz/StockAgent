from __future__ import annotations

import dataclasses
import logging
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, cast

from langgraph.graph.state import CompiledStateGraph

from src.application.agent_graph import (
    AgentGraphDeps,
    AgentState,
    create_agent_graph,
    create_gate_graph,
    create_paid_graph,
    required_prediction_columns,
)
from src.application.cycle_telemetry import CycleTelemetry
from src.application.market_features import CycleQuoteCache, collect_market_features
from src.application.ports import (
    AttestationPublisherPort,
    LatestPrediction,
    NewsPort,
    SentimentPort,
)
from src.domain.alpha_fusion import AlphaFusionScore
from src.domain.asset import DRIFT_WINDOW_DAYS, Asset, drift_reference, is_reference_stale
from src.domain.corporate_events import CorporateEventSnapshot
from src.domain.experiment import active_arms
from src.domain.value_objects import AssetType, Money

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateOutcome:
    """EXP-27: stan symbolu po fazie 1. `passed` = czeka na fazę płatną; inaczej
    symbol skończył cykl (pod progiem: snapshot zapisany, `status` ustawiony)."""

    state: AgentState

    @property
    def passed(self) -> bool:
        return "status" not in self.state


class PaidPhase:
    """EXP-27: skompilowana faza płatna cyklu z portami feedu fazy 2."""

    def __init__(self, app: CompiledStateGraph[AgentState]) -> None:
        self._app = app

    def run(
        self, state: AgentState, *, peer_context: tuple[tuple[str, str], ...] = ()
    ) -> dict[str, Any]:
        """Kończy symbol za bramką od stanu fazy 1. `peer_context` (#5) liczy DI
        dopiero w fazie 2, z werdyktów symboli przeanalizowanych wcześniej."""
        phase_two_state: AgentState = {**state, "peer_context": peer_context}
        result: dict[str, Any] = self._app.invoke(phase_two_state)
        return result


class AnalyzeMarketUseCase:
    """Fast Loop — pełen cykl decyzyjny dla pojedynczego symbolu.

    Spina:
    1. odczyt poprzedniej ceny z repozytorium (FIN-14: wraz z jej wiekiem),
    2. uruchomienie grafu LangGraph (check_price → reflect/sentiment/news/predict/save).
    """

    def __init__(self, deps: AgentGraphDeps) -> None:
        if deps.market_features_enabled:
            # ML-31: faza cech pobiera kurs przed grafem, a `check_price` go reużywa.
            deps = dataclasses.replace(deps, market_port=CycleQuoteCache(deps.market_port))
        self._deps = deps
        self._repository = deps.repository_port
        # Telemetria CAŁEGO cyklu (ARCH-11) — jedna na use case, nie na symbol.
        # Wystawiamy TEN SAM egzemplarz, który dostaną węzły grafu (`deps` idzie
        # do fabryki nietknięty), inaczej licznik płatnych wywołań zostałby pusty
        # i raport FinOps pokazałby zerowy koszt cyklu.
        self.telemetry: CycleTelemetry = deps.telemetry
        self._ml_port = deps.ml_port
        self._experiments = deps.experiments
        workflow = create_agent_graph(deps)
        # Kompilacja jest deterministyczna (zależy tylko od topologii + portów),
        # więc kompilujemy RAZ tutaj i reużywamy aplikację w każdym run().
        # Inaczej przy 43 symbolach na cykl rekompilowalibyśmy graf 43×.
        self._app = workflow.compile()
        # EXP-27: graf fazy 1 kompilowany raz, dopiero gdy cykl dwufazowy go użyje
        # (faza 1 idzie równolegle, stąd lock).
        self._gate_app: CompiledStateGraph[AgentState] | None = None
        self._gate_app_lock = threading.Lock()
        # ML-31: cechy rynkowe per symbol z `prepare_market_features` (raz na cykl).
        self._market_features: dict[str, dict[str, Any]] = {}

    @property
    def paid_calls(self) -> Counter[str]:
        """Żywy licznik płatnych wywołań cyklu (`telemetry.paid_calls`); spójny
        odczyt wszystkich metryk daje `telemetry.snapshot()`."""
        return self.telemetry.paid_calls

    @property
    def ml_model_loaded(self) -> bool:
        """G03: czy port ML ma wagi. Raport bierze stąd fakt o cold-starcie,
        zamiast zgadywać go z porównania ceny docelowej z bieżącą."""
        return self._ml_port.is_trained

    @property
    def attestation_publisher(self) -> AttestationPublisherPort | None:
        """WIA-60: publisher commitmentów grafu (None = atestacja off) — DI domyka nim
        cykl jedną linią rootu Merkle'a po przebiegu symboli."""
        return self._deps.attestation_publisher

    def required_prediction_columns(self) -> frozenset[str]:
        """FIN-05: kolumny `prediction_logs`, które zapis predykcji tego cyklu może
        dotknąć — DI sprawdza je w bazie raz, przed pierwszym symbolem."""
        return required_prediction_columns(self._deps)

    def prepare_market_features(
        self,
        symbols: Sequence[str],
        *,
        asset_types: Mapping[str, AssetType] | None = None,
        macro_features: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        """ML-31: cechy rynkowe symboli cyklu — RAZ, przed pierwszym `run`/`run_gate`.

        Bez flag `market_features_enabled` i `feature_snapshots_enabled` nic nie robi
        (zero odczytów — cechy i tak nie miałyby dokąd trafić). Z nimi czyta kurs
        (zapamiętany dla `check_price`) i historię snapshotów każdego symbolu; zero
        płatnych wywołań. Błąd per symbol daje puste cechy. `asset_types` (ML-32,
        brak = STOCK) to klasy aktywów symboli do mediany klasy i szerokości rynku;
        `macro_features` (DAT-22) to cechy makro cyklu, te same dla każdego symbolu."""
        if not (self._deps.market_features_enabled and self._deps.feature_snapshots_enabled):
            return
        self._market_features = collect_market_features(
            symbols,
            market_port=self._deps.market_port,
            repository=self._repository,
            as_of=now if now is not None else datetime.now(UTC),
            asset_types=asset_types or {},
            gate_thresholds=self._base_gate_thresholds(),
            macro_features=macro_features,
        )

    def _base_gate_thresholds(self) -> dict[AssetType, float]:
        """ML-32: bazowy próg bramki klasy aktywa (bez mnożników reżimu i earnings)."""
        base = float(self._deps.threshold.value)
        crypto = self._deps.crypto_threshold
        thresholds = dict.fromkeys(AssetType, base)
        thresholds[AssetType.CRYPTO] = float(crypto.value) if crypto is not None else base
        return thresholds

    def run(
        self,
        symbol: str,
        asset: Asset | None = None,
        *,
        regime_context: str = "",
        regime_multiplier: float = 1.0,
        earnings_multiplier: float = 1.0,
        alpha_fusion: AlphaFusionScore | None = None,
        peer_context: tuple[tuple[str, str], ...] = (),
        now: datetime | None = None,
        corporate_events: CorporateEventSnapshot | None = None,
    ) -> dict[str, Any]:
        """`now` to zegar cyklu (orkiestrator podaje `started_at`); od niego liczy
        się wiek referencji ceny (FIN-14) i dzień przydziału ramion eksperymentów
        (EXP-05). None = bieżący czas."""
        initial_state = self._initial_state(
            symbol,
            asset,
            regime_context=regime_context,
            regime_multiplier=regime_multiplier,
            earnings_multiplier=earnings_multiplier,
            alpha_fusion=alpha_fusion,
            peer_context=peer_context,
            now=now,
            corporate_events=corporate_events,
        )
        result: dict[str, Any] = self._app.invoke(initial_state)
        return result

    def run_gate(
        self,
        symbol: str,
        asset: Asset | None = None,
        *,
        regime_context: str = "",
        regime_multiplier: float = 1.0,
        earnings_multiplier: float = 1.0,
        alpha_fusion: AlphaFusionScore | None = None,
        now: datetime | None = None,
        corporate_events: CorporateEventSnapshot | None = None,
    ) -> GateOutcome:
        """EXP-27 faza 1: cena, reflect, fundamenty z cache i bramka domenowa —
        ten sam stan wejściowy co `run`, bez peerów (liczy je faza 2)."""
        initial_state = self._initial_state(
            symbol,
            asset,
            regime_context=regime_context,
            regime_multiplier=regime_multiplier,
            earnings_multiplier=earnings_multiplier,
            alpha_fusion=alpha_fusion,
            peer_context=(),
            now=now,
            corporate_events=corporate_events,
        )
        # Stan końcowy grafu to AgentState (ten sam schemat); stuby typują go szerzej.
        final = cast(AgentState, self._compiled_gate_graph().invoke(initial_state))
        return GateOutcome(state=final)

    def paid_phase(self, *, sentiment_port: SentimentPort, news_port: NewsPort) -> PaidPhase:
        """EXP-27 faza 2 na porty feedu zbudowane po fazie 1 (klient AV tylko dla
        symboli za bramką). Reszta zależności, w tym telemetria cyklu, ta sama."""
        deps = dataclasses.replace(
            self._deps, sentiment_port=sentiment_port, news_port=news_port
        )
        return PaidPhase(create_paid_graph(deps).compile())

    def _compiled_gate_graph(self) -> CompiledStateGraph[AgentState]:
        with self._gate_app_lock:
            if self._gate_app is None:
                self._gate_app = create_gate_graph(self._deps).compile()
            return self._gate_app

    def _initial_state(
        self,
        symbol: str,
        asset: Asset | None,
        *,
        regime_context: str,
        regime_multiplier: float,
        earnings_multiplier: float,
        alpha_fusion: AlphaFusionScore | None,
        peer_context: tuple[tuple[str, str], ...],
        now: datetime | None,
        corporate_events: CorporateEventSnapshot | None = None,
    ) -> AgentState:
        clock = now if now is not None else datetime.now(UTC)
        previous = self._repository.get_last_price(symbol)
        reference_at = self._reference_at(symbol) if previous else None
        stale_reference = is_reference_stale(reference_at, clock)

        # Cold start: brak historii → previous=0 → delta=0 (guard w domenie) → ignore.
        initial_state: AgentState = {
            "symbol": symbol,
            "previous_price": previous.amount if previous else Decimal("0"),
            # FIN-14: stęchła referencja (pauza) zamyka bramkę w domenie — cykl
            # zapisuje nowy punkt odniesienia bez płatnych wywołań.
            "reference_at": reference_at,
            "stale_reference": stale_reference,
            # #7 reżim (label + mnożnik progu) i #5 contagion (peery z tego cyklu).
            "regime_context": regime_context,
            "regime_multiplier": regime_multiplier,
            # #6 bramka earnings — mnożnik progu z domeny (zawsze >= 1.0).
            "earnings_multiplier": earnings_multiplier,
            # #14 composite alfa — do promptu LLM i jako kandydat na cechę ML
            # (zapis w prediction_logs, poza `ML_FEATURE_COLUMNS`).
            "alpha_fusion": alpha_fusion,
            "peer_context": peer_context,
        }
        # Przekazujemy asset z klasyfikacją (STOCK/ETF), gdy dostępny z zewnątrz.
        if asset is not None:
            initial_state["asset"] = asset
        # ALR-10: referencja dryfu tylko za flagą; brak = bramka jak przed ALR-10.
        if self._deps.drift_gate_enabled:
            reference = self._drift_reference(symbol, clock)
            if reference is not None:
                initial_state["drift_reference"] = reference
        market_features = self._market_features.get(symbol)
        if market_features is not None:
            initial_state["market_features"] = dict(market_features)
        # DAT-17: tylko świeże 8-K (domena liczy wiek od dnia cyklu); brak snapshotu
        # (flaga off, ETF, EDGAR niedostępny) = brak klucza, czyli cecha „nie wiem”.
        if corporate_events is not None:
            initial_state["fresh_corporate_events"] = corporate_events.fresh_events(
                clock.date()
            )
        # EXP-05: ramiona aktywnych eksperymentów per symbol-dzień UTC cyklu.
        arms = active_arms(self._experiments, symbol, _utc_day(clock))
        if arms:
            logger.info("experiment arms for %s: %s", symbol, arms)
            initial_state["experiment_arms"] = arms
        return initial_state

    def _drift_reference(self, symbol: str, clock: datetime) -> Decimal | None:
        """ALR-10: cena odniesienia dryfu z darmowych odczytów Supabase (ostatnia
        predykcja, snapshoty z okna). Fail-closed: błąd albo odczyt spoza kontraktu
        portu → None, czyli bramka bez dryfu — nigdy dodatkowa płatna analiza."""
        try:
            latest = self._repository.get_latest_prediction(symbol)
            history = self._repository.get_price_history(symbol, DRIFT_WINDOW_DAYS)
        except Exception:
            logger.warning(
                "ALR-10: nie odczytano referencji dryfu %s — bramka bez dryfu",
                symbol,
                exc_info=True,
            )
            return None
        prediction = (
            (latest.timestamp, latest.price.amount)
            if isinstance(latest, LatestPrediction)
            else None
        )
        return drift_reference(prediction, _price_points(history), clock)

    def _reference_at(self, symbol: str) -> datetime | None:
        """FIN-14: moment zapisu referencji ceny (darmowy odczyt `price_snapshots`).

        Best-effort jak reszta odczytów pomocniczych: błąd odczytu nie wywala
        symbolu, tylko zostawia wiek nieznany — bramka działa wtedy jak przed
        FIN-14 (sam próg volatility)."""
        try:
            snapshot = self._repository.get_last_price_snapshot(symbol)
            return snapshot[1] if snapshot is not None else None
        except Exception:
            logger.warning(
                "FIN-14: nie odczytano wieku referencji ceny %s — bramka bez "
                "reguły stęchłej referencji",
                symbol,
                exc_info=True,
            )
            return None


def _price_points(history: object) -> list[tuple[datetime, Decimal]]:
    """Punkty historii cen w kształcie kontraktu portu; reszta odpada (fail-closed)."""
    if not isinstance(history, list):
        return []
    return [
        (stamp, price.amount)
        for stamp, price in (point for point in history if isinstance(point, tuple))
        if isinstance(stamp, datetime) and isinstance(price, Money)
    ]


def _utc_day(clock: datetime) -> date:
    """Dzień UTC zegara cyklu — ten sam klucz cyklu co ewaluacja (`timestamp` UTC).
    Zegar bez strefy traktujemy jak UTC."""
    return clock.astimezone(UTC).date() if clock.tzinfo is not None else clock.date()
