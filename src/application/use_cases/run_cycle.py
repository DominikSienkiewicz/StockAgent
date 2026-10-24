"""ARCH-07: przebieg symboli jednego cyklu Fast Loopa.

`RunFastLoopCycle` prowadzi listę symboli przez `AnalyzeMarketUseCase`:
sekwencyjnie (contagion #5, throttle między symbolami), równolegle w puli
wątków albo dwufazowo (EXP-27). Wejścia grafu liczone przed nim — klasa
aktywa, mnożnik earnings (#6), sygnały alfa i ich fuzja (#14) — też żyją tutaj.
Pojedynczy błąd symbolu nie przerywa cyklu: wynik dostaje `is_failure`.

Konfigurację przebiegu niesie jeden `CycleOptions` zamiast 9–11 kwargs na
każdej ścieżce. Zależności od świata (`sleep`, redakcja sekretów w komunikacie
błędu, kontekst logu symbolu) wstrzykuje composition root — application nie
importuje infrastruktury.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence, Set
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from typing import Any, NamedTuple

from src.application.agent_graph import AgentState
from src.application.alpha_signals import AlphaSignals
from src.application.errors import PortError
from src.application.ports import NewsPort, SentimentPort
from src.application.report_builder import SymbolResult, to_symbol_result
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase, PaidPhase
from src.application.use_cases.enrich_alpha import AlphaEnrichmentUseCase
from src.domain.alpha_fusion import AlphaFusionScore, fuse_alpha_signals
from src.domain.asset import Asset
from src.domain.earnings import EarningsEvent, earnings_threshold_multiplier
from src.domain.finops import rank_by_magnitude
from src.domain.peer_group import build_peer_context
from src.domain.value_objects import AssetType

logger = logging.getLogger(__name__)

# EXP-27: fabryka feedu fazy 2 — sentyment i newsy tylko dla symboli za bramką.
PhaseTwoFeeds = Callable[[Sequence[str]], tuple[SentimentPort, NewsPort]]
# Kontekst logu jednego symbolu (NIEZ-21); composition root podaje `symbol_context`.
SymbolScope = Callable[[str], AbstractContextManager[None]]


def classify_asset(symbol: str, crypto_set: Set[str], etf_set: Set[str]) -> AssetType:
    """STOCK/ETF/CRYPTO — pomijanie fundamentali dla ETF/CRYPTO, osobne progi CRYPTO."""
    if symbol in crypto_set:
        return AssetType.CRYPTO
    if symbol in etf_set:
        return AssetType.ETF
    return AssetType.STOCK


@dataclass(frozen=True)
class CycleOptions:
    """Ustawienia przebiegu symboli jednego cyklu, mapowane z `Settings`.

    `earnings_gate_enabled` (#6) i `alpha_fusion_enabled` (#14) domyślnie
    WYŁĄCZONE — wtedy ścieżka jest bit-w-bit taka jak przed nimi. Równolegle
    tylko przy `symbol_concurrency > 1` i bez contagion (#5 zależy od kolejności).
    `cycle_at` to start cyklu: od niego liczy się wiek referencji ceny (FIN-14)
    i odznaki proweniencji (ARCH-26).
    """

    regime_context: str = ""
    regime_multiplier: float = 1.0
    collect_alpha: bool = False
    earnings_gate_enabled: bool = False
    alpha_fusion_enabled: bool = False
    cycle_at: datetime | None = None
    crypto_symbols: frozenset[str] = frozenset()
    etf_symbols: frozenset[str] = frozenset()
    symbol_concurrency: int = 1
    contagion_enabled: bool = False
    peer_groups: Mapping[str, Sequence[str]] = field(default_factory=dict)
    throttle_seconds: float = 0.0

    @property
    def parallel(self) -> bool:
        """Równolegle tylko przy concurrency > 1 i bez contagion (#5 zależy od kolejności)."""
        return self.symbol_concurrency > 1 and not self.contagion_enabled


class ProcessedSymbol(NamedTuple):
    """Wynik przetworzenia jednego symbolu — niezależny od ścieżki (seq/parallel)."""

    result: SymbolResult
    is_failure: bool
    alpha_signal: AlphaSignals | None
    alpha_price: Decimal | None


class CycleOutcome(NamedTuple):
    """Wyniki w kolejności wejścia, liczba błędów oraz alfa i jej ceny per symbol."""

    results: list[SymbolResult]
    failures: int
    alpha_signals: dict[str, AlphaSignals]
    alpha_prices: dict[str, Decimal]


class _SymbolInputs(NamedTuple):
    """Wejścia grafu liczone PRZED nim: klasa aktywa, mnożnik earnings, alfa."""

    asset: Asset
    earnings_multiplier: float
    alpha_signal: AlphaSignals | None
    alpha_fusion: AlphaFusionScore | None


class _GatedSymbol(NamedTuple):
    """EXP-27: symbol po fazie 1 — skończony (`done`) albo czekający na fazę 2."""

    symbol: str
    done: ProcessedSymbol | None
    state: AgentState | None
    alpha_signal: AlphaSignals | None


def _outcome(order: Sequence[str], processed: Sequence[ProcessedSymbol]) -> CycleOutcome:
    return CycleOutcome(
        [p.result for p in processed],
        sum(1 for p in processed if p.is_failure),
        {
            sym: p.alpha_signal
            for sym, p in zip(order, processed, strict=True)
            if p.alpha_signal is not None
        },
        {
            sym: p.alpha_price
            for sym, p in zip(order, processed, strict=True)
            if p.alpha_price is not None
        },
    )


class RunFastLoopCycle:
    """ARCH-07: przebieg symboli cyklu z `CycleOptions`.

    `sleep` to throttle między symbolami (ochrona TPM dostawcy LLM), `redact`
    czyści komunikat błędu symbolu z sekretów, zanim trafi do maila (BEZ-07),
    a `symbol_scope` ustawia symbol w kontekście logu workera (NIEZ-21).
    Z `phase_two_feeds` (EXP-27, `two_phase_cycle_enabled`) cykl jest dwufazowy.
    """

    def __init__(
        self,
        *,
        use_case: AnalyzeMarketUseCase,
        alpha_use_case: AlphaEnrichmentUseCase,
        options: CycleOptions,
        sleep: Callable[[float], None],
        redact: Callable[[str], str],
        symbol_scope: SymbolScope,
        phase_two_feeds: PhaseTwoFeeds | None = None,
    ) -> None:
        self._use_case = use_case
        self._alpha = alpha_use_case
        self._options = options
        self._sleep = sleep
        self._redact = redact
        self._symbol_scope = symbol_scope
        self._phase_two_feeds = phase_two_feeds

    def run(self, eligible: Sequence[str]) -> CycleOutcome:
        """Dispatcher: wybiera ścieżkę równoległą lub sekwencyjną; z `phase_two_feeds`
        cykl dwufazowy."""
        options = self._options
        if self._phase_two_feeds is not None:
            return self._run_two_phase(eligible, self._phase_two_feeds)
        # Ścieżka równoległa: tylko gdy concurrency > 1 ORAZ contagion off (contagion
        # #5 zależy od kolejności przetwarzania, więc jest niekompatybilny z out-of-order).
        if options.parallel:
            return self._run_parallel(eligible)

        if options.symbol_concurrency > 1 and options.contagion_enabled:
            logger.warning(
                "symbol_concurrency=%d zignorowane — contagion_enabled wymaga "
                "przetwarzania sekwencyjnego (peer_context zależy od kolejności).",
                options.symbol_concurrency,
            )

        return self._run_sequential(eligible)

    def process_one(
        self, symbol: str, peer_ctx: tuple[tuple[str, str], ...] = ()
    ) -> ProcessedSymbol:
        """Przetwarza jeden symbol. NIGDY nie rzuca — łapie wyjątek i zwraca wynik
        z `is_failure=True`. Dzięki temu pula wątków (ścieżka równoległa) nigdy
        nie widzi wyjątku, a semantyka resilience jest identyczna z dawną pętlą
        sekwencyjną.

        `alpha_fusion_enabled` (#14) zmienia prompt LLM i zapisuje kandydata na
        cechę ML (`alpha_fusion_score`, poza `ML_FEATURE_COLUMNS` — model go nie
        widzi), więc jej wdrożenie ma być świadomą, odwracalną decyzją; o promocji
        do cech modelu rozstrzyga porównanie A/B (EXP-19), nie intuicja
        dobierającego wagi. Off → `alpha_fusion=None` → prompt i zapis jak przed #14.

        NIEZ-21: symbol trafia do każdej linii logu z tego wywołania — także z wątku
        puli i z wątków rady (`copy_context`).
        """
        options = self._options
        with self._symbol_scope(symbol):
            try:
                inputs = self._symbol_inputs(symbol)
                raw = self._use_case.run(
                    symbol,
                    asset=inputs.asset,
                    regime_context=options.regime_context,
                    regime_multiplier=options.regime_multiplier,
                    earnings_multiplier=inputs.earnings_multiplier,
                    alpha_fusion=inputs.alpha_fusion,
                    peer_context=peer_ctx,
                    # FIN-14: wiek referencji ceny liczy się od startu cyklu.
                    now=options.cycle_at,
                )
                return self._finished(symbol, raw, inputs.alpha_signal)
            except PortError as exc:
                # ARCH-33: awaria dostawcy przetłumaczona przez adapter to stan znany —
                # jedna linijka WARNING; traceback zostaje dla błędów kodu.
                return self._failed(symbol, exc, provider_failure=True)
            except Exception as exc:
                return self._failed(symbol, exc)

    def _symbol_inputs(self, symbol: str) -> _SymbolInputs:
        options = self._options
        asset = Asset(
            symbol=symbol,
            asset_type=classify_asset(symbol, options.crypto_symbols, options.etf_symbols),
        )
        # #6: bramka earnings. Event pobieramy PRZED grafem, bo mnożnik progu
        # musi być znany zanim graf zdecyduje o odpaleniu płatnych portów.
        # Gdy kalendarz earnings jest wyłączony, nie ruszamy portu w ogóle.
        earnings_event: EarningsEvent | None = None
        earnings_multiplier = 1.0
        if options.earnings_gate_enabled:
            earnings_event = self._alpha.fetch_earnings(symbol)
            if earnings_event is not None:
                earnings_multiplier = earnings_threshold_multiplier(earnings_event.proximity())
        # #14: enrichment biegnie PRZED grafem. Dawniej działał po nim, więc
        # LLM i cechy XGBoost nigdy nie widziały sygnałów alfa — 5 pobieranych
        # źródeł było render-only. Liczba płatnych wywołań się NIE zmienia:
        # `enrich` i tak leciał raz na symbol, tylko później.
        alpha_signal: AlphaSignals | None = None
        alpha_fusion: AlphaFusionScore | None = None
        if options.collect_alpha:
            # De-dup: gdy bramka earnings pobrała już event, wstrzykujemy go,
            # zamiast płacić za drugie wywołanie Alpha Vantage na ten sam symbol.
            # Z wyłączoną bramką wołamy `enrich(symbol)` dokładnie jak dawniej.
            if options.earnings_gate_enabled:
                alpha_signal = self._alpha.enrich(
                    symbol, earnings=earnings_event, earnings_prefetched=True
                )
            else:
                alpha_signal = self._alpha.enrich(symbol)
            if options.alpha_fusion_enabled and alpha_signal is not None:
                alpha_fusion = fuse_alpha_signals(
                    insider=alpha_signal.insider,
                    options=alpha_signal.options,
                    social=alpha_signal.social,
                    analyst=alpha_signal.analyst,
                    earnings=alpha_signal.earnings,
                )
        return _SymbolInputs(asset, earnings_multiplier, alpha_signal, alpha_fusion)

    def _finished(
        self, symbol: str, raw: dict[str, Any], alpha_signal: AlphaSignals | None
    ) -> ProcessedSymbol:
        options = self._options
        logger.info(
            "%s: status=%s delta=%s prediction_id=%s",
            symbol,
            raw.get("status"),
            raw.get("delta"),
            raw.get("prediction_id"),
        )
        sr = to_symbol_result(symbol, raw, now=options.cycle_at)
        alpha_price: Decimal | None = None
        if options.collect_alpha and sr.current_price is not None:
            alpha_price = sr.current_price
        return ProcessedSymbol(sr, False, alpha_signal, alpha_price)

    def _failed(
        self, symbol: str, exc: Exception, *, provider_failure: bool = False
    ) -> ProcessedSymbol:
        # BEZ-07: komunikat idzie do maila — ani URL z kluczem, ani dosłowna
        # wartość sekretu z Settings nie może w nim zostać (`redact`).
        error = self._redact(str(exc))
        if provider_failure:
            logger.warning("Failed to process symbol %s — %s: %s", symbol, type(exc).__name__, error)
        else:
            logger.exception("Failed to process symbol %s", symbol)
        result = to_symbol_result(symbol, None, error=error)
        # NIEZ-22: typ wyjątku osobno — step summary liczy błędy per klasa, bez komunikatu.
        return ProcessedSymbol(replace(result, error_type=type(exc).__name__), True, None, None)

    def _run_parallel(self, eligible: Sequence[str]) -> CycleOutcome:
        """Ścieżka równoległa: concurrency > 1 ORAZ contagion off (contagion #5 zależy
        od kolejności przetwarzania, więc jest niekompatybilny z out-of-order).
        executor.map zachowuje kolejność wejścia → wyniki deterministyczne."""
        with ThreadPoolExecutor(max_workers=self._options.symbol_concurrency) as executor:
            processed = list(executor.map(lambda sym: self.process_one(sym, ()), eligible))
        return _outcome(eligible, processed)

    def _run_sequential(self, eligible: Sequence[str]) -> CycleOutcome:
        """Ścieżka sekwencyjna: zachowuje kolejność, buduje peer_context dla contagion
        (#5) i throttluje między symbolami (ochrona OpenAI TPM)."""
        options = self._options
        results: list[SymbolResult] = []
        failures = 0
        # #5: akumulator werdyktów tego cyklu (symbol → stance) dla contagion.
        processed_stances: dict[str, str] = {}
        alpha_signals: dict[str, AlphaSignals] = {}
        alpha_prices: dict[str, Decimal] = {}
        for index, symbol in enumerate(eligible):
            peer_ctx = (
                build_peer_context(symbol, options.peer_groups, processed_stances)
                if options.contagion_enabled
                else ()
            )
            proc = self.process_one(symbol, peer_ctx)
            results.append(proc.result)
            if proc.is_failure:
                failures += 1
            elif proc.result.trend:  # zasil akumulator contagion dla kolejnych symboli
                processed_stances[symbol] = proc.result.trend
            if proc.alpha_signal is not None:
                alpha_signals[symbol] = proc.alpha_signal
            if proc.alpha_price is not None:
                alpha_prices[symbol] = proc.alpha_price
            # Throttle MIĘDZY symbolami — chroni OpenAI TPM. Po ostatnim brak sleepa.
            if options.throttle_seconds > 0 and index < len(eligible) - 1:
                self._sleep(options.throttle_seconds)
        return CycleOutcome(results, failures, alpha_signals, alpha_prices)

    def _gate_one(self, symbol: str) -> _GatedSymbol:
        """EXP-27 faza 1 jednego symbolu: te same wejścia co `process_one`
        (earnings, alfa) i graf bramki. NIGDY nie rzuca — błąd to wynik `is_failure`."""
        options = self._options
        try:
            inputs = self._symbol_inputs(symbol)
            outcome = self._use_case.run_gate(
                symbol,
                asset=inputs.asset,
                regime_context=options.regime_context,
                regime_multiplier=options.regime_multiplier,
                earnings_multiplier=inputs.earnings_multiplier,
                alpha_fusion=inputs.alpha_fusion,
                now=options.cycle_at,
            )
            if outcome.passed:
                return _GatedSymbol(symbol, None, outcome.state, inputs.alpha_signal)
            done = self._finished(symbol, dict(outcome.state), inputs.alpha_signal)
            return _GatedSymbol(symbol, done, None, inputs.alpha_signal)
        except Exception as exc:
            return _GatedSymbol(symbol, self._failed(symbol, exc), None, None)

    def _analyze_gated(
        self,
        gated: _GatedSymbol,
        paid_phase: PaidPhase,
        peer_ctx: tuple[tuple[str, str], ...],
    ) -> ProcessedSymbol:
        """EXP-27 faza 2 jednego symbolu za bramką. NIGDY nie rzuca."""
        assert gated.state is not None, "faza 2 dostaje tylko symbole za bramką"
        try:
            raw = paid_phase.run(gated.state, peer_context=peer_ctx)
            return self._finished(gated.symbol, raw, gated.alpha_signal)
        except Exception as exc:
            return self._failed(gated.symbol, exc)

    def _run_paid_phase(
        self,
        order: list[str],
        gated: Mapping[str, _GatedSymbol],
        paid_phase: PaidPhase,
    ) -> dict[str, ProcessedSymbol]:
        """EXP-27 faza 2 w kolejności `order` (malejące |Δ|). Równolegle jak cykl
        jednofazowy (pula startuje symbole w tej kolejności); sekwencyjnie z contagion
        liczonym z werdyktów wcześniejszych symboli fazy 2 i throttle między nimi."""
        options = self._options
        if options.parallel:
            with ThreadPoolExecutor(max_workers=options.symbol_concurrency) as executor:
                processed = list(
                    executor.map(
                        lambda sym: self._analyze_gated(gated[sym], paid_phase, ()), order
                    )
                )
            return dict(zip(order, processed, strict=True))

        results: dict[str, ProcessedSymbol] = {}
        processed_stances: dict[str, str] = {}
        for index, symbol in enumerate(order):
            peer_ctx = (
                build_peer_context(symbol, options.peer_groups, processed_stances)
                if options.contagion_enabled
                else ()
            )
            proc = self._analyze_gated(gated[symbol], paid_phase, peer_ctx)
            results[symbol] = proc
            if not proc.is_failure and proc.result.trend:
                processed_stances[symbol] = proc.result.trend
            # Throttle MIĘDZY płatnymi symbolami — chroni OpenAI TPM.
            if options.throttle_seconds > 0 and index < len(order) - 1:
                self._sleep(options.throttle_seconds)
        return results

    def _run_two_phase(
        self, eligible: Sequence[str], phase_two_feeds: PhaseTwoFeeds
    ) -> CycleOutcome:
        """EXP-27: cykl dwufazowy. Faza 1 (wszystkie symbole): cena, reflect, bramka
        domenowa; symbol pod progiem kończy jak dziś. Faza 2: feed AV zbudowany dla
        symboli za bramką i płatna analiza w kolejności `rank_by_magnitude`, więc
        sufit FIN-12 odcina najmniejsze |Δ|. Wyniki w kolejności `eligible`."""
        # Faza 1 nie zależy od kolejności (contagion żyje w fazie 2), więc idzie
        # równolegle przy każdym concurrency > 1.
        if self._options.symbol_concurrency > 1:
            with ThreadPoolExecutor(max_workers=self._options.symbol_concurrency) as executor:
                gated_list = list(executor.map(self._gate_one, eligible))
        else:
            gated_list = [self._gate_one(symbol) for symbol in eligible]

        processed = {g.symbol: g.done for g in gated_list if g.done is not None}
        passed = {g.symbol: g for g in gated_list if g.state is not None}
        order = rank_by_magnitude(
            {symbol: g.state["delta"] for symbol, g in passed.items() if g.state is not None}
        )
        logger.info(
            "EXP-27 two-phase cycle: %d/%d symbols past the gate, paid phase by |Δ|: %s",
            len(order),
            len(eligible),
            order,
        )
        if order:
            sentiment_port, news_port = phase_two_feeds(order)
            processed.update(
                self._run_paid_phase(
                    order,
                    passed,
                    self._use_case.paid_phase(sentiment_port=sentiment_port, news_port=news_port),
                )
            )

        return _outcome(eligible, [processed[symbol] for symbol in eligible])
