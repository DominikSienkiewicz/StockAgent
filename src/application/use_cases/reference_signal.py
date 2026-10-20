"""WIA-24 — darmowy sygnał referencyjny `reference-v1` (krok Slow Loopa).

Raz na przebieg: własny odczyt darmowej ceny (Finnhub /quote dla akcji, CoinGecko
dla krypto), rozliczenie zeszłotygodniowego wywołania TYM SAMYM odczytem i nowe
wywołanie albo abstynencja według zamrożonej reguły z `src/domain/reference_signal.py`
(protokół: docs/reference-protocol-v1.md).

Zero płatnych portów: use case przyjmuje wyłącznie darmową cenę i repozytorium.
Zapis idzie do osobnej tabeli `reference_predictions` (migracja 042) — nigdy do
`prediction_logs` (ścieżki LLM) ani `price_snapshots` (referencje Fast Loopa).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from src.application.ports import MarketDataPort, OpenReferenceCall, RepositoryPort
from src.domain.reference_signal import (
    BASELINE_MAX_AGE,
    ReferenceOutcome,
    ReferenceRead,
    may_call,
    momentum_call,
    select_baseline,
    settle_call,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceSignalSummary:
    """Liczby przebiegu, bez symboli: zapisane odczyty, wywołania, abstynencje,
    rozliczenia (trafienia, pudła, wygasłe) i symbole, które padły (spośród `total`).

    `held` — odczyty bez wywołania, bo symbol ma już wywołanie sprzed < 6 dni."""

    reads: int
    calls: int
    abstained: int
    hits: int
    misses: int
    expired: int
    failures: int
    total: int
    held: int = 0


class RunReferenceSignalUseCase:
    """Jeden przebieg reguły referencyjnej na podanych symbolach."""

    def __init__(self, *, market: MarketDataPort, repository: RepositoryPort) -> None:
        self._market = market
        self._repository = repository

    def run(self, symbols: Sequence[str], now: datetime | None = None) -> ReferenceSignalSummary:
        """Błąd odczytu historii przerywa przebieg PRZED zapisem — bez niej każde
        wywołanie byłoby abstynencją, a tydzień miałby dziurę zamiast błędu.
        Błąd pojedynczego symbolu nie przerywa pozostałych."""
        moment = now or datetime.now(UTC)
        reads = self._repository.get_reference_reads(moment - BASELINE_MAX_AGE)
        open_calls = self._repository.get_open_reference_calls()

        tally: Counter[str] = Counter()
        for symbol in symbols:
            try:
                self._run_symbol(symbol, moment, reads, open_calls, tally)
            except Exception:
                logger.exception("Reference signal failed for symbol %s", symbol)
                tally["failures"] += 1

        return ReferenceSignalSummary(
            reads=tally["reads"],
            calls=tally["calls"],
            abstained=tally["abstained"],
            hits=tally["hits"],
            misses=tally["misses"],
            expired=tally["expired"],
            failures=tally["failures"],
            total=len(symbols),
            held=tally["held"],
        )

    def _run_symbol(
        self,
        symbol: str,
        moment: datetime,
        reads: Sequence[ReferenceRead],
        open_calls: Sequence[OpenReferenceCall],
        tally: Counter[str],
    ) -> None:
        price = self._market.get_current_price(symbol)
        current = ReferenceRead(symbol=symbol, price=price.amount, read_at=moment)

        # Rozliczenie zeszłotygodniowego wywołania tym samym odczytem (horyzont
        # „do następnego przebiegu”).
        for open_call in open_calls:
            if open_call.call.symbol != symbol:
                continue
            outcome = settle_call(open_call.call, current)
            if outcome is ReferenceOutcome.PENDING:
                continue
            judged = outcome in (ReferenceOutcome.HIT, ReferenceOutcome.MISS)
            self._repository.settle_reference_call(
                open_call.id, outcome, current.price if judged else None, moment
            )
            tally[_OUTCOME_KEYS[outcome]] += 1

        # Wywołanie sprzed < 6 dni nie może być jeszcze rozliczone (rozliczenie zaczyna
        # się od 6 dni), więc lista otwartych wywołań zawiera każde, które blokuje.
        if not may_call((open_call.call for open_call in open_calls), current):
            self._repository.save_reference_read(current, None)
            tally["reads"] += 1
            tally["held"] += 1
            return

        call = momentum_call(select_baseline(reads, current), current)
        self._repository.save_reference_read(current, call)
        tally["reads"] += 1
        tally["calls" if call is not None else "abstained"] += 1


_OUTCOME_KEYS = {
    ReferenceOutcome.HIT: "hits",
    ReferenceOutcome.MISS: "misses",
    ReferenceOutcome.EXPIRED: "expired",
}
