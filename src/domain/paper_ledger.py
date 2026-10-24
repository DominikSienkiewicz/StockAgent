"""Append-only paper ledger z łańcuchem hashy (WIA-59) — czysta domena.

PnL track recordu był dotąd przeliczany z mutowalnych kolumn `prediction_logs`:
`update_prediction_accuracy` nadpisuje rozliczenie, a ręczny re-run upsertem
nadpisuje predykcję z tej samej godziny. Ledger zapisuje pozycję papierową raz,
przy zapisie predykcji, i zamyka ją raz, przy rozliczeniu.

Łańcuch: `row_hash = sha256(canonicalize(wpis) + prev_hash)`, gdzie `prev_hash`
to hash poprzedniego wiersza (pierwszy wiersz: `GENESIS_HASH`). Podmiana,
usunięcie albo wstawienie wiersza w środku rozjeżdża łańcuch od tego miejsca
(`first_broken_row`), a opublikowana głowa łańcucha kotwiczy całą historię przed
nią. Hash obejmuje wejście pozycji (decyzję); zamknięcie (cena wyjścia z rynku,
sprawdzalna z zewnątrz) chroni trigger bazy, który pozwala je zapisać tylko raz.

Kanonizacja to kontrakt danych jak w commitmentach (`attestation.canonicalize`):
ceny w skali kolumny DECIMAL(14,4), koszt NUMERIC(8,2), waga NUMERIC(8,4),
znacznik UTC z mikrosekundami — dokładnie to, co baza przechowa i odda. Zmiana
reguły = nowa `LEDGER_VERSION`, stare wiersze weryfikuje stara.

Zero I/O, tylko stdlib i domena.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any

from src.domain.attestation import canonicalize
from src.domain.equity_curve import signal_of
from src.domain.value_objects import AssetType, Money

# Hash „przed pierwszym wierszem” — `prev_hash` pierwszego wpisu łańcucha.
GENESIS_HASH = "0" * 64
# Wersja kanonizacji wpisu (pole payloadu, obok `schema_version` z `canonicalize`).
LEDGER_VERSION = 1

_COST_SCALE = Decimal("0.01")
_WEIGHT_SCALE = Decimal("0.0001")
# PnL jako ułamek w NUMERIC(14,8) — 8 miejsc wystarcza dla zwrotu w bps.
_PNL_SCALE = Decimal("0.00000001")


class TradeSide(StrEnum):
    """Strona pozycji z kierunku predykcji; FLAT (SIDEWAYS) trzyma gotówkę."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


_TREND_OF_SIDE = {TradeSide.LONG: "BULLISH", TradeSide.SHORT: "BEARISH", TradeSide.FLAT: "SIDEWAYS"}


def side_of(predicted_trend: str) -> TradeSide:
    """BULLISH → LONG, BEARISH → SHORT, reszta → FLAT (ta sama reguła co `signal_of`)."""
    signal = signal_of(predicted_trend)
    if signal > 0:
        return TradeSide.LONG
    if signal < 0:
        return TradeSide.SHORT
    return TradeSide.FLAT


def trend_of(side: TradeSide) -> str:
    """Kierunek predykcji pozycji — wejście krzywej kapitału (`signal_of`)."""
    return _TREND_OF_SIDE[side]


class ExitReason(StrEnum):
    """Dlaczego pozycja się zamknęła (`paper_trades.exit_reason`).

    STALE_QUOTE (DAT-14): rozliczenie na notowaniu sprzed sesji po predykcji —
    pozycja zamknięta, ale poza krzywą, jak predykcja poza trafnością."""

    SETTLED = "settled"
    STALE_QUOTE = "stale_quote"


@dataclass(frozen=True)
class LedgerCosts:
    """Koszt jednostronny pozycji w bps (WIA-50): krypto ma własny, reszta — akcyjny."""

    cost_bps: Decimal
    crypto_cost_bps: Decimal

    def for_asset(self, asset_type: AssetType | None) -> Decimal:
        """Koszt klasy aktywa; symbol bez klasy płaci koszt akcyjny."""
        return self.crypto_cost_bps if asset_type is AssetType.CRYPTO else self.cost_bps


@dataclass(frozen=True)
class PaperTradeEntry:
    """Pozycja papierowa otwarta przy zapisie predykcji `prediction_id`.

    `weight` to jednostka pozycji (1 = cała stawka jednej predykcji); krzywa
    portfela dzieli kapitał cyklu po równo, więc waga nie zależy od liczby
    symboli cyklu. `cost_bps` — jednostronny koszt obrotu symbolu (WIA-50)."""

    prediction_id: str
    symbol: str
    side: TradeSide
    entry_price: Decimal
    entry_ts: datetime
    cost_bps: Decimal
    weight: Decimal = Decimal(1)

    def payload(self) -> dict[str, Any]:
        """Treść hashowana w łańcuchu — w skali kolumn bazy (patrz docstring modułu)."""
        return {
            "ledger_version": LEDGER_VERSION,
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": str(Money(self.entry_price).for_storage().amount),
            "entry_ts": utc_timestamp(self.entry_ts),
            "cost_bps": str(self.cost_bps.quantize(_COST_SCALE, rounding=ROUND_HALF_UP)),
            "weight": str(self.weight.quantize(_WEIGHT_SCALE, rounding=ROUND_HALF_UP)),
        }


def utc_timestamp(moment: datetime) -> str:
    """Znacznik UTC z mikrosekundami (precyzja TIMESTAMPTZ); naiwny `datetime` to UTC."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class LedgerRow:
    """Wpis łańcucha: pozycja, hash poprzedniego wiersza i własny hash."""

    entry: PaperTradeEntry
    prev_hash: str
    row_hash: str


def row_hash(entry: PaperTradeEntry, prev_hash: str) -> str:
    """`sha256(canonicalize(wpis) + prev_hash)` jako hex."""
    material = canonicalize(entry.payload()) + prev_hash
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def chain_entry(entry: PaperTradeEntry, head: str) -> LedgerRow:
    """Wiersz dopisywany za głową łańcucha `head` (pusty ledger: `GENESIS_HASH`)."""
    return LedgerRow(entry=entry, prev_hash=head, row_hash=row_hash(entry, head))


def first_broken_row(rows: Sequence[LedgerRow], start: str = GENESIS_HASH) -> int | None:
    """Indeks pierwszego wiersza, który nie wynika z poprzedniego; None = łańcuch cały.

    `rows` w kolejności łańcucha. `start` to hash wiersza tuż przed `rows` —
    weryfikacja okna zaczyna od ostatniego zweryfikowanego (albo opublikowanego)
    hasha; cały ledger od `GENESIS_HASH`."""
    expected_prev = start
    for index, row in enumerate(rows):
        if row.prev_hash != expected_prev or row.row_hash != row_hash(row.entry, row.prev_hash):
            return index
        expected_prev = row.row_hash
    return None


@dataclass(frozen=True)
class PaperTradeClose:
    """Zamknięcie pozycji przy rozliczeniu predykcji.

    `pnl` — zwrot brutto pozycji jako ułamek: sygnał strony × zmiana ceny od
    wejścia. Koszt obrotu liczy krzywa portfela (zależy od poprzedniej pozycji
    symbolu), nie pojedyncze zamknięcie."""

    exit_price: Decimal
    exit_ts: datetime
    pnl: Decimal
    exit_reason: ExitReason


def close_trade(
    side: TradeSide,
    entry_price: Decimal,
    exit_price: Decimal,
    at: datetime,
    reason: ExitReason,
) -> PaperTradeClose:
    """Zamknięcie pozycji po cenie `exit_price`. Wejście bez ceny (0, skażony
    rekord) nie ma punktu odniesienia — PnL 0 zamiast dzielenia przez zero."""
    exit_stored = Money(exit_price).for_storage().amount
    pnl = Decimal(0)
    if entry_price > 0:
        change = (exit_stored - entry_price) / entry_price
        pnl = (signal_of(trend_of(side)) * change).quantize(_PNL_SCALE, rounding=ROUND_HALF_UP)
    return PaperTradeClose(exit_price=exit_stored, exit_ts=at, pnl=pnl, exit_reason=reason)


@dataclass(frozen=True)
class LedgerTrade:
    """Wiersz ledgera odczytany z bazy: łańcuch i zamknięcie (None = pozycja otwarta)."""

    seq: int
    row: LedgerRow
    close: PaperTradeClose | None = None


@dataclass(frozen=True)
class LedgerHead:
    """Głowa łańcucha — `row_hash` wiersza o największym `seq` (do publikacji)."""

    seq: int
    row_hash: str
