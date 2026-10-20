"""Sygnał referencyjny WIA-24: darmowa, pre-rejestrowana reguła momentum tygodniowego.

Reguła (odczyt, okno bazowe, wywołanie, horyzont, rozliczenie) jest opisana
w zamrożonym pliku `docs/reference-protocol-v1.md`; jego sha256 to
`REFERENCE_PROTOCOL_SHA256`. Test pada, gdy plik i stała się rozjadą — zmiana
reguły to nowa wersja (`reference-v2`), nie edycja v1, żeby nikt nie dobierał
reguły po fakcie.

Czysta logika, zero I/O: odczyt ceny z darmowego /quote i trwały zapis odczytów
należą do warstw wyżej.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from src.domain.prediction import Prediction, TrendDirection

# Etykieta `recipe_version` każdego wywołania — odróżnia je od predykcji LLM.
REFERENCE_RECIPE_VERSION = "reference-v1"
REFERENCE_PROTOCOL_PATH = "docs/reference-protocol-v1.md"
REFERENCE_PROTOCOL_SHA256 = "dab92dae5415d5aa2ab561c3a4aa934621812f5edc69db374d587d2f9eb24eea"

# Okno „zeszły tydzień” dla odczytu bazowego i dla odczytu rozliczającego: tydzień
# ± dzień zapasu na opóźnienie crona GitHuba (deadman Slow Loopa liczy 192 h).
BASELINE_MIN_AGE = timedelta(days=6)
BASELINE_MAX_AGE = timedelta(days=8)


@dataclass(frozen=True)
class ReferenceRead:
    """Własny odczyt ceny sygnału (nigdy `price_snapshots`). `read_at` z strefą (UTC)."""

    symbol: str
    price: Decimal
    read_at: datetime


@dataclass(frozen=True)
class ReferenceCall:
    """Wywołanie reguły: kierunek do następnego przebiegu od ceny z chwili wywołania."""

    symbol: str
    trend: TrendDirection
    price_at_call: Decimal
    called_at: datetime


class ReferenceOutcome(Enum):
    """Wynik rozliczenia: trafienie, pudło, jeszcze otwarte albo wygasłe (poza próbą)."""

    HIT = "HIT"
    MISS = "MISS"
    PENDING = "PENDING"
    EXPIRED = "EXPIRED"


def protocol_digest(text: str) -> str:
    """sha256 tekstu protokołu. Końce linii ujednolicone, bo checkout na Windows
    (CRLF) nie zmienia reguły, a zmieniłby hash."""
    normalized = text.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_week_apart(earlier: datetime, later: datetime) -> bool:
    return BASELINE_MIN_AGE <= later - earlier <= BASELINE_MAX_AGE


def select_baseline(
    reads: Iterable[ReferenceRead], current: ReferenceRead
) -> ReferenceRead | None:
    """Najnowszy wcześniejszy odczyt tego symbolu sprzed 6–8 dni albo None.

    Ręczny przebieg w środku tygodnia nie staje się bazą (za świeży), a pominięty
    tydzień daje abstynencję zamiast momentum z dwóch tygodni."""
    candidates = [
        read
        for read in reads
        if read.symbol == current.symbol and _is_week_apart(read.read_at, current.read_at)
    ]
    return max(candidates, key=lambda read: read.read_at, default=None)


def may_call(recent_calls: Iterable[ReferenceCall], current: ReferenceRead) -> bool:
    """Najwyżej jedno wywołanie na symbol w ciągu 6 dni (poprawka v1 przed startem).

    Ponowny przebieg w tym oknie (np. re-run joba) dałby drugie, prawie identyczne
    wywołanie z tej samej bazy — skorelowany duplikat pompujący próbę. Taki przebieg
    tylko rozlicza dojrzałe wywołania i zapisuje odczyt. Wywołanie sprzed dokładnie
    6 dni już nie blokuje."""
    return not any(
        call.symbol == current.symbol and current.read_at - call.called_at < BASELINE_MIN_AGE
        for call in recent_calls
    )


def momentum_call(
    baseline: ReferenceRead | None, current: ReferenceRead
) -> ReferenceCall | None:
    """Znak zmiany od odczytu bazowego: BULLISH w górę, BEARISH w dół.

    Brak bazy (pierwszy przebieg), cena bez zmian albo niedodatnia → abstynencja.
    Reguła nigdy nie wywołuje SIDEWAYS."""
    if baseline is None or baseline.price <= 0 or current.price <= 0:
        return None
    if current.price > baseline.price:
        trend = TrendDirection.BULLISH
    elif current.price < baseline.price:
        trend = TrendDirection.BEARISH
    else:
        return None
    return ReferenceCall(
        symbol=current.symbol,
        trend=trend,
        price_at_call=current.price,
        called_at=current.read_at,
    )


def settle_call(call: ReferenceCall, read: ReferenceRead) -> ReferenceOutcome:
    """Rozlicza wywołanie odczytem z następnego przebiegu.

    Odczyt sprzed 6 dni zostawia wywołanie otwarte, po 8 dniach wywołanie
    wygasa. Trafienie liczy ta sama reguła kierunku co track record LLM
    (`Prediction.is_trend_correct`): cena bez zmian to pudło."""
    if read.symbol != call.symbol:
        raise ValueError(f"Read of {read.symbol} cannot settle a call on {call.symbol}")
    elapsed = read.read_at - call.called_at
    if elapsed < BASELINE_MIN_AGE:
        return ReferenceOutcome.PENDING
    if elapsed > BASELINE_MAX_AGE:
        return ReferenceOutcome.EXPIRED
    prediction = Prediction(
        symbol=call.symbol,
        predicted_trend=call.trend,
        price_at_prediction=call.price_at_call,
        predicted_target_price=call.price_at_call,
    )
    if prediction.is_trend_correct(read.price):
        return ReferenceOutcome.HIT
    return ReferenceOutcome.MISS
