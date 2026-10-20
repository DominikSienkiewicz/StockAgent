"""Domena wykrywania szoku rynkowego (alert poza cyklem digestu).

Czysta logika bez I/O: decyduje, czy nagły ruch ceny zasługuje na natychmiastowy
alert. Próg szoku jest CELOWO wyższy niż bramka volatility Fast Loopa — chodzi
o rzadkie, gwałtowne ruchy, nie o codzienny szum. Krypto handluje 24/7 i rusza
się natywnie 3–5% dziennie, więc dostaje osobny, wyższy próg.

Persystencja (tabela `shock_alerts`, `UNIQUE(symbol, alert_date)`) NIE należy do
tej warstwy — domena dostarcza jedynie czysty predykat debounce `should_emit`.

Dwa źródła ruchu (ALR-02): delta wobec snapshotu Fast Loopa (`detect_shock`, z guardem
wieku) albo dzienna zmiana z darmowego /quote (`DailyChange`, z guardem dnia
`is_daily_change_fresh`). Oba kończą w tej samej klasyfikacji `classify_shock`.
"""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum

from src.domain.asset import Asset, PriceDelta
from src.domain.value_objects import AssetType, Threshold

# Mnożnik progu szoku względem bramki volatility Fast Loopa. Szok dla equity to
# 2× typowej bramki (np. 2% → 4%): tylko wyraźnie ponadnormatywny ruch alarmuje.
SHOCK_THRESHOLD_MULTIPLIER = Decimal("2")

# Osobny, stały próg dla krypto (BTC/ETH). Rynek 24/7 rusza się natywnie 3–5%
# dziennie, więc podwojona bramka equity dałaby zbyt wiele fałszywych alertów —
# krypto potrzebuje wyższej poprzeczki (ułamek, 0.06 = 6%).
CRYPTO_SHOCK_THRESHOLD = Decimal("0.06")

# Guard świeżości snapshotu: gdy poprzedni punkt odniesienia jest starszy niż
# tyle godzin, delta liczona jest wobec stęchłej ceny i szok NIE jest zgłaszany
# (fałszywy alarm). Wartość mieści się w bezpiecznym oknie 24–36h.
SNAPSHOT_MAX_AGE_HOURS = 30.0

# Okno zmiany 24 h CoinGecko (ALR-02): ruch, za który alert wyszedł mniej niż tyle
# temu, wciąż siedzi w kolejnym odczycie — także po północy UTC, pod nowym kluczem.
ROLLING_CHANGE_WINDOW = timedelta(hours=24)


class ShockDirection(Enum):
    """Kierunek szoku — spadek albo wzrost ceny."""

    UP = "UP"
    DOWN = "DOWN"


@dataclass(frozen=True)
class ShockAlert:
    """Wykryty szok rynkowy gotowy do wypchnięcia jednym alertem."""

    symbol: str
    delta: PriceDelta
    direction: ShockDirection


@dataclass(frozen=True)
class DailyChange:
    """Dzienna zmiana ceny z darmowego notowania (ALR-02).

    `fraction` to ułamek (-0.09 = -9%): dla akcji wobec poprzedniego zamknięcia
    (Finnhub `dp`), dla krypto w oknie 24 h (CoinGecko). `as_of` to chwila
    notowania — w sobotę Finnhub oddaje piątkowe `dp` z piątkowym `t`.
    """

    fraction: Decimal
    as_of: datetime


def _shock_threshold(asset: Asset, volatility_threshold: Threshold) -> Decimal:
    """Zwraca próg szoku (ułamek) właściwy dla danego typu aktywa."""
    if asset.asset_type is AssetType.CRYPTO:
        return CRYPTO_SHOCK_THRESHOLD
    return volatility_threshold.value * SHOCK_THRESHOLD_MULTIPLIER


def is_reference_stale(snapshot_age_hours: float) -> bool:
    """Czy punkt odniesienia jest za stary, żeby liczyć wobec niego szok.

    Dokładnie `SNAPSHOT_MAX_AGE_HOURS` jest jeszcze świeże. Watch woła ten
    predykat PRZED pobraniem ceny bieżącej i liczy stęchłe referencje — gdy
    stęchłe są wszystkie, nie może zgłosić żadnego alertu.
    """
    return snapshot_age_hours > SNAPSHOT_MAX_AGE_HOURS


def _as_utc(moment: datetime) -> datetime:
    # Naiwny datetime traktujemy jako UTC, jak całe repo.
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def is_daily_change_fresh(as_of: datetime, now: datetime) -> bool:
    """Czy dzienna zmiana należy do bieżącego dnia UTC — dnia klucza debounce.

    Odczyt z innego dnia UTC (sobota z piątkowym `dp`, piątkowy wieczór ET po
    północy UTC) opisuje sesję, za którą alert mógł już wyjść pod poprzednim
    `alert_date`. Ten sam dzień co klucz `UNIQUE(symbol, alert_date)` gwarantuje,
    że jedna sesja daje najwyżej jeden alert.
    """
    return _as_utc(as_of).date() == _as_utc(now).date()


def is_move_already_alerted(last_alert_at: datetime | None, now: datetime) -> bool:
    """Czy zmiana z okna 24 h wciąż zawiera ruch, za który alert już wyszedł.

    Alert sprzed mniej niż `ROLLING_CHANGE_WINDOW` liczył ruch z tego samego okna,
    więc kolejny alert z niego byłby duplikatem. Dokładnie 24 h to już nowe okno.
    """
    if last_alert_at is None:
        return False
    return _as_utc(now) - _as_utc(last_alert_at) < ROLLING_CHANGE_WINDOW


def classify_shock(
    asset: Asset, delta: PriceDelta, volatility_threshold: Threshold
) -> ShockAlert | None:
    """Klasyfikuje ruch jako szok albo zwraca None — bez oceny świeżości źródła.

    Świeżość ocenia wołający: `detect_shock` dla snapshotu, `is_daily_change_fresh`
    dla dziennej zmiany z notowania.
    """
    if abs(delta.fraction) < _shock_threshold(asset, volatility_threshold):
        return None

    direction = ShockDirection.DOWN if delta.fraction < 0 else ShockDirection.UP
    return ShockAlert(symbol=asset.symbol, delta=delta, direction=direction)


def detect_shock(
    asset: Asset,
    delta: PriceDelta,
    volatility_threshold: Threshold,
    snapshot_age_hours: float,
) -> ShockAlert | None:
    """Wykrywa szok cenowy albo zwraca None, gdy ruch nie kwalifikuje się.

    `volatility_threshold` to bramka Fast Loopa — próg szoku jest od niej wyższy
    (2× dla equity, osobna stała dla krypto). `snapshot_age_hours` to wiek
    poprzedniego snapshotu ceny: gdy przekracza `SNAPSHOT_MAX_AGE_HOURS`, punkt
    odniesienia jest stęchły i alert jest tłumiony niezależnie od wielkości
    delty.
    """
    if is_reference_stale(snapshot_age_hours):
        return None
    return classify_shock(asset, delta, volatility_threshold)


def should_emit(
    symbol: str,
    alert_date: date,
    already_sent: AbstractSet[tuple[str, date]],
) -> bool:
    """Predykat debounce: jeden alert per symbol per dzień.

    Twardy debounce OD PIERWSZEGO COMMITA — spam przy chaotycznym rynku uczy
    ignorować kanał. `already_sent` to zbiór par (symbol, dzień) już zgłoszonych
    alertów (odpowiednik `UNIQUE(symbol, alert_date)` w warstwie persystencji).
    """
    return (symbol, alert_date) not in already_sent
