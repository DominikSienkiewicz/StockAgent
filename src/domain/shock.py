"""Domena wykrywania szoku rynkowego (alert poza cyklem digestu).

Czysta logika bez I/O: decyduje, czy nagły ruch ceny zasługuje na natychmiastowy
alert. Próg szoku jest CELOWO wyższy niż bramka volatility Fast Loopa — chodzi
o rzadkie, gwałtowne ruchy, nie o codzienny szum. Krypto handluje 24/7 i rusza
się natywnie 3–5% dziennie, więc dostaje osobny, wyższy próg.

Persystencja (tabela `shock_alerts`, `UNIQUE(symbol, alert_date)`) NIE należy do
tej warstwy — domena dostarcza jedynie czysty predykat debounce `should_emit`.
"""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import date
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


def _shock_threshold(asset: Asset, volatility_threshold: Threshold) -> Decimal:
    """Zwraca próg szoku (ułamek) właściwy dla danego typu aktywa."""
    if asset.asset_type is AssetType.CRYPTO:
        return CRYPTO_SHOCK_THRESHOLD
    return volatility_threshold.value * SHOCK_THRESHOLD_MULTIPLIER


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
    if snapshot_age_hours > SNAPSHOT_MAX_AGE_HOURS:
        return None

    if abs(delta.fraction) < _shock_threshold(asset, volatility_threshold):
        return None

    direction = ShockDirection.DOWN if delta.fraction < 0 else ShockDirection.UP
    return ShockAlert(symbol=asset.symbol, delta=delta, direction=direction)


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
