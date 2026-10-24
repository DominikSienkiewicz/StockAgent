"""Domena: cechy zmienności i momentum z darmowych snapshotów cen (ML-31).

5 z 7 cech modelu to pochodne newsów i LLM, a jedyna cecha cenowa (`price_delta`)
liczy ruch od poprzedniej PREDYKCJI o zmiennym odstępie. Momentum, odwrócenie
i reżim zmienności to najlepiej udokumentowane darmowe sygnały krótkoterminowe.

Jedna funkcja dla treningu i servingu: `price_features(historia, as_of)` widzi
wyłącznie ceny z chwilą <= `as_of` (point-in-time, zero podglądu przyszłości).
Snapshoty są per cykl, więc szereg idzie do zamknięć dziennych (ostatnia cena
każdego dnia UTC). Wszystkie cechy są bezwymiarowe (zwroty i ich ilorazy), więc
nie zależą od skali ceny — warunek modelu wspólnego na wielu symbolach.

Maska braków: snapshoty istnieją tylko z dni, w które pętla chodziła. Okno, w którym
między kolejnymi zamknięciami jest przerwa dłuższa niż `MAX_GAP_DAYS` dni
kalendarzowych (dłuższa niż weekend ze świętem), nie daje cechy — None zamiast
„zwrotu 1-dniowego” liczonego przez tydzień pauzy. Stdlib only.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta

# Najdłuższa dopuszczalna przerwa między kolejnymi zamknięciami w oknie (dni).
MAX_GAP_DAYS = 5
# Okno krótkie (zmienność 5 dni, zwrot 5 dni) i długie (20 dni).
SHORT_WINDOW = 5
LONG_WINDOW = 20

PRICE_FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_1_over_vol_20",
    "log_realized_vol_20",
    "vol_ratio_5_20",
    "dist_from_20d_high",
    "gate_move_z",
)


@dataclass(frozen=True)
class PriceFeatures:
    """Cechy cenowe symbolu w chwili `as_of`; None = za mało historii albo dziura.

    Pola:
        ret_1, ret_5, ret_20: zwroty z 1, 5 i 20 zamknięć dziennych,
        ret_1_over_vol_20: ret_1 / σ_20 (σ = odchylenie standardowe 20 dziennych
            log-zwrotów) — ruch dnia w jednostkach zwykłej zmienności,
        log_realized_vol_20: ln σ_20 — reżim zmienności,
        vol_ratio_5_20: σ_5 / σ_20 — czy zmienność właśnie rośnie,
        dist_from_20d_high: cena / maksimum 20 zamknięć − 1 (≤ 0),
        gate_move_z: ruch od poprzedniego snapshotu (ten, który ocenia bramka
            volatility) / σ_20 — z-score ruchu bramki.
    """

    ret_1: float | None = None
    ret_5: float | None = None
    ret_20: float | None = None
    ret_1_over_vol_20: float | None = None
    log_realized_vol_20: float | None = None
    vol_ratio_5_20: float | None = None
    dist_from_20d_high: float | None = None
    gate_move_z: float | None = None

    def as_features(self) -> dict[str, float | None]:
        """Klucze `PRICE_FEATURE_NAMES` w kolejności kontraktu."""
        values = asdict(self)
        return {name: values[name] for name in PRICE_FEATURE_NAMES}


def _utc_day(moment: datetime) -> date:
    return moment.astimezone(UTC).date() if moment.tzinfo is not None else moment.date()


def _daily_closes(points: Sequence[tuple[datetime, float]]) -> list[tuple[date, float]]:
    """Ostatnia cena każdego dnia UTC, rosnąco po dniu (wejście posortowane)."""
    closes: dict[date, float] = {}
    for moment, price in points:
        closes[_utc_day(moment)] = price
    return sorted(closes.items())


def _window(closes: Sequence[tuple[date, float]], size: int) -> list[float] | None:
    """Ostatnie `size` zamknięć albo None (za krótko albo dziura > MAX_GAP_DAYS)."""
    if len(closes) < size:
        return None
    tail = closes[-size:]
    for (earlier, _), (later, _) in zip(tail, tail[1:], strict=False):
        if (later - earlier).days > MAX_GAP_DAYS:
            return None
    return [price for _, price in tail]


def _period_return(closes: Sequence[tuple[date, float]], periods: int) -> float | None:
    prices = _window(closes, periods + 1)
    return None if prices is None else prices[-1] / prices[0] - 1.0


def _volatility(closes: Sequence[tuple[date, float]], periods: int) -> float | None:
    """Odchylenie standardowe (próbkowe) `periods` dziennych log-zwrotów."""
    prices = _window(closes, periods + 1)
    if prices is None:
        return None
    log_returns = [math.log(b / a) for a, b in zip(prices, prices[1:], strict=False)]
    return statistics.stdev(log_returns)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def price_features(history: Iterable[tuple[datetime, float]], as_of: datetime) -> PriceFeatures:
    """Cechy cenowe z historii `(chwila, cena)` widzianej w chwili `as_of`.

    Historia może zawierać ceny późniejsze — są pomijane, tak jak niedodatnie
    i nieskończone. Bieżąca cena to ostatni punkt <= `as_of` (serving dokłada
    kurs cyklu jako punkt w `as_of`), a referencja bramki to punkt przed nim."""
    points = sorted(
        (moment, float(price))
        for moment, price in history
        if moment <= as_of and math.isfinite(price) and price > 0
    )
    if not points:
        return PriceFeatures()
    closes = _daily_closes(points)
    ret_1 = _period_return(closes, 1)
    vol_short = _volatility(closes, SHORT_WINDOW)
    vol_long = _volatility(closes, LONG_WINDOW)
    high_window = _window(closes, LONG_WINDOW)
    gate_move = points[-1][1] / points[-2][1] - 1.0 if len(points) >= 2 else None
    return PriceFeatures(
        ret_1=ret_1,
        ret_5=_period_return(closes, SHORT_WINDOW),
        ret_20=_period_return(closes, LONG_WINDOW),
        ret_1_over_vol_20=_ratio(ret_1, vol_long),
        log_realized_vol_20=math.log(vol_long) if vol_long else None,
        vol_ratio_5_20=_ratio(vol_short, vol_long),
        dist_from_20d_high=(
            high_window[-1] / max(high_window) - 1.0 if high_window is not None else None
        ),
        gate_move_z=_ratio(gate_move, vol_long),
    )



# ML-11: σ zwrotów symbolu do normalizacji targetu modelu wspólnego (zwrot / σ).
# Zwroty liczone między KOLEJNYMI snapshotami ceny (cykl do cyklu — ta sama skala co
# target „zwrot do następnego cyklu”), z ostatnich `SIGMA_WINDOW_CYCLES` cykli.
SIGMA_WINDOW_CYCLES = 20
# Krótsza historia daje niestabilną σ — wtedy brak σ (wiersz poza treningiem,
# serving bez predykcji modelu), zamiast dzielić przez przypadkową liczbę.
SIGMA_MIN_RETURNS = 10
# Podłoga σ: weekend akcji i stęchłe notowania dają zwroty 0, a σ bliska zeru
# rozdmuchałaby target. 0,25% na cykl leży poniżej najspokojniejszych ETF-ów
# (σ dzienna ≈ 0,8%), więc obcina tylko zdegenerowane okna.
SIGMA_FLOOR = 0.0025
# Punkt bieżącego cyklu nie jest „poprzednim cyklem”: węzeł terminalny zapisuje
# snapshot ceny tuż PRZED wektorem cech, więc w treningu ma chwilę sprzed `as_of`
# wektora, a w servingu (predict) jeszcze go nie ma. Cykle tego samego symbolu dzieli
# co najmniej godzina (upsert per godzina), a jeden cykl trwa minuty.
SAME_CYCLE_TOLERANCE = timedelta(minutes=30)


def return_sigma(history: Iterable[tuple[datetime, float]], as_of: datetime) -> float | None:
    """σ (próbkowe odchylenie standardowe) zwrotów z poprzednich cykli w chwili `as_of`.

    Jedna funkcja dla treningu (as_of = chwila wektora) i servingu (as_of = zegar
    cyklu): bierze wyłącznie punkty sprzed `as_of − SAME_CYCLE_TOLERANCE`, z dodatnią
    i skończoną ceną, i ostatnie `SIGMA_WINDOW_CYCLES` zwrotów między kolejnymi
    punktami. Mniej niż `SIGMA_MIN_RETURNS` zwrotów → None; wynik nie spada poniżej
    `SIGMA_FLOOR`."""
    cutoff = as_of - SAME_CYCLE_TOLERANCE
    prices = [
        float(price)
        for moment, price in sorted(
            (moment, price) for moment, price in history if moment < cutoff
        )
        if math.isfinite(price) and price > 0
    ]
    returns = [b / a - 1.0 for a, b in zip(prices, prices[1:], strict=False)][
        -SIGMA_WINDOW_CYCLES:
    ]
    if len(returns) < SIGMA_MIN_RETURNS:
        return None
    return max(statistics.stdev(returns), SIGMA_FLOOR)
