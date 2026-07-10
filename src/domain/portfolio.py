"""Domena portfela użytkownika — realne wagi, P/L i ekspozycja klastrowa.

Cała warstwa ryzyka liczyła fikcję: `stress_test` i `portfolio_pnl` używały
RÓWNYCH wag, więc user trzymający 40% w NVDA dostawał VaR policzony, jakby
miał po 2.3% w 43 tickerach. Ten agregat liczy prawdę:

  - `weights()` — realne wagi wg wartości rynkowej, renormalizowane po
    symbolach, dla których jest bieżąca cena (`usable_symbols`). Symbol bez
    ceny wypada z mianownika, inaczej wagi nie sumują się do 1.
  - `unrealized_pnl()` — P/L per pozycja + łącznie, na `Decimal`.
  - `cluster_exposure()` — ekspozycja klastra korelacji ("Twoje 3 największe
    pozycje to jeden klaster — 61% kapitału"). Bierze GOTOWĄ mapę
    symbol→klaster; korelacji NIE liczymy tutaj.

Osobno żyje reguła świeżości: ręcznie aktualizowana tabela pozycji potrafi
się zestarzeć, a stęchły portfel daje FAŁSZYWY P/L — gorszy niż brak sekcji.
Dlatego `is_stale`/`freshness` z zamrożonym progiem `DEFAULT_MAX_POSITION_AGE_DAYS`
to wymaganie; warstwa raportu renderuje z tego badge STALE (przez `provenance`).

Zero I/O, zero importów zewnętrznych — tylko stdlib. `now` jest wstrzykiwane,
domena nigdy nie woła zegara. Pieniądze na `Decimal` (jak `Money`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

# Zamrożony próg świeżości portfela. Ręcznie aktualizowana tabela pozycji
# starsza niż tyle dni jest STALE — badge to wymaganie, więc zmiana tej
# wartości ma świadomie łamać test `test_default_max_age_is_frozen_constant`.
DEFAULT_MAX_POSITION_AGE_DAYS = 7

_SECONDS_PER_DAY = 86_400.0


def _require_finite(name: str, value: Decimal) -> None:
    """Odrzuca NaN/Inf, które cicho zatruwają porównania i sumy w domenie."""
    if value.is_nan() or value.is_infinite():
        raise ValueError(f"{name} must be a finite number (got {value})")


@dataclass(frozen=True)
class Position:
    """Pojedyncza pozycja (lot) w portfelu.

    `quantity` — liczba jednostek (musi być dodatnia; short poza zakresem
    MVP retail). `purchase_price` — cena zakupu za jednostkę (nieujemna).
    `purchase_date` — data nabycia (do FIFO/podatków w przyszłości).
    """

    symbol: str
    quantity: Decimal
    purchase_price: Decimal
    purchase_date: date

    def __post_init__(self) -> None:
        _require_finite("quantity", self.quantity)
        _require_finite("purchase_price", self.purchase_price)
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive (got {self.quantity})")
        if self.purchase_price < 0:
            raise ValueError(
                f"purchase_price must be non-negative (got {self.purchase_price})"
            )

    def cost_basis(self) -> Decimal:
        """Kapitał zainwestowany w tę pozycję (ilość × cena zakupu)."""
        return self.quantity * self.purchase_price

    def market_value(self, price: Decimal) -> Decimal:
        """Bieżąca wartość rynkowa pozycji przy podanej cenie."""
        return self.quantity * price

    def unrealized_pnl(self, price: Decimal) -> Decimal:
        """Niezrealizowany P/L: (cena bieżąca − cena zakupu) × ilość."""
        return (price - self.purchase_price) * self.quantity


@dataclass(frozen=True)
class PositionPnL:
    """P/L pojedynczej pozycji w danym momencie.

    `return_pct` to zwrot względem kosztu (0.5 == +50%); `None`, gdy koszt
    bazowy jest zerowy (np. pozycja o cenie zakupu 0) — nie dzielimy przez 0.
    """

    symbol: str
    quantity: Decimal
    purchase_price: Decimal
    current_price: Decimal
    cost_basis: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    return_pct: Decimal | None


@dataclass(frozen=True)
class PortfolioPnL:
    """Zagregowany P/L portfela — pozycje policzalne + sumy łączne.

    `positions` zawiera tylko pozycje, dla których była bieżąca cena;
    reszta wypada (bez ceny nie ma P/L). `total_return_pct` jest `None`
    dla pustego/zerokosztowego portfela.
    """

    positions: tuple[PositionPnL, ...]
    total_cost_basis: Decimal
    total_market_value: Decimal
    total_unrealized_pnl: Decimal
    total_return_pct: Decimal | None


@dataclass(frozen=True)
class ClusterExposure:
    """Ekspozycja jednego klastra korelacji jako udział kapitału.

    `weight` to suma wag (po renormalizacji na `usable_symbols`) symboli
    należących do klastra. `symbols` — członkowie klastra obecni w portfelu.
    """

    cluster: str
    symbols: tuple[str, ...]
    weight: Decimal


class PortfolioFreshness(Enum):
    """Świeżość ręcznie aktualizowanej tabeli pozycji.

    FRESH — dane w progu wieku, P/L wiarygodny.
    STALE — pozycje starsze niż próg; P/L może być fałszywy → badge STALE.
    """

    FRESH = "FRESH"
    STALE = "STALE"


def _age_days(as_of: datetime, now: datetime) -> float:
    """Wiek danych w dniach. Naiwne i aware daty porównujemy spójnie —
    jeśli jedna strona jest naiwna, sprowadzamy obie do tego samego zegara."""
    if now.tzinfo is None and as_of.tzinfo is not None:
        as_of = as_of.replace(tzinfo=None)
    elif now.tzinfo is not None and as_of.tzinfo is None:
        now = now.replace(tzinfo=None)
    return (now - as_of).total_seconds() / _SECONDS_PER_DAY


def is_stale(
    as_of: datetime,
    now: datetime,
    max_age_days: int = DEFAULT_MAX_POSITION_AGE_DAYS,
) -> bool:
    """Czy portfel jest przeterminowany (wiek ≥ próg).

    Próg włączny: dokładnie `max_age_days` liczy się już jako STALE — wolimy
    fałszywy alarm świeżości niż ukryty stęchły P/L. `now` w przeszłości
    względem `as_of` (dane z przyszłości) traktujemy jako świeże."""
    return _age_days(as_of, now) >= max_age_days


def freshness(
    as_of: datetime,
    now: datetime,
    max_age_days: int = DEFAULT_MAX_POSITION_AGE_DAYS,
) -> PortfolioFreshness:
    """Enum świeżości — cukier na `is_stale` dla warstwy raportu."""
    if is_stale(as_of, now, max_age_days):
        return PortfolioFreshness.STALE
    return PortfolioFreshness.FRESH


@dataclass(frozen=True)
class Portfolio:
    """Agregat portfela: zbiór pozycji + znacznik ostatniej aktualizacji.

    `positions` to krotka (niemutowalny agregat). `as_of` to moment, w którym
    tabela pozycji była ostatnio odświeżona — wejście reguły świeżości.
    """

    positions: tuple[Position, ...]
    as_of: datetime

    def _usable_market_values(
        self, current_prices: Mapping[str, Decimal]
    ) -> dict[str, Decimal]:
        """Wartość rynkowa per symbol, TYLKO dla symboli z bieżącą ceną.

        Loty tego samego symbolu są agregowane. Symbol bez ceny wypada —
        to jest renormalizacja po `usable_symbols`: wypadnięcie z mianownika.
        """
        values: dict[str, Decimal] = {}
        for pos in self.positions:
            price = current_prices.get(pos.symbol)
            if price is None:
                continue
            values[pos.symbol] = values.get(pos.symbol, Decimal("0")) + (
                pos.market_value(price)
            )
        return values

    def weights(
        self, current_prices: Mapping[str, Decimal]
    ) -> dict[str, Decimal]:
        """Realne wagi kapitału wg wartości rynkowej, sumujące się do 1.

        Renormalizacja po `usable_symbols`: symbole bez bieżącej ceny nie
        wchodzą do mianownika. Pusty portfel lub brak policzalnej wartości →
        `{}` (nigdy dzielenie przez zero)."""
        values = self._usable_market_values(current_prices)
        total = sum(values.values(), Decimal("0"))
        if total == 0:
            return {}
        return {symbol: value / total for symbol, value in values.items()}

    def unrealized_pnl(
        self, current_prices: Mapping[str, Decimal]
    ) -> PortfolioPnL:
        """P/L per pozycja + sumy łączne. Pozycje bez ceny są pomijane."""
        rows: list[PositionPnL] = []
        total_cost = Decimal("0")
        total_value = Decimal("0")
        for pos in self.positions:
            price = current_prices.get(pos.symbol)
            if price is None:
                continue
            cost = pos.cost_basis()
            value = pos.market_value(price)
            pnl = pos.unrealized_pnl(price)
            rows.append(
                PositionPnL(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    purchase_price=pos.purchase_price,
                    current_price=price,
                    cost_basis=cost,
                    market_value=value,
                    unrealized_pnl=pnl,
                    return_pct=(pnl / cost) if cost != 0 else None,
                )
            )
            total_cost += cost
            total_value += value
        total_pnl = total_value - total_cost
        return PortfolioPnL(
            positions=tuple(rows),
            total_cost_basis=total_cost,
            total_market_value=total_value,
            total_unrealized_pnl=total_pnl,
            total_return_pct=(total_pnl / total_cost) if total_cost != 0 else None,
        )

    def cluster_exposure(
        self,
        clusters: Mapping[str, str],
        current_prices: Mapping[str, Decimal],
    ) -> list[ClusterExposure]:
        """Ekspozycja per klaster korelacji, posortowana malejąco po wadze.

        `clusters` to GOTOWA mapa symbol→id_klastra (korelacji nie liczymy
        tutaj). Symbol nieobecny w mapie tworzy własny singleton (klaster ==
        symbol), żeby nie zniknął z sumy. Wagi to te z `weights()` — więc
        symbole bez ceny naturalnie nie wchodzą. Pusty portfel → `[]`."""
        weights = self.weights(current_prices)
        grouped: dict[str, list[str]] = {}
        weight_by_cluster: dict[str, Decimal] = {}
        for symbol, weight in weights.items():
            cluster = clusters.get(symbol, symbol)
            grouped.setdefault(cluster, []).append(symbol)
            weight_by_cluster[cluster] = (
                weight_by_cluster.get(cluster, Decimal("0")) + weight
            )
        exposures = [
            ClusterExposure(
                cluster=cluster,
                symbols=tuple(sorted(grouped[cluster])),
                weight=weight_by_cluster[cluster],
            )
            for cluster in grouped
        ]
        # Malejąco po wadze; przy remisie stabilnie po nazwie klastra.
        exposures.sort(key=lambda e: (-e.weight, e.cluster))
        return exposures

    def is_stale(
        self, now: datetime, max_age_days: int = DEFAULT_MAX_POSITION_AGE_DAYS
    ) -> bool:
        """Czy tabela pozycji jest przeterminowana względem `now`."""
        return is_stale(self.as_of, now, max_age_days)

    def freshness(
        self, now: datetime, max_age_days: int = DEFAULT_MAX_POSITION_AGE_DAYS
    ) -> PortfolioFreshness:
        """Enum świeżości portfela względem `now` (wejście do badge STALE)."""
        return freshness(self.as_of, now, max_age_days)
