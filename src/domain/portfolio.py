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
  - `change_since()` (PORT-06) — zmiana wartości portfela między dwoma zestawami
    cen (poprzedni cykl → bieżący) z top kontrybutorami.
  - `market_value_pln()` (PORT-18) — wartość w PLN kursem waluty KAŻDEJ pozycji
    (pozycja EUR kursem EUR/PLN, nie USD/PLN); pozycja bez kursu swojej waluty
    jest wymieniona, a nie wyceniona cudzym kursem.

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

# PORT-06: ilu kontrybutorów zmiany portfela pokazuje raport.
TOP_CHANGE_CONTRIBUTORS = 3

# PORT-18: waluta kwot „zł” w raporcie i waluta pozycji, gdy źródło jej nie podaje
# (tabela `positions` przed migracją 035 nie ma kolumny `currency`).
HOME_CURRENCY = "PLN"
DEFAULT_POSITION_CURRENCY = "USD"


def _require_finite(name: str, value: Decimal) -> None:
    """Odrzuca NaN/Inf, które cicho zatruwają porównania i sumy w domenie."""
    if value.is_nan() or value.is_infinite():
        raise ValueError(f"{name} must be a finite number (got {value})")


def is_iso_currency(code: str) -> bool:
    """Czy `code` ma format kodu ISO 4217: trzy wielkie litery ASCII."""
    return len(code) == 3 and code.isascii() and code.isalpha() and code.isupper()


def pln_rate(currency: str, rates: Mapping[str, Decimal]) -> Decimal | None:
    """PORT-18: kurs PLN za 1 jednostkę `currency` — PLN to 1, inna waluta
    z mapy `rates` albo None (nigdy kurs innej waluty w zastępstwie)."""
    if currency == HOME_CURRENCY:
        return Decimal("1")
    return rates.get(currency)


@dataclass(frozen=True)
class Position:
    """Pojedyncza pozycja (lot) w portfelu.

    `quantity` — liczba jednostek (musi być dodatnia; short poza zakresem
    MVP retail). `purchase_price` — cena zakupu za jednostkę (nieujemna),
    w walucie `currency`. `purchase_date` — prawdziwa data nabycia lotu
    (PORT-18); None = nieznana — wtedy kurs z dnia zakupu też jest nieznany,
    a nie przybliżany datą aktualizacji tabeli. `currency` — waluta notowania
    (kod ISO 4217); ceny zakupu i bieżąca są w tej walucie.
    """

    symbol: str
    quantity: Decimal
    purchase_price: Decimal
    purchase_date: date | None = None
    currency: str = DEFAULT_POSITION_CURRENCY

    def __post_init__(self) -> None:
        _require_finite("quantity", self.quantity)
        _require_finite("purchase_price", self.purchase_price)
        if not is_iso_currency(self.currency):
            raise ValueError(f"currency must be an ISO 4217 code (got {self.currency!r})")
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
    # PORT-18: waluta, w której są ceny i kwoty tego wiersza.
    currency: str = DEFAULT_POSITION_CURRENCY


@dataclass(frozen=True)
class PlnValuation:
    """PORT-18: wartość rynkowa w PLN kursem waluty każdej pozycji.

    `value` sumuje wycenione pozycje z kursem swojej waluty. `unrated_symbols`
    — wycenione pozycje bez kursu swojej waluty (posortowane, bez duplikatów):
    nie wchodzą do sumy, bo cudzy kurs dałby fałszywą kwotę.
    """

    value: Decimal
    unrated_symbols: tuple[str, ...]


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


@dataclass(frozen=True)
class SymbolChange:
    """PORT-06: wkład jednego symbolu w zmianę wartości portfela.

    `change` — Δ wartości pozycji (ilość × (cena bieżąca − poprzednia), loty
    zsumowane). `contribution` — ten Δ jako ułamek wartości portfela PRZED
    zmianą; wkłady wszystkich symboli sumują się do `PortfolioChange.change_pct`.
    """

    symbol: str
    change: Decimal
    contribution: Decimal


@dataclass(frozen=True)
class PortfolioChange:
    """PORT-06: zmiana wartości portfela między dwoma zestawami cen.

    Liczona TYLKO z pozycji, które mają cenę poprzednią i bieżącą — pozycja bez
    punktu odniesienia nie może udawać „bez zmian”. Te wypadnięte są wymienione
    w `unmeasured_symbols`. `top_contributors` — co najwyżej
    `TOP_CHANGE_CONTRIBUTORS` symboli o największym |Δ|, bez pozycji stojących.
    """

    previous_value: Decimal
    current_value: Decimal
    change: Decimal
    change_pct: Decimal
    top_contributors: tuple[SymbolChange, ...]
    unmeasured_symbols: tuple[str, ...]


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

    def unpriced_symbols(
        self, current_prices: Mapping[str, Decimal]
    ) -> tuple[str, ...]:
        """PORT-03: symbole bez bieżącej ceny, posortowane i bez duplikatów.

        To dokładnie te pozycje, które `weights` i `unrealized_pnl` pomijają —
        raport musi je wymienić, bo bez nich wagi reszty są zawyżone."""
        return tuple(
            sorted({pos.symbol for pos in self.positions if pos.symbol not in current_prices})
        )

    def unpriced_cost_share(
        self, current_prices: Mapping[str, Decimal]
    ) -> Decimal | None:
        """PORT-03: udział pozycji bez ceny w koszcie CAŁEGO portfela.

        Koszt, nie wartość rynkowa — bez ceny wartości nie znamy. Portfel bez
        żadnego kosztu (pusty albo same zerowe ceny zakupu) → None."""
        total = sum((pos.cost_basis() for pos in self.positions), Decimal("0"))
        if total == 0:
            return None
        unpriced = sum(
            (pos.cost_basis() for pos in self.positions if pos.symbol not in current_prices),
            Decimal("0"),
        )
        return unpriced / total

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
                    currency=pos.currency,
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

    def market_value_pln(
        self,
        current_prices: Mapping[str, Decimal],
        rates: Mapping[str, Decimal],
    ) -> PlnValuation:
        """PORT-18: wartość wycenionych pozycji w PLN kursem ich waluty.

        `rates` — kurs PLN za 1 jednostkę waluty (np. {"USD": 4.00, "EUR": 4.30});
        PLN nie potrzebuje kursu. Pozycja bez ceny wypada (jak w `weights`),
        pozycja bez kursu swojej waluty trafia do `unrated_symbols`."""
        value = Decimal("0")
        unrated: set[str] = set()
        for pos in self.positions:
            price = current_prices.get(pos.symbol)
            if price is None:
                continue
            rate = pln_rate(pos.currency, rates)
            if rate is None:
                unrated.add(pos.symbol)
                continue
            value += pos.market_value(price) * rate
        return PlnValuation(value=value, unrated_symbols=tuple(sorted(unrated)))

    def change_since(
        self,
        previous_prices: Mapping[str, Decimal],
        current_prices: Mapping[str, Decimal],
    ) -> PortfolioChange | None:
        """PORT-06: Δ wartości portfela od `previous_prices` do `current_prices`.

        Ruch symbolu waży tyle, ile jego wartość rynkowa: NVDA +10% przy 75%
        kapitału to +7,5% portfela. Pozycje bez obu cen wypadają z obu stron.
        None = żadnej mierzalnej pozycji albo zerowa wartość poprzednia (nie
        dzielimy przez zero)."""
        quantities: dict[str, Decimal] = {}
        for pos in self.positions:
            quantities[pos.symbol] = quantities.get(pos.symbol, Decimal("0")) + pos.quantity
        measured = [
            symbol
            for symbol in quantities
            if symbol in previous_prices and symbol in current_prices
        ]
        previous_value = sum(
            (quantities[s] * previous_prices[s] for s in measured), Decimal("0")
        )
        if not measured or previous_value == 0:
            return None
        current_value = sum(
            (quantities[s] * current_prices[s] for s in measured), Decimal("0")
        )
        moves = [
            SymbolChange(symbol=s, change=change, contribution=change / previous_value)
            for s in measured
            if (change := quantities[s] * (current_prices[s] - previous_prices[s])) != 0
        ]
        # Malejąco po |Δ|; przy remisie stabilnie po symbolu.
        moves.sort(key=lambda m: (-abs(m.change), m.symbol))
        change = current_value - previous_value
        return PortfolioChange(
            previous_value=previous_value,
            current_value=current_value,
            change=change,
            change_pct=change / previous_value,
            top_contributors=tuple(moves[:TOP_CHANGE_CONTRIBUTORS]),
            unmeasured_symbols=tuple(sorted(set(quantities) - set(measured))),
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
