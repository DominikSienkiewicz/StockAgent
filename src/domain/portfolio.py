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
  - `pln_pnl()` (PORT-11) — P/L w złotych: koszt po kursie NBP z dnia zakupu,
    wartość po kursie bieżącym, rozbicie na wynik instrumentu i wynik walutowy.
    Bez daty zakupu (albo kursu z tego dnia) P/L w zł jest jawnie nieznany —
    nigdy liczony bieżącym kursem po obu stronach, bo to ukrywa efekt walutowy.
  - `pln_prices()` — ceny trzymanych symboli w złotych (cena × kurs bieżący
    waluty pozycji). Podane do `weights`, `cluster_exposure` i `change_since`
    dają wagi, ekspozycję sektora i zmianę portfela w JEDNEJ walucie zamiast
    sumy dolarów z euro. Symbol bez kursu wypada (`unrated_symbols`), nigdy nie
    jest liczony jak USD.

Osobno żyje reguła świeżości: ręcznie aktualizowana tabela pozycji potrafi
się zestarzeć, a stęchły portfel daje FAŁSZYWY P/L — gorszy niż brak sekcji.
Dlatego `is_stale`/`freshness` z zamrożonym progiem `DEFAULT_MAX_POSITION_AGE_DAYS`
to wymaganie; warstwa raportu renderuje z tego badge STALE (przez `provenance`).

Zero I/O, zero importów zewnętrznych — tylko stdlib. `now` jest wstrzykiwane,
domena nigdy nie woła zegara. Pieniądze na `Decimal` (jak `Money`).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from src.domain.value_objects import Ratio

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

    `return_pct` to zwrot względem kosztu jako UŁAMEK (`Ratio`: 0.5 == +50%),
    mimo sufiksu `_pct` (ARCH-04); `None`, gdy koszt bazowy jest zerowy (np.
    pozycja o cenie zakupu 0) — nie dzielimy przez 0.
    """

    symbol: str
    quantity: Decimal
    purchase_price: Decimal
    current_price: Decimal
    cost_basis: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    return_pct: Ratio | None
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
class PlnRates:
    """PORT-11: kursy NBP (PLN za 1 jednostkę) do P/L w złotych.

    `current` — kurs bieżący per waluta; `on_purchase` — kurs z dnia zakupu per
    (waluta, data zakupu). PLN nie potrzebuje kursu (zawsze 1). Brak klucza =
    kurs nieznany; domena nigdy nie podstawia w zamian innego kursu.
    """

    current: Mapping[str, Decimal] = field(default_factory=dict)
    on_purchase: Mapping[tuple[str, date], Decimal] = field(default_factory=dict)

    def current_rate(self, currency: str) -> Decimal | None:
        return pln_rate(currency, self.current)

    def purchase_rate(self, currency: str, day: date) -> Decimal | None:
        if currency == HOME_CURRENCY:
            return Decimal("1")
        return self.on_purchase.get((currency, day))


@dataclass(frozen=True)
class PlnPrices:
    """Ceny trzymanych symboli w złotych i symbole, których nie da się przeliczyć.

    `prices` — symbol → cena × kurs bieżący waluty pozycji (PLN to 1); tylko
    symbole portfela z ceną. `unrated_symbols` (posortowane) — wycenione, ale bez
    kursu swojej waluty albo trzymane w dwóch walutach naraz: poza wagami i sumami.
    """

    prices: Mapping[str, Decimal]
    unrated_symbols: tuple[str, ...]


class PlnPnlGap(Enum):
    """PORT-11: dlaczego pozycja nie ma pełnego P/L w złotych.

    CURRENT_RATE_MISSING — brak bieżącego kursu waluty (nie ma nawet wartości zł).
    PURCHASE_DATE_UNKNOWN — brak daty zakupu, więc i kursu z tego dnia.
    PURCHASE_RATE_MISSING — data jest, ale źródło nie dało kursu z tego dnia.
    """

    CURRENT_RATE_MISSING = "CURRENT_RATE_MISSING"
    PURCHASE_DATE_UNKNOWN = "PURCHASE_DATE_UNKNOWN"
    PURCHASE_RATE_MISSING = "PURCHASE_RATE_MISSING"


@dataclass(frozen=True)
class PositionPlnPnL:
    """PORT-11: P/L jednej pozycji w złotych z rozbiciem.

    `cost_pln` = ilość × cena zakupu × kurs z dnia zakupu; `value_pln` = ilość ×
    cena bieżąca × kurs bieżący; `pnl_pln` = wartość − koszt. Rozbicie:
    `instrument_pln` = ilość × (cena bieżąca − cena zakupu) × kurs bieżący (ruch
    samego instrumentu), `fx_pln` = ilość × cena zakupu × (kurs bieżący − kurs
    z dnia zakupu) (zmiana kursu na zainwestowanym kapitale); suma = `pnl_pln`.
    `gap` niepusty → kwoty bez pełnego kursu są None.
    """

    symbol: str
    currency: str
    value_pln: Decimal | None
    cost_pln: Decimal | None
    pnl_pln: Decimal | None
    instrument_pln: Decimal | None
    fx_pln: Decimal | None
    gap: PlnPnlGap | None


@dataclass(frozen=True)
class PortfolioPlnPnL:
    """PORT-11: P/L portfela w złotych — wiersze w kolejności `unrealized_pnl`.

    `total_value_pln` sumuje pozycje z kursem bieżącym; `total_pnl_pln`,
    `total_instrument_pln` i `total_fx_pln` — tylko pozycje z pełnym rozbiciem.
    `incomplete_symbols` (posortowane) to pozycje, których P/L zł nie znamy.
    """

    positions: tuple[PositionPlnPnL, ...]
    total_value_pln: Decimal
    total_pnl_pln: Decimal
    total_instrument_pln: Decimal
    total_fx_pln: Decimal
    incomplete_symbols: tuple[str, ...]


@dataclass(frozen=True)
class PortfolioPnL:
    """Zagregowany P/L portfela — pozycje policzalne + sumy łączne.

    `positions` zawiera tylko pozycje, dla których była bieżąca cena;
    reszta wypada (bez ceny nie ma P/L). `total_return_pct` to zwrot łączny
    jako UŁAMEK (`Ratio`, mimo sufiksu `_pct`, ARCH-04) albo `None` dla
    pustego/zerokosztowego portfela.
    """

    positions: tuple[PositionPnL, ...]
    total_cost_basis: Decimal
    total_market_value: Decimal
    total_unrealized_pnl: Decimal
    total_return_pct: Ratio | None


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
    `change_pct` to zmiana względem wartości poprzedniej jako UŁAMEK (`Ratio`,
    mimo sufiksu `_pct`, ARCH-04).
    """

    previous_value: Decimal
    current_value: Decimal
    change: Decimal
    change_pct: Ratio
    top_contributors: tuple[SymbolChange, ...]
    unmeasured_symbols: tuple[str, ...]


class PortfolioFreshness(Enum):
    """Świeżość ręcznie aktualizowanej tabeli pozycji.

    FRESH — dane w progu wieku, P/L wiarygodny.
    STALE — pozycje starsze niż próg; P/L może być fałszywy → badge STALE.
    """

    FRESH = "FRESH"
    STALE = "STALE"


def _position_pln_pnl(position: Position, price: Decimal, rates: PlnRates) -> PositionPlnPnL:
    """PORT-11: jeden wiersz P/L w złotych (patrz `PositionPlnPnL`)."""
    current_rate = rates.current_rate(position.currency)
    if current_rate is None:
        return _incomplete(position, None, PlnPnlGap.CURRENT_RATE_MISSING)
    value = position.market_value(price) * current_rate
    if position.purchase_date is None:
        return _incomplete(position, value, PlnPnlGap.PURCHASE_DATE_UNKNOWN)
    purchase_rate = rates.purchase_rate(position.currency, position.purchase_date)
    if purchase_rate is None:
        return _incomplete(position, value, PlnPnlGap.PURCHASE_RATE_MISSING)
    cost = position.cost_basis() * purchase_rate
    return PositionPlnPnL(
        symbol=position.symbol,
        currency=position.currency,
        value_pln=value,
        cost_pln=cost,
        pnl_pln=value - cost,
        instrument_pln=position.unrealized_pnl(price) * current_rate,
        fx_pln=position.cost_basis() * (current_rate - purchase_rate),
        gap=None,
    )


def _incomplete(position: Position, value: Decimal | None, gap: PlnPnlGap) -> PositionPlnPnL:
    return PositionPlnPnL(
        symbol=position.symbol,
        currency=position.currency,
        value_pln=value,
        cost_pln=None,
        pnl_pln=None,
        instrument_pln=None,
        fx_pln=None,
        gap=gap,
    )


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
        self, current_prices: Mapping[str, Decimal], *, rates: PlnRates | None = None
    ) -> Decimal | None:
        """PORT-03: udział pozycji bez ceny w koszcie CAŁEGO portfela.

        Koszt, nie wartość rynkowa — bez ceny wartości nie znamy. Portfel bez
        żadnego kosztu (pusty albo same zerowe ceny zakupu) → None.

        Z `rates` koszty liczymy w złotych po kursie bieżącym waluty pozycji
        (udział, nie kwota). Wyceniona pozycja bez kursu wypada z mianownika;
        niewyceniona bez kursu → None, bo jej udziału nie znamy."""
        costs: list[tuple[Position, Decimal]] = []
        for pos in self.positions:
            rate = Decimal("1") if rates is None else rates.current_rate(pos.currency)
            if rate is None:
                if pos.symbol not in current_prices:
                    return None
                continue
            costs.append((pos, pos.cost_basis() * rate))
        total = sum((cost for _, cost in costs), Decimal("0"))
        if total == 0:
            return None
        unpriced = sum(
            (cost for pos, cost in costs if pos.symbol not in current_prices), Decimal("0")
        )
        return unpriced / total

    def pln_prices(
        self, current_prices: Mapping[str, Decimal], rates: PlnRates
    ) -> PlnPrices:
        """Ceny trzymanych symboli w złotych (patrz `PlnPrices`)."""
        currencies: dict[str, set[str]] = {}
        for pos in self.positions:
            currencies.setdefault(pos.symbol, set()).add(pos.currency)
        converted: dict[str, Decimal] = {}
        unrated: list[str] = []
        for symbol, codes in currencies.items():
            price = current_prices.get(symbol)
            if price is None:
                continue
            rate = rates.current_rate(next(iter(codes))) if len(codes) == 1 else None
            if rate is None:
                unrated.append(symbol)
                continue
            converted[symbol] = price * rate
        return PlnPrices(prices=converted, unrated_symbols=tuple(sorted(unrated)))

    def excluding(self, symbols: Collection[str]) -> Portfolio:
        """Ten sam portfel (i ta sama świeżość) bez pozycji w `symbols`."""
        return Portfolio(
            positions=tuple(pos for pos in self.positions if pos.symbol not in symbols),
            as_of=self.as_of,
        )

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
                    return_pct=Ratio(pnl / cost) if cost != 0 else None,
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
            total_return_pct=Ratio(total_pnl / total_cost) if total_cost != 0 else None,
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

    def pln_pnl(
        self, current_prices: Mapping[str, Decimal], rates: PlnRates
    ) -> PortfolioPlnPnL:
        """PORT-11: P/L w złotych per pozycja i łącznie (pozycje bez ceny pomijane).

        Kurs z dnia zakupu bierze `purchase_date` pozycji — bez niej P/L zł jest
        nieznany (`PlnPnlGap.PURCHASE_DATE_UNKNOWN`), a nie przybliżony."""
        rows = tuple(
            _position_pln_pnl(pos, price, rates)
            for pos in self.positions
            if (price := current_prices.get(pos.symbol)) is not None
        )
        complete = [row for row in rows if row.gap is None]
        return PortfolioPlnPnL(
            positions=rows,
            total_value_pln=sum(
                (row.value_pln for row in rows if row.value_pln is not None), Decimal("0")
            ),
            total_pnl_pln=sum((row.pnl_pln or Decimal("0") for row in complete), Decimal("0")),
            total_instrument_pln=sum(
                (row.instrument_pln or Decimal("0") for row in complete), Decimal("0")
            ),
            total_fx_pln=sum((row.fx_pln or Decimal("0") for row in complete), Decimal("0")),
            incomplete_symbols=tuple(sorted({row.symbol for row in rows if row.gap is not None})),
        )

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
            change_pct=Ratio(change / previous_value),
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
