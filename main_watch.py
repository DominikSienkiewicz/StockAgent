"""Watch entry point — darmowy alert szoku rynkowego poza cyklem (#11).

Czwarty entry point obok `main_agent` (Fast Loop), `main_trainer` (Slow Loop)
i `main_backtest`. Zaprojektowany pod cron CO GODZINĘ, 24/7 — nie tylko w sesji
NYSE, bo BTC/ETH handlują całą dobę.

Dwa źródła ruchu (ALR-02). Pierwszeństwo ma dzienna zmiana z darmowego notowania
(`DailyChangePort`: Finnhub `dp` od poprzedniego zamknięcia, CoinGecko zmiana 24 h)
— niezależna od Fast Loopa. Snapshot Fast Loopa jest fallbackiem, gdy odczytu nie ma,
padł albo pochodzi z innego dnia UTC (sobota z piątkowym `dp`). Gdy oba źródła są
niedostępne albo stęchłe, watch jest ślepy i mówi to WARNING-iem zamiast udawać
zdrowe „0 alertów".

FinOps: **zero płatnych wywołań**. Watch dotyka wyłącznie darmowych źródeł ceny
(Finnhub / CoinGecko), odczytu Supabase i darmowych kanałów push. Żadnego LLM,
Alpha Vantage ani embeddingów — próg szoku jest wyższy od bramki volatility,
ale to i tak nie uprawnia do płatnych portów.

Dwa nienaruszalne kontrakty:

1. **Watch NIGDY nie zapisuje snapshotu ceny.** `save_price_snapshot` należy
   wyłącznie do Fast Loopa. Nadpisywanie snapshotu co godzinę policzyłoby deltę
   Fast Loopa względem ceny sprzed godziny, a nie sprzed cyklu — bramka
   volatility przestałaby cokolwiek przepuszczać.
2. **Debounce: jeden alert per symbol per dzień.** Spam przy chaotycznym rynku
   uczy użytkownika ignorować kanał. Egzekwowany dwustopniowo: predykatem
   domenowym `should_emit` i twardym `UNIQUE(symbol, alert_date)` z migracji 021.

ALR-07: za flagą `portfolio_shock_enabled` ten sam przebieg liczy też szok PORTFELA —
Σ waga × Δ po pozycjach (tabela `positions`) wobec `portfolio_shock_threshold`. Łapie
skoordynowany spadek wielu spółek, którego progi per symbol (2× bramki) nie widzą;
jeden push dziennie, debounce w `shock_alerts` pod symbolem „__PORTFOLIO__”.

ALR-11: ten sam przebieg sprawdza alerty poziomów cenowych użytkownika
(`scan_price_levels`, flaga `price_level_alerts_enabled`, tabela z migracji 057) —
darmowe ceny, przekroczenie liczone od `last_seen_price` alertu, nie od snapshotu.

Użycie:
    uv run python main_watch.py
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from functools import partial

from src.application.alert_context import (
    AlertContext,
    PositionContext,
    comparable_prices,
    position_context,
    render_alert_context,
)
from src.application.portfolio_fx import collect_current_pln_rates, nbp_today
from src.application.ports import (
    DailyChangePort,
    FxRatePort,
    LatestPrediction,
    MarketDataPort,
    PreviousVerdict,
    ReportNotifierPort,
    RepositoryPort,
)
from src.application.report_formatting import format_local
from src.application.report_ops import render_watch_summary_markdown
from src.config import Settings
from src.domain.asset import Asset, PriceDelta
from src.domain.number_format import format_number_pl, format_pct_pl
from src.domain.portfolio import PlnRates, Portfolio
from src.domain.price_levels import LevelDirection, PriceLevelAlert
from src.domain.quiet_hours import is_quiet
from src.domain.shock import (
    PORTFOLIO_ALERT_SYMBOL,
    ROLLING_CHANGE_WINDOW,
    SNAPSHOT_MAX_AGE_HOURS,
    DailyChange,
    PortfolioShock,
    ShockAlert,
    classify_shock,
    detect_portfolio_shock,
    detect_shock,
    is_daily_change_fresh,
    is_move_already_alerted,
    is_reference_stale,
    should_emit,
)
from src.domain.value_objects import AssetType, Money, Threshold
from src.infrastructure.logging_setup import configure_logging

logger = logging.getLogger(__name__)

# Punkt odniesienia ruchu w treści pusha — zależy od źródła (ALR-02).
SNAPSHOT_BASIS = "od ostatniego snapshotu ceny"
PREVIOUS_CLOSE_BASIS = "od poprzedniego zamknięcia"
ROLLING_24H_BASIS = "w ciągu ostatnich 24 h"


def _render_alert(
    alert: ShockAlert,
    *,
    basis: str,
    detected_at: datetime,
    display_timezone: str,
    context_lines: Sequence[str] = (),
) -> tuple[str, str]:
    """Treść pusha — krótka i jednoznaczna (temat, ciało). Po polsku, także liczby
    („-5,2%”, MAIL-58). Chwila wykrycia w strefie odbiorcy (MAIL-14) — logi watcha
    zostają w UTC. `basis` mówi, wobec czego liczony jest ruch. `context_lines`
    (ALR-08: pozycja, rada, ostatnia predykcja) stoją między ruchem a stopką."""
    arrow = "📉" if alert.direction.name == "DOWN" else "📈"
    move = format_pct_pl(alert.delta.percent, signed=True)
    subject = f"{arrow} Szok rynkowy: {alert.symbol} {move}"
    lines = [
        f"{arrow} {alert.symbol} {move} {basis} "
        f"(stan na {format_local(detected_at, display_timezone)}).",
        *context_lines,
        "Alert poza cyklem — pełna analiza w najbliższym dziennym digeście.",
    ]
    return subject, "\n".join(lines)


def _current_pln_rates(
    fx: FxRatePort | None, portfolio: Portfolio, moment: datetime
) -> PlnRates | None:
    """ALR-08: kursy NBP z dziś (Warszawa) obcych walut portfela. Portfel w jednej
    walucie albo brak portu → None bez zapytań; błąd pojedynczego kursu to brak tego
    kursu (`collect_current_pln_rates`), nigdy przerwany alert."""
    if fx is None or len({lot.currency for lot in portfolio.positions}) <= 1:
        return None
    return collect_current_pln_rates(portfolio, fx, today=nbp_today(moment))


def _load_user_portfolio(settings: Settings) -> Portfolio | None:
    """ALR-08: portfel z fabryki Fast Loopa — ta sama flaga `portfolio_positions_enabled`
    i ta sama reguła (błąd albo brak pozycji → None). Sam odczyt Supabase."""
    # Import lokalny — fabryka żyje w DI Containerze Fast Loopa.
    from main_agent import _build_user_portfolio

    return _build_user_portfolio(settings)


class _AlertContextReader:
    """ALR-08: darmowe odczyty kontekstu alertu, każdy best-effort.

    Błąd odczytu kosztuje tylko WARNING i linię kontekstu mniej — alert idzie dalej.
    Wynik spoza kontraktu portu (inny typ) traktujemy jak brak danych, żeby push nie
    niósł śmieci. Portfel czytamy raz na przebieg, przy pierwszym alercie; ceny pozycji
    to darmowy /quote. Żadnego zapisu — zwłaszcza snapshotu ceny."""

    def __init__(
        self,
        repository: RepositoryPort,
        market: MarketDataPort,
        portfolio_loader: Callable[[], Portfolio | None],
        rates_loader: Callable[[Portfolio], PlnRates | None] = lambda _portfolio: None,
    ) -> None:
        self._repository = repository
        self._market = market
        self._portfolio_loader = portfolio_loader
        self._rates_loader = rates_loader
        self._portfolio: Portfolio | None = None
        self._portfolio_read = False
        self._rates: PlnRates | None = None
        self._rates_read = False

    def read(self, symbol: str) -> AlertContext:
        return AlertContext(
            position=self._position(symbol),
            verdict=self._verdict(symbol),
            prediction=self._prediction(symbol),
        )

    def _verdict(self, symbol: str) -> PreviousVerdict | None:
        try:
            verdict = self._repository.get_previous_verdict(symbol)
        except Exception:
            logger.warning(
                "Alert context: previous verdict of %s unavailable", symbol, exc_info=True
            )
            return None
        return verdict if isinstance(verdict, PreviousVerdict) else None

    def _prediction(self, symbol: str) -> LatestPrediction | None:
        try:
            latest = self._repository.get_latest_prediction(symbol)
        except Exception:
            logger.warning(
                "Alert context: latest prediction of %s unavailable", symbol, exc_info=True
            )
            return None
        return latest if isinstance(latest, LatestPrediction) else None

    def _position(self, symbol: str) -> PositionContext | None:
        portfolio = self.user_portfolio()
        if portfolio is None or symbol not in {lot.symbol for lot in portfolio.positions}:
            return None
        return position_context(
            portfolio, symbol, self.held_prices(portfolio), self.rates(portfolio)
        )

    def rates(self, portfolio: Portfolio) -> PlnRates | None:
        """Kursy NBP do wag w złotych (ALR-08), raz na przebieg. None = bez kursów
        (portfel w jednej walucie ich nie potrzebuje; reszta pomija udział)."""
        if not self._rates_read:
            self._rates_read = True
            self._rates = self._rates_loader(portfolio)
        return self._rates

    def user_portfolio(self) -> Portfolio | None:
        if not self._portfolio_read:
            self._portfolio_read = True
            try:
                loaded = self._portfolio_loader()
            except Exception:
                logger.warning("Alert context: portfolio unavailable", exc_info=True)
                loaded = None
            self._portfolio = loaded if isinstance(loaded, Portfolio) else None
        return self._portfolio

    def held_prices(self, portfolio: Portfolio) -> dict[str, Decimal]:
        prices: dict[str, Decimal] = {}
        for symbol in sorted({lot.symbol for lot in portfolio.positions}):
            try:
                price = self._market.get_current_price(symbol)
            except Exception:
                logger.warning("Alert context: no free quote for %s", symbol, exc_info=True)
                continue
            if isinstance(price, Money):
                prices[symbol] = price.amount
        return prices


def _classify(symbol: str, crypto_symbols: set[str]) -> Asset:
    from src.domain.value_objects import AssetType

    asset_type = AssetType.CRYPTO if symbol in crypto_symbols else AssetType.STOCK
    return Asset(symbol=symbol, asset_type=asset_type)


def _snapshot_fraction(symbol: str, *, market: MarketDataPort, previous: Money) -> Decimal | None:
    """Zmiana ceny (ułamek) od referencji ze snapshotu. Brak ceny bieżącej albo cena
    odniesienia 0 → None (nie ma od czego liczyć delty)."""
    current = market.get_current_price(symbol)
    if current is None or previous.amount == 0:
        return None
    return (current.amount - previous.amount) / previous.amount


def _fresh_daily_change(
    daily_change: DailyChangePort | None, symbol: str, moment: datetime
) -> DailyChange | None:
    """Świeży odczyt dziennej zmiany albo None — wtedy watch wraca do snapshotu.

    Błąd źródła nie przerywa symbolu: snapshot Fast Loopa jest fallbackiem, więc
    padnięty /quote kosztuje tylko WARNING."""
    if daily_change is None:
        return None
    try:
        reading = daily_change.get_daily_change(symbol)
    except Exception:
        logger.warning(
            "Daily change unavailable for %s — falling back to the price snapshot",
            symbol,
            exc_info=True,
        )
        return None
    if not is_daily_change_fresh(reading.as_of, moment):
        logger.info(
            "Daily change for %s is from %s, not today (UTC) — falling back to the snapshot",
            symbol,
            reading.as_of.isoformat(),
        )
        return None
    return reading


def _crypto_alert_times(
    repository: RepositoryPort,
    daily_change: DailyChangePort | None,
    symbols: list[str],
    crypto: set[str],
    moment: datetime,
) -> dict[str, datetime]:
    """Chwile ostatnich alertów z okna 24 h — tylko gdy krypto idzie dzienną zmianą.

    Bez portu dziennej zmiany albo bez krypto na liście nie ma czego tłumić, więc
    nie ma też odczytu. Błąd odczytu → {} z logiem: lepiej ryzykować powtórzony
    alert niż zgubić nowy."""
    if daily_change is None or not crypto.intersection(symbols):
        return {}
    try:
        return repository.get_shock_alert_times(moment - ROLLING_CHANGE_WINDOW)
    except Exception:
        logger.exception("Failed to read shock alert times — crypto 24 h window not applied")
        return {}


def _daily_change_basis(asset: Asset) -> str:
    return ROLLING_24H_BASIS if asset.asset_type is AssetType.CRYPTO else PREVIOUS_CLOSE_BASIS


def _persist_shock_alert(
    repository: RepositoryPort, symbol: str, today: date, alert: ShockAlert
) -> None:
    """Persystencja PO wysyłce — lepiej wysłać dwa razy niż zgubić alert
    przy padzie bazy. UNIQUE(symbol, alert_date) i tak blokuje duplikat."""
    try:
        repository.save_shock_alert(
            symbol, today, float(alert.delta.fraction), alert.direction.value
        )
    except Exception:
        logger.exception("Failed to persist shock alert for %s", symbol)


@dataclass(frozen=True)
class WatchOutcome:
    """Wynik przebiegu: wysłane alerty i symbole, które padły (spośród `total`).

    `checked` — symbole z referencją ceny, `stale` — z referencją zbyt starą na
    alert (NIEZ-22: step summary pokazuje ślepotę watcha liczbami)."""

    sent: int
    failures: int
    total: int
    checked: int = 0
    stale: int = 0

    @property
    def all_failed(self) -> bool:
        return self.total > 0 and self.failures == self.total


def _silent_delivery(settings: Settings, moment: datetime) -> bool:
    """ALR-09: czy push tego przebiegu ma dotrzeć bez dźwięku (cisza nocna
    w strefie odbiorcy). Flaga off → push zawsze z dźwiękiem, jak przed ALR-09."""
    return settings.quiet_hours_enabled and is_quiet(
        moment,
        settings.display_timezone,
        settings.quiet_hours_start,
        settings.quiet_hours_end,
    )


def _watched_stocks(settings: Settings, moment: datetime) -> list[str]:
    """FIN-34: w dzień zamknięcia NYSE ceny akcji stoją — ta sama reguła i ten sam
    log co w Fast Loopie; krypto handluje w święta, więc zostaje."""
    # Import lokalny jak w `main()` — reguła żyje w DI Containerze Fast Loopa.
    from main_agent import STOCKS_SKIPPED_LOG, nyse_closure_label

    closure = nyse_closure_label(settings, moment)
    if closure is None:
        return list(settings.symbols)
    logger.info(STOCKS_SKIPPED_LOG, closure, len(settings.symbols), settings.crypto_symbols)
    return []


def _render_portfolio_alert(
    shock: PortfolioShock, *, detected_at: datetime, display_timezone: str
) -> tuple[str, str]:
    """ALR-07: „💼 Portfel -3,4% (NVDA -6,0%, AMD -5,0%)” — temat i treść pusha.
    Gdy część portfela nie miała policzonej zmiany, push mówi, z ilu procent liczy."""
    move = format_pct_pl(shock.delta.percent, signed=True)
    movers = ", ".join(
        f"{symbol} {format_pct_pl(PriceDelta(change).percent, signed=True)}"
        for symbol, change in shock.movers
    )
    lines = [
        f"💼 Portfel {move} ({movers}) od ostatnich referencji cen "
        f"(stan na {format_local(detected_at, display_timezone)}).",
    ]
    if shock.measured_weight < 1:
        share = format_pct_pl(shock.measured_weight * 100, signed=False, decimals=0)
        lines.append(f"Liczone z {share} portfela — reszta bez policzonej zmiany ceny.")
    lines.append("Alert poza cyklem — pełna analiza w najbliższym dziennym digeście.")
    return f"💼 Szok portfela: {move}", "\n".join(lines)


def _pre_move_prices(
    prices: dict[str, Decimal], deltas: dict[str, Decimal]
) -> dict[str, Decimal]:
    """Ceny SPRZED ruchu: cena bieżąca / (1 + Δ) dla symboli z policzoną zmianą,
    reszta bez zmian (jej wartość uznajemy za nieruszoną). Wagi sprzed ruchu to
    prawidłowe wagi dla Σ waga × Δ; cena spadająca do zera pomija symbol."""
    return {
        symbol: price / (1 + deltas[symbol]) if symbol in deltas and deltas[symbol] > -1 else price
        for symbol, price in prices.items()
    }


def _portfolio_shock_step(
    settings: Settings,
    *,
    repository: RepositoryPort,
    push_notifier: ReportNotifierPort | None,
    reader: _AlertContextReader,
    portfolio: Portfolio,
    deltas: dict[str, Decimal],
    moment: datetime,
    silent: bool,
    already_sent: set[tuple[str, date]],
) -> int:
    """ALR-07: szok portfela z ruchów policzonych w tym przebiegu. Zwraca 1, gdy
    push wyszedł. Bez kanału push: log błędu, bez wysyłki i bez zapisu debounce
    (zapis zablokowałby alert na dobę, choć nikt go nie dostał). Błąd wysyłki leci
    do pętli watcha, która liczy go jak błąd symbolu."""
    comparable = comparable_prices(
        portfolio, reader.held_prices(portfolio), reader.rates(portfolio)
    )
    if comparable is None:
        logger.warning(
            "Shock watch: portfel w kilku walutach bez kursów NBP — szok portfela pominięty"
        )
        return 0
    weights = portfolio.weights(_pre_move_prices(dict(comparable.prices), deltas))
    shock = detect_portfolio_shock(weights, deltas, settings.portfolio_shock_threshold)
    if shock is None:
        return 0
    if push_notifier is None:
        logger.error("Shock watch: szok portfela wykryty, ale brak kanału push")
        return 0
    subject, body = _render_portfolio_alert(
        shock, detected_at=moment, display_timezone=settings.display_timezone
    )
    push_notifier.send_report(subject=subject, html_body=body, plain_text=body, silent=silent)
    today = moment.date()
    _persist_shock_alert(
        repository,
        PORTFOLIO_ALERT_SYMBOL,
        today,
        ShockAlert(symbol=PORTFOLIO_ALERT_SYMBOL, delta=shock.delta, direction=shock.direction),
    )
    already_sent.add((PORTFOLIO_ALERT_SYMBOL, today))
    return 1


def scan_watch(
    settings: Settings,
    *,
    repository: RepositoryPort,
    market: MarketDataPort,
    push_notifier: ReportNotifierPort | None,
    now: datetime | None = None,
    daily_change: DailyChangePort | None = None,
    portfolio_loader: Callable[[], Portfolio | None] | None = None,
    fx_port: FxRatePort | None = None,
) -> WatchOutcome:
    """Jeden przebieg watcha.

    Resilience jak w Fast Loopie: błąd pojedynczego symbolu nie przerywa
    przebiegu. Wyłączona flaga → natychmiastowy zwrot, zero odczytów.
    `push_notifier=None` (brak kanału push): skan biegnie dalej, żeby sygnał
    ślepoty nie zniknął, ale wykryty szok nie jest ani wysyłany, ani zapisywany —
    zapis zablokowałby debounce na dobę dla alertu, którego nikt nie dostał (ALR-01).
    `daily_change` (ALR-02): świeży odczyt rozstrzyga o symbolu sam, bez snapshotu
    i bez ceny bieżącej — jedno źródło, najwyżej jeden alert. None → wyłącznie
    snapshoty, jak przed ALR-02.
    `portfolio_loader` (ALR-08): skąd wziąć portfel do linii „Twoja pozycja”;
    None → fabryka Fast Loopa (za `portfolio_positions_enabled`).
    `fx_port` (ALR-08): kursy NBP do udziału pozycji w złotych, gdy portfel trzyma
    kilka walut; None albo brak kursu → linia udziału pominięta, alert idzie.
    `portfolio_shock_enabled` (ALR-07): watch mierzy też symbole trzymane w portfelu,
    które dziś już zaalarmowały (ich ruch wchodzi do szoku portfela), a po pętli
    liczy szok portfela jako dodatkowy element przebiegu — jego błąd liczy się jak
    błąd symbolu i nie przerywa reszty.
    """
    if not settings.shock_alerts_enabled:
        logger.info("Shock watch wyłączony (shock_alerts_enabled=false).")
        return WatchOutcome(sent=0, failures=0, total=0)

    moment = now or datetime.now(UTC)
    today = moment.date()
    crypto = set(settings.crypto_symbols)
    unsupported = set(settings.symbols_unsupported_price)
    symbols = [
        s for s in (*_watched_stocks(settings, moment), *settings.crypto_symbols)
        if s not in unsupported
    ]

    try:
        already_sent: set[tuple[str, date]] = repository.get_sent_shock_alerts(today)
    except Exception:
        logger.exception("Failed to read shock alert history — debounce in-memory only")
        already_sent = set()
    crypto_alert_times = _crypto_alert_times(repository, daily_change, symbols, crypto, moment)
    silent = _silent_delivery(settings, moment)
    context_reader = _AlertContextReader(
        repository,
        market,
        portfolio_loader or partial(_load_user_portfolio, settings),
        lambda portfolio: _current_pln_rates(fx_port, portfolio, moment),
    )

    # ALR-07: szok portfela raz dziennie; portfel czytamy tylko, gdy jest co mierzyć.
    held: frozenset[str] = frozenset()
    portfolio: Portfolio | None = None
    if settings.portfolio_shock_enabled and should_emit(
        PORTFOLIO_ALERT_SYMBOL, today, already_sent
    ):
        portfolio = context_reader.user_portfolio()
        if portfolio is not None:
            held = frozenset(lot.symbol for lot in portfolio.positions)
    deltas: dict[str, Decimal] = {}
    work = [*symbols, *([PORTFOLIO_ALERT_SYMBOL] if portfolio is not None else [])]

    sent = 0
    failures = 0
    checked = 0
    stale = 0
    for symbol in work:
        try:
            if symbol == PORTFOLIO_ALERT_SYMBOL and portfolio is not None:
                sent += _portfolio_shock_step(
                    settings,
                    repository=repository,
                    push_notifier=push_notifier,
                    reader=context_reader,
                    portfolio=portfolio,
                    deltas=deltas,
                    moment=moment,
                    silent=silent,
                    already_sent=already_sent,
                )
                continue
            measure_only = False
            if not should_emit(symbol, today, already_sent):
                if symbol not in held:
                    continue
                # Symbol już zaalarmował, ale jego ruch należy do szoku portfela.
                measure_only = True
            reading = _fresh_daily_change(daily_change, symbol, moment)
            if reading is not None:
                checked += 1
                if symbol in held:
                    deltas[symbol] = reading.fraction
                if measure_only:
                    continue
                asset = _classify(symbol, crypto)
                # Decyzja 2: zmiana 24 h krypto wciąż zawiera ruch zaalarmowany < 24 h
                # temu — po północy UTC byłby to drugi alert za ten sam ruch.
                if asset.asset_type is AssetType.CRYPTO and is_move_already_alerted(
                    crypto_alert_times.get(symbol), moment
                ):
                    logger.info("Shock watch: %s zaalarmowany < 24 h temu — pomijam", symbol)
                    continue
                alert = classify_shock(
                    asset, PriceDelta(reading.fraction), Threshold(settings.volatility_threshold)
                )
                basis = _daily_change_basis(asset)
            else:
                snapshot = repository.get_last_price_snapshot(symbol)
                if snapshot is None:
                    continue
                previous, taken_at = snapshot
                age_hours = (moment - taken_at).total_seconds() / 3600.0
                checked += 1
                # Wiek PRZED pobraniem ceny: stęchła referencja i tak tłumi alert.
                if is_reference_stale(age_hours):
                    stale += 1
                    continue
                fraction = _snapshot_fraction(symbol, market=market, previous=previous)
                if fraction is None:
                    continue
                if symbol in held:
                    deltas[symbol] = fraction
                if measure_only:
                    continue
                alert = detect_shock(
                    _classify(symbol, crypto),
                    PriceDelta(fraction),
                    Threshold(settings.volatility_threshold),
                    age_hours,
                )
                basis = SNAPSHOT_BASIS
            if alert is None:
                continue
            if push_notifier is None:
                logger.error("Shock watch: szok %s wykryty, ale brak kanału push", symbol)
                continue

            subject, body = _render_alert(
                alert,
                basis=basis,
                detected_at=moment,
                display_timezone=settings.display_timezone,
                context_lines=render_alert_context(
                    context_reader.read(symbol),
                    now=moment,
                    display_timezone=settings.display_timezone,
                ),
            )
            push_notifier.send_report(
                subject=subject, html_body=body, plain_text=body, silent=silent
            )
            _persist_shock_alert(repository, symbol, today, alert)
            already_sent.add((symbol, today))
            sent += 1
        except Exception:
            logger.exception("Shock watch failed for symbol %s", symbol)
            failures += 1

    if checked and stale == checked:
        # Bez świeżej dziennej zmiany jedyne referencje to snapshoty Fast Loopa —
        # gdy on stoi, watch nie widzi niczego, a samo „0 alertów" wyglądałoby jak
        # zdrowy przebieg.
        logger.warning(
            "Shock watch ślepy: wszystkie %d referencje cen są starsze niż %.0f h "
            "(snapshoty zapisuje wyłącznie Fast Loop, dziennej zmiany brak) — "
            "żaden alert nie jest możliwy.",
            checked,
            SNAPSHOT_MAX_AGE_HOURS,
        )
    logger.info("Shock watch done — %d alert(ów) wysłanych", sent)
    return WatchOutcome(
        sent=sent, failures=failures, total=len(work), checked=checked, stale=stale
    )


def _level_text(value: Decimal) -> str:
    """Poziom po polsku z tyloma miejscami, ile ma wpis („150”, „150,5”)."""
    exponent = value.normalize().as_tuple().exponent
    decimals = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return format_number_pl(value, signed=False, decimals=decimals)


def _render_level_alert(
    alert: PriceLevelAlert, price: Decimal, *, detected_at: datetime, display_timezone: str
) -> tuple[str, str]:
    """ALR-11: „🎯 NVDA przebiła 150 (Twoja notatka: realizacja części zysku)”."""
    level = alert.level
    verb = "przebiła" if level.direction is LevelDirection.ABOVE else "spadła do"
    subject = f"🎯 {level.symbol} {verb} {_level_text(level.level)}"
    headline = f"{subject} (Twoja notatka: {level.note})" if level.note else subject
    body = "\n".join(
        [
            f"{headline}.",
            f"Cena: {format_number_pl(price, signed=False, decimals=2)} "
            f"(stan na {format_local(detected_at, display_timezone)}).",
            "Alert wyłączony po trafieniu — nowy poziom dopiszesz w price_alerts (config.toml).",
        ]
    )
    return subject, body


def _level_prices(
    market: MarketDataPort, symbols: set[str], unsupported: set[str]
) -> dict[str, Decimal]:
    """Darmowe ceny symboli z aktywnymi alertami — jeden /quote na symbol. Błąd
    symbolu kosztuje WARNING i pominięcie jego alertów w tym przebiegu."""
    prices: dict[str, Decimal] = {}
    for symbol in sorted(symbols - unsupported):
        try:
            prices[symbol] = market.get_current_price(symbol).amount
        except Exception:
            logger.warning("Price levels: no free quote for %s", symbol, exc_info=True)
    return prices


def scan_price_levels(
    settings: Settings,
    *,
    repository: RepositoryPort,
    market: MarketDataPort,
    push_notifier: ReportNotifierPort | None,
    now: datetime | None = None,
) -> int:
    """ALR-11: jeden przebieg alertów poziomów cenowych. Zwraca liczbę wysłanych.

    Flaga `price_level_alerts_enabled` off → zero odczytów i zapisów (przed migracją
    057 tabeli nie ma). Lista startowa `price_alerts` dopisuje brakujące poziomy.
    Przekroczenie liczy domena (`PriceLevelAlert.is_hit`) od `last_seen_price` —
    watch nie zapisuje `price_snapshots`. Push przed zapisem trafienia (lepiej dwa
    razy niż wcale); bez kanału push trafienie czeka nietknięte na następny przebieg.
    Zero płatnych portów."""
    if not settings.price_level_alerts_enabled:
        return 0
    moment = now or datetime.now(UTC)
    seeds = settings.price_level_seeds()
    try:
        if seeds:
            repository.add_price_levels(seeds)
        alerts = repository.get_active_price_levels()
    except Exception:
        logger.exception("Price levels unavailable (apply migration 057) — skipped")
        return 0
    prices = _level_prices(
        market,
        {alert.level.symbol for alert in alerts},
        set(settings.symbols_unsupported_price),
    )
    silent = _silent_delivery(settings, moment)
    sent = 0
    for alert in alerts:
        price = prices.get(alert.level.symbol)
        if price is None:
            continue
        try:
            if not alert.is_hit(price):
                repository.record_price_level_reading(alert.id, price)
                continue
            if push_notifier is None:
                logger.error(
                    "Price level %s %s hit, but no push channel", alert.level.symbol, alert.id
                )
                continue
            subject, body = _render_level_alert(
                alert, price, detected_at=moment, display_timezone=settings.display_timezone
            )
            push_notifier.send_report(
                subject=subject, html_body=body, plain_text=body, silent=silent
            )
            repository.mark_price_level_triggered(alert.id, price, moment)
            sent += 1
        except Exception:
            logger.exception("Price level alert %s failed", alert.id)
    logger.info("Price levels done — %d alert(ów) wysłanych", sent)
    return sent


def run_watch(
    settings: Settings,
    *,
    repository: RepositoryPort,
    market: MarketDataPort,
    push_notifier: ReportNotifierPort,
    now: datetime | None = None,
    daily_change: DailyChangePort | None = None,
    portfolio_loader: Callable[[], Portfolio | None] | None = None,
    fx_port: FxRatePort | None = None,
) -> int:
    """Jeden przebieg watcha. Zwraca liczbę WYSŁANYCH alertów.

    `daily_change=None` → wyłącznie snapshoty Fast Loopa (zachowanie sprzed ALR-02).
    `portfolio_loader` i `fx_port` jak w `scan_watch` (ALR-08)."""
    return scan_watch(
        settings,
        repository=repository,
        market=market,
        push_notifier=push_notifier,
        now=now,
        daily_change=daily_change,
        portfolio_loader=portfolio_loader,
        fx_port=fx_port,
    ).sent


def main(settings: Settings | None = None) -> int:
    resolved = settings or Settings.from_env()
    configure_logging("main_watch", resolved.secret_values())
    # Import lokalny — reguła pingu żyje w DI Containerze Fast Loopa.
    from main_agent import run_with_heartbeat

    # NIEZ-13: ping zewnętrznego heartbeatu na koniec (ok = exit 0).
    return run_with_heartbeat(resolved, lambda: _run_watch(resolved))


def _run_watch(settings: Settings) -> int:
    # Import lokalny — DI Container Fast Loopa buduje darmowe adaptery ceny
    # i kanały push; watch tylko je składa. Zero płatnych portów.
    from main_agent import (
        build_fx_port,
        build_market_port,
        build_push_channels,
        build_push_notifier,
        build_repository,
        write_step_summary,
    )

    push_notifier = build_push_notifier(settings) if build_push_channels(settings) else None
    if settings.shock_alerts_enabled and push_notifier is None:
        logger.error(
            "Shock watch bez kanału push (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID albo "
            "SLACK_WEBHOOK_URL) — wykryte szoki nie zostaną wysłane ani zapisane."
        )
    market = build_market_port(settings)
    repository = build_repository(settings)
    scan = partial(
        scan_watch,
        settings,
        repository=repository,
        market=market,
        push_notifier=push_notifier,
    )
    if settings.nbp_enabled:
        # ALR-08: darmowe kursy NBP do udziału pozycji w złotych, tylko za `nbp_enabled`.
        scan = partial(scan, fx_port=build_fx_port(settings))
    # ALR-02: router darmowych cen (Finnhub /quote, CoinGecko) jest też źródłem dziennej
    # zmiany. Port ceny bez tej zdolności → przebieg wyłącznie na snapshotach.
    outcome = scan(daily_change=market) if isinstance(market, DailyChangePort) else scan()
    # ALR-11: alerty poziomów cenowych — te same darmowe ceny i kanały push.
    scan_price_levels(
        settings, repository=repository, market=market, push_notifier=push_notifier
    )
    write_step_summary(
        lambda: render_watch_summary_markdown(
            checked=outcome.checked,
            stale=outcome.stale,
            sent=outcome.sent,
            failures=outcome.failures,
            total=outcome.total,
        )
    )
    # Brak alertów to normalny, zdrowy przebieg; padnięte WSZYSTKIE symbole (np. baza
    # nie odpowiada) już nie — jak w Fast Loopie, exit 1 (NIEZ-05).
    if outcome.all_failed:
        logger.error("Shock watch: wszystkie %d symbole padły", outcome.total)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
