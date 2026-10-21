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
    position_context,
    render_alert_context,
)
from src.application.ports import (
    DailyChangePort,
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
from src.domain.number_format import format_pct_pl
from src.domain.portfolio import Portfolio
from src.domain.quiet_hours import is_quiet
from src.domain.shock import (
    ROLLING_CHANGE_WINDOW,
    SNAPSHOT_MAX_AGE_HOURS,
    DailyChange,
    ShockAlert,
    classify_shock,
    detect_shock,
    is_daily_change_fresh,
    is_move_already_alerted,
    is_reference_stale,
    should_emit,
)
from src.domain.value_objects import AssetType, Money, Threshold
from src.infrastructure.logging_redaction import configure_logging

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
    ) -> None:
        self._repository = repository
        self._market = market
        self._portfolio_loader = portfolio_loader
        self._portfolio: Portfolio | None = None
        self._portfolio_read = False

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
        portfolio = self._user_portfolio()
        if portfolio is None or symbol not in {lot.symbol for lot in portfolio.positions}:
            return None
        return position_context(portfolio, symbol, self._held_prices(portfolio))

    def _user_portfolio(self) -> Portfolio | None:
        if not self._portfolio_read:
            self._portfolio_read = True
            try:
                loaded = self._portfolio_loader()
            except Exception:
                logger.warning("Alert context: portfolio unavailable", exc_info=True)
                loaded = None
            self._portfolio = loaded if isinstance(loaded, Portfolio) else None
        return self._portfolio

    def _held_prices(self, portfolio: Portfolio) -> dict[str, Decimal]:
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


def _detect_symbol_shock(
    symbol: str,
    *,
    settings: Settings,
    market: MarketDataPort,
    crypto: set[str],
    previous: Money,
    age_hours: float,
) -> ShockAlert | None:
    """Czy symbol kwalifikuje się na alert szoku wobec świeżej referencji.
    Brak ceny bieżącej albo cena odniesienia 0 → None (nie ma od czego liczyć
    delty)."""
    current = market.get_current_price(symbol)
    if current is None or previous.amount == 0:
        return None

    fraction = (current.amount - previous.amount) / previous.amount
    return detect_shock(
        _classify(symbol, crypto),
        PriceDelta(fraction),
        Threshold(settings.volatility_threshold),
        age_hours,
    )


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


def scan_watch(
    settings: Settings,
    *,
    repository: RepositoryPort,
    market: MarketDataPort,
    push_notifier: ReportNotifierPort | None,
    now: datetime | None = None,
    daily_change: DailyChangePort | None = None,
    portfolio_loader: Callable[[], Portfolio | None] | None = None,
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
        repository, market, portfolio_loader or partial(_load_user_portfolio, settings)
    )

    sent = 0
    failures = 0
    checked = 0
    stale = 0
    for symbol in symbols:
        try:
            if not should_emit(symbol, today, already_sent):
                continue
            reading = _fresh_daily_change(daily_change, symbol, moment)
            if reading is not None:
                checked += 1
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
                alert = _detect_symbol_shock(
                    symbol,
                    settings=settings,
                    market=market,
                    crypto=crypto,
                    previous=previous,
                    age_hours=age_hours,
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
        sent=sent, failures=failures, total=len(symbols), checked=checked, stale=stale
    )


def run_watch(
    settings: Settings,
    *,
    repository: RepositoryPort,
    market: MarketDataPort,
    push_notifier: ReportNotifierPort,
    now: datetime | None = None,
    daily_change: DailyChangePort | None = None,
    portfolio_loader: Callable[[], Portfolio | None] | None = None,
) -> int:
    """Jeden przebieg watcha. Zwraca liczbę WYSŁANYCH alertów.

    `daily_change=None` → wyłącznie snapshoty Fast Loopa (zachowanie sprzed ALR-02).
    `portfolio_loader` jak w `scan_watch` (ALR-08)."""
    return scan_watch(
        settings,
        repository=repository,
        market=market,
        push_notifier=push_notifier,
        now=now,
        daily_change=daily_change,
        portfolio_loader=portfolio_loader,
    ).sent


def main(settings: Settings | None = None) -> int:
    resolved = settings or Settings()  # type: ignore[call-arg]
    configure_logging(
        resolved.secret_values(), fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    # Import lokalny — reguła pingu żyje w DI Containerze Fast Loopa.
    from main_agent import run_with_heartbeat

    # NIEZ-13: ping zewnętrznego heartbeatu na koniec (ok = exit 0).
    return run_with_heartbeat(resolved, lambda: _run_watch(resolved))


def _run_watch(settings: Settings) -> int:
    # Import lokalny — DI Container Fast Loopa buduje darmowe adaptery ceny
    # i kanały push; watch tylko je składa. Zero płatnych portów.
    from main_agent import (
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
    scan = partial(
        scan_watch,
        settings,
        repository=build_repository(settings),
        market=market,
        push_notifier=push_notifier,
    )
    # ALR-02: router darmowych cen (Finnhub /quote, CoinGecko) jest też źródłem dziennej
    # zmiany. Port ceny bez tej zdolności → przebieg wyłącznie na snapshotach.
    outcome = scan(daily_change=market) if isinstance(market, DailyChangePort) else scan()
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
