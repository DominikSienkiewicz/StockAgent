"""Watch entry point — darmowy alert szoku rynkowego poza cyklem (#11).

Czwarty entry point obok `main_agent` (Fast Loop), `main_trainer` (Slow Loop)
i `main_backtest`. Zaprojektowany pod cron CO GODZINĘ, 24/7 — nie tylko w sesji
NYSE, bo BTC/ETH handlują całą dobę. Cron jest wstrzymany razem z Fast Loopem:
jedynymi referencjami cen są jego snapshoty, więc bez niego watch jest ślepy
(i mówi to WARNING-iem zamiast udawać zdrowe „0 alertów").

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
from dataclasses import dataclass
from datetime import UTC, date, datetime

from src.application.ports import MarketDataPort, ReportNotifierPort, RepositoryPort
from src.application.report_formatting import format_local
from src.application.report_ops import render_watch_summary_markdown
from src.config import Settings
from src.domain.asset import Asset, PriceDelta
from src.domain.number_format import format_pct_pl
from src.domain.shock import (
    SNAPSHOT_MAX_AGE_HOURS,
    ShockAlert,
    detect_shock,
    is_reference_stale,
    should_emit,
)
from src.domain.value_objects import Money, Threshold
from src.infrastructure.logging_redaction import configure_logging

logger = logging.getLogger(__name__)


def _render_alert(
    alert: ShockAlert, *, detected_at: datetime, display_timezone: str
) -> tuple[str, str]:
    """Treść pusha — krótka i jednoznaczna (temat, ciało). Po polsku, także liczby
    („-5,2%”, MAIL-58). Chwila wykrycia w strefie odbiorcy (MAIL-14) — logi watcha
    zostają w UTC."""
    arrow = "📉" if alert.direction.name == "DOWN" else "📈"
    move = format_pct_pl(alert.delta.percent, signed=True)
    subject = f"{arrow} Szok rynkowy: {alert.symbol} {move}"
    body = (
        f"{arrow} {alert.symbol} {move} od ostatniego snapshotu ceny "
        f"(stan na {format_local(detected_at, display_timezone)}).\n"
        "Alert poza cyklem — pełna analiza w najbliższym dziennym digeście."
    )
    return subject, body


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
) -> WatchOutcome:
    """Jeden przebieg watcha.

    Resilience jak w Fast Loopie: błąd pojedynczego symbolu nie przerywa
    przebiegu. Wyłączona flaga → natychmiastowy zwrot, zero odczytów.
    `push_notifier=None` (brak kanału push): skan biegnie dalej, żeby sygnał
    ślepoty nie zniknął, ale wykryty szok nie jest ani wysyłany, ani zapisywany —
    zapis zablokowałby debounce na dobę dla alertu, którego nikt nie dostał (ALR-01).
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

    sent = 0
    failures = 0
    checked = 0
    stale = 0
    for symbol in symbols:
        try:
            if not should_emit(symbol, today, already_sent):
                continue
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
            if alert is None:
                continue
            if push_notifier is None:
                logger.error("Shock watch: szok %s wykryty, ale brak kanału push", symbol)
                continue

            subject, body = _render_alert(
                alert, detected_at=moment, display_timezone=settings.display_timezone
            )
            push_notifier.send_report(
                subject=subject, html_body=body, plain_text=body
            )
            _persist_shock_alert(repository, symbol, today, alert)
            already_sent.add((symbol, today))
            sent += 1
        except Exception:
            logger.exception("Shock watch failed for symbol %s", symbol)
            failures += 1

    if checked and stale == checked:
        # Jedyne referencje to snapshoty Fast Loopa — gdy on stoi, watch nie
        # widzi niczego, a samo „0 alertów" wyglądałoby jak zdrowy przebieg.
        logger.warning(
            "Shock watch ślepy: wszystkie %d referencje cen są starsze niż %.0f h "
            "(snapshoty zapisuje wyłącznie Fast Loop) — żaden alert nie jest możliwy.",
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
) -> int:
    """Jeden przebieg watcha. Zwraca liczbę WYSŁANYCH alertów."""
    return scan_watch(
        settings, repository=repository, market=market, push_notifier=push_notifier, now=now
    ).sent


def main(settings: Settings | None = None) -> int:
    settings = settings or Settings()  # type: ignore[call-arg]
    configure_logging(
        settings.secret_values(), fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

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
    outcome = scan_watch(
        settings,
        repository=build_repository(settings),
        market=build_market_port(settings),
        push_notifier=push_notifier,
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
