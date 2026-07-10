"""Watch entry point — darmowy alert szoku rynkowego poza cyklem (#11).

Czwarty entry point obok `main_agent` (Fast Loop), `main_trainer` (Slow Loop)
i `main_backtest`. Uruchamiany cronem CO GODZINĘ, 24/7 — nie tylko w sesji NYSE,
bo BTC/ETH handlują całą dobę.

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
from datetime import UTC, date, datetime

from src.application.ports import MarketDataPort, ReportNotifierPort, RepositoryPort
from src.config import Settings
from src.domain.asset import Asset, PriceDelta
from src.domain.shock import ShockAlert, detect_shock, should_emit
from src.domain.value_objects import Threshold

logger = logging.getLogger(__name__)


def _render_alert(alert: ShockAlert) -> tuple[str, str]:
    """Treść pusha — krótka i jednoznaczna (temat, ciało). Po polsku."""
    arrow = "📉" if alert.direction.name == "DOWN" else "📈"
    pct = alert.delta.fraction * 100
    subject = f"{arrow} Szok rynkowy: {alert.symbol} {pct:+.1f}%"
    body = (
        f"{arrow} {alert.symbol} {pct:+.1f}% od ostatniego snapshotu ceny.\n"
        "Alert poza cyklem — pełna analiza w najbliższym dziennym digeście."
    )
    return subject, body


def _classify(symbol: str, crypto_symbols: set[str]) -> Asset:
    from src.domain.value_objects import AssetType

    asset_type = AssetType.CRYPTO if symbol in crypto_symbols else AssetType.STOCK
    return Asset(symbol=symbol, asset_type=asset_type)


def run_watch(
    settings: Settings,
    *,
    repository: RepositoryPort,
    market: MarketDataPort,
    push_notifier: ReportNotifierPort,
    now: datetime | None = None,
) -> int:
    """Jeden przebieg watcha. Zwraca liczbę WYSŁANYCH alertów.

    Resilience jak w Fast Loopie: błąd pojedynczego symbolu nie przerywa
    przebiegu. Wyłączona flaga → natychmiastowy zwrot, zero odczytów.
    """
    if not settings.shock_alerts_enabled:
        logger.info("Shock watch wyłączony (shock_alerts_enabled=false).")
        return 0

    moment = now or datetime.now(UTC)
    today = moment.date()
    crypto = set(settings.crypto_symbols)
    symbols = [*settings.symbols, *settings.crypto_symbols]
    unsupported = set(settings.symbols_unsupported_price)

    try:
        already_sent: set[tuple[str, date]] = repository.get_sent_shock_alerts(today)
    except Exception:
        logger.exception("Failed to read shock alert history — debounce in-memory only")
        already_sent = set()

    sent = 0
    for symbol in symbols:
        if symbol in unsupported:
            continue
        try:
            if not should_emit(symbol, today, already_sent):
                continue
            snapshot = repository.get_last_price_snapshot(symbol)
            if snapshot is None:
                continue
            previous, taken_at = snapshot
            current = market.get_current_price(symbol)
            if current is None or previous.amount == 0:
                continue

            fraction = (current.amount - previous.amount) / previous.amount
            delta = PriceDelta(fraction)
            age_hours = (moment - taken_at).total_seconds() / 3600.0
            alert = detect_shock(
                _classify(symbol, crypto),
                delta,
                Threshold(settings.volatility_threshold),
                age_hours,
            )
            if alert is None:
                continue

            subject, body = _render_alert(alert)
            push_notifier.send_report(
                subject=subject, html_body=body, plain_text=body
            )
            # Persystencja PO wysyłce — lepiej wysłać dwa razy niż zgubić alert
            # przy padzie bazy. UNIQUE(symbol, alert_date) i tak blokuje duplikat.
            try:
                repository.save_shock_alert(
                    symbol, today, float(alert.delta.fraction), alert.direction.value
                )
            except Exception:
                logger.exception("Failed to persist shock alert for %s", symbol)
            already_sent.add((symbol, today))
            sent += 1
        except Exception:
            logger.exception("Shock watch failed for symbol %s", symbol)

    logger.info("Shock watch done — %d alert(ów) wysłanych", sent)
    return sent


def main(settings: Settings | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = settings or Settings()  # type: ignore[call-arg]

    # Import lokalny — DI Container Fast Loopa buduje darmowe adaptery ceny
    # i kanały push; watch tylko je składa. Zero płatnych portów.
    from main_agent import build_market_port, build_push_notifier, build_repository

    run_watch(
        settings,
        repository=build_repository(settings),
        market=build_market_port(settings),
        push_notifier=build_push_notifier(settings),
    )
    # Watch jest best-effort: brak alertów to normalny, zdrowy przebieg.
    return 0


if __name__ == "__main__":
    sys.exit(main())
