"""Kursy NBP do P/L portfela w złotych (PORT-11) — sama orkiestracja portu.

Zbiera z `FxRatePort` kurs bieżący każdej obcej waluty portfela oraz kurs z dnia
zakupu każdej pary (waluta, data zakupu). Liczenie P/L, rozbicie na wynik
instrumentu i walutowy oraz decyzja, czego nie wiemy, siedzą w domenie
(`Portfolio.pln_pnl`). Pozycja bez daty zakupu nie dostaje żadnego kursu
z dnia zakupu — domena mówi wtedy wprost, że go nie zna.

FinOps: NBP jest darmowe; to wywołania poza bramką volatility, w fazie raportu.
Błąd pojedynczego kursu to brak tego kursu, nie przerwany raport.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.application.ports import FxRatePort
from src.domain.fx import FxRate
from src.domain.portfolio import HOME_CURRENCY, PlnRates, Portfolio

logger = logging.getLogger(__name__)

# NBP publikuje tabele wg kalendarza polskiego — „dziś” liczymy w Warszawie.
NBP_TIMEZONE = ZoneInfo("Europe/Warsaw")


def nbp_today(now: datetime) -> date:
    """Dzień kalendarzowy w Warszawie dla chwili `now` (aware)."""
    return now.astimezone(NBP_TIMEZONE).date()


def collect_pln_rates(portfolio: Portfolio, fx: FxRatePort, *, today: date) -> PlnRates:
    """Kursy bieżące (na `today`) i z dni zakupu dla obcych walut portfela."""
    foreign = [p for p in portfolio.positions if p.currency != HOME_CURRENCY]
    current: dict[str, Decimal] = {}
    for currency in sorted({p.currency for p in foreign}):
        rate = _rate(fx, currency, today)
        if rate is not None:
            current[currency] = rate.rate
    on_purchase: dict[tuple[str, date], Decimal] = {}
    purchases: set[tuple[str, date]] = set()
    for position in foreign:
        if position.purchase_date is not None:
            purchases.add((position.currency, position.purchase_date))
    for currency, day in sorted(purchases):
        rate = _rate(fx, currency, day)
        if rate is not None:
            on_purchase[(currency, day)] = rate.rate
    return PlnRates(current=current, on_purchase=on_purchase)


def _rate(fx: FxRatePort, currency: str, day: date) -> FxRate | None:
    try:
        return fx.get_rate(currency, day)
    except Exception:
        logger.exception("FX rate for %s on %s failed — left without a rate", currency, day)
        return None
