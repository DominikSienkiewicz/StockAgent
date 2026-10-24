"""Domena: odporny detektor anomalii ceny (DAT-15).

Pojedynczy zły tick (0,01, cena innego instrumentu, błąd agregatora) trafiał
jednocześnie do `price_snapshots`, wyznaczał deltę bramki (płatne wywołania), zwrot
rozliczenia i cechę ML. Tu mierzymy zwrot odczytu od ostatniej ceny historii
symbolu w jednostkach MAD (mediana odchyleń bezwzględnych) zwrotów tej historii —
miara odporna na pojedyncze skoki w samej historii, w przeciwieństwie do σ.

Prawdziwy szok (−20% po wynikach) też bywa wiele MAD od mediany, więc odczyt
z niezależnym potwierdzeniem nie jest anomalią: `pc` notowania (oficjalne zamknięcie
poprzedniej sesji) już niesie skok, a bieżąca cena leży blisko niego. Zero importów
zewnętrznych.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from statistics import median

# Próg w jednostkach MAD (katalog DAT-15): zły tick leży setki MAD od mediany,
# zwykły dzień kilka.
MAD_K = Decimal(8)
# Krótsza historia nie daje wiarygodnej skali — wtedy nic nie flagujemy.
MIN_HISTORY_RETURNS = 10
# Okno historii snapshotów, z którego graf liczy skalę (~20 sesji akcji, 30 dni krypto).
OUTLIER_HISTORY_DAYS = 30


def is_price_outlier(
    price: Decimal, history: Sequence[Decimal], *, k: Decimal = MAD_K
) -> bool:
    """Czy `price` leży dalej niż `k` MAD od mediany zwrotów `history` (ceny
    rosnąco po czasie), licząc zwrot od ostatniej ceny historii.

    Cena niedodatnia jest anomalią zawsze. Historia z mniej niż
    `MIN_HISTORY_RETURNS` zwrotami albo ze stałą ceną (MAD = 0) nie daje skali,
    więc odczyt przechodzi — detektor nie tłumi, gdy nie umie ocenić."""
    if k <= 0:
        raise ValueError(f"k must be positive (got {k})")
    if price <= 0:
        return True
    prices = [value for value in history if value > 0]
    returns = [
        current / previous - 1
        for previous, current in zip(prices, prices[1:], strict=False)
    ]
    if len(returns) < MIN_HISTORY_RETURNS:
        return False
    center = median(returns)
    mad = median(abs(value - center) for value in returns)
    if mad == 0:
        return False
    return abs(price / prices[-1] - 1 - center) > k * mad


def is_unconfirmed_outlier(
    price: Decimal,
    history: Sequence[Decimal],
    *,
    prev_close: Decimal | None,
    k: Decimal = MAD_K,
) -> bool:
    """Anomalia bez potwierdzenia: `price` jest anomalią względem `history`, a `pc`
    tego nie potwierdza. Potwierdzenie = `prev_close` sam jest anomalią (skok zaszedł
    już w poprzedniej sesji), a `price` nie jest nią względem historii z `prev_close`."""
    if not is_price_outlier(price, history, k=k):
        return False
    if prev_close is None or not is_price_outlier(prev_close, history, k=k):
        return True
    return is_price_outlier(price, [*history, prev_close], k=k)
