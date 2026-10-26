"""Domena skuteczności hedge'a — Hedge-Effectiveness Score (R3).

Czysta logika, ZERO importów zewnętrznych (tylko stdlib + inne moduły domeny).
Mierzymy, jak dobrze instrument zabezpieczający (inverse ETF, np. SH) faktycznie
ZNOSI ruch książki. Idealny inverse hedge porusza się przeciwnie do portfela
(korelacja -1), więc jego skuteczność to po prostu odwrócony znak korelacji.

Korzystamy z gotowej, sprawdzonej korelacji Pearsona z `src.domain.correlation`
(truncacja do krótszego szeregu, bramka na zerową wariancję i za krótkie serie,
domknięcie do [-1, 1]). Karmione DARMOWYMI snapshotami zwrotów — żadnych
płatnych portów.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.domain.correlation import pearson
from src.domain.macro_risk import MacroRiskInstrumentType

# PORT-04 (decyzja 2026-10-19): w sizingu hedge'em portfela akcji są tylko
# instrumenty odwrotne do akcji i długa zmienność. Złoto (SAFE_HAVEN), proxy
# suwerenne (EPOL) czy odwrotne obligacje trzymane w portfelu to inwestycja,
# nie zabezpieczenie akcji — nie mogą obniżać bandy jako „nieskuteczny hedge”.
EQUITY_HEDGE_TYPES: frozenset[MacroRiskInstrumentType] = frozenset(
    {
        MacroRiskInstrumentType.INVERSE_EQUITY,
        MacroRiskInstrumentType.VOLATILITY,
    }
)


def equity_hedge_symbols(
    symbol_types: Mapping[str, MacroRiskInstrumentType],
) -> frozenset[str]:
    """Symbole, które w sizingu liczą się jako hedge portfela akcji."""
    return frozenset(symbol for symbol, kind in symbol_types.items() if kind in EQUITY_HEDGE_TYPES)


def realized_correlation(book_returns: Sequence[float], hedge_returns: Sequence[float]) -> float:
    """Zrealizowana korelacja Pearsona między zwrotami książki a hedge'a.

    Wynik w [-1, 1]. Różne długości serii są przycinane do krótszej. Mniej niż
    2 punkty albo zerowa wariancja którejkolwiek serii → 0.0 (brak mierzalnej
    współzależności). Delegujemy do `correlation.pearson`, by zachować jedną
    spójną definicję korelacji w domenie.
    """
    return pearson(list(book_returns), list(hedge_returns))


def hedge_effectiveness(
    book_returns: Sequence[float],
    hedge_returns: Sequence[float],
    *,
    expect_inverse: bool = True,
) -> float:
    """Skuteczność hedge'a w [-1, 1]: +1 = idealny hedge, <=0 = zepsuty.

    Inverse hedge (np. SH, inverse ETF) powinien poruszać się PRZECIWNIE do
    książki, więc jego skuteczność to odwrócony znak korelacji
    (`-correlation`). Gdy `expect_inverse=False` (hedge ma podążać za książką),
    skuteczność równa się wprost korelacji (`+correlation`).
    """
    corr = realized_correlation(book_returns, hedge_returns)
    return -corr if expect_inverse else corr


@dataclass(frozen=True)
class HedgeAssessment:
    """Ocena pojedynczego instrumentu zabezpieczającego (value object).

    `hedge_symbol` to ticker hedge'a, `correlation` to surowa korelacja zwrotów
    z książką, `effectiveness` to znormalizowana skuteczność w [-1, 1], a
    `label` to gotowa polska etykieta ("skuteczny"/"nieskuteczny"/"odwrócony").
    """

    hedge_symbol: str
    correlation: float
    effectiveness: float
    label: str


def assess_hedge(
    hedge_symbol: str,
    book_returns: Sequence[float],
    hedge_returns: Sequence[float],
    *,
    expect_inverse: bool = True,
    effective_threshold: float = 0.3,
) -> HedgeAssessment:
    """Pełna ocena hedge'a wraz z czytelną etykietą po polsku.

    Reguła etykiet (próg domknięty obustronnie):
      - `effectiveness >= effective_threshold`  → "skuteczny"
      - `-threshold < effectiveness < threshold` → "nieskuteczny"
      - `effectiveness <= -effective_threshold` → "odwrócony"
    """
    corr = realized_correlation(book_returns, hedge_returns)
    effectiveness = -corr if expect_inverse else corr
    if effectiveness >= effective_threshold:
        label = "skuteczny"
    elif effectiveness <= -effective_threshold:
        label = "odwrócony"
    else:
        label = "nieskuteczny"
    return HedgeAssessment(
        hedge_symbol=hedge_symbol,
        correlation=corr,
        effectiveness=effectiveness,
        label=label,
    )
