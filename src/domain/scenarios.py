"""Domena stress-testów scenariuszowych — beta do rynku + "co jeśli" szoki.

Czysta logika, ZERO importów zewnętrznych (tylko stdlib, BEZ numpy). Dla każdego
symbolu liczymy betę względem rynku (wrażliwość zwrotów symbolu na zwroty rynku),
a potem rzutujemy hipotetyczny szok rynkowy na P&L per symbol i całego portfela.

Beta = cov(asset, market) / var(market) — populacyjne kowariancja i wariancja
(dzielnik n, nie n-1; szacujemy parametr próbki, nie wnioskujemy o populacji).
Karmione DARMOWYMI snapshotami cen / zwrotów — żadnych płatnych portów.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Minimalna liczba sparowanych punktów, by kowariancja/wariancja miały sens.
_MIN_PAIRED = 2


def beta_to_portfolio(
    asset_returns: Sequence[float], market_returns: Sequence[float]
) -> float:
    """Beta symbolu względem rynku: cov(asset, market) / var(market).

    Kowariancja i wariancja populacyjne (dzielnik n). Serie o różnej długości są
    truncowane do KRÓTSZEJ (wspólny prefiks) — historia bywa niewyrównana. Gdy
    rynek jest płaski (var(market) == 0), zwracamy 0.0 zamiast dzielenia przez
    zero — brak zmienności rynku = brak mierzalnej wrażliwości. Mniej niż 2
    sparowane punkty → 0.0.
    """
    n = min(len(asset_returns), len(market_returns))
    if n < _MIN_PAIRED:
        return 0.0
    assets = asset_returns[:n]
    markets = market_returns[:n]
    mean_a = sum(assets) / n
    mean_m = sum(markets) / n
    cov = sum((a - mean_a) * (m - mean_m) for a, m in zip(assets, markets, strict=True))
    var_m = sum((m - mean_m) ** 2 for m in markets)
    if var_m == 0:
        return 0.0
    return cov / var_m


@dataclass(frozen=True)
class Scenario:
    """Hipotetyczny scenariusz rynkowy: nazwa + ułamkowy szok rynku.

    `market_shock` jest ułamkowy (np. -0.10 = spadek rynku o 10%).
    """

    name: str
    market_shock: float


@dataclass(frozen=True)
class ScenarioImpact:
    """Wynik stress-testu jednego scenariusza na portfel.

    `per_symbol` to mapa symbol → szacowany P&L (ułamkowy) pod tym szokiem,
    `portfolio_pnl` to równoważona średnia tych wartości, a `worst_symbol`
    /`worst_pnl` wskazują najbardziej poszkodowany symbol (None/0.0 dla pustego
    portfela).
    """

    scenario_name: str
    market_shock: float
    portfolio_pnl: float
    per_symbol: dict[str, float]
    worst_symbol: str | None
    worst_pnl: float


def stress_test(
    returns_by_symbol: Mapping[str, Sequence[float]],
    market_returns: Sequence[float],
    scenarios: Sequence[Scenario],
) -> list[ScenarioImpact]:
    """Rzutuje każdy scenariusz na portfel przez betę symboli względem rynku.

    Najpierw liczymy betę raz na symbol (reuse między scenariuszami). Dla
    każdego scenariusza:

    - `per_symbol[sym] = beta_sym * market_shock`,
    - `portfolio_pnl` = równoważona średnia po `per_symbol` (0.0 dla pustego
      portfela),
    - `worst_symbol` = symbol o najbardziej UJEMNYM P&L (None, gdy brak symboli),
      `worst_pnl` = jego P&L (0.0, gdy brak).

    Brak scenariuszy → pusta lista. Pusty portfel → po jednym pustym
    `ScenarioImpact` na scenariusz (graceful).
    """
    betas = {
        symbol: beta_to_portfolio(returns, market_returns)
        for symbol, returns in returns_by_symbol.items()
    }

    impacts: list[ScenarioImpact] = []
    for scenario in scenarios:
        per_symbol = {
            symbol: beta * scenario.market_shock for symbol, beta in betas.items()
        }
        if per_symbol:
            portfolio_pnl = sum(per_symbol.values()) / len(per_symbol)
            worst_symbol, worst_pnl = min(per_symbol.items(), key=lambda kv: kv[1])
        else:
            portfolio_pnl = 0.0
            worst_symbol, worst_pnl = None, 0.0
        impacts.append(
            ScenarioImpact(
                scenario_name=scenario.name,
                market_shock=scenario.market_shock,
                portfolio_pnl=portfolio_pnl,
                per_symbol=per_symbol,
                worst_symbol=worst_symbol,
                worst_pnl=worst_pnl,
            )
        )
    return impacts
