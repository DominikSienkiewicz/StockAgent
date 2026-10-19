"""Domena Equity Curve — krzywa kapitału z track recordu predykcji agenta.

Czysta domena (stdlib only). Bierze prymitywy, NIE `ResolvedPrediction` —
mapowanie z modelu raportu na pary `(trend, realized_change_pct)` robi
orkiestrator. Dzięki temu moduł jest odsprzężony od warstwy aplikacji.

Każdy trade to `(predicted_trend, realized_change_pct)`, gdzie zmiana ceny jest
ułamkiem (np. 0.03 = +3%). Sygnał kierunku: BULLISH→+1, BEARISH→-1,
SIDEWAYS/inne→0. Zwrot okresu = sygnał * zmiana ceny, więc trafiony short
(cena spadła) daje zysk, a chybiony long — stratę. Kapitał składa się
procentowo: equity_i = equity_{i-1} * (1 + period_return).

Dwie krzywe:
  - `equity_curve` — każdy trade to osobny okres na 100% kapitału (szeregowo),
  - `portfolio_equity_curve` (WIA-17) — jeden okres na CYKL: kapitał dzielony
    po równo między predykcje cyklu, SIDEWAYS trzyma swoją część w gotówce.
    Równoległe trade'y jednego dnia nie składają się więc jak kolejne zakłady.
    `paper_portfolio_curve` (WIA-50) to ta sama krzywa z kosztem od obrotu
    per symbol (`trading_costs`): trade niesie symbol i własny koszt w bps.

Koszt zmienia PnL, nie trafność: `win_rate` liczy się z zwrotu brutto.

Tu mieszka też jedyna definicja hit-rate kierunkowego (`directional_hit_rate`,
WIA-05): SIDEWAYS nie stawia zakładu, więc nie wchodzi do mianownika.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.domain.trading_costs import net_return


def signal_of(predicted_trend: str) -> int:
    """Mapuje przewidziany trend na sygnał ekspozycji (+1 / -1 / 0).

    BULLISH→+1 (long), BEARISH→-1 (short), SIDEWAYS oraz cokolwiek nieznanego→0
    (brak ekspozycji). Porównanie jest niewrażliwe na wielkość liter — domena
    nie zakłada, w jakim case'ie przyjdzie etykieta z warstwy aplikacji.
    """
    normalized = predicted_trend.strip().upper()
    if normalized == "BULLISH":
        return 1
    if normalized == "BEARISH":
        return -1
    return 0


@dataclass(frozen=True)
class DirectionalHitRate:
    """Hit-rate kierunkowy: trafienia wśród predykcji BULLISH/BEARISH.

    `directional` to mianownik (predykcje z sygnałem != 0), `sideways` to
    predykcje bez zakładu (SIDEWAYS i nieznane etykiety) — liczone osobno,
    żeby raport mógł pokazać, ile ich było, zamiast po cichu je pominąć.
    """

    hits: int
    directional: int
    sideways: int

    @property
    def rate(self) -> float | None:
        """Odsetek trafień kierunkowych; None, gdy nie było żadnego zakładu."""
        if self.directional == 0:
            return None
        return self.hits / self.directional


def directional_hit_rate(calls: Iterable[tuple[str, bool]]) -> DirectionalHitRate:
    """Liczy hit-rate kierunkowy z par `(predicted_trend, is_correct)`.

    Kierunkowość rozstrzyga `signal_of`: sygnał 0 (SIDEWAYS, nieznany trend)
    nie wchodzi do mianownika, bo nie stawia zakładu na kierunek.
    """
    hits = 0
    directional = 0
    sideways = 0
    for trend, is_correct in calls:
        if signal_of(trend) == 0:
            sideways += 1
            continue
        directional += 1
        if is_correct:
            hits += 1
    return DirectionalHitRate(hits=hits, directional=directional, sideways=sideways)


@dataclass(frozen=True)
class EquityPoint:
    """Pojedynczy punkt krzywej kapitału po rozliczeniu jednego trade'u.

    `index` to pozycja w serii (0-based), `period_return` to zwrot tego okresu
    (sygnał * zmiana ceny), a `equity` to skumulowany kapitał po tym okresie.
    """

    index: int
    period_return: float
    equity: float


@dataclass(frozen=True)
class EquityCurve:
    """Zagregowana krzywa kapitału plus metryki track recordu.

    `points` to kolejne punkty krzywej, `total_return` to łączny zwrot względem
    kapitału startowego, `win_rate` to ułamek trafionych trade'ów KIERUNKOWYCH,
    `max_drawdown` to największe obsunięcie szczyt→dołek (dodatni ułamek), a
    `n_trades` to liczba wszystkich rozliczonych trade'ów.
    """

    points: tuple[EquityPoint, ...]
    total_return: float
    win_rate: float
    max_drawdown: float
    n_trades: int


def equity_curve(
    trades: Sequence[tuple[str, float]],
    starting_equity: float = 1.0,
    *,
    cost_bps: float = 0.0,
) -> EquityCurve:
    """Buduje krzywą kapitału ze składania kolejnych trade'ów.

    Każdy trade = `(predicted_trend, realized_change_pct)` (zmiana jako ułamek).
    Zwrot okresu = `signal_of(trend) * realized_change_pct` minus koszt zmiany
    pozycji względem poprzedniego trade'u (WIA-50, `cost_bps` jednostronnie,
    start z gotówki); kapitał składa się procentowo. Metryki:

    - `total_return` = final_equity / starting_equity - 1,
    - `win_rate` = ułamek trade'ów KIERUNKOWYCH (sygnał != 0) z dodatnim zwrotem;
      trade'y SIDEWAYS / o zerowym sygnale są wyłączone z mianownika (brak
      kierunkowych → 0.0, bez dzielenia przez zero),
    - `max_drawdown` = największy spadek szczyt→dołek serii kapitału jako DODATNI
      ułamek (monotoniczny wzrost → 0.0),
    - pusta historia → krzywa z samymi zerami.
    """
    period_returns: list[float] = []
    prev_signal = 0
    for trend, change_pct in trades:
        signal = signal_of(trend)
        period_returns.append(net_return(signal, prev_signal, signal * change_pct, cost_bps))
        prev_signal = signal
    return _compound(period_returns, trades, starting_equity)


@dataclass(frozen=True)
class PaperTrade:
    """Rozliczona predykcja jako trade paper-PnL (WIA-50).

    `change` to zrealizowana zmiana ceny w oknie predykcji (ułamek), a
    `cost_bps` to jednostronny koszt obrotu tego symbolu (np. wyższy dla krypto).
    Symbol służy wyłącznie do śledzenia pozycji między cyklami.
    """

    symbol: str
    predicted_trend: str
    change: float
    cost_bps: float = 0.0


def portfolio_equity_curve(
    cycles: Sequence[Sequence[tuple[str, float]]],
    starting_equity: float = 1.0,
) -> EquityCurve:
    """Krzywa kapitału portfela o równej wadze — jeden punkt na cykl (WIA-17).

    `cycles` to kolejne cykle w porządku chronologicznym, każdy z listą trade'ów
    `(predicted_trend, realized_change_pct)`. Zwrot cyklu to ŚREDNIA
    `signal_of(trend) * change` po wszystkich trade'ach cyklu: każda predykcja
    dostaje 1/N kapitału, a SIDEWAYS (sygnał 0) trzyma swoją część w gotówce.
    Puste cykle są pomijane. `win_rate`, `max_drawdown` i `n_trades` liczone
    są tymi samymi regułami co w `equity_curve` (n_trades = wszystkie trade'y,
    `points` = cykle).
    """
    # Bez kosztu obrót nie ma znaczenia, więc symbol trade'u też nie.
    return paper_portfolio_curve(
        [[PaperTrade("", trend, change) for trend, change in cycle] for cycle in cycles],
        starting_equity,
    )


def paper_portfolio_curve(
    cycles: Sequence[Sequence[PaperTrade]],
    starting_equity: float = 1.0,
) -> EquityCurve:
    """Krzywa portfela o równej wadze per cykl po kosztach obrotu (WIA-50).

    Reguły jak w `portfolio_equity_curve`, ale zwrot trade'u to
    `net_return(sygnał, poprzedni sygnał TEGO symbolu, brutto, trade.cost_bps)`:
    koszt płaci się tylko przy zmianie pozycji symbolu (pierwsza predykcja
    symbolu startuje z gotówki), a waga 1/N skaluje go jak zwrot. `cycles` idą
    chronologicznie, trade'y w cyklu — w kolejności czasu.
    """
    non_empty = [cycle for cycle in cycles if cycle]
    positions: dict[str, int] = {}
    period_returns: list[float] = []
    for cycle in non_empty:
        returns: list[float] = []
        for trade in cycle:
            signal = signal_of(trade.predicted_trend)
            prev_signal = positions.get(trade.symbol, 0)
            returns.append(
                net_return(signal, prev_signal, signal * trade.change, trade.cost_bps)
            )
            positions[trade.symbol] = signal
        period_returns.append(sum(returns) / len(returns))
    trades = [(trade.predicted_trend, trade.change) for cycle in non_empty for trade in cycle]
    return _compound(period_returns, trades, starting_equity)


def _compound(
    period_returns: Sequence[float],
    trades: Sequence[tuple[str, float]],
    starting_equity: float,
) -> EquityCurve:
    """Składa zwroty okresów w krzywą; win-rate liczy z pojedynczych trade'ów.

    Okres to trade (`equity_curve`) albo cykl (`portfolio_equity_curve`), więc
    punkty krzywej i trade'y win-rate'u to osobne sekwencje.
    """
    points: list[EquityPoint] = []
    equity = starting_equity
    peak = starting_equity
    max_drawdown = 0.0

    for index, period_return in enumerate(period_returns):
        equity = equity * (1.0 + period_return)
        points.append(
            EquityPoint(index=index, period_return=period_return, equity=equity)
        )

        # Obsunięcie liczone na bieżącym szczycie kapitału (peak-to-trough).
        if equity > peak:
            peak = equity
        elif peak > 0:
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    if not points:
        return EquityCurve(
            points=(),
            total_return=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            n_trades=0,
        )

    # Win-rate liczymy tylko z trade'ów kierunkowych (sygnał != 0) —
    # SIDEWAYS / nieznany trend nie stawia zakładu, więc nie zaniża trafień.
    directional_returns = [
        signal_of(trend) * change for trend, change in trades if signal_of(trend) != 0
    ]
    wins = sum(1 for trade_return in directional_returns if trade_return > 0)
    total_return = equity / starting_equity - 1.0 if starting_equity != 0 else 0.0
    win_rate = wins / len(directional_returns) if directional_returns else 0.0

    return EquityCurve(
        points=tuple(points),
        total_return=total_return,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        n_trades=len(trades),
    )
