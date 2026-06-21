"""Domena Equity Curve — krzywa kapitału z track recordu predykcji agenta.

Czysta domena (stdlib only). Bierze prymitywy, NIE `ResolvedPrediction` —
mapowanie z modelu raportu na pary `(trend, realized_change_pct)` robi
orkiestrator. Dzięki temu moduł jest odsprzężony od warstwy aplikacji.

Każdy trade to `(predicted_trend, realized_change_pct)`, gdzie zmiana ceny jest
ułamkiem (np. 0.03 = +3%). Sygnał kierunku: BULLISH→+1, BEARISH→-1,
SIDEWAYS/inne→0. Zwrot okresu = sygnał * zmiana ceny, więc trafiony short
(cena spadła) daje zysk, a chybiony long — stratę. Kapitał składa się
procentowo: equity_i = equity_{i-1} * (1 + period_return).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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
) -> EquityCurve:
    """Buduje krzywą kapitału ze składania kolejnych trade'ów.

    Każdy trade = `(predicted_trend, realized_change_pct)` (zmiana jako ułamek).
    Zwrot okresu = `signal_of(trend) * realized_change_pct`; kapitał składa się
    procentowo. Metryki:

    - `total_return` = final_equity / starting_equity - 1,
    - `win_rate` = ułamek trade'ów KIERUNKOWYCH (sygnał != 0) z dodatnim zwrotem;
      trade'y SIDEWAYS / o zerowym sygnale są wyłączone z mianownika (brak
      kierunkowych → 0.0, bez dzielenia przez zero),
    - `max_drawdown` = największy spadek szczyt→dołek serii kapitału jako DODATNI
      ułamek (monotoniczny wzrost → 0.0),
    - pusta historia → krzywa z samymi zerami.
    """
    points: list[EquityPoint] = []
    equity = starting_equity
    peak = starting_equity
    max_drawdown = 0.0
    directional_count = 0
    wins = 0

    for index, (trend, change_pct) in enumerate(trades):
        signal = signal_of(trend)
        period_return = signal * change_pct
        equity = equity * (1.0 + period_return)
        points.append(
            EquityPoint(index=index, period_return=period_return, equity=equity)
        )

        # Win-rate liczymy tylko z trade'ów kierunkowych (sygnał != 0) —
        # SIDEWAYS / nieznany trend nie stawia zakładu, więc nie zaniża trafień.
        if signal != 0:
            directional_count += 1
            if period_return > 0:
                wins += 1

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

    total_return = equity / starting_equity - 1.0 if starting_equity != 0 else 0.0
    win_rate = wins / directional_count if directional_count > 0 else 0.0

    return EquityCurve(
        points=tuple(points),
        total_return=total_return,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        n_trades=len(points),
    )
