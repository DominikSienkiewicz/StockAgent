"""Domena kosztów transakcyjnych paper-PnL i backtestu (WIA-50).

Czysta domena (stdlib only). Koszt `cost_bps` jest JEDNOSTRONNY i naliczany od
obrotu, czyli wyłącznie przy zmianie pozycji symbolu — nie przy każdej
predykcji (to by go przeszacowało). Pozycja to sygnał kierunku: +1 long,
-1 short, 0 gotówka (SIDEWAYS oznacza wyjście z pozycji).

Obrót przy przejściu `prev_signal → signal` to `|signal - prev_signal|` w
jednostkach kapitału: wejście z gotówki i wyjście do gotówki = 1, odwrócenie
long↔short = 2, trzymanie = 0. Pełny round trip (wejście + wyjście) kosztuje
więc 2 × `cost_bps`.
"""

from __future__ import annotations

BPS_PER_UNIT = 10_000.0


def turnover(signal: int, prev_signal: int) -> int:
    """Obrócony kapitał (0, 1 albo 2) przy przejściu z `prev_signal` na `signal`."""
    return abs(signal - prev_signal)


def trading_cost(traded: float, cost_bps: float) -> float:
    """Koszt obrotu `traded` (ułamek kapitału) przy koszcie jednostronnym `cost_bps`."""
    return traded * cost_bps / BPS_PER_UNIT


def net_return(signal: int, prev_signal: int, gross: float, cost_bps: float) -> float:
    """Zwrot okresu po koszcie zmiany pozycji `prev_signal → signal`.

    `gross` to zwrot brutto okresu (sygnał × zmiana ceny). Trzymanie tej samej
    pozycji i pozostawanie w gotówce są darmowe.
    """
    return gross - trading_cost(turnover(signal, prev_signal), cost_bps)
