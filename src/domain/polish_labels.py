"""Wspólny słownik polskich etykiet sygnałów (MAIL-27, MAIL-15).

Jedno źródło prawdy dla etykiet, które trafiają do maila, leadu i tematu:
trend modelu jako sygnał akcji (KUP / SPRZEDAJ / WSTRZYMAJ), nazwa trendu
(Wzrostowy / Spadkowy / Boczny), prognoza w zdaniu leadu („trafiona prognoza
wzrostu”) i werdykt rady (KUP / SPRZEDAJ / TRZYMAJ). Domena (`digest_lead`)
i renderery warstwy application biorą etykiety stąd — moduł nie zna nic spoza
domeny.

Etykieta to „sygnał modelu”, nie rekomendacja: mail niesie zastrzeżenie, że
treść nie stanowi rekomendacji inwestycyjnej (MAIL-08), więc nagłówek nie może
mówić inaczej. Angielskie wartości (BULLISH, BUY…) zostają w logach i w bazie.
"""

from __future__ import annotations

MODEL_SIGNAL_TITLE = "Sygnał modelu"

# Brak / nieznany trend → WSTRZYMAJ: brak kierunku to brak akcji, nie „KUP”.
_HOLD_SIGNAL = "WSTRZYMAJ"

_MODEL_SIGNAL_PL: dict[str, str] = {
    "BULLISH": "KUP",
    "BEARISH": "SPRZEDAJ",
    "SIDEWAYS": _HOLD_SIGNAL,
}

_TREND_PL: dict[str, str] = {
    "BULLISH": "Wzrostowy",
    "BEARISH": "Spadkowy",
    "SIDEWAYS": "Boczny",
}

# Dopełniacz do zdania leadu: „trafiona prognoza wzrostu”.
_TREND_FORECAST_PL: dict[str, str] = {
    "BULLISH": "wzrostu",
    "BEARISH": "spadku",
    "SIDEWAYS": "ruchu bocznego",
}

_COUNCIL_RECOMMENDATION_PL: dict[str, str] = {
    "BUY": "KUP",
    "SELL": "SPRZEDAJ",
    "HOLD": "TRZYMAJ",
}


def model_signal_label(trend: str | None) -> str:
    """Trend modelu (BULLISH/BEARISH/SIDEWAYS) jako polski sygnał akcji."""
    return _MODEL_SIGNAL_PL.get(trend or "", _HOLD_SIGNAL)


def trend_label(trend: str | None) -> str:
    """Nazwa trendu po polsku; nieznany trend przechodzi bez zmian, brak → „—”."""
    return _TREND_PL.get(trend or "", trend or "—")


def trend_forecast_label(trend: str) -> str:
    """„prognoza wzrostu” / „prognoza spadku” / „prognoza ruchu bocznego”.

    Nieznana wartość zostaje widoczna („prognoza FLAT”), zamiast udawać kierunek."""
    return f"prognoza {_TREND_FORECAST_PL.get(trend, trend)}"


def council_recommendation_label(recommendation: str) -> str:
    """Werdykt rady (BUY/SELL/HOLD) po polsku; nieznana wartość przechodzi bez
    zmian, żeby nowy werdykt był widoczny w mailu zamiast zniknąć."""
    return _COUNCIL_RECOMMENDATION_PL.get(recommendation, recommendation)
