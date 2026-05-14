from __future__ import annotations

from decimal import Decimal

_TREND_PL = {
    "BULLISH": "Wzrostowy",
    "BEARISH": "Spadkowy",
    "SIDEWAYS": "Boczny",
}

_SENTIMENT_PL = {
    "Bullish": "Pozytywny",
    "Somewhat-Bullish": "Lekko pozytywny",
    "Neutral": "Neutralny",
    "Somewhat-Bearish": "Lekko negatywny",
    "Bearish": "Negatywny",
}


def trend_label(trend: str | None) -> str:
    return _TREND_PL.get(trend or "", trend or "—")


def sentiment_label(label: str | None) -> str:
    return _SENTIMENT_PL.get(label or "", label or "—")


def pct(value: Decimal | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    if signed:
        return f"{value * 100:+.2f}%"
    return f"{value * 100:.2f}%"


def money(value: Decimal | None) -> str:
    if value is None:
        return "—"
    return f"${value:.2f}"


def trend_color(trend: str | None) -> str:
    return {
        "BULLISH": "#16a34a",
        "BEARISH": "#dc2626",
        "SIDEWAYS": "#737373",
    }.get(trend or "", "#737373")


def delta_color(delta: Decimal | None) -> str:
    if delta is None:
        return "#737373"
    if delta > 0:
        return "#16a34a"
    if delta < 0:
        return "#dc2626"
    return "#737373"


def score_to_pl_label(score: float) -> str:
    if score <= -0.35:
        return "Negatywny"
    if score <= -0.15:
        return "Lekko negatywny"
    if score < 0.15:
        return "Neutralny"
    if score < 0.35:
        return "Lekko pozytywny"
    return "Pozytywny"
