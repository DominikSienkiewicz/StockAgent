from __future__ import annotations

import json
import urllib.parse
from decimal import Decimal

from src.application.report_formatting import delta_color
from src.application.report_models import SymbolResult

# Kolor siatki wykresów QuickChart — wspólny dla wszystkich osi/wykresów.
_GRID_COLOR = "rgba(0,0,0,0.06)"


def build_chart_url(results: list[SymbolResult]) -> str | None:
    """Generuje URL do QuickChart.io z bar-chartem zmiany ceny per symbol."""
    valid = [r for r in results if r.delta is not None and r.status != "error"]
    if not valid:
        return None

    valid_sorted = sorted(valid, key=lambda r: abs(r.delta or Decimal("0")), reverse=True)
    labels = [r.symbol for r in valid_sorted]
    deltas_pct = [float((r.delta or Decimal("0")) * Decimal("100")) for r in valid_sorted]
    colors = [delta_color(r.delta) for r in valid_sorted]

    config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Zmiana 12h (%)",
                "data": deltas_pct,
                "backgroundColor": colors,
                "borderWidth": 0,
            }],
        },
        "options": {
            "plugins": {
                "legend": {"display": False},
                "title": {
                    "display": True,
                    "text": "Zmiana ceny w ciągu 12h",
                    "font": {"size": 14, "weight": "bold"},
                },
            },
            "scales": {
                "y": {
                    "title": {"display": True, "text": "Zmiana (%)"},
                    "grid": {"color": _GRID_COLOR},
                },
                "x": {"grid": {"display": False}},
            },
        },
    }

    encoded = urllib.parse.quote(json.dumps(config, separators=(",", ":")))
    return f"https://quickchart.io/chart?c={encoded}&w=720&h=320&bkg=white&v=4"


def build_correlation_chart_url(results: list[SymbolResult]) -> str | None:
    """Scatter plot: sentyment (oś X) vs Δ12h (oś Y)."""
    points = [
        r for r in results
        if r.sentiment_score is not None and r.delta is not None and r.status != "error"
    ]
    if len(points) < 3:
        return None

    data_points = [
        {
            "x": float(r.sentiment_score or 0),
            "y": float((r.delta or Decimal("0")) * Decimal("100")),
        }
        for r in points
    ]
    colors = [delta_color(r.delta) for r in points]
    labels = [r.symbol for r in points]

    config = {
        "type": "scatter",
        "data": {
            "datasets": [{
                "label": "Symbole",
                "data": data_points,
                "backgroundColor": colors,
                "pointRadius": 7,
                "pointHoverRadius": 9,
            }],
        },
        "options": {
            "plugins": {
                "legend": {"display": False},
                "title": {
                    "display": True,
                    "text": "Korelacja sentymentu z ruchem ceny",
                    "font": {"size": 14, "weight": "bold"},
                },
                "datalabels": {
                    "align": "top",
                    "anchor": "end",
                    "font": {"size": 10},
                    "labels": {"title": {"color": "#374151"}},
                    "formatter": "function(value, ctx) { "
                    f"return {json.dumps(labels)}[ctx.dataIndex]; "
                    "}",
                },
            },
            "scales": {
                "x": {
                    "title": {"display": True, "text": "Sentyment AV (-1 ... +1)"},
                    "grid": {"color": _GRID_COLOR},
                },
                "y": {
                    "title": {"display": True, "text": "Zmiana 12h (%)"},
                    "grid": {"color": _GRID_COLOR},
                },
            },
        },
    }
    encoded = urllib.parse.quote(json.dumps(config, separators=(",", ":")))
    return f"https://quickchart.io/chart?c={encoded}&w=720&h=380&bkg=white&v=4"


def build_forecast_chart_url(results: list[SymbolResult]) -> str | None:
    """Wykres prognozowanych zmian (current -> target) dla symboli `saved`."""
    saved = [
        r for r in results
        if r.status == "saved" and r.expected_change is not None
    ]
    if not saved:
        return None

    saved_sorted = sorted(
        saved,
        key=lambda r: abs(r.expected_change or Decimal("0")),
        reverse=True,
    )
    labels = [r.symbol for r in saved_sorted]
    changes_pct = [
        float((r.expected_change or Decimal("0")) * Decimal("100")) for r in saved_sorted
    ]
    colors = [delta_color(r.expected_change) for r in saved_sorted]

    config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Prognozowana zmiana (%)",
                "data": changes_pct,
                "backgroundColor": colors,
                "borderWidth": 0,
            }],
        },
        "options": {
            "plugins": {
                "legend": {"display": False},
                "title": {
                    "display": True,
                    "text": "Prognoza ruchu cenowego w ciągu 12h",
                    "font": {"size": 14, "weight": "bold"},
                },
            },
            "scales": {
                "y": {
                    "title": {"display": True, "text": "Oczekiwana zmiana (%)"},
                    "grid": {"color": _GRID_COLOR},
                },
                "x": {"grid": {"display": False}},
            },
        },
    }

    encoded = urllib.parse.quote(json.dumps(config, separators=(",", ":")))
    return f"https://quickchart.io/chart?c={encoded}&w=720&h=320&bkg=white&v=4"
