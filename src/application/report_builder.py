"""Buduje raport HTML+text z wyników cyklu Fast Loop.

Czysta logika prezentacji — bez I/O, łatwo testowalne.
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Polskie etykiety dla wartości z LLM (EN) i AV (EN)
# ---------------------------------------------------------------------------

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


def _trend_label(trend: str | None) -> str:
    return _TREND_PL.get(trend or "", trend or "—")


def _sentiment_label(label: str | None) -> str:
    return _SENTIMENT_PL.get(label or "", label or "—")


@dataclass(frozen=True)
class TopNewsItem:
    title: str
    source: str | None
    url: str | None
    relevance: float
    sentiment: float


@dataclass(frozen=True)
class SymbolResult:
    """Pojedynczy wynik analizy dla symbolu."""

    symbol: str
    status: str  # "saved" | "ignored" | "error"
    delta: Decimal | None = None
    current_price: Decimal | None = None
    trend: str | None = None
    target_price: Decimal | None = None
    confidence_score: float | None = None
    reasoning: str | None = None
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    news_volume: int | None = None
    av_llm_agreement: float | None = None       # 0.0-1.0 — zgoda LLM ↔ AV sentiment
    reflection_insight: str | None = None       # wniosek z poprzedniego cyklu
    top_news: list[TopNewsItem] = field(default_factory=list)
    error_message: str | None = None

    @property
    def expected_change(self) -> Decimal | None:
        """Procentowa zmiana z obecnej ceny do prognozowanej (target)."""
        if self.current_price is None or self.target_price is None:
            return None
        if self.current_price == 0:
            return None
        return (self.target_price - self.current_price) / self.current_price


def _pct(value: Decimal | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    if signed:
        return f"{value * 100:+.2f}%"
    return f"{value * 100:.2f}%"


def _money(value: Decimal | None) -> str:
    if value is None:
        return "—"
    return f"${value:.2f}"


def _trend_color(trend: str | None) -> str:
    return {
        "BULLISH": "#16a34a",   # green-600
        "BEARISH": "#dc2626",   # red-600
        "SIDEWAYS": "#737373",  # neutral-500
    }.get(trend or "", "#737373")


def _delta_color(delta: Decimal | None) -> str:
    if delta is None:
        return "#737373"
    if delta > 0:
        return "#16a34a"
    if delta < 0:
        return "#dc2626"
    return "#737373"


def build_chart_url(results: list[SymbolResult]) -> str | None:
    """Generuje URL do QuickChart.io z bar-chartem zmiany ceny per symbol.

    Klienty poczty (Gmail, Outlook, Apple Mail) renderują `<img src="https://...">`
    z QuickChart bez problemu. Bar chart sortowany malejąco po abs(delta),
    kolor: zielony (wzrost), czerwony (spadek), szary (zerowa zmiana).
    Zwraca None gdy brak danych do wykresu.
    """
    valid = [r for r in results if r.delta is not None and r.status != "error"]
    if not valid:
        return None

    valid_sorted = sorted(valid, key=lambda r: abs(r.delta or Decimal("0")), reverse=True)
    labels = [r.symbol for r in valid_sorted]
    deltas_pct = [float((r.delta or Decimal("0")) * Decimal("100")) for r in valid_sorted]
    colors = [_delta_color(r.delta) for r in valid_sorted]

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
                    "grid": {"color": "rgba(0,0,0,0.06)"},
                },
                "x": {"grid": {"display": False}},
            },
        },
    }

    encoded = urllib.parse.quote(json.dumps(config, separators=(",", ":")))
    return f"https://quickchart.io/chart?c={encoded}&w=720&h=320&bkg=white&v=4"


def build_correlation_chart_url(results: list[SymbolResult]) -> str | None:
    """Scatter plot: sentyment (oś X) vs Δ12h (oś Y).

    Pokazuje, czy AV-sentyment koreluje z ruchem ceny w tym cyklu.
    Wymaga ≥ 3 punktów, żeby miało sens wizualne.
    """
    points = [
        r for r in results
        if r.sentiment_score is not None and r.delta is not None and r.status != "error"
    ]
    if len(points) < 3:
        return None

    data_points = [
        {"x": float(r.sentiment_score or 0), "y": float((r.delta or Decimal("0")) * Decimal("100"))}
        for r in points
    ]
    colors = [_delta_color(r.delta) for r in points]
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
                    "labels": {
                        "title": {"color": "#374151"},
                    },
                    "formatter": "function(value, ctx) { "
                                 f"return {json.dumps(labels)}[ctx.dataIndex]; "
                                 "}",
                },
            },
            "scales": {
                "x": {
                    "title": {"display": True, "text": "Sentyment AV (-1 ... +1)"},
                    "grid": {"color": "rgba(0,0,0,0.06)"},
                },
                "y": {
                    "title": {"display": True, "text": "Zmiana 12h (%)"},
                    "grid": {"color": "rgba(0,0,0,0.06)"},
                },
            },
        },
    }
    encoded = urllib.parse.quote(json.dumps(config, separators=(",", ":")))
    return f"https://quickchart.io/chart?c={encoded}&w=720&h=380&bkg=white&v=4"


# ---------------------------------------------------------------------------
# Trade signals — top BUY/SELL po sile (confidence × |expected_change|)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeSignal:
    symbol: str
    direction: str   # "KUP" | "SPRZEDAJ" | "OBSERWUJ"
    confidence: float
    expected_change: Decimal
    strength: float  # confidence × |expected_change| × 100 (skala 0-100)
    current_price: Decimal | None
    target_price: Decimal | None


def _trend_to_direction(trend: str | None) -> str:
    return {
        "BULLISH": "KUP",
        "BEARISH": "SPRZEDAJ",
        "SIDEWAYS": "OBSERWUJ",
    }.get(trend or "", "OBSERWUJ")


def build_trade_signals(
    results: list[SymbolResult], top_n: int = 5
) -> list[TradeSignal]:
    """Top sygnały transakcyjne — posortowane po sile sygnału."""
    signals: list[TradeSignal] = []
    for r in results:
        if r.status != "saved":
            continue
        if r.confidence_score is None or r.expected_change is None:
            continue
        strength = r.confidence_score * float(abs(r.expected_change)) * 100
        signals.append(TradeSignal(
            symbol=r.symbol,
            direction=_trend_to_direction(r.trend),
            confidence=r.confidence_score,
            expected_change=r.expected_change,
            strength=strength,
            current_price=r.current_price,
            target_price=r.target_price,
        ))
    signals.sort(key=lambda s: s.strength, reverse=True)
    return signals[:top_n]


# ---------------------------------------------------------------------------
# Risk signals — wykrywanie anomalii / niespójności sygnałów
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskSignal:
    symbol: str
    type: str    # "DIVERGENCE" | "AV_LLM_CONFLICT" | "LOW_SIGNAL"
    severity: str  # "high" | "medium" | "low"
    description: str


DIVERGENCE_PRICE_THRESHOLD = Decimal("0.02")
DIVERGENCE_SENTIMENT_THRESHOLD = 0.2
AV_LLM_CONFLICT_THRESHOLD = 0.3
LOW_SIGNAL_NEWS_THRESHOLD = 3


def detect_risk_signals(results: list[SymbolResult]) -> list[RiskSignal]:
    """Flagi ostrzegawcze per symbol:

    1. **DIVERGENCE** — cena i sentyment idą w przeciwnych kierunkach
       (np. cena +3%, sentyment -0.3 → możliwa korekta / pump)
    2. **AV_LLM_CONFLICT** — LLM się nie zgadza z AV sentymentem
       (av_agreement < 0.3 → fake news?, manipulacja?)
    3. **LOW_SIGNAL** — predykcja oparta na zbyt małej liczbie newsów
    """
    out: list[RiskSignal] = []
    for r in results:
        if r.status != "saved":
            continue

        # Divergence
        if r.delta is not None and r.sentiment_score is not None:
            if (
                r.delta > DIVERGENCE_PRICE_THRESHOLD
                and r.sentiment_score < -DIVERGENCE_SENTIMENT_THRESHOLD
            ):
                out.append(RiskSignal(
                    symbol=r.symbol, type="DIVERGENCE", severity="high",
                    description=(
                        f"cena rośnie {_pct(r.delta)} ale sentyment "
                        f"{r.sentiment_score:+.2f} → możliwa korekta lub pump"
                    ),
                ))
            elif (
                r.delta < -DIVERGENCE_PRICE_THRESHOLD
                and r.sentiment_score > DIVERGENCE_SENTIMENT_THRESHOLD
            ):
                out.append(RiskSignal(
                    symbol=r.symbol, type="DIVERGENCE", severity="high",
                    description=(
                        f"cena spada {_pct(r.delta)} ale sentyment "
                        f"{r.sentiment_score:+.2f} → okazja lub odbicie?"
                    ),
                ))

        # AV-LLM conflict
        if (
            r.av_llm_agreement is not None
            and r.av_llm_agreement < AV_LLM_CONFLICT_THRESHOLD
        ):
            out.append(RiskSignal(
                symbol=r.symbol, type="AV_LLM_CONFLICT", severity="high",
                description=(
                    f"LLM nie zgadza się z AV (zgoda={r.av_llm_agreement:.2f})"
                    " → potencjalny fake news / manipulacja"
                ),
            ))

        # Low signal
        if (
            r.news_volume is not None
            and r.news_volume < LOW_SIGNAL_NEWS_THRESHOLD
        ):
            out.append(RiskSignal(
                symbol=r.symbol, type="LOW_SIGNAL", severity="medium",
                description=(
                    f"tylko {r.news_volume} relevantnych newsów — decyzja"
                    " na słabych danych"
                ),
            ))

    return out


# ---------------------------------------------------------------------------
# Day-over-day — porównanie z poprzednim cyklem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedPrediction:
    """Predykcja zamknięta w bieżącym cyklu (dostała accuracy_score)."""
    symbol: str
    predicted_trend: str
    accuracy_score: float
    is_correct: bool


def parse_resolved_predictions(
    rows: list[dict[str, Any]],
) -> list[ResolvedPrediction]:
    """Mapuje rekordy z `prediction_logs` na DTO `ResolvedPrediction`."""
    out: list[ResolvedPrediction] = []
    for row in rows:
        score = row.get("accuracy_score")
        if score is None:
            continue
        out.append(ResolvedPrediction(
            symbol=str(row.get("symbol", "?")),
            predicted_trend=str(row.get("predicted_trend", "?")),
            accuracy_score=float(score),
            is_correct=float(score) > 0.5,
        ))
    return out


# ---------------------------------------------------------------------------
# Portfolio mood — overview całego cyklu
# ---------------------------------------------------------------------------


def _score_to_pl_label(score: float) -> str:
    if score <= -0.35:
        return "Negatywny"
    if score <= -0.15:
        return "Lekko negatywny"
    if score < 0.15:
        return "Neutralny"
    if score < 0.35:
        return "Lekko pozytywny"
    return "Pozytywny"


def build_portfolio_mood(results: list[SymbolResult]) -> dict[str, Any]:
    """Liczy agregaty dla całego portfela: avg sentiment, top mover, etc."""
    saved = [r for r in results if r.status == "saved"]
    with_sentiment = [r for r in results if r.sentiment_score is not None]
    sentiments = [r.sentiment_score for r in with_sentiment if r.sentiment_score is not None]

    avg = sum(sentiments) / len(sentiments) if sentiments else 0.0
    most_bullish = max(with_sentiment, key=lambda r: r.sentiment_score or 0, default=None)
    most_bearish = min(with_sentiment, key=lambda r: r.sentiment_score or 0, default=None)
    high_conf = sum(
        1 for r in saved if r.confidence_score is not None and r.confidence_score >= 0.7
    )
    total_news = sum(r.news_volume or 0 for r in results)

    return {
        "avg_sentiment": avg,
        "avg_sentiment_label": _score_to_pl_label(avg),
        "most_bullish": most_bullish,
        "most_bearish": most_bearish,
        "high_confidence_count": high_conf,
        "saved_count": len(saved),
        "total_news": total_news,
    }


# ---------------------------------------------------------------------------
# Status sesji giełdowej (NYSE — uproszczone, bez świąt)
# ---------------------------------------------------------------------------


NYSE_TZ = ZoneInfo("America/New_York")
NYSE_OPEN = time(9, 30)
NYSE_CLOSE = time(16, 0)


def market_status(now_utc: datetime) -> dict[str, str]:
    """Status sesji NYSE (uproszczony — bez świąt). Zwraca:
    `{"label": "...", "detail": "..."}`."""
    nyc = now_utc.astimezone(NYSE_TZ)
    if nyc.weekday() >= 5:
        next_monday = nyc + timedelta(days=(7 - nyc.weekday()))
        return {
            "label": "🌙 Zamknięta (weekend)",
            "detail": f"Najbliższe otwarcie: {next_monday.strftime('%A %d.%m')} 09:30 ET",
        }
    cur = nyc.time()
    if NYSE_OPEN <= cur < NYSE_CLOSE:
        close_today = nyc.replace(hour=NYSE_CLOSE.hour, minute=NYSE_CLOSE.minute, second=0)
        delta = close_today - nyc
        return {
            "label": "🟢 Otwarta (regular hours)",
            "detail": f"Zamknięcie za {_humanize_delta(delta)}",
        }
    if cur < NYSE_OPEN:
        open_today = nyc.replace(hour=NYSE_OPEN.hour, minute=NYSE_OPEN.minute, second=0)
        delta = open_today - nyc
        return {
            "label": "🌅 Premarket",
            "detail": f"Otwarcie za {_humanize_delta(delta)}",
        }
    # after-hours
    return {
        "label": "🌆 After-hours",
        "detail": "Otwarcie jutro 09:30 ET",
    }


def _humanize_delta(delta: timedelta) -> str:
    total_min = max(0, int(delta.total_seconds() // 60))
    h, m = divmod(total_min, 60)
    if h > 0:
        return f"{h}h {m}min"
    return f"{m}min"


def build_forecast_chart_url(results: list[SymbolResult]) -> str | None:
    """Wykres prognozowanych zmian (current → target) dla symboli `saved`."""
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
    colors = [_delta_color(r.expected_change) for r in saved_sorted]

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
                    "grid": {"color": "rgba(0,0,0,0.06)"},
                },
                "x": {"grid": {"display": False}},
            },
        },
    }

    encoded = urllib.parse.quote(json.dumps(config, separators=(",", ":")))
    return f"https://quickchart.io/chart?c={encoded}&w=720&h=320&bkg=white&v=4"


def build_html_report(
    results: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    accuracy_stats: dict[str, Any] | None = None,
    resolved_predictions: list[ResolvedPrediction] | None = None,
) -> tuple[str, str]:
    """Zwraca (html_body, plain_text) — oba reprezentacje raportu.

    Parametry opcjonalne:
        accuracy_stats: wynik `RepositoryPort.get_accuracy_stats(days)`.
        resolved_predictions: zamknięte predykcje z ostatnich N godzin
            (do sekcji day-over-day).
    """
    saved = [r for r in results if r.status == "saved"]
    ignored = [r for r in results if r.status == "ignored"]
    errors = [r for r in results if r.status == "error"]
    mood = build_portfolio_mood(results)
    session = market_status(started_at)
    trade_signals = build_trade_signals(results)
    risk_signals = detect_risk_signals(results)

    html = _render_html(
        results, saved, ignored, errors, started_at, duration_seconds,
        mood, session, accuracy_stats, trade_signals, risk_signals,
        resolved_predictions or [],
    )
    text = _render_plain(
        results, saved, ignored, errors, started_at, duration_seconds,
        mood, session, accuracy_stats, trade_signals, risk_signals,
        resolved_predictions or [],
    )
    return html, text


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _render_html(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    mood: dict[str, Any],
    session: dict[str, str],
    accuracy_stats: dict[str, Any] | None,
    trade_signals: list[TradeSignal],
    risk_signals: list[RiskSignal],
    resolved_predictions: list[ResolvedPrediction],
) -> str:
    sections = []

    # Header
    sections.append(f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; max-width: 720px; margin: 0 auto; color: #1f2937;">
      <h1 style="font-size: 22px; margin: 0 0 4px 0;">📊 StockAgent — raport cyklu</h1>
      <div style="font-size: 13px; color: #6b7280; margin-bottom: 24px;">
        Czas rozpoczęcia: {started_at.strftime("%Y-%m-%d %H:%M:%S UTC")}
        · Czas trwania: {duration_seconds:.1f}s
        · Liczba symboli: {len(results)}
      </div>
    """)

    # Status sesji giełdowej
    sections.append(f"""
      <div style="padding: 10px 14px; background: #f9fafb; border-radius: 4px;
                  margin-bottom: 16px; font-size: 13px;">
        <strong>Sesja NYSE:</strong> {session["label"]}
        <span style="color: #6b7280;">· {session["detail"]}</span>
      </div>
    """)

    # Summary boxes
    sections.append(f"""
      <div style="display: flex; gap: 8px; margin-bottom: 16px;">
        <div style="flex: 1; padding: 12px; background: #f0fdf4; border-left: 3px solid #16a34a; border-radius: 4px;">
          <div style="font-size: 11px; color: #15803d; text-transform: uppercase;">Predykcje</div>
          <div style="font-size: 24px; font-weight: 600; color: #14532d;">{len(saved)}</div>
        </div>
        <div style="flex: 1; padding: 12px; background: #f9fafb; border-left: 3px solid #9ca3af; border-radius: 4px;">
          <div style="font-size: 11px; color: #4b5563; text-transform: uppercase;">Pominięte</div>
          <div style="font-size: 24px; font-weight: 600; color: #1f2937;">{len(ignored)}</div>
        </div>
        <div style="flex: 1; padding: 12px; background: {'#fef2f2' if errors else '#f9fafb'}; border-left: 3px solid {'#dc2626' if errors else '#9ca3af'}; border-radius: 4px;">
          <div style="font-size: 11px; color: {'#991b1b' if errors else '#4b5563'}; text-transform: uppercase;">Błędy</div>
          <div style="font-size: 24px; font-weight: 600; color: {'#7f1d1d' if errors else '#1f2937'};">{len(errors)}</div>
        </div>
      </div>
    """)

    # 🎯 Trade ideas (najsilniejsze sygnały transakcyjne)
    if trade_signals:
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
            "🎯 Najsilniejsze sygnały</h2>"
        )
        for sig in trade_signals:
            dir_color = {
                "KUP": "#16a34a",
                "SPRZEDAJ": "#dc2626",
                "OBSERWUJ": "#737373",
            }[sig.direction]
            dir_bg = {
                "KUP": "#f0fdf4",
                "SPRZEDAJ": "#fef2f2",
                "OBSERWUJ": "#f9fafb",
            }[sig.direction]
            change_color = _delta_color(sig.expected_change)
            sections.append(f"""
              <div style="margin-bottom: 8px; padding: 10px 14px; background: {dir_bg};
                          border-left: 4px solid {dir_color}; border-radius: 4px;
                          font-size: 13px; display: flex; gap: 12px; align-items: center;">
                <span style="font-weight: 700; color: {dir_color}; min-width: 90px;">
                  {sig.direction}
                </span>
                <span style="font-weight: 600; min-width: 60px;">{sig.symbol}</span>
                <span style="color: #4b5563;">
                  pewność <strong>{sig.confidence * 100:.0f}%</strong>
                </span>
                <span style="color: {change_color};">
                  prognoza <strong>{_pct(sig.expected_change, signed=True)}</strong>
                </span>
                <span style="color: #6b7280; margin-left: auto; font-size: 12px;">
                  siła sygnału: <strong>{sig.strength:.2f}</strong>
                </span>
              </div>
            """)

    # 🚨 Risk signals (anomalie / niespójności)
    if risk_signals:
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0; color: #991b1b;'>"
            "🚨 Sygnały ostrzegawcze</h2>"
        )
        type_label = {
            "DIVERGENCE": "Rozbieżność cena ↔ sentyment",
            "AV_LLM_CONFLICT": "Niezgodność AV ↔ LLM",
            "LOW_SIGNAL": "Słaby sygnał",
        }
        for rs in risk_signals:
            sev_color = {"high": "#dc2626", "medium": "#f59e0b", "low": "#6b7280"}[rs.severity]
            sections.append(f"""
              <div style="margin-bottom: 6px; padding: 8px 12px; background: #fffbeb;
                          border-left: 3px solid {sev_color}; border-radius: 4px;
                          font-size: 12px;">
                <strong>{rs.symbol}</strong>
                <span style="color: {sev_color}; font-weight: 600;">
                  · {type_label.get(rs.type, rs.type)}
                </span><br/>
                <span style="color: #4b5563;">{rs.description}</span>
              </div>
            """)

    # Portfolio mood box
    bullish_part = (
        f"<strong>{mood['most_bullish'].symbol}</strong> "
        f"({mood['most_bullish'].sentiment_score:+.2f})"
        if mood["most_bullish"] else "—"
    )
    bearish_part = (
        f"<strong>{mood['most_bearish'].symbol}</strong> "
        f"({mood['most_bearish'].sentiment_score:+.2f})"
        if mood["most_bearish"] else "—"
    )
    sections.append(f"""
      <div style="padding: 12px 14px; background: #eff6ff; border-left: 3px solid #2563eb;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #1e3a8a;">
          📊 Nastroje portfela
        </div>
        <div style="color: #1e40af; line-height: 1.7;">
          Średni sentyment: <strong>{mood['avg_sentiment']:+.2f}</strong>
          ({mood['avg_sentiment_label']})<br/>
          Najbardziej pozytywny: {bullish_part} · Najbardziej negatywny: {bearish_part}<br/>
          Predykcje z wysoką pewnością (≥70%): {mood['high_confidence_count']} / {mood['saved_count']}
          · Łącznie newsów: {mood['total_news']}
        </div>
      </div>
    """)

    # 📊 Day-over-day — zamknięte predykcje z ostatnich 24h
    if resolved_predictions:
        correct = [p for p in resolved_predictions if p.is_correct]
        wrong = [p for p in resolved_predictions if not p.is_correct]
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
            "📊 Zamknięte predykcje (ostatnie 24h)</h2>"
        )
        for p in resolved_predictions:
            mark = "✅" if p.is_correct else "❌"
            color = "#16a34a" if p.is_correct else "#dc2626"
            sections.append(f"""
              <div style="margin-bottom: 4px; padding: 6px 10px; background: #fafafa;
                          border-left: 3px solid {color}; border-radius: 4px;
                          font-size: 12px;">
                {mark} <strong>{p.symbol}</strong>
                <span style="color: #6b7280;">·
                  prognoza {_trend_label(p.predicted_trend)} ·
                  trafność <strong style="color: {color};">
                    {p.accuracy_score * 100:.0f}%
                  </strong>
                </span>
              </div>
            """)
        sections.append(
            "<div style='font-size: 11px; color: #6b7280; margin: 4px 0 16px 0;'>"
            f"Suma: {len(correct)} trafionych / {len(wrong)} błędnych "
            f"({len(correct) / max(1, len(resolved_predictions)) * 100:.0f}% accuracy)"
            "</div>"
        )

    # Historia trafności
    if accuracy_stats and accuracy_stats.get("mean_accuracy") is not None:
        acc = accuracy_stats["mean_accuracy"]
        n = accuracy_stats["sample_count"]
        correct = accuracy_stats["correct_count"]
        days = accuracy_stats["days_window"]
        sections.append(f"""
      <div style="padding: 12px 14px; background: #fef3c7; border-left: 3px solid #f59e0b;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #92400e;">
          🎯 Historia trafności (ostatnie {days} dni)
        </div>
        <div style="color: #78350f;">
          Średnia trafność: <strong>{acc * 100:.1f}%</strong>
          · Predykcji ocenionych: {n}
          · Poprawnych (accuracy > 0.5): {correct}
        </div>
      </div>
        """)
    elif accuracy_stats is not None:
        sections.append("""
      <div style="padding: 10px 14px; background: #f9fafb; border-radius: 4px;
                  margin-bottom: 20px; font-size: 12px; color: #6b7280;">
        🎯 Historia trafności: brak ocenionych predykcji (potrzeba ≥1 zamkniętego cyklu).
      </div>
        """)

    # Wykres zmiany cen
    chart_url = build_chart_url(results)
    if chart_url:
        sections.append(f"""
      <div style="margin-bottom: 16px; text-align: center;">
        <img src="{chart_url}" alt="Wykres zmiany cen 12h"
             style="max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;" />
      </div>
        """)

    # Self-Reflection — wnioski z poprzednich cykli
    reflections = [r for r in saved if r.reflection_insight]
    if reflections:
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
            "🧠 Wnioski z poprzednich cykli (Self-Reflection)</h2>"
        )
        for r in reflections:
            sections.append(f"""
              <div style="margin-bottom: 8px; padding: 10px 12px; background: #faf5ff;
                          border-left: 3px solid #9333ea; border-radius: 4px;
                          font-size: 12px;">
                <strong>{r.symbol}:</strong>
                <span style="color: #581c87;">{r.reflection_insight}</span>
              </div>
            """)

    # Predykcje (saved)
    if saved:
        sections.append("<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>🔮 Wygenerowane predykcje</h2>")
        sections.append("<table style='width: 100%; border-collapse: collapse; font-size: 13px;'>")
        sections.append("""
          <tr style="background: #f3f4f6; text-align: left;">
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Symbol</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Cena</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Zmiana 12h</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Trend</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Prognoza (12h)</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Pewność</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Sentyment</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Newsy</th>
          </tr>
        """)
        for r in saved:
            sentiment_text = (
                f"{_sentiment_label(r.sentiment_label)} ({r.sentiment_score:+.2f})"
                if r.sentiment_score is not None
                else "—"
            )
            confidence_text = (
                f"{r.confidence_score * 100:.0f}%"
                if r.confidence_score is not None
                else "—"
            )
            forecast_text = (
                f"{_money(r.target_price)} "
                f"<span style='color: {_delta_color(r.expected_change)};'>"
                f"({_pct(r.expected_change, signed=True)})</span>"
                if r.target_price is not None
                else "—"
            )
            sections.append(f"""
              <tr>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; font-weight: 600;">{r.symbol}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{_money(r.current_price)}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; color: {_delta_color(r.delta)};">{_pct(r.delta)}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; color: {_trend_color(r.trend)}; font-weight: 600;">{_trend_label(r.trend)}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{forecast_text}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{confidence_text}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{sentiment_text}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{r.news_volume or 0}</td>
              </tr>
            """)
        sections.append("</table>")

        # Wykres prognozy
        forecast_chart_url = build_forecast_chart_url(saved)
        if forecast_chart_url:
            sections.append(f"""
      <div style="margin: 16px 0; text-align: center;">
        <img src="{forecast_chart_url}" alt="Prognoza zmian cen"
             style="max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;" />
      </div>
            """)

        # Uzasadnienia
        sections.append("<h3 style='font-size: 14px; margin: 16px 0 8px 0;'>💡 Uzasadnienia</h3>")
        for r in saved:
            if not r.reasoning:
                continue
            move_line = ""
            if r.current_price is not None and r.target_price is not None:
                conf_part = (
                    f" · Pewność {r.confidence_score * 100:.0f}%"
                    if r.confidence_score is not None
                    else ""
                )
                move_line = (
                    f"<div style='font-size: 11px; color: #6b7280; margin-top: 2px;'>"
                    f"Oczekiwany ruch: {_money(r.current_price)} → {_money(r.target_price)} "
                    f"<span style='color: {_delta_color(r.expected_change)};'>"
                    f"({_pct(r.expected_change, signed=True)})</span>"
                    f"{conf_part}"
                    f"</div>"
                )
            news_block = ""
            if r.top_news:
                items = []
                for n in r.top_news:
                    src = f"<strong>[{n.source or 'unknown'}]</strong>"
                    if n.url:
                        title_html = (
                            f"<a href='{n.url}' target='_blank' "
                            f"style='color: #2563eb; text-decoration: none;'>"
                            f"{n.title}</a>"
                        )
                    else:
                        title_html = n.title
                    items.append(
                        f"<li style='margin: 2px 0;'>{src} {title_html}"
                        f" <span style='color: #6b7280;'>"
                        f"(relevance {n.relevance:.2f}, sentyment {n.sentiment:+.2f})"
                        f"</span></li>"
                    )
                news_block = (
                    "<div style='font-size: 11px; color: #4b5563; margin-top: 8px;'>"
                    "📰 <strong>Top newsy:</strong>"
                    f"<ul style='margin: 4px 0 0 20px; padding: 0;'>{''.join(items)}</ul>"
                    "</div>"
                )
            sections.append(f"""
              <div style="margin-bottom: 10px; padding: 10px 12px; background: #fafafa; border-left: 3px solid {_trend_color(r.trend)}; border-radius: 4px;">
                <div style="font-weight: 600; font-size: 13px;">{r.symbol} <span style="color: {_trend_color(r.trend)};">{_trend_label(r.trend)}</span></div>
                {move_line}
                <div style="font-size: 12px; color: #4b5563; margin-top: 6px;">{r.reasoning}</div>
                {news_block}
              </div>
            """)

    # Scatter: sentyment vs cena (jeśli ≥3 punkty)
    scatter_url = build_correlation_chart_url(results)
    if scatter_url:
        sections.append(f"""
      <h3 style="font-size: 14px; margin: 20px 0 8px 0;">📈 Korelacja sentymentu z ruchem ceny</h3>
      <div style="margin-bottom: 16px; text-align: center;">
        <img src="{scatter_url}" alt="Scatter sentyment vs cena"
             style="max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;" />
      </div>
        """)

    # Pominięte
    if ignored:
        sections.append("<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>⏸ Pominięte (poniżej progu zmienności)</h2>")
        rows = "".join(
            f"<span style='display: inline-block; padding: 2px 8px; margin: 2px; background: #f3f4f6; border-radius: 3px; font-size: 12px;'>"
            f"<strong>{r.symbol}</strong> <span style='color: {_delta_color(r.delta)}'>{_pct(r.delta)}</span></span>"
            for r in ignored
        )
        sections.append(f"<div>{rows}</div>")

    # Błędy
    if errors:
        sections.append("<h2 style='font-size: 16px; margin: 20px 0 8px 0; color: #991b1b;'>⚠️ Błędy</h2>")
        for r in errors:
            sections.append(f"""
              <div style="margin-bottom: 8px; padding: 10px; background: #fef2f2; border-left: 3px solid #dc2626; border-radius: 4px; font-size: 12px;">
                <strong>{r.symbol}</strong>: {r.error_message}
              </div>
            """)

    sections.append("""
      <div style="margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e7eb; font-size: 11px; color: #9ca3af;">
        StockAgent · Agentic RAG + Continual Learning · raport automatyczny
      </div>
    </div>
    """)

    return "".join(sections)


# ---------------------------------------------------------------------------
# Plain text fallback
# ---------------------------------------------------------------------------


def _render_plain(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    mood: dict[str, Any],
    session: dict[str, str],
    accuracy_stats: dict[str, Any] | None,
    trade_signals: list[TradeSignal],
    risk_signals: list[RiskSignal],
    resolved_predictions: list[ResolvedPrediction],
) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("STOCKAGENT — RAPORT CYKLU")
    lines.append("=" * 64)
    lines.append(f"Czas rozpoczęcia:  {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Czas trwania:      {duration_seconds:.1f}s")
    lines.append(f"Sesja NYSE:        {session['label']} — {session['detail']}")
    lines.append(
        f"Symboli:           {len(results)} "
        f"(predykcji={len(saved)}, pominiętych={len(ignored)}, błędów={len(errors)})"
    )
    lines.append("")

    # Trade ideas
    if trade_signals:
        lines.append("NAJSILNIEJSZE SYGNAŁY")
        lines.append("-" * 64)
        for sig in trade_signals:
            lines.append(
                f"  {sig.direction:9s} {sig.symbol:6s} "
                f"pewność {sig.confidence * 100:.0f}%  "
                f"prognoza {_pct(sig.expected_change, signed=True):>8s}  "
                f"siła {sig.strength:.2f}"
            )
        lines.append("")

    # Risk signals
    if risk_signals:
        lines.append("SYGNAŁY OSTRZEGAWCZE")
        lines.append("-" * 64)
        type_label = {
            "DIVERGENCE": "Rozbieżność",
            "AV_LLM_CONFLICT": "Niezgodność AV↔LLM",
            "LOW_SIGNAL": "Słaby sygnał",
        }
        for rs in risk_signals:
            lines.append(
                f"  ⚠ {rs.symbol}: [{type_label.get(rs.type, rs.type)}] {rs.description}"
            )
        lines.append("")

    # Zamknięte predykcje
    if resolved_predictions:
        correct = sum(1 for p in resolved_predictions if p.is_correct)
        lines.append("ZAMKNIĘTE PREDYKCJE (ostatnie 24h)")
        lines.append("-" * 64)
        for p in resolved_predictions:
            mark = "✅" if p.is_correct else "❌"
            lines.append(
                f"  {mark} {p.symbol:6s}  trend {_trend_label(p.predicted_trend):11s} "
                f"trafność {p.accuracy_score * 100:.0f}%"
            )
        lines.append(
            f"  Suma: {correct}/{len(resolved_predictions)} "
            f"({correct / max(1, len(resolved_predictions)) * 100:.0f}% accuracy)"
        )
        lines.append("")

    # Nastroje portfela
    lines.append("NASTROJE PORTFELA")
    lines.append("-" * 64)
    lines.append(
        f"  Średni sentyment: {mood['avg_sentiment']:+.2f} ({mood['avg_sentiment_label']})"
    )
    if mood["most_bullish"]:
        mb = mood["most_bullish"]
        lines.append(f"  Najbardziej pozytywny: {mb.symbol} ({mb.sentiment_score:+.2f})")
    if mood["most_bearish"]:
        mbe = mood["most_bearish"]
        lines.append(f"  Najbardziej negatywny: {mbe.symbol} ({mbe.sentiment_score:+.2f})")
    lines.append(
        f"  Wysoka pewność (≥70%): {mood['high_confidence_count']} / {mood['saved_count']}"
        f"  ·  Newsów: {mood['total_news']}"
    )
    lines.append("")

    if accuracy_stats and accuracy_stats.get("mean_accuracy") is not None:
        lines.append(f"HISTORIA TRAFNOŚCI (ostatnie {accuracy_stats['days_window']} dni)")
        lines.append("-" * 64)
        lines.append(
            f"  Średnia trafność: {accuracy_stats['mean_accuracy'] * 100:.1f}%"
            f"  ·  Predykcji ocenionych: {accuracy_stats['sample_count']}"
            f"  ·  Poprawnych: {accuracy_stats['correct_count']}"
        )
        lines.append("")

    # Wnioski z poprzednich cykli
    reflections = [r for r in saved if r.reflection_insight]
    if reflections:
        lines.append("WNIOSKI Z POPRZEDNICH CYKLI (Self-Reflection)")
        lines.append("-" * 64)
        for r in reflections:
            lines.append(f"  {r.symbol}: {r.reflection_insight}")
        lines.append("")

    if saved:
        lines.append("PREDYKCJE")
        lines.append("-" * 64)
        for r in saved:
            conf_part = (
                f" · pewność {r.confidence_score * 100:.0f}%"
                if r.confidence_score is not None
                else ""
            )
            forecast_part = (
                f"prognoza {_money(r.target_price)} "
                f"({_pct(r.expected_change, signed=True)})"
                if r.target_price is not None and r.expected_change is not None
                else "prognoza —"
            )
            sentiment_part = (
                f"sentyment {r.sentiment_score:+.2f}"
                if r.sentiment_score is not None
                else "sentyment —"
            )
            lines.append(
                f"  {r.symbol:6s} {_money(r.current_price):>10s}  "
                f"Δ12h {_pct(r.delta, signed=True):>8s}  →  "
                f"{_trend_label(r.trend):11s} {forecast_part}{conf_part}"
            )
            lines.append(f"         {sentiment_part}")
            if r.reasoning:
                lines.append(f"        └ {r.reasoning}")
            for n in r.top_news:
                lines.append(
                    f"        📰 [{n.source or '?'}] {n.title[:78]}"
                    f" (rel={n.relevance:.2f}, sent={n.sentiment:+.2f})"
                )
        lines.append("")

    if ignored:
        lines.append("POMINIĘTE (poniżej progu zmienności)")
        lines.append("-" * 64)
        ignored_str = ", ".join(f"{r.symbol}({_pct(r.delta)})" for r in ignored)
        lines.append(f"  {ignored_str}")
        lines.append("")

    if errors:
        lines.append("BŁĘDY")
        lines.append("-" * 64)
        for r in errors:
            lines.append(f"  {r.symbol}: {r.error_message}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: convert use case output → SymbolResult
# ---------------------------------------------------------------------------


def to_symbol_result(symbol: str, raw: dict[str, Any] | None, error: str | None = None) -> SymbolResult:
    """Mapper z wyjścia AnalyzeMarketUseCase.run() na SymbolResult."""
    if error is not None:
        return SymbolResult(symbol=symbol, status="error", error_message=error)
    if raw is None:
        return SymbolResult(symbol=symbol, status="error", error_message="No result")

    status = str(raw.get("status", "unknown"))
    sentiment = raw.get("sentiment") or {}
    llm = raw.get("llm_analysis") or {}
    confidence = llm.get("confidence_score")
    av_agreement = llm.get("av_agreement")

    return SymbolResult(
        symbol=symbol,
        status=status,
        delta=_as_decimal(raw.get("delta")),
        current_price=_as_decimal(raw.get("current_price")),
        trend=llm.get("trend_direction"),
        target_price=_as_decimal(raw.get("ml_target_price")),
        confidence_score=float(confidence) if confidence is not None else None,
        reasoning=llm.get("reasoning"),
        sentiment_score=sentiment.get("av_sentiment_score"),
        sentiment_label=sentiment.get("av_sentiment_label"),
        news_volume=sentiment.get("news_volume_24h"),
        av_llm_agreement=float(av_agreement) if av_agreement is not None else None,
        reflection_insight=_clean_reflection(raw.get("reflection_context")),
        top_news=_extract_top_news(raw.get("news") or []),
    )


def _clean_reflection(raw_ctx: Any) -> str | None:
    """Filtruje cold-start placeholdery — pokazujemy reflection tylko gdy
    jest realny wniosek (LLM analyze_mistake lub potwierdzenie trafności)."""
    if not raw_ctx or not isinstance(raw_ctx, str):
        return None
    txt = raw_ctx.strip()
    if not txt:
        return None
    # Skip cold-start placeholder messages
    if "brak danych historycznych" in txt.lower():
        return None
    return txt


def _extract_top_news(news_items: list[dict[str, Any]], top_n: int = 2) -> list[TopNewsItem]:
    """Top N artykułów po relevance_score (z bonusem dla high-sentiment)."""
    valid = [
        item for item in news_items
        if item.get("relevance_score") is not None and item.get("title")
    ]
    sorted_news = sorted(valid, key=lambda x: float(x["relevance_score"] or 0), reverse=True)
    return [
        TopNewsItem(
            title=str(item["title"]),
            source=item.get("source"),
            url=item.get("url"),
            relevance=float(item.get("relevance_score") or 0),
            sentiment=float(item.get("ticker_sentiment_score") or 0),
        )
        for item in sorted_news[:top_n]
    ]


def _as_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None
