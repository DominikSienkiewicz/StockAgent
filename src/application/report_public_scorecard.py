"""Publiczny scorecard kalibracji jako samodzielna strona HTML (roadmap #10).

Cały track record żyje w prywatnym mailu; `web_digest_enabled` publikowałby
PEŁNY prywatny digest, który nie nadaje się do upublicznienia. Ten moduł
produkuje osobny, PUBLICZNY artefakt: kompletną, samodzielną stronę HTML
zawierającą WYŁĄCZNIE agregaty — liczbę predykcji, hit-rate, ECE i krzywą
kalibracji jako div-sparkline.

Bezwzględne inwarianty prywatności:
  - ZERO PII (adresy e-mail, dane odbiorców),
  - ZERO symboli watchlisty,
  - ZERO kwot (ceny, waluty).

`render_public_scorecard_html` jest CZYSTĄ funkcją — nie przyjmuje żadnego
portu, nie robi I/O. Metryki equity czyta z pól frozen dataclass `EquityCurve`
(funkcja `summary` nie istnieje). Krzywą i ECE liczy przez `reliability_curve`
i `expected_calibration_error` z domeny — nic nie duplikuje. Metryki liczone
są z przekazanych danych NIEZALEŻNIE od jakichkolwiek flag konfiguracji.
"""

from __future__ import annotations

import html as _html
from collections.abc import Sequence

from src.application.report_models import ResolvedPrediction
from src.domain.calibration_curve import (
    CalibrationBucket,
    expected_calibration_error,
    reliability_curve,
)
from src.domain.equity_curve import EquityCurve

# Parametry div-sparkline (technika z report_track_record._render_sparkline_html):
# słupki jako <div> o wysokości w px, skalowane do zakresu serii.
_SPARKLINE_MAX_PX = 40
_EQUITY_BARS = 40


def render_public_scorecard_html(
    resolved: Sequence[ResolvedPrediction] | None,
    equity: EquityCurve | None,
    calibration_samples: Sequence[tuple[float, bool]] | None,
    *,
    generated_on: str = "",
) -> str:
    """Renderuje publiczny scorecard kalibracji jako samodzielną stronę HTML.

    Parametry:
      - `resolved` — zamknięte predykcje (niosą symbol, ceny, uzasadnienia);
        używane WYŁĄCZNIE do policzenia agregatów N i hit-rate. Żadne pole PII
        nie trafia na stronę.
      - `equity` — gotowa krzywa kapitału; metryki (total_return, win_rate,
        max_drawdown, n_trades) czytane z jej PÓL, a `points` renderowane jako
        div-sparkline.
      - `calibration_samples` — surowe próbki `(confidence, was_correct)`;
        krzywa i ECE liczone przez domenę (`reliability_curve`).
      - `generated_on` — opcjonalny znacznik daty publikacji (escapowany).

    Samosupresja: brak jakichkolwiek danych → `""` (nie publikujemy pustej
    strony). Wszystko, co idzie do outputu, jest escapowane.
    """
    predictions = list(resolved) if resolved else []
    samples = list(calibration_samples) if calibration_samples else []
    has_equity = equity is not None and equity.n_trades > 0

    # Samosupresja — pusta strona nie ma wartości dowodowej.
    if not predictions and not samples and not has_equity:
        return ""

    buckets = reliability_curve(samples) if samples else []

    sections: list[str] = [
        _render_headline(predictions, equity),
        _render_calibration_section(buckets),
        _render_equity_section(equity),
        _render_methodology_section(),
    ]
    body = "".join(section for section in sections if section)
    return _wrap_page(body, generated_on)


def _hit_rate(predictions: Sequence[ResolvedPrediction]) -> tuple[int, float]:
    """Zwraca `(N, hit_rate)` policzone WYŁĄCZNIE z liczbowych agregatów.

    Symbole, ceny i uzasadnienia predykcji są celowo ignorowane — na stronę
    trafia tylko liczba predykcji i odsetek trafień kierunkowych.
    """
    n = len(predictions)
    if n == 0:
        return 0, 0.0
    correct = sum(1 for p in predictions if p.is_correct)
    return n, correct / n


def _render_headline(
    predictions: Sequence[ResolvedPrediction], equity: EquityCurve | None
) -> str:
    """Nagłówek z kluczowymi agregatami: N predykcji, hit-rate, ECE-anchor.

    Gdy brak zamkniętych predykcji, N i hit-rate biorą się z pól `EquityCurve`
    (n_trades / win_rate) — nadal czyste agregaty, zero PII.
    """
    n, hit_rate = _hit_rate(predictions)
    if n == 0 and equity is not None:
        n = equity.n_trades
        hit_rate = equity.win_rate

    stat_cards = (
        _stat_card("Predykcji rozliczonych", str(n))
        + _stat_card("Hit-rate kierunkowy", f"{hit_rate * 100:.1f}%")
    )
    return (
        "<section class='hero'>"
        "<h1>Publiczny scorecard kalibracji</h1>"
        "<p class='lede'>Weryfikowalny track record agenta — same agregaty, "
        "łącznie z naszymi pomyłkami. Bez obietnic, bez cudzych danych.</p>"
        f"<div class='cards'>{stat_cards}</div>"
        "</section>"
    )


def _stat_card(label: str, value: str) -> str:
    """Pojedyncza karta metryki (label + duża wartość). Oba pola escapowane."""
    return (
        "<div class='card'>"
        f"<div class='card-value'>{_html.escape(value)}</div>"
        f"<div class='card-label'>{_html.escape(label)}</div>"
        "</div>"
    )


def _render_calibration_section(buckets: Sequence[CalibrationBucket]) -> str:
    """Sekcja krzywej kalibracji: div-sparkline hit-rate per kubełek + ECE.

    Reliability diagram i ECE pochodzą z domeny (`reliability_curve` policzyło
    kubełki, `expected_calibration_error` liczy błąd). Nic nie duplikujemy.
    """
    if not buckets:
        return ""
    ece = expected_calibration_error(buckets)
    sparkline = _render_calibration_sparkline(buckets)
    return (
        "<section class='block'>"
        "<h2>Krzywa kalibracji</h2>"
        "<p class='desc'>Każdy słupek to jeden kubełek deklarowanej pewności; "
        "wysokość odpowiada realnemu hit-rate. Idealna kalibracja rośnie "
        "równo od lewej do prawej.</p>"
        f"{sparkline}"
        "<p class='ece'>Błąd kalibracji "
        f"<strong>ECE = {ece * 100:.1f}%</strong> — im niżej, tym wierniej "
        "deklarowana pewność odpowiada realnej trafności.</p>"
        "</section>"
    )


def _render_calibration_sparkline(buckets: Sequence[CalibrationBucket]) -> str:
    """Div-sparkline krzywej kalibracji (technika z `_render_sparkline_html`).

    Słupki to <div> o wysokości proporcjonalnej do `hit_rate` kubełka. Etykiety
    to wyłącznie zakresy pewności (liczby) — żadnych symboli ani kwot.
    """
    cells: list[str] = []
    for bucket in buckets:
        # Wysokość ≥2px, by nawet zerowy hit-rate był widoczny jako kreska.
        height = 2 + int(bucket.hit_rate * (_SPARKLINE_MAX_PX - 2))
        label = f"{bucket.lower * 100:.0f}–{bucket.upper * 100:.0f}%"
        cells.append(
            "<td class='spark-cell'>"
            f"<div class='spark-bar' style='height: {height}px;'></div>"
            f"<div class='spark-label'>{_html.escape(label)}</div>"
            "</td>"
        )
    return (
        "<table class='sparkline'><tr>" + "".join(cells) + "</tr></table>"
    )


def _render_equity_section(equity: EquityCurve | None) -> str:
    """Sekcja krzywej kapitału: agregaty z PÓL EquityCurve + div-sparkline.

    Czyta `total_return`, `win_rate`, `max_drawdown`, `n_trades` oraz `points`.
    Funkcja `summary` NIE ISTNIEJE — korzystamy wprost z pól dataclassy.
    """
    if equity is None or equity.n_trades == 0:
        return ""
    total = equity.total_return
    sign = "pos" if total >= 0 else "neg"
    stats = (
        _stat_card("Wynik łączny (paper)", f"{total * 100:+.1f}%")
        + _stat_card("Max obsunięcie", f"{equity.max_drawdown * 100:.1f}%")
        + _stat_card("Transakcji", str(equity.n_trades))
    )
    return (
        "<section class='block'>"
        "<h2>Krzywa kapitału (paper-PnL)</h2>"
        "<p class='desc'>Symulacja bez pieniędzy: kierunkowa ekspozycja wg "
        "prognoz, składana procentowo. Ilustruje jakość sygnału, nie realne "
        "zyski.</p>"
        f"<div class='cards'>{stats}</div>"
        f"<div class='equity-{sign}'>{_render_equity_sparkline(equity)}</div>"
        "</section>"
    )


def _render_equity_sparkline(equity: EquityCurve) -> str:
    """Div-sparkline serii kapitału (technika z `_render_sparkline_html`).

    Słupki skalowane do zakresu [min, max] equity z ostatnich N punktów.
    """
    points = list(equity.points)[-_EQUITY_BARS:]
    if not points:
        return ""
    equities = [p.equity for p in points]
    lo, hi = min(equities), max(equities)
    span = (hi - lo) or 1.0
    cells: list[str] = []
    for point in points:
        height = 2 + int((point.equity - lo) / span * (_SPARKLINE_MAX_PX - 2))
        tone = "up" if point.period_return >= 0 else "down"
        cells.append(
            "<td class='spark-cell'>"
            f"<div class='spark-bar {tone}' style='height: {height}px;'></div>"
            "</td>"
        )
    return "<table class='sparkline'><tr>" + "".join(cells) + "</tr></table>"


def _render_methodology_section() -> str:
    """Sekcja „Metodologia" — uczciwe, polskie copy o okresach słabości.

    Publicznie żenujący hit-rate w słabym okresie to FEATURE, nie bug:
    weryfikowalna pokora buduje zaufanie mocniej niż obietnice. Copy mówi
    wprost, że pokazujemy też własne pomyłki.
    """
    return (
        "<section class='block methodology'>"
        "<h2>Metodologia</h2>"
        "<p>Ten scorecard pokazuje <strong>wszystkie</strong> rozliczone "
        "predykcje z okna track-recordu — także te chybione. Nie ma tu "
        "cherry-pickingu ani „wyróżnionych trafień”.</p>"
        "<p>Hit-rate liczymy na predykcjach <em>kierunkowych</em> (wzrost / "
        "spadek); prognozy „bez ruchu” nie stawiają zakładu i nie zaniżają "
        "wyniku. ECE mierzy rozjazd między deklarowaną pewnością a realną "
        "trafnością — publikujemy go, bo przepewny model jest gorszy niż "
        "niepewny.</p>"
        "<p><strong>Okresy słabości są tu celowo widoczne.</strong> Zmiana "
        "reżimu rynku albo pechowa seria potrafią zjechać hit-rate poniżej "
        "rzutu monetą; taki wynik zostaje na stronie. Uczciwie pokazana "
        "pomyłka jest wiarygodniejsza niż wygładzony wykres — dlatego słabe "
        "okresy traktujemy jak część dowodu, nie jak usterkę do ukrycia.</p>"
        "<p class='fineprint'>Wynik paper-PnL jest symulacją bez realnych "
        "transakcji i nie stanowi rekomendacji inwestycyjnej.</p>"
        "</section>"
    )


def _wrap_page(body: str, generated_on: str) -> str:
    """Owija treść w kompletny, samodzielny dokument HTML (inline CSS)."""
    footer = ""
    if generated_on:
        footer = (
            "<footer class='foot'>Wygenerowano: "
            f"{_html.escape(generated_on)}</footer>"
        )
    return (
        "<!DOCTYPE html>"
        "<html lang='pl'>"
        "<head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Publiczny scorecard kalibracji</title>"
        f"<style>{_page_css()}</style>"
        "</head>"
        "<body>"
        f"<main class='wrap'>{body}{footer}</main>"
        "</body>"
        "</html>"
    )


def _page_css() -> str:
    """Wbudowany CSS strony — bez zewnętrznych zasobów (samodzielność).

    Świadomie bez reguł `@media`/`@import`, żeby znak „at” nie pojawił się w
    outpucie (twardy inwariant anty-PII: brak „at” = brak ryzyka e-maila).
    """
    return (
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,"
        "sans-serif;color:#0f172a;background:#f8fafc;margin:0;padding:24px;"
        "line-height:1.5}"
        ".wrap{max-width:720px;margin:0 auto}"
        ".hero h1{font-size:26px;margin:0 0 6px}"
        ".lede{color:#475569;font-size:15px;margin:0 0 18px}"
        ".cards{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0 8px}"
        ".card{flex:1 1 120px;background:#fff;border:1px solid #e2e8f0;"
        "border-radius:10px;padding:14px}"
        ".card-value{font-size:24px;font-weight:700;color:#1d4ed8}"
        ".card-label{font-size:12px;color:#64748b;text-transform:uppercase;"
        "letter-spacing:.04em;margin-top:4px}"
        ".block{background:#fff;border:1px solid #e2e8f0;border-radius:10px;"
        "padding:16px;margin:16px 0}"
        ".block h2{font-size:17px;margin:0 0 8px}"
        ".desc{color:#475569;font-size:13px;margin:0 0 12px}"
        ".sparkline{border-collapse:collapse;height:" + str(_SPARKLINE_MAX_PX)
        + "px;margin:8px 0}"
        ".spark-cell{vertical-align:bottom;padding:0 2px;text-align:center}"
        ".spark-bar{width:14px;background:#0ea5e9;border-radius:2px;"
        "margin:0 auto}"
        ".spark-bar.up{background:#10b981}"
        ".spark-bar.down{background:#f97316}"
        ".spark-label{font-size:9px;color:#94a3b8;margin-top:3px}"
        ".ece{font-size:13px;color:#334155;margin:8px 0 0}"
        ".methodology p{font-size:13px;color:#334155;margin:0 0 10px}"
        ".fineprint{color:#94a3b8;font-size:11px}"
        ".foot{color:#94a3b8;font-size:11px;text-align:center;margin:20px 0 0}"
    )
