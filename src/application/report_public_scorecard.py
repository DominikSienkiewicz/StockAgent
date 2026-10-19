"""Publiczny scorecard kalibracji jako samodzielna strona HTML (roadmap #10).

Cały track record żyje w prywatnym mailu; `web_digest_enabled` publikowałby
PEŁNY prywatny digest, który nie nadaje się do upublicznienia. Ten moduł
produkuje osobny, PUBLICZNY artefakt: kompletną, samodzielną stronę HTML
zawierającą WYŁĄCZNIE agregaty — liczbę predykcji, hit-rate kierunkowy z
przedziałem Wilsona, n_eff, naiwnymi baseline'ami i werdyktem istotności,
Brier / BSS / ECE, krzywą kalibracji jako div-sparkline (z n per kubełek),
krzywą kapitału portfela oraz okno i datę generacji.

Bezwzględne inwarianty prywatności:
  - ZERO PII (adresy e-mail, dane odbiorców),
  - ZERO symboli watchlisty,
  - ZERO kwot (ceny, waluty).

`render_public_scorecard_html` jest CZYSTĄ funkcją — nie przyjmuje żadnego
portu, nie robi I/O. Metryki equity czyta z pól frozen dataclass `EquityCurve`
(funkcja `summary` nie istnieje). Hit-rate liczy jedna domenowa definicja
(`directional_hit_rate`, przez `report_evidence`), krzywą, ECE i Brier —
`calibration_curve` z domeny; nic nie duplikuje. Metryki liczone są z
przekazanych danych NIEZALEŻNIE od jakichkolwiek flag konfiguracji.
"""

from __future__ import annotations

import html as _html
from collections.abc import Sequence
from datetime import date, timedelta

from src.application.report_evidence import (
    PORTFOLIO_ASSUMPTIONS_PL,
    PORTFOLIO_METHOD_CHANGE_PL,
    baselines_label,
    hit_rate_detail,
    hit_rate_evidence,
    hit_rate_value,
    murphy_label,
    probability_label,
    uncertainty_note,
)
from src.application.report_formatting import DOT_NUMBERS
from src.application.report_models import ResolvedPrediction
from src.domain.calibration_curve import (
    CalibrationBucket,
    expected_calibration_error,
    probability_scores,
    reliability_curve,
)
from src.domain.equity_curve import EquityCurve
from src.domain.track_record import HitRateEvidence

# Parametry div-sparkline (technika z report_track_record._render_sparkline_html):
# słupki jako <div> o wysokości w px, skalowane do zakresu serii.
_SPARKLINE_MAX_PX = 40
_EQUITY_BARS = 40


def render_public_scorecard_html(
    resolved: Sequence[ResolvedPrediction] | None,
    equity: EquityCurve | None,
    calibration_samples: Sequence[tuple[float, bool]] | None,
    *,
    generated_on: date | str = "",
    window_days: int | None = None,
) -> str:
    """Renderuje publiczny scorecard kalibracji jako samodzielną stronę HTML.

    Parametry:
      - `resolved` — zamknięte predykcje (niosą symbol, ceny, uzasadnienia);
        używane WYŁĄCZNIE do policzenia agregatów (N, hit-rate kierunkowy,
        baseline'y, n_eff). Żadne pole PII nie trafia na stronę.
      - `equity` — gotowa krzywa kapitału (orkiestrator podaje krzywą portfela
        o równej wadze per cykl); metryki czytane z jej PÓL, a `points`
        renderowane jako div-sparkline.
      - `calibration_samples` — surowe próbki `(confidence, was_correct)`;
        krzywa, ECE, Brier i rozkład Murphy'ego liczone przez domenę.
      - `generated_on` — data publikacji (escapowana); jako `date` razem z
        `window_days` daje zakres okna „od–do”.
      - `window_days` — długość okna track-recordu ujawniana na stronie.

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
        _render_headline(predictions, generated_on, window_days),
        _render_calibration_section(buckets, samples),
        _render_equity_section(equity),
        _render_methodology_section(),
    ]
    body = "".join(section for section in sections if section)
    return _wrap_page(body, str(generated_on))


def _render_headline(
    predictions: Sequence[ResolvedPrediction],
    generated_on: date | str,
    window_days: int | None,
) -> str:
    """Nagłówek: N, hit-rate kierunkowy z przedziałem, karta SIDEWAYS, dowód, okno.

    Hit-rate liczy jedna domenowa definicja (SIDEWAYS poza mianownikiem). Brak
    predykcji kierunkowych → „—”, bez podstawiania `EquityCurve.win_rate`.
    Symbole, ceny i uzasadnienia są celowo ignorowane — zero PII.
    """
    evidence = hit_rate_evidence(predictions)
    sideways = 0 if evidence is None else evidence.hit_rate.sideways
    stat_cards = (
        _stat_card("Predykcji rozliczonych", str(len(predictions)))
        + _stat_card(
            "Hit-rate kierunkowy",
            hit_rate_value(evidence, numbers=DOT_NUMBERS),
            "" if evidence is None else hit_rate_detail(evidence, numbers=DOT_NUMBERS),
        )
        + _stat_card("Prognozy „bez ruchu” (poza hit-rate)", str(sideways))
    )
    return (
        "<section class='hero'>"
        "<h1>Publiczny scorecard kalibracji</h1>"
        "<p class='lede'>Weryfikowalny track record agenta — same agregaty, "
        "łącznie z naszymi pomyłkami. Bez obietnic, bez cudzych danych.</p>"
        f"{_render_window(generated_on, window_days)}"
        f"<div class='cards'>{stat_cards}</div>"
        f"{_render_evidence(evidence)}"
        "</section>"
    )


def _render_window(generated_on: date | str, window_days: int | None) -> str:
    """„Okno: ostatnie 30 dni (2026-09-18 – 2026-10-18)” — daty ISO, bez znaku „at”."""
    if window_days is None:
        return ""
    text = f"Okno: ostatnie {window_days} dni"
    if isinstance(generated_on, date):
        start = generated_on - timedelta(days=window_days)
        text += f" ({start.isoformat()} – {generated_on.isoformat()})"
    return f"<p class='window'>{_html.escape(text)}</p>"


def _render_evidence(evidence: HitRateEvidence | None) -> str:
    """Baseline'y, werdykt istotności i uwagi o niepewności pod kartami."""
    if evidence is None or evidence.hit_rate.rate is None:
        return ""
    lines: list[str] = []
    baselines = baselines_label(evidence, numbers=DOT_NUMBERS)
    if baselines:
        lines.append(f"Naiwne strategie na tych samych predykcjach: {baselines}.")
    if evidence.verdict is not None:
        lines.append(
            f"Werdykt: {evidence.verdict} — test dwumianowy względem lepszej z "
            "naiwnych strategii kierunkowych."
        )
    note = uncertainty_note(evidence, numbers=DOT_NUMBERS)
    if note:
        lines.append(f"Niepewność: {note}.")
    return "".join(f"<p class='evidence'>{_html.escape(line)}</p>" for line in lines)


def _stat_card(label: str, value: str, note: str = "") -> str:
    """Pojedyncza karta metryki (wartość + label + opcjonalna notka). Escapowane."""
    note_html = f"<div class='card-note'>{_html.escape(note)}</div>" if note else ""
    return (
        "<div class='card'>"
        f"<div class='card-value'>{_html.escape(value)}</div>"
        f"<div class='card-label'>{_html.escape(label)}</div>"
        f"{note_html}"
        "</div>"
    )


def _render_calibration_section(
    buckets: Sequence[CalibrationBucket], samples: Sequence[tuple[float, bool]]
) -> str:
    """Sekcja pewności: Brier / BSS / log loss, rozkład Murphy'ego, sparkline + ECE.

    Brier i Brier Skill Score są metrykami głównymi; ECE zostaje pomocniczo,
    bo bywa 0 przy zerowej umiejętności. Wszystko liczy domena.
    """
    if not buckets:
        return ""
    ece = expected_calibration_error(buckets)
    sparkline = _render_calibration_sparkline(buckets)
    scores = probability_scores(samples)
    scores_html = ""
    if scores is not None:
        probability = probability_label(scores, numbers=DOT_NUMBERS)
        murphy = murphy_label(scores, numbers=DOT_NUMBERS)
        scores_html = (
            f"<p class='ece'><strong>{_html.escape(probability)}</strong>"
            " — Brier im niżej, tym lepiej; BSS mówi, o ile bijemy prognozę "
            "„zawsze stopa bazowa” (0% = brak umiejętności).</p>"
            "<p class='ece'>Rozkład Murphy'ego: "
            f"{_html.escape(murphy)} (rzetelność, rozdzielczość, "
            "niepewność).</p>"
        )
    return (
        "<section class='block'>"
        "<h2>Krzywa kalibracji</h2>"
        f"{scores_html}"
        "<p class='desc'>Każdy słupek to jeden kubełek deklarowanej pewności; "
        "wysokość odpowiada realnemu hit-rate, pod spodem liczba predykcji. "
        "Idealna kalibracja rośnie równo od lewej do prawej.</p>"
        f"{sparkline}"
        "<p class='ece'>Pomocniczo: błąd kalibracji "
        f"<strong>ECE = {ece * 100:.1f}%</strong> — im niżej, tym wierniej "
        "deklarowana pewność odpowiada realnej trafności.</p>"
        "</section>"
    )


def _render_calibration_sparkline(buckets: Sequence[CalibrationBucket]) -> str:
    """Div-sparkline krzywej kalibracji (technika z `_render_sparkline_html`).

    Słupki to <div> o wysokości proporcjonalnej do `hit_rate` kubełka. Etykiety
    to wyłącznie zakresy pewności i liczność kubełka — żadnych symboli ani kwot.
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
            f"<div class='spark-count'>n={bucket.count}</div>"
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
        "<p class='desc'>Symulacja bez pieniędzy: portfel kierunkowych "
        "ekspozycji wg prognoz. Ilustruje jakość sygnału, nie realne zyski. "
        f"{_html.escape(PORTFOLIO_ASSUMPTIONS_PL)} "
        f"{_html.escape(PORTFOLIO_METHOD_CHANGE_PL)}</p>"
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
        "spadek); prognozy „bez ruchu” nie stawiają zakładu, więc nie wchodzą "
        "do mianownika — ich liczbę podajemy osobno. Obok trafności stoi 95% "
        "przedział Wilsona liczony na efektywnej liczności n_eff (predykcje "
        "jednego dnia ruszają się razem, więc nie są niezależne), naiwne "
        "strategie „zawsze wzrost / spadek / bez ruchu” policzone na tych "
        "samych predykcjach oraz test dwumianowy względem lepszej z naiwnych "
        "strategii kierunkowych.</p>"
        "<p>Pewność oceniamy Brier score i Brier Skill Score względem prognozy "
        "klimatologicznej (stała stopa bazowa). ECE publikujemy pomocniczo: "
        "przepewny model jest gorszy niż niepewny, ale ECE bywa zerowe także "
        "przy zerowej umiejętności.</p>"
        "<p><strong>Okresy słabości są tu celowo widoczne.</strong> Zmiana "
        "reżimu rynku albo pechowa seria potrafią zjechać hit-rate poniżej "
        "rzutu monetą; taki wynik zostaje na stronie. Uczciwie pokazana "
        "pomyłka jest wiarygodniejsza niż wygładzony wykres — dlatego słabe "
        "okresy traktujemy jak część dowodu, nie jak usterkę do ukrycia.</p>"
        # WIA-48: survivorship — watchlista to dzisiejsi zwycięzcy. Bez nazw
        # spółek: strona publiczna nie ujawnia symboli watchlisty.
        "<p><strong>Uniwersum wybrane ręcznie (2026-06).</strong> Listę "
        "obserwowanych spółek wybraliśmy ręcznie, z wiedzą o ich wcześniejszych "
        "wzrostach — to nie losowa próba rynku. Przewaga prognoz wzrostowych "
        "(long-bias) na takim uniwersum nie jest umiejętnością: dlatego obok "
        "trafności stoi naiwna strategia „zawsze wzrost” policzona na tych "
        "samych predykcjach.</p>"
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
        ".card-note{font-size:11px;color:#64748b;margin-top:6px}"
        ".evidence{font-size:13px;color:#334155;margin:6px 0 0}"
        ".window{font-size:13px;color:#475569;margin:0 0 6px}"
        ".spark-count{font-size:9px;color:#64748b}"
        ".methodology p{font-size:13px;color:#334155;margin:0 0 10px}"
        ".fineprint{color:#94a3b8;font-size:11px}"
        ".foot{color:#94a3b8;font-size:11px;text-align:center;margin:20px 0 0}"
    )
