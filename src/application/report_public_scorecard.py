"""Publiczny scorecard kalibracji jako samodzielna strona HTML (roadmap #10).

Cały track record żyje w prywatnym mailu; `web_digest_enabled` publikowałby
PEŁNY prywatny digest, który nie nadaje się do upublicznienia. Ten moduł
produkuje osobny, PUBLICZNY artefakt: kompletną, samodzielną stronę HTML
zawierającą WYŁĄCZNIE agregaty — liczbę predykcji, hit-rate kierunkowy z
przedziałem Wilsona, n_eff, naiwnymi baseline'ami i werdyktem istotności,
Brier / BSS / ECE, krzywą kalibracji jako div-sparkline (z n per kubełek),
krzywą kapitału portfela (po kosztach obrotu, obok koszyka i indeksu S&P 500 —
WIA-50/21) oraz okno i datę generacji.

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
from decimal import Decimal

from src.application.report_evidence import (
    NO_BENCHMARK_PL,
    PORTFOLIO_METHOD_CHANGE_PL,
    PUBLIC_BENCHMARK_LABEL,
    PaperCosts,
    baselines_label,
    benchmark_detail,
    benchmark_line,
    benchmark_method,
    hit_rate_detail,
    hit_rate_evidence,
    hit_rate_value,
    murphy_label,
    paper_pnl_label,
    portfolio_assumptions,
    probability_label,
    uncertainty_note,
)
from src.application.report_formatting import DOT_NUMBERS
from src.application.report_models import ResolvedPrediction
from src.domain.benchmark import BenchmarkComparison
from src.domain.calibration_curve import (
    CalibrationBucket,
    calibration_error_evidence,
    expected_calibration_error,
    probability_scores,
    reliability_curve,
)
from src.domain.equity_curve import EquityCurve
from src.domain.evaluation_protocol import EvaluationProtocol
from src.domain.track_record import HitRateEvidence

# Parametry div-sparkline (technika z report_track_record._render_sparkline_html):
# słupki jako <div> o wysokości w px, skalowane do zakresu serii.
_SPARKLINE_MAX_PX = 40
_EQUITY_BARS = 40

# WIA-42: etykiety identyfikatorów z `evaluation_protocol.toml` na stronie
# publicznej. Indeks nazywa się „S&P 500”, nie tickerem — żaden symbol
# watchlisty nie trafia na stronę.
PROTOCOL_LABELS: dict[str, str] = {
    "always_bullish": "zawsze wzrost",
    "always_bearish": "zawsze spadek",
    "always_sideways": "zawsze bez ruchu",
    "basket_buy_and_hold": "koszyk kup-i-trzymaj",
    "sp500_buy_and_hold": f"{PUBLIC_BENCHMARK_LABEL} kup-i-trzymaj",
    "directional_hit_rate": "hit-rate kierunkowy",
    "brier_score": "Brier score",
    "brier_skill_score": "Brier Skill Score",
    "expected_calibration_error": "ECE",
    "market_relative_hit_rate": "hit-rate względem rynku",
    "paper_pnl_net": "paper-PnL po kosztach",
}


def render_public_scorecard_html(
    resolved: Sequence[ResolvedPrediction] | None,
    equity: EquityCurve | None,
    calibration_samples: Sequence[tuple[float, bool]] | None,
    *,
    generated_on: date | str = "",
    window_days: int | None = None,
    costs: PaperCosts | None = None,
    benchmark: BenchmarkComparison | None = None,
    protocol: EvaluationProtocol | None = None,

    calibration_by_cycle: Sequence[Sequence[tuple[float, bool]]] | None = None,
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
      - `costs` — założenia kosztowe krzywej (WIA-50); None = krzywa brutto.
        Strona podaje same koszty w bps, nigdy symbole z `crypto_symbols`.
      - `benchmark` — agent obok koszyka i indeksu (WIA-21); sparkline dostaje
        drugą linię (koszyk). Koszyk to agregat, a indeks nazywa się „S&P 500”
        zamiast tickera — żaden symbol nie trafia na stronę. None = „Na tle
        rynku: brak danych benchmarku”.
      - `protocol` — protokół ewaluacji z hashem (WIA-42), renderowany w sekcji
        metodologii; ten sam sha256 niesie każdy commitment v2. None = sekcja
        bez protokołu, jak przed WIA-42.

      - `calibration_by_cycle` — te same próbki pogrupowane po cyklu (dniu UTC,
        WIA-58); z nimi ECE dostaje kubełki równoliczne, korektę obciążenia,
        n_eff i przedział z bootstrapu po cyklach. None = sama liczba ECE.

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
        _render_calibration_section(buckets, samples, calibration_by_cycle),
        _render_equity_section(equity, costs, benchmark),
        _render_methodology_section(protocol),
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


def _ece_line(
    ece: float, calibration_by_cycle: Sequence[Sequence[tuple[float, bool]]] | None
) -> str:
    """„ECE = 7.0%” albo z cyklami (WIA-58): przedział, n, n_eff i ECE po korekcie."""
    evidence = (
        calibration_error_evidence(calibration_by_cycle) if calibration_by_cycle else None
    )
    if evidence is None:
        return f"<strong>ECE = {ece * 100:.1f}%</strong>"
    if evidence.interval is None:
        spread = " (za mało cykli na bootstrap)"
    else:
        low, high = evidence.interval
        spread = f" [{low * 100:.1f}%; {high * 100:.1f}%]"
    sample = f"n={evidence.n}, n_eff≈{evidence.n_eff:.0f}"
    debiased = max(0.0, evidence.debiased) * 100
    return (
        f"<strong>ECE = {evidence.ece * 100:.1f}%{spread}, {sample}</strong> "
        f"(kubełki równoliczne; po korekcie obciążenia {debiased:.1f}%)"
    )


def _render_calibration_section(
    buckets: Sequence[CalibrationBucket],
    samples: Sequence[tuple[float, bool]],
    calibration_by_cycle: Sequence[Sequence[tuple[float, bool]]] | None = None,
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
        "Idealna kalibracja rośnie równo od lewej do prawej. Same zakłady na "
        "kierunek, bez prognoz „bez ruchu” — te trafiają za darmo w spokojne "
        "dni.</p>"
        f"{sparkline}"
        "<p class='ece'>Pomocniczo: błąd kalibracji "
        f"{_ece_line(ece, calibration_by_cycle)} — im niżej, tym wierniej "
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


def _render_equity_section(
    equity: EquityCurve | None,
    costs: PaperCosts | None,
    benchmark: BenchmarkComparison | None,
) -> str:
    """Sekcja krzywej kapitału: agregaty z PÓL EquityCurve + div-sparkline.

    Czyta `total_return`, `win_rate`, `max_drawdown`, `n_trades` oraz `points`.
    Funkcja `summary` NIE ISTNIEJE — korzystamy wprost z pól dataclassy.
    Nagłówek i założenia mówią, z jakim kosztem liczono krzywą (WIA-50).
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
    title = (
        "Krzywa kapitału (paper-PnL)"
        if costs is None
        else f"Krzywa kapitału — {paper_pnl_label(costs, numbers=DOT_NUMBERS)}"
    )
    assumptions = portfolio_assumptions(costs, numbers=DOT_NUMBERS)
    return (
        "<section class='block'>"
        f"<h2>{_html.escape(title)}</h2>"
        "<p class='desc'>Symulacja bez pieniędzy: portfel kierunkowych "
        "ekspozycji wg prognoz. Ilustruje jakość sygnału, nie realne zyski. "
        f"{_html.escape(assumptions)} "
        f"{_html.escape(PORTFOLIO_METHOD_CHANGE_PL)}</p>"
        f"<div class='cards'>{stats}</div>"
        f"{_render_benchmark(benchmark)}"
        f"<div class='equity-{sign}'>{_render_equity_sparkline(equity, benchmark)}</div>"
        "</section>"
    )


def _render_benchmark(benchmark: BenchmarkComparison | None) -> str:
    """WIA-21 — agent obok koszyka i S&P 500 (kropka dziesiętna) albo jawny brak danych."""
    if benchmark is None:
        return f"<p class='evidence'>Na tle rynku: {NO_BENCHMARK_PL}</p>"
    label = PUBLIC_BENCHMARK_LABEL
    line = benchmark_line(benchmark, numbers=DOT_NUMBERS, index_label=label)
    detail = benchmark_detail(benchmark, numbers=DOT_NUMBERS, index_label=label)
    method = benchmark_method(label)
    return (
        f"<p class='evidence'>Na tle rynku: <strong>{_html.escape(line)}</strong></p>"
        f"<p class='evidence'>{_html.escape(detail)}. {_html.escape(method)} "
        "Na wykresie: agent (kolor), koszyk (szary).</p>"
    )


def _render_equity_sparkline(
    equity: EquityCurve, benchmark: BenchmarkComparison | None = None
) -> str:
    """Div-sparkline serii kapitału (technika z `_render_sparkline_html`).

    Słupki skalowane do zakresu [min, max] equity z ostatnich N punktów. Z
    benchmarkiem (WIA-21) drugi wiersz to koszyk we wspólnej skali.
    """
    points = list(equity.points)[-_EQUITY_BARS:]
    if not points:
        return ""
    bench = [] if benchmark is None else list(benchmark.basket.points)[-_EQUITY_BARS:]
    equities = [p.equity for p in points] + [p.equity for p in bench]
    lo, hi = min(equities), max(equities)
    span = (hi - lo) or 1.0

    def bar(value: float, tone: str) -> str:
        height = 2 + int((value - lo) / span * (_SPARKLINE_MAX_PX - 2))
        return (
            "<td class='spark-cell'>"
            f"<div class='spark-bar {tone}' style='height: {height}px;'></div>"
            "</td>"
        )

    agent = "".join(
        bar(point.equity, "up" if point.period_return >= 0 else "down") for point in points
    )
    rows = f"<tr>{agent}</tr>"
    if bench:
        rows += "<tr>" + "".join(bar(point.equity, "bench") for point in bench) + "</tr>"
    return f"<table class='sparkline'>{rows}</table>"


def _render_methodology_section(protocol: EvaluationProtocol | None = None) -> str:
    """Sekcja „Metodologia" — uczciwe, polskie copy o okresach słabości.

    Publicznie żenujący hit-rate w słabym okresie to FEATURE, nie bug:
    weryfikowalna pokora buduje zaufanie mocniej niż obietnice. Copy mówi
    wprost, że pokazujemy też własne pomyłki. WIA-42: z protokołem sekcja
    kończy się jego regułami i hashem.
    """
    protocol_block = "" if protocol is None else _render_protocol(protocol)
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
        "przy zerowej umiejętności. Liczymy je na 10 kubełkach równolicznych "
        "(granice kwantylowe, bo pewności skupiają się w wąskim paśmie), więc "
        "różnią się one od kubełków równej szerokości na wykresie; przy małej "
        "próbie ECE jest zawyżone, dlatego obok podajemy wartość po odjęciu "
        "ECE oczekiwanego przy idealnej kalibracji i przedział z bootstrapu "
        "losującego całe dni.</p>"
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
        f"{protocol_block}"
        "<p class='fineprint'>Wynik paper-PnL jest symulacją bez realnych "
        "transakcji i nie stanowi rekomendacji inwestycyjnej.</p>"
        "</section>"
    )


def _render_protocol(protocol: EvaluationProtocol) -> str:
    """WIA-42: reguły protokołu ewaluacji i sha256 pliku, z którego pochodzą."""

    def labels(names: Sequence[str]) -> str:
        return ", ".join(PROTOCOL_LABELS.get(name, name) for name in names)

    def band(fraction: Decimal) -> str:
        return DOT_NUMBERS.pct(float(fraction), decimals=1)

    sideways = (
        "wchodzą do mianownika hit-rate"
        if protocol.sideways_in_hit_rate
        else "nie wchodzą do mianownika hit-rate"
    )
    selection = (
        "ręcznie" if protocol.universe_selection == "manual" else protocol.universe_selection
    )
    items = [
        f"Horyzont: {protocol.horizon_hours} h.",
        "Trafność: wzrost, gdy cena po horyzoncie jest wyższa niż przy prognozie, "
        "spadek, gdy niższa (martwa strefa "
        f"{band(protocol.directional_dead_zone)}); bez ruchu, gdy zmiana mieści się "
        f"w ±{band(protocol.sideways_band)} (krypto ±{band(protocol.crypto_sideways_band)}). "
        f"Prognozy „bez ruchu” {sideways}.",
        f"Naiwne strategie: {labels(protocol.naive_baselines)}; "
        f"benchmarki: {labels(protocol.benchmarks)}.",
        f"Metryki główne: {labels(protocol.primary_metrics)}; "
        f"pomocnicze: {labels(protocol.secondary_metrics)}.",
        f"Istotność: α = {DOT_NUMBERS.number(protocol.alpha, decimals=2)}, "
        f"FDR q = {DOT_NUMBERS.number(protocol.fdr_q, decimals=2)}.",
        f"Okno: {protocol.window_days} dni.",
        f"Uniwersum: wybrane {selection} ({protocol.universe_selected}); "
        "wersja listy jest w przepisie każdej predykcji.",
    ]
    rows = "".join(f"<li>{_html.escape(item)}</li>" for item in items)
    return (
        "<h3>Protokół ewaluacji</h3>"
        f"<p>Wersja {protocol.version}, przyjęta "
        f"{_html.escape(protocol.adopted_on.isoformat())}. Reguły zamrożono przed "
        "wynikami, które oceniają; hash pliku protokołu niesie każdy commitment "
        "predykcji, więc metryk nie da się dobrać po fakcie.</p>"
        f"<ul>{rows}</ul>"
        f"<p class='fineprint'>sha256 protokołu: <code>{_html.escape(protocol.sha256)}</code></p>"
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
        ".spark-bar.bench{background:#94a3b8}"
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
