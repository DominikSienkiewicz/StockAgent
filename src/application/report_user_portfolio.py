"""Render sekcji "💼 Twój portfel" w raporcie e-mail (roadmap #15).

Moment "to jest O MNIE": realny P/L per pozycja i łącznie, realne wagi kapitału,
opcjonalnie kwoty w PLN po kursie USD/PLN oraz ostrzeżenie o koncentracji
sektora ("Twoje 3 pozycji (…) to jeden sektor «…» — 61% kapitału"). Grupy to
sektory akcji z `sectors` w config.toml (PORT-05), nie policzona korelacja.

Renderer jest CZYSTY (jak `report_persona_leaderboard`): bierze agregat
`Portfolio` z domeny + bieżące ceny (już pobrane w cyklu) + gotową mapę
klastrów + `now` i produkuje fragment HTML/tekst. Zero I/O, zero portów.

Kurs USD/PLN jest PARAMETREM (`usd_pln`), nie wstrzykniętym portem —
orkiestrator bierze go z `MacroRiskReport.polish_macro`. Gdy `None`, sekcja
PLN znika (pokazujemy tylko USD), bez wyjątku.

Badge STALE to WYMAGANIE, nie ozdobnik: ręcznie aktualizowana tabela pozycji
potrafi się zestarzeć, a stęchły portfel daje FAŁSZYWY P/L — gorszy niż brak
sekcji. Gdy `Portfolio.freshness(now)` == STALE, renderujemy widoczny badge
ostrzegający, że P/L może być nieaktualny.

Nie reużywamy `provenance.build_provenance_badges` — tamten agregat ocenia
proweniencję sygnałów predykcji (TTL fundamentów, `degraded_reason`, flagi
walidacji wejścia); świeżość ręcznej tabeli pozycji to inny sygnał i wymuszanie
go przez tamten kontrakt oznaczałoby podawanie fikcyjnego `fundamentals_fetched_at`.
Prosty, dedykowany badge z `PortfolioFreshness` jest czytelniejszy.

Sekcja samosupresująca: brak policzalnych pozycji (pusty portfel albo żadna
pozycja nie ma bieżącej ceny) → `render_*` zwraca "".

PORT-03: pozycje bez ceny w tym cyklu (np. tickery UNSUPPORTED_PRICE) są
wymienione pod tabelą z udziałem w koszcie, a sumy podpisane „bez pozycji
niewycenionych” — wagi i P/L liczą się tylko z wycenionych pozycji.

PORT-06: pod sumami linia „Zmiana od ostatniego maila” (kwota, %, top 3
kontrybutorów) z gotowego `PortfolioChange` z domeny, a na pierwszym ekranie
box z tą samą liczbą. Po pauzie podpis mówi „od <data>”.

PORT-11: z `pln_rates` (kursy NBP: bieżące i z dni zakupu) tabela dostaje
kolumny „P/L zł” (koszt po kursie z dnia zakupu, wartość po bieżącym) i „w tym
kurs” (wynik walutowy z `Portfolio.pln_pnl`). Bez daty zakupu albo kursu z tego
dnia komórka mówi to wprost — bez cichego przybliżenia bieżącym kursem. Kwoty
pozycji są w ich walucie (PORT-18); sumy w walucie natywnej tylko, gdy wszystkie
pozycje mają tę samą. `pln_rates` wygrywa z `usd_pln`, które zostaje dla
wołających bez kursów z dnia zakupu (stara kolumna „P/L (PLN)”).

Jedna waluta: z `pln_rates` kolumna „Waga”, ostrzeżenie o koncentracji sektora
i udział niewycenionych w koszcie liczą się w złotych (`Portfolio.pln_prices`),
a pozycja bez kursu NBP wypada z nich z notą „Bez kursu NBP: …”. Linia zmiany
dostaje walutę od wołającego (`change_currency`, w zł gdy znane są kursy).
"""

from __future__ import annotations

import dataclasses
import html as _html
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.application.report_formatting import PALETTE, format_local_date, number_exact
from src.application.report_models import SymbolResult
from src.application.reporting.section import ReportSection, SectionInput, text_block
from src.domain.asset import is_reference_stale
from src.domain.cycle_maturity import SkipReason
from src.domain.number_format import format_money_pl, format_pct_pl
from src.domain.portfolio import (
    DEFAULT_MAX_POSITION_AGE_DAYS,
    DEFAULT_POSITION_CURRENCY,
    HOME_CURRENCY,
    ClusterExposure,
    PlnPnlGap,
    PlnPrices,
    PlnRates,
    Portfolio,
    PortfolioChange,
    PortfolioFreshness,
    PortfolioPlnPnL,
    PortfolioPnL,
    PositionPlnPnL,
    PositionPnL,
)

# Zamrożony próg koncentracji grupy (sektora). Grupa (≥2 symbole) o udziale
# ≥ 40% kapitału wywołuje ostrzeżenie. Wartość celowo stała — zmiana ma łamać
# `test_threshold_is_frozen_constant`, bo to decyzja o ryzyku, nie kosmetyka.
_CLUSTER_CONCENTRATION_THRESHOLD = Decimal("0.40")

# Kolory zysku/straty — spójne z resztą raportu (report_persona_leaderboard).
_COLOR_GAIN = PALETTE.positive
_COLOR_LOSS = PALETTE.negative


def _pnl_color(value: Decimal) -> str:
    """Zielony dla zysku (i zera), czerwony dla straty."""
    return _COLOR_LOSS if value < 0 else _COLOR_GAIN


def _format_pln(value: Decimal) -> str:
    """Kwota w PLN (np. „-800,00 zł”)."""
    return format_money_pl(value, "PLN")


def _format_signed_pln(value: Decimal) -> str:
    """P/L w PLN z jawnym znakiem (+/-)."""
    return format_money_pl(value, "PLN", signed=True)


def _format_money(value: Decimal, currency: str, *, signed: bool = False) -> str:
    """PORT-18: kwota pozycji w jej walucie („$1 400,00”, „800,00 EUR”)."""
    return format_money_pl(value, currency, signed=signed)


# PORT-11: dlaczego komórka „w tym kurs” nie ma kwoty — po polsku, wprost.
_GAP_NOTES = {
    PlnPnlGap.PURCHASE_DATE_UNKNOWN: "kurs z dnia zakupu nieznany",
    PlnPnlGap.PURCHASE_RATE_MISSING: "brak kursu NBP z dnia zakupu",
}


def _gap_note(row: PositionPlnPnL) -> str:
    if row.gap is PlnPnlGap.CURRENT_RATE_MISSING:
        return f"brak kursu {row.currency}/PLN"
    return _GAP_NOTES[row.gap] if row.gap is not None else ""


def _pln_pnl_cell(row: PositionPlnPnL) -> str:
    return "—" if row.pnl_pln is None else _format_signed_pln(row.pnl_pln)


def _fx_cell(row: PositionPlnPnL) -> str:
    return _gap_note(row) if row.fx_pln is None else _format_signed_pln(row.fx_pln)


def _single_currency(pnl: PortfolioPnL) -> str | None:
    """Waluta wspólna wszystkich wycenionych pozycji albo None (różne waluty)."""
    currencies = {row.currency for row in pnl.positions}
    return currencies.pop() if len(currencies) == 1 else None


def _currencies_label(pnl: PortfolioPnL) -> str:
    return ", ".join(sorted({row.currency for row in pnl.positions}))


def _rates_label(pln: PortfolioPlnPnL, rates: PlnRates) -> str:
    """„ (kursy NBP: EUR 4,2710 · USD 3,6820)” — kursy bieżące użyte w sumach."""
    used = sorted(
        {row.currency for row in pln.positions if row.currency != HOME_CURRENCY}
        & set(rates.current)
    )
    if not used:
        return ""
    joined = " · ".join(f"{code} {number_exact(rates.current[code])}" for code in used)
    return f" (kursy NBP: {joined})"


def _pln_totals(pln: PortfolioPlnPnL, rates: PlnRates) -> tuple[str, str, str]:
    """Sumy w zł jako (prefiks z wartością, P/L z kursem albo "", pominięte)."""
    head = f"W PLN{_rates_label(pln, rates)}: wartość {_format_pln(pln.total_value_pln)}"
    complete = any(row.gap is None for row in pln.positions)
    pnl_part = (
        f"P/L {_format_signed_pln(pln.total_pnl_pln)} "
        f"(w tym kurs {_format_signed_pln(pln.total_fx_pln)})"
        if complete
        else ""
    )
    left_out = (
        f"bez P/L w zł: {', '.join(pln.incomplete_symbols)}" if pln.incomplete_symbols else ""
    )
    return head, pnl_part, left_out


def _format_pct(value: Decimal, *, signed: bool = False) -> str:
    """Ułamek jako procent z jednym miejscem po przecinku („+12,5%”)."""
    return format_pct_pl(value * 100, signed, 1)


def _format_return_pct(return_pct: Decimal | None) -> str:
    """Zwrot procentowy pozycji lub "—" gdy nie da się policzyć (koszt 0)."""
    if return_pct is None:
        return "—"
    return _format_pct(return_pct, signed=True)


def _row_weight(row: PositionPnL, total_market_value: Decimal) -> Decimal:
    """Realna waga wiersza wg wartości rynkowej (0 gdy brak wartości łącznej)."""
    if total_market_value == 0:
        return Decimal("0")
    return row.market_value / total_market_value


def _row_weights(pnl: PortfolioPnL, valuation: PlnPrices | None) -> list[str]:
    """Kolumna „Waga” per wiersz: natywnie bez kursów, w złotych z kursami.

    W złotych lot waży ilość × cena w zł / suma; pozycja bez kursu NBP nie ma
    wagi („—”) i nie wchodzi do mianownika — jak w `Portfolio.weights`."""
    if valuation is None:
        return [_format_pct(_row_weight(row, pnl.total_market_value)) for row in pnl.positions]
    values = [
        None if (price := valuation.prices.get(row.symbol)) is None else row.quantity * price
        for row in pnl.positions
    ]
    total = sum((value for value in values if value is not None), Decimal("0"))
    return [
        "—" if value is None or total == 0 else _format_pct(value / total) for value in values
    ]


def _unrated_note(valuation: PlnPrices | None) -> str:
    """„Bez kursu NBP: SAP.DE — poza wagami i sumami w zł” albo ""."""
    if valuation is None or not valuation.unrated_symbols:
        return ""
    return f"Bez kursu NBP: {', '.join(valuation.unrated_symbols)} — poza wagami i sumami w zł"


def _dominant_cluster(exposures: list[ClusterExposure]) -> ClusterExposure | None:
    """Największa grupa (sektor, ≥2 symbole) przekraczająca próg, albo None.

    `exposures` jest już posortowane malejąco po wadze (kontrakt domeny), więc
    pierwszy kwalifikujący się klaster jest zarazem największy. Klaster
    jednosymbolowy to koncentracja pojedynczej pozycji, nie sektora, więc go
    pomijamy — ostrzeżenie dotyczy kilku pozycji z jednego sektora.
    """
    for exp in exposures:
        if len(exp.symbols) >= 2 and exp.weight >= _CLUSTER_CONCENTRATION_THRESHOLD:
            return exp
    return None


# PORT-06: podpis zmiany portfela, gdy referencja jest z poprzedniego cyklu.
PORTFOLIO_CHANGE_DEFAULT_LABEL = "od ostatniego maila"


def portfolio_change_label(
    since: datetime | None, now: datetime, display_timezone: str
) -> str:
    """PORT-06: „od ostatniego maila” albo — po pauzie (stęchła referencja wg
    domeny, FIN-14) — „od 19.09.2026”, bo wtedy Δ obejmuje wiele dni."""
    if since is not None and is_reference_stale(since, now):
        return f"od {format_local_date(since, display_timezone)}"
    return PORTFOLIO_CHANGE_DEFAULT_LABEL


def _change_amount(change: PortfolioChange, currency: str = DEFAULT_POSITION_CURRENCY) -> str:
    """„+$70,00 (+7,0%)” albo „+280,00 zł (+7,0%)” — kwota i procent zmiany."""
    amount = _format_money(change.change, currency, signed=True)
    return f"{amount} ({_format_pct(change.change_pct, signed=True)})"


def _contributors(change: PortfolioChange, currency: str = DEFAULT_POSITION_CURRENCY) -> str:
    """„NVDA +$75,00, AAPL -$5,00” — top kontrybutorzy w kolejności domeny."""
    return ", ".join(
        f"{c.symbol} {_format_money(c.change, currency, signed=True)}"
        for c in change.top_contributors
    )


def _usd_extra(usd_pln: Decimal | None, currency: str) -> Decimal | None:
    """Dopisek „w zł” po kursie USD/PLN ma sens tylko dla zmiany liczonej w USD."""
    return usd_pln if currency == DEFAULT_POSITION_CURRENCY else None


def _change_details(
    change: PortfolioChange,
    usd_pln: Decimal | None,
    currency: str = DEFAULT_POSITION_CURRENCY,
) -> list[str]:
    """Człony linii zmiany po kwocie: PLN, kontrybutorzy, pozycje bez odniesienia."""
    details: list[str] = []
    extra = _usd_extra(usd_pln, currency)
    if extra is not None:
        details.append(_format_signed_pln(change.change * extra))
    if change.top_contributors:
        details.append(f"najwięcej: {_contributors(change, currency)}")
    if change.unmeasured_symbols:
        details.append(f"bez punktu odniesienia: {', '.join(change.unmeasured_symbols)}")
    return details


def _render_change_html(
    change: PortfolioChange,
    label: str,
    usd_pln: Decimal | None,
    currency: str = DEFAULT_POSITION_CURRENCY,
) -> str:
    color = _pnl_color(change.change)
    details = "".join(
        f" · {_html.escape(detail)}" for detail in _change_details(change, usd_pln, currency)
    )
    return (
        "<div style='padding: 10px 14px; background: #f9fafb; "
        "border-radius: 4px; font-size: 13px; margin-bottom: 12px;'>"
        f"Zmiana {_html.escape(label)}: <strong style='color: {color};'>"
        f"{_change_amount(change, currency)}</strong>{details}</div>"
    )


def _render_change_text(
    change: PortfolioChange,
    label: str,
    usd_pln: Decimal | None,
    currency: str = DEFAULT_POSITION_CURRENCY,
) -> str:
    details = "".join(f" · {detail}" for detail in _change_details(change, usd_pln, currency))
    return f"  Zmiana {label}: {_change_amount(change, currency)}{details}"


def _change_amounts(
    change: PortfolioChange, usd_pln: Decimal | None, currency: str
) -> list[str]:
    amounts = [_format_money(change.change, currency, signed=True)]
    extra = _usd_extra(usd_pln, currency)
    if extra is not None:
        amounts.append(_format_signed_pln(change.change * extra))
    return amounts


def render_portfolio_change_box_html(
    change: PortfolioChange,
    *,
    label: str = PORTFOLIO_CHANGE_DEFAULT_LABEL,
    usd_pln: Decimal | None = None,
    currency: str = DEFAULT_POSITION_CURRENCY,
) -> str:
    """PORT-06: box pierwszego ekranu „Twój portfel od ostatniego maila” —
    procent dużą cyfrą, pod nim kwota (i PLN, gdy znamy kurs). `currency` to
    waluta, w której policzono zmianę (PLN przy kursach NBP)."""
    color = _pnl_color(change.change)
    amounts = _change_amounts(change, usd_pln, currency)
    return (
        '<td style="width: 33%; padding: 12px; background: #f9fafb; '
        f'border-left: 3px solid {color}; border-radius: 4px; vertical-align: top;">'
        '<div style="font-size: 11px; color: #4b5563; text-transform: uppercase;">'
        f"Twój portfel {_html.escape(label)}</div>"
        f'<div style="font-size: 24px; font-weight: 600; color: {color};">'
        f"{_format_pct(change.change_pct, signed=True)}</div>"
        f'<div style="font-size: 12px; color: {color};">{" · ".join(amounts)}</div>'
        "</td>"
    )


def render_portfolio_change_text_line(
    change: PortfolioChange,
    *,
    label: str = PORTFOLIO_CHANGE_DEFAULT_LABEL,
    usd_pln: Decimal | None = None,
    currency: str = DEFAULT_POSITION_CURRENCY,
) -> str:
    """PORT-06: linia pierwszego ekranu wersji tekstowej:
    „Twój portfel od ostatniego maila: +7,0% (+$70,00)”."""
    amounts = _change_amounts(change, usd_pln, currency)
    return (
        f"Twój portfel {label}: {_format_pct(change.change_pct, signed=True)} "
        f"({' · '.join(amounts)})"
    )


def render_user_portfolio_html(
    portfolio: Portfolio,
    current_prices: Mapping[str, Decimal],
    *,
    clusters: Mapping[str, str] | None = None,
    now: datetime,
    usd_pln: Decimal | None = None,
    max_age_days: int = DEFAULT_MAX_POSITION_AGE_DAYS,
    change: PortfolioChange | None = None,
    change_label: str = PORTFOLIO_CHANGE_DEFAULT_LABEL,
    pln_rates: PlnRates | None = None,
    change_currency: str = DEFAULT_POSITION_CURRENCY,
) -> str:
    """Renderuje sekcję HTML portfela lub "" gdy brak policzalnych pozycji."""
    pnl = portfolio.unrealized_pnl(current_prices)
    if not pnl.positions:
        return ""
    pln = portfolio.pln_pnl(current_prices, pln_rates) if pln_rates is not None else None
    valuation = portfolio.pln_prices(current_prices, pln_rates) if pln_rates is not None else None

    exposures = portfolio.cluster_exposure(
        clusters or {}, current_prices if valuation is None else valuation.prices
    )
    freshness = portfolio.freshness(now, max_age_days)

    parts: list[str] = [
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>"
        + "💼 Twój portfel</h2>"
    ]
    if freshness is PortfolioFreshness.STALE:
        parts.append(_render_stale_badge_html(max_age_days))
    parts.append(_render_table_html(pnl, usd_pln, pln, _row_weights(pnl, valuation)))
    unpriced_note = _unpriced_note(portfolio, current_prices, pln_rates)
    for note in (unpriced_note, _unrated_note(valuation)):
        if note:
            parts.append(
                "<p style='font-size: 12px; color: #92400e; margin: 0 0 8px 0;'>"
                f"⚠️ {_html.escape(note)}</p>"
            )
    parts.append(
        _render_totals_html(
            pnl, usd_pln, partial=bool(unpriced_note), pln=pln, pln_rates=pln_rates
        )
    )
    if change is not None:
        parts.append(_render_change_html(change, change_label, usd_pln, change_currency))

    cluster = _dominant_cluster(exposures)
    if cluster is not None:
        parts.append(_render_cluster_warning_html(cluster))

    return "".join(parts)


def _render_stale_badge_html(max_age_days: int) -> str:
    return (
        "<div style='display: inline-block; padding: 4px 10px; "
        "background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; "
        "border-radius: 4px; font-size: 12px; font-weight: 600; "
        "margin-bottom: 8px;'>"
        f"⚠️ STALE — pozycje starsze niż {max_age_days} dni; "
        "P/L może być nieaktualny</div>"
    )


_TH_RIGHT = "<th scope='col' style='padding: 6px 8px; text-align: right;'>"


def _render_table_html(
    pnl: PortfolioPnL,
    usd_pln: Decimal | None,
    pln: PortfolioPlnPnL | None,
    weights: list[str],
) -> str:
    if pln is not None:
        pln_header = f"{_TH_RIGHT}P/L zł</th>{_TH_RIGHT}w tym kurs</th>"
        rows = [
            _render_row_html(row, weight, None, pln_row)
            for row, weight, pln_row in zip(pnl.positions, weights, pln.positions, strict=True)
        ]
    else:
        pln_header = f"{_TH_RIGHT}P/L (PLN)</th>" if usd_pln is not None else ""
        rows = [
            _render_row_html(row, weight, usd_pln)
            for row, weight in zip(pnl.positions, weights, strict=True)
        ]
    return (
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 8px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Symbol</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: right;'>Waga</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: right;'>Wartość</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: right;'>P/L</th>"
        f"{pln_header}"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_row_html(
    row: PositionPnL,
    weight: str,
    usd_pln: Decimal | None,
    pln_row: PositionPlnPnL | None = None,
) -> str:
    symbol = _html.escape(row.symbol)
    color = _pnl_color(row.unrealized_pnl)
    pnl_cell = (
        f"{_format_money(row.unrealized_pnl, row.currency, signed=True)} "
        f"({_format_return_pct(row.return_pct)})"
    )
    if pln_row is not None:
        pln_cell = _render_pln_cells_html(pln_row)
    elif usd_pln is not None:
        pln_cell = (
            "<td style='padding: 6px 8px; text-align: right; "
            f"color: {color};'>"
            f"{_format_signed_pln(row.unrealized_pnl * usd_pln)}</td>"
        )
    else:
        pln_cell = ""
    return (
        "<tr>"
        f"<td style='padding: 6px 8px;'><strong>{symbol}</strong></td>"
        f"<td style='padding: 6px 8px; text-align: right; color: #6b7280;'>"
        f"{weight}</td>"
        f"<td style='padding: 6px 8px; text-align: right;'>"
        f"{_format_money(row.market_value, row.currency)}</td>"
        f"<td style='padding: 6px 8px; text-align: right; color: {color}; "
        f"font-weight: 600;'>{pnl_cell}</td>"
        f"{pln_cell}"
        "</tr>"
    )


def _render_pln_cells_html(row: PositionPlnPnL) -> str:
    """PORT-11: komórki „P/L zł” i „w tym kurs”; brak kursu → szara notka."""
    pnl_color = "#6b7280" if row.pnl_pln is None else _pnl_color(row.pnl_pln)
    fx_style = (
        "color: #6b7280; font-size: 12px;"
        if row.fx_pln is None
        else f"color: {_pnl_color(row.fx_pln)};"
    )
    return (
        f"<td style='padding: 6px 8px; text-align: right; color: {pnl_color};'>"
        f"{_pln_pnl_cell(row)}</td>"
        f"<td style='padding: 6px 8px; text-align: right; {fx_style}'>"
        f"{_html.escape(_fx_cell(row))}</td>"
    )


_PARTIAL_TOTALS_LABEL = "bez pozycji niewycenionych"


def _unpriced_note(
    portfolio: Portfolio,
    current_prices: Mapping[str, Decimal],
    rates: PlnRates | None = None,
) -> str:
    """PORT-03: „Bez wyceny w tym cyklu: A, B (≈ X% kosztu)” albo "" gdy wszystko
    wycenione."""
    symbols = portfolio.unpriced_symbols(current_prices)
    if not symbols:
        return ""
    note = f"Bez wyceny w tym cyklu: {', '.join(symbols)}"
    share = portfolio.unpriced_cost_share(current_prices, rates=rates)
    if share is not None:
        note += f" (≈ {_format_pct(share)} kosztu)"
    return note


def _totals_label(partial: bool) -> str:
    return f"Łącznie ({_PARTIAL_TOTALS_LABEL})" if partial else "Łącznie"


def _render_totals_html(
    pnl: PortfolioPnL,
    usd_pln: Decimal | None,
    *,
    partial: bool = False,
    pln: PortfolioPlnPnL | None = None,
    pln_rates: PlnRates | None = None,
) -> str:
    color = _pnl_color(pnl.total_unrealized_pnl)
    currency = _single_currency(pnl)
    if currency is None:
        total_line = (
            f"{_totals_label(partial)}: pozycje w różnych walutach "
            f"({_html.escape(_currencies_label(pnl))})"
        )
    else:
        total_line = (
            f"{_totals_label(partial)}: "
            f"<strong>{_format_money(pnl.total_market_value, currency)}</strong> · "
            f"P/L <strong style='color: {color};'>"
            f"{_format_money(pnl.total_unrealized_pnl, currency, signed=True)} "
            f"({_format_return_pct(pnl.total_return_pct)})</strong>"
        )
    pln_line = ""
    if pln is not None and pln_rates is not None:
        pln_line = _render_pln_totals_html(pln, pln_rates)
    elif usd_pln is not None:
        pln_line = (
            "<br><span style='color: #6b7280;'>"
            f"W PLN (kurs USD/PLN {number_exact(usd_pln)}): "
            f"wartość {_format_pln(pnl.total_market_value * usd_pln)}, "
            f"P/L <span style='color: {color};'>"
            f"{_format_signed_pln(pnl.total_unrealized_pnl * usd_pln)}</span>"
            "</span>"
        )
    return (
        "<div style='padding: 10px 14px; background: #f9fafb; "
        "border-radius: 4px; font-size: 13px; margin-bottom: 12px;'>"
        f"{total_line}{pln_line}</div>"
    )


def _render_pln_totals_html(pln: PortfolioPlnPnL, rates: PlnRates) -> str:
    head, pnl_part, left_out = _pln_totals(pln, rates)
    body = _html.escape(head)
    if pnl_part:
        body += (
            f", <span style='color: {_pnl_color(pln.total_pnl_pln)};'>"
            f"{_html.escape(pnl_part)}</span>"
        )
    if left_out:
        body += f"; {_html.escape(left_out)}"
    return f"<br><span style='color: #6b7280;'>{body}</span>"


def _cluster_warning_message(cluster: ClusterExposure) -> str:
    """Wspólna treść ostrzeżenia o klastrze (HTML i tekst różnią się opakowaniem)."""
    count = len(cluster.symbols)
    symbols = ", ".join(cluster.symbols)
    pct = _format_pct(cluster.weight)
    return (
        f"Twoje {count} pozycji ({symbols}) to jeden sektor "
        f"«{cluster.cluster}» — {pct} kapitału."
    )


def _render_cluster_warning_html(cluster: ClusterExposure) -> str:
    message = _html.escape(_cluster_warning_message(cluster))
    return (
        "<div style='padding: 10px 14px; background: #fffbeb; "
        "border-left: 3px solid #d97706; border-radius: 4px; "
        "font-size: 13px; margin-bottom: 16px; color: #92400e;'>"
        f"⚠️ {message}</div>"
    )


def render_user_portfolio_text(
    portfolio: Portfolio,
    current_prices: Mapping[str, Decimal],
    *,
    clusters: Mapping[str, str] | None = None,
    now: datetime,
    usd_pln: Decimal | None = None,
    max_age_days: int = DEFAULT_MAX_POSITION_AGE_DAYS,
    change: PortfolioChange | None = None,
    change_label: str = PORTFOLIO_CHANGE_DEFAULT_LABEL,
    pln_rates: PlnRates | None = None,
    change_currency: str = DEFAULT_POSITION_CURRENCY,
) -> str:
    """Renderuje sekcję plain-text portfela lub "" gdy brak policzalnych pozycji."""
    pnl = portfolio.unrealized_pnl(current_prices)
    if not pnl.positions:
        return ""
    pln = portfolio.pln_pnl(current_prices, pln_rates) if pln_rates is not None else None
    pln_rows = pln.positions if pln is not None else (None,) * len(pnl.positions)
    valuation = portfolio.pln_prices(current_prices, pln_rates) if pln_rates is not None else None

    exposures = portfolio.cluster_exposure(
        clusters or {}, current_prices if valuation is None else valuation.prices
    )
    freshness = portfolio.freshness(now, max_age_days)

    lines: list[str] = ["=== 💼 Twój portfel ==="]
    if freshness is PortfolioFreshness.STALE:
        lines.append(
            f"[STALE] Pozycje starsze niż {max_age_days} dni — "
            "P/L może być nieaktualny."
        )
    lines.append("")

    weights = _row_weights(pnl, valuation)
    for row, weight, pln_row in zip(pnl.positions, weights, pln_rows, strict=True):
        line = (
            f"  {row.symbol:8s} "
            f"waga {weight:>6s} "
            f"wart. {_format_money(row.market_value, row.currency):>14s} "
            f"P/L {_format_money(row.unrealized_pnl, row.currency, signed=True):>12s} "
            f"({_format_return_pct(row.return_pct)})"
        )
        if pln_row is not None:
            line += f" | P/L zł {_pln_pnl_cell(pln_row)} ({_fx_text(pln_row)})"
        elif usd_pln is not None:
            line += f" | {_format_signed_pln(row.unrealized_pnl * usd_pln)}"
        lines.append(line)

    unpriced_note = _unpriced_note(portfolio, current_prices, pln_rates)
    for note in (unpriced_note, _unrated_note(valuation)):
        if note:
            lines.append(f"  ⚠️ {note}")

    lines.append("")
    lines.append(_totals_text(pnl, partial=bool(unpriced_note)))
    if pln is not None and pln_rates is not None:
        head, pnl_part, left_out = _pln_totals(pln, pln_rates)
        lines.append(
            f"  {head}"
            + (f", {pnl_part}" if pnl_part else "")
            + (f"; {left_out}" if left_out else "")
        )
    elif usd_pln is not None:
        lines.append(
            f"  W PLN (USD/PLN {number_exact(usd_pln)}): "
            f"wartość {_format_pln(pnl.total_market_value * usd_pln)}, "
            f"P/L {_format_signed_pln(pnl.total_unrealized_pnl * usd_pln)}"
        )
    if change is not None:
        lines.append(_render_change_text(change, change_label, usd_pln, change_currency))

    cluster = _dominant_cluster(exposures)
    if cluster is not None:
        lines.append("")
        lines.append(f"  ⚠️ {_cluster_warning_message(cluster)}")

    return "\n".join(lines)


def _fx_text(row: PositionPlnPnL) -> str:
    """Wersja tekstowa „w tym kurs”: kwota z prefiksem albo sama notka o braku."""
    return _gap_note(row) if row.fx_pln is None else f"w tym kurs {_fx_cell(row)}"


def _totals_text(pnl: PortfolioPnL, *, partial: bool) -> str:
    currency = _single_currency(pnl)
    if currency is None:
        return f"  {_totals_label(partial)}: pozycje w różnych walutach ({_currencies_label(pnl)})"
    return (
        f"  {_totals_label(partial)}: {_format_money(pnl.total_market_value, currency)} · "
        f"P/L {_format_money(pnl.total_unrealized_pnl, currency, signed=True)} "
        f"({_format_return_pct(pnl.total_return_pct)})"
    )


# ---------------------------------------------------------------------------
# PORT-06: zmiana portfela od ostatniego maila — sekcja 💼 i pierwszy ekran
# ---------------------------------------------------------------------------


def _previous_prices_of(results: list[SymbolResult]) -> dict[str, Decimal]:
    """PORT-06: cena referencji odtworzona z delty cyklu: prev = current / (1 + Δ).

    Działa dla każdego symbolu z ceną i deltą — także pominiętego bramką albo
    z referencją sprzed pauzy (FIN-14). Cold-start nie ma referencji (Δ = 0 to
    brak punktu odniesienia, nie „bez zmian”), więc nie wchodzi."""
    previous: dict[str, Decimal] = {}
    for r in results:
        if r.current_price is None or r.delta is None:
            continue
        if r.skip_reason is SkipReason.COLD_START or r.delta == -1:
            continue
        previous[r.symbol] = r.current_price / (1 + r.delta)
    return previous


@dataclasses.dataclass(frozen=True)
class PortfolioChangeView:
    """PORT-06: zmiana portfela z domeny + podpis okna („od ostatniego maila”)."""

    change: PortfolioChange
    label: str
    currency: str = DEFAULT_POSITION_CURRENCY


def portfolio_change_view(
    results: list[SymbolResult],
    portfolio: Portfolio | None,
    current_prices: Mapping[str, Decimal],
    *,
    now: datetime,
    display_timezone: str,
    rates: PlnRates | None = None,
) -> PortfolioChangeView | None:
    """PORT-06: Δ portfela od referencji cyklu. Okno zaczyna się od NAJSTARSZEJ
    referencji zmierzonych pozycji — po pauzie podpis mówi „od <data>”.
    Brak portfela (np. obcy subskrybent, BEZ-01) → None, sekcje się chowają.

    Z kursami NBP zmiana jest w złotych: obie ceny po BIEŻĄCYM kursie waluty
    pozycji (ruch kursu między mailami nie wchodzi), a pozycja bez kursu wypada
    z obu stron — sekcja 💼 wymienia ją w nocie „Bez kursu NBP”."""
    if portfolio is None:
        return None
    previous = _previous_prices_of(results)
    currency = DEFAULT_POSITION_CURRENCY
    if rates is not None:
        valuation = portfolio.pln_prices(current_prices, rates)
        portfolio = portfolio.excluding(valuation.unrated_symbols)
        previous = dict(portfolio.pln_prices(previous, rates).prices)
        current_prices = valuation.prices
        currency = HOME_CURRENCY
    change = portfolio.change_since(previous, current_prices)
    if change is None:
        return None
    measured = {p.symbol for p in portfolio.positions} - set(change.unmeasured_symbols)
    references = [
        r.reference_at for r in results if r.symbol in measured and r.reference_at is not None
    ]
    since = min(references) if references else None
    return PortfolioChangeView(
        change=change,
        label=portfolio_change_label(since, now, display_timezone),
        currency=currency,
    )


def _change_kwargs(change: PortfolioChangeView | None) -> dict[str, Any]:
    """PORT-06: opcjonalne argumenty linii zmiany dla rendererów sekcji 💼."""
    if change is None:
        return {}
    return {
        "change": change.change,
        "change_label": change.label,
        "change_currency": change.currency,
    }


# ---------------------------------------------------------------------------
# Sekcja 💼 Twój portfel (ARCH-12)
# ---------------------------------------------------------------------------


def _section_html(data: SectionInput) -> str:
    """#15: sekcja realnego portfela. Kursy przychodzą jako wartości (PORT-11:
    `pln_rates`, inaczej USD/PLN z `MacroRiskReport`), nie z portu. Brak
    portfela → "" (sekcja sama się chowa). PORT-06: linia zmiany od ostatniego
    maila, gdy da się ją policzyć."""
    context = data.context
    if context.user_portfolio is None:
        return ""
    return render_user_portfolio_html(
        context.user_portfolio,
        data.current_prices,
        clusters=context.portfolio_clusters,
        now=data.started_at,
        usd_pln=data.usd_pln,
        pln_rates=data.pln_rates,
        **_change_kwargs(data.portfolio_change),
    )


def _section_text(data: SectionInput) -> list[str]:
    """Wariant plain-text sekcji portfela (decyzja 2026-10-19) — te same dane co
    HTML. Bez `user_portfolio_in_text` (stary push) tekst nie ma kwot portfela."""
    context = data.context
    if context.user_portfolio is None or not context.user_portfolio_in_text:
        return []
    text = render_user_portfolio_text(
        context.user_portfolio,
        data.current_prices,
        clusters=context.portfolio_clusters,
        now=data.started_at,
        usd_pln=data.usd_pln,
        pln_rates=data.pln_rates,
        **_change_kwargs(data.portfolio_change),
    )
    return text_block(text, spaced=True)


# #15: 💼 realny portfel (wagi, P/L, sektor, PLN) — zaraz za pierwszym ekranem.
USER_PORTFOLIO_SECTION = ReportSection(
    key="user_portfolio", render_html=_section_html, render_text=_section_text
)
