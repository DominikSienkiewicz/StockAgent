from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.domain import polish_labels
from src.domain.number_format import format_money_pl, format_number_pl, format_pct_pl
from src.domain.provenance import ProvenanceBadge, ProvenanceLevel

COMPANY_NAMES: dict[str, str] = {
    # Mega-cap tech
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "AVGO": "Broadcom Inc.",
    "ORCL": "Oracle Corporation",
    # Semi / hardware
    "AMD": "Advanced Micro Devices Inc.",
    "INTC": "Intel Corporation",
    "QCOM": "Qualcomm Inc.",
    "TXN": "Texas Instruments Inc.",
    "ASML": "ASML Holding N.V.",
    "TSM": "Taiwan Semiconductor Mfg. Co.",
    "MU": "Micron Technology Inc.",
    "AMAT": "Applied Materials Inc.",
    "KLAC": "KLA Corporation",
    "LRCX": "Lam Research Corporation",
    "SNDK": "Sandisk Corporation",
    "SSNLF": "Samsung Electronics Co. Ltd.",
    # Software / cloud
    "CRM": "Salesforce Inc.",
    "ADBE": "Adobe Inc.",
    "NOW": "ServiceNow Inc.",
    "SNOW": "Snowflake Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "PANW": "Palo Alto Networks Inc.",
    "CRWD": "CrowdStrike Holdings Inc.",
    "NET": "Cloudflare Inc.",
    "MDB": "MongoDB Inc.",
    "DDOG": "Datadog Inc.",
    "TEAM": "Atlassian Corporation",
    "FROG": "JFrog Ltd.",
    # Cybersecurity
    "OKTA": "Okta Inc.",
    "S": "SentinelOne Inc.",
    "SAIL": "SailPoint Inc.",
    # Enterprise / legacy tech
    "IBM": "IBM Corp.",
    "SAP": "SAP SE",
    "CSCO": "Cisco Systems Inc.",
    "HPQ": "HP Inc.",
    "DELL": "Dell Technologies Inc.",
    # Finance
    "JPM": "JPMorgan Chase & Co.",
    "GS": "Goldman Sachs Group Inc.",
    "MS": "Morgan Stanley",
    "BAC": "Bank of America Corp.",
    "C": "Citigroup Inc.",
    "WFC": "Wells Fargo & Co.",
    "BLK": "BlackRock Inc.",
    "V": "Visa Inc.",
    "MA": "Mastercard Inc.",
    "PYPL": "PayPal Holdings Inc.",
    "AXP": "American Express Co.",
    "SQ": "Block Inc.",
    # Healthcare
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Inc.",
    "LLY": "Eli Lilly and Co.",
    "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co. Inc.",
    "PFE": "Pfizer Inc.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "ABT": "Abbott Laboratories",
    "NVO": "Novo Nordisk A/S",
    # Consumer
    "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale Corp.",
    "HD": "The Home Depot Inc.",
    "NKE": "Nike Inc.",
    "MCD": "McDonald's Corp.",
    "SBUX": "Starbucks Corp.",
    "DIS": "The Walt Disney Co.",
    "NFLX": "Netflix Inc.",
    "CMCSA": "Comcast Corp.",
    # Energy
    "XOM": "Exxon Mobil Corp.",
    "CVX": "Chevron Corp.",
    "COP": "ConocoPhillips",
    # Telecom
    "T": "AT&T Inc.",
    "VZ": "Verizon Communications Inc.",
    # Other large-cap
    "BRK.B": "Berkshire Hathaway Inc.",
    "BRK-B": "Berkshire Hathaway Inc.",
    "PG": "Procter & Gamble Co.",
    "KO": "The Coca-Cola Co.",
    "PEP": "PepsiCo Inc.",
    "PM": "Philip Morris International Inc.",
    "RTX": "RTX Corporation",
    "LMT": "Lockheed Martin Corp.",
    "CAT": "Caterpillar Inc.",
    "DE": "Deere & Company",
    "BA": "The Boeing Co.",
    "GE": "GE Aerospace",
    "MMM": "3M Company",
    "SIEGY": "Siemens AG",
    "UBER": "Uber Technologies Inc.",
    # ETFs
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "GLD": "SPDR Gold Shares",
    "VOO": "Vanguard S&P 500 ETF",
    "IVV": "iShares Core S&P 500 ETF",
    "VT": "Vanguard Total World Stock ETF",
    "VB": "Vanguard Small-Cap ETF",
    "QUAL": "iShares MSCI USA Quality Factor ETF",
    "IHI": "iShares U.S. Medical Devices ETF",
    "EWY": "iShares MSCI South Korea ETF",
    "CSPX.L": "iShares Core S&P 500 UCITS ETF",
    "XDWD.DE": "Xtrackers MSCI World UCITS ETF",
    "IUSN.DE": "iShares MSCI World Small Cap UCITS ETF",
    # Risk Watch — proxy suwerenne, ETF-y odwrotne i zmienności
    "EPOL": "iShares MSCI Poland ETF",
    "SH": "ProShares Short S&P 500",
    "PSQ": "ProShares Short QQQ",
    "RWM": "ProShares Short Russell 2000",
    "EUM": "ProShares Short MSCI Emerging Markets",
    "SQQQ": "ProShares UltraPro Short QQQ",
    "TBT": "ProShares UltraShort 20+ Year Treasury",
    "VIXY": "ProShares VIX Short-Term Futures ETF",
    "UVXY": "ProShares Ultra VIX Short-Term Futures ETF",
    # Crypto (tickery używane przez niektóre feedy)
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "BNB": "BNB",
    "SOL": "Solana",
    "XRP": "XRP",
    # Inne często używane w testach
    "ASMIY": "ASM International N.V.",
}


def company_label(symbol: str) -> str:
    """Zwraca 'SYMBOL (Nazwa Spółki)' lub samo 'SYMBOL' gdy nazwa nieznana."""
    if not symbol:
        return symbol
    name = COMPANY_NAMES.get(symbol)
    if name:
        return f"{symbol} ({name})"
    return symbol


# Mapowanie symbol → sektor (PL), używane w sekcji PREDYKCJE maila. Membership
# w słowniku pełni też rolę klasyfikacji klasy aktywa: ETF-y → "ETF",
# krypto → "Krypto", akcje → konkretny sektor. Symbol spoza słownika nie
# dostaje etykiety (po cichu, tak jak company_label pomija nieznaną nazwę).
# Pokrywa domyślne portfolio (43) + krypto; rozszerzaj wraz z SYMBOLS.
# Klasyfikacja zgrupowana per sektor — każda nazwa sektora pada raz (łatwiej
# rozszerzać niż płaską mapę symbol→sektor z powtarzaną etykietą). `SECTORS`
# spłaszczamy do symbol→sektor dla szybkiego lookupu.
_SECTOR_SYMBOLS: dict[str, list[str]] = {
    "Big Tech": ["AAPL", "AMZN", "GOOGL", "GOOG", "META"],
    "Półprzewodniki/AI": [
        "NVDA", "AMD", "TSM", "ASML", "ASMIY", "MU", "QCOM", "INTC", "SNDK",
    ],
    "Chmura/Software": [
        "MSFT", "ORCL", "NET", "SAP", "PLTR", "IBM", "TEAM", "FROG", "SNOW", "DDOG",
    ],
    "Cyberbezpieczeństwo": ["CRWD", "PANW", "OKTA", "S", "SAIL"],
    "Hardware": ["DELL", "SSNLF"],
    "Mobilność": ["TSLA", "UBER"],
    "Przemysł/Pharma": ["SIEGY", "NVO"],
    "Finanse": ["BLK"],
    "ETF": ["VT", "QUAL", "IHI", "VB", "EWY", "IVV", "XDWD.DE", "IUSN.DE"],
    "Krypto": ["BTC", "ETH"],
}
SECTORS: dict[str, str] = {
    symbol: sector
    for sector, symbols in _SECTOR_SYMBOLS.items()
    for symbol in symbols
}


# Kuratorowana pula peerów per sektor akcji — nadzbiór monitorowanych. Służy
# sekcji 💡 WARTE UWAGI: gdy sektor jest "gorący" w bieżącym cyklu, sugerujemy
# stąd tickery, których jeszcze nie monitorujemy. Tylko sektory akcji
# (ETF/Krypto mają inną dynamikę i nie są tu uwzględniane).
PEERS: dict[str, list[str]] = {
    "Big Tech": ["NFLX", "DIS"],
    "Półprzewodniki/AI": ["AVGO", "TXN", "AMAT", "KLAC", "LRCX", "ARM", "MRVL"],
    "Chmura/Software": ["CRM", "ADBE", "NOW", "MDB", "WDAY", "HUBS"],
    "Cyberbezpieczeństwo": ["ZS", "FTNT", "CYBR", "TENB", "QLYS"],
    "Hardware": ["HPQ", "HPE", "WDC", "STX", "SMCI", "LOGI"],
    "Mobilność": ["LYFT", "RIVN", "LCID", "GM", "F"],
    "Przemysł/Pharma": ["LLY", "MRK", "PFE", "ABBV", "NVS"],
    "Finanse": ["JPM", "GS", "MS", "V", "MA"],
}


def sector_label(symbol: str) -> str | None:
    """Zwraca sektor (PL) dla symbolu lub None, gdy nieznany.

    None → renderer pomija etykietę (brak '—'), analogicznie do company_label.
    """
    if not symbol:
        return None
    return SECTORS.get(symbol)


def company_label_with_sector(symbol: str) -> str:
    """'SYMBOL (Nazwa) · Sektor' lub 'SYMBOL (Nazwa)' gdy sektor nieznany.

    Używane w sekcji PREDYKCJE — sektor doklejony przy nazwie spółki.
    """
    label = company_label(symbol)
    sector = sector_label(symbol)
    return f"{label} · {sector}" if sector else label


_SENTIMENT_PL = {
    "Bullish": "Pozytywny",
    "Somewhat-Bullish": "Lekko pozytywny",
    "Neutral": "Neutralny",
    "Somewhat-Bearish": "Lekko negatywny",
    "Bearish": "Negatywny",
}


def trend_label(trend: str | None) -> str:
    """Nazwa trendu ze wspólnego słownika domeny (MAIL-15)."""
    return polish_labels.trend_label(trend)


def sentiment_label(label: str | None) -> str:
    return _SENTIMENT_PL.get(label or "", label or "—")


# MAIL-08 (G21): zastrzeżenie inwestycyjne. Raport podaje sygnały modelu i bandy
# wielkości pozycji, więc każdy kanał, który je pokazuje, niesie też to zdanie —
# bezwarunkowo, poza sekcjami, które same się chowają.
INVESTMENT_DISCLAIMER = (
    "Zastrzeżenie: raport generuje automatycznie model statystyczny wspierany "
    "przez LLM, wyłącznie w celach informacyjnych i edukacyjnych. Treść nie "
    "stanowi rekomendacji inwestycyjnej ani porady inwestycyjnej, podatkowej lub "
    "prawnej i nie jest ofertą kupna ani sprzedaży instrumentów finansowych. "
    "Sygnały i prognozy modelu mogą być błędne, a historyczna trafność nie "
    "gwarantuje przyszłych wyników. Decyzje inwestycyjne podejmujesz na własne ryzyko."
)
# Wariant pushu: ma się zmieścić w budżecie digestu (MAX_DIGEST_CHARS).
MESSENGER_DISCLAIMER = "ℹ️ Sygnały modelu nie stanowią rekomendacji inwestycyjnej."


# MAIL-14: strefa czasu w temacie, nagłówku maila i pushu. Logi zostają w UTC.
DEFAULT_DISPLAY_TIMEZONE = "Europe/Warsaw"

_WEEKDAYS_PL = ("pon", "wt", "śr", "czw", "pt", "sob", "nd")


def format_local(moment: datetime, tz: str) -> str:
    """Chwila w strefie `tz` (nazwa IANA) jako np. „pt 16.10 07:30”.

    Naiwny `datetime` traktujemy jako UTC — tak zapisuje czas całe repo, a odczyt
    jako czasu lokalnego maszyny zależałby od strefy runnera.
    """
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    local = aware.astimezone(ZoneInfo(tz))
    return f"{_WEEKDAYS_PL[local.weekday()]} {local:%d.%m %H:%M}"


def format_local_date(moment: datetime, tz: str) -> str:
    """Dzień chwili w strefie `tz` jako „19.09.2026” (naiwny `datetime` = UTC,
    jak w `format_local`). Rok jest jawny: daty sprzed pauzy bywają odległe."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return f"{aware.astimezone(ZoneInfo(tz)):%d.%m.%Y}"


def _decimal(value: Decimal | float) -> Decimal:
    """Float ze statystyk → Decimal przez najkrótszy zapis (0.0125 → „0.0125”), żeby
    remis zaokrąglał się jak w zapisie dziesiętnym, a nie jak w binarnym."""
    return value if isinstance(value, Decimal) else Decimal(str(value))


def pct(value: Decimal | float | None, signed: bool = False, decimals: int = 2) -> str:
    """Ułamek jako procent po polsku (0.0123 → „1,23%”), przez `format_pct_pl`.

    ARCH-28: jeden formater procentów sekcji maila — przecinek, ROUND_HALF_UP,
    minus ASCII, zero bez minusa. None → „—”.
    """
    if value is None:
        return "—"
    return format_pct_pl(_decimal(value) * 100, signed, decimals)


def direction_pct(value: Decimal | float | None, decimals: int = 2) -> str:
    """MAIL-57: zmiana ze strzałką i znakiem („▲ +0,42%”, „▼ -0,42%”) — kierunek
    czytelny bez koloru. Ruch zaokrąglony do zera nie dostaje strzałki. None → „—”."""
    if value is None:
        return "—"
    text = pct(value, signed=True, decimals=decimals)
    if text == pct(0, signed=True, decimals=decimals):
        return text
    return f"{'▲' if value > 0 else '▼'} {text}"


def pct_range(low: Decimal | float, high: Decimal | float, decimals: int = 0) -> str:
    """Przedział ułamków jako procenty z jednym „%” na końcu: „41–63%”."""
    return f"{pct(low, decimals=decimals).removesuffix('%')}–{pct(high, decimals=decimals)}"


def number(value: Decimal | float | None, decimals: int, signed: bool = False) -> str:
    """Liczba po polsku (sentyment „+0,42”, kurs „4,27”), przez `format_number_pl`."""
    if value is None:
        return "—"
    return format_number_pl(_decimal(value), signed, decimals)


@dataclass(frozen=True, slots=True)
class NumberStyle:
    """Zapis liczb w etykietach współdzielonych przez mail i publiczny scorecard.

    MAIL-58: mail pisze po polsku (`POLISH_NUMBERS`), a publiczny scorecard
    zostaje przy kropce i dotychczasowym zaokrąglaniu (`DOT_NUMBERS`).
    """

    polish: bool

    @property
    def decimal_separator(self) -> str:
        return "," if self.polish else "."

    def pct(self, fraction: float, *, decimals: int, signed: bool = False) -> str:
        """Ułamek jako procent: „52,0%” (polski) albo „52.0%” (kropka)."""
        if self.polish:
            return pct(fraction, signed=signed, decimals=decimals)
        # `+ 0.0` zamienia ujemne zero z zaokrąglenia na dodatnie — „+0.0%”.
        rounded = round(fraction * 100, decimals) + 0.0
        sign = "+" if signed else ""
        return f"{rounded:{sign}.{decimals}f}%"

    def pct_range(self, low: float, high: float, *, decimals: int = 0) -> str:
        """Przedział procentów z jednym „%” na końcu: „41–63%”."""
        low_pct = self.pct(low, decimals=decimals).removesuffix("%")
        return f"{low_pct}–{self.pct(high, decimals=decimals)}"

    def number(self, value: float, *, decimals: int) -> str:
        """Liczba: „0,231” (polski) albo „0.231” (kropka)."""
        if self.polish:
            return number(value, decimals=decimals)
        return f"{value:.{decimals}f}"


POLISH_NUMBERS = NumberStyle(polish=True)
DOT_NUMBERS = NumberStyle(polish=False)


def plural_pl(n: int, one: str, few: str, many: str) -> str:
    """Polska forma rzeczownika po liczebniku `n` (ARCH-28).

    1 → `one`; końcówka 2–4 poza nastkami 12–14 (2, 22, 104) → `few`; reszta
    (0, 5–21, 25, 112) → `many`. Liczy się wartość bezwzględna.
    """
    count = abs(n)
    if count == 1:
        return one
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


@dataclass(frozen=True, slots=True)
class ReportPalette:
    """Kolory raportu w jednym miejscu (ARCH-28); sekcje przechodzą na nie stopniowo.

    MAIL-57: kolory tekstu mają mieć ≥ 4.5:1 na białym i na `surface` (WCAG AA dla
    tekstu 11–13 px, `contrast_ratio`). `caution` i `subtle` ściemnione z #ca8a04
    (2.9:1) i #9ca3af (2.5:1) — jaśniejszy szary tekst niż `muted` nie przechodzi
    AA, więc `subtle` mu równy. `positive` (#16a34a, 3.3:1) czeka na decyzję
    właściciela (test ARCH-28 przypina obecną wartość)."""

    positive: str = "#16a34a"
    negative: str = "#dc2626"
    neutral: str = "#737373"
    caution: str = "#a16207"
    text: str = "#111827"
    text_secondary: str = "#374151"
    muted: str = "#6b7280"
    subtle: str = "#6b7280"
    border: str = "#e5e7eb"
    surface: str = "#f9fafb"


PALETTE = ReportPalette()


def _relative_luminance(color: str) -> float:
    """Luminancja względna sRGB koloru `#rrggbb` albo `#rgb` (WCAG 2.x)."""
    hex_digits = color.removeprefix("#")
    if len(hex_digits) == 3:
        hex_digits = "".join(digit * 2 for digit in hex_digits)
    channels = [int(hex_digits[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """MAIL-57: kontrast dwóch kolorów wg WCAG 2.x — od 1 (ten sam kolor) do 21
    (czarny na białym). Kolejność argumentów nie ma znaczenia."""
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def number_exact(value: Decimal) -> str:
    """Decimal po polsku z jego własną precyzją: kurs NBP „4,2710”, cena „186,00”."""
    exponent = value.as_tuple().exponent
    decimals = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return format_number_pl(value, False, decimals)


def money(value: Decimal | None, signed: bool = False) -> str:
    """Kwota w USD po polsku („$1 234,50”), przez `format_money_pl`. None → „—”."""
    if value is None:
        return "—"
    return format_money_pl(value, "USD", signed=signed)


def trend_color(trend: str | None) -> str:
    return {
        "BULLISH": PALETTE.positive,
        "BEARISH": PALETTE.negative,
    }.get(trend or "", PALETTE.neutral)


def delta_color(delta: Decimal | None) -> str:
    if delta is None or delta == 0:
        return PALETTE.neutral
    return PALETTE.positive if delta > 0 else PALETTE.negative


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


# Q6: kolor i tło chipa proweniencji per poziom. FRESH celowo nie ma chipa
# (brak szumu), więc nie ma go tu — render pomija FRESH.
_PROVENANCE_STYLE: dict[ProvenanceLevel, tuple[str, str]] = {
    # (kolor tekstu/obwódki, tło)
    ProvenanceLevel.STALE: ("#92400e", "#fef3c7"),     # bursztynowy — cache
    ProvenanceLevel.DEGRADED: ("#991b1b", "#fef2f2"),  # czerwony — skażone
    ProvenanceLevel.FLAGGED: ("#1e40af", "#eff6ff"),   # niebieski — flagi
}


def provenance_badges_html(badges: Sequence[ProvenanceBadge]) -> str:
    """Renderuje listę odznak proweniencji jako inline'owe chipy HTML.

    FRESH jest pomijany (oznacza "wszystko OK" — chip byłby tylko szumem).
    Etykieta i detail mogą pochodzić z danych zewnętrznych (degraded_reason,
    nazwy flag), więc OBA przechodzą przez `html.escape` — chip nie może być
    sinkiem XSS. Pusta lista / same FRESH → pusty string (renderer pomija blok).
    """
    chips: list[str] = []
    for badge in badges:
        if badge.level is ProvenanceLevel.FRESH:
            continue
        color, bg = _PROVENANCE_STYLE.get(badge.level, ("#4b5563", "#f3f4f6"))
        label = html.escape(badge.label, quote=True)
        title_attr = (
            f' title="{html.escape(badge.detail, quote=True)}"'
            if badge.detail
            else ""
        )
        chips.append(
            f"<span{title_attr} style=\"display: inline-block; padding: 1px 6px; "
            f"margin: 0 2px; font-size: 10px; font-weight: 600; border-radius: 3px; "
            f"color: {color}; background: {bg};\">{label}</span>"
        )
    return "".join(chips)
