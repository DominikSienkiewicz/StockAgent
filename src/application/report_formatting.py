from __future__ import annotations

import html
from collections.abc import Sequence
from decimal import Decimal

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
    # ETFs
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "GLD": "SPDR Gold Shares",
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
SECTORS: dict[str, str] = {
    # Big Tech
    "AAPL": "Big Tech",
    "AMZN": "Big Tech",
    "GOOGL": "Big Tech",
    "GOOG": "Big Tech",
    "META": "Big Tech",
    # Półprzewodniki / AI
    "NVDA": "Półprzewodniki/AI",
    "AMD": "Półprzewodniki/AI",
    "TSM": "Półprzewodniki/AI",
    "ASML": "Półprzewodniki/AI",
    "ASMIY": "Półprzewodniki/AI",
    "MU": "Półprzewodniki/AI",
    "QCOM": "Półprzewodniki/AI",
    "INTC": "Półprzewodniki/AI",
    "SNDK": "Półprzewodniki/AI",
    # Chmura / Software
    "MSFT": "Chmura/Software",
    "ORCL": "Chmura/Software",
    "NET": "Chmura/Software",
    "SAP": "Chmura/Software",
    "PLTR": "Chmura/Software",
    "IBM": "Chmura/Software",
    "TEAM": "Chmura/Software",
    "FROG": "Chmura/Software",
    "SNOW": "Chmura/Software",
    "DDOG": "Chmura/Software",
    # Cyberbezpieczeństwo
    "CRWD": "Cyberbezpieczeństwo",
    "PANW": "Cyberbezpieczeństwo",
    "OKTA": "Cyberbezpieczeństwo",
    "S": "Cyberbezpieczeństwo",
    "SAIL": "Cyberbezpieczeństwo",
    # Hardware
    "DELL": "Hardware",
    "SSNLF": "Hardware",
    # Mobilność
    "TSLA": "Mobilność",
    "UBER": "Mobilność",
    # Przemysł / Pharma
    "SIEGY": "Przemysł/Pharma",
    "NVO": "Przemysł/Pharma",
    # Finanse
    "BLK": "Finanse",
    # ETF
    "VT": "ETF",
    "QUAL": "ETF",
    "IHI": "ETF",
    "VB": "ETF",
    "EWY": "ETF",
    "IVV": "ETF",
    "XDWD.DE": "ETF",
    "IUSN.DE": "ETF",
    # Krypto
    "BTC": "Krypto",
    "ETH": "Krypto",
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
