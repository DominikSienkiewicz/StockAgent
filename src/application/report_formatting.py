from __future__ import annotations

from decimal import Decimal

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
