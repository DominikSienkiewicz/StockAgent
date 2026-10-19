"""Polski zapis liczb w treści dla czytelnika (MAIL-58).

Przecinek dziesiętny, twarda spacja (U+00A0) jako separator tysięcy i między
kwotą a jednostką (żeby klient poczty nie złamał „12 345,50 zł” w pół),
„zł” po kwocie PLN, „$” przed kwotą USD. Minus zostaje znakiem ASCII „-”:
odbiorca wyszukuje w skrzynce „-4,2%”, a U+2212 by tego nie znalazł.

Zaokrąglenie ROUND_HALF_UP („4,25” → „4,3”), a nie bankierskie
ROUND_HALF_EVEN, które `format` stosuje dla `Decimal` domyślnie.

Moduł domeny, bo lead (`digest_lead`) składa temat maila i nie może sięgać do
warstwy application. Czyste funkcje, tylko stdlib.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

NBSP = " "

_CURRENCY_PREFIX = {"USD": "$"}
_CURRENCY_SUFFIX = {"PLN": "zł"}


def _format_number(value: Decimal, *, decimals: int, signed: bool) -> str:
    """Liczba po polsku: zaokrąglona w górę od połowy, z grupowaniem tysięcy."""
    rounded = value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    sign = "+" if signed else ""
    english = f"{rounded:{sign},.{decimals}f}"
    return english.replace(",", NBSP).replace(".", ",")


def format_pct_pl(value: Decimal, signed: bool, decimals: int = 1) -> str:
    """Procent (4.2 = 4,2%, nie ułamek) po polsku, np. „-4,2%” albo „+1 234,5%”.

    `signed` dokleja „+” przed wartością dodatnią; ujemna zawsze ma „-”.
    """
    return f"{_format_number(value, decimals=decimals, signed=signed)}%"


def format_money_pl(value: Decimal, currency: str) -> str:
    """Kwota z groszami po polsku: „12 345,50 zł”, „$12 345,50”, „-$5,00”.

    Waluta spoza PLN/USD dostaje kod ISO po kwocie („99,90 EUR”).
    """
    code = currency.upper()
    number = _format_number(value, decimals=2, signed=False)
    if code in _CURRENCY_PREFIX:
        sign, digits = ("-", number[1:]) if number.startswith("-") else ("", number)
        return f"{sign}{_CURRENCY_PREFIX[code]}{digits}"
    return f"{number}{NBSP}{_CURRENCY_SUFFIX.get(code, code)}"
