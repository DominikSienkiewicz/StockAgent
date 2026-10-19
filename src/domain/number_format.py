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


def _round_half_up(value: Decimal, decimals: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def _render(rounded: Decimal, *, decimals: int, signed: bool) -> str:
    """Zaokrąglona liczba po polsku, z grupowaniem tysięcy."""
    sign = "+" if signed else ""
    english = f"{rounded:{sign},.{decimals}f}"
    return english.replace(",", NBSP).replace(".", ",")


def format_number_pl(value: Decimal, signed: bool, decimals: int) -> str:
    """Liczba po polsku z `decimals` miejscami, np. „+0,42” albo „65 040,0”.

    `signed` dokleja „+” przed wartością dodatnią i przed zerem; ujemna zawsze
    ma „-”. Wartość zaokrąglona do zera traci minus („-0,001” → „0,00”), bo
    „-0,00” sugerowałoby spadek, którego w zapisie nie ma.
    """
    rounded = _round_half_up(value, decimals)
    if rounded.is_zero():
        rounded = rounded.copy_abs()
    return _render(rounded, decimals=decimals, signed=signed)


def format_pct_pl(value: Decimal, signed: bool, decimals: int = 1) -> str:
    """Procent (4.2 = 4,2%, nie ułamek) po polsku, np. „-4,2%” albo „+1 234,5%”.

    Znak i zero jak w `format_number_pl`: „+0,0%” ze znakiem, nigdy „-0,0%”.
    """
    return f"{format_number_pl(value, signed, decimals)}%"


def format_money_pl(value: Decimal, currency: str, *, signed: bool = False) -> str:
    """Kwota z groszami po polsku: „12 345,50 zł”, „$12 345,50”, „-$5,00”.

    `signed` dokleja „+” przed zyskiem i zerem („+$0,00”), znak stoi przed „$”.
    Kwota zaokrąglona do zera traci minus. Waluta spoza PLN/USD dostaje kod ISO
    po kwocie („99,90 EUR”).
    """
    code = currency.upper()
    number = format_number_pl(value, signed, 2)
    if code in _CURRENCY_PREFIX:
        sign, digits = (number[0], number[1:]) if number[0] in "+-" else ("", number)
        return f"{sign}{_CURRENCY_PREFIX[code]}{digits}"
    return f"{number}{NBSP}{_CURRENCY_SUFFIX.get(code, code)}"
