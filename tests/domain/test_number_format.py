"""MAIL-58: polski zapis liczb w treści dla czytelnika (temat, lead, push, mail)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.number_format import format_money_pl, format_number_pl, format_pct_pl

NBSP = "\u00a0"


def test_format_pct_pl_uses_decimal_comma() -> None:
    assert format_pct_pl(Decimal("-4.2"), signed=True) == "-4,2%"
    assert format_money_pl(Decimal("12345.5"), "PLN") == f"12{NBSP}345,50{NBSP}zł"


class TestFormatPctPl:
    def test_signed_value_shows_a_plus(self) -> None:
        assert format_pct_pl(Decimal("4.2"), signed=True) == "+4,2%"

    def test_unsigned_value_has_no_plus(self) -> None:
        assert format_pct_pl(Decimal("4.2"), signed=False) == "4,2%"

    def test_minus_is_ascii_for_inbox_search(self) -> None:
        assert format_pct_pl(Decimal("-4.2"), signed=False) == "-4,2%"

    def test_decimals_set_the_precision(self) -> None:
        assert format_pct_pl(Decimal("24.4"), signed=False, decimals=0) == "24%"
        assert format_pct_pl(Decimal("1.2345"), signed=True, decimals=2) == "+1,23%"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("4.25", "+4,3%"), ("-4.25", "-4,3%"), ("4.35", "+4,4%"), ("4.249", "+4,2%")],
    )
    def test_rounds_half_up(self, value: str, expected: str) -> None:
        # Zapis bankierski (ROUND_HALF_EVEN, domyślny w `format`) dałby „+4,2%”.
        assert format_pct_pl(Decimal(value), signed=True) == expected

    def test_thousands_are_grouped_with_a_hard_space(self) -> None:
        assert format_pct_pl(Decimal("1234.5"), signed=True) == f"+1{NBSP}234,5%"

    def test_zero_keeps_the_plus_when_signed(self) -> None:
        # Konwencja dotychczasowego `_format_delta`: zero ze znakiem to „+0.0%”.
        assert format_pct_pl(Decimal("0"), signed=True) == "+0,0%"
        assert format_pct_pl(Decimal("0"), signed=False) == "0,0%"

    @pytest.mark.parametrize(
        ("value", "signed", "decimals", "expected"),
        [
            ("-0.04", True, 1, "+0,0%"),
            ("-0.04", False, 1, "0,0%"),
            ("-0.004", True, 2, "+0,00%"),
            ("-0.4", True, 0, "+0%"),
        ],
    )
    def test_negative_value_rounded_to_zero_has_no_minus(
        self, value: str, signed: bool, decimals: int, expected: str
    ) -> None:
        # „-0,0%” sugerowało spadek, którego w zapisie nie ma.
        assert format_pct_pl(Decimal(value), signed=signed, decimals=decimals) == expected


class TestFormatMoneyPl:
    def test_pln_puts_the_currency_after_the_amount(self) -> None:
        assert format_money_pl(Decimal("1234.5"), "PLN") == f"1{NBSP}234,50{NBSP}zł"

    def test_usd_puts_the_dollar_before_the_amount(self) -> None:
        assert format_money_pl(Decimal("12345.5"), "USD") == f"$12{NBSP}345,50"

    def test_negative_usd_keeps_the_minus_before_the_dollar(self) -> None:
        assert format_money_pl(Decimal("-1234.5"), "USD") == f"-$1{NBSP}234,50"

    def test_negative_pln_uses_an_ascii_minus(self) -> None:
        assert format_money_pl(Decimal("-0.5"), "PLN") == f"-0,50{NBSP}zł"

    def test_amount_rounds_half_up_to_cents(self) -> None:
        assert format_money_pl(Decimal("0.125"), "PLN") == f"0,13{NBSP}zł"

    def test_large_amount_groups_every_thousand(self) -> None:
        assert format_money_pl(Decimal("1234567"), "PLN") == f"1{NBSP}234{NBSP}567,00{NBSP}zł"

    def test_other_currency_code_follows_the_amount(self) -> None:
        assert format_money_pl(Decimal("99.9"), "EUR") == f"99,90{NBSP}EUR"

    def test_amount_rounded_to_zero_has_no_minus(self) -> None:
        assert format_money_pl(Decimal("-0.004"), "PLN") == f"0,00{NBSP}zł"
        assert format_money_pl(Decimal("-0.004"), "USD") == "$0,00"

    def test_signed_amount_shows_a_plus_before_the_currency_sign(self) -> None:
        assert format_money_pl(Decimal("1234.5"), "USD", signed=True) == f"+$1{NBSP}234,50"
        assert format_money_pl(Decimal("-200"), "USD", signed=True) == "-$200,00"
        assert format_money_pl(Decimal("800"), "PLN", signed=True) == f"+800,00{NBSP}zł"

    def test_signed_zero_amount_keeps_the_plus(self) -> None:
        assert format_money_pl(Decimal("0"), "USD", signed=True) == "+$0,00"
        assert format_money_pl(Decimal("-0.004"), "USD", signed=True) == "+$0,00"


class TestFormatNumberPl:
    def test_plain_decimal_uses_a_comma(self) -> None:
        assert format_number_pl(Decimal("0.42"), signed=True, decimals=2) == "+0,42"
        assert format_number_pl(Decimal("48.2"), signed=False, decimals=2) == "48,20"

    def test_thousands_are_grouped_with_a_hard_space(self) -> None:
        assert format_number_pl(Decimal("65040"), signed=False, decimals=1) == f"65{NBSP}040,0"

    def test_rounds_half_up(self) -> None:
        assert format_number_pl(Decimal("0.125"), signed=False, decimals=2) == "0,13"

    def test_value_rounded_to_zero_has_no_minus(self) -> None:
        assert format_number_pl(Decimal("-0.001"), signed=True, decimals=2) == "+0,00"
        assert format_number_pl(Decimal("-0.001"), signed=False, decimals=2) == "0,00"
