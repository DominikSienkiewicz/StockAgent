"""Testy etykiet sektorowych w raporcie mailowym.

Sektor pokazywany przy nazwie spółki w sekcji PREDYKCJE (decyzja: PL, przy
nazwie, wszystkie klasy aktywów — akcje sektor, ETF→'ETF', krypto→'Krypto').
"""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.application.report_formatting import (
    COMPANY_NAMES,
    DEFAULT_DISPLAY_TIMEZONE,
    PALETTE,
    company_label,
    delta_color,
    format_local,
    pct,
    plural_pl,
    provenance_badges_html,
    sector_label,
    trend_color,
)
from src.domain.provenance import ProvenanceBadge, ProvenanceLevel

CONFIG_TOML = Path(__file__).resolve().parents[2] / "config.toml"
# Listy symboli z config.toml, które trafiają do maila (lead, temat, tabele,
# Risk Watch). Niezmiennik „config ⊆ COMPANY_NAMES", nie zamrożona lista —
# nowy symbol w configu bez nazwy wywala ten test.
_CONFIGURED_SYMBOL_LISTS = ("symbols", "symbols_etf", "crypto_symbols", "risk_symbols")


class TestSectorLabel:
    def test_stock_returns_polish_sector(self) -> None:
        assert sector_label("AAPL") == "Big Tech"
        assert sector_label("NVDA") == "Półprzewodniki/AI"
        assert sector_label("MSFT") == "Chmura/Software"

    def test_cybersecurity_bucket(self) -> None:
        # Nowo dodane spółki cyber muszą mieć sektor.
        for sym in ("CRWD", "PANW", "OKTA", "S", "SAIL"):
            assert sector_label(sym) == "Cyberbezpieczeństwo", sym

    def test_etf_maps_to_etf(self) -> None:
        for sym in ("VT", "QUAL", "IHI", "VB", "EWY", "IVV", "XDWD.DE", "IUSN.DE"):
            assert sector_label(sym) == "ETF", sym

    def test_crypto_maps_to_krypto(self) -> None:
        assert sector_label("BTC") == "Krypto"
        assert sector_label("ETH") == "Krypto"

    def test_unknown_symbol_returns_none(self) -> None:
        assert sector_label("ZZZZ") is None

    def test_empty_symbol_returns_none(self) -> None:
        assert sector_label("") is None

    def test_every_configured_symbol_has_a_sector(self) -> None:
        # Każdy ticker z domyślnego portfolio (43) + krypto MUSI mieć sektor —
        # inaczej w mailu pojawi się symbol bez etykiety mimo że jest w SYMBOLS.
        portfolio = [
            "AAPL", "AMZN", "GOOGL", "MSFT", "META", "NVDA", "TSLA", "AMD",
            "NET", "PLTR", "ORCL", "UBER", "TSM", "ASML", "ASMIY", "SAP",
            "SIEGY", "NVO", "DELL", "IBM", "MU", "QCOM", "CRWD", "INTC",
            "SNDK", "BLK", "SSNLF", "TEAM", "FROG", "SNOW", "DDOG", "SAIL",
            "OKTA", "S", "PANW", "VT", "QUAL", "IHI", "VB", "EWY", "IVV",
            "XDWD.DE", "IUSN.DE", "BTC", "ETH",
        ]
        missing = [s for s in portfolio if sector_label(s) is None]
        assert not missing, f"brak sektora dla: {missing}"


class TestProvenanceBadgesHtml:
    def test_empty_list_renders_nothing(self) -> None:
        assert provenance_badges_html([]) == ""

    def test_fresh_badge_omitted_to_avoid_noise(self) -> None:
        # FRESH to "wszystko OK" — nie zaśmiecamy nim wiersza symbolu.
        badges = [ProvenanceBadge(level=ProvenanceLevel.FRESH, label="Świeże")]
        assert provenance_badges_html(badges) == ""

    def test_degraded_badge_renders_label(self) -> None:
        badges = [
            ProvenanceBadge(
                level=ProvenanceLevel.DEGRADED,
                label="Dane zdegradowane",
                detail="powód: av_keys_exhausted",
            )
        ]
        out = provenance_badges_html(badges)
        assert "Dane zdegradowane" in out
        assert "<span" in out

    def test_stale_and_flagged_both_rendered(self) -> None:
        badges = [
            ProvenanceBadge(level=ProvenanceLevel.STALE, label="Dane z cache"),
            ProvenanceBadge(level=ProvenanceLevel.FLAGGED, label="Flagi jakości"),
        ]
        out = provenance_badges_html(badges)
        assert "Dane z cache" in out
        assert "Flagi jakości" in out

    def test_untrusted_detail_is_escaped(self) -> None:
        # Detail może pochodzić z degraded_reason / flag (dane zewnętrzne) —
        # musi przejść przez escapowanie, inaczej chip jest sinkiem XSS.
        badges = [
            ProvenanceBadge(
                level=ProvenanceLevel.DEGRADED,
                label="Dane zdegradowane",
                detail='"><img src=x onerror=alert(1)>',
            )
        ]
        out = provenance_badges_html(badges)
        assert "<img src=x onerror=alert(1)>" not in out
        assert "&lt;img src=x onerror=alert(1)&gt;" in out

    def test_untrusted_label_is_escaped(self) -> None:
        badges = [
            ProvenanceBadge(
                level=ProvenanceLevel.FLAGGED,
                label="<script>alert(1)</script>",
            )
        ]
        out = provenance_badges_html(badges)
        assert "<script>alert(1)</script>" not in out
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out


class TestCompanyNamesCoverConfig:
    def test_every_configured_symbol_has_company_name(self) -> None:
        config = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
        configured = {
            symbol for key in _CONFIGURED_SYMBOL_LISTS for symbol in config[key]
        }

        missing = sorted(configured - COMPANY_NAMES.keys())

        assert missing == []

    def test_single_letter_ticker_renders_with_name(self) -> None:
        # „S" (SentinelOne) jako goły ticker w temacie maila jest nieczytelne.
        assert company_label("S") == "S (SentinelOne Inc.)"


class TestFormatLocal:
    """MAIL-14: czas w temacie, nagłówku i pushu jest lokalny, nie UTC."""

    def test_summer_time_is_utc_plus_two(self) -> None:
        moment = datetime(2026, 10, 16, 5, 30, tzinfo=UTC)

        assert format_local(moment, "Europe/Warsaw") == "pt 16.10 07:30"

    def test_winter_time_is_utc_plus_one(self) -> None:
        moment = datetime(2026, 11, 2, 5, 30, tzinfo=UTC)

        assert format_local(moment, "Europe/Warsaw") == "pon 02.11 06:30"

    def test_local_date_can_be_the_next_day(self) -> None:
        # 23:30 UTC w czwartek to już piątek w Warszawie — dzień tygodnia i data
        # liczą się po przeliczeniu, nie z daty UTC.
        moment = datetime(2026, 5, 14, 23, 30, tzinfo=UTC)

        assert format_local(moment, "Europe/Warsaw") == "pt 15.05 01:30"

    def test_naive_datetime_is_read_as_utc(self) -> None:
        # Wszystkie znaczniki w repo są w UTC; naiwny datetime nie może zostać
        # zinterpretowany jako czas lokalny maszyny (runner GHA, laptop).
        assert format_local(datetime(2026, 10, 16, 5, 30), "Europe/Warsaw") == "pt 16.10 07:30"

    def test_default_display_timezone_is_warsaw(self) -> None:
        assert DEFAULT_DISPLAY_TIMEZONE == "Europe/Warsaw"


class TestPluralPl:
    """ARCH-28: jedna reguła polskiej odmiany przez liczebnik zamiast kopii per sekcja."""

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1, "predykcja"),
            (2, "predykcje"),
            (4, "predykcje"),
            (22, "predykcje"),
            (104, "predykcje"),
            (0, "predykcji"),
            (5, "predykcji"),
            (11, "predykcji"),
            (12, "predykcji"),
            (14, "predykcji"),
            (21, "predykcji"),
            (112, "predykcji"),
            (-3, "predykcje"),
        ],
    )
    def test_plural_pl_matches_polish_rules(self, count: int, expected: str) -> None:
        assert plural_pl(count, "predykcja", "predykcje", "predykcji") == expected


class TestPalette:
    """ARCH-28: kolory raportu w jednym miejscu; helpery kolorów biorą je stąd."""

    def test_palette_keeps_the_report_colors(self) -> None:
        assert PALETTE.positive == "#16a34a"
        assert PALETTE.negative == "#dc2626"
        assert PALETTE.neutral == "#737373"
        assert PALETTE.muted == "#6b7280"

    def test_color_helpers_read_the_palette(self) -> None:
        assert trend_color("BULLISH") == PALETTE.positive
        assert trend_color("BEARISH") == PALETTE.negative
        assert trend_color(None) == PALETTE.neutral
        assert delta_color(Decimal("0.01")) == PALETTE.positive
        assert delta_color(Decimal("-0.01")) == PALETTE.negative
        assert delta_color(Decimal("0")) == PALETTE.neutral
        assert delta_color(None) == PALETTE.neutral


class TestPct:
    """ARCH-28: `pct` dostaje liczbę miejsc po przecinku. Sekcje maila zostają przy
    dotychczasowym zapisie, dopóki etap MAIL-58 nie przeniesie ich na polski."""

    def test_pct_takes_the_number_of_decimals(self) -> None:
        assert pct(Decimal("0.01234"), signed=True, decimals=1) == "+1.2%"
        assert pct(Decimal("0.24"), decimals=0) == "24%"

    def test_pct_defaults_keep_the_current_section_format(self) -> None:
        assert pct(Decimal("0.01234")) == "1.23%"
        assert pct(Decimal("-0.042"), signed=True) == "-4.20%"
        assert pct(None) == "—"
