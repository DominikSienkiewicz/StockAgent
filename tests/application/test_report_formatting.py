"""Testy etykiet sektorowych w raporcie mailowym.

Sektor pokazywany przy nazwie spółki w sekcji PREDYKCJE (decyzja: PL, przy
nazwie, wszystkie klasy aktywów — akcje sektor, ETF→'ETF', krypto→'Krypto').
"""

from __future__ import annotations

from src.application.report_formatting import (
    provenance_badges_html,
    sector_label,
)
from src.domain.provenance import ProvenanceBadge, ProvenanceLevel


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
