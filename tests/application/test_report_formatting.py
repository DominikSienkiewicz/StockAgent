"""Testy etykiet sektorowych w raporcie mailowym.

Sektor pokazywany przy nazwie spółki w sekcji PREDYKCJE (decyzja: PL, przy
nazwie, wszystkie klasy aktywów — akcje sektor, ETF→'ETF', krypto→'Krypto').
"""

from __future__ import annotations

from src.application.report_formatting import sector_label


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
