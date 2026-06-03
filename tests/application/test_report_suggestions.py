"""Testy sekcji 💡 WARTE UWAGI — sugestie peerów z 'gorących' sektorów.

Kryterium 'warte uwagi' liczone z DANYCH BIEŻĄCEGO cyklu (zero dodatkowych
calli): sektor jest gorący, gdy jego najsilniejszy monitorowany ruch
|Δ12h| ≥ próg LUB średni |sentyment| ≥ próg. Sugerujemy peerów z tego
sektora, których jeszcze nie monitorujemy.
"""

from __future__ import annotations

from decimal import Decimal

from src.application.report_models import SymbolResult
from src.application.report_suggestions import (
    SectorSuggestion,
    build_sector_suggestions,
    render_suggestions_html,
    render_suggestions_text,
)


def _saved(symbol: str, delta: str, sentiment: float = 0.0) -> SymbolResult:
    return SymbolResult(
        symbol=symbol,
        status="saved",
        delta=Decimal(delta),
        sentiment_score=sentiment,
    )


class TestBuildSectorSuggestions:
    def test_hot_sector_by_price_move_yields_peers(self) -> None:
        results = [_saved("CRWD", "0.06"), _saved("PANW", "0.042")]
        out = build_sector_suggestions(results)
        cyber = [s for s in out if s.sector == "Cyberbezpieczeństwo"]
        assert cyber, "gorący sektor cyber powinien dać sugestię"
        sug = cyber[0]
        assert sug.candidates, "powinni być kandydaci-peerzy"
        # Driverem jest najsilniejszy monitorowany ruch.
        assert sug.drivers[0][0] == "CRWD"

    def test_candidates_exclude_already_monitored(self) -> None:
        # ZS to peer cyber; jeśli już monitorowany, nie może być sugerowany.
        results = [_saved("CRWD", "0.06"), _saved("ZS", "0.01")]
        out = build_sector_suggestions(results)
        cyber = next(s for s in out if s.sector == "Cyberbezpieczeństwo")
        assert "ZS" not in cyber.candidates

    def test_cold_sector_is_not_suggested(self) -> None:
        # Mały ruch i neutralny sentyment → brak sugestii dla Big Tech.
        results = [_saved("AAPL", "0.004", sentiment=0.05)]
        out = build_sector_suggestions(results)
        assert all(s.sector != "Big Tech" for s in out)

    def test_hot_by_sentiment_only(self) -> None:
        # Ruch ceny mały, ale silny sentyment → sektor kwalifikuje się.
        results = [_saved("CRWD", "0.005", sentiment=0.45)]
        out = build_sector_suggestions(results)
        assert any(s.sector == "Cyberbezpieczeństwo" for s in out)

    def test_etf_and_crypto_never_suggested(self) -> None:
        results = [_saved("VT", "0.06"), _saved("BTC", "0.10")]
        out = build_sector_suggestions(results)
        assert all(s.sector not in ("ETF", "Krypto") for s in out)

    def test_empty_results_returns_empty(self) -> None:
        assert build_sector_suggestions([]) == []

    def test_unknown_symbol_ignored(self) -> None:
        assert build_sector_suggestions([_saved("ZZZZ", "0.20")]) == []

    def test_sectors_capped_and_sorted_by_intensity(self) -> None:
        # Cyber mocniejszy (0.09) niż mobilność (0.04) → cyber pierwszy.
        results = [_saved("CRWD", "0.09"), _saved("UBER", "0.04")]
        out = build_sector_suggestions(results)
        assert out[0].sector == "Cyberbezpieczeństwo"


class TestRender:
    def test_html_contains_sector_and_candidates(self) -> None:
        sug = SectorSuggestion(
            sector="Cyberbezpieczeństwo",
            drivers=[("CRWD", Decimal("0.06"))],
            candidates=["ZS", "FTNT"],
        )
        html = render_suggestions_html([sug])
        assert "Cyberbezpieczeństwo" in html
        assert "ZS" in html and "FTNT" in html
        assert "CRWD" in html

    def test_empty_suggestions_render_to_empty_string(self) -> None:
        assert render_suggestions_html([]) == ""
        assert render_suggestions_text([]) == ""

    def test_text_contains_sector_and_candidates(self) -> None:
        sug = SectorSuggestion(
            sector="Mobilność",
            drivers=[("UBER", Decimal("0.05"))],
            candidates=["LYFT"],
        )
        text = render_suggestions_text([sug])
        assert "Mobilność" in text
        assert "LYFT" in text
