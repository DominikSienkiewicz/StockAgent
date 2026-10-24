"""Testy sekcji powitalnej "Dzień 1" w raporcie e-mail (R-4).

Najważniejsze ryzyko zamrożone tu testem: pomylenie cold-startu z masową
awarią źródeł. Renderer NIE decyduje sam o tonie — bierze `CycleMaturity`
z domeny. STEADY_STATE musi znikać (samosupresja), nawet gdy portfel jest duży;
FIRST_RUN musi dać niepusty, powitalny tekst z liczbą instrumentów.
"""

from __future__ import annotations

from src.application.report_onboarding import (
    onboarding_subject,
    render_onboarding_html,
    render_onboarding_text,
)
from src.domain.cycle_maturity import CycleMaturity

_SECTORS_SAMPLE = {"Big Tech": 3, "Półprzewodniki/AI": 2}


class TestSelfSuppression:
    """STEADY_STATE = ścieżka domyślna: żadnego powitania, żadnego tematu."""

    def test_steady_state_suppresses_html(self) -> None:
        assert (
            render_onboarding_html(
                CycleMaturity.STEADY_STATE,
                instrument_count=43,
                sectors=_SECTORS_SAMPLE,
            )
            == ""
        )

    def test_steady_state_suppresses_text(self) -> None:
        assert (
            render_onboarding_text(
                CycleMaturity.STEADY_STATE,
                instrument_count=43,
                sectors=_SECTORS_SAMPLE,
            )
            == ""
        )

    def test_steady_state_returns_no_subject_even_for_large_portfolio(self) -> None:
        # Rdzeń ryzyka: duża liczba instrumentów NIE może włączyć powitania,
        # jeśli domena mówi STEADY_STATE (to np. masowa awaria, nie Dzień 1).
        assert (
            onboarding_subject(CycleMaturity.STEADY_STATE, instrument_count=999)
            is None
        )


class TestFirstRun:
    """FIRST_RUN = "Dzień 1": powitalny ton z liczbą instrumentów i sektorami."""

    def test_first_run_html_is_non_empty_and_mentions_instrument_count(self) -> None:
        html = render_onboarding_html(
            CycleMaturity.FIRST_RUN,
            instrument_count=43,
            sectors=_SECTORS_SAMPLE,
        )

        assert html != ""
        assert "43" in html

    def test_first_run_html_lists_portfolio_sectors(self) -> None:
        html = render_onboarding_html(
            CycleMaturity.FIRST_RUN,
            instrument_count=5,
            sectors=_SECTORS_SAMPLE,
        )

        assert "Big Tech" in html
        assert "Półprzewodniki/AI" in html

    def test_first_run_html_escapes_sector_labels(self) -> None:
        html = render_onboarding_html(
            CycleMaturity.FIRST_RUN,
            instrument_count=1,
            sectors={"<b>Hack</b>": 1},
        )

        assert "<b>Hack</b>" not in html
        assert "&lt;b&gt;Hack&lt;/b&gt;" in html

    def test_first_run_text_is_non_empty_and_mentions_instrument_count(self) -> None:
        text = render_onboarding_text(
            CycleMaturity.FIRST_RUN,
            instrument_count=43,
            sectors=_SECTORS_SAMPLE,
        )

        assert text != ""
        assert "43" in text

    def test_first_run_subject_is_present_and_mentions_instrument_count(self) -> None:
        subject = onboarding_subject(CycleMaturity.FIRST_RUN, instrument_count=43)

        assert subject is not None
        assert "43" in subject

    def test_first_run_html_without_sectors_still_welcomes(self) -> None:
        # Sektory mogą być puste (nietypowa konfiguracja) — powitanie i tak działa.
        html = render_onboarding_html(
            CycleMaturity.FIRST_RUN,
            instrument_count=7,
            sectors={},
        )

        assert html != ""
        assert "7" in html


class TestInstrumentPluralization:
    """Polska odmiana "instrument" przez liczebnik — po końcówce liczby."""

    def test_singular(self) -> None:
        assert "1 instrument" in onboarding_subject(
            CycleMaturity.FIRST_RUN, instrument_count=1
        )

    def test_few(self) -> None:
        assert "3 instrumenty" in onboarding_subject(
            CycleMaturity.FIRST_RUN, instrument_count=3
        )

    def test_many(self) -> None:
        assert "5 instrumentów" in onboarding_subject(
            CycleMaturity.FIRST_RUN, instrument_count=5
        )

    def test_teens_are_genitive(self) -> None:
        # 12-14 to wyjątek: mimo końcówki 2-4 → "instrumentów", nie "instrumenty".
        assert "13 instrumentów" in onboarding_subject(
            CycleMaturity.FIRST_RUN, instrument_count=13
        )


def test_sectors_follow_the_config_order() -> None:
    """PORT-05: kanoniczna kolejność sektorów idzie z mapy configu; sektor
    spoza niej (np. „Inne”) ląduje na końcu."""
    text = render_onboarding_text(
        CycleMaturity.FIRST_RUN,
        instrument_count=4,
        sectors={"Inne": 1, "Krypto": 1, "Big Tech": 2},
        sector_order=("Big Tech", "ETF", "Krypto"),
    )

    assert text.index("Big Tech") < text.index("Krypto") < text.index("Inne")


def test_first_run_explains_how_to_read_signals() -> None:
    # MAIL-22: pierwszy mail tłumaczy, jak czytać sygnały — nie tylko cold-start.
    html = render_onboarding_html(
        CycleMaturity.FIRST_RUN, instrument_count=3, sectors=_SECTORS_SAMPLE
    )
    text = render_onboarding_text(
        CycleMaturity.FIRST_RUN, instrument_count=3, sectors=_SECTORS_SAMPLE
    )

    for out in (html, text):
        assert "Jak czytać ten mail" in out
        assert "nie prawdopodobieństwo" in out
        assert "nie rekomendacja inwestycyjna" in out
        assert "paper-PnL" in out
        assert "Track Record" in out
    assert html.count("<li") >= 4 + len(_SECTORS_SAMPLE)


def test_steady_state_has_no_reading_guide() -> None:
    html = render_onboarding_html(
        CycleMaturity.STEADY_STATE, instrument_count=3, sectors=_SECTORS_SAMPLE
    )

    assert "Jak czytać ten mail" not in html
