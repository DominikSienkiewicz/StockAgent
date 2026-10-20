"""DX-01: przykładowy mail raportu bez kluczy, bazy i sieci — czysta funkcja.

`render_sample_report` składa fikcyjny cykl z `sample_data` tym samym
`build_html_report`, którym Fast Loop składa prawdziwy mail, więc podgląd nie
rozjeżdża się z produktem. Zapis plików robi podkomenda
`main_tools.py render-sample-report` (ARCH-39); tu nie ma I/O ani `Settings`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.application.report_builder import build_html_report
from src.tools.sample_data import (
    SAMPLE_DURATION_SECONDS,
    SAMPLE_STARTED_AT,
    sample_context,
    sample_results,
)

SAMPLE_HTML_NAME = "report-email.html"
SAMPLE_TEXT_NAME = "report-email.txt"


@dataclass(frozen=True)
class SampleReport:
    """Obie reprezentacje maila, jak z `build_html_report`."""

    html: str
    text: str


def render_sample_report() -> SampleReport:
    """Mail fikcyjnego cyklu; dwa wywołania dają identyczny wynik co do znaku."""
    html, text = build_html_report(
        sample_results(), SAMPLE_STARTED_AT, SAMPLE_DURATION_SECONDS, sample_context()
    )
    return SampleReport(html=html, text=text)
