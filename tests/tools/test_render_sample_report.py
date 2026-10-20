"""DX-01: przykładowy mail bez kluczy — deterministyczny, bez PII, z każdą klasą
aktywa i każdym statusem symbolu, przez ten sam builder co Fast Loop."""

from __future__ import annotations

import re

from src.tools.render_sample_report import render_sample_report

_EMAIL_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def test_sample_report_is_deterministic_and_has_no_pii() -> None:
    first = render_sample_report()
    second = render_sample_report()

    assert first == second
    for body in (first.html, first.text):
        assert "@" not in body
        assert not _EMAIL_ADDRESS.search(body)
    assert "Krypto" in first.html


def test_sample_report_shows_every_status_and_asset_class() -> None:
    report = render_sample_report()

    for symbol in ("NVDA", "AMD", "VOO", "BTC", "ETH", "MSFT", "XDWD.DE", "ASML"):
        assert symbol in report.html, symbol
        assert symbol in report.text, symbol
    # Błąd źródła, odznaka degradacji i stopka przykładowego przebiegu.
    assert "429 Too Many Requests" in report.html
    assert "przebieg #sample" in report.text
