"""ARCH-12: rejestr sekcji maila — kolejność, w jakiej mail je drukuje.

Jedna krotka dla HTML i wersji tekstowej: sekcja tylko w HTML ma `no_text`, więc
tekst idzie tą samą kolejnością. Nowa sekcja to wpis tutaj (patrz `section.py`).
"""

from __future__ import annotations

from src.application.report_lead import LEAD_SECTION, PREHEADER_SECTION
from src.application.report_onboarding import ONBOARDING_SECTION
from src.application.report_quota_banner import QUOTA_BANNER_SECTION
from src.application.reporting.section import ReportSection

REPORT_SECTIONS: tuple[ReportSection, ...] = (
    PREHEADER_SECTION,
    QUOTA_BANNER_SECTION,
    LEAD_SECTION,
    ONBOARDING_SECTION,
)
